"""Board encoding and the trap-policy network.

One module owns both the tensor layout and the model, because every shape
mismatch in a pipeline like this comes from two files disagreeing about the
encoding. The generator, the trainer and the inference wrapper all import the
constants from here; none of them writes a literal ``8`` or ``4096``.

Move encoding is ``from_square * 64 + to_square``. Promotions collapse onto the
same index as the non-promotion move between those squares, so the head cannot
distinguish ``e7e8=Q`` from ``e7e8=N``. That is a deliberate, bounded loss: this
network proposes *candidates* and the search evaluates them, so an
underpromotion arrives as its queen twin and is scored properly one layer down.
The alternative, AlphaZero's 4672-way encoding, buys underpromotion at the cost
of a much larger head for a case that arises in well under 1% of positions.
"""

from __future__ import annotations

from typing import Final, List, Optional, Sequence

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "INPUT_PLANES",
    "POLICY_SIZE",
    "TrapPolicyNet",
    "encode_board",
    "encode_batch",
    "legal_move_mask",
    "move_to_index",
    "index_to_moves",
]

BOARD_SIZE: Final[int] = 8
NUM_SQUARES: Final[int] = BOARD_SIZE * BOARD_SIZE
POLICY_SIZE: Final[int] = NUM_SQUARES * NUM_SQUARES  # 4096: from-square x to-square

# Plane layout, fixed once and referenced everywhere:
#   0-5    our pieces      (pawn, knight, bishop, rook, queen, king)
#   6-11   their pieces    (same order)
#   12     legal-move destinations
#   13     castling rights (four corners) and en-passant file
#   14     opponent rating, broadcast
PIECE_PLANES: Final[int] = 12
PLANE_LEGAL: Final[int] = 12
PLANE_RIGHTS: Final[int] = 13
PLANE_RATING: Final[int] = 14
INPUT_PLANES: Final[int] = 15
"""15, not the 14 in the brief: conditioning on opponent rating needs its own
plane, and folding it into the castling plane would make two unrelated signals
share a scale."""

RATING_MIN: Final[float] = 800.0
RATING_MAX: Final[float] = 2400.0

PIECE_ORDER: Final[Sequence[chess.PieceType]] = (
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING,
)


def move_to_index(move: chess.Move) -> int:
    """``from_square * 64 + to_square``. Promotions share their base index."""
    return move.from_square * NUM_SQUARES + move.to_square


def index_to_moves(board: chess.Board, index: int) -> List[chess.Move]:
    """Legal moves matching a policy index, queen promotion first.

    Returns a list because an index can name several legal moves once
    promotions collapse. Empty when the index names nothing legal, which is how
    the caller stays incapable of emitting an illegal move.
    """
    from_square, to_square = divmod(index, NUM_SQUARES)
    matches = [
        move
        for move in board.legal_moves
        if move.from_square == from_square and move.to_square == to_square
    ]
    matches.sort(key=lambda move: (move.promotion != chess.QUEEN, move.promotion or 0))
    return matches


def _normalised_rating(rating: int) -> float:
    span = RATING_MAX - RATING_MIN
    return float(min(max(rating, RATING_MIN), RATING_MAX) - RATING_MIN) / span


def encode_board(board: chess.Board, rating: int) -> torch.Tensor:
    """``[INPUT_PLANES, 8, 8]`` float tensor, from the side-to-move's view.

    Planes are always "us then them", so the network never has to learn colour
    symmetry twice. Squares are indexed ``rank * 8 + file`` with rank 0 at the
    bottom, matching ``chess.square`` exactly.
    """
    planes = torch.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    us = board.turn
    flat = planes.view(INPUT_PLANES, NUM_SQUARES)

    for offset, colour in ((0, us), (len(PIECE_ORDER), not us)):
        for piece_index, piece_type in enumerate(PIECE_ORDER):
            for square in board.pieces(piece_type, colour):
                flat[offset + piece_index, square] = 1.0

    for move in board.legal_moves:
        flat[PLANE_LEGAL, move.to_square] = 1.0

    for colour, kingside, queenside in (
        (us, chess.BB_H1 if us else chess.BB_H8, chess.BB_A1 if us else chess.BB_A8),
        (not us, chess.BB_H1 if not us else chess.BB_H8, chess.BB_A1 if not us else chess.BB_A8),
    ):
        value = 1.0 if colour == us else 0.5
        if board.castling_rights & kingside:
            flat[PLANE_RIGHTS, (kingside.bit_length() - 1)] = value
        if board.castling_rights & queenside:
            flat[PLANE_RIGHTS, (queenside.bit_length() - 1)] = value
    if board.ep_square is not None:
        flat[PLANE_RIGHTS, board.ep_square] = 0.25

    planes[PLANE_RATING].fill_(_normalised_rating(rating))
    return planes


def encode_batch(
    boards: Sequence[chess.Board], ratings: Sequence[int], device: Optional[torch.device] = None
) -> torch.Tensor:
    """``[B, INPUT_PLANES, 8, 8]`` for a batch of positions."""
    if len(boards) != len(ratings):
        raise ValueError(f"{len(boards)} boards but {len(ratings)} ratings")
    if not boards:
        return torch.zeros((0, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    stacked = torch.stack([encode_board(board, rating) for board, rating in zip(boards, ratings)])
    return stacked.to(device) if device is not None else stacked


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """``[POLICY_SIZE]`` boolean mask; True where a legal move lands."""
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU twice with a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)


class TrapPolicyNet(nn.Module):
    """Small AlphaZero-style trunk with a spatially structured policy head.

    The head is a 1x1 convolution to 64 planes of 8x8, flattened to 4096, so the
    output is literally "for each from-square, a board of to-squares". Flattening
    the trunk into ``nn.Linear(..., 4096)`` instead would cost around 8.4M
    parameters -- more than twenty times the whole rest of the network -- to
    express the same mapping while discarding the spatial structure that makes
    it learnable.
    """

    def __init__(self, channels: int = 64, blocks: int = 4) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {blocks}")
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy = nn.Conv2d(channels, NUM_SQUARES, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, INPUT_PLANES, 8, 8]`` -> ``[B, POLICY_SIZE]`` raw logits."""
        if x.dim() != 4 or x.shape[1] != INPUT_PLANES or x.shape[2:] != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"expected [B, {INPUT_PLANES}, {BOARD_SIZE}, {BOARD_SIZE}], got {tuple(x.shape)}"
            )
        features = self.tower(self.stem(x))
        logits: torch.Tensor = self.policy(features).flatten(start_dim=1)
        return logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
