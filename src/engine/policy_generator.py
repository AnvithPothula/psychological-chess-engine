"""Neural candidate generator: the trained policy, wrapped for the search.

Proposes the moves worth searching, so Stockfish scores a handful of candidates
instead of ranking every legal move. One forward pass replaces the widest part
of the root scan.

**It proposes; it does not decide.** The searcher's safety guarantees rest on
knowing the objectively best move, and a policy net emits an ordering with no
scores attached. Feeding these candidates to Stockfish (``root_moves=``) keeps
the scores and the fallback intact; using the policy *instead of* evaluation
would quietly delete the safety filter that Milestones 2 and 5 are built on.

Every returned move comes out of ``board.legal_moves``, never reconstructed from
an index, so a mis-trained or corrupt checkpoint can degrade move quality but
can never produce an illegal move.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

import chess
import torch
import torch.nn.functional as F

from src.training.model import (
    INPUT_PLANES,
    POLICY_SIZE,
    TrapPolicyNet,
    encode_board,
    index_to_moves,
    legal_move_mask,
    move_to_index,
)
from src.types import MoveDistribution

__all__ = ["NeuralCandidateGenerator", "PolicyUnavailableError", "load_proposer"]

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT: Final[Path] = Path("models/trap_policy.pth")
DEFAULT_TOP_K: Final[int] = 5


class PolicyUnavailableError(RuntimeError):
    """The checkpoint is missing, unreadable, or built for another encoding."""


def _select_device(preference: Optional[str]) -> torch.device:
    """CPU by default, and that is not an oversight.

    The search calls this one board at a time, and at batch size 1 with a
    308k-parameter network the accelerator loses: measured on an M4, MPS runs
    1.64ms per board steady-state and 9.09ms over the first ten calls, against
    0.68ms and 0.77ms on CPU. Kernel-launch overhead dominates when there is no
    batch to amortise it over. Training, which is batched, still picks the
    accelerator -- see ``train_dpo.select_device``.
    """
    if preference:
        return torch.device(preference)
    return torch.device("cpu")


class NeuralCandidateGenerator:
    """Loads ``trap_policy.pth`` and ranks legal moves for a given opponent."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        *,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = _select_device(device)
        self.model = self._load(checkpoint_path, self.device)
        self.model.eval()
        logger.info(
            "policy: loaded %s on %s (%s parameters)",
            checkpoint_path.name, self.device, f"{self.model.parameter_count:,}",
        )

    @staticmethod
    def _load(path: Path, device: torch.device) -> TrapPolicyNet:
        if not path.exists():
            raise PolicyUnavailableError(
                f"no checkpoint at {path}; run python -m src.training.train_dpo"
            )
        try:
            payload: Dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        except (OSError, RuntimeError, EOFError) as exc:
            raise PolicyUnavailableError(f"could not read {path}: {exc}") from exc

        # Encoding drift between trainer and wrapper is the classic way this
        # pipeline breaks, and it fails as silent nonsense rather than a crash.
        # The checkpoint carries its shapes so the mismatch is caught on load.
        for key, expected in (("input_planes", INPUT_PLANES), ("policy_size", POLICY_SIZE)):
            actual = payload.get(key)
            if actual is not None and int(actual) != expected:
                raise PolicyUnavailableError(
                    f"{path} was trained with {key}={actual}, this build expects {expected}"
                )

        model = TrapPolicyNet(
            channels=int(payload.get("channels", 64)), blocks=int(payload.get("blocks", 4))
        )
        try:
            model.load_state_dict(payload["state_dict"])
        except (KeyError, RuntimeError) as exc:
            raise PolicyUnavailableError(f"{path} does not match TrapPolicyNet: {exc}") from exc
        return model.to(device)

    # -- inference ----------------------------------------------------------

    @torch.no_grad()
    def logits(self, board: chess.Board, rating: int) -> torch.Tensor:
        """Raw ``[POLICY_SIZE]`` logits for one position."""
        planes = encode_board(board, rating).unsqueeze(0).to(self.device)
        out: torch.Tensor = self.model(planes).squeeze(0)
        return out

    @torch.no_grad()
    def move_probabilities(self, board: chess.Board, rating: int) -> MoveDistribution:
        """Softmax over legal moves only, as a validated distribution.

        Promotions share a policy index, so an index's mass is split evenly
        across the legal moves it names; the distribution still sums to one.
        """
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError(f"no legal moves in {board.fen()}")

        mask = legal_move_mask(board).to(self.device)
        masked = self.logits(board, rating).masked_fill(~mask, float("-inf"))
        probabilities = F.softmax(masked, dim=0)

        shared: Dict[int, int] = {}
        for move in legal:
            index = move_to_index(move)
            shared[index] = shared.get(index, 0) + 1

        distribution = {
            move: float(probabilities[move_to_index(move)].item()) / shared[move_to_index(move)]
            for move in legal
        }
        total = sum(distribution.values())
        if total <= 0.0:
            uniform = 1.0 / len(legal)
            return MoveDistribution({move: uniform for move in legal})
        return MoveDistribution({move: value / total for move, value in distribution.items()})

    @torch.no_grad()
    def get_candidates(
        self, board: chess.Board, rating: int, top_k: int = DEFAULT_TOP_K
    ) -> List[chess.Move]:
        """The ``top_k`` legal moves the policy rates highest, best first."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        legal = list(board.legal_moves)
        if not legal:
            return []
        if len(legal) <= top_k:
            # Nothing to filter; skip the forward pass entirely.
            return legal

        mask = legal_move_mask(board).to(self.device)
        masked = self.logits(board, rating).masked_fill(~mask, float("-inf"))
        # More indices than needed, because one index can name several legal
        # moves and some ranked indices may name none.
        wanted = min(POLICY_SIZE, top_k * 2 + 8)
        ranked = torch.topk(masked, k=wanted).indices.tolist()

        candidates: List[chess.Move] = []
        for index in ranked:
            for move in index_to_moves(board, int(index)):
                if move not in candidates:
                    candidates.append(move)
                if len(candidates) >= top_k:
                    return candidates
        # A checkpoint that ranks nothing legal still must not stall the search.
        for move in legal:
            if move not in candidates:
                candidates.append(move)
            if len(candidates) >= top_k:
                break
        return candidates


def load_proposer(
    checkpoint_path: Path = DEFAULT_CHECKPOINT, *, device: Optional[str] = None
) -> Optional[NeuralCandidateGenerator]:
    """The generator, or ``None`` when no usable checkpoint exists.

    A missing checkpoint must not stop the engine from starting: the search
    falls back to the Stockfish scan on its own when there is no proposer.
    """
    try:
        return NeuralCandidateGenerator(checkpoint_path, device=device)
    except PolicyUnavailableError as exc:
        logger.warning("policy: running without a neural proposer (%s)", exc)
        return None
