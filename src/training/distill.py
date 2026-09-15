"""Warm-start ``TrapPolicyNet`` by distilling Maia's raw policy.

Milestone 8 trained the network from random initialisation on 170 preference
pairs and it memorised them completely without learning anything transferable
(validation preference accuracy stayed at 0.529, chance being 0.500). The
network had no notion of chess to align in the first place.

Distillation supplies that notion. Maia at one node is a pure policy forward
pass, so tens of thousands of positions can be labelled in minutes, and matching
its distribution teaches piece values, threats and plausible-move structure --
the representation the preference stage needs before it can shift anything.

**A note on the rating channel.** Distillation labels each position with a
randomly chosen Maia band and sets the rating plane to that band, so the channel
means "the Maia band in play" in both stages. It does *not* mean the same thing
to the output: distillation teaches "what a player of band R does here", while
DPO teaches "what beats a player of band R here". The trunk transfers; the head
is deliberately re-purposed, and whether the preference stage actually moves it
is the thing ``train_dpo`` reports as ``val_pref``.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Optional, Protocol, Sequence, Tuple

import chess
import chess.polyglot
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src import config as engine_config
from src.types import MoveDistribution
from src.training.model import (
    INPUT_PLANES,
    POLICY_SIZE,
    TrapPolicyNet,
    encode_board,
    legal_move_mask,
    move_to_index,
)

__all__ = [
    "DistillationDataset",
    "build_corpus",
    "distill",
    "sample_positions",
    "top_k_agreement",
    "main",
]

logger = logging.getLogger(__name__)

DEFAULT_CORPUS: Final[Path] = Path("build/distill_corpus.npz")
DEFAULT_CHECKPOINT: Final[Path] = Path("models/policy_warm.pth")
DEFAULT_POSITIONS: Final[int] = 30_000

WALK_PLIES: Final[Tuple[int, int]] = (4, 40)
"""Random-walk depth. Shallower than 4 is opening theory the book already
covers; deeper than 40 drifts into endgames the search rarely reaches."""

BOOK_WALK_FRACTION: Final[float] = 0.5
"""Half the corpus starts from a book line so the opening -- where traps live --
is represented far better than uniform random play would manage."""

PROGRESS_INTERVAL: Final[int] = 2_000
TARGET_TOP3_ACCURACY: Final[float] = 0.60


def sample_positions(
    count: int,
    *,
    rng: random.Random,
    books: Sequence[Path] = (),
    max_attempts_factor: int = 8,
) -> List[chess.Board]:
    """Distinct, non-terminal positions from random walks.

    Deduplicated on EPD rather than FEN: the same position at two move numbers
    is one training example, and counting it twice just reweights the loss.
    """
    readers = []
    for path in books:
        try:
            readers.append(chess.polyglot.open_reader(path))
        except (OSError, ValueError) as exc:
            logger.warning("distill: book %s unavailable (%s)", path, exc)

    seen: set[str] = set()
    positions: List[chess.Board] = []
    attempts = 0
    limit = count * max_attempts_factor

    try:
        while len(positions) < count and attempts < limit:
            attempts += 1
            board = chess.Board()
            target_plies = rng.randint(*WALK_PLIES)
            use_book = bool(readers) and rng.random() < BOOK_WALK_FRACTION

            for _ in range(target_plies):
                if board.is_game_over(claim_draw=False):
                    break
                move = None
                if use_book:
                    move = _book_move(readers, board, rng)
                if move is None:
                    move = rng.choice(list(board.legal_moves))
                board.push(move)

            if board.is_game_over(claim_draw=False):
                continue
            key = board.epd()
            if key in seen:
                continue
            seen.add(key)
            positions.append(board.copy(stack=False))
            if len(positions) % PROGRESS_INTERVAL == 0:
                logger.info("distill: sampled %d/%d positions", len(positions), count)
    finally:
        for reader in readers:
            reader.close()

    if len(positions) < count:
        logger.warning(
            "distill: only %d distinct positions after %d walks", len(positions), attempts
        )
    return positions


def _book_move(
    readers: Sequence[chess.polyglot.MemoryMappedReader], board: chess.Board, rng: random.Random
) -> Optional[chess.Move]:
    for reader in readers:
        entries = list(reader.find_all(board))
        if entries:
            moves = [entry.move for entry in entries]
            weights = [entry.weight for entry in entries]
            return rng.choices(moves, weights=weights, k=1)[0]
    return None


class MaiaLike(Protocol):
    """What labelling needs of Maia. Declared structurally so the corpus builder
    can be driven by a scripted policy without starting Lc0."""

    rating: int

    def set_rating(self, rating: int) -> None: ...

    def predict_move_probabilities(
        self, board: chess.Board, *, temperature: Optional[float] = ...
    ) -> MoveDistribution: ...


def build_corpus(
    positions: Sequence[chess.Board],
    ratings: Sequence[int],
    maia: "MaiaLike",
    destination: Path,
) -> Dict[str, float]:
    """Label positions with Maia's policy and write a compressed corpus.

    Targets are stored sparsely -- indices and probabilities with per-position
    offsets. A dense ``[N, 4096]`` array would be 480MB for 30k positions to
    hold roughly thirty non-zero entries each.
    """
    if len(positions) != len(ratings):
        raise ValueError(f"{len(positions)} positions but {len(ratings)} ratings")

    fens: List[str] = []
    kept_ratings: List[int] = []
    indices: List[int] = []
    probabilities: List[float] = []
    offsets: List[int] = [0]

    started = time.perf_counter()
    current_band = -1
    for number, (board, rating) in enumerate(zip(positions, ratings), start=1):
        if rating != current_band:
            maia.set_rating(rating)
            current_band = rating
        try:
            distribution = maia.predict_move_probabilities(board)
        except (ValueError, Exception) as exc:  # noqa: BLE001 - one bad position is a skip
            logger.debug("distill: skipping %s (%s)", board.fen(), exc)
            continue

        fens.append(board.fen())
        kept_ratings.append(rating)
        for move, probability in distribution.probabilities.items():
            indices.append(move_to_index(move))
            probabilities.append(probability)
        offsets.append(len(indices))

        if number % PROGRESS_INTERVAL == 0:
            rate = number / (time.perf_counter() - started)
            logger.info("distill: labelled %d/%d (%.0f pos/s)", number, len(positions), rate)

    elapsed = time.perf_counter() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        fens=np.array(fens),
        ratings=np.array(kept_ratings, dtype=np.int32),
        indices=np.array(indices, dtype=np.int32),
        probabilities=np.array(probabilities, dtype=np.float32),
        offsets=np.array(offsets, dtype=np.int64),
    )
    rate = len(fens) / elapsed if elapsed else 0.0
    logger.info(
        "distill: wrote %d positions to %s in %.1fs (%.0f pos/s, %.1f MB)",
        len(fens), destination, elapsed, rate, destination.stat().st_size / 1e6,
    )
    return {"positions": float(len(fens)), "seconds": elapsed, "rate": rate}


@dataclass(frozen=True, slots=True)
class DistillSample:
    planes: Tensor
    target: Tensor
    """Dense ``[POLICY_SIZE]`` distribution; zero everywhere Maia gave no mass."""

    legal: Tensor


class DistillationDataset(Dataset[DistillSample]):
    """Corpus loaded into memory once, encoded once.

    Planes are encoded up front rather than per epoch: encoding is pure Python
    and would otherwise dominate every pass. Thirty thousand positions is about
    115MB of float32, which is cheaper than paying the encode cost thirty times.
    """

    def __init__(self, path: Path, limit: Optional[int] = None) -> None:
        payload = np.load(path, allow_pickle=False)
        fens = payload["fens"]
        ratings = payload["ratings"]
        indices = payload["indices"]
        probabilities = payload["probabilities"]
        offsets = payload["offsets"]

        total = len(fens) if limit is None else min(limit, len(fens))
        self.planes = torch.empty((total, INPUT_PLANES, 8, 8), dtype=torch.float32)
        self.targets = torch.zeros((total, POLICY_SIZE), dtype=torch.float32)
        self.legal = torch.zeros((total, POLICY_SIZE), dtype=torch.bool)
        self.fens: List[str] = []
        self.ratings: List[int] = []

        for row in range(total):
            board = chess.Board(str(fens[row]))
            rating = int(ratings[row])
            self.planes[row] = encode_board(board, rating)
            self.legal[row] = legal_move_mask(board)
            start, stop = int(offsets[row]), int(offsets[row + 1])
            self.targets[row, torch.from_numpy(indices[start:stop].astype(np.int64))] = (
                torch.from_numpy(probabilities[start:stop])
            )
            self.fens.append(str(fens[row]))
            self.ratings.append(rating)

        logger.info("distill: loaded %d positions (%.0f MB of planes)",
                    total, self.planes.numel() * 4 / 1e6)

    def __len__(self) -> int:
        return self.planes.shape[0]

    def __getitem__(self, index: int) -> DistillSample:
        return DistillSample(self.planes[index], self.targets[index], self.legal[index])


def collate(batch: Sequence[DistillSample]) -> Tuple[Tensor, Tensor, Tensor]:
    return (
        torch.stack([sample.planes for sample in batch]),
        torch.stack([sample.target for sample in batch]),
        torch.stack([sample.legal for sample in batch]),
    )


def distillation_loss(logits: Tensor, target: Tensor, legal: Tensor) -> Tensor:
    """Cross-entropy against Maia's distribution, over legal moves only.

    Masking before the log-softmax is what keeps illegal moves at zero
    probability: an unmasked softmax would spend capacity pushing 4000-odd
    illegal logits down, which the mask does for free at inference anyway.
    """
    log_probabilities = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
    return -(target * log_probabilities.nan_to_num(neginf=0.0)).sum(dim=1).mean()


def top_k_agreement(logits: Tensor, target: Tensor, legal: Tensor, k: int) -> float:
    """Fraction of positions where Maia's favourite is in the model's top ``k``."""
    masked = logits.masked_fill(~legal, float("-inf"))
    top = masked.topk(k=min(k, masked.shape[1]), dim=1).indices
    favourite = target.argmax(dim=1, keepdim=True)
    return float((top == favourite).any(dim=1).float().mean().item())


def distill(
    corpus_path: Path,
    checkpoint_path: Path,
    *,
    epochs: int = 12,
    batch_size: int = 256,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.05,
    channels: int = 64,
    blocks: int = 4,
    device: Optional[torch.device] = None,
    seed: int = 20260911,
) -> Dict[str, object]:
    """Train the warm checkpoint. Returns the run's metrics."""
    from src.training.train_dpo import select_device

    torch.manual_seed(seed)
    random.seed(seed)
    target_device = device if device is not None else select_device()

    dataset = DistillationDataset(corpus_path)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = max(1, int(len(indices) * validation_fraction))
    validation_ids, training_ids = indices[:split], indices[split:]

    train_loader: DataLoader[DistillSample] = DataLoader(
        torch.utils.data.Subset(dataset, training_ids),
        batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True,
    )
    validation_loader: DataLoader[DistillSample] = DataLoader(
        torch.utils.data.Subset(dataset, validation_ids),
        batch_size=batch_size, shuffle=False, collate_fn=collate,
    )

    model = TrapPolicyNet(channels=channels, blocks=blocks).to(target_device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, epochs))
    logger.info(
        "distill: %d train / %d val on %s, %s parameters",
        len(training_ids), len(validation_ids), target_device, f"{model.parameter_count:,}",
    )

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for planes, target, legal in train_loader:
            planes = planes.to(target_device)
            target, legal = target.to(target_device), legal.to(target_device)
            logits = model(planes)
            loss = distillation_loss(logits, target, legal)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            running += float(loss.item()) * planes.shape[0]
            seen += planes.shape[0]
        schedule.step()

        metrics = evaluate(model, validation_loader, target_device)
        entry = {"epoch": float(epoch), "loss": round(running / max(1, seen), 4), **metrics}
        history.append(entry)
        logger.info("distill: %s", entry)

    final = history[-1] if history else {}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "state_dict": model.state_dict(),
            "channels": channels,
            "blocks": blocks,
            "input_planes": INPUT_PLANES,
            "policy_size": POLICY_SIZE,
            "records": len(dataset),
            "stage": "distilled",
            "metrics": final,
        },
        checkpoint_path,
    )
    logger.info("distill: wrote %s", checkpoint_path)

    top3 = float(final.get("top3", 0.0))
    if top3 < TARGET_TOP3_ACCURACY:
        logger.warning(
            "distill: top-3 agreement %.3f is below the %.2f gate; the warm start is weak",
            top3, TARGET_TOP3_ACCURACY,
        )
    return {"history": history, "positions": len(dataset), "top3": top3}


@torch.no_grad()
def evaluate(
    model: TrapPolicyNet, loader: DataLoader[DistillSample], device: torch.device
) -> Dict[str, float]:
    """Validation loss and top-1/3/5 agreement with Maia."""
    model.eval()
    totals = {"val_loss": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0}
    seen = 0
    for planes, target, legal in loader:
        planes = planes.to(device)
        target, legal = target.to(device), legal.to(device)
        logits = model(planes)
        count = planes.shape[0]
        totals["val_loss"] += float(distillation_loss(logits, target, legal).item()) * count
        for k in (1, 3, 5):
            totals[f"top{k}"] += top_k_agreement(logits, target, legal, k) * count
        seen += count
    return {key: round(value / max(1, seen), 4) for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.training.distill")
    parser.add_argument("--positions", type=int, default=DEFAULT_POSITIONS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--build-only", action="store_true", help="Label positions, do not train.")
    parser.add_argument("--train-only", action="store_true", help="Train from an existing corpus.")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("chess.engine").setLevel(logging.ERROR)

    if not args.train_only:
        from src.engine import Maia2Evaluator

        rng = random.Random(args.seed)
        books = [engine_config.trap_book_path(), engine_config.standard_book_path()]
        positions = sample_positions(args.positions, rng=rng, books=[b for b in books if b.exists()])
        bands = engine_config.AVAILABLE_MAIA_RATINGS
        # A random band per position, so the rating plane carries real variance
        # rather than a constant the network can ignore.
        ratings = [rng.choice(bands) for _ in positions]
        order = sorted(range(len(positions)), key=lambda i: ratings[i])
        with Maia2Evaluator(bands[0]) as maia:
            build_corpus([positions[i] for i in order], [ratings[i] for i in order], maia, args.corpus)

    if args.build_only:
        return 0

    from src.training.train_dpo import select_device

    distill(
        args.corpus, args.checkpoint,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        device=select_device(args.device), seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
