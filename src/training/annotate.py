"""Attach per-move engine features to mined preference pairs.

Matilda's ablation credits its entire high-rating gain to engine features on
candidate moves rather than to parameter count, so a late-fusion re-ranker needs
a centipawn score, a loss against best, and a rank for *each* move in a pair.
Mining never recorded those. It logged ``objective_cost_cp`` for the chosen move
only, and nothing at all for the rejected one, because at the time the only
consumer was a CNN that read the board and the two move indices.

One MultiPV scan over every legal move supplies all of it, and the scan is the
same one ``blunder_potential`` uses -- depth 6 was measured at 0.015 mean
absolute error against a depth-12 reference for roughly a quarter of the cost.

Worth knowing before trusting ``rank`` and ``is_top_choice``: the rejected move
is the search's *expectimax* best, not the engine's. Measured over 40 sampled
pairs, it coincides with Stockfish's top move only 42% of the time against 25%
for the chosen move, and the chosen move's engine rank runs from 1 to 11 with a
median of 3. The features are informative rather than a restatement of the
label, which is the only reason they are worth computing.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import chess

from src.engine.search import _rank_mover_relative
from src.engine.stockfish import StockfishEvaluator

__all__ = ["MoveFeatures", "annotate_record", "annotate", "main"]

logger = logging.getLogger(__name__)

DEFAULT_INPUT: Final[Path] = Path("build/dpo_dataset_clean.jsonl")
DEFAULT_OUTPUT: Final[Path] = Path("build/dpo_dataset_annotated.jsonl")
DEFAULT_DEPTH: Final[int] = 6
"""Matches the blunder-potential scan: 0.015 mean absolute error against a
depth-12 reference at a quarter of depth 8's cost."""

CP_SCALE: Final[float] = 100.0
"""Centipawns are divided by this before they reach the network. A raw -900
alongside a rank of 3 would dominate the first gradient step outright."""

CP_CLAMP: Final[float] = 10.0
"""Scaled scores are clamped to +/-10 pawns. Mate scores saturate at six
figures of centipawns and would otherwise swamp every other feature."""


@dataclass(frozen=True, slots=True)
class MoveFeatures:
    """Engine evidence for one move, from the mover's point of view."""

    centipawns: int
    """Mover-relative score of the position this move reaches."""

    loss_vs_best: int
    """Centipawns given up against the engine's preferred move. Zero when this
    move *is* the engine's choice; never negative."""

    rank: int
    """1-based position in the engine's ordering."""

    is_top_choice: bool
    legal_moves: int

    def as_vector(self) -> List[float]:
        """Normalised features, in the order the network consumes them."""
        scale = CP_SCALE
        return [
            max(-CP_CLAMP, min(CP_CLAMP, self.centipawns / scale)),
            max(-CP_CLAMP, min(CP_CLAMP, self.loss_vs_best / scale)),
            self.rank / max(1, self.legal_moves),
            1.0 if self.is_top_choice else 0.0,
        ]


FEATURE_WIDTH: Final[int] = 4
"""Length of :meth:`MoveFeatures.as_vector`."""


def _features(
    ranked: List[Tuple[chess.Move, int]], move: chess.Move
) -> Optional[MoveFeatures]:
    best = ranked[0][1]
    for position, (candidate, centipawns) in enumerate(ranked, start=1):
        if candidate == move:
            return MoveFeatures(
                centipawns=centipawns,
                loss_vs_best=max(0, best - centipawns),
                rank=position,
                is_top_choice=position == 1,
                legal_moves=len(ranked),
            )
    return None


def annotate_record(
    evaluator: StockfishEvaluator, record: Dict[str, Any], *, depth: int
) -> Optional[Dict[str, Any]]:
    """Add ``chosen_features`` and ``rejected_features`` to one mined pair.

    Returns ``None`` when the record cannot be scored -- an unparsable FEN, a
    move that is not legal in it, or a position with nothing to search. Those
    are dropped rather than filled with zeros, which would teach the re-ranker
    that a broken record looks like a balanced position.
    """
    try:
        board = chess.Board(str(record["fen"]))
        chosen = board.parse_san(str(record["chosen"]))
        rejected = board.parse_san(str(record["rejected"]))
    except (KeyError, ValueError):
        return None
    if chosen == rejected or not board.legal_moves:
        return None

    ranked = _rank_mover_relative(
        evaluator.analyse_root_moves(board, depth=depth, multipv=board.legal_moves.count()),
        board.turn,
    )
    if not ranked:
        return None

    chosen_features = _features(ranked, chosen)
    rejected_features = _features(ranked, rejected)
    if chosen_features is None or rejected_features is None:
        return None

    annotated = dict(record)
    annotated["chosen_features"] = chosen_features.as_vector()
    annotated["rejected_features"] = rejected_features.as_vector()
    return annotated


def annotate(source: Path, destination: Path, *, depth: int = DEFAULT_DEPTH) -> int:
    """Annotate every record in ``source``. Returns the number written."""
    records = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    logger.info("annotate: %d records from %s at depth %d", len(records), source, depth)

    destination.parent.mkdir(parents=True, exist_ok=True)
    written = dropped = 0
    started = time.perf_counter()
    with StockfishEvaluator() as evaluator, destination.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            annotated = annotate_record(evaluator, record, depth=depth)
            if annotated is None:
                dropped += 1
                continue
            handle.write(json.dumps(annotated) + "\n")
            written += 1
            if index % 1000 == 0:
                rate = index / max(1e-9, time.perf_counter() - started)
                remaining = (len(records) - index) / max(1e-9, rate)
                logger.info(
                    "annotate: %d/%d (%.0f/s, %.0f min left)",
                    index, len(records), rate, remaining / 60,
                )

    logger.info(
        "annotate: wrote %d records to %s (%d dropped) in %.0fs",
        written, destination, dropped, time.perf_counter() - started,
    )
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.annotate",
        description="Attach per-move Stockfish features to mined preference pairs.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help="MultiPV depth. 6 matches the blunder-potential scan.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.input.exists():
        logger.error("annotate: %s does not exist", args.input)
        return 1
    return 0 if annotate(args.input, args.output, depth=args.depth) else 1


if __name__ == "__main__":
    raise SystemExit(main())
