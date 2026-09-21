"""Pack mined skew entries into a Polyglot book the existing reader can open.

``OpeningBook`` already reads two books and filters entries by a nine-bit rating
mask packed into Polyglot's ``learn`` field, so a skew book is a third file in
the same format rather than a new mechanism. Reusing the format also means the
book stays readable by any standard Polyglot tool.

Weight carries the magnitude of the skew, because the reader chooses among
legal entries in proportion to weight. A move eight points above baseline
should be played more often than one six points above it, and mapping skew to
weight linearly is what makes that happen without a second selection rule.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
from pathlib import Path
from typing import Dict, Final, List, Optional, Sequence, Tuple

import chess
import chess.polyglot

from src import config

__all__ = ["compile_book", "skew_weight", "band_mask", "main"]

logger = logging.getLogger(__name__)

ENTRY_STRUCT: Final[struct.Struct] = struct.Struct(">QHHI")
"""Polyglot: 8-byte Zobrist key, 2-byte move, 2-byte weight, 4-byte learn."""

DEFAULT_INPUT: Final[Path] = Path("build/skew_positions.jsonl")
DEFAULT_OUTPUT: Final[Path] = Path("src/engine/books/skew.bin")

MIN_WEIGHT: Final[int] = 1
MAX_WEIGHT: Final[int] = 4096
"""Polyglot weight is 16-bit. Staying well under the limit leaves room for a
book to be merged with another without rescaling."""

WEIGHT_PER_POINT: Final[float] = 40_000.0
"""Weight per unit of skew. Six points of skew (the mining floor) lands near
240 and eighteen points (the strongest positions the source study reports)
near 720, so the range in play spans about a factor of three -- enough for the
reader's weighted choice to prefer the better lines without ignoring the rest."""


def skew_weight(skew: float) -> int:
    """Polyglot weight for a skew, clamped into the representable range."""
    return max(MIN_WEIGHT, min(MAX_WEIGHT, int(round(skew * WEIGHT_PER_POINT))))


def band_mask(ratings: Sequence[int]) -> int:
    """Nine-bit mask over ``config.AVAILABLE_MAIA_RATINGS``.

    Zero means unbanded and the reader treats it as eligible everywhere, which
    is the right default for a book mined across a rating range rather than for
    one band.
    """
    mask = 0
    for rating in ratings:
        nearest = config.nearest_maia_rating(rating)
        mask |= 1 << config.AVAILABLE_MAIA_RATINGS.index(nearest)
    return mask


def polyglot_raw_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a move the way Polyglot does.

    Castling is stored as king-takes-own-rook, not as the two-square king move
    python-chess reports, so a book written with the naive encoding produces
    entries no reader will ever match.
    """
    to_square = move.to_square
    if board.is_castling(move):
        rooks = board.occupied_co[board.turn] & board.rooks
        to_square = chess.msb(rooks) if move.to_square > move.from_square else chess.lsb(rooks)
    promotion = 0 if move.promotion is None else move.promotion - 1
    return (
        (promotion << 12)
        | (chess.square_rank(move.from_square) << 9)
        | (chess.square_file(move.from_square) << 6)
        | (chess.square_rank(to_square) << 3)
        | chess.square_file(to_square)
    )


def compile_book(
    records: Sequence[Dict[str, object]], destination: Path, *, ratings: Sequence[int] = ()
) -> int:
    """Write ``destination`` from mined records. Returns the entry count.

    Duplicate (position, move) pairs keep the larger weight rather than summing:
    the same move can be mined twice through different transpositions, and
    adding the weights would count one measurement as two.
    """
    entries: Dict[Tuple[int, int], int] = {}
    mask = band_mask(ratings)
    skipped = 0

    for record in records:
        try:
            board = chess.Board(str(record["fen"]))
            move = chess.Move.from_uci(str(record["uci"]))
            skew = float(record["skew"])  # type: ignore[arg-type]
        except (KeyError, ValueError):
            skipped += 1
            continue
        if move not in board.legal_moves:
            skipped += 1
            continue

        key = (chess.polyglot.zobrist_hash(board), polyglot_raw_move(board, move))
        weight = skew_weight(skew)
        entries[key] = max(entries.get(key, 0), weight)

    if skipped:
        logger.warning("compile: skipped %d unusable records", skipped)

    packed = [
        ENTRY_STRUCT.pack(key, raw_move, weight, mask)
        for (key, raw_move), weight in sorted(entries.items())
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(destination.suffix + ".tmp")
    staging.write_bytes(b"".join(packed))
    staging.replace(destination)
    logger.info("compile: wrote %d entries to %s", len(packed), destination)
    return len(packed)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.polyglot_compiler",
        description="Pack mined skew entries into a Polyglot book.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ratings", type=int, nargs="*", default=[],
                        help="Rating bands to mark eligible. Empty means unbanded.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    if not args.input.exists():
        logger.error("no mined positions at %s; run src.training.skew_miner first", args.input)
        return 1

    records = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    written = compile_book(records, args.output, ratings=args.ratings)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
