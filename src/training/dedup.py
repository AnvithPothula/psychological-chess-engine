"""Deduplicate the mined preference dataset.

Self-play revisits opening positions constantly, and duplicate preferences do
not merely waste a row: they silently reweight the objective, so a position the
rollout happened to reach forty times counts forty times in the gradient. That
is a bias toward whatever the book plays, which is the opposite of what the
network is meant to learn.

Records collapse on ``(position, opponent_rating)``. The same position against a
different opponent is a genuinely different preference and is kept.

The key is the **EPD**, not the raw FEN the brief specifies. A FEN carries the
halfmove clock and fullmove number, so the identical position reached on move 9
and on move 14 produces two different strings and never collides -- the dedup
would silently do nothing for exactly the repeats it exists to catch. At the
current dataset size this changes no rows; at fifty thousand it will.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, Iterator, Sequence, Tuple, Union

import chess

__all__ = ["DedupStats", "deduplicate", "load_records", "position_key", "main"]

logger = logging.getLogger(__name__)

DEFAULT_INPUT: Final[Path] = Path("build/dpo_pairs.jsonl")
DEFAULT_OUTPUT: Final[Path] = Path("build/dpo_dataset_clean.jsonl")
REQUIRED_FIELDS: Final[Tuple[str, ...]] = ("fen", "chosen", "rejected", "opponent_rating")
RANK_FIELD: Final[str] = "bait_probability"
"""Tie-break: keep the record whose bait the opponent was most likely to take."""


def position_key(fen: str, rating: int) -> Tuple[str, int]:
    """Counter-free identity for a position, paired with the opponent."""
    try:
        return (chess.Board(fen).epd(), rating)
    except ValueError:
        return (fen, rating)  # Unparseable: fall back to the literal string.


@dataclass(frozen=True, slots=True)
class DedupStats:
    read: int
    written: int
    duplicates: int
    malformed: int

    @property
    def duplicate_rate(self) -> float:
        return self.duplicates / self.read if self.read else 0.0


def load_records(path: Path) -> Iterator[Dict[str, object]]:
    """Yield well-formed JSON objects, skipping and counting the rest.

    A partially written final line is normal -- the generator appends and
    flushes per game -- so a bad line is a skip, not a crash.
    """
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("dedup: line %d is not valid JSON, skipping", number)
                continue
            if not isinstance(record, dict) or any(field not in record for field in REQUIRED_FIELDS):
                logger.debug("dedup: line %d is missing required fields, skipping", number)
                continue
            yield record


def deduplicate(sources: Union[Path, Sequence[Path]], destination: Path) -> DedupStats:
    """Collapse duplicates across one or more shards, keeping the strongest bait.

    Parallel rollouts must write separate files -- two processes appending to one
    is a corruption risk for any game that emits more than a page of pairs -- so
    merging shards is the normal case, not an edge case.
    """
    paths = [sources] if isinstance(sources, Path) else list(sources)
    best: Dict[Tuple[str, int], Dict[str, object]] = {}
    read = 0
    raw_lines = 0

    for path in paths:
        if not path.exists():
            logger.warning("dedup: %s does not exist, skipping it", path)
            continue
        with path.open("r", encoding="utf-8") as handle:
            raw_lines += sum(1 for line in handle if line.strip())

    for record in (r for path in paths if path.exists() for r in load_records(path)):
        read += 1
        rating = record["opponent_rating"]
        key = position_key(str(record["fen"]), int(rating) if isinstance(rating, (int, float, str)) else 0)
        incumbent = best.get(key)
        if incumbent is None or _rank(record) > _rank(incumbent):
            best[key] = record
    malformed = raw_lines - read

    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(best.items(), key=lambda item: (item[0][1], item[0][0]))
    with destination.open("w", encoding="utf-8") as sink:
        for _, record in ordered:
            sink.write(json.dumps(record, separators=(",", ":")) + "\n")

    stats = DedupStats(read=read, written=len(best), duplicates=read - len(best), malformed=malformed)
    by_rating = Counter(int(key[1]) for key in best)
    logger.info(
        "dedup: %d read -> %d unique (%d duplicates, %.0f%%; %d malformed)",
        stats.read, stats.written, stats.duplicates, stats.duplicate_rate * 100, stats.malformed,
    )
    logger.info("dedup: per rating band %s", dict(sorted(by_rating.items())))
    return stats


def _rank(record: Dict[str, object]) -> float:
    value = record.get(RANK_FIELD, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.training.dedup")
    parser.add_argument("--input", type=Path, nargs="+", default=[DEFAULT_INPUT],
                        metavar="PATH", help="One or more shards to merge.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    present = [path for path in args.input if path.exists()]
    if not present:
        logger.error("no dataset at %s; run python -m src.training.dpo_generator first",
                     ", ".join(str(path) for path in args.input))
        return 1
    stats = deduplicate(present, args.output)
    logger.info("dedup: wrote %d records to %s", stats.written, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
