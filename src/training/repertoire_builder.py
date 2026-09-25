"""Steer games toward the mined skew positions.

The skew book fired in 2 of 40 arena games from the initial position. Its
entries sit at plies 7-10 of particular lines, and nothing sent the bot down
those lines. This builds the book that does, from the tree the miner walked
(``--tree``), by expectimax over the human moves in it:

- On the bot's turn, the best of a mined skew move here (worth its skew) or an
  engine-equal move the miner followed (worth the position it leads to). Only
  moves that passed the miner's |eval| gate are candidates, so the steering never
  costs the bot the position.
- On the opponent's turn, the average over their replies, weighted by how often
  1100-1700 blitz and rapid players chose each one. The weights are explorer
  counts, so replies the miner did not follow -- rare ones, and ones that failed
  the gate -- keep their share of the denominator and are worth nothing: those
  are the games where the opponent leaves the repertoire.

The value is expected skew under that opponent, and a skew move scores exactly
its skew, so at a node that is both a skew position and a waypoint to a better
one the two compete on one scale. That is why this book replaces skew.bin in
the front slot instead of sitting in front of it: a chain of books always
prefers whichever book comes first, not whichever move is worth more.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import chess

from src.training.polyglot_compiler import compile_book

__all__ = ["build_repertoire", "main"]

logger = logging.getLogger(__name__)

DEFAULT_TREE: Final[Path] = Path("build/skew_tree.jsonl")
DEFAULT_POSITIONS: Final[Path] = Path("build/skew_positions.jsonl")
DEFAULT_OUTPUT: Final[Path] = Path("src/engine/books/repertoire.bin")

Value = Tuple[float, float]
"""Expected skew, and the probability of reaching a skew move at all."""


def build_repertoire(
    tree_rows: Sequence[Mapping[str, Any]], skew_rows: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, object]], Dict[chess.Color, Value]]:
    """Book records (``fen``, ``uci``, value as ``skew``) and each colour's value.

    One record per bot position worth anything: the move with the highest
    expected skew. The value from the initial position is the dosage: the share
    of games that reach a skew move, which sets how many live games a
    measurement needs.
    """
    tree = {chess.Board(str(row["fen"])).epd(): row for row in tree_rows}
    skews: Dict[str, List[Tuple[str, float]]] = {}
    for row in skew_rows:
        epd = chess.Board(str(row["fen"])).epd()
        skews.setdefault(epd, []).append((str(row["uci"]), float(row["skew"])))

    chosen: Dict[str, Tuple[str, str, float]] = {}
    memo: Dict[Tuple[str, chess.Color], Value] = {}

    def child(board: chess.Board, uci: str, bot: chess.Color) -> Value:
        board.push(chess.Move.from_uci(uci))
        try:
            return value(board, bot)
        finally:
            board.pop()

    def value(board: chess.Board, bot: chess.Color) -> Value:
        epd = board.epd()
        if (epd, bot) in memo:
            return memo[(epd, bot)]
        memo[(epd, bot)] = (0.0, 0.0)  # a repetition back into this line is worth nothing
        node = tree.get(epd)
        moves = [move for move in (node["moves"] if node else ()) if move["equal"]]

        if board.turn == bot:
            best, reach, pick = 0.0, 0.0, ""
            for uci, skew in skews.get(epd, ()):
                if skew > best:
                    best, reach, pick = skew, 1.0, uci
            for move in moves:
                expected, probability = child(board, str(move["uci"]), bot)
                if expected > best:
                    best, reach, pick = expected, probability, str(move["uci"])
            if pick:
                chosen[epd] = (board.fen(), pick, best)
            result = (best, reach)
        else:
            total = int(node["games"]) if node else 0
            expected = probability = 0.0
            for move in moves if total else ():
                share = int(move["games"]) / total
                sub_expected, sub_probability = child(board, str(move["uci"]), bot)
                expected += share * sub_expected
                probability += share * sub_probability
            result = (expected, probability)

        memo[(epd, bot)] = result
        return result

    values = {colour: value(chess.Board(), colour) for colour in (chess.WHITE, chess.BLACK)}
    records: List[Dict[str, object]] = [
        {"fen": fen, "uci": uci, "skew": worth} for fen, uci, worth in chosen.values() if worth > 0
    ]
    return records, values


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.repertoire_builder",
        description="Build the opening book that steers games into mined skew positions.",
    )
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for path in (args.tree, args.positions):
        if not path.exists():
            logger.error("no %s; run src.training.skew_miner with --tree first", path)
            return 1

    records, values = build_repertoire(_read_jsonl(args.tree), _read_jsonl(args.positions))
    for colour, (expected, reach) in values.items():
        logger.info(
            "repertoire: as %s, %.1f%% of games reach a skew move, expected skew %+.4f",
            chess.COLOR_NAMES[colour], 100 * reach, expected,
        )
    written = compile_book(records, args.output)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
