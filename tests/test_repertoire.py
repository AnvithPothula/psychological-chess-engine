"""The repertoire builder: expectimax over the mined tree, not a lookup."""

from __future__ import annotations

import chess
import pytest

from src.training.repertoire_builder import build_repertoire


def _fen(*moves: str) -> str:
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return board.fen()


def _move(uci: str, games: int = 0, equal: bool = True) -> dict[str, object]:
    return {"uci": uci, "games": games, "equal": equal}


TREE = [
    {"fen": _fen(), "games": 3000,
     "moves": [_move("e2e4", 1000), _move("d2d4", 1000), _move("g2g4", 1000, equal=False)]},
    # after 1.e4 most club players answer e5, which leads to a modest skew
    {"fen": _fen("e2e4"), "games": 1000, "moves": [_move("e7e5", 750), _move("c7c5", 250)]},
    # after 1.d4 most avoid d5, so the bigger skew behind it is rarely reached
    {"fen": _fen("d2d4"), "games": 1000, "moves": [_move("d7d5", 250), _move("g8f6", 750)]},
    # 1.g4 failed the engine gate; the huge skew behind it must stay out of reach
    {"fen": _fen("g2g4"), "games": 1000, "moves": [_move("e7e5", 1000)]},
]
SKEWS = [
    {"fen": _fen("e2e4", "e7e5"), "uci": "g1f3", "skew": 0.10},
    {"fen": _fen("d2d4", "d7d5"), "uci": "c2c4", "skew": 0.20},
    {"fen": _fen("g2g4", "e7e5"), "uci": "f1g2", "skew": 0.90},
]


def test_reach_is_weighed_against_skew_and_unsound_moves_are_never_steered_through() -> None:
    """0.75 x 0.10 beats 0.25 x 0.20, and 1.g4's 0.90 is unreachable by construction."""
    records, values = build_repertoire(TREE, SKEWS)
    book = {(r["fen"], r["uci"]): r["skew"] for r in records}

    assert book[(_fen(), "e2e4")] == pytest.approx(0.075)
    assert (_fen(), "d2d4") not in book, "one move per position: the best one"
    assert (_fen(), "g2g4") not in book
    assert book[(_fen("e2e4", "e7e5"), "g1f3")] == pytest.approx(0.10)
    assert values[chess.WHITE] == pytest.approx((0.075, 0.75))


def test_a_colour_with_no_skew_moves_gets_no_entries() -> None:
    records, values = build_repertoire(TREE, SKEWS)
    assert values[chess.BLACK] == (0.0, 0.0)
    assert all(chess.Board(str(r["fen"])).turn == chess.WHITE for r in records)
