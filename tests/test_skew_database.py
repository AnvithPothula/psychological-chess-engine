"""Database analysis: PGN parsing, the fixed-position design, and stratification."""

from __future__ import annotations

import io
import json

import chess
import pytest

from src.eval.skew_database import iter_games, mh_difference, report, scan

PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/aaa"]
[WhiteElo "1500"]
[BlackElo "1450"]
[Result "1-0"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%eval 0.3] [%clk 0:03:00] } 2. Nf3?! Nc6 3. Bb5 a6 1-0

[Event "Rated Blitz game"]
[Site "https://lichess.org/bbb"]
[WhiteElo "1500"]
[BlackElo "1450"]
[Result "0-1"]

1. e4 e5 2. Bc4 Nf6 0-1

[Event "Rated Bullet game"]
[Site "https://lichess.org/ccc"]
[WhiteElo "1500"]
[BlackElo "1450"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Rated Rapid game"]
[Site "https://lichess.org/ddd"]
[WhiteElo "1900"]
[BlackElo "1450"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""


def test_movetext_is_read_without_clocks_evals_glyphs_or_move_numbers() -> None:
    games = list(iter_games(iter(PGN.splitlines(keepends=True))))
    assert len(games) == 4
    headers, sans = games[0]
    assert headers["Site"] == "https://lichess.org/aaa"
    assert sans == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]


def test_scan_keeps_only_the_band_and_speeds_and_splits_at_the_mined_position() -> None:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    targets = {board.epd(): {"g1f3"}}
    sink = io.StringIO()
    counts = scan(iter_games(iter(PGN.splitlines(keepends=True))), targets, sink,
                  min_elo=1100, max_elo=1700, per_group=10, max_ply=11)

    rows = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert counts == {"read": 4, "eligible": 2, "kept": 2}, "bullet and an 1900 player are out"
    assert [(r["site"][-3:], r["treated"], r["played"]) for r in rows] == [
        ("aaa", True, "g1f3"), ("bbb", False, "f1c4"),
    ]
    assert rows[0]["mover"] == "white" and rows[0]["mover_elo"] == 1500
    assert rows[0]["moves"][0] == "Nf3", "scoring starts with the move at the position"


def _row(epd: str, treated: bool, blunders: int, moves: int, mover: int = 1500) -> dict[str, object]:
    return {"epd": epd, "treated": treated, "blunders": blunders, "opponent_moves": moves,
            "mover_elo": mover, "opponent_elo": 1500, "cp_lost": 0, "max_error": 0, "mover_score": 0.5}


def test_stratifying_removes_a_difference_that_is_only_which_position_was_reached() -> None:
    """Simpson: treated games sit mostly in the blunder-heavy position, and
    within each position there is no difference at all."""
    rows = (
        [_row("sharp", True, 3, 10)] * 9 + [_row("sharp", False, 3, 10)] * 1
        + [_row("quiet", True, 1, 10)] * 1 + [_row("quiet", False, 1, 10)] * 9
    )
    blunders, moves = (lambda r: float(r["blunders"])), (lambda r: float(r["opponent_moves"]))
    assert mh_difference(rows, blunders, moves) == pytest.approx(0.0)

    crude_treated = sum(r["blunders"] for r in rows if r["treated"]) / 100
    crude_control = sum(r["blunders"] for r in rows if not r["treated"]) / 100
    assert crude_treated - crude_control == pytest.approx(0.16), "the crude gap is all confounding"


def test_a_real_within_position_difference_survives_stratification() -> None:
    rows = [_row("p", True, 2, 10)] * 20 + [_row("p", False, 1, 10)] * 20
    estimate = mh_difference(rows, lambda r: float(r["blunders"]), lambda r: float(r["opponent_moves"]))
    assert estimate == pytest.approx(0.1)
    assert "stratified" in report(rows)
