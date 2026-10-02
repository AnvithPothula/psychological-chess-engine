"""Clock analysis: clock parsing, think time with increment, forcing flags, ranks."""

from __future__ import annotations

import random

import pytest

from src.eval.clock_analysis import clocked_moves, report, sample_position, spearman

MOVETEXT = (
    "1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Nf3 { [%eval 0.3] [%clk 0:02:58] } "
    "2... Nc6 { [%clk 0:02:55] } 3. Bc4 { [%clk 0:02:50] } 3... Nd4 { [%clk 0:02:40] } "
    "4. Nxe5 { [%clk 0:02:45] } 4... Qg5 { [%clk 0:02:20] } 5. Nxf7 { [%clk 0:02:30] } 1-0"
)


def test_clocks_are_read_per_move_and_a_game_missing_any_is_dropped() -> None:
    moves = clocked_moves(MOVETEXT)
    assert [san for san, _ in moves][:3] == ["e4", "e5", "Nf3"]
    assert moves[2] == ("Nf3", 178), "an eval comment before the clock must not hide it"
    assert clocked_moves("1. e4 { [%clk 0:03:00] } 1... e5 2. Nf3 { [%clk 0:02:58] }") == []


def test_think_time_adds_the_increment_and_flags_a_recapture_position() -> None:
    """The clock shown includes the increment, so a 3+2 clock that rises was a fast move."""
    headers = {"TimeControl": "180+2", "Event": "Rated Blitz game", "WhiteElo": "1500",
               "BlackElo": "1400", "Site": "x"}
    moves = clocked_moves(MOVETEXT)
    row = None
    for seed in range(100):
        row = sample_position(headers, moves, random.Random(seed), first_ply=7, last_ply=7)
        if row:
            break
    assert row is not None and row["ply"] == 7, "black's 4...Qg5, after white's 4.Nxe5 capture"
    assert row["think"] == 160 - 140 + 2
    assert row["position"] == "after capture" and row["speed"] == "blitz" and row["elo"] == 1400
    assert sample_position({**headers, "TimeControl": "-"}, moves, random.Random(0),
                           first_ply=2, last_ply=8) is None, "correspondence has no clock"


def test_rank_correlation_handles_ties_and_direction() -> None:
    rho, _ = spearman([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
    assert rho == pytest.approx(-1.0)
    rho, _ = spearman([1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3])
    assert rho == pytest.approx(1.0)


def test_the_report_separates_forcing_positions_from_quiet_ones() -> None:
    rows = [{"speed": "blitz", "position": kind, "src": src, "think": think, "blunder": blunder,
             "remaining": 120}
            for kind in ("quiet", "in check")
            for src, think, blunder in ((1, 20, True), (2, 18, False), (6, 5, False), (8, 4, False))
            for _ in range(5)]
    text = report(rows)
    assert "quiet (n=10 easy, 10 narrow)" in text and "in check" in text
