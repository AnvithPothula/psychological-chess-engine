"""Live analysis: move scoring, clustered errors, and which comparison is causal."""

from __future__ import annotations

import math
from pathlib import Path

import chess
import pytest

from src.eval.live_analyzer import GameStats, read_jsonl, report, score_game, summarise


def _epd(*sans: str) -> str:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.epd()


def test_a_blunder_is_the_movers_score_before_plus_the_opponents_after() -> None:
    """Scholar's mate: only 3...Nf6?? loses, and the bot's own moves are not scored."""
    line = ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]
    scores = {_epd(*line[:5]): -30, _epd(*line[:6]): 900}
    stats = score_game(
        "g", "skew", "mate", line, chess.WHITE,
        skew_moves={(_epd("e4", "e5"), "d1h5")},
        evaluate=lambda board: scores.get(board.epd(), 0),
    )
    assert stats.opponent_moves == 3
    assert stats.blunders == 1 and stats.cp_lost == 870 and stats.max_error == 870
    assert stats.decisive_ply == 6, "decided by the blunder and never given back"
    assert stats.skew_moves == 1, "exposure is the bot playing the mined move"


def test_the_error_on_a_rate_is_clustered_by_game_not_counted_over_moves() -> None:
    games = [
        GameStats("a", "x", "mate", 10, 0, 0, 0, None, 0),
        GameStats("b", "x", "mate", 10, 2, 0, 0, None, 0),
    ]
    summary = summarise(games)
    assert summary.blunder_rate == pytest.approx(0.1)
    assert summary.blunder_se == pytest.approx(0.1), "binomial over 20 moves would say 0.067"

    same_person = [GameStats(g.game, g.arm, g.status, g.opponent_moves, g.blunders, 0, 0, None, 0,
                             opponent="one") for g in games]
    assert math.isnan(summarise(same_person).blunder_se), "two games, one opponent: no spread yet"


def test_one_game_against_two_reports_no_sigma_and_an_empty_group_no_table() -> None:
    """Seen live: one game against two printed +4.5 sigma, and an empty group -7.5."""
    games = [GameStats("s1", "standard", "mate", 19, 1, 1299, 254, 12, 0, opponent="a"),
             GameStats("s2", "standard", "mate", 17, 3, 1774, 509, 26, 0, opponent="b"),
             GameStats("k1", "skew", "resign", 17, 1, 1284, 955, 25, 0, opponent="c")]
    text = report(games, "standard", "skew")
    max_error_row = next(line for line in text.splitlines() if "max error" in line)
    assert max_error_row.rstrip().endswith("n/a")
    assert "no games in 'exposed' yet" in text


def test_the_report_leads_with_the_arms_and_scales_by_the_exposure_uplift() -> None:
    games = [GameStats(f"s{i}", "standard", "mate", 20, 2, 400, 300, 30, 0) for i in range(4)]
    games += [GameStats(f"k{i}", "skew", "mate", 20, 3, 500, 400, 28, i % 2) for i in range(4)]
    games += [GameStats("ab", "skew", "aborted", 0, 0, 0, 0, None, 1)]
    text = report(games, "standard", "skew")

    assert text.index("BY ARM") < text.index("SCALED") < text.index("confounded")
    assert "50.0% more games playing a skew move" in text, "the aborted game is not counted"
    assert "+10.00" in text, "a 5-point arm difference over 50% exposure is 10 points"


def test_a_half_written_last_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    log = tmp_path / "live.jsonl"
    log.write_text('{"event": "start", "game": "a"}\n{"event": "sta')
    assert [row["game"] for row in read_jsonl(log)] == ["a"]
    assert list(read_jsonl(tmp_path / "missing.jsonl")) == []
