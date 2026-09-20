"""Arena bookkeeping: aggregation, worker sizing, and arm isolation.

The games themselves need two engines and a checkpoint, so they are not run
here. What is tested is everything that could silently corrupt a result --
counting moves as trials, splitting work without losing a game, and refusing to
let an arm quietly become its own control.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.engine.bot_factory import ARENA_SEARCH, ARMS, BASELINE, TRAP, BotSpec
from src.eval.arena import GameResult, _chunks, safe_workers, summarise


def _result(arm: str, game: int, *, moves: int, blunders: int, cp: int, outcome: float) -> GameResult:
    return GameResult(
        arm=arm, game=game, bot_white=game % 2 == 0, outcome=outcome,
        adjudicated=False, opponent_moves=moves, opponent_blunders=blunders,
        opponent_cp_lost=cp, plies=60,
    )


def test_blunder_rate_counts_moves_not_games() -> None:
    """A 40-move game and a 4-move game are not one trial each.

    Averaging per-game rates would weight a four-move game as heavily as a
    forty-move one, which is how a short decisive game can swing an arm.
    """
    results = [
        _result("trap", 0, moves=40, blunders=4, cp=400, outcome=1.0),
        _result("trap", 1, moves=4, blunders=2, cp=200, outcome=0.0),
    ]
    summary = summarise("trap", results, seconds=1.0)

    assert summary.opponent_moves == 44
    assert summary.blunder_rate == pytest.approx(6 / 44)
    assert summary.blunder_rate != pytest.approx((4 / 40 + 2 / 4) / 2), "per-game mean is wrong"


def test_the_standard_error_is_binomial_over_moves() -> None:
    results = [_result("trap", i, moves=100, blunders=15, cp=1000, outcome=0.5) for i in range(4)]
    summary = summarise("trap", results, seconds=1.0)

    expected = math.sqrt(0.15 * 0.85 / 400)
    assert summary.blunder_rate == pytest.approx(0.15)
    assert summary.blunder_rate_stderr == pytest.approx(expected)


def test_an_arm_with_no_opponent_moves_does_not_divide_by_zero() -> None:
    summary = summarise("trap", [_result("trap", 0, moves=0, blunders=0, cp=0, outcome=1.0)], 1.0)
    assert summary.blunder_rate == 0.0
    assert summary.mean_cp_lost == 0.0
    assert summary.blunder_rate_stderr == 0.0


def test_summarising_nothing_is_not_an_error() -> None:
    summary = summarise("trap", [], seconds=0.0)
    assert summary.games == 0 and summary.score == 0.0


def test_every_game_is_dealt_exactly_once() -> None:
    """A lost or duplicated game silently changes the sample size."""
    for games, workers in ((100, 7), (4, 8), (1, 4), (13, 3)):
        buckets = _chunks(games, workers)
        dealt = [game for bucket in buckets for game in bucket]
        assert sorted(dealt) == list(range(games)), (games, workers)
        assert all(bucket for bucket in buckets), "an empty bucket would start a pool for nothing"


def test_colours_are_spread_across_workers() -> None:
    """Round-robin, so one worker does not draw only White."""
    buckets = _chunks(100, 4)
    for bucket in buckets:
        whites = sum(1 for game in bucket if game % 2 == 0)
        assert 0 < whites < len(bucket), "a worker should see both colours"


def test_worker_count_is_bounded_and_respects_an_explicit_request() -> None:
    assert safe_workers(3) == 3
    assert safe_workers(0) == 1, "a pool of zero would never run"
    assert safe_workers(-5) == 1
    automatic = safe_workers(None)
    assert 1 <= automatic <= (32 if automatic else 1)


def test_the_arms_differ_only_in_where_candidates_come_from() -> None:
    """If the arms differ in depth too, the result measures the depth."""
    assert BASELINE.search is TRAP.search is ARENA_SEARCH
    assert BASELINE.use_scorer is False and TRAP.use_scorer is True
    assert set(ARMS) == {"baseline", "trap"}


def test_a_missing_scorer_is_fatal_rather_than_a_silent_downgrade() -> None:
    """An arm that quietly becomes the control produces a fake null result."""
    from src.engine.policy_generator import PolicyUnavailableError
    from src.engine.scorer_proposer import ScorerCandidateProposer

    with pytest.raises(PolicyUnavailableError):
        ScorerCandidateProposer.__new__(ScorerCandidateProposer)._load_scorer(
            Path("models/does-not-exist.pth"), __import__("torch").device("cpu")
        )


def test_a_spec_carries_its_own_checkpoints() -> None:
    spec = BotSpec(name="custom", description="x", use_scorer=True,
                   scorer_path=Path("a.pth"), prior_path=Path("b.pth"))
    assert spec.scorer_path == Path("a.pth") and spec.prior_path == Path("b.pth")
