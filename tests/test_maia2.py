"""Maia-2 evaluator: output contract, rating handling, and the 2000 ceiling.

The checkpoint is ~280MB and downloads on first use, so everything that can be
tested without it is tested without it. The one integration test skips when the
checkpoint is not already cached rather than pulling it in CI.
"""

from __future__ import annotations

from pathlib import Path

import chess
import pytest

from src.engine import EvaluatorError
from src.engine.maia2_model import MAIA2_RATING_CEILING, Maia2Evaluator

MODEL_CACHE = Path("maia2_models") / "rapid_model.pt"
needs_checkpoint = pytest.mark.skipif(
    not MODEL_CACHE.exists(), reason="Maia-2 checkpoint not cached; skipping the 280MB download"
)


def test_illegal_keys_are_dropped_and_the_rest_renormalised() -> None:
    """Maia-2 emits a zero-mass 'a1h8' sentinel and sums to about 0.9996."""
    board = chess.Board()
    legal = list(board.legal_moves)
    raw = {move.uci(): 0.01 for move in legal}
    raw["a1h8"] = 0.0
    raw["e7e5"] = 0.5  # legal for Black, not for White at the start

    priors = Maia2Evaluator._legal_priors(raw, legal)

    assert set(priors) == set(legal)
    assert sum(priors.values()) == pytest.approx(1.0)
    assert all(value == pytest.approx(1.0 / len(legal)) for value in priors.values())


def test_a_dead_policy_falls_back_to_uniform() -> None:
    """A model that assigns nothing must not produce a zero distribution."""
    board = chess.Board()
    legal = list(board.legal_moves)
    priors = Maia2Evaluator._legal_priors({}, legal)
    assert sum(priors.values()) == pytest.approx(1.0)
    assert all(value == pytest.approx(1.0 / len(legal)) for value in priors.values())


def test_negative_mass_cannot_survive() -> None:
    """Clamping matters: a negative prior would invert the softmax."""
    board = chess.Board()
    legal = list(board.legal_moves)
    raw = {move.uci(): -1.0 for move in legal}
    raw[legal[0].uci()] = 1.0
    priors = Maia2Evaluator._legal_priors(raw, legal)
    assert priors[legal[0]] == pytest.approx(1.0)
    assert all(value >= 0.0 for value in priors.values())


def test_ratings_above_the_ceiling_are_stored_not_clamped() -> None:
    """The caller's real rating is kept; only the network cannot use it."""
    evaluator = Maia2Evaluator(1500)
    evaluator.set_rating(2400)
    assert evaluator.rating == 2400
    assert MAIA2_RATING_CEILING == 2000


def test_both_seats_are_tracked_independently() -> None:
    """Maia-2 conditions on the opponent too, so the bot's rating is state."""
    evaluator = Maia2Evaluator(1500, opponent_rating=2100)
    assert (evaluator.rating, evaluator.opponent_rating) == (1500, 2100)
    evaluator.set_opponent_rating(1200)
    assert (evaluator.rating, evaluator.opponent_rating) == (1500, 1200)


def test_a_non_positive_temperature_is_refused() -> None:
    with pytest.raises(ValueError):
        Maia2Evaluator(1500, temperature=0.0)


def test_a_finished_position_has_nothing_to_predict() -> None:
    checkmated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert checkmated.is_game_over()
    with pytest.raises(EvaluatorError):
        Maia2Evaluator(1500).predict_move_probabilities(checkmated)


@needs_checkpoint
def test_integration_the_model_is_a_usable_human_model() -> None:
    board = chess.Board("r1bqk2r/pp1nnppp/4p3/1N1pP3/1bpP4/3B1N2/PPP2PPP/R1BQ1K1R w kq - 0 9")
    with Maia2Evaluator(1500) as evaluator:
        distribution = evaluator.predict_move_probabilities(board)

    probabilities = distribution.probabilities
    assert set(probabilities) <= set(board.legal_moves)
    assert sum(probabilities.values()) == pytest.approx(1.0)


@needs_checkpoint
def test_integration_the_model_saturates_at_two_thousand() -> None:
    """The measured ceiling, pinned so a model swap has to notice it."""
    board = chess.Board("r1bqk2r/pp1nnppp/4p3/1N1pP3/1bpP4/3B1N2/PPP2PPP/R1BQ1K1R w kq - 0 9")
    with Maia2Evaluator(1500) as evaluator:
        def top(rating: int) -> float:
            evaluator.set_rating(rating)
            return max(evaluator.predict_move_probabilities(board).probabilities.values())

        assert top(1100) != pytest.approx(top(1900)), "conditioning must do something"
        assert top(2000) == pytest.approx(top(2400))
        assert top(2400) == pytest.approx(top(3200))
        assert top(2000) == pytest.approx(top(9999)), "the top bucket is unbounded"
