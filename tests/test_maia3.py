"""Maia-3 evaluator: the rating ceiling is gone, and the top of the range is not.

Integration tests skip when the checkpoint is not in the Hugging Face cache
rather than downloading it during a test run.
"""

from __future__ import annotations

from pathlib import Path

import chess
import pytest

from src.engine import EvaluatorError
from src.engine.maia3_model import MAIA3_MAX_RATING, MAIA3_MIN_RATING, Maia3Evaluator

MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--UofTCSSLab--Maia3-5M"
needs_checkpoint = pytest.mark.skipif(
    not MODEL_CACHE.exists(), reason="Maia-3 checkpoint not cached; skipping the download"
)

POSITION = "r1bqk2r/pp1nnppp/4p3/1N1pP3/1bpP4/3B1N2/PPP2PPP/R1BQ1K1R w kq - 0 9"


def test_ratings_are_clamped_to_the_usable_range() -> None:
    """At 5000 the interpolation reaches its engine anchor and degenerates."""
    assert Maia3Evaluator._clamp(5000) == MAIA3_MAX_RATING
    assert Maia3Evaluator._clamp(0) == MAIA3_MIN_RATING
    assert Maia3Evaluator._clamp(2400) == 2400


def test_the_callers_rating_survives_clamping() -> None:
    """Clamping happens at inference; the real rating stays readable."""
    evaluator = Maia3Evaluator(1500)
    evaluator.set_rating(4200)
    assert evaluator.rating == 4200


def test_both_seats_are_tracked_independently() -> None:
    evaluator = Maia3Evaluator(1500, opponent_rating=2100)
    assert (evaluator.rating, evaluator.opponent_rating) == (1500, 2100)
    evaluator.set_opponent_rating(1200)
    assert evaluator.opponent_rating == 1200


def test_a_non_positive_temperature_is_refused() -> None:
    with pytest.raises(ValueError):
        Maia3Evaluator(1500, temperature=0.0)


def test_a_finished_position_has_nothing_to_predict() -> None:
    checkmated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert checkmated.is_game_over()
    with pytest.raises(EvaluatorError):
        Maia3Evaluator(1500).predict_move_probabilities(checkmated)


@needs_checkpoint
def test_integration_the_output_is_a_legal_distribution() -> None:
    board = chess.Board(POSITION)
    with Maia3Evaluator(1500) as evaluator:
        probabilities = evaluator.predict_move_probabilities(board).probabilities
    assert set(probabilities) <= set(board.legal_moves)
    assert sum(probabilities.values()) == pytest.approx(1.0)


@needs_checkpoint
def test_integration_the_rating_ceiling_is_gone() -> None:
    """The whole point of the migration: 2000, 2400 and 3200 must differ.

    Maia-1 stopped at 1900 and Maia-2 returned bit-identical distributions for
    every rating at or above 2000. Measured here: 0.5309 / 0.5805 / 0.5788.
    """
    board = chess.Board(POSITION)
    with Maia3Evaluator(1500) as evaluator:
        def top(rating: int) -> float:
            evaluator.set_rating(rating)
            return max(evaluator.predict_move_probabilities(board).probabilities.values())

        assert top(2000) != pytest.approx(top(2400))
        assert top(2400) != pytest.approx(top(2000))
        assert top(1500) != pytest.approx(top(3200))


@needs_checkpoint
def test_integration_the_opponent_seat_moves_the_prediction() -> None:
    """A fixed player is modelled differently against a stronger opponent."""
    board = chess.Board(POSITION)
    with Maia3Evaluator(1500) as evaluator:
        def top(oppo: int) -> float:
            evaluator.set_opponent_rating(oppo)
            return max(evaluator.predict_move_probabilities(board).probabilities.values())

        assert top(1500) > top(3200), "a stronger opponent should flatten the prediction"


@needs_checkpoint
def test_integration_move_history_reaches_the_model() -> None:
    """Maia-3 reads seven prior positions; a board carries them in move_stack."""
    played = chess.Board()
    for move in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"):
        played.push_san(move)
    bare = chess.Board(played.fen())  # same position, no history

    with Maia3Evaluator(1500) as evaluator:
        with_history = evaluator.predict_move_probabilities(played).probabilities
        without = evaluator.predict_move_probabilities(bare).probabilities

    assert max(with_history.values()) != pytest.approx(max(without.values()))
