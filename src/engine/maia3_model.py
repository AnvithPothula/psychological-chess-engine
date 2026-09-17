"""Maia-3 opponent model.

Maia-1 banded skill into nine checkpoints and stopped at 1900. Maia-2 folded
them into one network but still bucketed the top of the range, so everything at
or above 2000 came back bit-identical. Maia-3 does neither: it interpolates the
rating embedding continuously, ``e_k = g*e_weak + (1-g)*e_strong`` with
``g = (5000-k)/5000``, so there is no bucket to fall into.

Measured on a fixed position, top-move policy at SelfElo 2000/2400/3200 is
0.5309 / 0.5805 / 0.5788, and the OppoElo sweep falls monotonically
0.5526 -> 0.4200 across 1500..3200. Distinct values above 2000 in both
directions, which is what the previous two models could not produce.

**The interpolation has an unusable top end.** At SelfElo 5000 the distribution
collapses to near-uniform noise -- top move 0.0858 on the same position where
3200 gives 0.5788. 5000 is the ``e_strong`` anchor, not a chess player, so
ratings are clamped to ``MAIA3_MAX_RATING``.

Because the ceiling is gone, the rating-conditioned sharpening that
``config.opponent_temperature`` performed for Maia-1 and Maia-2 is no longer
needed and has been removed with this model.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, Final, List, Optional

import chess
import torch

from src import config
from src.engine import EvaluatorError
from src.engine.maia import temperature_softmax
from src.types import MoveDistribution

__all__ = ["Maia3Evaluator", "MAIA3_MAX_RATING"]

logger = logging.getLogger(__name__)

MAIA3_MAX_RATING: Final[int] = 3200
"""Highest rating passed to the network. The embedding interpolates towards an
engine-level anchor at 5000 which is not a playable opponent: at 5000 the policy
degenerates to near-uniform. 3200 is above every human Lichess rating and still
produces a sharp, sane distribution."""

MAIA3_MIN_RATING: Final[int] = 100
"""The other end of the same interpolation, kept off zero for the same reason."""

_DEFAULT_MODEL: Final[str] = "maia3-5m"
"""5M reaches 55.4% move-matching against Maia-2's 52.0%, and runs on CPU in
3.4ms. The 23M and 79M variants buy roughly one more point for several times
the latency, which a per-node opponent model cannot afford."""


class Maia3Evaluator:
    """Rating-conditioned human move model, satisfying the ``HumanModel`` protocol.

    Conditions on both seats: ``rating`` is the human being modelled,
    ``opponent_rating`` is the bot facing them.
    """

    def __init__(
        self,
        rating: int = config.DEFAULT_MAIA_RATING,
        *,
        opponent_rating: Optional[int] = None,
        model: str = _DEFAULT_MODEL,
        device: str = "cpu",
        temperature: float = config.DEFAULT_MAIA_TEMPERATURE,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.rating = rating
        self.opponent_rating = rating if opponent_rating is None else opponent_rating
        self.temperature = temperature
        self.model_name = model
        self.device = device
        self._engine: Optional[Any] = None
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def _require_engine(self) -> Any:
        """Load the checkpoint on first use."""
        if self._engine is None:
            try:
                from maia3.uci import Maia3UCIEngine, parse_args  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover - install-time failure
                raise EvaluatorError(
                    "maia3 is not installed. It is not on PyPI: clone "
                    "github.com/CSSLab/maia3 and 'pip install .' from there."
                ) from exc
            try:
                engine = Maia3UCIEngine(
                    parse_args(["--model", self.model_name, "--device", self.device, "--no-use-amp"])
                )
                engine.ensure_model_loaded()
            except Exception as exc:
                raise EvaluatorError(f"Could not load Maia-3 ({self.model_name}): {exc}") from exc
            self._engine = engine
            logger.info(
                "maia3: %s on %s, modelling %d vs %d",
                self.model_name, self.device, self.rating, self.opponent_rating,
            )
        return self._engine

    def close(self) -> None:
        """Drop the checkpoint. Idempotent; there is no subprocess to reap."""
        self._engine = None

    def __enter__(self) -> Maia3Evaluator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- rating -------------------------------------------------------------

    @staticmethod
    def _clamp(rating: int) -> int:
        return max(MAIA3_MIN_RATING, min(MAIA3_MAX_RATING, rating))

    def set_rating(self, rating: int) -> None:
        """Point the model at a different opponent strength. An assignment."""
        self.rating = rating

    def set_opponent_rating(self, rating: int) -> None:
        """Set the rating of the player facing the modelled human."""
        self.opponent_rating = rating

    # -- inference ----------------------------------------------------------

    def predict_move_probabilities(
        self, board: chess.Board, *, temperature: Optional[float] = None
    ) -> MoveDistribution:
        """Human move distribution over the legal moves of ``board``."""
        legal = list(board.legal_moves)
        if not legal:
            raise EvaluatorError(f"No legal moves to predict in {board.fen()}")

        engine = self._require_engine()
        with self._lock:
            try:
                priors = self._policy(engine, board)
            except Exception as exc:
                raise EvaluatorError(f"Maia-3 failed on {board.fen()}: {exc}") from exc

        scaled = temperature_softmax(
            priors, self.temperature if temperature is None else temperature
        )
        return MoveDistribution(scaled)

    @torch.no_grad()
    def _policy(self, engine: Any, board: chess.Board) -> Dict[chess.Move, float]:
        """One forward pass, policy head only.

        The packaged ``score_moves`` also runs the value head over every
        candidate board to produce WDL for UCI output. That second batched pass
        costs 64.6ms against 3.4ms for the policy alone, for a number the
        opponent model never reads.
        """
        from maia3.dataset import (  # type: ignore[import-untyped]
            get_legal_moves_mask, tokenize_board,
        )

        engine.board = board
        engine.history = self._history(engine, board, tokenize_board)

        mask = get_legal_moves_mask(board, engine.all_moves_dict)
        tokens = engine._tokens_from_history(engine.history).unsqueeze(0).to(engine.cfg.device)
        ratings = (self._clamp(self.rating), self._clamp(self.opponent_rating))
        seats = [
            torch.tensor([value], dtype=torch.long, device=engine.cfg.device) for value in ratings
        ]

        logits_move, _value, _ = engine.model(tokens, seats[0], seats[1])
        logits = logits_move[0].float().masked_fill(~mask.to(engine.cfg.device), float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)

        priors: Dict[chess.Move, float] = {}
        for index in torch.nonzero(mask).flatten().tolist():
            move = engine._move_from_index(index)
            if move is not None:
                priors[move] = float(probabilities[index])
        if not priors:
            raise EvaluatorError(f"Maia-3 masked every legal move in {board.fen()}")
        return priors

    @staticmethod
    def _history(engine: Any, board: chess.Board, tokenize: Any) -> Deque[Any]:
        """Recent positions, newest last.

        Maia-3 reads the last seven positions as well as the current one, and
        the paper measures that history as worth 1.4 points of move-matching.
        The protocol only hands us a board, but a board carries its own
        ``move_stack``, so the history is reconstructable rather than padded.
        """
        span: int = engine.cfg.history
        replay = board.copy(stack=True)
        positions: List[chess.Board] = [replay.copy(stack=False)]
        for _ in range(span):
            if not replay.move_stack:
                break
            replay.pop()
            positions.append(replay.copy(stack=False))
        positions.reverse()
        return deque((tokenize(position) for position in positions), maxlen=span)
