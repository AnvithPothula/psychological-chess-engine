"""Maia-2 opponent model.

Maia-1 is nine separate networks, one per 100-Elo band, served through an Lc0
subprocess. Everything awkward about the old evaluator follows from that: the
``WeightsFile`` hot-swap, the ``ucinewgame`` token rotation that makes a swap
actually take effect, the engine-death supervisor, the cognitive-Elo hysteresis
that exists only to stop the checkpoint thrashing. Maia-2 is one network that
takes a rating as an input, so all of it goes away -- ``set_rating`` is an
assignment, and switching opponents costs nothing.

**It still has a ceiling, just a higher one.** Maia-2 encodes skill as a
categorical embedding whose top bucket is ``(2000, +inf)``, so every rating at
or above 2000 produces a bit-identical distribution. Measured on a fixed
position, top-move probability climbs 0.7065 -> 0.8221 across 1100..1900,
reaches 0.8408 at 2000, and then does not move again at 2100, 2400, or 3200.
Modelling a 2400 therefore still needs ``config.opponent_temperature`` on top,
with the hinge at 2000 rather than 1900. Maia-3 interpolates rating
continuously from 0 to 5000 and has no bucket at all; when it is available it
is a drop-in for this class.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Final, Mapping, Optional, Tuple

import chess

from src import config
from src.engine import EvaluatorError
from src.engine.maia import temperature_softmax
from src.types import MoveDistribution

__all__ = ["Maia2Evaluator", "MAIA2_RATING_CEILING"]

logger = logging.getLogger(__name__)

MAIA2_RATING_CEILING: Final[int] = 2000
"""Above this, Maia-2's skill embedding stops changing. Measured, not documented
upstream: 2000, 2100, 2400 and 3200 return identical distributions."""

_SENTINEL_MOVE: Final[str] = "a1h8"
"""Maia-2 emits this key in every position, always with zero mass. Filtering by
legality catches it anyway; the name is recorded so the extra key is not
mistaken for a bug."""

_DEFAULT_GAME_TYPE: Final[str] = "rapid"


class Maia2Evaluator:
    """Rating-conditioned human move model, satisfying the ``HumanModel`` protocol.

    Holds two ratings because Maia-2 conditions on both seats: ``rating`` is the
    human being modelled, ``opponent_rating`` is the bot sitting across from
    them. The second one matters -- on a fixed position the same 1500 is
    predicted to play the top move 0.634 of the time against a 1100 and 0.757
    against a 2400.
    """

    def __init__(
        self,
        rating: int = config.DEFAULT_MAIA_RATING,
        *,
        opponent_rating: Optional[int] = None,
        game_type: str = _DEFAULT_GAME_TYPE,
        device: str = "cpu",
        temperature: float = config.DEFAULT_MAIA_TEMPERATURE,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.rating = rating
        self.opponent_rating = rating if opponent_rating is None else opponent_rating
        self.temperature = temperature
        self.game_type = game_type
        self.device = device
        self._model: Optional[Any] = None
        self._prepared: Optional[Tuple[Any, ...]] = None
        # The checkpoint is ~280MB of shared state; one process may drive the
        # model from the UI thread and a rollout thread at once.
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def _require_model(self) -> Tuple[Any, Tuple[Any, ...]]:
        """Load the checkpoint on first use. Cached load is ~0.1s."""
        if self._model is None or self._prepared is None:
            try:
                from maia2 import (  # type: ignore[import-untyped]
                    inference, model as maia2_model,
                )
            except ImportError as exc:  # pragma: no cover - install-time failure
                raise EvaluatorError(
                    "maia2 is not installed. Note that its pinned pyzstd does not "
                    "build on Python 3.13: install with --no-deps plus "
                    "'pyzstd>=0.19 gdown einops'."
                ) from exc
            try:
                self._model = maia2_model.from_pretrained(
                    type=self.game_type, device=self.device
                )
                self._prepared = inference.prepare()
            except Exception as exc:
                raise EvaluatorError(f"Could not load Maia-2 ({self.game_type}): {exc}") from exc
            logger.info(
                "maia2: %s model on %s, modelling %d vs %d",
                self.game_type, self.device, self.rating, self.opponent_rating,
            )
        return self._model, self._prepared

    def close(self) -> None:
        """Drop the checkpoint. Idempotent; there is no subprocess to reap."""
        self._model = None
        self._prepared = None

    def __enter__(self) -> Maia2Evaluator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- rating -------------------------------------------------------------

    def set_rating(self, rating: int) -> None:
        """Point the model at a different opponent strength.

        Free, unlike the Maia-1 equivalent: no weights file, no ``ucinewgame``,
        no risk of the engine serving a stale policy for positions it has
        already seen. Ratings above ``MAIA2_RATING_CEILING`` are accepted and
        stored, but the network cannot distinguish them.
        """
        if rating > MAIA2_RATING_CEILING and self.rating <= MAIA2_RATING_CEILING:
            logger.info(
                "maia2: rating %d is above the %d ceiling; the model sees %d. "
                "Sharpening via config.opponent_temperature carries it further.",
                rating, MAIA2_RATING_CEILING, MAIA2_RATING_CEILING,
            )
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

        model, prepared = self._require_model()
        with self._lock:
            try:
                raw, _ = self._infer(model, prepared, board.fen())
            except Exception as exc:
                raise EvaluatorError(f"Maia-2 failed on {board.fen()}: {exc}") from exc

        priors = self._legal_priors(raw, legal)
        scaled = temperature_softmax(
            priors, self.temperature if temperature is None else temperature
        )
        return MoveDistribution(scaled)

    def _infer(
        self, model: Any, prepared: Tuple[Any, ...], fen: str
    ) -> Tuple[Mapping[str, float], float]:
        from maia2 import inference

        result: Tuple[Mapping[str, float], float] = inference.inference_each(
            model, prepared, fen, self.rating, self.opponent_rating
        )
        return result

    @staticmethod
    def _legal_priors(
        raw: Mapping[str, float], legal: list[chess.Move]
    ) -> Dict[chess.Move, float]:
        """Filter to legal moves and renormalise.

        Maia-2's output is neither legality-masked nor exactly normalised: it
        carries one zero-mass sentinel key and sums to about 0.9996. Both are
        handled by rebuilding the distribution over the real legal moves.
        """
        priors = {move: max(0.0, float(raw.get(move.uci(), 0.0))) for move in legal}
        total = sum(priors.values())
        if total <= 0.0:
            uniform = 1.0 / len(legal)
            return {move: uniform for move in legal}
        return {move: value / total for move, value in priors.items()}
