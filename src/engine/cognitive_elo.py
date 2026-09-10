"""In-game estimate of how strong the opponent is actually playing.

A Lichess rating is a prior, not an observation. It is an average over months
of games, it lags form, and on any given evening the 1600 sitting opposite may
be playing like a 1200. The searcher's whole edge comes from modelling *this*
opponent, so the Maia checkpoint should track what they are doing now rather
than what their profile says.

Each opponent move is scored on how much it cost them, that centipawn loss is
mapped to the rating it typically implies, and the estimate is pulled toward it
by an exponential moving average. How far it moves depends on how much the
observation is worth: a blunder the opponent model *predicted* is corroborating
evidence about their level, while one it never saw coming says more about the
position being weird than about the player.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final, List, Optional, Sequence, Tuple

import chess

from src import config
from src.types import MoveDistribution

__all__ = ["CognitiveTracker", "MoveObservation", "elo_from_centipawn_loss"]

logger = logging.getLogger(__name__)

# Average centipawn loss against the rating it typically implies. Anchors follow
# the well-known ACPL-by-rating relationship on large online samples; the exact
# numbers are a calibration knob, not a law, and are the first thing to revisit
# if the tracker drifts systematically.
ACPL_ANCHORS: Final[Tuple[Tuple[float, float], ...]] = (
    (200.0, 900.0),
    (120.0, 1100.0),
    (80.0, 1300.0),
    (55.0, 1500.0),
    (38.0, 1700.0),
    (26.0, 1900.0),
    (16.0, 2100.0),
    (8.0, 2400.0),
)

DEFAULT_ALPHA: Final[float] = 0.12
"""EMA weight for a fully-informative move. Roughly a 15-move memory: long
enough to ignore one bad move, short enough to notice a collapse."""

MIN_ELO: Final[float] = 600.0
MAX_ELO: Final[float] = 2800.0

BASE_CONFIDENCE: Final[float] = 0.35
"""Weight floor for a move the opponent model did not anticipate at all."""

BAND_HYSTERESIS_ELO: Final[float] = 80.0
"""How far past a band boundary the estimate must travel before a swap is
suggested. Reloading Maia weights forces a ``ucinewgame`` that flushes Lc0's
cache, so a tracker that thrashes across a boundary would pay that repeatedly
for no change in behaviour."""

QUIET_MOVE_CP: Final[float] = 10.0
"""Losses under this are noise: engine eval jitter, not a human error."""


def elo_from_centipawn_loss(loss: float) -> float:
    """The rating a single move of this quality typically implies.

    Piecewise-linear through :data:`ACPL_ANCHORS`, clamped at both ends. A
    single move is a very noisy rating estimate, which is exactly why the caller
    blends it in slowly rather than trusting it.
    """
    clamped_loss = max(0.0, loss)
    anchors = ACPL_ANCHORS
    if clamped_loss >= anchors[0][0]:
        return anchors[0][1]
    if clamped_loss <= anchors[-1][0]:
        return anchors[-1][1]

    for (high_loss, low_elo), (low_loss, high_elo) in zip(anchors, anchors[1:]):
        if low_loss <= clamped_loss <= high_loss:
            span = high_loss - low_loss
            position = (high_loss - clamped_loss) / span if span else 0.0
            return low_elo + position * (high_elo - low_elo)
    return anchors[-1][1]  # pragma: no cover - the loop is exhaustive


@dataclass(frozen=True, slots=True)
class MoveObservation:
    """One scored opponent move, kept for telemetry and tests."""

    move: chess.Move
    centipawn_loss: float
    predicted_probability: float
    implied_elo: float
    weight: float
    elo_after: float


@dataclass
class CognitiveTracker:
    """EMA estimate of the opponent's playing strength, updated move by move."""

    starting_elo: float
    alpha: float = DEFAULT_ALPHA
    bands: Sequence[int] = field(default=config.AVAILABLE_MAIA_RATINGS)
    _elo: float = field(init=False)
    _observations: List[MoveObservation] = field(init=False, default_factory=list)
    _band: int = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must lie in (0.0, 1.0], got {self.alpha}")
        if not self.bands:
            raise ValueError("at least one Maia band is required")
        self._elo = float(min(max(self.starting_elo, MIN_ELO), MAX_ELO))
        self._band = config.nearest_maia_rating(int(self._elo))

    @property
    def current_elo(self) -> int:
        """Live strength estimate, for the searcher to band its opponent model."""
        return int(round(self._elo))

    @property
    def drift(self) -> float:
        """Elo moved since the start. Negative means playing below their rating."""
        return self._elo - self.starting_elo

    @property
    def observations(self) -> Sequence[MoveObservation]:
        return tuple(self._observations)

    @property
    def suggested_band(self) -> int:
        """Maia band to load, with hysteresis around the boundaries."""
        candidate = config.nearest_maia_rating(self.current_elo)
        if candidate == self._band:
            return self._band
        # Only cross once the estimate is clearly past the midpoint, so a
        # tracker hovering on a boundary does not reload weights every move.
        if abs(self._elo - self._band) > BAND_HYSTERESIS_ELO:
            logger.info(
                "cognitive: opponent playing at ~%d, switching model band %d -> %d",
                self.current_elo, self._band, candidate,
            )
            self._band = candidate
        return self._band

    def update(
        self,
        board: chess.Board,
        opponent_move: chess.Move,
        exact_eval_drop: float,
        maia_probs: Optional[MoveDistribution] = None,
    ) -> MoveObservation:
        """Fold one opponent move into the estimate.

        ``exact_eval_drop`` is what the move cost *them*, in centipawns, as a
        non-negative number: Stockfish's value of the position before the move
        minus its value after, from the opponent's own perspective.
        """
        loss = max(0.0, float(exact_eval_drop))
        predicted = 0.0 if maia_probs is None else maia_probs[opponent_move]
        implied = elo_from_centipawn_loss(loss)

        if loss < QUIET_MOVE_CP:
            # A move that costs nothing is played by everyone; it carries no
            # information about strength and must not drag the estimate upward.
            weight = 0.0
            implied = self._elo
        else:
            # A blunder the model saw coming corroborates the current band; one
            # it missed is more likely to be positional noise than evidence.
            weight = self.alpha * min(1.0, BASE_CONFIDENCE + predicted)

        self._elo = min(max((1.0 - weight) * self._elo + weight * implied, MIN_ELO), MAX_ELO)
        observation = MoveObservation(
            move=opponent_move,
            centipawn_loss=loss,
            predicted_probability=predicted,
            implied_elo=implied,
            weight=weight,
            elo_after=self._elo,
        )
        self._observations.append(observation)
        logger.debug(
            "cognitive: %s cost %.0fcp (p=%.2f) -> implied %.0f, estimate %.0f",
            board.san(opponent_move) if opponent_move in board.legal_moves else opponent_move.uci(),
            loss, predicted, implied, self._elo,
        )
        return observation

    def reset(self, starting_elo: Optional[float] = None) -> None:
        """Start a fresh game, optionally against a different opponent."""
        if starting_elo is not None:
            self.starting_elo = float(starting_elo)
        self._elo = float(min(max(self.starting_elo, MIN_ELO), MAX_ELO))
        self._band = config.nearest_maia_rating(int(self._elo))
        self._observations.clear()
