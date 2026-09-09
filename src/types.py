"""Immutable value objects exchanged between the evaluators and the search.

Sign convention (applies to every score in this module):
    **positive always favours White**, regardless of whose turn it is.
The search layer is responsible for flipping to a node-relative view; the
evaluators never hand out side-to-move-relative numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

import chess
import chess.engine

from src.config import MATE_SCORE_CP, PROBABILITY_SUM_TOLERANCE, WIN_PROBABILITY_SCALE

__all__ = ["EngineEval", "MoveDistribution", "win_probability"]


def win_probability(centipawns: float) -> float:
    """Map a White-relative centipawn score onto ``[0.0, 1.0]``.

    Standard logistic used across chess tooling: ``1 / (1 + 10^(-cp/400))``.
    0.5 is a dead-equal position, 1.0 is a won game for White.
    """
    return 1.0 / (1.0 + math.pow(10.0, -centipawns / WIN_PROBABILITY_SCALE))


@dataclass(frozen=True, slots=True)
class EngineEval:
    """Ground-truth evaluation of a position from White's point of view.

    Attributes:
        centipawns: White-relative score. Mates are saturated to
            ``±MATE_SCORE_CP`` (offset by distance) so the value stays orderable.
        is_mate: True when the engine reported a forced mate.
        mate_in: Signed distance to mate in moves (positive: White mates,
            negative: Black mates). ``None`` unless ``is_mate``.
        win_probability: ``centipawns`` squashed to ``[0.0, 1.0]``; exactly 1.0
            or 0.0 for a forced mate.
    """

    centipawns: int
    is_mate: bool
    mate_in: Optional[int]
    win_probability: float

    def __post_init__(self) -> None:
        if self.is_mate != (self.mate_in is not None):
            raise ValueError("mate_in must be set if and only if is_mate is True")
        if not 0.0 <= self.win_probability <= 1.0:
            raise ValueError(f"win_probability out of range: {self.win_probability}")

    @classmethod
    def from_pov_score(cls, score: chess.engine.PovScore) -> EngineEval:
        """Build from a python-chess score, normalising to White-positive."""
        white = score.white()
        centipawns = white.score(mate_score=MATE_SCORE_CP)
        if centipawns is None:  # pragma: no cover - mate_score makes this total
            raise ValueError(f"Unscorable engine score: {score!r}")

        mate_in = white.mate()
        if mate_in is not None:
            return cls(
                centipawns=centipawns,
                is_mate=True,
                mate_in=mate_in,
                win_probability=1.0 if centipawns > 0 else 0.0,
            )
        return cls(
            centipawns=centipawns,
            is_mate=False,
            mate_in=None,
            win_probability=win_probability(centipawns),
        )


@dataclass(frozen=True, slots=True)
class MoveDistribution:
    """A probability distribution over the legal moves of one position.

    Construction validates rather than normalises: the mapping must already sum
    to 1.0 within ``PROBABILITY_SUM_TOLERANCE``. Producers normalise; this type
    refuses to silently paper over a broken producer.
    """

    probabilities: Mapping[chess.Move, float]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("MoveDistribution requires at least one move")

        for move, probability in self.probabilities.items():
            if not 0.0 <= probability <= 1.0 or math.isnan(probability):
                raise ValueError(f"Probability for {move.uci()} out of range: {probability}")

        total = math.fsum(self.probabilities.values())
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"Probabilities must sum to 1.0, got {total!r}")

        object.__setattr__(self, "probabilities", MappingProxyType(dict(self.probabilities)))

    def __getitem__(self, move: chess.Move) -> float:
        return self.probabilities.get(move, 0.0)

    def __len__(self) -> int:
        return len(self.probabilities)

    @property
    def most_likely(self) -> chess.Move:
        """The move the opponent model considers most probable."""
        return max(self.probabilities, key=lambda move: self.probabilities[move])
