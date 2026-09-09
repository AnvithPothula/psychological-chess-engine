"""Immutable value objects exchanged between the evaluators and the search.

Sign convention (applies to every score in this module):
    **positive always favours White**, regardless of whose turn it is.
The search layer is responsible for flipping to a node-relative view; the
evaluators never hand out side-to-move-relative numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import chess
import chess.engine

from src.config import MATE_SCORE_CP, PROBABILITY_SUM_TOLERANCE, WIN_PROBABILITY_SCALE

__all__ = [
    "CandidateStats",
    "MoveSource",
    "EngineEval",
    "MoveDistribution",
    "PredictedReply",
    "SearchConfig",
    "SearchResult",
    "win_probability",
]


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


class MoveSource(Enum):
    """Where a played move came from. Drives telemetry, not decisions."""

    SEARCH = "search"
    MATE_IN_ONE = "mate-in-1"
    BOOK_TRAP = "trap-book"
    BOOK_STANDARD = "standard-book"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Tuning knobs for :class:`~src.engine.search.AdversarialSearcher`.

    Attributes:
        safety_threshold: ``tau``. A candidate is discarded if any retained human
            reply drops the bot below ``-tau`` centipawns. Guards against playing
            a trap that a competent opponent simply refutes.
        root_margin: ``delta_root``. Candidate moves must score within this many
            centipawns of Stockfish's best root move.
        min_reply_probability: ``p_min``. Human replies rarer than this are pruned.
        target_reply_mass: ``P_target``. Stop adding replies once the retained
            cumulative probability reaches this.
        max_candidates: Hard width bound on the root branching factor. This, not
            ``root_margin``, is what makes search time predictable: in a quiet
            position a 150cp window can admit 15+ moves.
        max_replies: Hard width bound on human replies per candidate, applied
            after ``min_reply_probability`` and ``target_reply_mass``.
        root_depth: Depth for the single MultiPV root scan. Deliberately shallower
            than ``leaf_depth``: the root scan only has to rank candidates, while
            leaves decide utility and safety. MultiPV cost grows steeply with depth.
        leaf_depth: Depth for each expectimax leaf evaluation.
        book_safety_threshold: Deliberately looser floor for *book* moves. A
            search-invented trap has nothing vouching for it, so it must clear
            ``safety_threshold``; a curated gambit has a human accepting its
            objective cost up front, and the measured cost of a real trap
            repertoire runs to roughly -250cp (Halloween, Englund, Stafford).
            This floor exists to catch a corrupt or wrong-sided book entry, not
            to second-guess the repertoire.
        cache_size: Maximum entries held by the transposition cache.
    """

    safety_threshold: int = 180
    root_margin: int = 150
    min_reply_probability: float = 0.02
    target_reply_mass: float = 0.92
    max_candidates: int = 6
    max_replies: int = 6
    root_depth: int = 10
    leaf_depth: int = 12
    book_safety_threshold: int = 300
    cache_size: int = 100_000

    def __post_init__(self) -> None:
        if self.safety_threshold < 0:
            raise ValueError("safety_threshold must be non-negative")
        if self.root_margin < 0:
            raise ValueError("root_margin must be non-negative")
        if not 0.0 <= self.min_reply_probability < 1.0:
            raise ValueError("min_reply_probability must lie in [0.0, 1.0)")
        if not 0.0 < self.target_reply_mass <= 1.0:
            raise ValueError("target_reply_mass must lie in (0.0, 1.0]")
        if self.max_candidates < 1 or self.max_replies < 1:
            raise ValueError("max_candidates and max_replies must be >= 1")
        if self.root_depth < 1 or self.leaf_depth < 1:
            raise ValueError("search depths must be >= 1")
        if self.book_safety_threshold < 0:
            raise ValueError("book_safety_threshold must be non-negative")
        if self.cache_size < 1:
            raise ValueError("cache_size must be >= 1")


@dataclass(frozen=True, slots=True)
class PredictedReply:
    """One human reply the opponent model expects, and where it leads."""

    move: chess.Move
    probability: float
    evaluation: int
    """Bot-relative centipawns of the position after this reply."""


@dataclass(frozen=True, slots=True)
class CandidateStats:
    """Per-candidate telemetry produced while scoring one bot move.

    All scores are **bot relative**: positive is good for the searching side,
    whichever colour that is.
    """

    move: chess.Move
    expected_utility: float
    """Probability-weighted bot-relative score over the retained human replies."""

    worst_case: int
    """Safety floor: the minimum bot-relative score across retained replies."""

    objective_score: int
    """Stockfish's minimax value of the position after this move, at ``leaf_depth``
    -- i.e. what the move is worth against *best* defence rather than likely
    defence. Doubles as the objective half of the safety floor."""

    blunder_trap_delta: float
    """``expected_utility - objective_score``: centipawns the move is expected to
    win purely from human error. Large and positive means a genuine trap."""

    is_safe: bool
    """``min(worst_case, objective_score) >= -safety_threshold``. Both halves are
    required: the human-reply floor alone can be blinded by reply truncation,
    which drops exactly the low-probability refutation that busts a trap."""

    top_replies: Sequence[PredictedReply] = field(default_factory=tuple)
    """The three most probable human replies, descending."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Outcome of one :meth:`~src.engine.search.AdversarialSearcher.search` call."""

    move: chess.Move
    expected_utility: float
    is_trap: bool
    """True when the chosen move is *not* Stockfish's objective best, i.e. the
    psychological model overrode the engine's preference."""

    fallback_triggered: bool
    """True when every candidate failed the safety filter and the searcher fell
    back to Stockfish's top minimax move."""

    candidates: Sequence[CandidateStats]
    """Every scored candidate, unsafe ones included, sorted by utility descending."""

    nodes_evaluated: int
    """Expectimax leaf positions scored, cache hits included."""

    duration_ms: float

    source: MoveSource = MoveSource.SEARCH
    """Which mechanism produced the move. ``is_trap`` describes the *search's*
    reasoning and stays False for book moves, so this is the field to read when
    asking "did that come from the book?"."""

    def summary(self) -> str:
        """One-line human-readable digest, used for logging and test output."""
        tags = [self.source.value] if self.source is not MoveSource.SEARCH else []
        tags += [t for t, on in (("TRAP", self.is_trap), ("FALLBACK", self.fallback_triggered)) if on]
        return (
            f"{self.move.uci()} utility={self.expected_utility:+.1f}cp "
            f"candidates={len(self.candidates)} nodes={self.nodes_evaluated} "
            f"{self.duration_ms:.0f}ms {' '.join(tags)}".strip()
        )
