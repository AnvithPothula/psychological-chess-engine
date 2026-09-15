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
        proposal_count: Moves asked of the neural proposer, when one is attached.
        max_proposals: Cap on how many of those may join the candidate list.
            Separate from ``proposal_count`` because the proposer may return
            moves Stockfish already covered, which cost nothing to drop.
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
    proposal_count: int = 3
    max_proposals: int = 3
    cache_size: int = 100_000

    blunder_threshold: int = 200
    """Centipawn loss against best play that makes a reply a *blunder*, in the
    Anderson-Kleinberg sense. 200cp is a clear piece-or-more error."""

    beta_depth: int = 6
    """Depth for the blunder-potential scan. Measured over 25 mined positions
    against a depth-12 reference: depth 6 costs 29ms and lands within 0.015
    mean absolute error, depth 8 costs 109ms for no accuracy gain, and depth 10
    costs 584ms and is *worse*. Classifying a 200cp loss is a coarse call."""

    beta_weight: float = 0.0
    """Gamma on blunder potential. **Measured not to work, and left at zero.**

    Anderson et al. measure beta as skill-*free* danger: the chance a uniformly
    random mover errs. Steering by it does move the search -- over 30 self-play
    games against maia2@1700, mean beta faced rose 0.397 -> 0.495 as gamma went
    0 -> 600 -- but the opponent's blunder rate did not follow (0.149 -> 0.153,
    inside one standard error) and the blunders got *cheaper* (452cp -> 311cp).

    The reason is that beta counts what share of *legal moves* are blunders,
    while Maia concentrates its probability on good ones. Raising the share does
    not raise the mass, and counting blunders by number rather than severity
    trades away the positions whose one available blunder is fatal. Use
    ``blunder_mass_weight`` instead."""

    blunder_mass_weight: float = 0.0
    """Delta on Maia-weighted blunder mass: the probability *this* opponent puts
    on a blundering reply, rather than the share of moves that are blunders.

    Targets the right quantity and still does not clearly work. Over 30 games
    delta=1200 looked strong, lifting the opponent's blunder rate 0.149 -> 0.192;
    at 90 games the same comparison shrank to 0.159 -> 0.170, about +7% and
    1.5 standard errors on ~2,700 opponent moves. It also cheapens the errors
    the way beta does (453cp -> 390cp).

    Treat the effect as unproven rather than absent: separating +7% from zero
    needs roughly 600 games per arm, which nobody has run. Zero reproduces the
    pre-Milestone-10 ranking exactly, and zero is the honest default until that
    experiment exists."""

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
        if self.proposal_count < 1 or self.max_proposals < 0:
            raise ValueError("proposal_count must be >= 1 and max_proposals >= 0")
        if self.cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        if self.blunder_threshold < 0:
            raise ValueError("blunder_threshold must be non-negative")
        if self.beta_depth < 1:
            raise ValueError("beta_depth must be >= 1")


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

    beta: float = 0.0
    """Blunder potential of the position this move creates: the fraction of the
    opponent's legal replies that lose more than ``blunder_threshold``."""

    beta_exact: bool = True
    """False when the MultiPV scan could not pin beta and it is an upper bound."""

    blunder_mass: float = 0.0
    """Opponent-model probability mass sitting on blundering replies."""

    is_safe: bool = True
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
