"""Named bot configurations, so an A/B test differs in one thing and not four.

An arena comparing two engines is only worth running if the arms are identical
apart from the variable under test. Building searchers ad hoc at each call site
is how a depth or a safety threshold quietly drifts between them, and then the
result measures the drift.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Iterator, Mapping, Optional

from src.engine.book import OpeningBook
from src.engine.maia3_model import Maia3Evaluator
from src.engine.search import AdversarialSearcher, CandidateProposer
from src.engine.stockfish import StockfishEvaluator
from src.training.train_dpo import WARM_CHECKPOINT
from src.types import SearchConfig

__all__ = [
    "BotSpec", "BASELINE", "TRAP", "STANDARD", "SKEW", "PSYCH", "PSYCH_CAPPED", "ARMS",
    "HUMAN_PLAY", "for_humans", "build_book", "build_searcher",
]

logger = logging.getLogger(__name__)

REPERTOIRE_BOOK: Final[Path] = Path("src/engine/books/repertoire.bin")
"""Built by src.training.repertoire_builder from the miner's tree and finds."""

ARENA_SEARCH: Final[SearchConfig] = SearchConfig(
    root_depth=8, leaf_depth=8, max_candidates=5, max_replies=5
)
"""The control. Static safety floor, Stockfish candidates only."""

GAMBIT_SEARCH: Final[SearchConfig] = SearchConfig(
    root_depth=8, leaf_depth=8, max_candidates=5, max_replies=5,
    max_proposals=5, proposal_count=5,
    gambit_lambda=0.5, gambit_floor=400,
)
"""The treatment. Same depths, so the arms still differ in one idea: the
candidate pool is the union of Stockfish's top five and the prior's top five,
and the floor slides with expected utility down to a hard bottom of 400cp."""


@dataclass(frozen=True, slots=True)
class BotSpec:
    """One arm of an arena match."""

    name: str
    description: str
    use_prior_candidates: bool = False
    prior_path: Path = field(default_factory=lambda: WARM_CHECKPOINT)
    search: SearchConfig = ARENA_SEARCH

    book: bool = False
    """Open an opening book at all. The arena ran without one until Milestone 16,
    which is worth knowing when reading its earlier results: every game was
    played out of the search from move one."""

    standard_path: Optional[Path] = None
    """Overrides the standard book. ``None`` keeps the configured default."""

    trap_path: Optional[Path] = None


BASELINE: Final[BotSpec] = BotSpec(
    name="baseline",
    description="Stockfish MultiPV candidates, adversarial expectimax, no re-ranking.",
)

STANDARD: Final[BotSpec] = BotSpec(
    name="standard",
    description="Opens from the standard book, then adversarial expectimax.",
    book=True,
)

SKEW: Final[BotSpec] = BotSpec(
    name="skew",
    description=(
        "Steers toward mined engine-equal skew positions and plays the skew move on "
        "arrival, the standard book once the opponent leaves the repertoire, then "
        "the same expectimax."
    ),
    book=True,
    trap_path=REPERTOIRE_BOOK,
)
"""The repertoire sits in the front slot, which is probed first and falls through
to the standard book on a miss. It holds the skew moves as well as the path to
them, so skew.bin has no slot of its own: see src.training.repertoire_builder.

History, because both mistakes produced clean-looking nulls. Milestone 16 put
the skew book in the standard slot, which *replaced* 578,126 entries with 708
and left the arm bookless from ply 0. Moving it to the front slot fixed coverage
but not dosage: from the initial position it fired in 2 of 40 games, because
nothing steered toward plies 7-10 of the mined lines."""

TRAP: Final[BotSpec] = BotSpec(
    name="trap",
    description=(
        "Same search, candidate pool widened by the distilled human prior and a "
        "risk floor that slides with expected utility."
    ),
    use_prior_candidates=True,
    search=GAMBIT_SEARCH,
)

HUMAN_PLAY: Final[Mapping[str, Any]] = {
    "safety_threshold": 250,
    "gambit_lambda": 0.5,
    "gambit_floor": 450,
    "narrow_path_weight": 10.0,
    "winning_margin": 10_000,
}
"""How the bot plays humans: psychology first, at some cost in objective strength.

- A looser floor (-250cp, sliding to -450 when the expected payout is large):
  the odds bots perform at 2000-2700 against humans from positions a queen down,
  so objective soundness is not what beats humans.
- The quiet narrow-path term on: in August 2026 human games, quiet positions
  with two or fewer safe replies were blundered 22% of the time against 3%.
- Won positions must stay won (+300), but need not stay within 200cp of the best
  move, so traps are still set while winning. Against bots the full rule holds:
  engines do not fall for them, and that cost five won games.

Never weaker by losing on purpose: Lichess flags bots that throw games."""

HUMAN_DEPTH: Final[int] = 6
"""Search depth cap against humans, so the bot is not simply out-calculating
them. Measured against maia3@1900, 300 games per arm: the cap made the
opponent blunder more, 11.74% -> 13.48% of moves (+3.2 sigma against the
uncapped human profile), while the score dropped only 0.997 -> 0.990. The arena
cannot say how much weaker that is against people; Maia 1900 loses to either."""


def for_humans(config: SearchConfig) -> SearchConfig:
    """``config`` with the human profile applied and its depth capped."""
    return replace(
        config, **HUMAN_PLAY,
        root_depth=min(config.root_depth, HUMAN_DEPTH),
        leaf_depth=min(config.leaf_depth, HUMAN_DEPTH),
    )


PSYCH: Final[BotSpec] = BotSpec(
    name="psych",
    description="The human profile without the depth cap, for comparison.",
    search=replace(ARENA_SEARCH, **HUMAN_PLAY),
)

PSYCH_CAPPED: Final[BotSpec] = BotSpec(
    name="psych-capped",
    description="The human profile as played live: looser floor, narrow paths, depth 6.",
    search=for_humans(ARENA_SEARCH),
)

ARMS: Final[dict[str, BotSpec]] = {
    arm.name: arm for arm in (BASELINE, TRAP, STANDARD, SKEW, PSYCH, PSYCH_CAPPED)
}


def build_book(spec: BotSpec, *, opponent_rating: int) -> Optional[OpeningBook]:
    """The arm's books, or ``None`` for a bookless arm. Raises if one is missing.

    ``OpeningBook`` itself treats a missing file as empty, which is right for a
    bot in production and wrong for an experiment: the arm would quietly become
    its own control. Shared with the Lichess bridge for the same reason.
    """
    if not spec.book:
        return None
    for path in (spec.trap_path, spec.standard_path):
        if path is not None and not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist; run src.training.skew_miner --tree "
                "and src.training.repertoire_builder first"
            )
    return OpeningBook(
        trap_path=spec.trap_path,
        standard_path=spec.standard_path,
        opponent_rating=opponent_rating,
    )


def build_searcher(
    spec: BotSpec,
    stockfish: StockfishEvaluator,
    opponent: Maia3Evaluator,
    *,
    opponent_rating: int,
) -> AdversarialSearcher:
    """Assemble one arm's searcher. Raises if a required checkpoint is absent.

    A missing checkpoint is fatal rather than a silent downgrade to the
    baseline: an arena arm that quietly becomes its own control produces a null
    result that looks like evidence.
    """
    proposer: Optional[CandidateProposer] = None
    if spec.use_prior_candidates:
        from src.engine.policy_generator import NeuralCandidateGenerator

        proposer = NeuralCandidateGenerator(spec.prior_path)

    book = build_book(spec, opponent_rating=opponent_rating)
    searcher = AdversarialSearcher(
        stockfish, opponent, config=spec.search, book=book, proposer=proposer
    )
    searcher.opponent_rating = opponent_rating
    return searcher


def engines(rating: int) -> Iterator[tuple[StockfishEvaluator, Maia3Evaluator]]:
    """One Stockfish and one Maia-3 for the caller's lifetime.

    A generator rather than a pair of context managers so a worker process can
    hold both for the duration of its games and drop them together. Stockfish is
    a subprocess and is not reentrant; one per process is the contract.
    """
    with ExitStack() as stack:
        stockfish = stack.enter_context(StockfishEvaluator())
        opponent = stack.enter_context(Maia3Evaluator(rating))
        yield stockfish, opponent
