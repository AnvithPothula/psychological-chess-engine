"""Named bot configurations, so an A/B test differs in one thing and not four.

An arena comparing two engines is only worth running if the arms are identical
apart from the variable under test. Building searchers ad hoc at each call site
is how a depth or a safety threshold quietly drifts between them, and then the
result measures the drift.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterator, Optional

from src.engine.book import OpeningBook
from src.engine.maia3_model import Maia3Evaluator
from src.engine.search import AdversarialSearcher, CandidateProposer
from src.engine.stockfish import StockfishEvaluator
from src.training.train_dpo import WARM_CHECKPOINT
from src.types import SearchConfig

__all__ = [
    "BotSpec", "BASELINE", "TRAP", "STANDARD", "SKEW", "ARMS", "build_searcher",
]

logger = logging.getLogger(__name__)

SKEW_BOOK: Final[Path] = Path("src/engine/books/skew.bin")
"""Mined by src.training.skew_miner and packed by src.training.polyglot_compiler."""

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
        "Plays the mined engine-equal skew book where it has an entry, the standard "
        "book everywhere else, then the same expectimax."
    ),
    book=True,
    trap_path=SKEW_BOOK,
)
"""The skew book sits in the front slot, which is probed first and falls through
to the standard book on a miss. Milestone 16 put it in the standard slot
instead, which *replaced* 578,126 entries with 708: the arm had no book at plies
0-2 and left book at once, so the run compared a book against almost none and
never tested which openings to steer into. The control's front slot holds the
185-entry trap book, so the arms now differ only in that small front book."""

TRAP: Final[BotSpec] = BotSpec(
    name="trap",
    description=(
        "Same search, candidate pool widened by the distilled human prior and a "
        "risk floor that slides with expected utility."
    ),
    use_prior_candidates=True,
    search=GAMBIT_SEARCH,
)

ARMS: Final[dict[str, BotSpec]] = {
    arm.name: arm for arm in (BASELINE, TRAP, STANDARD, SKEW)
}


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

    book: Optional[OpeningBook] = None
    if spec.book:
        for path in (spec.trap_path, spec.standard_path):
            if path is not None and not path.exists():
                raise FileNotFoundError(
                    f"{path} does not exist; run src.training.skew_miner "
                    "and src.training.polyglot_compiler first"
                )
        book = OpeningBook(
            trap_path=spec.trap_path,
            standard_path=spec.standard_path,
            opponent_rating=opponent_rating,
        )

    searcher = AdversarialSearcher(stockfish, opponent, config=spec.search, book=book)
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
