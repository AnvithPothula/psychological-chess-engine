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

from src.engine.maia3_model import Maia3Evaluator
from src.engine.search import AdversarialSearcher, CandidateProposer
from src.engine.stockfish import StockfishEvaluator
from src.training.train_dpo import WARM_CHECKPOINT
from src.types import SearchConfig

__all__ = ["BotSpec", "BASELINE", "TRAP", "ARMS", "build_searcher"]

logger = logging.getLogger(__name__)

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


BASELINE: Final[BotSpec] = BotSpec(
    name="baseline",
    description="Stockfish MultiPV candidates, adversarial expectimax, no re-ranking.",
)

TRAP: Final[BotSpec] = BotSpec(
    name="trap",
    description=(
        "Same search, candidate pool widened by the distilled human prior and a "
        "risk floor that slides with expected utility."
    ),
    use_prior_candidates=True,
    search=GAMBIT_SEARCH,
)

ARMS: Final[dict[str, BotSpec]] = {BASELINE.name: BASELINE, TRAP.name: TRAP}


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
    searcher = AdversarialSearcher(stockfish, opponent, proposer=proposer)
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
