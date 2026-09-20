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
"""Shared by every arm. Both bots search identically; only the candidate source
differs, which is the whole point of the comparison."""


@dataclass(frozen=True, slots=True)
class BotSpec:
    """One arm of an arena match."""

    name: str
    description: str
    use_scorer: bool = False
    scorer_path: Path = Path("models/trap_scorer.pth")
    prior_path: Path = field(default_factory=lambda: WARM_CHECKPOINT)
    search: SearchConfig = ARENA_SEARCH


BASELINE: Final[BotSpec] = BotSpec(
    name="baseline",
    description="Stockfish MultiPV candidates, adversarial expectimax, no re-ranking.",
)

TRAP: Final[BotSpec] = BotSpec(
    name="trap",
    description="Same search, candidates re-ranked by the engine-annotated scorer.",
    use_scorer=True,
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

    A missing scorer is fatal rather than a silent downgrade to the baseline:
    an arena arm that quietly becomes its own control produces a null result
    that looks like evidence.
    """
    proposer: Optional[CandidateProposer] = None
    if spec.use_scorer:
        from src.engine.scorer_proposer import ScorerCandidateProposer

        proposer = ScorerCandidateProposer(
            stockfish, prior_path=spec.prior_path, scorer_path=spec.scorer_path
        )
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
