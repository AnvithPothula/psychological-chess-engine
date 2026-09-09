"""UCI engine wrappers: objective evaluation (Stockfish) and human policy (Maia)."""

from __future__ import annotations


class EvaluatorError(Exception):
    """Base class for every failure raised by this package."""


class EngineInitializationError(EvaluatorError):
    """The engine binary could not be launched, handshaken or configured."""


class EngineAnalysisError(EvaluatorError):
    """The engine started but produced unusable or no output for a position."""


# Re-exported below the exception definitions: the submodules import those names
# back out of this package, so they must exist before the submodules load.
from src.engine.cache import EvalCache  # noqa: E402
from src.engine.maia import MaiaEvaluator  # noqa: E402
from src.engine.stockfish import StockfishEvaluator  # noqa: E402
from src.engine.search import AdversarialSearcher, TerminalPositionError  # noqa: E402

__all__ = [
    "AdversarialSearcher",
    "EngineAnalysisError",
    "EngineInitializationError",
    "EvalCache",
    "EvaluatorError",
    "MaiaEvaluator",
    "StockfishEvaluator",
    "TerminalPositionError",
]
