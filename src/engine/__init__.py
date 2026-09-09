"""UCI engine wrappers: objective evaluation (Stockfish) and human policy (Maia)."""

from __future__ import annotations


class EvaluatorError(Exception):
    """Base class for every failure raised by this package."""


class EngineInitializationError(EvaluatorError):
    """The engine binary could not be launched, handshaken or configured."""


class EngineAnalysisError(EvaluatorError):
    """The engine started but produced unusable or no output for a position."""


from src.engine.maia import MaiaEvaluator  # noqa: E402  (re-export; avoids an import cycle)
from src.engine.stockfish import StockfishEvaluator  # noqa: E402

__all__ = [
    "EngineAnalysisError",
    "EngineInitializationError",
    "EvaluatorError",
    "MaiaEvaluator",
    "StockfishEvaluator",
]
