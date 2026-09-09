"""Stockfish wrapper producing objective, White-relative evaluations."""

from __future__ import annotations

import atexit
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

import chess
import chess.engine

from src import config
from src.engine import EngineAnalysisError, EngineInitializationError
from src.types import EngineEval

__all__ = ["StockfishEvaluator"]


class StockfishEvaluator:
    """Fixed-depth ground-truth evaluation from a persistent Stockfish process.

    Sign convention: every :class:`~src.types.EngineEval` returned is **White
    relative** -- positive favours White whether it is White or Black to move.
    UCI engines report scores relative to the side to move, so the flip happens
    here (via ``PovScore.white()``) and nowhere else in the codebase.

    The process is long-lived; construct once and reuse across a search. Use as
    a context manager, or call :meth:`close` explicitly, so the child process is
    never orphaned::

        with StockfishEvaluator() as sf:
            evaluation = sf.evaluate(board, depth=16)
    """

    def __init__(
        self,
        binary: Optional[Path] = None,
        *,
        threads: int = config.STOCKFISH_THREADS,
        hash_mb: int = config.STOCKFISH_HASH_MB,
        timeout: float = config.ENGINE_STARTUP_TIMEOUT,
    ) -> None:
        path = binary if binary is not None else config.stockfish_binary()
        try:
            self._engine: Optional[chess.engine.SimpleEngine] = chess.engine.SimpleEngine.popen_uci(
                str(path), timeout=timeout
            )
        except (OSError, chess.engine.EngineError, TimeoutError) as exc:
            raise EngineInitializationError(f"Could not start Stockfish at {path}: {exc}") from exc

        # Registered before configure() so a failure mid-setup still gets reaped.
        atexit.register(self.close)

        try:
            self._engine.configure({"Threads": threads, "Hash": hash_mb})
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
            self.close()
            raise EngineInitializationError(f"Could not configure Stockfish: {exc}") from exc

    def evaluate(self, board: chess.Board, depth: int = config.DEFAULT_STOCKFISH_DEPTH) -> EngineEval:
        """Search ``board`` to ``depth`` and return the White-relative evaluation."""
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        engine = self._require_engine()

        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError) as exc:
            raise EngineAnalysisError(f"Stockfish failed on {board.fen()}: {exc}") from exc

        score = info.get("score")
        if score is None:
            raise EngineAnalysisError(f"Stockfish returned no score for {board.fen()}")
        return EngineEval.from_pov_score(score)

    def close(self) -> None:
        """Terminate the engine process. Idempotent and safe to call from atexit."""
        engine, self._engine = self._engine, None
        if engine is None:
            return
        atexit.unregister(self.close)
        try:
            engine.quit()
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, OSError, TimeoutError):
            pass  # Fall through to close(), which kills the transport outright.
        finally:
            engine.close()

    def _require_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            raise EngineAnalysisError("StockfishEvaluator has been closed")
        return self._engine

    def __enter__(self) -> StockfishEvaluator:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        try:
            self.close()
        except Exception:
            pass
