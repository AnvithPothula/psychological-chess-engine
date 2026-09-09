"""Maia (Lc0) wrapper producing human move-likelihood distributions.

Maia is a Leela-architecture network trained to *predict the move a human of a
given rating actually plays*, not to find the best move. Its value therefore
lives entirely in the policy head, which is why this wrapper deliberately does
no tree search.

Why the policy priors and not MultiPV score lines: at ``go nodes 1`` Lc0 expands
the root and stops, so every child edge has ``N=0`` and inherits the *same* root
Q. MultiPV info lines at that budget carry one identical score per move --
softmaxing them yields a uniform distribution. The per-move priors emitted by
``VerboseMoveStats`` are the network's actual output and the only usable signal
at this budget. Raising the node count to differentiate the scores would replace
the human model with weak MCTS, which defeats the purpose.
"""

from __future__ import annotations

import atexit
import math
import re
from pathlib import Path
from types import TracebackType
from typing import Dict, Mapping, Optional, Type, Union

import chess
import chess.engine

from src import config
from src.engine import EngineAnalysisError, EngineInitializationError
from src.types import MoveDistribution

__all__ = ["MaiaEvaluator", "temperature_softmax"]

# Lc0 `VerboseMoveStats` line, e.g.
#   e2e4  (322 ) N: 0 (+ 0) (P: 21.05%) (WL: -.-----) ... (Q: 0.04558) ...
# python-chess strips the leading "info string ", so match from the move token.
_MOVE_STATS_PATTERN = re.compile(
    r"^(?P<move>[a-h][1-8][a-h][1-8][qrbnQRBN]?)\s.*?\(P:\s*(?P<prior>-?[\d.]+)%\)"
)


def temperature_softmax(priors: Mapping[chess.Move, float], temperature: float) -> Dict[chess.Move, float]:
    """Temperature-scaled softmax over move priors, returned normalised to 1.0.

    The priors are already probabilities, so the softmax is taken over their
    logits: ``softmax(log(p) / T)``, equivalently ``p^(1/T)`` renormalised.
    ``T == 1.0`` is therefore the identity and preserves Maia's calibration;
    ``T > 1`` flattens the opponent model, ``T < 1`` sharpens it. Applying a
    plain softmax to the probabilities themselves would destroy the calibration.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not priors:
        raise ValueError("temperature_softmax requires at least one move")

    logits = {
        move: (math.log(prior) / temperature if prior > 0.0 else -math.inf)
        for move, prior in priors.items()
    }
    peak = max(logits.values())
    if not math.isfinite(peak):
        raise ValueError("All move priors were zero")

    weights = {move: math.exp(logit - peak) for move, logit in logits.items()}
    total = math.fsum(weights.values())
    return {move: weight / total for move, weight in weights.items()}


class MaiaEvaluator:
    """Human move-likelihood model backed by an Lc0 process hosting Maia weights.

    The process is long-lived and the weights are loaded once at construction;
    swapping rating means constructing a new evaluator. Use as a context manager
    or call :meth:`close` so the ``lc0`` child process is never orphaned::

        with MaiaEvaluator(rating=1500) as maia:
            distribution = maia.predict_move_probabilities(board)
    """

    def __init__(
        self,
        rating: int = config.DEFAULT_MAIA_RATING,
        *,
        binary: Optional[Path] = None,
        weights: Optional[Path] = None,
        temperature: float = config.DEFAULT_MAIA_TEMPERATURE,
        backend: Optional[str] = None,
        timeout: float = config.ENGINE_STARTUP_TIMEOUT,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.rating = rating
        self.temperature = temperature
        self.weights_path = weights if weights is not None else config.maia_weights(rating)
        path = binary if binary is not None else config.lc0_binary()

        try:
            self._engine: Optional[chess.engine.SimpleEngine] = chess.engine.SimpleEngine.popen_uci(
                str(path), timeout=timeout
            )
        except (OSError, chess.engine.EngineError, TimeoutError) as exc:
            raise EngineInitializationError(f"Could not start Lc0 at {path}: {exc}") from exc

        atexit.register(self.close)

        options: Dict[str, Union[str, int, bool, None]] = {
            "WeightsFile": str(self.weights_path),
            "VerboseMoveStats": True,
            # Lc0 wants its float options as strings.
            "PolicyTemperature": str(config.LC0_POLICY_TEMPERATURE),
        }
        if backend is not None:
            options["Backend"] = backend

        try:
            self._engine.configure(options)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
            self.close()
            raise EngineInitializationError(
                f"Could not load Maia weights {self.weights_path} into Lc0: {exc}"
            ) from exc

    def predict_move_probabilities(
        self, board: chess.Board, *, temperature: Optional[float] = None
    ) -> MoveDistribution:
        """Probability that a human of this rating plays each legal move.

        Keys are restricted to the legal moves of ``board`` and the values sum
        to 1.0. ``temperature`` overrides the instance default for this call.
        """
        legal_moves = set(board.legal_moves)
        if not legal_moves:
            raise ValueError(f"No legal moves in position {board.fen()}")
        if len(legal_moves) == 1:
            return MoveDistribution({legal_moves.pop(): 1.0})

        priors = self._policy_priors(board, legal_moves)
        scaled = temperature_softmax(priors, self.temperature if temperature is None else temperature)
        return MoveDistribution(scaled)

    def _policy_priors(self, board: chess.Board, legal_moves: set[chess.Move]) -> Dict[chess.Move, float]:
        """Read raw policy priors (as fractions, unnormalised) out of Lc0."""
        engine = self._require_engine()
        priors: Dict[chess.Move, float] = {}

        try:
            with engine.analysis(board, chess.engine.Limit(nodes=config.MAIA_SEARCH_NODES)) as analysis:
                for info in analysis:
                    line = info.get("string")
                    if not isinstance(line, str):
                        continue
                    parsed = self._parse_move_stats(board, line, legal_moves)
                    if parsed is not None:
                        move, prior = parsed
                        priors[move] = prior
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError) as exc:
            raise EngineAnalysisError(f"Maia failed on {board.fen()}: {exc}") from exc

        if not priors:
            raise EngineAnalysisError(
                f"Lc0 emitted no usable policy priors for {board.fen()}. "
                "Check that the VerboseMoveStats option is supported by this Lc0 build."
            )
        return priors

    @staticmethod
    def _parse_move_stats(
        board: chess.Board, line: str, legal_moves: set[chess.Move]
    ) -> Optional[tuple[chess.Move, float]]:
        match = _MOVE_STATS_PATTERN.match(line.removeprefix("info string ").strip())
        if match is None:
            return None  # Root summary ("node ...") and other info strings.
        try:
            move = board.parse_uci(match.group("move"))
        except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            return None
        if move not in legal_moves:
            return None
        return move, float(match.group("prior")) / 100.0

    def close(self) -> None:
        """Terminate the Lc0 process. Idempotent and safe to call from atexit."""
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
            raise EngineAnalysisError("MaiaEvaluator has been closed")
        return self._engine

    def __enter__(self) -> MaiaEvaluator:
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
