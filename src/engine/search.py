"""Opponent-bounded adversarial expectimax search.

The bot does not play the objectively best move. It plays the move with the
highest *expected* value against a calibrated model of how this particular
opponent actually moves -- subject to a hard safety floor, so a trap that a
competent opponent refutes is never played.

One search does:

1. **Root scan.** One Stockfish MultiPV search scores every legal move. Moves
   within ``root_margin`` of the best are candidates, capped at ``max_candidates``.
   Stockfish's own best move is always first in that list, so it is always
   available as a fallback.
2. **Opponent model.** For each candidate, Maia predicts the human reply
   distribution, which is pruned (``min_reply_probability``, ``target_reply_mass``,
   ``max_replies``) and renormalised.
3. **Expectimax.** Each retained reply is scored by Stockfish; utility is the
   probability-weighted mean.
4. **Safety filter.** A candidate is discarded when either its worst retained
   human reply *or* its objective value against best defence drops the bot below
   ``-safety_threshold``, regardless of how attractive its utility is.

The objective half of that filter is not in the original formulation and is not
redundant. Safety measured only over *retained* human replies is blind by
construction: reply truncation drops the tail of the distribution, and the move
that refutes a trap is precisely the move a weak opponent is unlikely to find.
A 95%-probability reply alone reaches ``target_mass``, so the 5% refutation is
never scored and the trap is scored as safe. One extra Stockfish call per
candidate -- the minimax value of the position after the candidate move -- closes
that hole and makes the safety guarantee independent of the opponent model.

All scores inside this module are **bot relative**: positive is good for the
side that is searching. Stockfish's White-relative output is flipped exactly
once, in :meth:`AdversarialSearcher._to_bot`.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import chess

from src import config as engine_config
from src.engine import EvaluatorError
from src.engine.cache import EvalCache, position_key
from src.types import (
    CandidateStats,
    EngineEval,
    MoveDistribution,
    PredictedReply,
    SearchConfig,
    SearchResult,
)

__all__ = ["AdversarialSearcher", "HumanModel", "PositionEvaluator", "TerminalPositionError", "truncate_distribution"]

logger = logging.getLogger(__name__)

_TOP_REPLIES_REPORTED = 3


class TerminalPositionError(EvaluatorError):
    """Raised when asked to search a position that is already over."""


class PositionEvaluator(Protocol):
    """Objective evaluation source. Implemented by ``StockfishEvaluator``."""

    def evaluate(self, board: chess.Board, depth: int = ...) -> EngineEval: ...

    def analyse_root_moves(
        self, board: chess.Board, *, depth: int = ..., multipv: Optional[int] = ...
    ) -> Dict[chess.Move, EngineEval]: ...


class HumanModel(Protocol):
    """Opponent move-likelihood source. Implemented by ``MaiaEvaluator``."""

    def predict_move_probabilities(
        self, board: chess.Board, *, temperature: Optional[float] = ...
    ) -> MoveDistribution: ...


def truncate_distribution(
    distribution: MoveDistribution,
    *,
    min_probability: float,
    target_mass: float,
    max_replies: int,
) -> MoveDistribution:
    """Keep the head of a reply distribution and renormalise it to 1.0.

    Replies are taken in descending probability until any of three limits binds:
    a reply falls below ``min_probability``, the retained mass reaches
    ``target_mass`` (the crossing reply is kept), or ``max_replies`` is reached.
    The single most likely reply is always retained, even if it is itself below
    ``min_probability`` -- which happens in positions with very flat policies.
    """
    ordered = sorted(distribution.probabilities.items(), key=lambda item: (-item[1], item[0].uci()))
    kept: List[Tuple[chess.Move, float]] = []
    cumulative = 0.0

    for move, probability in ordered:
        if kept and probability < min_probability:
            break
        kept.append((move, probability))
        cumulative += probability
        if cumulative >= target_mass or len(kept) >= max_replies:
            break

    total = math.fsum(probability for _, probability in kept)
    return MoveDistribution({move: probability / total for move, probability in kept})


class AdversarialSearcher:
    """Selects moves by maximising expected utility against a human opponent model.

    Evaluators are injected; the searcher owns neither process lifecycle. Callers
    are responsible for closing them (both support the context-manager protocol)::

        with StockfishEvaluator() as sf, MaiaEvaluator(1500) as maia:
            searcher = AdversarialSearcher(sf, maia)
            result = searcher.search(board)
    """

    def __init__(
        self,
        evaluator: PositionEvaluator,
        human_model: HumanModel,
        *,
        config: Optional[SearchConfig] = None,
        cache: Optional[EvalCache] = None,
    ) -> None:
        self.evaluator = evaluator
        self.human_model = human_model
        self.config = config if config is not None else SearchConfig()
        self.cache = cache if cache is not None else EvalCache(self.config.cache_size)

    def search(self, board: chess.Board, config: Optional[SearchConfig] = None) -> SearchResult:
        """Choose a move for the side to move in ``board``.

        ``board`` is not mutated. Raises :class:`TerminalPositionError` if the
        position is already decided.
        """
        started = time.perf_counter()
        settings = config if config is not None else self.config
        work = board.copy()
        bot_color = work.turn

        if not any(work.legal_moves):
            raise TerminalPositionError(f"No legal moves in {work.fen()}")
        if self._terminal_bot_score(work, bot_color) is not None:
            raise TerminalPositionError(f"Position is already drawn or decided: {work.fen()}")

        immediate_mate = self._find_mate_in_one(work)
        if immediate_mate is not None:
            logger.info("search: %s is mate in 1, skipping expectimax", immediate_mate.uci())
            return SearchResult(
                move=immediate_mate,
                expected_utility=float(engine_config.MATE_SCORE_CP),
                is_trap=False,
                fallback_triggered=False,
                candidates=(),
                nodes_evaluated=0,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        root_scores, best_move = self._scan_root(work, bot_color, settings)
        candidates = self._select_candidates(root_scores, best_move, settings)
        logger.debug(
            "root: best=%s (%+dcp) candidates=%s",
            best_move.uci(),
            root_scores[best_move],
            [f"{move.uci()}:{root_scores[move]:+d}" for move in candidates],
        )

        nodes = 0
        stats: List[CandidateStats] = []
        for move in candidates:
            candidate, evaluated = self._score_candidate(work, move, bot_color, settings)
            nodes += evaluated
            stats.append(candidate)
            logger.debug(
                "candidate %s: utility=%+.1f worst=%+d trap_delta=%+.1f safe=%s replies=%s",
                move.uci(),
                candidate.expected_utility,
                candidate.worst_case,
                candidate.blunder_trap_delta,
                candidate.is_safe,
                [f"{r.move.uci()}@{r.probability:.0%}->{r.evaluation:+d}" for r in candidate.top_replies],
            )

        stats.sort(key=lambda candidate: (-candidate.expected_utility, candidate.move.uci()))
        safe = [candidate for candidate in stats if candidate.is_safe]

        if safe:
            chosen = safe[0]
            fallback_triggered = False
            utility = chosen.expected_utility
            selected = chosen.move
        else:
            fallback_triggered = True
            selected = best_move
            utility = next(
                (c.expected_utility for c in stats if c.move == best_move),
                float(root_scores[best_move]),
            )
            logger.info(
                "search: every candidate breached the -%dcp safety floor, "
                "falling back to Stockfish's best move %s",
                settings.safety_threshold,
                best_move.uci(),
            )

        result = SearchResult(
            move=selected,
            expected_utility=utility,
            is_trap=selected != best_move,
            fallback_triggered=fallback_triggered,
            candidates=tuple(stats),
            nodes_evaluated=nodes,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        cache_stats = self.cache.stats()
        logger.info("search: %s", result.summary())
        logger.debug(
            "cache: %d hits / %d misses (%.0f%%), %d entries, %d evictions",
            cache_stats.hits,
            cache_stats.misses,
            cache_stats.hit_rate * 100.0,
            cache_stats.size,
            cache_stats.evictions,
        )
        return result

    # -- root ---------------------------------------------------------------

    def _scan_root(
        self, board: chess.Board, bot_color: chess.Color, settings: SearchConfig
    ) -> Tuple[Dict[chess.Move, int], chess.Move]:
        """One MultiPV search: bot-relative scores for the plausible moves, plus the best.

        Only ``max_candidates`` PV lines are requested. Scoring every legal move
        is pure waste -- ``_select_candidates`` discards everything past that cap
        anyway -- and MultiPV cost scales hard with line count: 40 lines at depth
        10 costs ~1s against ~0.1s for 6, on identical output for our purposes.
        """
        white_scores = self.evaluator.analyse_root_moves(
            board, depth=settings.root_depth, multipv=settings.max_candidates
        )
        scores = {
            move: self._to_bot(evaluation.centipawns, bot_color)
            for move, evaluation in white_scores.items()
        }
        best_move = min(scores, key=lambda move: (-scores[move], move.uci()))
        return scores, best_move

    @staticmethod
    def _select_candidates(
        scores: Dict[chess.Move, int], best_move: chess.Move, settings: SearchConfig
    ) -> List[chess.Move]:
        """Moves within ``root_margin`` of the best, best first, width-capped."""
        floor = scores[best_move] - settings.root_margin
        ranked = sorted(scores, key=lambda move: (-scores[move], move.uci()))
        return [move for move in ranked if scores[move] >= floor][: settings.max_candidates]

    # -- candidate expansion ------------------------------------------------

    def _score_candidate(
        self,
        board: chess.Board,
        move: chess.Move,
        bot_color: chess.Color,
        settings: SearchConfig,
    ) -> Tuple[CandidateStats, int]:
        """Expectimax over the opponent's likely replies to ``move``.

        Returns the candidate's telemetry and the number of leaves evaluated.
        """
        board.push(move)
        try:
            terminal = self._terminal_bot_score(board, bot_color)
            if terminal is not None:
                # The opponent has no move, or the game is drawn here: the
                # position's value is certain and there is nothing to model.
                return (
                    self._build_stats(move, float(terminal), terminal, terminal, (), settings),
                    0,
                )

            # Objective floor: what this move is worth against best defence, not
            # against likely defence. Independent of the opponent model.
            objective_score = self._evaluate_bot(board, bot_color, settings.leaf_depth)

            replies = truncate_distribution(
                self.human_model.predict_move_probabilities(board),
                min_probability=settings.min_reply_probability,
                target_mass=settings.target_reply_mass,
                max_replies=settings.max_replies,
            )

            utility = 0.0
            worst = engine_config.MATE_SCORE_CP
            predicted: List[PredictedReply] = []
            for reply, probability in replies.probabilities.items():
                board.push(reply)
                try:
                    score = self._evaluate_bot(board, bot_color, settings.leaf_depth)
                finally:
                    board.pop()
                utility += probability * score
                worst = min(worst, score)
                predicted.append(PredictedReply(reply, probability, score))
        finally:
            board.pop()

        predicted.sort(key=lambda reply: (-reply.probability, reply.move.uci()))
        return (
            self._build_stats(
                move, utility, worst, objective_score, tuple(predicted[:_TOP_REPLIES_REPORTED]), settings
            ),
            len(predicted) + 1,  # + the objective-floor evaluation
        )

    @staticmethod
    def _build_stats(
        move: chess.Move,
        utility: float,
        worst: int,
        objective_score: int,
        replies: Sequence[PredictedReply],
        settings: SearchConfig,
    ) -> CandidateStats:
        return CandidateStats(
            move=move,
            expected_utility=utility,
            worst_case=worst,
            objective_score=objective_score,
            blunder_trap_delta=utility - objective_score,
            is_safe=min(worst, objective_score) >= -settings.safety_threshold,
            top_replies=replies,
        )

    # -- evaluation ---------------------------------------------------------

    def _evaluate_bot(self, board: chess.Board, bot_color: chess.Color, depth: int) -> int:
        """Bot-relative score of ``board``, via the cache, short-circuiting terminals."""
        terminal = self._terminal_bot_score(board, bot_color)
        if terminal is not None:
            return terminal

        key = position_key(board, depth)
        cached = self.cache.get(key)
        if cached is None:
            cached = self.evaluator.evaluate(board, depth)
            self.cache.put(key, cached)
        return self._to_bot(cached.centipawns, bot_color)

    @staticmethod
    def _to_bot(white_centipawns: int, bot_color: chess.Color) -> int:
        """The one place White-relative scores become bot-relative."""
        return white_centipawns if bot_color == chess.WHITE else -white_centipawns

    @staticmethod
    def _terminal_bot_score(board: chess.Board, bot_color: chess.Color) -> Optional[int]:
        """Bot-relative value of a finished position, or ``None`` if play continues.

        Claimable draws (threefold repetition, 50-move rule) count as draws: either
        side will claim one when it suits them, so modelling them as still-playable
        would overstate the bot's winning chances.
        """
        if board.is_checkmate():
            # The side to move is mated.
            return -engine_config.MATE_SCORE_CP if board.turn == bot_color else engine_config.MATE_SCORE_CP
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.halfmove_clock >= 100
            or board.is_repetition(3)
        ):
            return 0
        return None

    @staticmethod
    def _find_mate_in_one(board: chess.Board) -> Optional[chess.Move]:
        """A move that mates immediately, if one exists. Costs no engine calls."""
        for move in sorted(board.legal_moves, key=lambda candidate: candidate.uci()):
            board.push(move)
            mates = board.is_checkmate()
            board.pop()
            if mates:
                return move
        return None
