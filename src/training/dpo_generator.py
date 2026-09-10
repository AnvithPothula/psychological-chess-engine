"""Self-play rollouts that mine trap preferences from the searcher.

The bot plays Maia. Every time it prefers an objectively worse move because the
opponent model expects the human to fall in, that disagreement is a preference
pair: the trap is ``chosen``, Stockfish's move is ``rejected``.

**Every pair records the opponent it was mined against.** The preference is not
universal -- it is the whole point that it is not. "Play the Stafford" is
correct against a 1200 and losing against a 2000, so a pair stripped of its
opponent context teaches a policy to sacrifice material unconditionally. The
records therefore carry the rating band and the measured bait probability, so a
downstream objective can condition on them rather than average over them.

Generation is fully headless. A viewer, if attached, receives
:class:`RolloutUpdate` snapshots through a callback and is never awaited.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Final, List, Optional, Protocol, Sequence

import chess

from src.engine import EvaluatorError
from src.engine.cognitive_elo import CognitiveTracker
from src.engine.search import AdversarialSearcher, TerminalPositionError
from src.types import CandidateStats, MoveDistribution, SearchConfig, SearchResult

__all__ = [
    "AdaptiveOpponent",
    "DPOGenerator",
    "GeneratorConfig",
    "RolloutUpdate",
    "TrapPair",
    "main",
]


class AdaptiveOpponent(Protocol):
    """What the rollout needs of an opponent: a policy that can change band.

    ``MaiaEvaluator`` satisfies it. Declared structurally so a rollout can be
    driven by a scripted opponent without starting Lc0.
    """

    rating: int

    def set_rating(self, rating: int) -> None: ...

    def predict_move_probabilities(
        self, board: chess.Board, *, temperature: Optional[float] = ...
    ) -> MoveDistribution: ...

logger = logging.getLogger(__name__)

TRAP_GAIN_CP: Final[int] = 200
"""Centipawns the human's likely reply must cost them for the bait to count."""

BLUNDER_HAZARD_CP: Final[int] = 150
"""Threshold for the hazard meter: reply mass that loses at least this much."""

MIN_BAIT_PROBABILITY: Final[float] = 0.05
"""A bait nobody takes is not a trap, however large the payoff would be."""

CONTESTED_CP: Final[int] = 300
"""Only mine where the game is still live.

In a +900 position every move wins and the 300-500cp "gains" between replies are
the difference between two won games, not a trap working. Pairs harvested there
teach a policy that winning positions prefer quiet moves to queen grabs, which
is both false and useless. The signal only exists where the trap changes the
result."""

PROGRESS_INTERVAL_SECONDS: Final[float] = 30.0
MAX_GAME_PLIES: Final[int] = 160
"""Self-play games are for mining openings and middlegames; a 300-move endgame
shuffle produces no traps and burns the rollout budget."""


@dataclass(frozen=True, slots=True)
class TrapPair:
    """One preference record, with the context that makes it conditional."""

    fen: str
    chosen: str
    """SAN of the trap the searcher preferred."""

    rejected: str
    """SAN of Stockfish's objectively best move."""

    opponent_rating: int
    """Rating band this preference was mined against. Without it the pair is
    an instruction to sacrifice material unconditionally."""

    trap_gain_cp: int
    """Centipawns won if the bait is taken, over best defence."""

    bait_probability: float
    """Maia's probability that the opponent walks in."""

    objective_cost_cp: int
    """What the trap concedes against best defence. The price of the bet."""

    expected_utility_cp: float
    game_index: int
    ply: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RolloutUpdate:
    """Snapshot handed to an attached viewer. Pure data, no pygame."""

    game_index: int
    ply: int
    fen: str
    last_move: Optional[chess.Move]
    evaluation_cp: int
    starting_elo: int
    current_elo: int
    blunder_hazard: float
    pairs_total: int
    pairs_per_minute: float
    games_completed: int
    status: str


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    games: int = 20
    bot_plays_white: Optional[bool] = None
    """``None`` alternates colours, which keeps the mined pairs balanced."""

    search_config: SearchConfig = field(default_factory=lambda: SearchConfig(
        root_depth=8, leaf_depth=8, max_candidates=5, max_replies=5
    ))
    """Shallower than match play on purpose: rollouts are a throughput problem,
    and a trap that only exists at depth 12 is not one a human will fall for."""

    opponent_rating: int = 1500
    trap_gain_cp: int = TRAP_GAIN_CP
    min_bait_probability: float = MIN_BAIT_PROBABILITY
    contested_cp: int = CONTESTED_CP
    max_plies: int = MAX_GAME_PLIES
    adapt_opponent: bool = True
    """Let the cognitive tracker re-band Maia mid-game, as in a real game."""


@dataclass
class GeneratorStats:
    games: int = 0
    plies: int = 0
    pairs: int = 0
    searches: int = 0
    elapsed_seconds: float = 0.0

    @property
    def pairs_per_minute(self) -> float:
        return self.pairs * 60.0 / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


Observer = Callable[[RolloutUpdate], None]


class DPOGenerator:
    """Plays the searcher against Maia and writes the disagreements to JSONL."""

    def __init__(
        self,
        searcher: AdversarialSearcher,
        opponent: AdaptiveOpponent,
        *,
        config: Optional[GeneratorConfig] = None,
        observer: Optional[Observer] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.searcher = searcher
        self.opponent = opponent
        self.config = config if config is not None else GeneratorConfig()
        self.observer = observer
        self.rng = rng if rng is not None else random.Random()
        self.stats = GeneratorStats()
        self.tracker = CognitiveTracker(float(self.config.opponent_rating))
        self._stop = False
        self._started = 0.0
        self._last_progress = 0.0

    def stop(self) -> None:
        """Ask the rollout to finish after the current move."""
        self._stop = True

    # -- rollout ------------------------------------------------------------

    def run(self, output_path: Path) -> GeneratorStats:
        """Play the configured games, appending every pair to ``output_path``."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._started = time.perf_counter()
        self._last_progress = self._started

        with output_path.open("a", encoding="utf-8") as sink:
            for index in range(self.config.games):
                if self._stop:
                    break
                for pair in self.play_game(index):
                    sink.write(pair.to_json() + "\n")
                    self.stats.pairs += 1
                sink.flush()
                self.stats.games += 1
                self._log_progress(force=True)

        self.stats.elapsed_seconds = time.perf_counter() - self._started
        logger.info(
            "rollout: %d games, %d plies, %d pairs in %.1fs (%.1f pairs/min)",
            self.stats.games, self.stats.plies, self.stats.pairs,
            self.stats.elapsed_seconds, self.stats.pairs_per_minute,
        )
        return self.stats

    def play_game(self, index: int) -> List[TrapPair]:
        """One self-play game. Returns the pairs it produced."""
        board = chess.Board()
        bot_is_white = (
            index % 2 == 0 if self.config.bot_plays_white is None else self.config.bot_plays_white
        )
        bot_colour = chess.WHITE if bot_is_white else chess.BLACK
        self.tracker.reset(float(self.config.opponent_rating))
        self.opponent.set_rating(self.tracker.suggested_band)

        pairs: List[TrapPair] = []
        evaluation = 0
        hazard = 0.0

        while not board.is_game_over(claim_draw=True) and board.ply() < self.config.max_plies:
            if self._stop:
                break
            move: Optional[chess.Move]
            if board.turn == bot_colour:
                result = self._search(board)
                if result is None:
                    break
                self.stats.searches += 1
                chosen = next((c for c in result.candidates if c.move == result.move), None)
                hazard = self._blunder_hazard(chosen)
                pair = self._mine_pair(board, result, index)
                if pair is not None:
                    pairs.append(pair)
                    logger.debug("pair: %s over %s (+%dcp bait)", pair.chosen, pair.rejected, pair.trap_gain_cp)
                move = result.move
                evaluation = self._bot_relative(chosen, evaluation)
            else:
                move = self._opponent_move(board, bot_colour)
            if move is None:
                break

            board.push(move)
            self.stats.plies += 1
            self._publish(index, board, move, evaluation, hazard, "playing")
            self._log_progress()

        self._publish(index, board, board.peek() if board.move_stack else None,
                      evaluation, hazard, "game over")
        return pairs

    # -- one move -----------------------------------------------------------

    def _search(self, board: chess.Board) -> Optional[SearchResult]:
        try:
            return self.searcher.search(board, self.config.search_config)
        except TerminalPositionError:
            return None
        except EvaluatorError as exc:
            logger.error("rollout: search failed (%s), abandoning the game", exc)
            self._stop = True
            return None

    def _opponent_move(self, board: chess.Board, bot_colour: chess.Color) -> Optional[chess.Move]:
        """Sample Maia, then score the move so the tracker can re-band."""
        try:
            distribution = self.opponent.predict_move_probabilities(board)
        except (EvaluatorError, ValueError) as exc:
            logger.error("rollout: opponent model failed (%s)", exc)
            self._stop = True
            return None

        moves = list(distribution.probabilities)
        weights = [distribution[move] for move in moves]
        move = self.rng.choices(moves, weights=weights, k=1)[0]

        loss = self._centipawn_loss(board, move, bot_colour)
        if loss is not None:
            self.tracker.update(board, move, loss, distribution)
            if self.config.adapt_opponent:
                band = self.tracker.suggested_band
                if band != self.opponent.rating:
                    self.opponent.set_rating(band)
        return move

    def _centipawn_loss(
        self, board: chess.Board, move: chess.Move, bot_colour: chess.Color
    ) -> Optional[float]:
        """What ``move`` cost the opponent, from their own perspective."""
        settings = self.config.search_config
        try:
            before = self.searcher.evaluator.evaluate(board, settings.leaf_depth)
            board.push(move)
            try:
                after = self.searcher.evaluator.evaluate(board, settings.leaf_depth)
            finally:
                board.pop()
        except EvaluatorError as exc:
            logger.debug("rollout: could not score the opponent move (%s)", exc)
            return None

        sign = -1 if bot_colour == chess.WHITE else 1  # opponent's perspective
        return float(sign * (after.centipawns - before.centipawns))

    def _mine_pair(
        self, board: chess.Board, result: SearchResult, index: int
    ) -> Optional[TrapPair]:
        """A pair exists only where the searcher overrode Stockfish and it paid."""
        if not result.is_trap or not result.candidates:
            return None
        chosen = next((c for c in result.candidates if c.move == result.move), None)
        best = self._objective_best(result.candidates, exclude=result.move)
        if chosen is None or best is None:
            return None
        if abs(best.objective_score) > self.config.contested_cp:
            # The game is already decided; nothing here is a trap.
            return None

        bait = self._best_bait(chosen)
        if bait is None:
            return None
        gain, probability = bait
        if gain < self.config.trap_gain_cp or probability < self.config.min_bait_probability:
            return None

        return TrapPair(
            fen=board.fen(),
            chosen=board.san(chosen.move),
            rejected=board.san(best.move),
            opponent_rating=self.opponent.rating,
            trap_gain_cp=int(gain),
            bait_probability=round(probability, 4),
            objective_cost_cp=chosen.objective_score,
            expected_utility_cp=round(chosen.expected_utility, 1),
            game_index=index,
            ply=board.ply(),
        )

    @staticmethod
    def _objective_best(
        candidates: Sequence[CandidateStats], *, exclude: chess.Move
    ) -> Optional[CandidateStats]:
        """The candidate Stockfish rates highest -- the move we chose not to play."""
        others = [candidate for candidate in candidates if candidate.move != exclude]
        return max(others, key=lambda c: c.objective_score) if others else None

    @staticmethod
    def _best_bait(candidate: CandidateStats) -> Optional[tuple[float, float]]:
        """Largest ``(gain, probability)`` among the predicted human replies."""
        best: Optional[tuple[float, float]] = None
        for reply in candidate.top_replies:
            gain = float(reply.evaluation - candidate.objective_score)
            if best is None or gain > best[0]:
                best = (gain, reply.probability)
        return best

    @staticmethod
    def _blunder_hazard(candidate: Optional[CandidateStats]) -> float:
        """Reply mass that loses at least ``BLUNDER_HAZARD_CP`` for the human."""
        if candidate is None:
            return 0.0
        return sum(
            reply.probability
            for reply in candidate.top_replies
            if reply.evaluation - candidate.objective_score >= BLUNDER_HAZARD_CP
        )

    @staticmethod
    def _bot_relative(candidate: Optional[CandidateStats], fallback: int) -> int:
        return candidate.objective_score if candidate is not None else fallback

    # -- observation --------------------------------------------------------

    def _publish(
        self,
        index: int,
        board: chess.Board,
        move: Optional[chess.Move],
        evaluation: int,
        hazard: float,
        status: str,
    ) -> None:
        if self.observer is None:
            return
        elapsed = time.perf_counter() - self._started
        self.observer(
            RolloutUpdate(
                game_index=index,
                ply=board.ply(),
                fen=board.fen(),
                last_move=move,
                evaluation_cp=evaluation,
                starting_elo=int(self.tracker.starting_elo),
                current_elo=self.tracker.current_elo,
                blunder_hazard=hazard,
                pairs_total=self.stats.pairs,
                pairs_per_minute=self.stats.pairs * 60.0 / elapsed if elapsed > 0 else 0.0,
                games_completed=self.stats.games,
                status=status,
            )
        )

    def _log_progress(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_progress < PROGRESS_INTERVAL_SECONDS:
            return
        self._last_progress = now
        elapsed = now - self._started
        rate = self.stats.pairs * 60.0 / elapsed if elapsed > 0 else 0.0
        logger.info(
            "rollout: game %d, %d plies, %d pairs, %.1f pairs/min",
            self.stats.games, self.stats.plies, self.stats.pairs, rate,
        )


def main() -> int:
    """Entry point: ``python -m src.training.dpo_generator``."""
    import argparse

    from src.engine import MaiaEvaluator, StockfishEvaluator
    from src.engine.book import OpeningBook

    parser = argparse.ArgumentParser(prog="python -m src.training.dpo_generator")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--rating", type=int, default=1500, help="Opponent's starting rating.")
    parser.add_argument("--output", type=Path, default=Path("build/dpo_pairs.jsonl"))
    parser.add_argument("--viewer", action="store_true", help="Watch the rollout live.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("chess.engine").setLevel(logging.WARNING)

    from src import config as engine_config

    band = engine_config.nearest_maia_rating(args.rating)
    with (
        StockfishEvaluator() as stockfish,
        MaiaEvaluator(band) as maia,
        OpeningBook(opponent_rating=args.rating) as book,
    ):
        searcher = AdversarialSearcher(stockfish, maia, book=book)
        settings = GeneratorConfig(games=args.games, opponent_rating=args.rating)
        generator = DPOGenerator(searcher, maia, config=settings)

        if args.viewer:
            from src.ui.training_viewer import run_with_viewer

            stats = run_with_viewer(generator, args.output)
        else:
            stats = generator.run(args.output)

    logger.info("wrote %d pairs to %s", stats.pairs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
