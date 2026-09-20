"""Head-to-head arena: play a bot against Maia-3 and count the opponent's errors.

``val_pref`` stopped being informative once engine features entered the input.
Mining selects pairs whose chosen move is a worse-ranked trap and whose rejected
move is the search's own best, so "prefer the worse-ranked move" scores 0.6894
on held-out pairs by itself -- against 0.7004 for a trained 19k-parameter
network. The metric mostly asks whether the model noticed the rank column.

This measures the thing the project is actually for: put the bot in front of a
human model and count how often that model errs.

**Win rate will not discriminate and is reported anyway.** A Stockfish-backed
expectimax beats Maia at club ratings in essentially every game -- an earlier
sweep scored 0.999 across four configurations -- so the outcome column is a
sanity check that the games are real, not the result. Blunder rate and mean
centipawn loss are the measurements, because they have headroom.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import logging
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, List, Optional, Sequence, Tuple

import chess

from src.engine.bot_factory import ARMS, BotSpec, build_searcher, engines
from src.engine.search import _rank_mover_relative, _split_at_threshold
from src.engine.stockfish import StockfishEvaluator

__all__ = ["GameResult", "ArmSummary", "play_game", "run_arm", "main"]

logger = logging.getLogger(__name__)

BLUNDER_CP: Final[int] = 200
"""Anderson-Kleinberg's threshold, and the one the mining pipeline uses."""

JUDGE_DEPTH: Final[int] = 6
"""Depth for scoring the opponent's move. Measured at 0.015 mean absolute error
against a depth-12 reference on beta, for a quarter of depth 8's cost."""

DEFAULT_PLIES: Final[int] = 60
DEFAULT_GAMES: Final[int] = 100
DEFAULT_RATING: Final[int] = 1500

MEMORY_PER_WORKER_MB: Final[int] = 700
"""A worker holds torch plus a Stockfish subprocess. Measured near 400MB for the
engine pair alone during the overnight mining run; 700 leaves headroom for the
Maia-3 checkpoint and keeps the box from swapping."""


@dataclass(frozen=True, slots=True)
class GameResult:
    """One completed game, from the bot's point of view."""

    arm: str
    game: int
    bot_white: bool
    outcome: float
    """1.0 win, 0.5 draw, 0.0 loss. Adjudicated by material-free engine score
    when the ply cap is reached before a natural result."""

    adjudicated: bool
    opponent_moves: int
    opponent_blunders: int
    opponent_cp_lost: int
    plies: int

    @property
    def blunder_rate(self) -> float:
        return self.opponent_blunders / self.opponent_moves if self.opponent_moves else 0.0

    @property
    def mean_cp_lost(self) -> float:
        return self.opponent_cp_lost / self.opponent_moves if self.opponent_moves else 0.0


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Aggregate over one arm's games."""

    arm: str
    games: int
    score: float
    decisive: int
    blunder_rate: float
    blunder_rate_stderr: float
    mean_cp_lost: float
    opponent_moves: int
    seconds: float


def _judge(
    stockfish: StockfishEvaluator, board: chess.Board
) -> Tuple[dict[chess.Move, int], int, set[chess.Move]]:
    """Score every legal reply for the side to move, mover-relative."""
    ranked = _rank_mover_relative(
        stockfish.analyse_root_moves(
            board, depth=JUDGE_DEPTH, multipv=board.legal_moves.count()
        ),
        board.turn,
    )
    _sound, blunders = _split_at_threshold(ranked, BLUNDER_CP)
    return {move: score for move, score in ranked}, ranked[0][1], set(blunders)


def play_game(
    spec: BotSpec,
    stockfish: StockfishEvaluator,
    opponent: object,
    *,
    game: int,
    rating: int,
    plies: int,
) -> GameResult:
    """One game, bot against the human model, with the opponent's errors scored."""
    from src.engine.maia3_model import Maia3Evaluator

    assert isinstance(opponent, Maia3Evaluator)
    searcher = build_searcher(spec, stockfish, opponent, opponent_rating=rating)
    rng = random.Random(game)
    bot_white = game % 2 == 0
    board = chess.Board()
    blunders = moves = cp_lost = 0

    for _ in range(plies):
        if board.is_game_over():
            break
        if board.turn == (chess.WHITE if bot_white else chess.BLACK):
            board.push(searcher.search(board, spec.search).move)
            continue

        scores, best, blundering = _judge(stockfish, board)
        distribution = opponent.predict_move_probabilities(board).probabilities
        candidates = list(distribution)
        played = rng.choices(candidates, weights=[distribution[m] for m in candidates])[0]
        moves += 1
        blunders += played in blundering
        cp_lost += max(0, best - scores.get(played, best))
        board.push(played)

    adjudicated = not board.is_game_over()
    if adjudicated:
        evaluation = stockfish.evaluate(board, depth=12)
        white_score = evaluation.win_probability
    else:
        result = board.result()
        white_score = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5

    return GameResult(
        arm=spec.name, game=game, bot_white=bot_white,
        outcome=white_score if bot_white else 1.0 - white_score,
        adjudicated=adjudicated, opponent_moves=moves, opponent_blunders=blunders,
        opponent_cp_lost=cp_lost, plies=board.ply(),
    )


def _play_batch(
    arm: str, games: Sequence[int], rating: int, plies: int
) -> List[GameResult]:
    """Worker entry point. Owns its own engines for the whole batch.

    One Stockfish per process, held across the batch: it is a subprocess and not
    reentrant, and respawning it per game would cost more than the games do.
    """
    logging.basicConfig(level=logging.WARNING)
    spec = ARMS[arm]
    results: List[GameResult] = []
    for stockfish, opponent in engines(rating):
        for game in games:
            results.append(
                play_game(spec, stockfish, opponent, game=game, rating=rating, plies=plies)
            )
    return results


def _chunks(games: int, workers: int) -> List[List[int]]:
    """Deal colour pairs, so every worker plays both sides.

    Plain round-robin looks balanced and is not: with an even worker count,
    ``game % workers`` hands worker 0 every even-numbered game, and colour is
    assigned by game parity, so that worker plays White exclusively. The totals
    still come out 50/50, but no individual batch does -- which matters the
    moment a worker dies partway or a per-worker number is read.
    """
    buckets: List[List[int]] = [[] for _ in range(workers)]
    for pair in range((games + 1) // 2):
        bucket = buckets[pair % workers]
        for game in (2 * pair, 2 * pair + 1):
            if game < games:
                bucket.append(game)
    return [bucket for bucket in buckets if bucket]


def summarise(arm: str, results: Sequence[GameResult], seconds: float) -> ArmSummary:
    total_moves = sum(result.opponent_moves for result in results)
    total_blunders = sum(result.opponent_blunders for result in results)
    rate = total_blunders / total_moves if total_moves else 0.0
    # Binomial standard error over moves, not games: a game is not one trial.
    stderr = math.sqrt(rate * (1.0 - rate) / total_moves) if total_moves else 0.0
    return ArmSummary(
        arm=arm,
        games=len(results),
        score=statistics.fmean(result.outcome for result in results) if results else 0.0,
        decisive=sum(1 for result in results if not result.adjudicated),
        blunder_rate=rate,
        blunder_rate_stderr=stderr,
        mean_cp_lost=(
            sum(result.opponent_cp_lost for result in results) / total_moves
            if total_moves else 0.0
        ),
        opponent_moves=total_moves,
        seconds=seconds,
    )


def run_arm(
    arm: str, *, games: int, rating: int, plies: int, workers: int
) -> Tuple[ArmSummary, List[GameResult]]:
    """Play one arm's games across a process pool."""
    started = time.perf_counter()
    batches = _chunks(games, workers)
    results: List[GameResult] = []
    with futures.ProcessPoolExecutor(max_workers=len(batches)) as pool:
        pending = [
            pool.submit(_play_batch, arm, batch, rating, plies) for batch in batches
        ]
        for done in futures.as_completed(pending):
            results.extend(done.result())
    results.sort(key=lambda result: result.game)
    return summarise(arm, results, time.perf_counter() - started), results


def safe_workers(requested: Optional[int]) -> int:
    """Cap concurrency by memory rather than by core count.

    Each worker holds a Stockfish subprocess and a torch checkpoint. The
    overnight mining run showed the engine pair alone near 400MB, and a box that
    starts swapping runs slower than one with half the workers.
    """
    if requested is not None:
        return max(1, requested)
    # A worker is two processes, not one: the Python process holding torch and
    # the Stockfish subprocess it drives. Filling every core with workers
    # therefore doubles the machine's process count -- measured at load average
    # 54 on ten cores, which starves the games it is trying to run.
    cores = max(1, (os.cpu_count() or 4) // 2)
    try:
        total_mb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except (ValueError, OSError, AttributeError):  # pragma: no cover - non-POSIX
        return max(1, cores // 2)
    by_memory = max(1, int(total_mb * 0.6) // MEMORY_PER_WORKER_MB)
    return max(1, min(cores, by_memory))


def _report(summaries: Sequence[ArmSummary]) -> None:
    print(f"\n{'arm':<10}{'games':>7}{'score':>8}{'decisive':>10}"
          f"{'blunder rate':>15}{'mean cp lost':>14}{'opp moves':>11}{'min':>7}")
    for summary in summaries:
        print(
            f"{summary.arm:<10}{summary.games:>7}{summary.score:>8.3f}{summary.decisive:>10}"
            f"{summary.blunder_rate:>11.4f} +/-{summary.blunder_rate_stderr:.4f}"
            f"{summary.mean_cp_lost:>14.0f}{summary.opponent_moves:>11}"
            f"{summary.seconds / 60:>7.1f}"
        )
    if len(summaries) == 2:
        first, second = summaries
        delta = second.blunder_rate - first.blunder_rate
        stderr = math.sqrt(first.blunder_rate_stderr ** 2 + second.blunder_rate_stderr ** 2)
        sigma = delta / stderr if stderr else 0.0
        print(
            f"\n  {second.arm} - {first.arm}: {delta:+.4f} blunder rate "
            f"({sigma:+.1f} sigma). "
            + ("Significant at 2 sigma." if abs(sigma) >= 2 else "Not significant.")
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.arena",
        description="Play bot configurations against Maia-3 and count opponent errors.",
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="Games per arm.")
    parser.add_argument("--rating", type=int, default=DEFAULT_RATING,
                        help="Maia-3 rating, used for both seats.")
    parser.add_argument("--plies", type=int, default=DEFAULT_PLIES)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--workers", type=int, default=None,
                        help="Default: bounded by memory, not core count.")
    parser.add_argument("--output", type=Path, default=Path("build/arena.jsonl"))
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    workers = safe_workers(args.workers)
    logger.info(
        "arena: %d games per arm vs maia3@%d, %d plies, %d workers",
        args.games, args.rating, args.plies, workers,
    )

    summaries: List[ArmSummary] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for arm in args.arms:
            summary, results = run_arm(
                arm, games=args.games, rating=args.rating, plies=args.plies, workers=workers
            )
            summaries.append(summary)
            for result in results:
                handle.write(json.dumps(asdict(result)) + "\n")
            logger.info("arena: %s done in %.1f min", arm, summary.seconds / 60)

    _report(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
