"""Mine opening positions that are engine-equal but humanly lopsided.

Milestones 10 through 14 tested three ways of picking a *move* that induces
opponent error -- blunder potential, a Maia-weighted variant, and a trained
re-ranker -- and all three measured null over a properly powered 1,000-game
arena. Relaxing the safety floor did not help either: a gambit was legal in 10
of 40 sharp positions and won the expected-utility comparison in one, losing the
rest by a median of 340cp. The expectimax evaluates traps correctly and rejects
them on their merits.

What that leaves is choosing a different *position* rather than a different move
inside one. Park's Engine-Equal study measured the effect directly: across 16.1M
Lichess occurrences of 1,661 positions Stockfish scores within 10cp of zero,
human outcomes carry reproducible skews that survive account-disjoint splits,
time splits, rating-band splits, and an out-of-sample month. This mines that
signal live.

**Three things the obvious design gets wrong, and this one does not.**

The baseline is not 0.50. White scores 0.5190 from the starting position across
blitz and rapid at 1600-2000, so a threshold measured against even odds bakes
the first-move advantage into every result. Skew here is measured against the
root baseline for the same colour.

Draws are not noise to be dropped. ``W/(W+B)`` discards them, and equal
positions -- exactly the ones being mined -- are where they concentrate. Score,
``(W + D/2)/N``, is the primary measure; the win ratio is recorded beside it so
a threshold stated in either can be checked.

Rarity correlates with strength, and it turns out not to matter. Average rating
climbs from 1839 on 1.e4 to 1892 on 1.Nf3 in the same query, which looks like
the repertoire-selection confound the source study removes with a
rating-calibrated model -- stronger players choosing rarer lines and carrying
the results with them. Measured over 124 mined moves it explains 1% of the
variance in skew, with the sign running the wrong way for that story
(r = -0.113). Engine advantage explains 2%. ``average_rating`` and
``evaluation_cp`` are recorded on every entry anyway, because the check is
cheap and a larger mine could say otherwise.

What survives is the signal itself. Tightening the equality gate tenfold, from
100cp to the source study's own 10cp, moves mean skew from 0.0369 to 0.0354 --
so these are positions the engine calls level where humans do not score level,
which is what the study reports and what the move-selection milestones could
not produce.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Final, List, Optional, Sequence, Tuple

import chess
import requests

from src import config
from src.engine.stockfish import StockfishEvaluator

__all__ = ["SkewEntry", "ExplorerClient", "mine", "main"]

logger = logging.getLogger(__name__)

EXPLORER_URL: Final[str] = "https://explorer.lichess.ovh/lichess"
"""The endpoint requires an Authorization header. Unauthenticated requests get
an nginx 401, which is why earlier work in this repository scraped a third-party
mirror instead; a bot token is sufficient."""

DEFAULT_OUTPUT: Final[Path] = Path("build/skew_positions.jsonl")
DEFAULT_SPEEDS: Final[str] = "blitz,rapid"
DEFAULT_RATINGS: Final[str] = "1600,1800,2000"
DEFAULT_MAX_PLY: Final[int] = 20
DEFAULT_MIN_GAMES: Final[int] = 5_000
"""Minimum games behind a move before its skew is believed. The source study's
median skew is about two points of score, so a line with a few hundred games
cannot distinguish one from noise: the standard error on 500 games is 2.2
points."""

DEFAULT_MIN_SKEW: Final[float] = 0.06
"""Score points above the same-colour baseline. Six points sits in the tail the
source study reports (8-18 points for its strongest positions) rather than near
its median, which is where the signal is worth acting on."""

EQUAL_CP: Final[int] = 100
"""Default ``|eval|`` bound for "the engine calls this equal".

Park's study uses 10cp, an order of magnitude tighter. Measured over 124 mined
moves, the choice barely matters: mean skew is 0.0369 at 100cp and 0.0354 at
10cp, and engine advantage explains 2% of skew variance either way. The looser
gate is the default because it keeps five times as many positions for the same
mean skew; ``--max-eval 10`` reproduces the study's own criterion."""

EVAL_DEPTH: Final[int] = 12
BRANCHES_PER_NODE: Final[int] = 4
"""Explorer moves considered per position. The tail beyond this is rarely played
and rarely has the game count to clear ``min_games`` anyway."""

REQUEST_INTERVAL: Final[float] = 1.1
"""Seconds between calls. The endpoint is generous but not free, and a miner
that gets itself rate-limited finishes slower than one that waits."""

MAX_BACKOFF: Final[float] = 120.0


@dataclass(frozen=True, slots=True)
class SkewEntry:
    """One engine-equal move whose human results lean one way."""

    fen: str
    san: str
    uci: str
    ply: int
    bot_color: str
    """Whose side the skew favours: ``white`` or ``black``."""

    games: int
    score: float
    """``(W + D/2)/N`` from the favoured side's point of view."""

    win_ratio: float
    """``W/(W+B)``, the draw-free measure, recorded for comparison."""

    skew: float
    """``score`` minus the same-colour baseline. This is the number that ranks."""

    average_rating: int
    """Explorer's mean rating for this move. Rises with rarity, so a high skew on
    an unusually strong average is partly a property of who plays the line."""

    evaluation_cp: int
    """Stockfish's score after the move, from the favoured side's point of view."""


class ExplorerClient:
    """Lichess opening explorer, with backoff and a request floor."""

    def __init__(self, token: str, *, speeds: str, ratings: str) -> None:
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.speeds = speeds
        self.ratings = ratings
        self._last_call = 0.0

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> ExplorerClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def lookup(self, board: chess.Board) -> Optional[Dict[str, Any]]:
        """Explorer statistics for one position, or ``None`` if unavailable."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

        backoff = 2.0
        for attempt in range(6):
            self._last_call = time.monotonic()
            try:
                response = self.session.get(
                    EXPLORER_URL,
                    params={
                        "variant": "standard",
                        "speeds": self.speeds,
                        "ratings": self.ratings,
                        "fen": board.fen(),
                    },
                    timeout=30,
                )
            except requests.RequestException as exc:
                logger.warning("explorer: request failed (%s), retry %d", exc, attempt + 1)
                time.sleep(min(MAX_BACKOFF, backoff))
                backoff *= 2
                continue

            if response.status_code == 200:
                payload: Dict[str, Any] = response.json()
                return payload
            if response.status_code == 401:
                raise RuntimeError(
                    "explorer returned 401. The endpoint needs a Lichess token: "
                    "export LICHESS_API_TOKEN before mining."
                )
            if response.status_code in (429, 503):
                wait = float(response.headers.get("Retry-After", backoff))
                logger.info("explorer: %d, waiting %.0fs", response.status_code, wait)
                time.sleep(min(MAX_BACKOFF, wait))
                backoff *= 2
                continue
            logger.warning("explorer: HTTP %d, skipping this position", response.status_code)
            return None

        logger.error("explorer: gave up after repeated failures")
        return None


def _counts(entry: Dict[str, Any]) -> Tuple[int, int, int, int]:
    white = int(entry.get("white", 0))
    draws = int(entry.get("draws", 0))
    black = int(entry.get("black", 0))
    return white, draws, black, white + draws + black


def score_for(white: int, draws: int, black: int, color: chess.Color) -> float:
    """Score from ``color``'s point of view, draws counted as a half."""
    total = white + draws + black
    if total == 0:
        return 0.0
    wins = white if color == chess.WHITE else black
    return (wins + draws / 2.0) / total


def win_ratio_for(white: int, black: int, color: chess.Color) -> float:
    """The draw-free measure. Undefined with no decisive games; reported as 0.5."""
    decisive = white + black
    if decisive == 0:
        return 0.5
    wins = white if color == chess.WHITE else black
    return wins / decisive


def baseline(client: ExplorerClient) -> Tuple[float, float]:
    """Score for White and for Black from the starting position.

    Measured rather than assumed: White scores about 0.518 across these speeds
    and bands, and calling that 0.500 would credit every White line with two
    points of skew it has not earned.
    """
    payload = client.lookup(chess.Board())
    if payload is None:
        raise RuntimeError("could not read the explorer baseline")
    white, draws, black, total = _counts(payload)
    if total == 0:
        raise RuntimeError("explorer returned an empty baseline")
    white_score = score_for(white, draws, black, chess.WHITE)
    logger.info(
        "baseline: White %.4f over %s games (draws %.1f%%)",
        white_score, f"{total:,}", 100.0 * draws / total,
    )
    return white_score, 1.0 - white_score


def _evaluate(
    evaluator: StockfishEvaluator, board: chess.Board, color: chess.Color, depth: int
) -> int:
    """Stockfish score after the move, from ``color``'s point of view."""
    evaluation = evaluator.evaluate(board, depth=depth)
    return evaluation.centipawns if color == chess.WHITE else -evaluation.centipawns


def mine(
    client: ExplorerClient,
    evaluator: StockfishEvaluator,
    *,
    max_ply: int,
    min_games: int,
    min_skew: float,
    max_eval: int = EQUAL_CP,
    depth: int = EVAL_DEPTH,
    max_positions: Optional[int] = None,
) -> List[SkewEntry]:
    """Breadth-first walk of the opening tree, keeping equal-but-skewed moves.

    The frontier only extends through moves that are themselves popular and
    engine-equal. Following a line the engine already considers lost would find
    plenty of "skew" that is simply one side being better.
    """
    baselines = dict(zip((chess.WHITE, chess.BLACK), baseline(client)))
    found: List[SkewEntry] = []
    seen: set[str] = set()
    frontier: Deque[chess.Board] = deque([chess.Board()])
    visited = 0

    while frontier:
        if max_positions is not None and visited >= max_positions:
            break
        board = frontier.popleft()
        if board.ply() >= max_ply:
            continue
        key = board.epd()
        if key in seen:
            continue
        seen.add(key)
        visited += 1

        payload = client.lookup(board)
        if payload is None:
            continue

        mover = board.turn
        for entry in list(payload.get("moves", []))[:BRANCHES_PER_NODE]:
            white, draws, black, total = _counts(entry)
            if total < min_games:
                continue
            try:
                move = board.parse_uci(str(entry["uci"]))
            except (KeyError, ValueError):
                continue

            board.push(move)
            try:
                if board.is_game_over():
                    continue
                evaluation = _evaluate(evaluator, board, mover, depth)
                if abs(evaluation) >= max_eval:
                    continue  # not engine-equal; any skew here is just being better
                child = board.copy(stack=False)
            finally:
                board.pop()

            frontier.append(child)
            score = score_for(white, draws, black, mover)
            skew = score - baselines[mover]
            if skew < min_skew:
                continue

            found.append(
                SkewEntry(
                    fen=board.fen(),
                    san=board.san(move),
                    uci=move.uci(),
                    ply=board.ply(),
                    bot_color="white" if mover == chess.WHITE else "black",
                    games=total,
                    score=round(score, 4),
                    win_ratio=round(win_ratio_for(white, black, mover), 4),
                    skew=round(skew, 4),
                    average_rating=int(entry.get("averageRating", 0)),
                    evaluation_cp=evaluation,
                )
            )
            logger.info(
                "skew: ply %2d %-6s %s  score %.3f (%+.3f) over %s games, eval %+dcp, avg %d",
                board.ply(), board.san(move), found[-1].bot_color,
                score, skew, f"{total:,}", evaluation, found[-1].average_rating,
            )

    logger.info("mine: %d skewed moves from %d positions", len(found), visited)
    return found


def write_jsonl(entries: Sequence[SkewEntry], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for entry in sorted(entries, key=lambda item: -item.skew):
            handle.write(json.dumps(asdict(entry)) + "\n")
    return len(entries)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.skew_miner",
        description="Mine engine-equal opening moves whose human results are skewed.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--speeds", default=DEFAULT_SPEEDS)
    parser.add_argument("--ratings", default=DEFAULT_RATINGS)
    parser.add_argument("--max-ply", type=int, default=DEFAULT_MAX_PLY)
    parser.add_argument("--min-games", type=int, default=DEFAULT_MIN_GAMES,
                        help="Games behind a move before its skew is believed.")
    parser.add_argument("--min-skew", type=float, default=DEFAULT_MIN_SKEW,
                        help="Score points above the same-colour baseline, not above 0.5.")
    parser.add_argument("--max-eval", type=int, default=EQUAL_CP,
                        help="|eval| bound for engine-equal. 10 matches the source study.")
    parser.add_argument("--depth", type=int, default=EVAL_DEPTH)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    try:
        token = config.lichess_token()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    with (
        ExplorerClient(token, speeds=args.speeds, ratings=args.ratings) as client,
        StockfishEvaluator() as evaluator,
    ):
        entries = mine(
            client, evaluator,
            max_ply=args.max_ply, min_games=args.min_games, min_skew=args.min_skew,
            max_eval=args.max_eval, depth=args.depth, max_positions=args.max_positions,
        )
    written = write_jsonl(entries, args.output)
    logger.info("mine: wrote %d entries to %s", written, args.output)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
