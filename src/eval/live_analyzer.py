"""Score the live games per opponent move, and compare the randomised arms.

The bridge draws each game's book at random and logs the draw
(``build/live_games.jsonl``). This fetches the finished games, scores every
opponent move with Stockfish, caches the per-game totals, and reports three
tables, in the order they should be trusted:

1. **By arm.** Standard book against repertoire, every game in the arm it was
   assigned. The coin flip makes this the only unbiased comparison. It is
   diluted -- most repertoire games never reach a skew move -- but dilution
   costs power, not validity.
2. **Scaled to exposure.** The arm difference divided by how much more often the
   repertoire arm actually played a skew move: the effect on a game that got
   one. Same test, same sigma, rescaled.
3. **Exposed against not, pooled.** Whether a game reaches a skew move depends
   on the opponent's own opening choices, and opponents who follow the main line
   for ten plies are not the ones who leave it on ply three. This table measures
   that difference as much as the skew. Descriptive only.

"Exposed" means the bot played a mined skew move, not that the game passed a
mined position: every game passes the initial position, which the miner visited
first.

Safe to run on a cron while the bot plays. The game log is only read, never
locked, and a half-written last line is skipped until it is complete; the cache
is appended one game at a time, so an interrupted run loses nothing it finished.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Final, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import chess

from src.eval.arena import BLUNDER_CP, DECISIVE_CP, ERROR_CAP

__all__ = ["GameStats", "score_game", "summarise", "main"]

logger = logging.getLogger(__name__)

DEFAULT_LOG: Final[Path] = Path("build/live_games.jsonl")
DEFAULT_CACHE: Final[Path] = Path("build/evaluated_games.jsonl")
DEFAULT_SKEW: Final[Path] = Path("build/skew_positions.jsonl")
DEFAULT_DEPTH: Final[int] = 10

EXPORT_BATCH: Final[int] = 300
"""Lichess's limit on ids per ``/api/games/export/_ids`` request."""

UNFINISHED: Final[frozenset[str]] = frozenset({"created", "started"})
NOT_PLAYED: Final[frozenset[str]] = frozenset({"aborted", "noStart"})
"""Cached so they are not fetched again, and left out of every table."""

RATE_LIMIT_WAIT: Final[float] = 60.0
MAX_ATTEMPTS: Final[int] = 5


@dataclass(frozen=True, slots=True)
class GameStats:
    """One game's opponent-move totals. Sums rather than rates, so arms pool
    by adding and the standard errors can be clustered by game."""

    game: str
    arm: str
    status: str
    opponent_moves: int
    blunders: int
    cp_lost: int
    max_error: int
    decisive_ply: Optional[int]
    skew_moves: int
    """Mined skew moves the bot played. Non-zero is what "exposed" means."""


def _stm_score(board: chess.Board, evaluate: Callable[[chess.Board], int]) -> int:
    """Side-to-move score, clamped so one mate cannot dominate a sum."""
    if board.is_checkmate():
        return -ERROR_CAP
    if board.is_game_over(claim_draw=True):
        return 0
    return max(-ERROR_CAP, min(ERROR_CAP, evaluate(board)))


def score_game(
    game: str,
    arm: str,
    status: str,
    sans: Sequence[str],
    bot_colour: chess.Color,
    skew_moves: Set[Tuple[str, str]],
    evaluate: Callable[[chess.Board], int],
    initial_fen: Optional[str] = None,
) -> GameStats:
    """Score every opponent move by what it lost against the engine's best.

    ``evaluate`` returns the side-to-move score in centipawns. Each position is
    evaluated once, so a move's loss is the mover's score before it plus the
    opponent's score after it.
    """
    board = chess.Board(initial_fen) if initial_fen else chess.Board()
    scores = [_stm_score(board, evaluate)]
    moves = blunders = cp_lost = max_error = skewed = 0
    decisive: Optional[int] = None

    for san in sans:
        mover = board.turn
        move = board.parse_san(san)
        if mover == bot_colour and (board.epd(), move.uci()) in skew_moves:
            skewed += 1
        board.push(move)
        scores.append(_stm_score(board, evaluate))
        if mover != bot_colour:
            loss = min(ERROR_CAP, max(0, scores[-2] + scores[-1]))
            moves += 1
            blunders += loss > BLUNDER_CP
            cp_lost += loss
            max_error = max(max_error, loss)
        bot_score = scores[-1] if board.turn == bot_colour else -scores[-1]
        if bot_score >= DECISIVE_CP:
            decisive = board.ply() if decisive is None else decisive
        else:
            decisive = None  # gave it back; that was not the deciding moment

    return GameStats(
        game=game, arm=arm, status=status, opponent_moves=moves, blunders=blunders,
        cp_lost=cp_lost, max_error=max_error, decisive_ply=decisive, skew_moves=skewed,
    )


# -- statistics ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Summary:
    games: int
    moves: int
    blunder_rate: float
    blunder_se: float
    cp_loss: float
    cp_loss_se: float
    max_error: float
    max_error_se: float
    decisive_ply: float
    decisive_se: float
    exposed_share: float


def _ratio(numerators: Sequence[float], denominators: Sequence[float]) -> Tuple[float, float]:
    """A pooled rate and its standard error clustered by game.

    Moves within a game share an opponent, a position type and a clock, so they
    are not independent trials; the binomial error over moves the arena uses
    would overstate the precision here.
    """
    total = sum(denominators)
    if not total:
        return 0.0, 0.0
    rate = sum(numerators) / total
    n = len(denominators)
    if n < 2:
        return rate, 0.0
    spread = sum((x - rate * m) ** 2 for x, m in zip(numerators, denominators))
    return rate, math.sqrt(spread * n / (n - 1)) / total


def _mean(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return statistics.fmean(values), se


def summarise(games: Sequence[GameStats]) -> Summary:
    moves = [float(g.opponent_moves) for g in games]
    blunder_rate, blunder_se = _ratio([float(g.blunders) for g in games], moves)
    cp_loss, cp_loss_se = _ratio([float(g.cp_lost) for g in games], moves)
    max_error, max_error_se = _mean([float(g.max_error) for g in games if g.opponent_moves])
    decisive, decisive_se = _mean([float(g.decisive_ply) for g in games if g.decisive_ply is not None])
    return Summary(
        games=len(games), moves=int(sum(moves)),
        blunder_rate=blunder_rate, blunder_se=blunder_se,
        cp_loss=cp_loss, cp_loss_se=cp_loss_se,
        max_error=max_error, max_error_se=max_error_se,
        decisive_ply=decisive, decisive_se=decisive_se,
        exposed_share=sum(1 for g in games if g.skew_moves) / len(games) if games else 0.0,
    )


def _sigma(a: float, a_se: float, b: float, b_se: float) -> str:
    """Formatted, because zero spread is undefined, not zero: printing +0.0 there
    would read as "no difference" for a difference nothing can yet measure."""
    spread = math.hypot(a_se, b_se)
    return f"{(b - a) / spread:+.1f}" if spread else "n/a"


def _table(title: str, left: Tuple[str, Summary], right: Tuple[str, Summary]) -> List[str]:
    (left_name, a), (right_name, b) = left, right
    rows = [
        f"\n{title}",
        f"  {'':<22}{left_name:>16}{right_name:>16}{'difference':>14}{'sigma':>8}",
        f"  {'games / opp. moves':<22}{f'{a.games} / {a.moves}':>16}{f'{b.games} / {b.moves}':>16}",
        f"  {'played a skew move':<22}{a.exposed_share:>16.1%}{b.exposed_share:>16.1%}",
    ]
    for name, scale, pairs in (
        ("blunder rate %", 100.0, (a.blunder_rate, a.blunder_se, b.blunder_rate, b.blunder_se)),
        ("mean cp loss", 1.0, (a.cp_loss, a.cp_loss_se, b.cp_loss, b.cp_loss_se)),
        ("max error (cp)", 1.0, (a.max_error, a.max_error_se, b.max_error, b.max_error_se)),
        ("plies to decisive", 1.0, (a.decisive_ply, a.decisive_se, b.decisive_ply, b.decisive_se)),
    ):
        x, x_se, y, y_se = pairs
        rows.append(
            f"  {name:<22}{x * scale:>11.2f} ±{x_se * scale:<4.2f}{y * scale:>11.2f} ±{y_se * scale:<4.2f}"
            f"{(y - x) * scale:>+14.2f}{_sigma(x, x_se, y, y_se):>8}"
        )
    return rows


def report(games: Sequence[GameStats], control: str, treatment: str) -> str:
    played = [g for g in games if g.status not in NOT_PLAYED]
    if not played:
        return "no finished games yet"
    by_arm = {arm: summarise([g for g in played if g.arm == arm]) for arm in (control, treatment)}
    lines = _table("BY ARM (randomised; the unbiased comparison)",
                   (control, by_arm[control]), (treatment, by_arm[treatment]))

    a, b = by_arm[control], by_arm[treatment]
    uplift = b.exposed_share - a.exposed_share
    if uplift > 0:
        lines += [
            f"\nSCALED TO EXPOSURE (arm difference / {uplift:.1%} more games playing a skew move)",
            f"  blunder rate %{100 * (b.blunder_rate - a.blunder_rate) / uplift:>+12.2f}   "
            f"mean cp loss{(b.cp_loss - a.cp_loss) / uplift:>+10.1f}   "
            f"sigma as above: rescaling adds no evidence",
        ]
    else:
        lines.append("\nSCALED TO EXPOSURE: the repertoire arm is not playing more skew moves; nothing to scale")

    exposed = summarise([g for g in played if g.skew_moves])
    unexposed = summarise([g for g in played if not g.skew_moves])
    lines += _table("EXPOSED vs NOT, both arms (confounded by the opponent's opening: descriptive only)",
                    ("not exposed", unexposed), ("exposed", exposed))
    return "\n".join(lines)


# -- I/O ----------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Complete lines only. The bridge may be mid-write on the last one."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row


def load_skew_moves(path: Path) -> Set[Tuple[str, str]]:
    return {(chess.Board(str(row["fen"])).epd(), str(row["uci"])) for row in read_jsonl(path)}


def _export(client: Any, ids: Sequence[str]) -> List[Mapping[str, Any]]:
    """One batch, retried with exponential backoff on 429s and dropped connections."""
    from berserk.exceptions import ResponseError
    import requests

    wait = RATE_LIMIT_WAIT
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return list(client.games.export_multi(*ids, moves=True))
        except ResponseError as exc:
            if exc.status_code != 429 and exc.status_code < 500:
                raise
            logger.warning("lichess: HTTP %s on export, attempt %d/%d, waiting %.0fs",
                           exc.status_code, attempt, MAX_ATTEMPTS, wait)
        except (requests.RequestException, OSError) as exc:
            logger.warning("lichess: export failed (%s), attempt %d/%d, waiting %.0fs",
                           exc, attempt, MAX_ATTEMPTS, wait)
        time.sleep(wait)
        wait *= 2
    raise RuntimeError(f"export failed {MAX_ATTEMPTS} times; try again later")


def _batches(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.live_analyzer",
        description="Score finished live games per opponent move and compare the arms.",
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--skew", type=Path, default=DEFAULT_SKEW,
                        help="The mined positions the committed repertoire was built from.")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--report-only", action="store_true", help="Skip fetching; report the cache.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from src.engine.bot_factory import SKEW, STANDARD

    starts = {str(row["game"]): row for row in read_jsonl(args.log) if row.get("event") == "start"}
    cached = {str(row["game"]): row for row in read_jsonl(args.cache)}
    pending = [game for game in starts if game not in cached]
    logger.info("live: %d games logged, %d cached, %d to fetch", len(starts), len(cached), len(pending))

    if pending and not args.report_only:
        import berserk

        from src import config
        from src.engine.stockfish import StockfishEvaluator

        skew_moves = load_skew_moves(args.skew)
        if not skew_moves:
            logger.error("no mined skew moves at %s; exposure cannot be detected", args.skew)
            return 1
        client = berserk.Client(session=berserk.TokenSession(config.lichess_token()))
        args.cache.parent.mkdir(parents=True, exist_ok=True)

        with StockfishEvaluator() as stockfish, args.cache.open("a", encoding="utf-8") as sink:
            def evaluate(board: chess.Board) -> int:
                cp = stockfish.evaluate(board, depth=args.depth).centipawns
                return cp if board.turn == chess.WHITE else -cp

            for batch in _batches(pending, EXPORT_BATCH):
                for exported in _export(client, batch):
                    game = str(exported.get("id", ""))
                    status = str(exported.get("status", ""))
                    if game not in starts or status in UNFINISHED:
                        continue
                    start = starts[game]
                    stats = score_game(
                        game, str(start.get("arm", "")), status,
                        str(exported.get("moves", "")).split(),
                        chess.WHITE if start.get("colour") == "white" else chess.BLACK,
                        skew_moves, evaluate, exported.get("initialFen"),
                    )
                    sink.write(json.dumps(asdict(stats)) + "\n")
                    sink.flush()
                    cached[game] = asdict(stats)

    games = [GameStats(**row) for row in cached.values()]
    print(report(games, STANDARD.name, SKEW.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
