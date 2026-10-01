"""Do 1100-1700 humans err more after a skew move? Asked of the Lichess database.

The live bot cannot answer this at any useful pace: three hours online drew no
challenges, and bots cannot enter pools. Human-vs-human games can, by the
thousand, because the question was never about the bot. It is whether the
opponent of whoever plays a mined skew move goes on to make more mistakes.

The design is a comparison at a fixed position. A game that reaches a mined
position P with the skew side to move is *treated* if that player chose the
skew move, *control* if they chose anything else. Both groups reached the same
position through the same opponent choices, so the opponent has not been
selected by their opening the way "reached a skew position or not" would select
them. What remains is who chooses the skew move: stronger players might choose
it more or less often. So the comparison is stratified by position and by both
players' rating, 200 points to a band, and the report prints the rating balance
so the reader can see how much that mattered.

Three stages, each resumable and each streaming its output:

    scan      read a PGN dump (a URL, a file, or stdin; .zst is decompressed in
              process), keep rated 1100-1700 blitz and rapid games that reach a
              mined position, up to --per-group per position and group
    evaluate  score the opponent's moves after P with Stockfish, in parallel
    report    crude and stratified differences, with a cluster bootstrap

The rate differences use the Mantel-Haenszel estimator for rates, and the
errors are bootstrapped over games: a game's moves share an opponent and a
clock, so counting moves as independent trials would overstate precision.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import io
import json
import logging
import math
import random
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any, BinaryIO, Callable, Dict, Final, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

import chess

from src.eval.live_analyzer import GameStats, read_jsonl, score_game

__all__ = ["iter_games", "scan", "mh_difference", "report", "main"]

logger = logging.getLogger(__name__)

DEFAULT_SKEW: Final[Path] = Path("build/skew_positions.jsonl")
DEFAULT_CANDIDATES: Final[Path] = Path("build/db_candidates.jsonl")
DEFAULT_EVALUATED: Final[Path] = Path("build/db_evaluated.jsonl")

SPEEDS: Final[Tuple[str, ...]] = ("Rated Blitz", "Rated Rapid")
"""Lichess ``Event`` prefixes for the mined speeds; tournament games included."""

RATING_BAND: Final[int] = 200
CORE_BAND: Final[Tuple[int, int]] = (1100, 1700)
BOOTSTRAP_REPS: Final[int] = 200

_COMMENT = re.compile(r"\{[^}]*\}")
_MOVE_NUMBER = re.compile(r"^\d+\.+$")
_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2", "*"})


# -- reading the dump ---------------------------------------------------------


def iter_games(lines: Iterator[str]) -> Iterator[Tuple[Dict[str, str], List[str]]]:
    """Headers and SAN moves per game. Comments, clocks, evals and glyphs removed."""
    headers: Dict[str, str] = {}
    movetext: List[str] = []
    for line in lines:
        if line.startswith("["):
            if movetext:  # a game without a trailing blank line
                yield headers, _sans(" ".join(movetext))
                headers, movetext = {}, []
            key, _, value = line[1:].rstrip().rstrip("]").partition(" ")
            headers[key] = value.strip('"')
        elif line.strip():
            movetext.append(line.strip())
        elif movetext:
            yield headers, _sans(" ".join(movetext))
            headers, movetext = {}, []
    if movetext:
        yield headers, _sans(" ".join(movetext))


def _sans(movetext: str) -> List[str]:
    tokens = _COMMENT.sub(" ", movetext).split()
    return [
        token.rstrip("?!") for token in tokens
        if not _MOVE_NUMBER.match(token) and token not in _RESULTS and not token.startswith("$")
    ]


def _open_source(source: str) -> Iterator[str]:
    """Text lines from a URL, a path or ``-``; ``.zst`` is decompressed as it streams."""
    import sys
    import urllib.request

    import pyzstd

    raw: BinaryIO
    if source == "-":
        raw = sys.stdin.buffer
    elif source.startswith(("http://", "https://")):
        raw = cast(BinaryIO, urllib.request.urlopen(source))  # an explicit, user-supplied URL
    else:
        raw = open(source, "rb")
    if source.endswith(".zst"):
        raw = cast(BinaryIO, pyzstd.ZstdFile(raw))
    return iter(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))


def _skew_targets(path: Path) -> Dict[str, set[str]]:
    """Position (EPD, skew side to move) to its mined skew moves."""
    targets: Dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        targets[chess.Board(str(row["fen"])).epd()].add(str(row["uci"]))
    return dict(targets)


def _elo(headers: Mapping[str, str], key: str) -> int:
    try:
        return int(headers.get(key, ""))
    except ValueError:
        return 0


def scan(
    games: Iterator[Tuple[Dict[str, str], List[str]]],
    targets: Mapping[str, set[str]],
    sink: IO[str],
    *,
    min_elo: int,
    max_elo: int,
    per_group: int,
    max_ply: int,
    max_games: Optional[int] = None,
) -> Dict[str, int]:
    """Write one line per kept game, at its first mined position. Returns counts.

    Only a game's first mined position counts: a game is one sample, and taking
    a later position too would let one opponent appear in both groups.
    """
    filled: Dict[Tuple[str, bool], int] = defaultdict(int)
    counts = {"read": 0, "eligible": 0, "kept": 0}
    started = time.monotonic()

    for headers, sans in games:
        counts["read"] += 1
        if max_games is not None and counts["read"] > max_games:
            break
        if counts["read"] % 500_000 == 0:
            logger.info("scan: %s read, %s eligible, %s kept (%.0f games/s)",
                        f"{counts['read']:,}", f"{counts['eligible']:,}", f"{counts['kept']:,}",
                        counts["read"] / (time.monotonic() - started))
        if not headers.get("Event", "").startswith(SPEEDS) or headers.get("Variant", "Standard") != "Standard":
            continue
        if headers.get("Termination") == "Abandoned":
            continue
        if "BOT" in (headers.get("WhiteTitle"), headers.get("BlackTitle")):
            continue  # the dumps include Bot API games; the question is about humans
        white, black = _elo(headers, "WhiteElo"), _elo(headers, "BlackElo")
        if not (min_elo <= white <= max_elo and min_elo <= black <= max_elo):
            continue
        counts["eligible"] += 1

        board = chess.Board()
        for ply, san in enumerate(sans[:max_ply]):
            try:
                move = board.parse_san(san)
            except ValueError:
                break
            skew_moves = targets.get(board.epd())
            if skew_moves is not None:
                treated = move.uci() in skew_moves
                group = (board.epd(), treated)
                if filled[group] < per_group:
                    filled[group] += 1
                    counts["kept"] += 1
                    mover = board.turn
                    sink.write(json.dumps({
                        "site": headers.get("Site", ""), "fen": board.fen(), "ply": ply,
                        "played": move.uci(), "treated": treated,
                        "mover": "white" if mover == chess.WHITE else "black",
                        "mover_elo": white if mover == chess.WHITE else black,
                        "opponent_elo": black if mover == chess.WHITE else white,
                        "speed": "blitz" if "Blitz" in headers["Event"] else "rapid",
                        "result": headers.get("Result", "*"),
                        "moves": sans[ply:],
                    }) + "\n")
                    sink.flush()
                break
            board.push(move)

    logger.info("scan: done. %s read, %s eligible, %s kept in %d groups",
                f"{counts['read']:,}", f"{counts['eligible']:,}", f"{counts['kept']:,}", len(filled))
    return counts


# -- scoring ------------------------------------------------------------------


def _mover_score(result: str, mover: str) -> float:
    white = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
    return white if mover == "white" else 1.0 - white


def _evaluate_batch(rows: Sequence[Dict[str, Any]], depth: int, plies: int) -> List[Dict[str, Any]]:
    """Worker: one Stockfish for the batch, every opponent move after P scored."""
    from src.engine.stockfish import StockfishEvaluator

    logging.basicConfig(level=logging.WARNING)
    out: List[Dict[str, Any]] = []
    with StockfishEvaluator() as stockfish:
        def evaluate(board: chess.Board) -> int:
            cp = stockfish.evaluate(board, depth=depth).centipawns
            return cp if board.turn == chess.WHITE else -cp

        for row in rows:
            stats = score_game(
                str(row["site"]), "treated" if row["treated"] else "control", str(row["result"]),
                list(row["moves"])[:plies], chess.WHITE if row["mover"] == "white" else chess.BLACK,
                set(), evaluate, initial_fen=str(row["fen"]),
            )
            out.append({
                **asdict(stats), "epd": chess.Board(str(row["fen"])).epd(),
                "treated": bool(row["treated"]), "mover_elo": int(row["mover_elo"]),
                "opponent_elo": int(row["opponent_elo"]),
                "mover_score": _mover_score(str(row["result"]), str(row["mover"])),
            })
    return out


# -- statistics ---------------------------------------------------------------

Row = Mapping[str, Any]


def _stratum(row: Row) -> Tuple[str, int, int]:
    return (str(row["epd"]), int(row["mover_elo"]) // RATING_BAND, int(row["opponent_elo"]) // RATING_BAND)


def mh_difference(rows: Sequence[Row], value: Callable[[Row], float], exposure: Callable[[Row], float]) -> float:
    """Mantel-Haenszel treated-minus-control difference in ``value`` per ``exposure``.

    With exposure = opponent moves this is a rate difference per move; with
    exposure = 1 it is a difference in per-game means. Strata with only one
    group carry no comparison and drop out.
    """
    cells: Dict[Tuple[str, int, int], List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for row in rows:
        cell = cells[_stratum(row)]
        offset = 0 if row["treated"] else 2
        cell[offset] += value(row)
        cell[offset + 1] += exposure(row)
    numerator = denominator = 0.0
    for treated_value, treated_exposure, control_value, control_exposure in cells.values():
        total = treated_exposure + control_exposure
        if treated_exposure and control_exposure:
            numerator += (treated_value * control_exposure - control_value * treated_exposure) / total
            denominator += treated_exposure * control_exposure / total
    return numerator / denominator if denominator else 0.0


def _crude(rows: Sequence[Row], value: Callable[[Row], float], exposure: Callable[[Row], float]) -> float:
    total = sum(exposure(row) for row in rows)
    return sum(value(row) for row in rows) / total if total else 0.0


METRICS: Final[Tuple[Tuple[str, float, Callable[[Row], float], Callable[[Row], float]], ...]] = (
    ("opp. blunder rate %", 100.0, lambda r: float(r["blunders"]), lambda r: float(r["opponent_moves"])),
    ("opp. mean cp loss", 1.0, lambda r: float(r["cp_lost"]), lambda r: float(r["opponent_moves"])),
    ("opp. max error (cp)", 1.0, lambda r: float(r["max_error"]), lambda r: 1.0),
    ("skew side's score", 1.0, lambda r: float(r["mover_score"]), lambda r: 1.0),
)


def report(rows: Sequence[Row], seed: int = 0) -> str:
    rows = [row for row in rows if row["opponent_moves"]]
    treated = [row for row in rows if row["treated"]]
    control = [row for row in rows if not row["treated"]]
    if not treated or not control:
        return "need games in both groups"

    lines = [
        f"games: {len(treated)} treated (skew move played), {len(control)} control, "
        f"{len({str(r['epd']) for r in rows})} positions",
        "rating balance (mean): "
        f"skew side {statistics.fmean(r['mover_elo'] for r in treated):.0f} vs "
        f"{statistics.fmean(r['mover_elo'] for r in control):.0f}, opponent "
        f"{statistics.fmean(r['opponent_elo'] for r in treated):.0f} vs "
        f"{statistics.fmean(r['opponent_elo'] for r in control):.0f}",
        "",
        f"  {'':<22}{'control':>10}{'treated':>10}{'crude diff':>12}{'stratified':>12}{'± se':>9}{'sigma':>8}",
    ]
    rng = random.Random(seed)
    for name, scale, value, exposure in METRICS:
        estimate = mh_difference(rows, value, exposure)
        replicates = [
            mh_difference([rows[rng.randrange(len(rows))] for _ in rows], value, exposure)
            for _ in range(BOOTSTRAP_REPS)
        ]
        se = statistics.stdev(replicates)
        control_rate, treated_rate = _crude(control, value, exposure), _crude(treated, value, exposure)
        lines.append(
            f"  {name:<22}{control_rate * scale:>10.2f}{treated_rate * scale:>10.2f}"
            f"{(treated_rate - control_rate) * scale:>+12.2f}{estimate * scale:>+12.2f}"
            f"{se * scale:>9.2f}{(estimate / se if se else math.nan):>+8.1f}"
        )
    lines.append("\n  stratified = within position and 200-point bands of both players' ratings;"
                 f" se from {BOOTSTRAP_REPS} bootstrap resamples of games")
    return "\n".join(lines)


# -- command line -------------------------------------------------------------


def _chunks(rows: Sequence[Dict[str, Any]], size: int) -> Iterator[Sequence[Dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.eval.skew_database", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    scan_cmd = sub.add_parser("scan", help="Find games that reach a mined position.")
    scan_cmd.add_argument("--source", required=True, help="URL, path, or - for stdin; .zst is decompressed.")
    scan_cmd.add_argument("--skew", type=Path, default=DEFAULT_SKEW)
    scan_cmd.add_argument("--out", type=Path, default=DEFAULT_CANDIDATES)
    scan_cmd.add_argument("--min-elo", type=int, default=1000,
                          help="Wider than the mined 1100-1700, so the report can say whether "
                               "the effect holds outside it; the core band is reported on its own.")
    scan_cmd.add_argument("--max-elo", type=int, default=2000)
    scan_cmd.add_argument("--per-group", type=int, default=30,
                          help="Games kept per position and group (treated, control).")
    scan_cmd.add_argument("--max-ply", type=int, default=11)
    scan_cmd.add_argument("--max-games", type=int, default=None, help="Stop after reading this many games.")

    eval_cmd = sub.add_parser("evaluate", help="Score the opponent's moves after the position.")
    eval_cmd.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    eval_cmd.add_argument("--out", type=Path, default=DEFAULT_EVALUATED)
    eval_cmd.add_argument("--depth", type=int, default=8)
    eval_cmd.add_argument("--plies", type=int, default=40, help="Plies scored after the position.")
    eval_cmd.add_argument("--workers", type=int, default=None)

    report_cmd = sub.add_parser("report", help="Compare treated and control games.")
    report_cmd.add_argument("--evaluated", type=Path, default=DEFAULT_EVALUATED)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.stage == "scan":
        targets = _skew_targets(args.skew)
        logger.info("scan: %d mined positions from %s", len(targets), args.skew)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as sink:
            scan(iter_games(_open_source(args.source)), targets, sink,
                 min_elo=args.min_elo, max_elo=args.max_elo, per_group=args.per_group,
                 max_ply=args.max_ply, max_games=args.max_games)
        return 0

    if args.stage == "evaluate":
        from src.eval.arena import safe_workers

        done = {str(row["game"]) for row in read_jsonl(args.out)}
        pending = [row for row in read_jsonl(args.candidates) if str(row["site"]) not in done]
        workers = safe_workers(args.workers)
        logger.info("evaluate: %d to score, %d already done, %d workers", len(pending), len(done), workers)
        with futures.ProcessPoolExecutor(max_workers=workers) as pool, \
                args.out.open("a", encoding="utf-8") as sink:
            jobs = [pool.submit(_evaluate_batch, list(batch), args.depth, args.plies)
                    for batch in _chunks(pending, 25)]
            for finished, job in enumerate(futures.as_completed(jobs), 1):
                for row in job.result():
                    sink.write(json.dumps(row) + "\n")
                sink.flush()
                if finished % 20 == 0 or finished == len(jobs):
                    logger.info("evaluate: %d/%d batches", finished, len(jobs))
        return 0

    rows = list(read_jsonl(args.evaluated))
    core = [r for r in rows if CORE_BAND[0] <= r["mover_elo"] <= CORE_BAND[1]
            and CORE_BAND[0] <= r["opponent_elo"] <= CORE_BAND[1]]
    print(f"BOTH PLAYERS {CORE_BAND[0]}-{CORE_BAND[1]} (the band the skew was mined in)")
    print(report(core))
    print("\nEVERY GAME SCANNED (default 1000-2000)")
    print(report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
