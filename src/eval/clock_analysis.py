"""Do humans spend longer -- and blunder more -- when they have few safe replies?

Milestone 19 steers toward positions where the opponent has few moves within
50cp of their best (the safe reply count, SRC), on the theory that those drain
the clock. The arena cannot test that: Maia has no clock. Lichess games can,
because every move carries the clock after it.

One position per game, a uniformly random ply in the middlegame window, from
rated blitz and rapid games between two 1000-2000 humans. For each: the SRC
from the same bounded MultiPV-8 scan the engine uses, the human's think time
for the move they actually played, and whether that move lost more than 200cp.

Positions are split three ways, because the obvious objection to SRC is that
"one safe reply" is often a recapture or a check evasion, which takes a second:

    in check        the mover must answer a check
    after capture   the opponent's last move captured something
    quiet           neither -- the narrow paths SRC is meant to find

Think time is the clock before the mover's previous move minus the clock after
this one, plus the increment: Lichess's ``%clk`` already includes it, which the
dump shows directly (clocks rise across quick moves). Resolution is a second.
Moves made in a time scramble are fast for reasons that have nothing to do
with the position, so the report repeats the key comparison without them.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import logging
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Mapping, Optional, Sequence, Tuple

import chess

from src.eval.arena import BLUNDER_CP, ERROR_CAP
from src.eval.live_analyzer import read_jsonl
from src.eval.skew_database import SPEEDS, _elo, _open_source, _sans, iter_raw_games

__all__ = ["clocked_moves", "sample_position", "spearman", "report", "main"]

logger = logging.getLogger(__name__)

DEFAULT_SOURCE: Final[str] = "build/lichess_2026-08_q1.pgn.zst"
DEFAULT_POSITIONS: Final[Path] = Path("build/clock_positions.jsonl")
DEFAULT_SCORED: Final[Path] = Path("build/clock_scored.jsonl")

SAFE_MARGIN: Final[int] = 50
SAFE_CAP: Final[int] = 8
SCAN_DEPTH: Final[int] = 6
SCRAMBLE_SECONDS: Final[int] = 30
"""Below this much clock a move is fast because the clock says so."""

BUCKETS: Final[Tuple[Tuple[str, int, int], ...]] = (
    ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-7", 5, 7), ("8+", 8, 99),
)
CLASSES: Final[Tuple[str, ...]] = ("quiet", "after capture", "in check")

_COMMENTED_MOVE = re.compile(r"([A-Za-z][^\s{}]*)\s*\{([^}]*)\}")
_CLOCK = re.compile(r"\[%clk (\d+):(\d+):(\d+)(?:\.\d+)?\]")


# -- sampling -----------------------------------------------------------------


def clocked_moves(movetext: str) -> List[Tuple[str, int]]:
    """(SAN, clock after the move in seconds) per move, or [] if any move has no clock."""
    pairs: List[Tuple[str, int]] = []
    for san, comment in _COMMENTED_MOVE.findall(movetext):
        clock = _CLOCK.search(comment)
        if clock is None:
            return []
        hours, minutes, seconds = (int(x) for x in clock.groups())
        pairs.append((san.rstrip("?!"), hours * 3600 + minutes * 60 + seconds))
    return pairs if len(pairs) == len(_sans(movetext)) else []


def sample_position(
    headers: Mapping[str, str], moves: Sequence[Tuple[str, int]], rng: random.Random,
    *, first_ply: int, last_ply: int,
) -> Optional[Dict[str, Any]]:
    """One random middlegame position from a clocked game, with its think time."""
    try:
        base, increment = (int(x) for x in headers.get("TimeControl", "").split("+"))
    except ValueError:
        return None
    last = min(last_ply, len(moves) - 1)
    if last < max(first_ply, 2):
        return None
    ply = rng.randint(max(first_ply, 2), last)

    board = chess.Board()
    previous_capture = False
    try:
        for san, _clock in moves[:ply]:
            move = board.parse_san(san)
            previous_capture = board.is_capture(move)
            board.push(move)
        played = board.parse_san(moves[ply][0])
    except ValueError:
        return None

    remaining = moves[ply - 2][1]
    mover = board.turn
    return {
        "site": headers.get("Site", ""), "fen": board.fen(), "played": played.uci(), "ply": ply,
        "think": max(0, remaining - moves[ply][1] + increment), "remaining": remaining,
        "base": base, "increment": increment,
        "speed": headers.get("Event", "").split()[1].lower(),
        "elo": _elo(headers, "WhiteElo" if mover == chess.WHITE else "BlackElo"),
        "position": "in check" if board.is_check() else "after capture" if previous_capture else "quiet",
    }


def collect(
    source: str, sink: Any, *, count: int, min_elo: int, max_elo: int,
    first_ply: int, last_ply: int, seed: int,
) -> int:
    rng = random.Random(seed)
    kept = read = 0
    for headers, movetext in iter_raw_games(_open_source(source)):
        read += 1
        if not headers.get("Event", "").startswith(SPEEDS[:2]):  # blitz and rapid
            continue
        if "BOT" in (headers.get("WhiteTitle"), headers.get("BlackTitle")):
            continue
        if headers.get("Termination") == "Abandoned":
            continue
        if not all(min_elo <= _elo(headers, key) <= max_elo for key in ("WhiteElo", "BlackElo")):
            continue
        moves = clocked_moves(movetext)
        row = sample_position(headers, moves, rng, first_ply=first_ply, last_ply=last_ply) if moves else None
        if row is None:
            continue
        sink.write(json.dumps(row) + "\n")
        kept += 1
        if kept >= count:
            break
    sink.flush()
    logger.info("collect: %s games read, %s positions kept", f"{read:,}", f"{kept:,}")
    return kept


# -- scoring ------------------------------------------------------------------


def _score_batch(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Worker: SRC and the played move's loss from one bounded scan per position."""
    from src.engine.search import _rank_mover_relative, count_safe
    from src.engine.stockfish import StockfishEvaluator

    logging.basicConfig(level=logging.WARNING)
    out: List[Dict[str, Any]] = []
    with StockfishEvaluator() as stockfish:
        for row in rows:
            board = chess.Board(str(row["fen"]))
            legal = board.legal_moves.count()
            ranked = _rank_mover_relative(
                stockfish.analyse_root_moves(board, depth=SCAN_DEPTH, multipv=min(SAFE_CAP, legal)),
                board.turn,
            )
            scores = dict(ranked)
            played = chess.Move.from_uci(str(row["played"]))
            if played in scores:
                played_score = scores[played]
            else:
                board.push(played)
                if board.is_checkmate():
                    played_score = ERROR_CAP
                elif board.is_game_over(claim_draw=True):
                    played_score = 0
                else:
                    white = stockfish.evaluate(board, depth=SCAN_DEPTH).centipawns
                    played_score = -white if board.turn == chess.WHITE else white
                board.pop()
            loss = min(ERROR_CAP, max(0, ranked[0][1] - played_score))
            out.append({**row, "src": count_safe(ranked, SAFE_MARGIN), "legal": legal,
                        "loss": loss, "blunder": loss > BLUNDER_CP})
    return out


# -- statistics ---------------------------------------------------------------


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Rank correlation with tie-averaged ranks, and its approximate standard error."""
    n = len(xs)
    if n < 4:
        return math.nan, math.nan
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    rho = sxy / math.sqrt(sxx * syy) if sxx and syy else math.nan
    return rho, 1.03 / math.sqrt(n - 3)  # Fieller et al. for Spearman


def _bucket(src: int) -> str:
    return next(name for name, low, high in BUCKETS if low <= src <= high)


def _mean_se(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) < 2:
        return (values[0] if values else math.nan), math.nan
    return statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values))


def _narrow_vs_easy(rows: Sequence[Mapping[str, Any]], label: str) -> str:
    narrow = [r for r in rows if r["src"] <= 2]
    easy = [r for r in rows if r["src"] >= 5]
    if len(narrow) < 2 or len(easy) < 2:
        return f"  {label}: too few positions"
    lines = []
    for name, key in (("think time (s)", "think"), ("blunder rate", "blunder")):
        a, a_se = _mean_se([float(r[key]) for r in easy])
        b, b_se = _mean_se([float(r[key]) for r in narrow])
        sigma = (b - a) / math.hypot(a_se, b_se)
        lines.append(f"{name} {a:.3g} -> {b:.3g} ({b - a:+.3g}, {sigma:+.1f} sigma)")
    return f"  {label} (n={len(easy)} easy, {len(narrow)} narrow): " + "; ".join(lines)


def report(rows: Sequence[Mapping[str, Any]]) -> str:
    out: List[str] = []
    for speed in ("blitz", "rapid"):
        at_speed = [r for r in rows if r["speed"] == speed]
        if not at_speed:
            continue
        out.append(f"\n{speed.upper()}  ({len(at_speed):,} positions)")
        out.append(f"  {'position':<15}{'SRC':>5}{'n':>7}{'median think':>14}"
                   f"{'mean think':>16}{'blunder %':>16}")
        for kind in CLASSES:
            group = [r for r in at_speed if r["position"] == kind]
            cells: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
            for r in group:
                cells[_bucket(int(r["src"]))].append(r)
            for name, _low, _high in BUCKETS:
                cell = cells.get(name, [])
                if not cell:
                    continue
                think, think_se = _mean_se([float(r["think"]) for r in cell])
                blunder, blunder_se = _mean_se([float(r["blunder"]) for r in cell])
                out.append(
                    f"  {kind:<15}{name:>5}{len(cell):>7}"
                    f"{statistics.median(float(r['think']) for r in cell):>13.0f}s"
                    f"{think:>10.1f} ±{think_se:<4.1f}{100 * blunder:>10.1f} ±{100 * blunder_se:<4.1f}"
                )
            rho_t, se_t = spearman([float(r["src"]) for r in group], [float(r["think"]) for r in group])
            rho_b, se_b = spearman([float(r["src"]) for r in group], [float(r["blunder"]) for r in group])
            out.append(f"  {kind:<15}rank correlation with SRC: think {rho_t:+.3f} ±{se_t:.3f}, "
                       f"blunder {rho_b:+.3f} ±{se_b:.3f}")
        quiet = [r for r in at_speed if r["position"] == "quiet"]
        out.append("  narrow (SRC <= 2) against easy (SRC >= 5):")
        out.append(_narrow_vs_easy(quiet, "quiet"))
        out.append(_narrow_vs_easy([r for r in quiet if r["remaining"] >= SCRAMBLE_SECONDS],
                                   f"quiet, clock >= {SCRAMBLE_SECONDS}s"))
        out.append(_narrow_vs_easy([r for r in at_speed if r["position"] != "quiet"], "forcing"))
    return "\n".join(out)


# -- command line -------------------------------------------------------------


def _chunks(rows: Sequence[Dict[str, Any]], size: int) -> Iterator[Sequence[Dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.eval.clock_analysis", description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--positions", type=Path, default=DEFAULT_POSITIONS)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--count", type=int, default=30_000, help="Positions, one per game.")
    parser.add_argument("--min-elo", type=int, default=1000)
    parser.add_argument("--max-elo", type=int, default=2000)
    parser.add_argument("--first-ply", type=int, default=15)
    parser.add_argument("--last-ply", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if not args.report_only:
        if not args.positions.exists():
            with args.positions.open("w", encoding="utf-8") as sink:
                collect(args.source, sink, count=args.count, min_elo=args.min_elo, max_elo=args.max_elo,
                        first_ply=args.first_ply, last_ply=args.last_ply, seed=0)
        done = {str(r["site"]) for r in read_jsonl(args.scored)}
        pending = [r for r in read_jsonl(args.positions) if str(r["site"]) not in done]
        logger.info("score: %d to scan, %d already done", len(pending), len(done))
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool, \
                args.scored.open("a", encoding="utf-8") as sink:
            jobs = [pool.submit(_score_batch, list(batch)) for batch in _chunks(pending, 200)]
            for finished, job in enumerate(futures.as_completed(jobs), 1):
                for row in job.result():
                    sink.write(json.dumps(row) + "\n")
                sink.flush()
                if finished % 10 == 0 or finished == len(jobs):
                    logger.info("score: %d/%d batches", finished, len(jobs))

    print(report(list(read_jsonl(args.scored))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
