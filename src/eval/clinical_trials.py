"""Do the psychological levers work on people? Three hypotheses, human games only.

A. **Cadence.** Does how long the bot appears to think change how long the
   human then thinks, and how often they blunder? The cadence is chosen by the
   position, so comparing cadences would compare positions; instead the bridge
   withholds each wait on a coin flip, and the comparison is paced against
   unpaced replies *within* a cadence. Needs the per-move log, so only games
   played since that logging began count.
B. **Narrow paths.** In quiet positions (not in check, the bot's last move not
   a capture), do humans facing two or fewer safe replies think longer and
   blunder more than with five or more, as they did in the August 2026 Lichess
   games? And are they put there more often than in human-vs-human play?
   Recomputed from the game itself, so every human game counts.
C. **The trap floor.** Games where a bot move took its own evaluation from above
   -2.50 into -2.50 .. -4.50: do they end in wins, or just in losses?

Every game is analysed once at depth 10 and cached, and the export is one
request per run. Errors cluster by opponent, so a regular is not counted as
many independent people; a group with fewer than two opponents shows n/a.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import chess

from src.eval.live_analyzer import (
    NOT_PLAYED, UNFINISHED, _opponent_key, _ratio, _se, _sigma, read_jsonl,
)

__all__ = ["analyse_game", "dashboard", "main"]

logger = logging.getLogger(__name__)

ACCOUNT: Final[str] = "latentblunder"
DEPTH: Final[int] = 10
SAFE_MARGIN: Final[int] = 50
SAFE_CAP: Final[int] = 8
BLUNDER_CP: Final[int] = 200
CAP: Final[int] = 1000
TRAP_ZONE: Final[Tuple[int, int]] = (-450, -250)
MIN_OPPONENTS: Final[int] = 5
"""Below this many opponents in a group its error bar is shown as n/a. A
clustered standard error from two or three people is not an estimate: the
first run printed +5.1 sigma from three opponents."""

WINDOW: Final[Tuple[int, int]] = (15, 40)
"""Plies compared against the August baseline, which sampled the same window."""

DEFAULT_CACHE: Final[Path] = Path("build/clinical_games.jsonl")
DEFAULT_LOG: Final[Path] = Path("build/live_games.jsonl")
DEFAULT_BASELINE: Final[Path] = Path("build/clock_scored.jsonl")

Scan = Callable[[chess.Board, int], List[Tuple[chess.Move, int]]]
"""Best-first (move, centipawns) for the side to move, ``multipv`` lines."""

Evaluate = Callable[[chess.Board], int]
"""Centipawns for the side to move."""


def _clamp(cp: int) -> int:
    return max(-CAP, min(CAP, cp))


def analyse_game(game: Mapping[str, Any], scan: Scan, evaluate: Evaluate) -> Dict[str, Any]:
    """Every ply from the mover's side: loss, think time, and for the human's
    positions the safe reply count and whether the position was quiet."""
    white = game["players"]["white"]
    bot_colour = chess.WHITE if white.get("user", {}).get("id") == ACCOUNT else chess.BLACK
    winner = game.get("winner")
    clocks = [c / 100 for c in game.get("clocks", [])]
    clock = game.get("clock") or {}
    initial, increment = float(clock.get("initial", 0)), float(clock.get("increment", 0))
    last = {chess.WHITE: initial, chess.BLACK: initial}

    board = chess.Board(game["initialFen"]) if game.get("initialFen") else chess.Board()
    plies: List[Dict[str, Any]] = []
    previous_capture = False
    for ply, san in enumerate(str(game.get("moves", "")).split()):
        mover = board.turn
        human = mover != bot_colour
        ranked = scan(board, min(SAFE_CAP, board.legal_moves.count()) if human else 1)
        best = _clamp(ranked[0][1])
        scores = dict(ranked)
        move = board.parse_san(san)
        if move in scores:
            played = _clamp(scores[move])
        else:
            board.push(move)
            played = CAP if board.is_checkmate() else 0 if board.is_game_over() else -_clamp(evaluate(board))
            board.pop()
        think = None
        if ply < len(clocks):
            think = round(last[mover] - clocks[ply] + increment, 2)
            last[mover] = clocks[ply]
        loss = max(0, best - played)
        plies.append({
            "ply": ply, "side": "human" if human else "bot", "think": think, "loss": loss,
            "blunder": loss > BLUNDER_CP,
            "safe_replies": sum(1 for _m, cp in ranked if best - _clamp(cp) <= SAFE_MARGIN) if human else None,
            "quiet": human and not board.is_check() and not previous_capture,
            "bot_before": None if human else best, "bot_after": None if human else played,
        })
        previous_capture = board.is_capture(move)
        board.push(move)

    return {
        "id": game["id"], "opponent": _opponent_key(game, bot_colour),
        "result": "win" if winner == chess.COLOR_NAMES[bot_colour] else "loss" if winner else "draw",
        "speed": game.get("speed"), "plies": plies,
    }


# -- dashboard ----------------------------------------------------------------


def _cell(rows: Sequence[Mapping[str, Any]], key: str) -> Tuple[float, float]:
    """Mean of ``key`` over moves with a value, clustered by opponent; NaN where
    there is nothing to average, and a NaN error below ``MIN_OPPONENTS``."""
    kept = [r for r in rows if r.get(key) is not None]
    if not kept:
        return math.nan, math.nan
    mean, se = _ratio([float(r[key]) for r in kept], [1.0] * len(kept), [str(r["opponent"]) for r in kept])
    return mean, se if len({str(r["opponent"]) for r in kept}) >= MIN_OPPONENTS else math.nan


def _fmt(x: float) -> str:
    return "  -" if math.isnan(x) else f"{x:4.1f}"


def _compare(label: str, a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> str:
    think_a, think_a_se = _cell(a, "think")
    think_b, think_b_se = _cell(b, "think")
    blunder_a, blunder_a_se = _cell(a, "blunder")
    blunder_b, blunder_b_se = _cell(b, "blunder")
    return (f"  {label:<20} n {len(a):>4} / {len(b):<4}"
            f"  think {_fmt(think_a)}s {_se(think_a_se)} vs {_fmt(think_b)}s {_se(think_b_se)}"
            f" ({_sigma(think_a, think_a_se, think_b, think_b_se)} sigma)"
            f"  blunders {_fmt(100 * blunder_a)}% vs {_fmt(100 * blunder_b)}%"
            f" ({_sigma(blunder_a, blunder_a_se, blunder_b, blunder_b_se)} sigma)")


def dashboard(
    games: Sequence[Mapping[str, Any]], telemetry: Mapping[Tuple[str, int], Mapping[str, Any]],
    baseline_narrow_share: Optional[float] = None,
) -> str:
    if not games:
        return "no finished human games yet"
    opponents = {str(g["opponent"]) for g in games}
    with_log = {gid for gid, _ in telemetry}
    out = [f"CLINICAL TRIAL: {len(games)} human games, {len(opponents)} opponents, "
           f"{sum(1 for g in games if g['id'] in with_log)} with move telemetry"]

    # A: human replies after a bot move, by its cadence and whether the wait was applied.
    replies: Dict[Tuple[str, bool], List[Dict[str, Any]]] = defaultdict(list)
    for game in games:
        for ply in game["plies"]:
            move = telemetry.get((game["id"], ply["ply"] - 1))
            if ply["side"] == "human" and move is not None:
                replies[(str(move["cadence"]), bool(move["paced"]))].append({**ply, "opponent": game["opponent"]})
    out.append("\nA. CADENCE: the human's reply, after the bot's wait was applied vs withheld (coin flip)")
    for cadence in ("snap", "bait", "deliberate"):
        out.append(_compare(cadence, replies[(cadence, True)], replies[(cadence, False)]))

    # B: quiet positions the human faced.
    quiet = [{**p, "opponent": g["opponent"]} for g in games for p in g["plies"]
             if p["side"] == "human" and p["quiet"]]
    narrow = [p for p in quiet if p["safe_replies"] <= 2]
    broad = [p for p in quiet if p["safe_replies"] >= 5]
    out.append("\nB. NARROW PATHS: quiet positions the human faced, broad (5+ safe) vs narrow (<=2)")
    out.append(_compare("broad vs narrow", broad, narrow))
    windowed = [p for p in quiet if WINDOW[0] <= p["ply"] <= WINDOW[1]]
    if windowed:
        share = sum(1 for p in windowed if p["safe_replies"] <= 2) / len(windowed)
        reference = (f"; human-vs-human, August 2026: {100 * baseline_narrow_share:.0f}%"
                     if baseline_narrow_share is not None else "")
        out.append(f"  narrow share of quiet positions, plies {WINDOW[0]}-{WINDOW[1]}: "
                   f"{100 * share:.0f}% of {len(windowed)}{reference}")

    # C: games where a bot move dropped its own evaluation into the trap zone.
    low, high = TRAP_ZONE
    gambled = {g["id"] for g in games for p in g["plies"] if p["side"] == "bot"
               and p["bot_before"] >= high and low <= p["bot_after"] < high}
    out.append(f"\nC. TRAP FLOOR: games where a bot move went from above {high / 100:+.2f} "
               f"into {low / 100:+.2f}..{high / 100:+.2f}")
    for label, group in (("those games", [g for g in games if g["id"] in gambled]),
                         ("all other games", [g for g in games if g["id"] not in gambled])):
        tally = {r: sum(1 for g in group if g["result"] == r) for r in ("win", "draw", "loss")}
        out.append(f"  {label:<16} n {len(group):>3}: {tally['win']} won, {tally['draw']} drawn, "
                   f"{tally['loss']} lost")
    return "\n".join(out)


# -- command line -------------------------------------------------------------


def _baseline_share(path: Path) -> Optional[float]:
    rows = [r for r in read_jsonl(path) if r.get("position") == "quiet"]
    return sum(1 for r in rows if int(r["src"]) <= 2) / len(rows) if rows else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.eval.clinical_trials", description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--report-only", action="store_true", help="Use the cache; no API call.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cached = {str(row["id"]): row for row in read_jsonl(args.cache)}
    if not args.report_only:
        import berserk

        from src import config
        from src.engine.search import _rank_mover_relative
        from src.engine.stockfish import StockfishEvaluator

        client = berserk.Client(session=berserk.TokenSession(config.lichess_token()))
        exported: List[Dict[str, Any]] = [
            dict(g) for g in client.games.export_by_player(ACCOUNT, as_pgn=False, moves=True, clocks=True)
            if isinstance(g, dict)
        ]
        pending = [
            g for g in exported
            if str(g.get("id")) not in cached and g.get("moves")
            and str(g.get("status")) not in UNFINISHED | NOT_PLAYED
            and "BOT" not in {str(p.get("user", {}).get("title", "")) for p in g["players"].values()
                              if p.get("user", {}).get("id") != ACCOUNT}
        ]
        logger.info("clinical: %d games exported, %d human games new", len(exported), len(pending))
        if pending:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            with StockfishEvaluator() as stockfish, args.cache.open("a", encoding="utf-8") as sink:
                def scan(board: chess.Board, lines: int) -> List[Tuple[chess.Move, int]]:
                    return _rank_mover_relative(
                        stockfish.analyse_root_moves(board, depth=DEPTH, multipv=lines), board.turn)

                def evaluate(board: chess.Board) -> int:
                    cp = stockfish.evaluate(board, depth=DEPTH).centipawns
                    return cp if board.turn == chess.WHITE else -cp

                for game in pending:
                    row = analyse_game(game, scan, evaluate)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                    cached[row["id"]] = row

    telemetry = {(str(r["game"]), int(r["ply"])): r for r in read_jsonl(args.log) if r.get("event") == "move"}
    print(dashboard(list(cached.values()), telemetry, _baseline_share(args.baseline)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
