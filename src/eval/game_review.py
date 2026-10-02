"""Review every game the account has played, move by move, with a deep engine.

For each move of both sides: the evaluation before and after at depth 16, the
centipawns lost against the engine's best, what the best move was, whether the
move came from an opening book, the think time and the clock left. Per game:
where the bot's advantage became decisive and how long it took to cash it,
forced mates it had and did not take, and how the clock was spent. The output
is a markdown report with one section per game and the per-move table, so a
mistake can be found and replayed, plus a JSON file of every number.

Clock values in the Lichess export already include the increment, as the
``%clk`` tags do, so think time is the previous own clock minus this one plus
the increment.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import chess
import chess.polyglot

from src import config

__all__ = ["review_game", "classify", "main"]

logger = logging.getLogger(__name__)

ACCOUNT: Final[str] = "latentblunder"
DEPTH: Final[int] = 16
CAP: Final[int] = 1000
"""Evaluations are clamped here for loss arithmetic, so one mate cannot dominate a mean."""

DECISIVE_CP: Final[int] = 300
BOOKS: Final[Tuple[Tuple[str, Path], ...]] = (
    ("repertoire", config.BOOKS_DIR / "repertoire.bin"),
    ("traps", config.trap_book_path()),
    ("standard", config.standard_book_path()),
)


def classify(loss: int) -> str:
    """Lichess's thresholds, in centipawns lost."""
    if loss >= 300:
        return "blunder"
    if loss >= 100:
        return "mistake"
    if loss >= 50:
        return "inaccuracy"
    return ""


def _clamp(cp: int) -> int:
    return max(-CAP, min(CAP, cp))


def review_game(game: Mapping[str, Any], depth: int = DEPTH) -> Dict[str, Any]:
    """Every ply scored from the mover's side. Runs in a worker with its own Stockfish."""
    from src.engine.search import _rank_mover_relative
    from src.engine.stockfish import StockfishEvaluator

    logging.basicConfig(level=logging.WARNING)
    white = game["players"]["white"]
    bot_colour = chess.WHITE if white.get("user", {}).get("id") == ACCOUNT else chess.BLACK
    opponent = game["players"]["black" if bot_colour == chess.WHITE else "white"]
    sans = str(game.get("moves", "")).split()
    clocks = [c / 100 for c in game.get("clocks", [])]
    clock = game.get("clock") or {}
    initial, increment = float(clock.get("initial", 0)), float(clock.get("increment", 0))

    readers = []
    for name, path in BOOKS:
        try:
            readers.append((name, chess.polyglot.open_reader(path)))
        except OSError:
            continue

    board = chess.Board(game["initialFen"]) if game.get("initialFen") else chess.Board()
    plies: List[Dict[str, Any]] = []
    with StockfishEvaluator() as stockfish:
        def score(b: chess.Board) -> Tuple[int, Optional[int], Optional[chess.Move]]:
            """Side-to-move score, mate distance for the side to move (if any), best move."""
            if b.is_checkmate():
                return -CAP, None, None
            if b.is_game_over():
                return 0, None, None
            ranked = stockfish.analyse_root_moves(b, depth=depth, multipv=1)
            best, evaluation = next(iter(ranked.items()))
            cp = evaluation.centipawns if b.turn == chess.WHITE else -evaluation.centipawns
            mate = None
            if evaluation.is_mate and evaluation.mate_in is not None:
                mate = evaluation.mate_in if b.turn == chess.WHITE else -evaluation.mate_in
            return _clamp(cp), mate, best

        before = score(board)
        last_own = {chess.WHITE: initial, chess.BLACK: initial}
        for ply, san in enumerate(sans):
            mover = board.turn
            move = board.parse_san(san)
            book = next((name for name, reader in readers
                         if any(e.move == move for e in reader.find_all(board))), "")
            best_san = board.san(before[2]) if before[2] is not None else ""
            board.push(move)
            after = score(board)
            think = None
            left = clocks[ply] if ply < len(clocks) else None
            if left is not None:
                think = round(last_own[mover] - left + increment, 1)
                last_own[mover] = left
            loss = max(0, before[0] + after[0])
            plies.append({
                "ply": ply, "side": "bot" if mover == bot_colour else "opp", "san": san,
                "best": best_san, "eval_before": before[0] if mover == bot_colour else -before[0],
                "eval_after": -after[0] if mover == bot_colour else after[0],
                "loss": loss, "class": classify(loss), "book": book,
                "think": think, "clock": left,
                "had_mate": before[1] is not None and before[1] > 0,
                # Delivering mate keeps it: a mated position reports no distance.
                "kept_mate": board.is_checkmate() or (after[1] is not None and after[1] < 0),
            })
            before = after
    for _name, reader in readers:
        reader.close()

    # eval_before/eval_after above are bot-relative; the trajectory is too.
    trajectory = [p["eval_after"] for p in plies]
    decisive: Optional[int] = None
    for ply, value in enumerate(trajectory):
        if value >= DECISIVE_CP:
            decisive = ply if decisive is None else decisive
        else:
            decisive = None
    winner = game.get("winner")
    return {
        "id": game["id"], "speed": game.get("speed"), "rated": game.get("rated"),
        "clock": f"{int(initial // 60)}+{int(increment)}", "colour": chess.COLOR_NAMES[bot_colour],
        "opponent": opponent.get("user", {}).get("name", "?"),
        "opponent_title": opponent.get("user", {}).get("title", ""),
        "opponent_rating": opponent.get("rating"), "status": game.get("status"),
        "result": "win" if winner == chess.COLOR_NAMES[bot_colour] else "loss" if winner else "draw",
        "opening": (game.get("opening") or {}).get("name", ""), "plies": plies,
        "decisive_ply": decisive, "created": game.get("createdAt"),
    }


# -- report -------------------------------------------------------------------


def _side(review: Mapping[str, Any], side: str) -> List[Mapping[str, Any]]:
    return [p for p in review["plies"] if p["side"] == side]


def _summary(review: Mapping[str, Any]) -> Dict[str, Any]:
    bot, opp = _side(review, "bot"), _side(review, "opp")
    thinks = [p["think"] for p in bot if p["think"] is not None]
    clocks = [p["clock"] for p in bot if p["clock"] is not None]

    def count(moves: Sequence[Mapping[str, Any]], kind: str) -> int:
        return sum(1 for p in moves if p["class"] == kind)
    return {
        "bot_acpl": statistics.fmean(p["loss"] for p in bot) if bot else 0.0,
        "opp_acpl": statistics.fmean(p["loss"] for p in opp) if opp else 0.0,
        "bot_errors": (count(bot, "inaccuracy"), count(bot, "mistake"), count(bot, "blunder")),
        "opp_errors": (count(opp, "inaccuracy"), count(opp, "mistake"), count(opp, "blunder")),
        "book_moves": sum(1 for p in bot if p["book"]),
        "missed_mates": sum(1 for p in bot if p["had_mate"] and not p["kept_mate"]),
        "think_mean": statistics.fmean(thinks) if thinks else 0.0,
        "think_max": max(thinks) if thinks else 0.0,
        "clock_min": min(clocks) if clocks else 0.0,
        "plies_to_finish": (len(review["plies"]) - review["decisive_ply"]
                            if review["decisive_ply"] is not None else None),
    }


def _think(ply: Mapping[str, Any]) -> str:
    return "" if ply["think"] is None else f"{ply['think']:.1f}s"


def report(reviews: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Game review", ""]
    lines.append("| game | clock | colour | opponent | result | bot ACPL | bot inacc/mist/blund | "
                 "opp ACPL | opp blund | missed mates | think mean/max (s) | min clock (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in reviews:
        s = _summary(r)
        opp = f"{r['opponent_title'] + ' ' if r['opponent_title'] else ''}{r['opponent']} ({r['opponent_rating']})"
        lines.append(
            f"| [{r['id']}](https://lichess.org/{r['id']}) | {r['clock']} | {r['colour']} | {opp} | "
            f"{r['result']} ({r['status']}) | {s['bot_acpl']:.0f} | {'/'.join(map(str, s['bot_errors']))} | "
            f"{s['opp_acpl']:.0f} | {s['opp_errors'][2]} | {s['missed_mates']} | "
            f"{s['think_mean']:.1f}/{s['think_max']:.1f} | {s['clock_min']:.0f} |"
        )
    for r in reviews:
        s = _summary(r)
        lines += ["", f"## {r['id']}: {r['colour']} vs {r['opponent']} ({r['opponent_rating']}), "
                      f"{r['clock']} {r['speed']}, {r['result']} by {r['status']}", ""]
        lines.append(f"- opening: {r['opening'] or '?'}; bot book moves: {s['book_moves']}")
        if r["decisive_ply"] is not None:
            lines.append(f"- decisive (+{DECISIVE_CP}cp, held) after ply {r['decisive_ply'] + 1}; "
                         f"{s['plies_to_finish']} plies from there to the end")
        lines.append(f"- bot: ACPL {s['bot_acpl']:.0f}, inaccuracies/mistakes/blunders "
                     f"{'/'.join(map(str, s['bot_errors']))}, missed forced mates {s['missed_mates']}")
        lines.append(f"- opponent: ACPL {s['opp_acpl']:.0f}, inaccuracies/mistakes/blunders "
                     f"{'/'.join(map(str, s['opp_errors']))}")
        lines.append(f"- clock: mean think {s['think_mean']:.1f}s, longest {s['think_max']:.1f}s, "
                     f"lowest clock {s['clock_min']:.0f}s")
        notable = [p for p in r["plies"] if p["class"] in ("mistake", "blunder")
                   or (p["side"] == "bot" and p["had_mate"] and not p["kept_mate"])]
        if notable:
            lines.append("")
            lines.append("| move | side | played | best | eval before → after (bot) | lost | note | think |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for p in notable:
                number = f"{p['ply'] // 2 + 1}{'.' if p['ply'] % 2 == 0 else '...'}"
                note = p["class"] + (" · missed forced mate" if p["side"] == "bot" and p["had_mate"]
                                     and not p["kept_mate"] else "")
                lines.append(f"| {number} | {p['side']} | {p['san']} | {p['best']} | "
                             f"{p['eval_before']:+d} → {p['eval_after']:+d} | {p['loss']} | {note} | "
                             f"{_think(p)} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.eval.game_review", description=__doc__.split("\n")[0])
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", type=Path, default=Path("build/game_review.json"))
    parser.add_argument("--report", type=Path, default=Path("build/game_review.md"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    import berserk

    client = berserk.Client(session=berserk.TokenSession(config.lichess_token()))
    exported: List[Dict[str, Any]] = list(client.games.export_by_player(  # type: ignore[arg-type]
        ACCOUNT, as_pgn=False, moves=True, clocks=True, opening=True))
    games = [g for g in exported if g.get("moves")]
    games.sort(key=lambda g: str(g.get("createdAt")))
    logger.info("review: %d games at depth %d", len(games), args.depth)

    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        reviews = list(pool.map(review_game, games, [args.depth] * len(games)))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(reviews, default=str))
    args.report.write_text(report(reviews))
    logger.info("review: wrote %s and %s", args.report, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
