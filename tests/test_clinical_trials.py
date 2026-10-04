"""Clinical dashboard: per-ply bookkeeping and honest small-sample output."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import chess

from src.eval.clinical_trials import analyse_game, dashboard

GAME = {
    "id": "g1", "winner": "white", "speed": "blitz", "status": "mate",
    "players": {"white": {"user": {"id": "latentblunder"}}, "black": {"user": {"id": "someone"}}},
    "moves": "e4 e5 Qh5 Nc6 Bc4 Nf6 Qxf7#",
    "clocks": [18000, 18000, 17950, 17700, 17900, 17000, 17800], "clock": {"initial": 180, "increment": 2},
}


def _scan(board: chess.Board, lines: int) -> List[Tuple[chess.Move, int]]:
    # Everything level, except that after 3.Bc4 only ...g6/...Qe7 defend f7.
    fen = board.board_fen()
    good = {"g7g6", "d8e7"} if fen == "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR" else None
    ranked = [(m, 0 if good is None or m.uci() in good else -900) for m in board.legal_moves]
    return sorted(ranked, key=lambda x: -x[1])[:lines]


def test_the_human_side_is_scored_with_its_safe_replies_and_think_time() -> None:
    row = analyse_game(GAME, _scan, lambda b: 0)
    human = [p for p in row["plies"] if p["side"] == "human"]
    nf6 = human[2]
    assert nf6["safe_replies"] == 2 and nf6["blunder"] and nf6["quiet"]
    assert nf6["think"] == 177.0 - 170.0 + 2, "clock before its previous move, minus this one, plus increment"
    assert row["result"] == "win" and row["opponent"]


def test_too_few_opponents_reads_n_a_not_a_number() -> None:
    row = analyse_game(GAME, _scan, lambda b: 0)
    telemetry: Dict[Tuple[str, int], Dict[str, Any]] = {("g1", 4): {"cadence": "deliberate", "paced": True}}
    text = dashboard([row], telemetry)
    assert "1 human games, 1 opponents, 1 with move telemetry" in text
    assert "n/a sigma" in text and "C. TRAP FLOOR" in text
    assert dashboard([], {}) == "no finished human games yet"


def test_a_handful_of_opponents_gets_no_sigma_however_big_the_gap() -> None:
    """The first live run printed +5.1 sigma from three opponents."""
    from src.eval.clinical_trials import _compare

    def moves(opponents: int, blunder: bool) -> list[dict[str, object]]:
        # Rates vary by opponent, so the clustered error is defined.
        return [{"opponent": f"o{i}", "think": 5.0 + i, "blunder": (k < i + 2) == blunder}
                for i in range(opponents) for k in range(8)]

    assert _compare("x", moves(3, False), moves(3, True)).count("n/a sigma") == 2
    assert "n/a sigma" not in _compare("x", moves(6, False), moves(6, True)).split("blunders")[1]
    assert "  -" in _compare("x", [], moves(6, True)), "an empty group shows a dash, not 0.0"
