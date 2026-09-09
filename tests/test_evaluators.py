"""Smoke tests for the dual evaluation subsystem.

Runs under pytest, or standalone with no test dependencies:

    python -m tests.test_evaluators

Both evaluators spawn real engine processes, so these are integration tests:
they need the ``stockfish`` and ``lc0`` binaries and the Maia weights present.
"""

from __future__ import annotations

import math

import chess

from src.engine.maia import MaiaEvaluator, temperature_softmax
from src.engine.stockfish import StockfishEvaluator
from src.types import EngineEval, MoveDistribution, win_probability

SUM_TOLERANCE = 1e-5

STARTING_FEN = chess.STARTING_FEN
# 1.e4 e5 2.Bc4 Nc6 3.Qh5 -- Black to move, Qxf7# threatened. Maia at 1100
# should spread real mass onto moves that walk into it.
SCHOLARS_THREAT_FEN = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"
# White is a full queen down and it is *Black* to move: the raw UCI score is
# strongly positive, so this catches a missing side-to-move flip.
WHITE_QUEENLESS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"


def _assert_valid_eval(evaluation: EngineEval) -> None:
    assert isinstance(evaluation, EngineEval)
    assert isinstance(evaluation.centipawns, int)
    assert 0.0 <= evaluation.win_probability <= 1.0
    assert evaluation.is_mate == (evaluation.mate_in is not None)
    if not evaluation.is_mate:
        assert math.isclose(
            evaluation.win_probability, win_probability(evaluation.centipawns), rel_tol=1e-9
        )


def _assert_valid_distribution(distribution: MoveDistribution, board: chess.Board) -> None:
    assert isinstance(distribution, MoveDistribution)
    legal = set(board.legal_moves)
    assert set(distribution.probabilities) <= legal, "distribution contains illegal moves"
    assert len(distribution) > 0
    total = math.fsum(distribution.probabilities.values())
    assert abs(total - 1.0) <= SUM_TOLERANCE, f"probabilities sum to {total}"
    assert all(0.0 <= p <= 1.0 for p in distribution.probabilities.values())


def test_win_probability_sigmoid() -> None:
    assert math.isclose(win_probability(0), 0.5)
    assert math.isclose(win_probability(400), 10 / 11)
    assert win_probability(-400) < 0.5 < win_probability(400)


def test_temperature_softmax_is_identity_at_one() -> None:
    moves = [chess.Move.from_uci(uci) for uci in ("e2e4", "d2d4", "g1f3")]
    priors = dict(zip(moves, (0.5, 0.3, 0.2)))
    result = temperature_softmax(priors, 1.0)
    for move, prior in priors.items():
        assert math.isclose(result[move], prior, abs_tol=1e-12)

    flat = temperature_softmax(priors, 50.0)
    assert max(flat.values()) - min(flat.values()) < max(priors.values()) - min(priors.values())
    assert math.isclose(math.fsum(flat.values()), 1.0, abs_tol=SUM_TOLERANCE)


def test_move_distribution_rejects_unnormalised() -> None:
    bad = {chess.Move.from_uci("e2e4"): 0.5, chess.Move.from_uci("d2d4"): 0.2}
    try:
        MoveDistribution(bad)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("MoveDistribution accepted probabilities summing to 0.7")


def test_stockfish_evaluates_positions() -> None:
    with StockfishEvaluator() as stockfish:
        opening = stockfish.evaluate(chess.Board(STARTING_FEN), depth=12)
        _assert_valid_eval(opening)
        assert not opening.is_mate
        assert abs(opening.centipawns) < 200, f"start position should be near equal: {opening}"

        threat = stockfish.evaluate(chess.Board(SCHOLARS_THREAT_FEN), depth=12)
        _assert_valid_eval(threat)

        # White-positive convention: Black to move, White a queen down.
        queenless = stockfish.evaluate(chess.Board(WHITE_QUEENLESS_FEN), depth=12)
        _assert_valid_eval(queenless)
        assert queenless.centipawns < -500, f"expected a big minus for White, got {queenless}"
        assert queenless.win_probability < 0.2

        # Black blunders into Scholar's Mate: forced mate for White.
        mated = chess.Board(SCHOLARS_THREAT_FEN)
        mated.push_san("Nf6")
        mate_eval = stockfish.evaluate(mated, depth=12)
        _assert_valid_eval(mate_eval)
        assert mate_eval.is_mate and mate_eval.mate_in is not None and mate_eval.mate_in > 0
        assert mate_eval.win_probability == 1.0


def test_maia_returns_legal_normalised_distribution() -> None:
    with MaiaEvaluator(rating=1100) as maia:
        board = chess.Board(STARTING_FEN)
        opening = maia.predict_move_probabilities(board)
        _assert_valid_distribution(opening, board)
        # A real policy is not uniform; this catches reading degenerate scores.
        assert max(opening.probabilities.values()) > 2.0 / len(list(board.legal_moves))

        threat_board = chess.Board(SCHOLARS_THREAT_FEN)
        threat = maia.predict_move_probabilities(threat_board)
        _assert_valid_distribution(threat, threat_board)
        assert threat.most_likely in set(threat_board.legal_moves)

        flattened = maia.predict_move_probabilities(threat_board, temperature=5.0)
        _assert_valid_distribution(flattened, threat_board)
        assert max(flattened.probabilities.values()) < max(threat.probabilities.values())


def _main() -> int:
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - standalone runner reports everything
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print("all green" if not failures else f"{failures} failing test(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
