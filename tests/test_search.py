"""Tests for the opponent-bounded adversarial expectimax search.

Decision-rule tests drive the searcher with *scripted* evaluators: real engines
are non-deterministic across versions and hardware, and the point of these tests
is the arbitration logic, not Stockfish's opinion. Integration and performance
tests use the real engines.

Runs under pytest, or standalone with no test dependencies:

    python -m tests.test_search
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Mapping, Optional

import chess

from src.config import MATE_SCORE_CP
from src.engine.cache import EvalCache, position_key
from src.engine.maia import MaiaEvaluator
from src.engine.search import AdversarialSearcher, TerminalPositionError, truncate_distribution
from src.engine.stockfish import StockfishEvaluator
from src.types import EngineEval, MoveDistribution, SearchConfig, win_probability

# A 2-ply search must fit comfortably inside a blitz move budget. Only the upper
# bound is asserted: a lower bound would fail the build for being fast, and the
# 1.5s figure in the brief is a target, not a requirement.
MAX_SEARCH_SECONDS = 3.0

ITALIAN_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
MIDDLEGAME_FEN = "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
# Black to move, Qxf7# threatened.
SCHOLARS_THREAT_FEN = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"


def _eval(centipawns: int) -> EngineEval:
    return EngineEval(centipawns, False, None, win_probability(centipawns))


def _child_fen(board: chess.Board, *ucis: str) -> str:
    child = board.copy()
    for uci in ucis:
        child.push(chess.Move.from_uci(uci))
    return child.fen()


class ScriptedEvaluator:
    """Stand-in for ``StockfishEvaluator`` with hand-written scores.

    Scores are White-relative, matching the real contract. Root moves not named
    in ``root_scores`` and positions not named in ``leaf_scores`` fall back to
    ``default_cp``, which is low enough that they never survive root filtering.
    """

    def __init__(
        self,
        root_scores: Mapping[str, int],
        leaf_scores: Mapping[str, int],
        *,
        default_cp: int = -900,
    ) -> None:
        self.root_scores = dict(root_scores)
        self.leaf_scores = dict(leaf_scores)
        self.default_cp = default_cp
        self.evaluate_calls = 0
        self.root_calls = 0

    def evaluate(self, board: chess.Board, depth: int = 12) -> EngineEval:
        self.evaluate_calls += 1
        return _eval(self.leaf_scores.get(board.fen(), self.default_cp))

    def analyse_root_moves(
        self, board: chess.Board, *, depth: int = 12, multipv: Optional[int] = None
    ) -> Dict[chess.Move, EngineEval]:
        self.root_calls += 1
        return {
            move: _eval(self.root_scores.get(move.uci(), self.default_cp))
            for move in board.legal_moves
        }


class ScriptedHumanModel:
    """Stand-in for ``MaiaEvaluator``; uniform over legal moves when unscripted."""

    def __init__(self, replies: Mapping[str, Mapping[str, float]]) -> None:
        self.replies = {fen: dict(moves) for fen, moves in replies.items()}
        self.calls = 0

    def predict_move_probabilities(
        self, board: chess.Board, *, temperature: Optional[float] = None
    ) -> MoveDistribution:
        self.calls += 1
        scripted = self.replies.get(board.fen())
        if scripted is None:
            legal = list(board.legal_moves)
            return MoveDistribution({move: 1.0 / len(legal) for move in legal})
        return MoveDistribution(
            {chess.Move.from_uci(uci): probability for uci, probability in scripted.items()}
        )


# --- pure helpers ----------------------------------------------------------


def test_truncate_distribution_respects_all_three_limits() -> None:
    moves = [chess.Move.from_uci(uci) for uci in ("e2e4", "d2d4", "g1f3", "b1c3", "a2a3")]
    distribution = MoveDistribution(dict(zip(moves, (0.50, 0.30, 0.15, 0.04, 0.01))))

    # target_mass: 0.50 + 0.30 + 0.15 = 0.95 crosses 0.92, so it stops there.
    by_mass = truncate_distribution(
        distribution, min_probability=0.0, target_mass=0.92, max_replies=99
    )
    assert set(by_mass.probabilities) == set(moves[:3])
    assert math.isclose(math.fsum(by_mass.probabilities.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(by_mass[moves[0]], 0.50 / 0.95, abs_tol=1e-9)

    by_floor = truncate_distribution(
        distribution, min_probability=0.10, target_mass=1.0, max_replies=99
    )
    assert set(by_floor.probabilities) == set(moves[:3])

    by_width = truncate_distribution(
        distribution, min_probability=0.0, target_mass=1.0, max_replies=2
    )
    assert set(by_width.probabilities) == set(moves[:2])


def test_truncate_distribution_always_keeps_the_top_reply() -> None:
    """A very flat policy can put every reply under ``min_probability``."""
    moves = [chess.Move.from_uci(uci) for uci in ("e2e4", "d2d4", "g1f3", "b1c3")]
    flat = MoveDistribution({move: 0.25 for move in moves})
    kept = truncate_distribution(flat, min_probability=0.9, target_mass=0.92, max_replies=99)
    assert len(kept) == 1
    assert math.isclose(math.fsum(kept.probabilities.values()), 1.0, abs_tol=1e-9)


def test_eval_cache_evicts_least_recently_used() -> None:
    cache = EvalCache(max_size=2)
    board = chess.Board()
    keys = []
    for uci in ("e2e4", "d2d4", "g1f3"):
        child = board.copy()
        child.push(chess.Move.from_uci(uci))
        keys.append(position_key(child, 12))

    cache.put(keys[0], _eval(10))
    cache.put(keys[1], _eval(20))
    assert cache.get(keys[0]) is not None  # refreshes keys[0], so keys[1] is oldest
    cache.put(keys[2], _eval(30))

    assert cache.get(keys[1]) is None
    assert cache.get(keys[2]) is not None
    stats = cache.stats()
    assert stats.evictions == 1 and stats.size == 2 and 0.0 < stats.hit_rate < 1.0

    cache.clear()
    assert len(cache) == 0 and cache.stats().hits == 0


def test_cache_key_separates_depths_and_halfmove_clocks() -> None:
    board = chess.Board(MIDDLEGAME_FEN)
    assert position_key(board, 8) != position_key(board, 12)
    deeper = board.copy()
    deeper.halfmove_clock = 99
    assert position_key(board, 12) != position_key(deeper, 12)


# --- decision rules --------------------------------------------------------


def test_safety_filter_rejects_a_refutable_trap() -> None:
    """A trap with a huge expected payoff must lose to a sound, duller move.

    The refutation is deliberately invisible to the human model: Maia gives the
    losing reply 95% of the mass, so the 5% refutation is truncated away and the
    human-reply safety floor never sees it. Only the objective floor catches it.
    """
    board = chess.Board()
    after_sound = _child_fen(board, "e2e4")
    after_trap = _child_fen(board, "d2d4")

    evaluator = ScriptedEvaluator(
        root_scores={"e2e4": 50, "d2d4": 40},
        leaf_scores={
            after_sound: 50,
            _child_fen(board, "e2e4", "e7e5"): 50,
            _child_fen(board, "e2e4", "c7c5"): 50,
            # Objective value after d2d4: refuted, far past -tau.
            after_trap: -600,
            _child_fen(board, "d2d4", "g8f6"): 900,
            _child_fen(board, "d2d4", "d7d5"): -600,
        },
    )
    human = ScriptedHumanModel(
        {
            after_sound: {"e7e5": 0.6, "c7c5": 0.4},
            after_trap: {"g8f6": 0.95, "d7d5": 0.05},
        }
    )

    searcher = AdversarialSearcher(evaluator, human, config=SearchConfig(safety_threshold=180))
    result = searcher.search(board)

    assert result.move == chess.Move.from_uci("e2e4")
    assert not result.fallback_triggered
    assert not result.is_trap

    trap = next(c for c in result.candidates if c.move == chess.Move.from_uci("d2d4"))
    sound = next(c for c in result.candidates if c.move == chess.Move.from_uci("e2e4"))
    assert not trap.is_safe and sound.is_safe
    assert trap.expected_utility > sound.expected_utility, "the trap must be the tempting option"
    assert trap.worst_case >= -180, "truncation hid the refutation from the human-reply floor"
    assert trap.objective_score == -600, "the objective floor is what rejected it"


def test_engine_prefers_a_trap_over_a_dry_equal_line() -> None:
    """A slightly worse move that induces a 45% blunder beats the engine move."""
    board = chess.Board()
    dry = _child_fen(board, "e2e4")
    trap = _child_fen(board, "d2d4")

    evaluator = ScriptedEvaluator(
        root_scores={"e2e4": 30, "d2d4": -20},
        leaf_scores={
            dry: 30,
            _child_fen(board, "e2e4", "e7e5"): 30,
            _child_fen(board, "e2e4", "c7c5"): 30,
            trap: -20,
            _child_fen(board, "d2d4", "g8f6"): 400,  # the human blunder
            _child_fen(board, "d2d4", "d7d5"): -20,  # the correct reply
        },
    )
    human = ScriptedHumanModel(
        {
            dry: {"e7e5": 0.7, "c7c5": 0.3},
            trap: {"g8f6": 0.45, "d7d5": 0.55},
        }
    )

    searcher = AdversarialSearcher(evaluator, human)
    result = searcher.search(board)

    assert result.move == chess.Move.from_uci("d2d4")
    assert result.is_trap and not result.fallback_triggered

    chosen = next(c for c in result.candidates if c.move == result.move)
    assert chosen.is_safe
    # 0.45 * 400 + 0.55 * -20 = 169
    assert math.isclose(chosen.expected_utility, 169.0, abs_tol=1e-6)
    assert math.isclose(chosen.blunder_trap_delta, 189.0, abs_tol=1e-6)
    assert chosen.expected_utility > next(
        c.expected_utility for c in result.candidates if c.move == chess.Move.from_uci("e2e4")
    )


def test_fallback_when_every_candidate_is_unsafe() -> None:
    """A lost position: nothing clears -tau, so play Stockfish's move and don't crash."""
    board = chess.Board()
    losing = _child_fen(board, "e2e4")
    worse = _child_fen(board, "d2d4")

    evaluator = ScriptedEvaluator(
        root_scores={"e2e4": -400, "d2d4": -500},
        leaf_scores={
            losing: -400,
            _child_fen(board, "e2e4", "e7e5"): -400,
            worse: -500,
            _child_fen(board, "d2d4", "d7d5"): -500,
        },
        default_cp=-2000,
    )
    human = ScriptedHumanModel({losing: {"e7e5": 1.0}, worse: {"d7d5": 1.0}})

    result = AdversarialSearcher(evaluator, human).search(board)

    assert result.fallback_triggered
    assert result.move == chess.Move.from_uci("e2e4"), "must be Stockfish's top move"
    assert not result.is_trap
    assert result.candidates and not any(c.is_safe for c in result.candidates)


def test_mate_in_one_skips_the_expectimax_layer() -> None:
    # Back-rank mate: Ra8#.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    evaluator = ScriptedEvaluator(root_scores={}, leaf_scores={})
    human = ScriptedHumanModel({})

    result = AdversarialSearcher(evaluator, human).search(board)

    assert board.san(result.move) == "Ra8#"
    assert result.nodes_evaluated == 0
    assert evaluator.root_calls == 0 and evaluator.evaluate_calls == 0 and human.calls == 0
    assert result.expected_utility == float(MATE_SCORE_CP)


def test_terminal_positions_raise_rather_than_return_a_move() -> None:
    evaluator = ScriptedEvaluator(root_scores={}, leaf_scores={})
    human = ScriptedHumanModel({})
    searcher = AdversarialSearcher(evaluator, human)

    for fen in (
        "7k/5QK1/8/8/8/8/8/8 b - - 0 1",  # checkmate
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
        "7k/8/6K1/8/8/8/8/8 w - - 0 1",  # insufficient material
    ):
        try:
            searcher.search(chess.Board(fen))
        except TerminalPositionError:
            continue
        raise AssertionError(f"expected TerminalPositionError for {fen}")


def test_search_does_not_mutate_the_caller_board() -> None:
    board = chess.Board(MIDDLEGAME_FEN)
    before = board.fen()
    evaluator = ScriptedEvaluator(root_scores={"c4d5": 20}, leaf_scores={})
    AdversarialSearcher(evaluator, ScriptedHumanModel({})).search(board)
    assert board.fen() == before


def test_terminal_candidate_is_scored_without_engine_calls() -> None:
    """A candidate that ends the game needs no opponent model and no evaluation."""
    # King and knight against king and rook: Kxh2 removes the last mating
    # material and draws on the spot.
    board = chess.Board("7k/8/8/8/8/8/7r/1N4K1 w - - 0 1")
    evaluator = ScriptedEvaluator(root_scores={"g1h2": 0}, leaf_scores={})
    human = ScriptedHumanModel({})

    result = AdversarialSearcher(evaluator, human).search(board)

    assert board.san(result.move) == "Kxh2"
    assert result.expected_utility == 0.0
    assert result.nodes_evaluated == 0
    assert human.calls == 0, "no opponent model needed for a finished position"
    assert evaluator.evaluate_calls == 0, "terminal value is exact, not searched"

    candidate = result.candidates[0]
    assert candidate.worst_case == 0 and candidate.is_safe and not candidate.top_replies


# --- integration and performance -------------------------------------------


def test_integration_against_real_engines() -> None:
    with StockfishEvaluator() as stockfish, MaiaEvaluator(rating=1100) as maia:
        searcher = AdversarialSearcher(stockfish, maia)
        board = chess.Board(SCHOLARS_THREAT_FEN)
        result = searcher.search(board)

        assert result.move in set(board.legal_moves)
        assert result.candidates and result.nodes_evaluated > 0
        assert all(-MATE_SCORE_CP <= c.worst_case <= MATE_SCORE_CP for c in result.candidates)
        assert any(c.is_safe for c in result.candidates) or result.fallback_triggered
        print(f"    integration: {result.summary()}")


def test_two_ply_search_stays_within_the_time_budget() -> None:
    with StockfishEvaluator() as stockfish, MaiaEvaluator(rating=1100) as maia:
        searcher = AdversarialSearcher(stockfish, maia)
        # Warm the engines: first-call weight loading and backend init are not
        # part of per-move cost in a real game.
        searcher.search(chess.Board())

        timings: List[float] = []
        for fen in (ITALIAN_FEN, MIDDLEGAME_FEN, SCHOLARS_THREAT_FEN):
            started = time.perf_counter()
            result = searcher.search(chess.Board(fen))
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            print(f"    {elapsed:5.2f}s  {result.summary()}")
            assert elapsed < MAX_SEARCH_SECONDS, f"{fen} took {elapsed:.2f}s"

        assert max(timings) < MAX_SEARCH_SECONDS


def test_cache_absorbs_repeated_searches() -> None:
    with StockfishEvaluator() as stockfish, MaiaEvaluator(rating=1100) as maia:
        searcher = AdversarialSearcher(stockfish, maia)
        board = chess.Board(ITALIAN_FEN)
        first = searcher.search(board)
        second = searcher.search(board)

        assert first.move == second.move
        assert searcher.cache.stats().hits > 0, "identical searches must hit the cache"
        assert second.duration_ms < first.duration_ms


def _main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
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
