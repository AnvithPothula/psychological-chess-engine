"""Tests for the self-improvement pipeline.

Logic tests run against scripted engines; one end-to-end rollout uses the real
Stockfish and Maia and writes actual JSONL.

    python -m tests.test_training
"""

from __future__ import annotations

import json
import logging
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import chess

from src import config as engine_config
from src.engine.cognitive_elo import (
    BAND_HYSTERESIS_ELO,
    CognitiveTracker,
    elo_from_centipawn_loss,
)
from src.engine.maia import MaiaEvaluator
from src.engine.search import AdversarialSearcher
from src.engine.stockfish import StockfishEvaluator
from src.training.dpo_generator import (
    BLUNDER_HAZARD_CP,
    DPOGenerator,
    GeneratorConfig,
    RolloutUpdate,
    TrapPair,
)
from src.types import CandidateStats, MoveDistribution, PredictedReply, SearchConfig
from src.ui.constants import ROLLOUT_QUEUE_LIMIT
from src.ui.training_viewer import TrainingViewer
from tests.test_lichess import StubMaia
from tests.test_search import ScriptedEvaluator


def uniform(board: chess.Board) -> MoveDistribution:
    legal = list(board.legal_moves)
    return MoveDistribution({move: 1.0 / len(legal) for move in legal})


def candidate(
    move: str, objective: int, replies: List[tuple[str, float, int]]
) -> CandidateStats:
    return CandidateStats(
        move=chess.Move.from_uci(move),
        expected_utility=0.0,
        worst_case=objective,
        objective_score=objective,
        blunder_trap_delta=0.0,
        is_safe=True,
        top_replies=tuple(
            PredictedReply(chess.Move.from_uci(uci), probability, evaluation)
            for uci, probability, evaluation in replies
        ),
    )


# --- cognitive elo ---------------------------------------------------------


def test_elo_from_loss_is_monotone_and_clamped() -> None:
    losses = [0.0, 10.0, 26.0, 55.0, 80.0, 120.0, 200.0, 500.0]
    elos = [elo_from_centipawn_loss(loss) for loss in losses]
    assert elos == sorted(elos, reverse=True), f"worse moves must imply lower ratings: {elos}"
    assert elo_from_centipawn_loss(0.0) == elo_from_centipawn_loss(-50.0), "clamped at the top"
    assert elo_from_centipawn_loss(9999.0) == elo_from_centipawn_loss(500.0), "clamped at the bottom"
    assert 1400 < elo_from_centipawn_loss(55.0) < 1600, "55cp average loss is club strength"


def test_repeated_blunders_drag_the_estimate_down() -> None:
    tracker = CognitiveTracker(1800.0)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    probs = MoveDistribution({move: 0.9, chess.Move.from_uci("d2d4"): 0.1})

    for _ in range(25):
        tracker.update(board, move, 180.0, probs)

    assert tracker.current_elo < 1400, f"25 big blunders must not read as 1800: {tracker.current_elo}"
    assert tracker.drift < -300
    assert len(tracker.observations) == 25


def test_accurate_play_holds_the_estimate() -> None:
    tracker = CognitiveTracker(1500.0)
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    for _ in range(30):
        tracker.update(board, move, 4.0, uniform(board))
    assert abs(tracker.drift) < 1.0, "engine-quality moves must not move the estimate at all"


def test_unpredicted_blunders_move_the_estimate_less() -> None:
    """A blunder the model expected says more about the player than a freak one."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    expected = MoveDistribution({move: 0.95, chess.Move.from_uci("d2d4"): 0.05})
    surprising = MoveDistribution({move: 0.02, chess.Move.from_uci("d2d4"): 0.98})

    confident = CognitiveTracker(1800.0)
    confident.update(board, move, 200.0, expected)
    puzzled = CognitiveTracker(1800.0)
    puzzled.update(board, move, 200.0, surprising)

    assert confident.current_elo < puzzled.current_elo


def test_band_hysteresis_prevents_weight_thrashing() -> None:
    """Reloading Maia forces a ucinewgame; a boundary must not be crossed twice."""
    tracker = CognitiveTracker(1500.0)
    assert tracker.suggested_band == 1500

    tracker._elo = 1549.0  # just past the 1500/1600 midpoint, inside the deadband
    assert tracker.suggested_band == 1500, "a nudge past the midpoint must not swap"

    tracker._elo = 1500.0 + BAND_HYSTERESIS_ELO + 60.0
    assert tracker.suggested_band == 1600, "a decisive move must swap"


def test_reset_returns_to_the_starting_rating() -> None:
    tracker = CognitiveTracker(1500.0)
    board = chess.Board()
    for _ in range(10):
        tracker.update(board, chess.Move.from_uci("e2e4"), 250.0, uniform(board))
    assert tracker.current_elo < 1500
    tracker.reset(1900.0)
    assert tracker.current_elo == 1900 and not tracker.observations


# --- pair mining -----------------------------------------------------------


def _generator(**overrides: object) -> DPOGenerator:
    model = StubMaia()
    searcher = AdversarialSearcher(ScriptedEvaluator({}, {}, default_cp=0), model)
    settings = GeneratorConfig(**overrides)  # type: ignore[arg-type]
    return DPOGenerator(searcher, model, config=settings)


def test_blunder_hazard_sums_the_losing_reply_mass() -> None:
    stats = candidate("g1f3", objective=0, replies=[
        ("e7e5", 0.40, BLUNDER_HAZARD_CP + 50),   # counts
        ("c7c5", 0.25, BLUNDER_HAZARD_CP),        # counts, exactly at the line
        ("d7d5", 0.35, 10),                        # does not
    ])
    assert abs(DPOGenerator._blunder_hazard(stats) - 0.65) < 1e-9
    assert DPOGenerator._blunder_hazard(None) == 0.0


def test_pair_requires_both_a_big_payoff_and_a_taker() -> None:
    generator = _generator()
    board = chess.Board()

    rich = candidate("g1f3", objective=-50, replies=[("e7e5", 0.40, 400)])   # +450 at 40%
    assert DPOGenerator._best_bait(rich) == (450.0, 0.40)

    stingy = candidate("g1f3", objective=-50, replies=[("e7e5", 0.40, 100)])  # only +150
    gain, _ = DPOGenerator._best_bait(stingy) or (0.0, 0.0)
    assert gain < generator.config.trap_gain_cp

    ignored = candidate("g1f3", objective=-50, replies=[("e7e5", 0.01, 400)])  # nobody bites
    _, probability = DPOGenerator._best_bait(ignored) or (0.0, 0.0)
    assert probability < generator.config.min_bait_probability


def test_mined_pairs_carry_the_opponent_rating() -> None:
    """A preference without its opponent is an instruction to hang material."""
    pair = TrapPair(
        fen=chess.STARTING_FEN, chosen="Ng5", rejected="Nf3", opponent_rating=1200,
        trap_gain_cp=450, bait_probability=0.4, objective_cost_cp=-50,
        expected_utility_cp=130.0, game_index=0, ply=6,
    )
    record = json.loads(pair.to_json())
    assert record["opponent_rating"] == 1200
    assert set(record) >= {"fen", "chosen", "rejected", "opponent_rating", "bait_probability"}
    assert record["chosen"] != record["rejected"]


def test_decided_positions_are_not_mined() -> None:
    """A +900 position produces preference pairs that mean nothing."""
    generator = _generator()
    board = chess.Board()
    trap = candidate("g1f3", objective=-40, replies=[("e7e5", 0.40, 400)])
    won = candidate("e2e4", objective=950, replies=[])
    balanced = candidate("e2e4", objective=20, replies=[])

    from src.types import MoveSource, SearchResult

    def result_with(alternative: CandidateStats) -> SearchResult:
        return SearchResult(
            move=trap.move, expected_utility=130.0, is_trap=True, fallback_triggered=False,
            candidates=(trap, alternative), nodes_evaluated=8, duration_ms=1.0,
            source=MoveSource.SEARCH,
        )

    assert generator._mine_pair(board, result_with(won), 0) is None, "decided position must be skipped"
    assert generator._mine_pair(board, result_with(balanced), 0) is not None, "contested one must mine"


def test_objective_best_excludes_the_move_actually_played() -> None:
    played = candidate("g1f3", objective=-40, replies=[])
    sound = candidate("e2e4", objective=25, replies=[])
    weak = candidate("a2a3", objective=-90, replies=[])
    best = DPOGenerator._objective_best([played, sound, weak], exclude=played.move)
    assert best is not None and best.move == sound.move


# --- viewer decoupling -----------------------------------------------------


def test_observer_drops_frames_rather_than_blocking_the_rollout() -> None:
    viewer = TrainingViewer()
    update = RolloutUpdate(0, 1, chess.STARTING_FEN, None, 0, 1500, 1500, 0.0, 0, 0.0, 0, "playing")

    started = time.perf_counter()
    for _ in range(ROLLOUT_QUEUE_LIMIT * 3):
        viewer.observe(update)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, "a full queue must never stall the generator"
    assert viewer.updates.qsize() == ROLLOUT_QUEUE_LIMIT


def test_drain_keeps_only_the_newest_snapshot() -> None:
    viewer = TrainingViewer()
    board = chess.Board()
    for ply in range(5):
        board.push(list(board.legal_moves)[0])
        viewer.observe(
            RolloutUpdate(0, ply, board.fen(), board.peek(), 0, 1500, 1500 - ply, 0.0, ply, 0.0, 0, "playing")
        )
    viewer._drain()
    assert viewer._latest is not None and viewer._latest.ply == 4
    assert viewer._board.fen() == board.fen()
    assert viewer.updates.empty()


def test_generator_runs_headless_with_no_observer() -> None:
    generator = _generator(games=1, max_plies=6)
    assert generator.observer is None
    generator._publish(0, chess.Board(), None, 0, 0.0, "playing")  # must be a no-op


# --- end to end ------------------------------------------------------------


def test_rollout_against_real_engines_writes_valid_jsonl() -> None:
    with StockfishEvaluator() as stockfish, MaiaEvaluator(rating=1100) as maia:
        searcher = AdversarialSearcher(stockfish, maia)
        settings = GeneratorConfig(
            games=1,
            opponent_rating=1100,
            max_plies=16,
            search_config=SearchConfig(root_depth=6, leaf_depth=6, max_candidates=3, max_replies=3),
        )
        collected: List[RolloutUpdate] = []
        generator = DPOGenerator(searcher, maia, config=settings, observer=collected.append)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pairs.jsonl"
            stats = generator.run(output)
            lines = output.read_text().splitlines() if output.exists() else []

    assert stats.games == 1 and stats.plies > 0
    assert collected, "the observer must receive rollout snapshots"
    assert all(isinstance(update, RolloutUpdate) for update in collected)
    for line in lines:
        record = json.loads(line)
        board = chess.Board(record["fen"])
        assert board.parse_san(record["chosen"]) in board.legal_moves
        assert board.parse_san(record["rejected"]) in board.legal_moves
        assert record["opponent_rating"] in engine_config.AVAILABLE_MAIA_RATINGS
        assert record["trap_gain_cp"] >= settings.trap_gain_cp
    print(f"    {stats.plies} plies, {stats.pairs} pairs, {stats.pairs_per_minute:.1f} pairs/min")


def _main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
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
