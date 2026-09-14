"""Tests for the distillation warm start and its use in the search.

Distillation training runs on a small synthetic corpus so the suite stays fast;
the shipped warm checkpoint, when present, is checked separately against real
Maia.

    python -m tests.test_distillation
"""

from __future__ import annotations

import logging
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch
import torch.nn.functional as F

from src import config as engine_config
from src.engine.policy_generator import NeuralCandidateGenerator
from src.engine.search import AdversarialSearcher, DEFAULT_PROPOSAL_RATING
from src.training.distill import (
    DistillationDataset,
    TARGET_TOP3_ACCURACY,
    build_corpus,
    distill,
    distillation_loss,
    sample_positions,
    top_k_agreement,
)
from src.training.model import POLICY_SIZE, TrapPolicyNet, legal_move_mask, move_to_index
from src.training.train_dpo import TrainConfig, load_warm_start, train
from src.types import MoveDistribution, SearchConfig
from tests.test_search import ScriptedEvaluator, ScriptedHumanModel

WARM_CHECKPOINT = Path("models/policy_warm.pth")
CORPUS = Path("build/distill_corpus.npz")


class ScriptedMaia:
    """Deterministic stand-in: mass concentrated on the alphabetically first move."""

    def __init__(self, rating: int = 1500) -> None:
        self.rating = rating
        self.calls = 0

    def set_rating(self, rating: int) -> None:
        self.rating = rating

    def predict_move_probabilities(
        self, board: chess.Board, *, temperature: Optional[float] = None
    ) -> MoveDistribution:
        self.calls += 1
        legal = sorted(board.legal_moves, key=lambda move: move.uci())
        weights = [2.0 ** -index for index in range(len(legal))]
        total = sum(weights)
        return MoveDistribution({move: w / total for move, w in zip(legal, weights)})


def tiny_corpus(directory: Path, count: int = 240) -> Path:
    rng = random.Random(7)
    positions = sample_positions(count, rng=rng)
    ratings = [rng.choice(engine_config.AVAILABLE_MAIA_RATINGS) for _ in positions]
    path = directory / "corpus.npz"
    build_corpus(positions, ratings, ScriptedMaia(), path)
    return path


# --- corpus ----------------------------------------------------------------


def test_sampled_positions_are_distinct_and_playable() -> None:
    rng = random.Random(3)
    positions = sample_positions(200, rng=rng)
    assert len(positions) == 200
    assert len({board.epd() for board in positions}) == 200, "EPD-distinct, not FEN-distinct"
    for board in positions:
        assert not board.is_game_over(claim_draw=False)
        assert any(board.legal_moves)
    plies = [board.ply() for board in positions]
    assert min(plies) >= 4 and max(plies) <= 40, f"walk depth out of range: {min(plies)}-{max(plies)}"


def test_corpus_round_trips_the_target_distribution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        path = tiny_corpus(directory, count=40)
        dataset = DistillationDataset(path)

        assert len(dataset) == 40
        for index in range(len(dataset)):
            sample = dataset[index]
            board = chess.Board(dataset.fens[index])
            assert abs(float(sample.target.sum().item()) - 1.0) < 1e-5, "targets must be a distribution"
            assert int(sample.legal.sum().item()) == len(
                {(m.from_square, m.to_square) for m in board.legal_moves}
            )
            # Every gram of probability mass must sit on a legal move.
            assert float(sample.target[~sample.legal].sum().item()) == 0.0


def test_corpus_stores_targets_sparsely() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = tiny_corpus(Path(tmp), count=100)
        payload = np.load(path)
        dense_bytes = 100 * POLICY_SIZE * 4
        assert payload["indices"].size < 100 * 60, "about thirty legal moves per position"
        assert path.stat().st_size < dense_bytes / 10, "sparse storage must beat dense by 10x+"


# --- loss and masking ------------------------------------------------------


def test_illegal_moves_carry_no_probability_after_masking() -> None:
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    legal = legal_move_mask(board).unsqueeze(0)
    logits = torch.randn(1, POLICY_SIZE) * 5.0

    probabilities = F.softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
    assert float(probabilities[~legal].sum().item()) == 0.0, "illegal mass must be exactly zero"
    assert abs(float(probabilities.sum().item()) - 1.0) < 1e-6
    assert torch.isfinite(probabilities).all()


def test_distillation_loss_is_minimised_by_matching_the_target() -> None:
    board = chess.Board()
    legal = legal_move_mask(board).unsqueeze(0)
    target = torch.zeros(1, POLICY_SIZE)
    moves = list(board.legal_moves)[:3]
    for move in moves:
        target[0, move_to_index(move)] = 1.0 / len(moves)

    matched = torch.where(target > 0, 10.0, -10.0)
    mismatched = -matched
    assert matched.shape == (1, POLICY_SIZE)
    assert distillation_loss(matched, target, legal) < distillation_loss(mismatched, target, legal)
    # A uniform guess must sit between the two, and stay finite despite -inf masking.
    uniform = distillation_loss(torch.zeros(1, POLICY_SIZE), target, legal)
    assert torch.isfinite(uniform)
    assert distillation_loss(matched, target, legal) < uniform < distillation_loss(
        mismatched, target, legal
    )


def test_top_k_agreement_counts_the_targets_favourite() -> None:
    target = torch.zeros(2, POLICY_SIZE)
    legal = torch.zeros(2, POLICY_SIZE, dtype=torch.bool)
    legal[:, :10] = True
    target[0, 3] = 1.0
    target[1, 7] = 1.0
    logits = torch.zeros(2, POLICY_SIZE)
    logits[0, 3] = 5.0    # ranked first
    logits[1, 7] = -1.0   # ranked last of ten
    logits[1, :7] = 1.0
    assert top_k_agreement(logits, target, legal, 1) == 0.5
    assert top_k_agreement(logits, target, legal, 10) == 1.0


# --- warm start ------------------------------------------------------------


def test_distillation_learns_a_scripted_policy() -> None:
    """On a policy this simple the network must reach the top-3 gate easily."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        corpus = tiny_corpus(directory, count=400)
        checkpoint = directory / "warm.pth"
        result = distill(
            corpus, checkpoint, epochs=25, batch_size=32, learning_rate=3e-3,
            channels=16, blocks=2, device=torch.device("cpu"), validation_fraction=0.2,
        )
        history = result["history"]
        assert isinstance(history, list)
        assert history[-1]["loss"] < history[0]["loss"], "distillation loss must fall"
        top3 = result["top3"]
        assert isinstance(top3, float)
        assert top3 > TARGET_TOP3_ACCURACY, f"top-3 was {top3}"
        assert checkpoint.exists()
        print(f"    scripted distillation top3={top3:.3f}")


def test_warm_checkpoint_loads_into_the_dpo_trainer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        corpus = tiny_corpus(directory, count=120)
        warm = directory / "warm.pth"
        distill(corpus, warm, epochs=3, batch_size=32, channels=16, blocks=2,
                device=torch.device("cpu"), validation_fraction=0.2)

        model, channels, blocks = load_warm_start(warm, torch.device("cpu"))
        assert (channels, blocks) == (16, 2), "geometry must come from the checkpoint"
        assert model.parameter_count > 0

        dataset = directory / "pairs.jsonl"
        board = chess.Board()
        legal = list(board.legal_moves)
        dataset.write_text(
            "\n".join(
                f'{{"fen": "{board.fen()}", "chosen": "{board.san(legal[i])}", '
                f'"rejected": "{board.san(legal[i + 1])}", "opponent_rating": 1500}}'
                for i in range(0, 8)
            ) + "\n"
        )
        result = train(dataset, directory / "aligned.pth", TrainConfig(epochs=4, batch_size=4),
                       device=torch.device("cpu"), warm_start=warm)
        assert (directory / "aligned.pth").exists()
        payload = torch.load(directory / "aligned.pth", map_location="cpu", weights_only=True)
        assert payload["stage"] == "aligned"
        assert payload["channels"] == 16, "the aligned checkpoint keeps the warm geometry"
        assert isinstance(result["history"], list)


def test_mismatched_warm_checkpoint_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stale.pth"
        model = TrapPolicyNet(channels=16, blocks=2)
        torch.save({"state_dict": model.state_dict(), "channels": 16, "blocks": 2,
                    "input_planes": 99, "policy_size": POLICY_SIZE}, path)
        try:
            load_warm_start(path, torch.device("cpu"))
        except ValueError as exc:
            assert "input_planes" in str(exc)
        else:
            raise AssertionError("an encoding mismatch must not warm-start silently")


# --- search integration ----------------------------------------------------


class FixedProposer:
    """Proposes named moves, so the integration can be tested without a model."""

    def __init__(self, moves: List[str], *, explode: bool = False) -> None:
        self.moves = moves
        self.explode = explode
        self.calls = 0

    def get_candidates(self, board: chess.Board, rating: int, top_k: int = 3) -> List[chess.Move]:
        self.calls += 1
        if self.explode:
            raise RuntimeError("simulated model failure")
        return [chess.Move.from_uci(uci) for uci in self.moves][:top_k]


def _searcher(
    proposer: Optional[object],
    root: Optional[Dict[str, int]] = None,
    leaf: Optional[Dict[str, int]] = None,
) -> AdversarialSearcher:
    return AdversarialSearcher(
        ScriptedEvaluator(root or {}, leaf or {}, default_cp=0),
        ScriptedHumanModel({}),
        proposer=proposer,  # type: ignore[arg-type]
    )


def test_proposals_are_added_not_substituted() -> None:
    board = chess.Board()
    proposer = FixedProposer(["b1a3", "g1h3", "a2a3"])
    searcher = _searcher(proposer)
    result = searcher.search(board, SearchConfig(max_candidates=3, max_proposals=3))

    moves = {candidate.move for candidate in result.candidates}
    assert proposer.calls == 1
    assert len(result.candidates) > 3, "the proposer must widen, not replace, the scan"
    assert chess.Move.from_uci("b1a3") in moves
    for candidate in result.candidates:
        assert candidate.move in board.legal_moves


def test_stockfish_best_move_survives_the_proposals() -> None:
    """The safety fallback needs the objective best move, whatever the net says."""
    board = chess.Board()
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    # e2e4 is Stockfish's clear best at the root; the proposer suggests neither.
    searcher = _searcher(
        FixedProposer(["a2a3", "b1a3"]), root={"e2e4": 120}, leaf={after_e4: 120}
    )
    result = searcher.search(board, SearchConfig(max_candidates=3))

    best = chess.Move.from_uci("e2e4")
    assert any(candidate.move == best for candidate in result.candidates), (
        "the objective best move must survive whatever the policy proposes"
    )
    assert not result.fallback_triggered


def test_unsafe_proposals_are_filtered_like_any_other_candidate() -> None:
    """Milestone 2's tau applies to the net's suggestions too."""
    board = chess.Board()
    poisoned = "rnbqkbnr/pppppppp/8/8/8/N7/PPPPPPPP/R1BQKBNR b KQkq - 1 1"  # after b1a3
    searcher = _searcher(FixedProposer(["b1a3"]), leaf={poisoned: -900})
    result = searcher.search(board, SearchConfig(max_candidates=3, safety_threshold=180))

    proposal = next(
        (c for c in result.candidates if c.move == chess.Move.from_uci("b1a3")), None
    )
    assert proposal is not None, "the proposal must still be scored"
    assert not proposal.is_safe, "a losing proposal must fail the safety filter"
    assert result.move != proposal.move, "and must not be played"


def test_a_failing_proposer_degrades_to_the_stockfish_scan() -> None:
    board = chess.Board()
    searcher = _searcher(FixedProposer([], explode=True))
    result = searcher.search(board, SearchConfig(max_candidates=3))
    assert result.candidates, "a dud model must not end the search"
    assert all(c.move in board.legal_moves for c in result.candidates)


def test_no_proposer_behaves_exactly_as_before() -> None:
    board = chess.Board()
    settings = SearchConfig(max_candidates=4)
    with_none = _searcher(None).search(board, settings)
    assert len(with_none.candidates) == 4
    assert DEFAULT_PROPOSAL_RATING in engine_config.AVAILABLE_MAIA_RATINGS


def test_proposals_outside_the_multipv_scan_do_not_crash() -> None:
    """Regression: a real MultiPV scan ranks only its own lines.

    The scripted evaluator scores every legal move, which hid a lookup that
    assumed every candidate had a root score. Stockfish does not work that way,
    and logging arguments are evaluated whatever the log level.
    """
    board = chess.Board()

    class NarrowEvaluator(ScriptedEvaluator):
        def analyse_root_moves(self, board, *, depth=12, multipv=None):  # type: ignore[no-untyped-def]
            full = super().analyse_root_moves(board, depth=depth, multipv=multipv)
            ranked = sorted(full, key=lambda move: move.uci())[: (multipv or 3)]
            return {move: full[move] for move in ranked}

    searcher = AdversarialSearcher(
        NarrowEvaluator({}, {}, default_cp=0),
        ScriptedHumanModel({}),
        proposer=FixedProposer(["h2h3", "g1f3"]),
    )
    logging.getLogger("src.engine.search").setLevel(logging.DEBUG)
    try:
        result = searcher.search(board, SearchConfig(max_candidates=3))
    finally:
        logging.getLogger("src.engine.search").setLevel(logging.WARNING)

    moves = {candidate.move for candidate in result.candidates}
    assert chess.Move.from_uci("h2h3") in moves, "an unscored proposal must still be a candidate"
    assert all(candidate.move in board.legal_moves for candidate in result.candidates)


def test_illegal_proposals_are_discarded() -> None:
    board = chess.Board()
    searcher = _searcher(FixedProposer(["e7e5", "h8h1"]))  # both illegal for White here
    result = searcher.search(board, SearchConfig(max_candidates=3))
    for candidate in result.candidates:
        assert candidate.move in board.legal_moves


# --- the shipped warm checkpoint -------------------------------------------


def test_shipped_warm_policy_agrees_with_maia_and_is_fast() -> None:
    """Top-5 overlap against real Maia-1500, plus the latency budget."""
    if not WARM_CHECKPOINT.exists():
        print("    (no models/policy_warm.pth; run python -m src.training.distill)")
        return

    from src.engine.maia import MaiaEvaluator

    generator = NeuralCandidateGenerator(WARM_CHECKPOINT)
    rng = random.Random(11)
    boards = sample_positions(40, rng=rng)

    # Steady state, which is what the search experiences: the first calls on an
    # accelerator pay one-off compilation that a real game amortises instantly.
    for board in boards[:10]:
        generator.get_candidates(board, 1500, top_k=5)
    started = time.perf_counter()
    for board in boards:
        generator.get_candidates(board, 1500, top_k=5)
    latency = (time.perf_counter() - started) / len(boards) * 1000

    hits = total = 0
    with MaiaEvaluator(rating=1500) as maia:
        for board in boards:
            maia_top = [
                move for move, _ in sorted(
                    maia.predict_move_probabilities(board).probabilities.items(),
                    key=lambda item: -item[1],
                )[:5]
            ]
            policy_top = generator.get_candidates(board, 1500, top_k=5)
            hits += len(set(maia_top) & set(policy_top))
            total += len(maia_top)

    overlap = hits / max(1, total)
    baseline = sum(5 / max(1, board.legal_moves.count()) for board in boards) / len(boards)
    print(f"    top-5 overlap with Maia-1500: {overlap:.1%} (random baseline {baseline:.1%}), "
          f"{latency:.2f}ms/board")
    assert overlap > 0.40, f"warm policy must beat 40% overlap, got {overlap:.1%}"
    assert overlap > baseline * 1.5, "and must clearly beat the random baseline"
    assert latency < 2.0, f"inference must stay under 2ms/board, took {latency:.2f}ms"


def _main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
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
