"""Tests for the trap policy: board encoding, the network and inference.

Shape agreement between the encoding, the model and the wrapper is the thing
most likely to break silently, so it is asserted at every boundary rather than
trusted.

    python -m tests.test_policy
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from src.engine.policy_generator import NeuralCandidateGenerator, PolicyUnavailableError
from src.engine.policy_model import (
    INPUT_PLANES,
    PLANE_LEGAL,
    PLANE_RATING,
    POLICY_SIZE,
    TrapPolicyNet,
    encode_board,
    index_to_moves,
    legal_move_mask,
    move_to_index,
)

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9",
    "4k3/PPPPPPPP/8/8/8/8/8/4K3 w - - 0 1",          # eight promotions available
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",          # castling both sides
    "8/8/8/8/8/5k2/6q1/7K w - - 0 1",                # stalemate: no legal moves at all
]


def test_encoding_shape_matches_the_model_input() -> None:
    for fen in POSITIONS:
        planes = encode_board(chess.Board(fen), 1500)
        assert planes.shape == (INPUT_PLANES, 8, 8), f"{fen}: {tuple(planes.shape)}"
        assert planes.dtype == torch.float32
        assert torch.isfinite(planes).all()


def test_pieces_are_encoded_relative_to_the_side_to_move() -> None:
    """Planes 0-5 are always "us", so colour symmetry is not learned twice."""
    white = encode_board(chess.Board(), 1500)
    black = encode_board(chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"), 1500)
    assert white[0].sum() == 8 and black[0].sum() == 8, "own pawns land in plane 0 for both sides"
    assert white[0, 1].sum() == 8, "White to move: own pawns on rank 2"
    assert black[0, 6].sum() == 8, "Black to move: own pawns on rank 7"


def test_legal_plane_and_rating_plane_carry_their_signals() -> None:
    board = chess.Board(POSITIONS[1])
    planes = encode_board(board, 1900)
    destinations = {move.to_square for move in board.legal_moves}
    assert int(planes[PLANE_LEGAL].sum().item()) == len(destinations)

    low = encode_board(board, 1100)[PLANE_RATING]
    high = encode_board(board, 1900)[PLANE_RATING]
    assert low.std() == 0 and high.std() == 0, "the rating plane is constant"
    assert high.mean() > low.mean(), "and monotone in rating"


def test_move_index_round_trips_including_promotions_and_castling() -> None:
    for fen in POSITIONS:
        board = chess.Board(fen)
        for move in board.legal_moves:
            index = move_to_index(move)
            assert 0 <= index < POLICY_SIZE
            recovered = index_to_moves(board, index)
            assert move in recovered, f"{board.san(move)} did not round-trip"
            assert all(candidate in board.legal_moves for candidate in recovered)


def test_promotions_collapse_onto_one_index_and_queen_leads() -> None:
    board = chess.Board(POSITIONS[3])
    promotions = [move for move in board.legal_moves if move.promotion]
    assert promotions, "fixture assumption: promotions available"
    index = move_to_index(promotions[0])
    shared = index_to_moves(board, index)
    assert len(shared) == 4, f"four promotion pieces share an index, got {len(shared)}"
    assert shared[0].promotion == chess.QUEEN, "queen must be offered first"


def test_legal_mask_counts_distinct_source_target_pairs() -> None:
    for fen in POSITIONS:
        board = chess.Board(fen)
        mask = legal_move_mask(board)
        expected = len({(move.from_square, move.to_square) for move in board.legal_moves})
        assert int(mask.sum().item()) == expected, f"{fen}: mask {int(mask.sum())} vs {expected}"


def test_illegal_moves_carry_no_probability_after_masking() -> None:
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    legal = legal_move_mask(board).unsqueeze(0)
    logits = torch.randn(1, POLICY_SIZE) * 5.0

    probabilities = F.softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
    assert float(probabilities[~legal].sum().item()) == 0.0, "illegal mass must be exactly zero"
    assert abs(float(probabilities.sum().item()) - 1.0) < 1e-6
    assert torch.isfinite(probabilities).all()


def test_forward_pass_shape_and_speed() -> None:
    model = TrapPolicyNet().eval()
    batch = torch.stack([encode_board(chess.Board(fen), 1500) for fen in POSITIONS])
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (len(POSITIONS), POLICY_SIZE)
    assert torch.isfinite(logits).all()

    single = batch[:1]
    with torch.no_grad():
        model(single)
        started = time.perf_counter()
        for _ in range(20):
            model(single)
        per_call = (time.perf_counter() - started) / 20 * 1000
    print(f"    {model.parameter_count:,} params, {per_call:.2f}ms per forward pass")
    assert per_call < 10.0, f"forward pass must stay under 10ms, took {per_call:.2f}ms"


def test_model_rejects_the_wrong_input_shape() -> None:
    model = TrapPolicyNet().eval()
    for bad in (torch.zeros(1, INPUT_PLANES, 8), torch.zeros(1, INPUT_PLANES - 1, 8, 8), torch.zeros(1, INPUT_PLANES, 7, 8)):
        try:
            model(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected a shape error for {tuple(bad.shape)}")


def test_a_fresh_residual_head_is_exactly_zero() -> None:
    """Zero init is the whole guarantee: untrained, it contributes nothing."""
    net = TrapPolicyNet(channels=16, blocks=1)
    final = net.residual[-1]
    assert torch.count_nonzero(final.weight) == 0
    assert final.bias is not None
    assert torch.count_nonzero(final.bias) == 0

    planes = torch.randn(3, INPUT_PLANES, 8, 8)
    net.eval()
    with torch.no_grad():
        features = net.trunk(planes)
        assert torch.count_nonzero(net.residual(features)) == 0
        prior_only = net.policy(features).flatten(start_dim=1)
        assert torch.equal(net(planes), prior_only)


def test_a_pre_residual_checkpoint_still_loads() -> None:
    """Checkpoints distilled before the head existed carry no residual tensors."""
    net = TrapPolicyNet(channels=16, blocks=1)
    legacy = {k: v for k, v in net.state_dict().items() if not k.startswith("residual.")}

    fresh = TrapPolicyNet(channels=16, blocks=1)
    missing, unexpected = fresh.load_state_dict(legacy, strict=False)
    assert not unexpected
    assert all(name.startswith("residual.") for name in missing)
    assert torch.count_nonzero(fresh.residual[-1].weight) == 0


def test_a_hidden_layer_keeps_the_zero_guarantee_and_adds_a_non_linearity() -> None:
    """Only the output layer is zeroed, so the head still starts at the prior."""
    net = TrapPolicyNet(channels=16, blocks=1, residual_hidden=32)
    assert torch.count_nonzero(net.residual[-1].weight) == 0
    assert torch.count_nonzero(net.residual[0].weight) > 0, "the hidden layer must be live"

    planes = torch.randn(3, INPUT_PLANES, 8, 8)
    net.eval()
    with torch.no_grad():
        features = net.trunk(planes)
        assert torch.count_nonzero(net.residual(features)) == 0
        assert torch.equal(net(planes), net.policy(features).flatten(start_dim=1))

    assert sum(parameter.numel() for parameter in net.residual.parameters()) > 4_160


def test_checkpoint_encoding_mismatch_is_caught_on_load() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stale.pth"
        model = TrapPolicyNet(channels=16, blocks=2)
        torch.save({"version": 1, "state_dict": model.state_dict(), "channels": 16, "blocks": 2,
                    "input_planes": INPUT_PLANES - 1, "policy_size": POLICY_SIZE}, path)
        try:
            NeuralCandidateGenerator(path, device="cpu")
        except PolicyUnavailableError as exc:
            assert "input_planes" in str(exc)
        else:
            raise AssertionError("an encoding mismatch must not load silently")

        missing = Path(tmp) / "absent.pth"
        try:
            NeuralCandidateGenerator(missing, device="cpu")
        except PolicyUnavailableError:
            pass
        else:
            raise AssertionError("a missing checkpoint must raise")


def _generator(directory: Path) -> NeuralCandidateGenerator:
    """An untrained checkpoint: the wrapper's guarantees cannot depend on training."""
    torch.manual_seed(0)
    model = TrapPolicyNet(channels=16, blocks=2)
    checkpoint = directory / "trap_policy.pth"
    torch.save({"version": 1, "state_dict": model.state_dict(), "channels": 16, "blocks": 2,
                "input_planes": INPUT_PLANES, "policy_size": POLICY_SIZE}, checkpoint)
    return NeuralCandidateGenerator(checkpoint, device="cpu")


def test_candidates_are_always_legal_and_respect_top_k() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        generator = _generator(Path(tmp))
        for fen in POSITIONS:
            board = chess.Board(fen)
            legal = list(board.legal_moves)
            for top_k in (1, 3, 5, 40):
                candidates = generator.get_candidates(board, 1500, top_k=top_k)
                assert len(candidates) == min(top_k, len(legal)), f"{fen} k={top_k}: {len(candidates)}"
                assert len(set(candidates)) == len(candidates), "no duplicates"
                for move in candidates:
                    assert move in board.legal_moves, f"illegal candidate {move.uci()} in {fen}"


def test_probabilities_form_a_valid_distribution_over_legal_moves() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        generator = _generator(Path(tmp))
        scored = 0
        for fen in POSITIONS:
            board = chess.Board(fen)
            if not any(board.legal_moves):
                # A stalemate has no distribution to form; it must say so.
                try:
                    generator.move_probabilities(board, 1500)
                except ValueError:
                    continue
                raise AssertionError(f"{fen} has no legal moves but returned a distribution")
            distribution = generator.move_probabilities(board, 1500)
            assert set(distribution.probabilities) == set(board.legal_moves)
            assert abs(sum(distribution.probabilities.values()) - 1.0) < 1e-6
            scored += 1
        assert scored >= 4, "most fixtures must actually be scored"


def test_rating_changes_the_ranking() -> None:
    """The conditioning channel must actually reach the output."""
    with tempfile.TemporaryDirectory() as tmp:
        generator = _generator(Path(tmp))
        board = chess.Board(POSITIONS[1])
        low = generator.logits(board, 1100)
        high = generator.logits(board, 1900)
        assert not torch.allclose(low, high), "the rating plane must influence the logits"


def test_terminal_position_yields_no_candidates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        generator = _generator(Path(tmp))
        mated = chess.Board("7k/5QK1/8/8/8/8/8/8 b - - 0 1")
        assert generator.get_candidates(mated, 1500) == []


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
