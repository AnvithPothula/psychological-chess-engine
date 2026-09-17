"""Tests for the trap policy: encoding, loss, training and inference.

Shape agreement between the generator, the model and the wrapper is the thing
most likely to break silently, so it is asserted at every boundary rather than
trusted.

    python -m tests.test_policy
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List

import chess
import pytest
import torch
import torch.nn.functional as F

from src.engine.policy_generator import NeuralCandidateGenerator, PolicyUnavailableError
from src.training.dedup import deduplicate, position_key
from src.training.annotate import CP_CLAMP, MoveFeatures
from src.training.model import (
    ENGINE_FEATURE_WIDTH,
    TrapScorer,
    INPUT_PLANES,
    POLICY_SIZE,
    PLANE_LEGAL,
    PLANE_RATING,
    TrapPolicyNet,
    encode_batch,
    encode_board,
    index_to_moves,
    legal_move_mask,
    move_to_index,
)
from src.training.train_dpo import (
    PreferenceDataset,
    TrainConfig,
    anchor_loss,
    collate,
    preference_loss,
    select_device,
    train,
)

POSITIONS = [
    chess.STARTING_FEN,
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9",
    "4k3/PPPPPPPP/8/8/8/8/8/4K3 w - - 0 1",          # eight promotions available
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",          # castling both sides
    "8/8/8/8/8/5k2/6q1/7K w - - 0 1",                # stalemate: no legal moves at all
]


def sample_dataset(path: Path, count: int = 24) -> Path:
    """A synthetic preference file whose moves are guaranteed legal."""
    records = []
    for index in range(count):
        board = chess.Board(POSITIONS[index % 3])
        legal = list(board.legal_moves)
        chosen, rejected = legal[index % len(legal)], legal[(index + 1) % len(legal)]
        if chosen == rejected:
            continue
        records.append({
            "fen": board.fen(), "chosen": board.san(chosen), "rejected": board.san(rejected),
            "opponent_rating": (1100, 1500, 1900)[index % 3],
            "trap_gain_cp": 300, "bait_probability": 0.1 + 0.01 * index,
            "objective_cost_cp": -50, "expected_utility_cp": 100.0,
            "game_index": index, "ply": 4,
        })
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


# --- encoding --------------------------------------------------------------


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


def test_batch_encoding_stacks_cleanly() -> None:
    boards = [chess.Board(fen) for fen in POSITIONS]
    batch = encode_batch(boards, [1500] * len(boards))
    assert batch.shape == (len(boards), INPUT_PLANES, 8, 8)
    assert encode_batch([], []).shape == (0, INPUT_PLANES, 8, 8)
    try:
        encode_batch(boards, [1500])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched lengths must raise")


# --- model -----------------------------------------------------------------


def test_forward_pass_shape_and_speed() -> None:
    model = TrapPolicyNet().eval()
    batch = encode_batch([chess.Board(fen) for fen in POSITIONS], [1500] * len(POSITIONS))
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


# --- loss ------------------------------------------------------------------


def test_preference_loss_equals_the_full_log_softmax_formulation() -> None:
    """The brief's formula and the logit gap are the same number.

    ``log P(a) = z_a - logsumexp(z)``, so the normaliser cancels in the
    difference. Gathering raw logits is not an approximation.
    """
    torch.manual_seed(0)
    logits = torch.randn(4, POLICY_SIZE)
    chosen = torch.tensor([5, 90, 400, 4000])
    rejected = torch.tensor([6, 91, 401, 4001])

    log_probs = F.log_softmax(logits, dim=1)
    spelled_out = -F.logsigmoid(
        0.1 * (log_probs.gather(1, chosen.unsqueeze(1)) - log_probs.gather(1, rejected.unsqueeze(1))).squeeze(1)
    ).mean()
    assert torch.allclose(preference_loss(logits, chosen, rejected, 0.1), spelled_out, atol=1e-6)


def test_preference_loss_alone_touches_only_two_of_4096_logits() -> None:
    """The reason the anchor term exists.

    With no reference model, 4094 logits receive no gradient at all, so a
    softmax over them is the initialisation rather than a learned policy.
    """
    logits = torch.randn(1, POLICY_SIZE, requires_grad=True)
    chosen, rejected = torch.tensor([100]), torch.tensor([200])
    preference_loss(logits, chosen, rejected, 0.1).backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert int((logits.grad != 0).sum().item()) == 2, "exactly the two ranked moves get a gradient"


def test_anchor_loss_reaches_every_legal_move() -> None:
    board = chess.Board(POSITIONS[1])
    legal = legal_move_mask(board).unsqueeze(0)
    chosen = torch.tensor([move_to_index(next(iter(board.legal_moves)))])
    logits = torch.randn(1, POLICY_SIZE, requires_grad=True)

    anchor_loss(logits, chosen, legal).backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    touched = int((logits.grad != 0).sum().item())
    assert touched == int(legal.sum().item()), f"{touched} touched, {int(legal.sum())} legal"
    assert touched > 2, "the whole legal set, not just the pair"


def test_preference_loss_falls_as_the_gap_widens() -> None:
    chosen, rejected = torch.tensor([0]), torch.tensor([1])
    losses = []
    for gap in (-4.0, 0.0, 4.0, 20.0):
        logits = torch.zeros(1, POLICY_SIZE)
        logits[0, 0] = gap
        losses.append(float(preference_loss(logits, chosen, rejected, 1.0).item()))
    assert losses == sorted(losses, reverse=True), f"loss must fall with preference margin: {losses}"


# --- training --------------------------------------------------------------


def test_dataset_skips_records_whose_moves_are_illegal() -> None:
    records = [
        {"fen": chess.STARTING_FEN, "chosen": "e4", "rejected": "d4", "opponent_rating": 1500},
        {"fen": chess.STARTING_FEN, "chosen": "Qh5", "rejected": "d4", "opponent_rating": 1500},
        {"fen": chess.STARTING_FEN, "chosen": "e4", "rejected": "e4", "opponent_rating": 1500},
        {"fen": "not a fen", "chosen": "e4", "rejected": "d4", "opponent_rating": 1500},
    ]
    dataset = PreferenceDataset(records)
    assert len(dataset) == 1, "only the legal, non-degenerate pair survives"
    sample = dataset[0]
    assert sample.planes.shape == (INPUT_PLANES, 8, 8)
    assert bool(sample.legal[sample.chosen]) and bool(sample.legal[sample.rejected])


def test_training_reduces_the_loss_and_writes_a_loadable_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        dataset = sample_dataset(directory / "clean.jsonl")
        checkpoint = directory / "trap_policy.pth"
        config = TrainConfig(epochs=25, batch_size=8, validation_fraction=0.25)
        result = train(dataset, checkpoint, config, device=torch.device("cpu"), channels=16, blocks=2)

        history = result["history"]
        assert history[-1]["loss"] < history[0]["loss"], f"loss must fall: {history[0]} -> {history[-1]}"
        assert history[-1]["pref_acc"] >= history[0]["pref_acc"]
        assert checkpoint.exists()

        generator = NeuralCandidateGenerator(checkpoint, device="cpu")
        assert generator.model.parameter_count > 0
        print(f"    loss {history[0]['loss']:.3f} -> {history[-1]['loss']:.3f}, "
              f"pref_acc {history[-1]['pref_acc']:.2f}")


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


def test_device_selection_prefers_accelerators_but_honours_an_override() -> None:
    assert select_device("cpu") == torch.device("cpu")
    chosen = select_device()
    assert chosen.type in {"cuda", "mps", "cpu"}
    print(f"    default device: {chosen}")


# --- inference -------------------------------------------------------------


def _generator(directory: Path) -> NeuralCandidateGenerator:
    dataset = sample_dataset(directory / "clean.jsonl")
    checkpoint = directory / "trap_policy.pth"
    train(dataset, checkpoint, TrainConfig(epochs=6, batch_size=8), device=torch.device("cpu"),
          channels=16, blocks=2)
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


# --- dedup -----------------------------------------------------------------


def test_dedup_collapses_on_fen_and_rating_keeping_the_best_bait() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        source = directory / "raw.jsonl"
        fen = chess.STARTING_FEN
        rows = [
            {"fen": fen, "chosen": "e4", "rejected": "d4", "opponent_rating": 1500, "bait_probability": 0.2},
            {"fen": fen, "chosen": "Nf3", "rejected": "d4", "opponent_rating": 1500, "bait_probability": 0.7},
            {"fen": fen, "chosen": "c4", "rejected": "d4", "opponent_rating": 1900, "bait_probability": 0.3},
        ]
        source.write_text("\n".join(json.dumps(row) for row in rows) + "\n{ broken\n")

        destination = directory / "clean.jsonl"
        stats = deduplicate(source, destination)
        assert stats.read == 3 and stats.written == 2 and stats.duplicates == 1
        assert stats.malformed == 1, "a truncated final line must be counted, not fatal"

        kept = [json.loads(line) for line in destination.read_text().splitlines()]
        by_rating = {record["opponent_rating"]: record for record in kept}
        assert by_rating[1500]["chosen"] == "Nf3", "the strongest bait wins the collision"
        assert by_rating[1900]["chosen"] == "c4", "a different opponent is a different preference"


def test_dedup_key_ignores_move_counters() -> None:
    """The same position on move 9 and move 14 is one preference, not two."""
    early = chess.Board(POSITIONS[1])
    late = chess.Board(POSITIONS[1])
    late.halfmove_clock, late.fullmove_number = 40, 30

    assert early.fen() != late.fen(), "fixture assumption: the FENs differ"
    assert position_key(early.fen(), 1500) == position_key(late.fen(), 1500)
    assert position_key(early.fen(), 1500) != position_key(early.fen(), 1900)
    assert position_key("not a fen", 1500) == ("not a fen", 1500), "unparseable input must not crash"

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        source = directory / "raw.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in [
            {"fen": early.fen(), "chosen": "d3", "rejected": "d4", "opponent_rating": 1500,
             "bait_probability": 0.2},
            {"fen": late.fen(), "chosen": "Ng5", "rejected": "d4", "opponent_rating": 1500,
             "bait_probability": 0.9},
        ]) + "\n")
        stats = deduplicate(source, directory / "clean.jsonl")
        assert stats.written == 1, "differing move counters must not defeat the dedup"
        kept = json.loads((directory / "clean.jsonl").read_text().strip())
        assert kept["chosen"] == "Ng5"


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


# --- Matilda residual head --------------------------------------------------


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


def test_freezing_the_prior_routes_every_gradient_to_the_residual() -> None:
    """DPO must not be able to move the distilled representation."""
    net = TrapPolicyNet(channels=16, blocks=1)
    net.freeze_prior()
    net.train()

    planes = torch.randn(4, INPUT_PLANES, 8, 8)
    net(planes).sum().backward()

    for name, parameter in net.named_parameters():
        if name.startswith("residual."):
            assert parameter.requires_grad, name
            assert parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0, name
        else:
            assert not parameter.requires_grad, name
            assert parameter.grad is None, f"{name} received a gradient"

    trainable = {id(parameter) for parameter in net.residual_parameters}
    assert trainable == {id(p) for p in net.parameters() if p.requires_grad}


def test_a_frozen_trunk_cannot_drift_through_batchnorm() -> None:
    """requires_grad is not enough: BatchNorm updates running stats in train()."""
    net = TrapPolicyNet(channels=16, blocks=1)
    net.freeze_prior()
    net.train()  # must not re-enable the frozen modules

    before = [
        buffer.clone() for name, buffer in net.named_buffers() if "running_" in name
    ]
    for _ in range(3):
        net(torch.randn(4, INPUT_PLANES, 8, 8))
    after = [buffer for name, buffer in net.named_buffers() if "running_" in name]

    assert before, "expected BatchNorm running statistics to exist"
    assert all(torch.equal(a, b) for a, b in zip(before, after)), "the prior drifted"


def test_a_pre_residual_checkpoint_still_loads() -> None:
    """Checkpoints distilled before the head existed carry no residual tensors."""
    net = TrapPolicyNet(channels=16, blocks=1)
    legacy = {k: v for k, v in net.state_dict().items() if not k.startswith("residual.")}

    fresh = TrapPolicyNet(channels=16, blocks=1)
    missing, unexpected = fresh.load_state_dict(legacy, strict=False)
    assert not unexpected
    assert all(name.startswith("residual.") for name in missing)
    assert torch.count_nonzero(fresh.residual[-1].weight) == 0


def test_a_bare_residual_adds_no_capacity_over_the_policy_head() -> None:
    """Two 1x1 convolutions over the same features sum to one 1x1 convolution.

    This is why unfreezing the policy head beside a width-0 residual cannot
    recover preference accuracy: it widens the parameter count, not the
    function class.
    """
    net = TrapPolicyNet(channels=16, blocks=1, residual_hidden=0)
    features = torch.randn(2, 16, 8, 8)
    merged = torch.nn.Conv2d(16, 64, kernel_size=1)
    with torch.no_grad():
        merged.weight.copy_(net.policy.weight + net.residual[-1].weight)
        assert merged.bias is not None and net.policy.bias is not None
        merged.bias.copy_(net.policy.bias + net.residual[-1].bias)
        combined = net.policy(features) + net.residual(features)
        assert torch.allclose(combined, merged(features), atol=1e-6)


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

    assert sum(parameter.numel() for parameter in net.residual_parameters) > 4_160


def test_releasing_the_policy_head_keeps_the_trunk_frozen() -> None:
    """The trunk is the distilled representation and is never trainable."""
    net = TrapPolicyNet(channels=16, blocks=1, residual_hidden=32)
    net.freeze_prior(include_policy=False)
    net.train()

    net(torch.randn(4, INPUT_PLANES, 8, 8)).sum().backward()

    for name, parameter in net.named_parameters():
        frozen = name.startswith("stem.") or name.startswith("tower.")
        assert parameter.requires_grad != frozen, name
        if frozen:
            assert parameter.grad is None, f"{name} received a gradient"
    released = [n for n, q in net.named_parameters() if q.requires_grad and n.startswith("policy.")]
    assert released, "the policy head should be trainable when it is not frozen"


# --- late-fusion re-ranker --------------------------------------------------


def test_an_untrained_scorer_returns_exactly_zero() -> None:
    """Same guarantee as the residual head: it starts at the distilled prior."""
    scorer = TrapScorer()
    priors = torch.randn(7)
    features = torch.randn(7, ENGINE_FEATURE_WIDTH)
    with torch.no_grad():
        assert torch.count_nonzero(scorer(priors, features)) == 0


def test_the_scorer_reads_engine_evidence_once_trained() -> None:
    """After the output layer leaves zero, features must change the modifier."""
    scorer = TrapScorer()
    with torch.no_grad():
        for parameter in scorer.stack[-1].parameters():
            parameter.add_(0.1)

    priors = torch.zeros(2)
    losing = torch.tensor([[-5.0, 5.0, 0.9, 0.0], [-5.0, 5.0, 0.9, 0.0]])
    winning = torch.tensor([[5.0, 0.0, 0.05, 1.0], [5.0, 0.0, 0.05, 1.0]])
    with torch.no_grad():
        assert not torch.allclose(scorer(priors, losing), scorer(priors, winning))


def test_the_scorer_rejects_mismatched_shapes() -> None:
    scorer = TrapScorer()
    with pytest.raises(ValueError):
        scorer(torch.randn(3), torch.randn(3, ENGINE_FEATURE_WIDTH + 1))
    with pytest.raises(ValueError):
        scorer(torch.randn(3), torch.randn(4, ENGINE_FEATURE_WIDTH))
    with pytest.raises(ValueError):
        scorer(torch.randn(3, 1), torch.randn(3, ENGINE_FEATURE_WIDTH))


def test_the_scorer_is_small_next_to_the_convolutional_alternative() -> None:
    """The point is evidence, not capacity: this is a fraction of hidden-256."""
    assert TrapScorer().parameter_count < 20_000


def test_engine_features_are_bounded_and_ordered() -> None:
    """A mate score must not swamp a rank; clamping is what prevents that."""
    mate = MoveFeatures(centipawns=100_000, loss_vs_best=0, rank=1,
                        is_top_choice=True, legal_moves=30)
    vector = mate.as_vector()
    assert vector[0] == CP_CLAMP, "centipawns must clamp"
    assert 0.0 < vector[2] <= 1.0, "rank is a fraction of the legal moves"
    assert vector[3] == 1.0

    blunder = MoveFeatures(centipawns=-800, loss_vs_best=900, rank=25,
                           is_top_choice=False, legal_moves=30)
    assert blunder.as_vector()[1] > mate.as_vector()[1], "a worse move loses more"
    assert blunder.as_vector()[2] > vector[2], "a worse move ranks lower"


def test_records_without_engine_features_are_skipped_not_zero_filled() -> None:
    """An all-zero feature vector reads as a balanced position the engine likes."""
    from src.training.train_dpo import AnnotatedDataset

    good = {
        "fen": chess.Board().fen(), "chosen": "e4", "rejected": "d4",
        "opponent_rating": 1500,
        "chosen_features": [0.1, 0.2, 0.3, 1.0],
        "rejected_features": [0.4, 0.0, 0.1, 0.0],
    }
    missing = {k: v for k, v in good.items() if k != "chosen_features"}
    wrong_width = dict(good, chosen_features=[0.1, 0.2])

    assert len(AnnotatedDataset([good, missing, wrong_width])) == 1

    with pytest.raises(ValueError):
        AnnotatedDataset([missing])


def test_an_annotated_sample_carries_both_feature_vectors() -> None:
    from src.training.train_dpo import AnnotatedDataset

    record = {
        "fen": chess.Board().fen(), "chosen": "e4", "rejected": "d4",
        "opponent_rating": 1500,
        "chosen_features": [0.1, 0.2, 0.3, 1.0],
        "rejected_features": [0.4, 0.0, 0.1, 0.0],
    }
    sample = AnnotatedDataset([record])[0]
    assert sample.chosen_features.shape == (ENGINE_FEATURE_WIDTH,)
    assert sample.rejected_features.shape == (ENGINE_FEATURE_WIDTH,)
    assert sample.chosen != sample.rejected

