"""Tests for the neural proposer's place in the search: it widens the
candidate pool and never weakens the safety filter.

    python -m tests.test_proposer
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import chess

from src import config as engine_config
from src.engine.search import AdversarialSearcher, DEFAULT_PROPOSAL_RATING
from src.types import SearchConfig
from tests.test_search import ScriptedEvaluator, ScriptedHumanModel


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
