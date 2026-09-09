"""Tests for the dual opening book and its integration with the search.

Book files are built in a temp directory from hand-written entries, so the
fallback cascade and the weighting are tested against known contents rather
than against whatever happens to be shipped in ``src/engine/books``.

    python -m tests.test_book
"""

from __future__ import annotations

import logging
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import chess
import chess.polyglot

from src import config as engine_config
from src.engine.book import BookMove, OpeningBook
from src.engine.books.build_trap_book import (
    ALL_BANDS,
    RATING_BANDS,
    TRAP_LINES,
    compile_lines,
    eligible_bands,
    polyglot_raw_move,
    smooth,
    write_book,
)
from src.engine.search import AdversarialSearcher
from src.types import MoveSource, SearchConfig
from tests.test_search import ScriptedEvaluator, ScriptedHumanModel

SHIPPED_TRAP_BOOK = engine_config.BOOKS_DIR / "traps.bin"
SHIPPED_STANDARD_BOOK = engine_config.BOOKS_DIR / "standard.bin"


def build_book(
    directory: Path,
    name: str,
    moves: Dict[str, Sequence[Tuple[str, int]]],
    *,
    bands: int = ALL_BANDS,
) -> Path:
    """Write a Polyglot book from ``{fen: [(uci, weight), ...]}``."""
    entries: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for fen, options in moves.items():
        board = chess.Board(fen)
        for uci, weight in options:
            move = board.parse_uci(uci)
            key = (chess.polyglot.zobrist_hash(board), polyglot_raw_move(board, move))
            entries[key] = (weight, bands)
    path = directory / name
    write_book(entries, path)
    return path


# --- cascade ---------------------------------------------------------------


def test_cascade_prefers_trap_then_standard_then_nothing() -> None:
    start = chess.STARTING_FEN
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        trap = build_book(directory, "traps.bin", {start: [("b1c3", 500)]})
        standard = build_book(
            directory, "standard.bin", {start: [("e2e4", 900)], after_e4: [("c7c5", 700)]}
        )
        with OpeningBook(trap, standard, rng=random.Random(1)) as book:
            # 1. The trap book wins outright where it has an entry.
            chosen = book.probe(chess.Board())
            assert chosen is not None
            assert chosen.move == chess.Move.from_uci("b1c3")
            assert chosen.source is MoveSource.BOOK_TRAP
            assert chosen.weight == 500

            # 2. Standard book covers what the trap book does not.
            chosen = book.probe(chess.Board(after_e4))
            assert chosen is not None
            assert chosen.move == chess.Move.from_uci("c7c5")
            assert chosen.source is MoveSource.BOOK_STANDARD

            # 3. Neither book covers this position.
            assert book.get_book_move(chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")) is None


def test_weighted_choice_follows_the_entry_weights() -> None:
    """A 9:1 weighting must show up as roughly 9:1 over many draws."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        trap = build_book(
            directory, "traps.bin", {chess.STARTING_FEN: [("e2e4", 900), ("d2d4", 100)]}
        )
        empty = build_book(directory, "standard.bin", {})
        with OpeningBook(trap, empty, rng=random.Random(20240501)) as book:
            counts: Counter[str] = Counter()
            for _ in range(2000):
                chosen = book.probe(chess.Board())
                assert chosen is not None
                counts[chosen.move.uci()] += 1

    assert set(counts) == {"e2e4", "d2d4"}, "both weighted moves must appear"
    share = counts["e2e4"] / sum(counts.values())
    assert 0.85 < share < 0.95, f"expected roughly 90% e2e4, saw {share:.1%}"


def test_missing_book_files_warn_but_do_not_crash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        present = build_book(directory, "standard.bin", {chess.STARTING_FEN: [("e2e4", 10)]})

        with OpeningBook(directory / "absent.bin", present, rng=random.Random(3)) as book:
            assert not book.is_empty
            assert book.get_book_move(chess.Board()) == chess.Move.from_uci("e2e4")

        with OpeningBook(directory / "absent.bin", directory / "gone.bin") as book:
            assert book.is_empty
            assert book.get_book_move(chess.Board()) is None


def test_close_is_idempotent_and_releases_the_maps() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = build_book(Path(tmp), "traps.bin", {chess.STARTING_FEN: [("e2e4", 10)]})
        book = OpeningBook(path, path)
        assert book.get_book_move(chess.Board()) is not None
        book.close()
        book.close()  # must not raise
        assert book.is_empty
        assert book.get_book_move(chess.Board()) is None


def test_illegal_book_entries_are_ignored() -> None:
    """A book entry that is illegal in the queried position must not be played."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        # Store a move keyed to the starting position, then query a position that
        # shares nothing with it: find_all filters by legality, so nothing fires.
        trap = build_book(directory, "traps.bin", {chess.STARTING_FEN: [("e2e4", 10)]})
        empty = build_book(directory, "standard.bin", {})
        with OpeningBook(trap, empty) as book:
            assert book.get_book_move(chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")) is None


# --- the shipped repertoire ------------------------------------------------


def test_shipped_books_load_and_the_trap_book_branches() -> None:
    assert SHIPPED_TRAP_BOOK.exists(), "run python -m src.engine.books.build_trap_book"
    assert SHIPPED_STANDARD_BOOK.exists(), "standard.bin is missing from src/engine/books"

    with OpeningBook(rng=random.Random(11)) as book:
        assert not book.is_empty
        board = chess.Board()
        for san in ("e4", "e5", "Nf3", "Nc6", "Bc4"):
            board.push_san(san)
        seen = {board.san(book.probe(board).move) for _ in range(60) if book.probe(board)}  # type: ignore[union-attr]
        assert seen == {"Nd4", "Nf6"}, f"trap book should offer both gambits here, saw {seen}"


def _pairs_by_owner() -> Dict[chess.Color, set[Tuple[int, int]]]:
    """``{colour: {(zobrist, raw move)}}`` for the moves each colour actually owns."""
    owned: Dict[chess.Color, set[Tuple[int, int]]] = {chess.WHITE: set(), chess.BLACK: set()}
    for line in TRAP_LINES:
        board = chess.Board()
        for san in line.moves.split():
            move = board.parse_san(san)
            if board.turn == line.owner:
                owned[line.owner].add(
                    (chess.polyglot.zobrist_hash(board), polyglot_raw_move(board, move))
                )
            board.push(move)
    return owned


def test_trap_book_never_stores_a_move_no_line_owns() -> None:
    """Storing a victim's reply would have the bot spring its own traps.

    A move may legitimately appear from the victim's side of one line while
    being owned by another -- 1.e4 is the victim's move in the Stafford and the
    owner's move in the Vienna. What must never happen is a move stored purely
    because it appeared as a victim reply.
    """
    stored = set(compile_lines(TRAP_LINES))
    owned = _pairs_by_owner()
    assert stored == owned[chess.WHITE] | owned[chess.BLACK]

    for line in TRAP_LINES:
        board = chess.Board()
        for san in line.moves.split():
            move = board.parse_san(san)
            pair = (chess.polyglot.zobrist_hash(board), polyglot_raw_move(board, move))
            if board.turn != line.owner and pair in stored:
                assert pair in owned[board.turn], (
                    f"{line.name}: {san} is stored but no {'white' if board.turn else 'black'} "
                    "line plays it, so the bot would walk into its own trap"
                )
            board.push(move)


def test_book_does_not_offer_the_victims_losing_moves() -> None:
    """As White facing the Stafford, we must not be fed the refuted continuation.

    3.Nxe5 itself is legitimately ours -- the Cochrane Gambit plays it -- so the
    invariant bites one move later: once Black answers 3...Nc6 we are the victim,
    and the book must hand us nothing rather than 4.Nxc6 into the trap.
    """
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nf6", "Nxe5", "Nc6"):
        board.push_san(san)
    assert board.turn == chess.WHITE

    with OpeningBook(SHIPPED_TRAP_BOOK, SHIPPED_TRAP_BOOK, rng=random.Random(2)) as book:
        offered = {book.get_book_move(board) for _ in range(40)} - {None}
        assert chess.Move.from_uci("e5c6") not in offered, "book fed us the Stafford victim's move"
        assert not offered, f"the victim's side of a trap must be out of book, got {offered}"


def test_every_trap_line_is_legal_and_reachable() -> None:
    for line in TRAP_LINES:
        board = chess.Board()
        for san in line.moves.split():
            board.push(board.parse_san(san))  # raises if the curated line is wrong
        assert board.fullmove_number > 3, f"{line.name} is too short to be a trap"


# --- rating bands ----------------------------------------------------------


def test_set_opponent_rating_selects_the_nearest_maia_band() -> None:
    with OpeningBook(SHIPPED_TRAP_BOOK, SHIPPED_STANDARD_BOOK) as book:
        assert book.opponent_rating is None, "unbanded until told otherwise"
        book.set_opponent_rating(1630)
        assert book.opponent_rating == 1630
        book.set_opponent_rating(400)  # clamps to the weakest checkpoint
        book.set_opponent_rating(2800)  # clamps to the strongest


def test_band_mask_narrows_the_repertoire_as_rating_rises() -> None:
    """The whole point: fewer unsound traps offered to stronger opponents."""
    offered: Dict[int, int] = {}
    for rating in (1100, 1500, 1900):
        with OpeningBook(SHIPPED_TRAP_BOOK, SHIPPED_TRAP_BOOK, rng=random.Random(3)) as book:
            book.set_opponent_rating(rating)
            reader = chess.polyglot.open_reader(SHIPPED_TRAP_BOOK)
            try:
                live = sum(1 for entry in reader if book._band_allows(entry.learn))
            finally:
                reader.close()
            offered[rating] = live

    assert offered[1100] > offered[1900], f"repertoire must shrink with rating: {offered}"
    assert offered[1100] >= offered[1500] >= offered[1900], f"must shrink monotonically: {offered}"
    print(f"    entries offered: 1100={offered[1100]} 1500={offered[1500]} 1900={offered[1900]}")


def test_unsound_traps_switch_off_before_sound_gambits_do() -> None:
    masks = {line.name: line for line in TRAP_LINES}
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nf6", "Nxe5"):
        board.push_san(san)  # Black to move: the Stafford entry

    seen: Dict[int, set[str]] = {}
    for rating in (1100, 1900):
        with OpeningBook(SHIPPED_TRAP_BOOK, SHIPPED_TRAP_BOOK, rng=random.Random(9)) as book:
            book.set_opponent_rating(rating)
            seen[rating] = {board.san(book.probe(board).move) for _ in range(50) if book.probe(board)}  # type: ignore[union-attr]

    assert "Nc6" in seen[1100], "the Stafford must be on offer against a 1100"
    assert "Nc6" not in seen[1900], f"the Stafford must be off against a 1900, saw {seen[1900]}"


def test_unbanded_books_stay_eligible_at_every_rating() -> None:
    """A stock Polyglot book has learn == 0 and must not be filtered away."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        plain = build_book(directory, "standard.bin", {chess.STARTING_FEN: [("e2e4", 10)]}, bands=0)
        with OpeningBook(directory / "absent.bin", plain) as book:
            for rating in (1100, 1500, 1900):
                book.set_opponent_rating(rating)
                assert book.get_book_move(chess.Board()) == chess.Move.from_uci("e2e4")


def test_smoothing_damps_maia_volatility() -> None:
    """Maia-1 checkpoints are volatile across levels; raw thresholds would flicker."""
    spiky = [100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0]
    smoothed = smooth(spiky)
    assert max(smoothed) - min(smoothed) < max(spiky) - min(spiky)
    raw_flips = sum(1 for a, b in zip(spiky, spiky[1:]) if (a >= 50) != (b >= 50))
    smooth_flips = sum(1 for a, b in zip(smoothed, smoothed[1:]) if (a >= 50) != (b >= 50))
    assert smooth_flips < raw_flips, "smoothing must reduce band on/off flicker"


def test_every_band_keeps_a_usable_repertoire() -> None:
    """No rating may be left with nothing but the GM book."""
    for rating in RATING_BANDS:
        with OpeningBook(SHIPPED_TRAP_BOOK, SHIPPED_TRAP_BOOK, rng=random.Random(1)) as book:
            book.set_opponent_rating(rating)
            assert book.get_book_move(chess.Board()) is not None, f"no first move at {rating}"


# --- search integration ----------------------------------------------------


def _searcher(book: Optional[OpeningBook], **scores: int) -> AdversarialSearcher:
    evaluator = ScriptedEvaluator({}, dict(scores), default_cp=0)
    return AdversarialSearcher(evaluator, ScriptedHumanModel({}), book=book)


def test_search_returns_a_book_move_without_searching() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        trap = build_book(directory, "traps.bin", {chess.STARTING_FEN: [("e2e4", 500)]})
        empty = build_book(directory, "standard.bin", {})
        with OpeningBook(trap, empty, rng=random.Random(5)) as book:
            searcher = _searcher(book)
            result = searcher.search(chess.Board())

    assert result.move == chess.Move.from_uci("e2e4")
    assert result.source is MoveSource.BOOK_TRAP
    assert result.candidates == (), "a book move does no expectimax"
    assert not result.is_trap and not result.fallback_triggered
    assert result.nodes_evaluated == 1, "one node: the safety probe"
    evaluator = searcher.evaluator
    assert isinstance(evaluator, ScriptedEvaluator)
    assert evaluator.root_calls == 0, "the root MultiPV scan must be skipped entirely"


def test_unsafe_book_move_is_rejected_and_the_search_runs() -> None:
    """A book entry past the book floor must not be trusted over the search."""
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        trap = build_book(directory, "traps.bin", {chess.STARTING_FEN: [("e2e4", 500)]})
        empty = build_book(directory, "standard.bin", {})
        with OpeningBook(trap, empty, rng=random.Random(5)) as book:
            # White to move; the position after e2e4 scores -400 for White.
            searcher = _searcher(book, **{after_e4: -400})
            result = searcher.search(chess.Board(), SearchConfig(book_safety_threshold=300))

    assert result.source is MoveSource.SEARCH, "the poisoned book move must be discarded"
    assert result.candidates, "the normal search must have run instead"


def test_book_move_inside_the_floor_is_still_played() -> None:
    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        trap = build_book(directory, "traps.bin", {chess.STARTING_FEN: [("e2e4", 500)]})
        empty = build_book(directory, "standard.bin", {})
        with OpeningBook(trap, empty, rng=random.Random(5)) as book:
            # -250cp is what a real gambit costs; it must survive the book floor.
            searcher = _searcher(book, **{after_e4: -250})
            result = searcher.search(chess.Board(), SearchConfig(book_safety_threshold=300))

    assert result.source is MoveSource.BOOK_TRAP
    assert result.expected_utility == -250.0, "utility reports the real objective score"


def test_searcher_without_a_book_is_unchanged() -> None:
    result = _searcher(None).search(chess.Board())
    assert result.source is MoveSource.SEARCH
    assert result.candidates


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
