"""Skew mining arithmetic and Polyglot packing.

No network here. What is tested is the arithmetic that decides which positions
reach the book -- baseline-relative scoring, draw handling, weight mapping and
Polyglot's castling encoding -- because an error in any of them produces a book
that looks fine and contains the wrong moves.
"""

from __future__ import annotations

import struct
from pathlib import Path

import chess
import chess.polyglot
import pytest

from src import config
from src.training.polyglot_compiler import (
    ENTRY_STRUCT,
    MAX_WEIGHT,
    MIN_WEIGHT,
    band_mask,
    compile_book,
    polyglot_raw_move,
    skew_weight,
)
from src.training.skew_miner import SkewEntry, score_for, win_ratio_for


def test_draws_count_as_half_rather_than_being_discarded() -> None:
    """Equal positions are exactly where draws concentrate, so dropping them lies."""
    white, draws, black = 300, 400, 300
    assert score_for(white, draws, black, chess.WHITE) == pytest.approx(0.5)
    assert win_ratio_for(white, black, chess.WHITE) == pytest.approx(0.5)

    # Same decisive record, far more draws: the win ratio cannot tell them apart.
    assert win_ratio_for(300, 300, chess.WHITE) == win_ratio_for(300, 300, chess.WHITE)
    assert score_for(300, 9_400, 300, chess.WHITE) == pytest.approx(0.5)
    assert score_for(600, 0, 400, chess.WHITE) == pytest.approx(0.6)
    assert win_ratio_for(600, 400, chess.WHITE) == pytest.approx(0.6)


def test_score_is_symmetric_between_the_colours() -> None:
    white, draws, black = 550, 100, 350
    assert score_for(white, draws, black, chess.WHITE) + score_for(
        white, draws, black, chess.BLACK
    ) == pytest.approx(1.0)


def test_an_empty_record_does_not_divide_by_zero() -> None:
    assert score_for(0, 0, 0, chess.WHITE) == 0.0
    assert win_ratio_for(0, 0, chess.WHITE) == 0.5, "no decisive games is not a 100% record"


def test_weight_rises_with_skew_and_stays_representable() -> None:
    """Polyglot weight is 16-bit; the reader picks in proportion to it."""
    assert skew_weight(0.06) < skew_weight(0.10) < skew_weight(0.18)
    assert skew_weight(0.0) == MIN_WEIGHT, "never zero: a zero-weight entry is unreachable"
    assert skew_weight(-1.0) == MIN_WEIGHT
    assert skew_weight(100.0) == MAX_WEIGHT
    assert MAX_WEIGHT < 65_536


def test_castling_uses_polyglot_encoding_not_the_king_move() -> None:
    """Polyglot stores castling as king-takes-own-rook.

    Writing the two-square king move python-chess reports produces entries no
    standard reader will ever match, and the book silently does nothing.
    """
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castle = chess.Move.from_uci("e1g1")
    assert board.is_castling(castle)

    encoded = polyglot_raw_move(board, castle)
    to_file, to_rank = encoded & 0b111, (encoded >> 3) & 0b111
    assert (to_file, to_rank) == (chess.square_file(chess.H1), chess.square_rank(chess.H1))

    quiet = chess.Move.from_uci("a1b1")
    plain = polyglot_raw_move(board, quiet)
    assert (plain & 0b111, (plain >> 3) & 0b111) == (chess.square_file(chess.B1), 0)


def test_band_mask_is_zero_when_unbanded_and_set_when_not() -> None:
    """Zero means eligible everywhere, which the reader already honours."""
    assert band_mask([]) == 0
    mask = band_mask([1600, 1800])
    assert mask != 0
    for rating in (1600, 1800):
        index = config.AVAILABLE_MAIA_RATINGS.index(config.nearest_maia_rating(rating))
        assert mask & (1 << index)
    unset = config.AVAILABLE_MAIA_RATINGS.index(1100)
    assert not mask & (1 << unset)


def test_a_duplicate_move_keeps_the_larger_weight_rather_than_summing(tmp_path: Path) -> None:
    """Transpositions mine the same move twice; summing counts it twice."""
    board = chess.Board()
    records = [
        {"fen": board.fen(), "uci": "e2e4", "skew": 0.06},
        {"fen": board.fen(), "uci": "e2e4", "skew": 0.10},
    ]
    destination = tmp_path / "skew.bin"
    assert compile_book(records, destination) == 1

    payload = destination.read_bytes()
    _key, _move, weight, _learn = ENTRY_STRUCT.unpack(payload)
    assert weight == skew_weight(0.10)
    assert weight != skew_weight(0.06) + skew_weight(0.10)


def test_illegal_and_malformed_records_are_dropped(tmp_path: Path) -> None:
    board = chess.Board()
    records = [
        {"fen": board.fen(), "uci": "e2e4", "skew": 0.08},
        {"fen": board.fen(), "uci": "e2e5", "skew": 0.09},   # not legal here
        {"fen": "not a fen", "uci": "e2e4", "skew": 0.09},
        {"uci": "e2e4", "skew": 0.09},                        # no fen
    ]
    assert compile_book(records, tmp_path / "skew.bin") == 1


def test_the_written_book_is_readable_and_finds_the_move(tmp_path: Path) -> None:
    board = chess.Board()
    destination = tmp_path / "skew.bin"
    compile_book([{"fen": board.fen(), "uci": "d2d4", "skew": 0.12}], destination)

    with chess.polyglot.open_reader(destination) as reader:
        found = list(reader.find_all(board))
    assert [entry.move for entry in found] == [chess.Move.from_uci("d2d4")]
    assert found[0].weight == skew_weight(0.12)


def test_entries_are_sorted_by_key_as_polyglot_requires(tmp_path: Path) -> None:
    """Readers binary-search the file; an unsorted book returns misses."""
    board = chess.Board()
    records = [{"fen": board.fen(), "uci": uci, "skew": 0.07}
               for uci in ("e2e4", "d2d4", "g1f3", "c2c4")]
    destination = tmp_path / "skew.bin"
    count = compile_book(records, destination)

    payload = destination.read_bytes()
    keys = [
        struct.unpack(">Q", payload[i * ENTRY_STRUCT.size:i * ENTRY_STRUCT.size + 8])[0]
        for i in range(count)
    ]
    assert keys == sorted(keys)


def test_a_skew_entry_records_the_confound_it_cannot_remove() -> None:
    """Rarer lines draw stronger players, so average rating has to be visible.

    Measured live: 1.d4 carries an average of 1824 and 1.Nf3 1927 in the same
    query. Part of any skew belongs to who plays the line, and raw explorer
    counts cannot separate that from the position.
    """
    entry = SkewEntry(
        fen=chess.Board().fen(), san="d4", uci="d2d4", ply=0, bot_color="white",
        games=29_859_253, score=0.551, win_ratio=0.556, skew=0.033,
        average_rating=1824, evaluation_cp=-5,
    )
    assert entry.average_rating > 0
    assert entry.games > 0 and -100 < entry.evaluation_cp < 100
