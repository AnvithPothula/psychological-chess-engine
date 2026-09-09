"""Dual Polyglot opening book: a curated trap repertoire over a broad GM book.

The trap book is queried first and the standard book only picks up where it has
nothing, so the repertoire steers the opening while the GM book keeps the bot in
theory once the opponent leaves the script.

**Traps are filtered by opponent rating.** Trap entries carry a 9-bit mask over
the Maia rating bands in the Polyglot ``learn`` field, set by measurement rather
than by reputation (see ``books/build_trap_book.py``). An entry with ``learn``
of zero is unbanded and always eligible, which is what makes an off-the-shelf
Polyglot book like the GM fallback work untouched.

Both books are memory mapped, so a probe is a binary search over a file already
in the page cache -- microseconds, no allocation, and strictly local disk I/O.
A missing book file is logged once and then behaves as an empty book; the bot
plays without it rather than refusing to start.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import List, Optional, Sequence, Tuple, Type

import chess
import chess.polyglot

from src import config
from src.types import MoveSource

__all__ = ["BookMove", "OpeningBook"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BookMove:
    """A move drawn from one of the books, with the provenance for telemetry."""

    move: chess.Move
    weight: int
    source: MoveSource


class OpeningBook:
    """Weighted move selection across a trap book and a standard fallback book.

    ``rng`` is injectable so games are reproducible under test; production uses
    a fresh :class:`random.Random`, which is what makes the bot vary its
    repertoire between games instead of walking the same line every time.
    """

    def __init__(
        self,
        trap_path: Optional[Path] = None,
        standard_path: Optional[Path] = None,
        *,
        rng: Optional[random.Random] = None,
        opponent_rating: Optional[int] = None,
    ) -> None:
        self.trap_path = trap_path if trap_path is not None else config.trap_book_path()
        self.standard_path = standard_path if standard_path is not None else config.standard_book_path()
        self._rng = rng if rng is not None else random.Random()
        self._trap = self._open(self.trap_path, "trap")
        self._standard = self._open(self.standard_path, "standard")
        self._band_bit = 0
        self.opponent_rating: Optional[int] = None
        if opponent_rating is not None:
            self.set_opponent_rating(opponent_rating)

    def set_opponent_rating(self, rating: int) -> None:
        """Restrict traps to those measured as worth playing at ``rating``.

        Until this is called every entry is eligible, so a bot that never learns
        its opponent's rating still plays the full repertoire rather than none
        of it.
        """
        band = config.nearest_maia_rating(rating)
        self.opponent_rating = rating
        self._band_bit = 1 << config.AVAILABLE_MAIA_RATINGS.index(band)
        logger.info("book: opponent %d -> rating band %d", rating, band)

    def _band_allows(self, learn: int) -> bool:
        """``learn`` of zero means unbanded, so any book without bands still works."""
        return learn == 0 or self._band_bit == 0 or bool(learn & self._band_bit)

    @staticmethod
    def _open(path: Path, label: str) -> Optional[chess.polyglot.MemoryMappedReader]:
        try:
            reader = chess.polyglot.open_reader(path)
        except (FileNotFoundError, IsADirectoryError, PermissionError) as exc:
            logger.warning("book: %s book unavailable at %s (%s), treating it as empty", label, path, exc)
            return None
        except (OSError, ValueError) as exc:
            logger.warning("book: %s book at %s is unreadable (%s), treating it as empty", label, path, exc)
            return None
        logger.info("book: loaded %s book %s (%d entries)", label, path.name, len(reader))
        return reader

    @property
    def is_empty(self) -> bool:
        """True when neither book loaded, so callers can skip probing entirely."""
        return self._trap is None and self._standard is None

    def probe(self, board: chess.Board) -> Optional[BookMove]:
        """Weighted pick from the trap book, else the standard book, else ``None``."""
        for reader, source in (
            (self._trap, MoveSource.BOOK_TRAP),
            (self._standard, MoveSource.BOOK_STANDARD),
        ):
            chosen = self._weighted_choice(reader, board, source)
            if chosen is not None:
                return chosen
        return None

    def get_book_move(self, board: chess.Board) -> Optional[chess.Move]:
        """The book move for ``board``, or ``None`` if neither book covers it."""
        chosen = self.probe(board)
        return chosen.move if chosen is not None else None

    def _weighted_choice(
        self,
        reader: Optional[chess.polyglot.MemoryMappedReader],
        board: chess.Board,
        source: MoveSource,
    ) -> Optional[BookMove]:
        if reader is None:
            return None
        try:
            # find_all already filters to legal moves and to weight >= 1, so a
            # zero-weight population can never reach random.choices.
            entries = [
                entry for entry in reader.find_all(board) if self._band_allows(entry.learn)
            ]
        except (OSError, ValueError) as exc:
            logger.warning("book: probe of %s failed (%s), skipping it", source.value, exc)
            return None
        if not entries:
            return None

        moves: Sequence[Tuple[chess.Move, int]] = [(entry.move, entry.weight) for entry in entries]
        weights: List[int] = [weight for _, weight in moves]
        move, weight = self._rng.choices(moves, weights=weights, k=1)[0]
        logger.debug(
            "book: %s offers %s -> %s (weight %d)",
            source.value,
            [f"{board.san(m)}:{w}" for m, w in moves],
            board.san(move),
            weight,
        )
        return BookMove(move=move, weight=weight, source=source)

    def close(self) -> None:
        """Release both memory maps. Idempotent."""
        for reader, label in ((self._trap, "trap"), (self._standard, "standard")):
            if reader is not None:
                reader.close()
                logger.debug("book: closed the %s book", label)
        self._trap = None
        self._standard = None

    def __enter__(self) -> OpeningBook:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()
