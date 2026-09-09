"""Thread-safe LRU transposition cache for engine evaluations.

Keyed on ``(zobrist hash, halfmove clock, depth)``:

* the **zobrist hash** covers pieces, side to move, castling rights and the en
  passant square -- everything Stockfish's static evaluation depends on;
* the **halfmove clock** is included because it is *not* part of the zobrist
  hash but does change the evaluation near the 50-move rule;
* the **depth** is included because a depth-8 score must never be served to a
  caller that asked for depth 16.

Known limitation: repetition history is not part of the key. Two positions
identical in every other respect but reached by different move orders share an
entry even though one may be a repetition draw. The searcher checks terminal and
draw states explicitly before consulting the cache, which covers the cases that
matter for move selection.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import chess
import chess.polyglot

from src.types import EngineEval

__all__ = ["CacheKey", "CacheStats", "EvalCache", "position_key"]

CacheKey = Tuple[int, int, int]


def position_key(board: chess.Board, depth: int) -> CacheKey:
    """Cache key for evaluating ``board`` at ``depth``."""
    return (chess.polyglot.zobrist_hash(board), board.halfmove_clock, depth)


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Snapshot of cache counters. ``hit_rate`` is 0.0 before any lookup."""

    hits: int
    misses: int
    evictions: int
    size: int

    @property
    def hit_rate(self) -> float:
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


class EvalCache:
    """Bounded, thread-safe LRU map from position keys to :class:`EngineEval`.

    Sized in entries, not bytes. Safe to share across threads: every operation
    takes a single lock, which is ample given lookups are nanoseconds and the
    engine calls they avoid are milliseconds.
    """

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._entries: OrderedDict[CacheKey, EngineEval] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: CacheKey) -> Optional[EngineEval]:
        """Return the cached evaluation, marking it most-recently-used."""
        with self._lock:
            evaluation = self._entries.get(key)
            if evaluation is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return evaluation

    def put(self, key: CacheKey, evaluation: EngineEval) -> None:
        """Insert or refresh an entry, evicting the least-recently-used if full."""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = evaluation
            if len(self._entries) > self._max_size:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        """Drop every entry and reset the counters."""
        with self._lock:
            self._entries.clear()
            self._hits = self._misses = self._evictions = 0

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(self._hits, self._misses, self._evictions, len(self._entries))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
