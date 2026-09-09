"""Clock math that turns a Lichess clock into search constraints.

The rule the bridge is built around: **never flag**. Moving too fast costs a
little strength; running out of time costs the whole game. So every decision
here is made against a profile's *worst observed* cost, never its mean, and the
budget is trimmed by reserves for network latency and for the endgame scramble.

A warning about the inputs. berserk converts clock fields to
:class:`datetime.timedelta` on ``gameState`` events, but the identical fields
nested inside a ``gameFull`` event's ``state`` object come through as **raw
integer milliseconds** -- the converter only rewrites top-level keys. The same
logical field therefore arrives as two different types one event apart, so
:func:`clock_seconds` accepts both and numbers are always read as milliseconds,
matching the Lichess wire format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Tuple, Union

from src.types import SearchConfig

__all__ = ["ClockValue", "SearchProfile", "TimeManager", "DEFAULT_PROFILES", "clock_seconds"]

logger = logging.getLogger(__name__)

ClockValue = Union[timedelta, int, float]
MILLISECONDS_PER_SECOND: Final[float] = 1000.0

MOVES_REMAINING: Final[int] = 40
"""Divisor in the classic ``remaining / n + increment`` budget."""

EMERGENCY_RESERVE_SECONDS: Final[float] = 3.0
"""Never *plan* to spend down to zero; keep enough to physically send moves."""

NETWORK_RESERVE_SECONDS: Final[float] = 0.25
"""Round trip to Lichess. Deducted from every budget."""


def clock_seconds(value: ClockValue) -> float:
    """Normalise a Lichess clock field to seconds.

    ``timedelta`` is taken as-is; a bare number is taken as **milliseconds**,
    which is what the Lichess API puts on the wire and what berserk leaves
    unconverted inside ``gameFull.state``.
    """
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(value) / MILLISECONDS_PER_SECOND


@dataclass(frozen=True, slots=True)
class SearchProfile:
    """A named search configuration and the wall time it is trusted to fit in."""

    name: str
    config: SearchConfig
    budget_seconds: float
    """Worst-case cost with headroom, *not* the average. Recalibrate with
    ``tests/test_lichess.py::test_profile_budgets_hold_on_this_machine``."""


# Measured on an Apple M4 across five middlegame positions, worst case per
# profile: panic 0.03s, fast 0.26s, brisk 0.56s, steady 0.72s, full 2.32s.
# Budgets carry roughly 2x headroom for slower hardware and unluckier positions.
DEFAULT_PROFILES: Final[Tuple[SearchProfile, ...]] = (
    SearchProfile("panic", SearchConfig(root_depth=4, leaf_depth=4, max_candidates=2, max_replies=2), 0.08),
    SearchProfile("fast", SearchConfig(root_depth=6, leaf_depth=6, max_candidates=2, max_replies=3), 0.60),
    SearchProfile("brisk", SearchConfig(root_depth=8, leaf_depth=8, max_candidates=4, max_replies=4), 1.20),
    SearchProfile("steady", SearchConfig(root_depth=9, leaf_depth=10, max_candidates=5, max_replies=5), 1.60),
    SearchProfile("full", SearchConfig(), 4.00),
)


class TimeManager:
    """Chooses a :class:`SearchConfig` that fits the clock."""

    def __init__(
        self,
        profiles: Tuple[SearchProfile, ...] = DEFAULT_PROFILES,
        *,
        moves_remaining: int = MOVES_REMAINING,
        emergency_reserve: float = EMERGENCY_RESERVE_SECONDS,
        network_reserve: float = NETWORK_RESERVE_SECONDS,
    ) -> None:
        if not profiles:
            raise ValueError("at least one search profile is required")
        if moves_remaining < 1:
            raise ValueError(f"moves_remaining must be >= 1, got {moves_remaining}")
        # Ascending cost, so "deepest affordable" is just the last one that fits.
        self.profiles = tuple(sorted(profiles, key=lambda profile: profile.budget_seconds))
        self.moves_remaining = moves_remaining
        self.emergency_reserve = emergency_reserve
        self.network_reserve = network_reserve

    def calculate_search_config(
        self,
        wtime: ClockValue,
        btime: ClockValue,
        winc: ClockValue,
        binc: ClockValue,
        is_white: bool,
    ) -> SearchConfig:
        """Search constraints for the side to move, given both clocks."""
        return self.select_profile(
            clock_seconds(wtime if is_white else btime),
            clock_seconds(winc if is_white else binc),
        ).config

    def budget_seconds(self, remaining: float, increment: float) -> float:
        """Time this move may spend, after reserves. May be zero or negative."""
        target = remaining / self.moves_remaining + increment
        # The clamp is what saves games with a fat increment and a thin clock:
        # a 0+5 game at 4s left has a 5.1s "target" it cannot possibly afford.
        affordable = min(target, remaining - self.emergency_reserve)
        return affordable - self.network_reserve

    def select_profile(self, remaining: float, increment: float) -> SearchProfile:
        """The deepest profile whose worst case fits the budget.

        Falls back to the cheapest profile rather than refusing to move: a
        near-instant weak move always beats a flag.
        """
        budget = self.budget_seconds(remaining, increment)
        chosen = self.profiles[0]
        for profile in self.profiles:
            if profile.budget_seconds <= budget:
                chosen = profile
        logger.debug(
            "clock: %.1fs remaining +%.1fs -> %.2fs budget -> %s profile",
            remaining, increment, budget, chosen.name,
        )
        return chosen
