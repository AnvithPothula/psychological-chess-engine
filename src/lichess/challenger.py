"""Challenge other online bots, so the account plays, gets a rating, and is seen.

Three hours online drew no challenges: a bot with no rating and no games gives
nobody a reason to pick it. Bots that challenge each other are how most bots on
Lichess get both. This runs inside the bridge rather than as its own process,
because the bridge plays one game at a time: a separate challenger could have a
challenge accepted while the bridge had just accepted a human, and the second
game would be abandoned, which counts against the account. In-process, a
pending challenge holds the bridge's game slot, so incoming challenges are
declined with "later" until it is answered.

Pacing is deliberately slow -- one challenge at a time, minutes apart, each
opponent at most every few hours, and a long pause on any 429 -- because the
point is a steady trickle of games, and an account flagged for spamming
challenges gets none.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List, Mapping, Optional, Sequence, Tuple

from berserk.exceptions import ResponseError

from src.lichess.bot import NETWORK_ERRORS, LichessBot

__all__ = ["ChallengePolicy", "Challenger", "speed_of"]

logger = logging.getLogger(__name__)

RATE_LIMIT_PAUSE: Final[float] = 900.0
"""After a 429. Lichess asks for a minute; a quarter of an hour on the
challenge endpoint keeps the account well clear of being flagged."""


def speed_of(limit: int, increment: int) -> str:
    """Lichess's speed for a clock: estimated duration is limit + 40 x increment."""
    estimate = limit + 40 * increment
    if estimate < 180:
        return "bullet"
    if estimate < 480:
        return "blitz"
    if estimate < 1500:
        return "rapid"
    return "classical"


@dataclass(frozen=True, slots=True)
class ChallengePolicy:
    clocks: Tuple[Tuple[int, int], ...] = ((180, 2), (300, 3), (600, 5))
    """Blitz and rapid. Bullet is left out: the search budgets per move, and a
    loaded machine in a one-minute game loses on time before it loses on the board."""

    rated: bool = True
    """Rated, because a rating is half of what makes the account worth challenging."""

    max_rating_gap: int = 300
    default_rating: int = 1500
    """Lichess's starting rating, used until the account has one at a speed."""

    pending_seconds: float = 30.0
    """An unanswered challenge is cancelled after this, and the slot freed."""

    pause_seconds: float = 180.0
    """Between challenges, whatever their outcome."""

    repeat_after_seconds: float = 6 * 3600.0
    """Before the same bot is asked again."""

    candidates: int = 5
    """Picks at random among the closest few, so one bot is not asked every time."""


class Challenger:
    """Sends one challenge at a time to a nearby-rated online bot while the bridge is idle."""

    def __init__(
        self,
        bot: LichessBot,
        client: Any,
        policy: Optional[ChallengePolicy] = None,
        *,
        rng: Optional[random.Random] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bot = bot
        self.client = client
        self.policy = policy if policy is not None else ChallengePolicy()
        self._rng = rng if rng is not None else random.Random()
        self._clock = clock
        self._asked: Dict[str, float] = {}

    def run(self) -> None:
        """Until the bridge stops. Waits use the bridge's stop event, so shutdown is prompt."""
        logger.info("challenger: started")
        while not self.bot._stop.is_set():
            if not self.bot.is_idle():
                self.bot._stop.wait(10.0)
                continue
            try:
                self.attempt()
            except ResponseError as exc:
                if exc.status_code == 429:
                    logger.warning("challenger: rate limited, pausing %.0fs", RATE_LIMIT_PAUSE)
                    self.bot._stop.wait(RATE_LIMIT_PAUSE)
                    continue
                wait = self._own_daily_limit(exc)
                if wait is not None:
                    logger.warning("challenger: our daily bot-game limit is used up, pausing %.0fs", wait)
                    self.bot._stop.wait(wait)
                    continue
                logger.warning("challenger: HTTP %s (%s)", exc.status_code, exc)
            except NETWORK_ERRORS as exc:
                logger.warning("challenger: network error (%s)", exc)
            self.bot._stop.wait(self.policy.pause_seconds)
        logger.info("challenger: stopped")

    def attempt(self) -> Optional[str]:
        """One challenge, start to finish. Returns the opponent's id if one was sent."""
        limit, increment = self._rng.choice(self.policy.clocks)
        speed = speed_of(limit, increment)
        target = self.pick(speed)
        if target is None:
            logger.info("challenger: no suitable %s opponent online", speed)
            return None

        key = f"outbound:{target}"
        if not self.bot._reserve(key):
            return None
        self._asked[target] = self._clock()
        challenge_id = ""
        try:
            created = self.client.challenges.create(
                target, self.policy.rated, clock_limit=limit, clock_increment=increment,
            )
            challenge_id = _challenge_id(created)
            logger.info("challenger: challenged %s to %d+%d %s (%s)",
                        target, limit // 60, increment, "rated" if self.policy.rated else "casual",
                        challenge_id or "no id returned")
            if self._accepted(key):
                logger.info("challenger: %s accepted", target)
                return target
            if challenge_id:
                self._cancel(challenge_id)
                # An acceptance can cross the cancel; its gameStart replaces the
                # reservation, and the bridge then plays it like any other game.
                if self._accepted(key, seconds=3.0):
                    return target
            logger.info("challenger: %s did not accept", target)
            return target
        finally:
            self.bot._release(key)

    def pick(self, speed: str) -> Optional[str]:
        """A close-rated bot that is online, not playing, and not asked recently."""
        own = self._own_rating(speed)
        now = self._clock()
        nearby: List[Tuple[int, str]] = []
        for user in self.client.bots.get_online_bots(limit=300):
            user_id = str(user.get("id", "")).lower()
            if not user_id or user_id == self.bot.bot_id:
                continue
            if now - self._asked.get(user_id, -1e18) < self.policy.repeat_after_seconds:
                continue
            rating = _rating(user, speed)
            if rating is None or abs(rating - own) > self.policy.max_rating_gap:
                continue
            nearby.append((abs(rating - own), user_id))
        nearby.sort()
        free = self._not_playing([user_id for _gap, user_id in nearby[: 4 * self.policy.candidates]])
        choices = [user_id for _gap, user_id in nearby if user_id in free][: self.policy.candidates]
        return self._rng.choice(choices) if choices else None

    def _own_rating(self, speed: str) -> int:
        account: Mapping[str, Any] = self.client.account.get()
        rating = _rating(account, speed)
        return rating if rating is not None else self.policy.default_rating

    def _not_playing(self, user_ids: Sequence[str]) -> set[str]:
        if not user_ids:
            return set()
        statuses = self.client.users.get_realtime_statuses(*user_ids)
        return {
            str(status.get("id", "")).lower() for status in statuses
            if status.get("online") and not status.get("playing")
        }

    def _accepted(self, key: str, seconds: Optional[float] = None) -> bool:
        """True once a gameStart has taken the slot this challenge reserved."""
        deadline = self._clock() + (self.policy.pending_seconds if seconds is None else seconds)
        while self._clock() < deadline:
            if self.bot._reservation != key:
                return True
            if self.bot._stop.wait(1.0):
                return False
        return self.bot._reservation != key

    def _own_daily_limit(self, exc: ResponseError) -> Optional[float]:
        """Seconds to wait if *we* hit Lichess's bot-vs-bot daily limit, else None.

        Lichess caps a bot at 100 games a day against other bots and answers a
        challenge past it with HTTP 400, naming whichever side is over and how
        long until it resets. When that is the target, the target is already
        on cooldown; when it is us, every further challenge fails the same way.
        """
        if exc.status_code != 400 or exc.response is None:
            return None
        try:
            body = exc.response.json()
        except ValueError:
            return None
        limit = body.get("ratelimit") if isinstance(body, Mapping) else None
        if not isinstance(limit, Mapping) or limit.get("key") != "bot.vsBot.day":
            return None
        named = str(body.get("error", "")).split(" played ", 1)[0].strip().lower()
        if named != self.bot.bot_id:
            return None
        return float(limit.get("seconds", RATE_LIMIT_PAUSE))

    def _cancel(self, challenge_id: str) -> None:
        try:
            self.client.challenges.cancel(challenge_id)
        except ResponseError as exc:
            # Already declined or accepted: either way there is nothing to cancel.
            logger.debug("challenger: cancel %s returned HTTP %s", challenge_id, exc.status_code)


def _challenge_id(created: Any) -> str:
    if not isinstance(created, Mapping):
        return ""
    inner = created.get("challenge")
    source = inner if isinstance(inner, Mapping) else created
    return str(source.get("id", ""))


def _rating(user: Mapping[str, Any], speed: str) -> Optional[int]:
    """An established rating at ``speed``, or None.

    Provisional ratings count as none. Lichess starts a bot account at a
    provisional 3000 with no games -- this one read as 3000 before it had played
    a rated game -- and trusting that would aim every challenge at the strongest
    bots online.
    """
    perfs = user.get("perfs")
    perf = perfs.get(speed) if isinstance(perfs, Mapping) else None
    if not isinstance(perf, Mapping) or perf.get("prov"):
        return None
    rating = perf.get("rating")
    return int(rating) if isinstance(rating, (int, float)) else None
