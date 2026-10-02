"""Outbound challenges: who gets asked, and the single-game slot while pending."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from src.lichess.challenger import ChallengePolicy, Challenger, speed_of
from tests.test_lichess import BOT_ID, FakeBots, build_bot


class _Api:
    def __init__(self, **methods: Any) -> None:
        self.__dict__.update(methods)


class FakeClient:
    def __init__(self, bots: List[Dict[str, Any]], playing: Tuple[str, ...] = ()) -> None:
        self.created: List[Tuple[str, bool, Optional[int], Optional[int]]] = []
        self.cancelled: List[str] = []
        self.bots = _Api(get_online_bots=lambda limit=None: iter(bots))
        self.account = _Api(get=lambda: {"id": "latentblunder", "perfs": {"blitz": {"rating": 1500}}})
        self.users = _Api(get_realtime_statuses=lambda *ids: [
            {"id": i, "online": True, "playing": i in playing} for i in ids])
        self.challenges = _Api(create=self._create, cancel=self.cancelled.append)

    def _create(self, user: str, rated: bool, clock_limit: Optional[int] = None,
                clock_increment: Optional[int] = None) -> Dict[str, Any]:
        self.created.append((user, rated, clock_limit, clock_increment))
        return {"id": "ch1"}


def _bot(rating: int, name: str) -> Dict[str, Any]:
    return {"id": name, "perfs": {"blitz": {"rating": rating}, "rapid": {"rating": rating}}}


class _Time:
    """A clock and a stop event in one, so waiting advances time instead of sleeping."""

    def __init__(self) -> None:
        self.now = 0.0
        self.on_wait: Any = None

    def __call__(self) -> float:
        return self.now

    def is_set(self) -> bool:
        return False

    def wait(self, timeout: float) -> bool:
        self.now += timeout
        if self.on_wait:
            self.on_wait()
        return False


def _challenger(client: FakeClient, **policy: Any) -> Tuple[Challenger, Any, _Time]:
    bot, _ = build_bot(FakeBots())
    clock = _Time()
    bot._stop = clock  # type: ignore[assignment]
    challenger = Challenger(bot, client, ChallengePolicy(clocks=((180, 2),), **policy),
                            rng=random.Random(0), clock=clock)
    return challenger, bot, clock


def test_speed_follows_lichess_estimate() -> None:
    assert speed_of(180, 2) == "blitz" and speed_of(600, 5) == "rapid" and speed_of(60, 0) == "bullet"


def test_only_close_rated_free_bots_not_asked_lately_are_picked() -> None:
    client = FakeClient([_bot(1550, "near"), _bot(2400, "far"), _bot(1520, "busy"),
                         _bot(1500, BOT_ID)], playing=("busy",))
    challenger, _, clock = _challenger(client)
    assert challenger.pick("blitz") == "near"
    challenger._asked["near"] = clock.now
    assert challenger.pick("blitz") is None, "asked a moment ago; the rest are far, busy or us"


def test_a_pending_challenge_holds_the_slot_and_is_cancelled_when_unanswered() -> None:
    client = FakeClient([_bot(1550, "near")])
    challenger, bot, clock = _challenger(client)
    seen: List[Optional[str]] = []
    clock.on_wait = lambda: seen.append(bot._reservation)

    assert challenger.attempt() == "near"
    assert client.created == [("near", True, 180, 2)], "rated, at the policy's clock"
    assert seen and all(r == "outbound:near" for r in seen), "the slot is held while it waits"
    assert client.cancelled == ["ch1"] and bot.is_idle(), "cancelled and released"


def test_an_accepted_challenge_hands_the_slot_to_the_game() -> None:
    client = FakeClient([_bot(1550, "near")])
    challenger, bot, clock = _challenger(client)
    clock.on_wait = lambda: setattr(bot, "_reservation", "game1")  # gameStart arrives

    assert challenger.attempt() == "near"
    assert client.cancelled == [], "nothing to cancel"
    assert bot._reservation == "game1", "the challenger must not release the game's slot"


def test_provisional_ratings_are_ignored_for_us_and_for_them() -> None:
    """A new bot account reads as a provisional 3000; aiming at that picks the strongest bots."""
    client = FakeClient([_bot(1550, "near"), {"id": "new", "perfs": {"blitz": {"rating": 1500, "prov": True}}},
                         _bot(3000, "strong")])
    client.account = _Api(get=lambda: {"perfs": {"blitz": {"rating": 3000, "prov": True, "games": 0}}})
    challenger, _, _ = _challenger(client)
    assert challenger._own_rating("blitz") == 1500, "the provisional 3000 is not a rating"
    assert challenger.pick("blitz") == "near", "not the provisional bot, not the 3000"


def test_the_daily_bot_game_limit_pauses_us_only_when_it_is_ours() -> None:
    """Lichess caps bots at 100 games a day against bots; seen live on a target."""
    import json as _json

    import requests
    from berserk.exceptions import ResponseError

    def limit_error(name: str) -> ResponseError:
        response = requests.Response()
        response.status_code = 400
        response._content = _json.dumps({
            "error": f"{name} played 100 games against other bots today, please wait.",
            "ratelimit": {"key": "bot.vsBot.day", "seconds": 18899}}).encode()
        return ResponseError(response)

    challenger, bot, _ = _challenger(FakeClient([]))
    assert challenger._own_daily_limit(limit_error("Trainer-Bot")) is None, "theirs: just move on"
    assert challenger._own_daily_limit(limit_error(BOT_ID)) == 18899.0, "ours: wait it out"
