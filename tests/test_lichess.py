"""Tests for the Lichess bridge.

No token and no network: the API is driven through a fake client that mimics
berserk's shapes, including the ones berserk itself converts. The clock and
rating-swap tests use the real engines, because those are the two places where
the bridge can be silently wrong.

    python -m tests.test_lichess
"""

from __future__ import annotations

import statistics
import time
from datetime import timedelta
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

import chess
import requests
from berserk import models
from berserk.exceptions import ResponseError

from src import config as engine_config
from src.engine.maia import MaiaEvaluator
from src.engine.search import AdversarialSearcher
from src.engine.stockfish import StockfishEvaluator
from src.lichess.bot import BotConfig, LichessBot
from src.lichess.time_manager import DEFAULT_PROFILES, TimeManager, clock_seconds
from tests.test_search import ScriptedEvaluator, ScriptedHumanModel

BOT_ID = "ourbot"
CALIBRATION_FENS = (
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6",
    "r1bq1rk1/ppp1npbp/3p1np1/3Pp3/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 2 9",
    "rnbqkb1r/pp3ppp/4pn2/2pp2B1/3P4/2N1P3/PPP2PPP/R2QKBNR w KQkq - 0 5",
)


# --- fakes -----------------------------------------------------------------


class StubMaia(ScriptedHumanModel):
    """Satisfies both the searcher's HumanModel and the bridge's RatingAdaptiveModel."""

    def __init__(self, rating: int = 1100) -> None:
        super().__init__({})
        self.rating = rating
        self.ratings_loaded: List[int] = []

    def set_rating(self, rating: int) -> None:
        self.ratings_loaded.append(rating)
        self.rating = rating


class FakeAccount:
    def __init__(self, bot_id: str) -> None:
        self._bot_id = bot_id

    def get(self) -> Mapping[str, Any]:
        return {"id": self._bot_id, "username": self._bot_id}


class FakeBots:
    """Records every call so tests can assert on what the bridge sent."""

    def __init__(
        self,
        events: Sequence[Mapping[str, Any]] = (),
        game_states: Optional[Dict[str, Sequence[Mapping[str, Any]]]] = None,
    ) -> None:
        self.events = list(events)
        self.game_states = dict(game_states or {})
        self.moves: List[Tuple[str, str]] = []
        self.accepted: List[str] = []
        self.declined: List[Tuple[str, str]] = []
        self.resigned: List[str] = []
        self.event_stream_calls = 0
        self.raise_on_event_stream: List[Optional[BaseException]] = []
        self.reconnect_states: Dict[str, Sequence[Any]] = {}
        """What a second connection to a game's stream yields. An exception in
        either script is raised where it sits, as a dropped connection would be."""
        self.game_stream_calls: Dict[str, int] = {}
        self.raise_on_move: List[BaseException] = []

    def stream_incoming_events(self) -> Iterator[Mapping[str, Any]]:
        self.event_stream_calls += 1
        if self.raise_on_event_stream:
            failure = self.raise_on_event_stream.pop(0)
            if failure is not None:
                raise failure
        yield from self.events

    def stream_game_state(self, game_id: str) -> Iterator[Mapping[str, Any]]:
        calls = self.game_stream_calls[game_id] = self.game_stream_calls.get(game_id, 0) + 1
        script = self.game_states.get(game_id, []) if calls == 1 else self.reconnect_states.get(game_id, [])
        for item in script:
            if isinstance(item, BaseException):
                raise item
            yield item

    def make_move(self, game_id: str, move: str) -> None:
        if self.raise_on_move:
            raise self.raise_on_move.pop(0)
        self.moves.append((game_id, move))

    def accept_challenge(self, challenge_id: str) -> None:
        self.accepted.append(challenge_id)

    def decline_challenge(self, challenge_id: str, reason: str = "generic") -> None:
        self.declined.append((challenge_id, reason))

    def resign_game(self, game_id: str) -> None:
        self.resigned.append(game_id)


class FakeClient:
    def __init__(self, bots: FakeBots, bot_id: str = BOT_ID) -> None:
        self.bots = bots
        self.account = FakeAccount(bot_id)


def response_error(status: int, retry_after: Optional[str] = None) -> ResponseError:
    response = requests.Response()
    response.status_code = status
    response.reason = "Test"
    response.url = "https://lichess.org/api/test"
    response._content = b"{}"
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return ResponseError(response)


def challenge_event(
    challenge_id: str = "chal1",
    *,
    variant: str = "standard",
    control_type: str = "clock",
    limit: int = 300,
    increment: int = 3,
    rated: bool = True,
) -> Mapping[str, Any]:
    return {
        "type": "challenge",
        "challenge": {
            "id": challenge_id,
            "status": "created",
            "challenger": {"id": "human", "name": "human", "rating": 1630},
            "destUser": {"id": BOT_ID, "name": BOT_ID},
            "variant": {"key": variant, "name": variant},
            "rated": rated,
            "speed": "correspondence" if control_type != "clock" else "blitz",
            "timeControl": {"type": control_type, "limit": limit, "increment": increment},
            "color": "random",
            "perf": {"icon": "", "name": "Blitz"},
        },
    }


def build_bot(bots: FakeBots, maia: Optional[StubMaia] = None) -> Tuple[LichessBot, StubMaia]:
    model = maia if maia is not None else StubMaia()
    searcher = AdversarialSearcher(ScriptedEvaluator({}, {}, default_cp=0), model)
    bot = LichessBot(FakeClient(bots), searcher, model, bot_id=BOT_ID, config=BotConfig(
        reconnect_backoff_seconds=0.001, max_reconnect_backoff_seconds=0.001, pacing=False))
    return bot, model


# --- clock handling --------------------------------------------------------


def test_clock_seconds_accepts_both_berserk_representations() -> None:
    assert clock_seconds(timedelta(seconds=297)) == 297.0
    assert clock_seconds(297000) == 297.0  # raw milliseconds, as sent on the wire
    assert clock_seconds(0) == 0.0


def test_gamefull_and_gamestate_clocks_produce_the_same_config() -> None:
    """berserk converts wtime on gameState but not inside gameFull.state.

    The identical field arrives as timedelta on one event and as an int of
    milliseconds on the next. If the bridge reads them differently it will
    either stall or blitz out its whole clock, so pin the behaviour here.
    """
    raw_full = {
        "type": "gameFull",
        "id": "g1",
        "createdAt": 1700000000000,
        "state": {"type": "gameState", "moves": "", "wtime": 300000, "btime": 300000,
                  "winc": 3000, "binc": 3000, "status": "started"},
    }
    raw_state = {"type": "gameState", "moves": "e2e4", "wtime": 300000, "btime": 300000,
                 "winc": 3000, "binc": 3000, "status": "started"}

    full = cast(Mapping[str, Any], models.GameState.convert(raw_full))
    state = cast(Mapping[str, Any], models.GameState.convert(raw_state))
    nested = cast(Mapping[str, Any], full["state"])
    assert isinstance(nested["wtime"], int), "berserk leaves nested clocks unconverted"
    assert isinstance(state["wtime"], timedelta), "berserk converts top-level clocks"

    manager = TimeManager()
    from_full = manager.calculate_search_config(
        nested["wtime"], nested["btime"], nested["winc"], nested["binc"], True
    )
    from_state = manager.calculate_search_config(
        state["wtime"], state["btime"], state["winc"], state["binc"], True
    )
    assert from_full == from_state


def test_profile_ladder_degrades_as_the_clock_drains() -> None:
    manager = TimeManager()
    names = [manager.select_profile(remaining, 0.0).name for remaining in (600, 60, 30, 10, 2)]
    assert names[0] == "full"
    assert names[-1] == "panic"
    budgets = [manager.budget_seconds(remaining, 0.0) for remaining in (600, 60, 30, 10, 2)]
    assert budgets == sorted(budgets, reverse=True), "budget must fall with the clock"

    # A fat increment on a thin clock must not authorise a long think.
    assert manager.select_profile(4.0, 5.0).name in {"panic", "fast"}
    # And we never refuse to move.
    assert manager.select_profile(0.0, 0.0).name == "panic"


def test_profile_budgets_hold_on_this_machine() -> None:
    """Calibration: every profile must fit the budget the ladder promises."""
    with StockfishEvaluator() as stockfish, MaiaEvaluator(rating=1100) as maia:
        searcher = AdversarialSearcher(stockfish, maia)
        searcher.search(chess.Board())  # warm the engines

        for profile in DEFAULT_PROFILES:
            timings = []
            for fen in CALIBRATION_FENS:
                searcher.cache.clear()
                started = time.perf_counter()
                searcher.search(chess.Board(fen), profile.config)
                timings.append(time.perf_counter() - started)
            worst = max(timings)
            print(f"    {profile.name:7} worst {worst:5.2f}s / {profile.budget_seconds:4.2f}s budget "
                  f"(mean {statistics.mean(timings):.2f}s)")
            assert worst <= profile.budget_seconds, (
                f"{profile.name} took {worst:.2f}s against a {profile.budget_seconds:.2f}s budget; "
                "recalibrate DEFAULT_PROFILES for this hardware"
            )


# --- opponent model --------------------------------------------------------


def test_nearest_maia_rating_maps_and_clamps() -> None:
    nearest = engine_config.nearest_maia_rating
    assert nearest(1630) == 1600
    assert nearest(1200) == 1200
    assert nearest(400) == 1100, "clamped to the weakest checkpoint"
    assert nearest(2900) == 1900, "clamped to the strongest checkpoint"
    assert nearest(1150) == 1100, "ties break downward"
    for rating in range(800, 2600, 7):
        assert nearest(rating) in engine_config.AVAILABLE_MAIA_RATINGS


def test_maia_rating_swap_changes_a_repeated_position() -> None:
    """Regression: setting WeightsFile alone is a no-op on positions already seen.

    Lc0 keeps serving the old network until a ucinewgame, so the probe below
    deliberately evaluates the *same* position before and after the swap.
    """
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3")
    with MaiaEvaluator(rating=1100) as maia:
        weak = maia.predict_move_probabilities(board)
        maia.set_rating(1900)
        assert maia.rating == 1900
        strong = maia.predict_move_probabilities(board)

        weak_top = weak.most_likely
        assert weak.probabilities != strong.probabilities, "weights swap did not take effect"
        blunder = next(m for m in board.legal_moves if board.san(m) == "Nf6")
        assert weak[blunder] > strong[blunder], "the weaker model must blunder more often"
        print(f"    maia-1100 Nf6={weak[blunder]:.1%} -> maia-1900 Nf6={strong[blunder]:.1%} "
              f"(top move {board.san(weak_top)})")

        maia.set_rating(1100)
        assert maia.predict_move_probabilities(board).probabilities == weak.probabilities


# --- challenge policy ------------------------------------------------------


def test_challenge_filtering() -> None:
    cases = [
        (challenge_event("ok"), None),
        (challenge_event("v", variant="chess960"), "variant"),
        (challenge_event("h", variant="crazyhouse"), "variant"),
        (challenge_event("c", control_type="correspondence"), "timeControl"),
        (challenge_event("u", control_type="unlimited"), "timeControl"),
        (challenge_event("f", limit=15), "tooFast"),
        (challenge_event("s", limit=99999), "tooSlow"),
    ]
    for event, expected in cases:
        bots = FakeBots()
        bot, _ = build_bot(bots)
        bot._handle_event(event)
        challenge_id = event["challenge"]["id"]
        if expected is None:
            assert bots.accepted == [challenge_id], f"{challenge_id} should have been accepted"
            assert bots.declined == []
        else:
            assert bots.accepted == []
            assert bots.declined == [(challenge_id, expected)], f"{challenge_id} -> {bots.declined}"


def test_second_challenge_is_declined_while_a_game_is_reserved() -> None:
    """One game at a time: the opponent model is per-game state on one process."""
    bots = FakeBots()
    bot, _ = build_bot(bots)
    bot._handle_event(challenge_event("first"))
    bot._handle_event(challenge_event("second"))
    assert bots.accepted == ["first"]
    assert bots.declined == [("second", "later")]


def test_declining_frees_the_slot_for_the_next_challenge() -> None:
    bots = FakeBots()
    bot, _ = build_bot(bots)
    bot._handle_event(challenge_event("variant-one", variant="chess960"))
    bot._handle_event(challenge_event("good-one"))
    assert bots.accepted == ["good-one"]


# --- game loop -------------------------------------------------------------


def _game_full(moves: str = "", *, opponent_rating: int = 1630) -> Mapping[str, Any]:
    return {
        "type": "gameFull",
        "id": "g1",
        "white": {"id": BOT_ID, "name": BOT_ID, "rating": 2000},
        "black": {"id": "human", "name": "human", "rating": opponent_rating},
        "initialFen": "startpos",
        "state": {"type": "gameState", "moves": moves, "wtime": 300000, "btime": 300000,
                  "winc": 3000, "binc": 3000, "status": "started"},
    }


def _game_state(moves: str, status: str = "started", **extra: Any) -> Mapping[str, Any]:
    state = {"type": "gameState", "moves": moves, "wtime": timedelta(seconds=280),
             "btime": timedelta(seconds=290), "winc": timedelta(seconds=3),
             "binc": timedelta(seconds=3), "status": status}
    state.update(extra)
    return state


def test_play_game_moves_swaps_model_and_stops_at_the_result() -> None:
    bots = FakeBots(game_states={"g1": [
        _game_full(),                                    # our move as White
        _game_state("e2e4 e7e5"),                        # our move again
        _game_state("e2e4 e7e5 g1f3 b8c6", "resign", winner="white"),
    ]})
    bot, model = build_bot(bots)
    bot.play_game("g1")

    assert model.ratings_loaded == [1630], (
        "Maia-2 takes the rating as an input, so the opponent's real 1630 reaches "
        "the model -- there is no checkpoint to snap to a band"
    )
    assert len(bots.moves) == 2, f"expected two submitted moves, got {bots.moves}"

    board = chess.Board()
    assert chess.Move.from_uci(bots.moves[0][1]) in board.legal_moves
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert chess.Move.from_uci(bots.moves[1][1]) in board.legal_moves


def test_bot_waits_when_it_is_not_its_turn() -> None:
    bots = FakeBots(game_states={"g1": [
        {**_game_full(), "white": {"id": "human", "name": "human", "rating": 1400},
         "black": {"id": BOT_ID, "name": BOT_ID, "rating": 2000}},
    ]})
    bot, model = build_bot(bots)
    bot.play_game("g1")
    assert bots.moves == [], "White to move and we are Black, so we must not move"
    assert model.ratings_loaded == [1400]


def test_board_resyncs_from_the_moves_string_after_a_replay() -> None:
    """Lichess resends state after a reconnect; a bot that assumes +1 move desyncs."""
    bots = FakeBots(game_states={"g1": [
        _game_full("e2e4 e7e5 g1f3 b8c6"),
        _game_state("e2e4 e7e5 g1f3 b8c6"),   # duplicate: same position resent
        _game_state("e2e4 e7e5 g1f3 b8c6", "mate", winner="black"),
    ]})
    bot, _ = build_bot(bots)
    bot.play_game("g1")

    board = chess.Board()
    for uci in "e2e4 e7e5 g1f3 b8c6".split():
        board.push_uci(uci)
    assert len(bots.moves) == 2, "both identical states are our turn, both get a move"
    for _, uci in bots.moves:
        assert chess.Move.from_uci(uci) in board.legal_moves, "replayed state produced an illegal move"


def test_search_failure_resigns_rather_than_flagging() -> None:
    class BrokenEvaluator(ScriptedEvaluator):
        def analyse_root_moves(self, board, *, depth=12, multipv=None):  # type: ignore[no-untyped-def]
            from src.engine import EngineAnalysisError

            raise EngineAnalysisError("engine died")

    bots = FakeBots(game_states={"g1": [_game_full()]})
    model = StubMaia()
    bot = LichessBot(
        FakeClient(bots), AdversarialSearcher(BrokenEvaluator({}, {}), model), model, bot_id=BOT_ID
    )
    bot.play_game("g1")
    assert bots.resigned == ["g1"], "a dead engine must resign, not sit and flag"
    assert bots.moves == []


# --- resilience ------------------------------------------------------------


def test_rate_limited_move_is_not_retried_into_the_ground() -> None:
    bots = FakeBots()
    bot, _ = build_bot(bots)
    rejections = [response_error(400)]

    def rejecting_make_move(game_id: str, move: str) -> None:
        if rejections:
            raise rejections.pop(0)
        bots.moves.append((game_id, move))

    bots.make_move = rejecting_make_move  # type: ignore[method-assign]
    assert bot._submit_move("g1", "e2e4") is False, "a 4xx rejection must not be retried"
    assert bots.moves == []


def test_retry_after_header_is_honoured() -> None:
    bots = FakeBots()
    bot, _ = build_bot(bots)
    assert bot._retry_after(response_error(429, retry_after="7"), 60.0) == 7.0
    assert bot._retry_after(response_error(429), 60.0) == 60.0
    assert bot._retry_after(response_error(429, retry_after="nonsense"), 60.0) == 60.0


def test_event_stream_reconnects_after_a_drop() -> None:
    bots = FakeBots(events=[challenge_event("late")])
    bots.raise_on_event_stream = [
        requests.ConnectionError("dropped"),
        response_error(429, retry_after="0"),
        None,
    ]
    bot, _ = build_bot(bots)
    bot.config = BotConfig(reconnect_backoff_seconds=0.01, max_reconnect_backoff_seconds=0.02)

    original = bot._handle_event

    def stop_after_first(event: Mapping[str, Any]) -> None:
        original(event)
        bot.stop()

    bot._handle_event = stop_after_first  # type: ignore[method-assign]
    bot.run()

    assert bots.event_stream_calls == 3, "must reconnect past both failures"
    assert bots.accepted == ["late"], "the event after reconnecting must still be handled"


def _main() -> int:
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


# --- the live experiment: band, speed, and a randomised book per game --------


def _challenge(**challenger: Any) -> Dict[str, Any]:
    challenge = dict(challenge_event()["challenge"])
    challenge["challenger"] = {"id": "human", "name": "human", "rating": 1500, **challenger}
    return challenge


def test_only_humans_in_the_mined_band_at_the_mined_speeds_are_accepted() -> None:
    bot, _ = build_bot(FakeBots())
    assert bot._decline_reason(_challenge()) is None
    assert bot._decline_reason(_challenge(rating=1099)) == "generic"
    assert bot._decline_reason(_challenge(rating=1701)) == "generic"
    assert bot._decline_reason(_challenge(rating=None)) == "generic"
    assert bot._decline_reason(_challenge(title="BOT")) == "noBot"
    assert bot._decline_reason({**_challenge(), "speed": "bullet"}) == "tooFast"
    assert bot._decline_reason({**_challenge(), "speed": "classical"}) == "tooSlow"


class _Book:
    is_empty = True  # the search skips an empty book, so no probe is needed

    def __init__(self) -> None:
        self.rating: Optional[int] = None

    def set_opponent_rating(self, rating: int) -> None:
        self.rating = rating


def test_each_game_draws_its_book_at_random_and_logs_the_draw(tmp_path: Any) -> None:
    """A live run without a control arm could not attribute anything to the book."""
    import json
    import random

    books = {"standard": _Book(), "skew": _Book()}
    log = tmp_path / "games.jsonl"
    bot, _ = build_bot(FakeBots())
    bot.books = books  # type: ignore[assignment]
    bot.config = BotConfig(game_log=log, control_share=0.5)
    bot._rng = random.Random(0)

    arms = []
    for game in range(40):
        session = bot._start_session(f"g{game}", _game_full())
        assert bot.searcher.book is books[session.arm]
        assert books[session.arm].rating == 1630, "the chosen book is banded to the opponent"
        arms.append(session.arm)
    assert 10 < arms.count("skew") < 30, "a fair coin, not one arm"

    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row["arm"] for row in rows] == arms
    assert all(row["event"] == "start" and "human" not in json.dumps(row) for row in rows)


def test_bots_are_declined_unless_allowed_and_then_skip_the_human_band() -> None:
    bot, _ = build_bot(FakeBots())
    strong_bot = {**_challenge(title="BOT", rating=2900), "speed": "bullet"}
    assert bot._decline_reason(strong_bot) == "noBot"
    bot.config = BotConfig(allow_bots=True)
    assert bot._decline_reason(strong_bot) is None
    assert bot._decline_reason(_challenge(rating=2900)) == "generic", "humans keep the band"


def test_a_stream_closed_on_open_backs_off_instead_of_reconnecting_every_two_seconds() -> None:
    """Seen live: a second bot process made the server close the stream on open,
    and the bridge reopened it every 2s because a clean close reset the backoff."""
    bot, _ = build_bot(FakeBots(events=[]))
    bot.config = BotConfig()

    class Clock:
        waits: List[float] = []

        def is_set(self) -> bool:
            return False

        def wait(self, timeout: float) -> bool:
            self.waits.append(timeout)
            return len(self.waits) >= 7

    clock = Clock()
    bot._stop = clock  # type: ignore[assignment]
    bot.run()
    assert clock.waits == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


def test_our_own_outbound_challenge_echoed_by_the_stream_is_left_alone() -> None:
    """Seen live: the bridge declined its own challenge; only a 404 saved it."""
    bots = FakeBots()
    bot, _ = build_bot(bots)
    echo = dict(challenge_event("ours")["challenge"])
    echo["challenger"] = {"id": BOT_ID, "name": BOT_ID, "rating": 1500, "title": "BOT"}
    bot._handle_challenge(echo)
    assert bots.declined == [] and bots.accepted == []



# --- the four live losses on time ---------------------------------------------


def _dropped() -> BaseException:
    from berserk.exceptions import ApiError

    return ApiError(requests.ConnectionError("Remote end closed connection without response"))


def test_a_move_whose_connection_drops_is_retried_not_fatal() -> None:
    """EPngOS1B and aevBdKIp: berserk wraps the drop in ApiError, which killed the game thread."""
    bots = FakeBots()
    bots.raise_on_move = [_dropped()]
    bot, _ = build_bot(bots)
    bot.config = BotConfig(max_reconnect_backoff_seconds=0.001)
    assert bot._submit_move("g1", "e2e4")
    assert bots.moves == [("g1", "e2e4")]


def test_a_dropped_game_stream_is_rejoined_and_the_game_finished() -> None:
    finished = _game_state("e2e4 e7e5", status="resign", winner="white")
    bots = FakeBots(game_states={"g1": [_game_full(), _dropped()]})
    bots.reconnect_states = {"g1": [_game_full("e2e4 e7e5"), finished]}
    bot, _ = build_bot(bots)
    starts: List[str] = []
    original = bot._start_session
    bot._start_session = lambda gid, ev: (starts.append(gid), original(gid, ev))[1]  # type: ignore[method-assign]
    bot.play_game("g1")
    assert bots.game_stream_calls["g1"] == 2, "rejoined after the drop"
    assert starts == ["g1"], "one session, one book draw, one log line"


def test_a_claimable_repetition_is_played_on_not_waited_out() -> None:
    """fUeLxEJr and AqpmpSQx: the next move could repeat a third time, the bridge
    called that terminal, and the clock ran out while Lichess waited for a move."""
    moves = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1"  # black can claim with ...Ng8; nothing has repeated three times
    full = {**_game_full(moves), "white": {"id": "human", "name": "human", "rating": 1400},
            "black": {"id": BOT_ID, "name": BOT_ID, "rating": 2000}}
    bots = FakeBots(game_states={"g1": [full]})
    bot, _ = build_bot(bots)
    bot.play_game("g1")
    assert bots.moves, "the bot moved"


def test_a_position_the_search_declines_gets_a_move_not_a_resignation() -> None:
    moves = "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"  # the start position, a third time
    bots = FakeBots(game_states={"g1": [_game_full(moves)]})
    bot, _ = build_bot(bots)
    bot.play_game("g1")
    assert bots.moves and not bots.resigned



def test_every_request_gets_a_read_timeout_so_a_dead_stream_cannot_hang() -> None:
    """tq8SjaTx and wczLemzW: the game stream went silent for ten minutes and the bot flagged."""
    import pytest

    from src.lichess.bot import READ_TIMEOUT_SECONDS, TimedTokenSession

    seen: Dict[str, Any] = {}

    def capture(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> None:
        seen.update(kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(requests.Session, "request", capture)
        TimedTokenSession("token").request("GET", "https://lichess.org/api/bot/game/stream/x", stream=True)
        assert seen["timeout"][1] == READ_TIMEOUT_SECONDS
        TimedTokenSession("token").request("GET", "https://x", timeout=5)
        assert seen["timeout"] == 5, "an explicit timeout wins"
    assert READ_TIMEOUT_SECONDS > 2 * 7.0, "must outlast keep-alives"



# --- psychology first against humans, safety first against bots ---------------


def _searched_with(bot: LichessBot) -> List[Any]:
    seen: List[Any] = []
    real = bot.searcher.search

    def capture(board: chess.Board, config: Any = None) -> Any:
        seen.append(config)
        return real(board, config)

    bot.searcher.search = capture  # type: ignore[method-assign]
    return seen


def test_humans_get_the_psychological_search_and_more_of_it_when_short_of_time() -> None:
    from src.engine.bot_factory import HUMAN_PLAY

    bot, _ = build_bot(FakeBots())
    seen = _searched_with(bot)
    session = bot._start_session("g1", _game_full())
    bot._advance(session, {"moves": "", "status": "started", "wtime": 300000, "btime": 300000})
    bot._advance(session, {"moves": "", "status": "started", "wtime": 300000, "btime": 20000})
    relaxed, short = seen
    assert relaxed.safety_threshold == HUMAN_PLAY["safety_threshold"]
    assert relaxed.narrow_path_weight == HUMAN_PLAY["narrow_path_weight"]
    assert short.narrow_path_weight == 2 * HUMAN_PLAY["narrow_path_weight"], "the human has 20s"
    assert relaxed.leaf_depth <= 6 and relaxed.root_depth <= 6, "not simply out-calculating them"


def test_bots_get_the_conservative_search_and_no_trap_book() -> None:
    from src.engine.bot_factory import SKEW, STANDARD

    bot, _ = build_bot(FakeBots())
    bot.books = {STANDARD.name: _Book(), SKEW.name: _Book()}  # type: ignore[assignment]
    seen = _searched_with(bot)
    engine = {**_game_full(), "black": {"id": "otherbot", "name": "OtherBot", "rating": 2000, "title": "BOT"}}
    session = bot._start_session("g1", engine)
    assert session.arm == "vs-bot" and bot.searcher.book is bot.books[SKEW.name]
    bot._advance(session, {"moves": "", "status": "started", "wtime": 300000, "btime": 20000})
    assert seen[0].narrow_path_weight == 0.0 and seen[0].winning_margin == 200



# --- move pacing against humans -------------------------------------------------


def test_pacing_classifies_moves_and_never_spends_the_bots_own_time() -> None:
    import random

    from src.lichess.time_manager import AdaptivePacingController, Cadence

    pace = AdaptivePacingController()
    calm = dict(forced=False, book=False, trap=False, quiet=True, utility=20.0, opponent_clock=120.0)
    assert pace.classify(**calm) is Cadence.DELIBERATE
    assert pace.classify(**{**calm, "opponent_clock": 12.0}) is Cadence.SNAP, "their scramble: reply at once"
    assert pace.classify(**{**calm, "forced": True}) is Cadence.SNAP
    assert pace.classify(**{**calm, "book": True}) is Cadence.SNAP
    assert pace.classify(**{**calm, "trap": True}) is Cadence.BAIT
    assert pace.classify(**{**calm, "utility": 600.0}) is Cadence.NONE, "not balanced"
    assert pace.classify(**{**calm, "quiet": False}) is Cadence.NONE

    rng = random.Random(0)
    for cadence in Cadence:
        assert pace.delay(cadence, elapsed=0.0, own_clock=9.9, budget=30.0, rng=rng) == 0.0, "own clock < 10s"
    for _ in range(50):
        wait = pace.delay(Cadence.DELIBERATE, elapsed=0.4, own_clock=120.0, budget=30.0, rng=rng)
        assert 1.0 - 0.4 <= wait <= 2.5 - 0.4, "ranges are total move time, search included"
        assert pace.delay(Cadence.SNAP, elapsed=0.4, own_clock=120.0, budget=30.0, rng=rng) == 0.0
    assert pace.delay(Cadence.DELIBERATE, elapsed=0.5, own_clock=120.0, budget=0.8, rng=rng) <= 0.3 + 1e-9, \
        "never past the move's time budget"


def test_humans_get_paced_moves_and_bots_get_them_at_once() -> None:
    def waits_in(game: Mapping[str, Any]) -> Tuple[List[float], Any]:
        bot, _ = build_bot(FakeBots())
        bot.config = BotConfig(pacing=True, pacing_share=1.0)
        waits: List[float] = []

        class Stop:
            def is_set(self) -> bool:
                return False

            def wait(self, timeout: float) -> bool:
                waits.append(timeout)
                return False

        bot._stop = Stop()  # type: ignore[assignment]
        session = bot._start_session("g1", game)
        bot._advance(session, {"moves": "", "status": "started", "wtime": 300000, "btime": 300000,
                               "winc": 3000, "binc": 3000})
        return waits, session

    human_waits, human = waits_in(_game_full())
    assert sum(human.cadence.values()) == 1, "every move's cadence is counted for the log"
    assert human_waits and all(0.0 < w <= 2.5 for w in human_waits), "a quiet opening move deliberates"
    engine = {**_game_full(), "black": {"id": "otherbot", "name": "OtherBot", "rating": 2000, "title": "BOT"}}
    bot_waits, versus_bot = waits_in(engine)
    assert bot_waits == [] and versus_bot.cadence == {}



def test_each_human_move_is_logged_with_its_telemetry_and_a_coin_flipped_wait(tmp_path: Any) -> None:
    import json
    import random

    log = tmp_path / "games.jsonl"
    bot, _ = build_bot(FakeBots())
    bot.config = BotConfig(pacing=True, pacing_share=0.5, game_log=log)
    bot._rng = random.Random(3)

    class Stop:
        def is_set(self) -> bool:
            return False

        def wait(self, timeout: float) -> bool:
            return False

    bot._stop = Stop()  # type: ignore[assignment]
    session = bot._start_session("g1", _game_full())
    paced = []
    for _ in range(40):
        bot._advance(session, {"moves": "", "status": "started", "wtime": 300000, "btime": 300000})
    moves = [json.loads(line) for line in log.read_text().splitlines() if '"move"' in line]
    assert len(moves) == 40 and {"cadence", "paced", "safe_replies", "utility", "trap"} <= set(moves[0])
    paced = [m["paced"] for m in moves]
    assert 10 < sum(paced) < 30, "about half the waits applied, half withheld"



def test_the_bot_holds_the_mac_awake_for_its_own_lifetime() -> None:
    """Four games were lost on time with the Mac asleep mid-game."""
    import pytest

    from src.lichess import bot as bridge

    launched: List[List[str]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("sys.platform", "darwin")
        patch.setattr("shutil.which", lambda name: "/usr/bin/caffeinate")
        patch.setattr("subprocess.Popen", lambda args: launched.append(args) or "proc")
        assert bridge.keep_awake(4242) == "proc"
        patch.setattr("sys.platform", "linux")
        assert bridge.keep_awake(4242) is None
    assert launched == [["caffeinate", "-i", "-s", "-w", "4242"]]
