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

    def stream_incoming_events(self) -> Iterator[Mapping[str, Any]]:
        self.event_stream_calls += 1
        if self.raise_on_event_stream:
            failure = self.raise_on_event_stream.pop(0)
            if failure is not None:
                raise failure
        yield from self.events

    def stream_game_state(self, game_id: str) -> Iterator[Mapping[str, Any]]:
        yield from self.game_states.get(game_id, [])

    def make_move(self, game_id: str, move: str) -> None:
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
    bot = LichessBot(FakeClient(bots), searcher, model, bot_id=BOT_ID)
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

    assert model.ratings_loaded == [1600], "1630 must map to the maia-1600 checkpoint"
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
