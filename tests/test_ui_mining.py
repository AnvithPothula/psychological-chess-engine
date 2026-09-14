"""Tests for the mining tab and session autosave.

Headless via the SDL dummy driver. The mining run at the end drives the real
engines through the tab exactly as a click would.

    python -m tests.test_ui_mining
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import chess
import pygame

from src.engine.maia import MaiaEvaluator
from src.engine.search import AdversarialSearcher
from src.engine.stockfish import StockfishEvaluator
from src.ui.app import AppState, ChessApp, Tab
from src.ui.asset_manager import AssetManager
from src.ui.constants import SQUARE_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH
from src.ui.mining_view import (
    GAMES_CHOICES,
    PLIES_CHOICES,
    RATING_CHOICES,
    seconds_per_game,
)
from src.ui.session import SessionState, SessionStore
from tests.test_lichess import StubMaia
from tests.test_search import ScriptedEvaluator
from tests.test_ui import _display, _pump_until, _start_worker, _stop_worker


def _app(session: SessionStore, **kwargs: object) -> ChessApp:
    model = StubMaia()
    searcher = AdversarialSearcher(ScriptedEvaluator({}, {}, default_cp=0), model)
    app = ChessApp(searcher, AssetManager(), session=session, **kwargs)  # type: ignore[arg-type]
    app._build_views()
    return app


# --- session persistence ---------------------------------------------------


def test_no_path_means_no_disk() -> None:
    """Constructing the app must not resume a game the caller never asked for."""
    _display()
    store = SessionStore(path=None)
    store.state.pairs_mined_total = 99
    store.touch()
    store.flush()
    store.tick()
    assert store.path is None
    assert store.restore_board() is None

    app = _app(store)
    assert app._board.move_stack == [], "a fresh app starts a fresh game"
    assert not Path("build/session.json").with_suffix(".part").exists()


def test_session_round_trips_settings_and_counters() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.json"
        store = SessionStore(path)
        store.state.mining.rating = 1700
        store.state.mining.games = 120
        store.state.active_tab = "mine"
        store.state.pairs_mined_total = 41
        store.touch()
        store.flush()

        assert path.exists()
        reloaded = SessionStore(path)
        assert reloaded.state.mining.rating == 1700
        assert reloaded.state.mining.games == 120
        assert reloaded.state.active_tab == "mine"
        assert reloaded.state.pairs_mined_total == 41


def test_autosave_is_debounced_not_per_frame() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.json"
        store = SessionStore(path)
        store.touch()
        store.tick()  # inside the debounce window, so nothing is written yet
        assert not path.exists(), "a touch must not write immediately"
        store.flush()
        assert path.exists()

        first = path.stat().st_mtime_ns
        for _ in range(200):
            store.tick()
        assert path.stat().st_mtime_ns == first, "clean state must not rewrite the file"


def test_corrupt_or_stale_session_starts_fresh() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "broken.json"
        broken.write_text("{ not json")
        assert SessionStore(broken).state.mining.rating == SessionState().mining.rating

        stale = Path(tmp) / "stale.json"
        stale.write_text(json.dumps({"version": 999, "mining": {"rating": 1900}}))
        assert SessionStore(stale).state.mining.rating == SessionState().mining.rating


def test_game_autosaves_and_resumes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.json"
        store = SessionStore(path)
        board = chess.Board()
        for san in ("e4", "e5", "Nf3"):
            board.push_san(san)
        store.record_game(board, human_is_white=True)
        store.flush()

        resumed = SessionStore(path).restore_board()
        assert resumed is not None
        assert resumed.fen() == board.fen()
        assert [move.uci() for move in resumed.move_stack] == [m.uci() for m in board.move_stack]


def test_unreplayable_saved_game_is_discarded_not_fatal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.json"
        store = SessionStore(path)
        store.state.game_moves = ["e2e4", "e2e4"]  # the second is illegal
        store.touch()
        assert store.restore_board() is None
        assert store.state.game_moves == []


def test_finished_game_is_archived_as_pgn() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(Path(tmp) / "session.json")
        board = chess.Board()
        for san in ("f3", "e5", "g4", "Qh4#"):
            board.push_san(san)
        store.record_game(board, human_is_white=False)
        path = store.archive_pgn(board, human_is_white=False, directory=Path(tmp) / "games")

        assert path is not None and path.exists()
        text = path.read_text()
        assert '[Result "0-1"]' in text and "Qh4#" in text
        assert store.state.game_moves == [], "an archived game leaves the live slot"


# --- tab behaviour ---------------------------------------------------------


def test_tabs_switch_and_persist() -> None:
    _display()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.json"
        app = _app(SessionStore(path))
        assert app._tab.value == "play"

        assert app._handle_tab_click(app._tab_rects[Tab.MINE].center)
        assert app._tab.value == "mine"
        assert not app._handle_tab_click((5, WINDOW_HEIGHT - 5)), "clicks off the bar are not tabs"
        app.session.flush()

        assert SessionStore(path).state.active_tab == "mine"


def test_settings_snap_to_allowed_values_on_load() -> None:
    """A stored value the stepper cannot show must be corrected, not diverge."""
    _display()
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(Path(tmp) / "session.json")
        store.state.mining.games = 6      # below the minimum, not on the step
        store.state.mining.rating = 1234  # not a Maia band
        app = _app(store)

        view = app._mining_view
        assert app.session.state.mining.games == view.games.value
        assert app.session.state.mining.rating == view.rating.value
        assert view.rating.value in RATING_CHOICES
        assert view.games.value in GAMES_CHOICES
        assert view.plies.value in PLIES_CHOICES


def test_steppers_write_through_to_the_session() -> None:
    _display()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.json"
        app = _app(SessionStore(path))
        view = app._mining_view

        start = view.rating.value
        view.rating.handle_click(view.rating._plus.center)
        assert view.rating.value > start
        assert app.session.state.mining.rating == view.rating.value
        assert view.rating.value in RATING_CHOICES

        for _ in range(len(RATING_CHOICES) + 3):
            view.rating.handle_click(view.rating._plus.center)
        assert view.rating.value == RATING_CHOICES[-1], "stepping past the end must clamp"

        app.session.flush()
        assert SessionStore(path).state.mining.rating == RATING_CHOICES[-1]


def test_game_count_reaches_overnight_scale_in_few_clicks() -> None:
    """A linear +10 stepper needed 199 clicks to cross the range."""
    _display()
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(Path(tmp) / "session.json")
        store.state.mining.games = 10
        app = _app(store)
        view = app._mining_view

        clicks = 0
        while view.games.value < 10_000 and clicks < 30:
            view.games.handle_click(view.games._plus.center)
            clicks += 1
        assert view.games.value >= 10_000, "10k games must be reachable"
        assert clicks <= 12, f"and within a dozen clicks, took {clicks}"
        assert app.session.state.mining.games == view.games.value


def test_runtime_estimate_scales_with_the_settings() -> None:
    """The panel must not promise a coffee break for an overnight run."""
    _display()
    with tempfile.TemporaryDirectory() as tmp:
        app = _app(SessionStore(Path(tmp) / "session.json"))
        view = app._mining_view

        view.settings.games, view.settings.max_plies, view.settings.rating = 10_000, 30, 1100
        cheap_hours, cheap_pairs = view.estimate()
        view.settings.max_plies = 80
        long_hours, _ = view.estimate()
        view.settings.max_plies, view.settings.games = 30, 100
        small_hours, small_pairs = view.estimate()

        assert long_hours > cheap_hours * 2, "a higher ply cap must cost visibly more"
        assert small_hours < cheap_hours and small_pairs < cheap_pairs
        # 1100 is the slowest band at 3.92s/game, so 10k games really is ~11h.
        assert 5.0 < cheap_hours < 15.0, f"10k games at 30 plies should be hours, got {cheap_hours}"


def test_board_stays_playable_while_mining() -> None:
    """The whole point of separate processes: the game keeps working."""
    _display()
    with tempfile.TemporaryDirectory() as tmp:
        app = _app(SessionStore(Path(tmp) / "session.json"))
        app._begin()
        import threading as _threading

        gate = _threading.Event()
        app._mining_thread = _threading.Thread(target=gate.wait, daemon=True)
        app._mining_thread.start()
        try:
            assert app._is_mining
            app._handle_click(app._board_view.square_rect(chess.E2).center)
            assert app._selected == chess.E2, "the board must stay live during a rollout"
        finally:
            gate.set()
            app._mining_thread.join(timeout=5.0)


def test_board_clicks_are_ignored_on_the_mining_tab() -> None:
    _display()
    with tempfile.TemporaryDirectory() as tmp:
        app = _app(SessionStore(Path(tmp) / "session.json"))
        app._begin()
        app._handle_tab_click(app._tab_rects[Tab.MINE].center)

        pygame.event.post(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=app._board_view.square_rect(chess.E2).center, button=1
        ))
        app._handle_events()
        assert app._selected is None, "the mining tab must not select pieces"
        assert app._board.move_stack == []


def test_mining_does_not_block_on_the_game() -> None:
    """Separate processes: an engine mid-search is no reason to refuse a rollout."""
    _display()
    with tempfile.TemporaryDirectory() as tmp:
        app = _app(SessionStore(Path(tmp) / "session.json"))
        app._state = AppState.ENGINE_THINKING
        assert app._mining_blocker() == "", "the game must not gate mining"
        app._state = AppState.HUMAN_TURN
        assert app._mining_blocker() == ""


def test_only_one_rollout_runs_at_a_time() -> None:
    _display()
    with tempfile.TemporaryDirectory() as tmp:
        app = _app(SessionStore(Path(tmp) / "session.json"))
        import threading as _threading

        gate = _threading.Event()
        app._mining_thread = _threading.Thread(target=gate.wait, daemon=True)
        app._mining_thread.start()
        try:
            assert "already running" in app._mining_blocker()
        finally:
            gate.set()
            app._mining_thread.join(timeout=5.0)


def test_rollout_observer_drops_frames_rather_than_blocking() -> None:
    _display()
    from src.training.dpo_generator import RolloutUpdate

    with tempfile.TemporaryDirectory() as tmp:
        app = _app(SessionStore(Path(tmp) / "session.json"))
        update = RolloutUpdate(0, 1, chess.STARTING_FEN, None, 0, 1500, 1500, 0.0, 0, 0.0, 0, "playing")
        started = time.perf_counter()
        for _ in range(2000):
            app._observe_rollout(update)
        assert time.perf_counter() - started < 1.0, "a full queue must not stall the rollout"

        app._drain_rollout()
        assert app._mining_view.latest is not None
        assert app._rollout.empty()


# --- end to end ------------------------------------------------------------


def test_mining_tab_runs_a_real_rollout_and_persists_the_totals() -> None:
    screen = _display()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        path = directory / "session.json"
        output = directory / "pairs.jsonl"

        store = SessionStore(path)
        store.state.mining.games = 10
        store.state.mining.max_plies = 30
        store.state.mining.rating = 1100
        store.state.mining.output = str(output)

        with StockfishEvaluator() as stockfish, MaiaEvaluator(rating=1100) as maia:
            searcher = AdversarialSearcher(stockfish, maia)
            app = ChessApp(searcher, AssetManager(), session=store)
            app._build_views()
            worker = _start_worker(app)
            try:
                app._begin()
                app._handle_tab_click(app._tab_rects[Tab.MINE].center)
                app._mining_view.action.handle_click(app._mining_view.action.rect.center)
                # The generator appears only once the worker has booted its own
                # engine pair, so the thread is what is true immediately.
                assert app._mining_thread is not None and app._mining_thread.is_alive()
                assert app._mining_view.running

                assert _pump_until(app, screen, lambda: not app._is_mining, 240.0), "rollout timed out"
                app._drain_rollout()
            finally:
                app._stop_mining()
                if app._mining_thread is not None:
                    app._mining_thread.join(timeout=30.0)
                _stop_worker(app, worker)

        assert not app._mining_view.running, "the tab must return to idle when the run ends"
        assert store.state.games_mined_total == 10
        store.flush()

        reloaded = SessionStore(path)
        assert reloaded.state.games_mined_total == 10
        assert reloaded.state.pairs_mined_total == store.state.pairs_mined_total

        if output.exists():
            for line in output.read_text().splitlines():
                record = json.loads(line)
                board = chess.Board(record["fen"])
                assert board.parse_san(record["chosen"]) in board.legal_moves
        print(f"    mined {store.state.pairs_mined_total} pairs over 10 games via the tab")


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
    pygame.quit()
    print("all green" if not failures else f"{failures} failing test(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())


def test_weak_opponents_cost_more_time_per_game() -> None:
    """The 22,000-game run measured 1100 at 3.92s/game and 1900 at 1.50s.

    An earlier fit scaled the other way and under-predicted 1100 by 3x, which is
    the difference between an evening and an overnight run.
    """
    costs = [seconds_per_game(rating) for rating in RATING_CHOICES]
    assert costs == sorted(costs, reverse=True), costs

    for rating, measured in ((1100, 3.92), (1500, 2.16), (1700, 1.58), (1900, 1.50)):
        assert abs(seconds_per_game(rating) - measured) < 0.01

    assert seconds_per_game(900) == seconds_per_game(1100)
    assert seconds_per_game(2500) == seconds_per_game(1900)
