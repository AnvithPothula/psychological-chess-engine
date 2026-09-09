"""Headless checks for the Pygame front-end.

Runs without a display via the SDL ``dummy`` video driver, so it works over SSH
and in CI. The rendering test drives the real engines end to end and writes a
screenshot, which doubles as a visual check.

    python -m tests.test_ui
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import chess
import pygame

from src.engine.maia import MaiaEvaluator
from src.engine.search import AdversarialSearcher
from src.engine.stockfish import StockfishEvaluator
from src.types import SearchConfig
from src.ui.app import AppState, ChessApp
from src.ui.asset_manager import PIECE_SYMBOLS, AssetManager
from src.ui.board_view import BoardView
from src.ui.constants import SQUARE_SIZE, WINDOW_HEIGHT, WINDOW_WIDTH, load_fonts
from tests.test_search import ScriptedEvaluator, ScriptedHumanModel

SCREENSHOT_PATH = Path("build/ui_screenshot.png")
ENGINE_REPLY_TIMEOUT = 30.0


def _display() -> pygame.Surface:
    if not pygame.get_init():
        pygame.init()
    surface = pygame.display.get_surface()
    if surface is None:
        surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    return surface


def _pump_until(
    app: ChessApp, screen: pygame.Surface, predicate: Callable[[], bool], timeout: float
) -> bool:
    """Drive the outcome queue and renderer until ``predicate`` holds."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        app._drain_outcomes()
        app._render(screen)
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _start_worker(app: ChessApp) -> threading.Thread:
    worker = threading.Thread(target=app._engine_worker, name="test-engine-worker", daemon=True)
    worker.start()
    return worker


def _stop_worker(app: ChessApp, worker: threading.Thread) -> None:
    app._jobs.put(None)
    worker.join(timeout=10.0)


def test_asset_manager_produces_every_piece_offline() -> None:
    _display()
    manager = AssetManager()
    assert manager.ensure_assets() == (), "all pieces must be obtainable without a network"
    assert manager.missing_symbols() == ()

    surfaces = manager.load_pieces(SQUARE_SIZE)
    assert set(surfaces) == set(PIECE_SYMBOLS)
    for symbol, surface in surfaces.items():
        assert surface.get_size() == (SQUARE_SIZE, SQUARE_SIZE)
        opaque = sum(
            1
            for x in range(0, SQUARE_SIZE, 4)
            for y in range(0, SQUARE_SIZE, 4)
            if surface.get_at((x, y)).a > 0
        )
        assert opaque > 0, f"{symbol} rendered fully transparent"


def test_square_at_round_trips_in_both_orientations() -> None:
    _display()
    fonts = load_fonts()
    pieces = AssetManager().load_pieces(SQUARE_SIZE)
    for orientation in (chess.WHITE, chess.BLACK):
        view = BoardView(pieces, fonts.coordinate, orientation=orientation)
        for square in chess.SQUARES:
            assert view.square_at(view.square_rect(square).center) == square
        assert view.square_at((-5, -5)) is None
        assert view.square_at((WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1)) is None

    white_view = BoardView(pieces, fonts.coordinate, orientation=chess.WHITE)
    black_view = BoardView(pieces, fonts.coordinate, orientation=chess.BLACK)
    assert white_view.square_rect(chess.A1).bottomleft == black_view.square_rect(chess.H8).bottomleft


def test_click_to_move_selects_then_commits_and_locks_input() -> None:
    """Two clicks play a move; further clicks are ignored while the engine thinks."""
    _display()
    board = chess.Board()
    after_e4 = board.copy()
    after_e4.push(chess.Move.from_uci("e2e4"))

    # The engine plays Black here, and ScriptedEvaluator speaks White-relative
    # centipawns, so the move we want it to choose must be the most negative.
    evaluator = ScriptedEvaluator(root_scores={"e7e5": -900}, leaf_scores={}, default_cp=0)
    human = ScriptedHumanModel({})
    app = ChessApp(AdversarialSearcher(evaluator, human), AssetManager())
    app._build_views()
    worker = _start_worker(app)
    screen = _display()
    try:
        app._begin()
        assert app._state is AppState.HUMAN_TURN, "human moves first as White"

        # First click selects and exposes the legal destinations.
        app._handle_click(app._board_view.square_rect(chess.E2).center)
        assert app._selected == chess.E2, "first click must select the pawn"
        assert {move.to_square for move in app._legal_moves} == {chess.E3, chess.E4}

        # Clicking an empty, illegal square clears the selection instead of moving.
        app._handle_click(app._board_view.square_rect(chess.A5).center)
        assert app._selected is None and app._board.move_stack == []

        app._handle_click(app._board_view.square_rect(chess.E2).center)
        app._handle_click(app._board_view.square_rect(chess.E4).center)
        assert app._board.fen() == after_e4.fen(), "second click must commit the move"
        assert app._state is AppState.ENGINE_THINKING, "engine takes the turn immediately"
        assert app._selected is None
        assert app._visible_hints() == (), "no move hints while the engine owns the turn"

        # Input is dropped for as long as the engine holds the turn.
        app._handle_click(app._board_view.square_rect(chess.D7).center)
        assert app._selected is None

        assert _pump_until(app, screen, lambda: app._state is AppState.HUMAN_TURN, ENGINE_REPLY_TIMEOUT)
        assert app._report is not None and app._report.san == "e5"
        assert app._last_move == chess.Move.from_uci("e7e5")
    finally:
        _stop_worker(app, worker)


def test_promotion_click_auto_queens() -> None:
    _display()
    board = chess.Board("8/4P3/8/8/8/8/8/4K1k1 w - - 0 1")
    app = ChessApp(
        AdversarialSearcher(ScriptedEvaluator({}, {}), ScriptedHumanModel({})),
        AssetManager(),
        board=board,
    )
    app._build_views()
    move = app._resolve_move(chess.E7, chess.E8)
    assert move is not None, "e7-e8 must resolve to a legal promotion"
    assert move.promotion == chess.QUEEN, f"expected auto-queen, got {move.uci()}"


def test_engine_failure_is_surfaced_not_swallowed() -> None:
    """A worker exception must reach the panel, not hang the UI in THINKING."""
    _display()

    class BrokenEvaluator(ScriptedEvaluator):
        def analyse_root_moves(self, board, *, depth=12, multipv=None):  # type: ignore[no-untyped-def]
            raise ValueError("simulated engine crash")

    app = ChessApp(
        AdversarialSearcher(BrokenEvaluator({}, {}), ScriptedHumanModel({})),
        AssetManager(),
        human_color=chess.BLACK,
    )
    app._build_views()
    worker = _start_worker(app)
    screen = _display()
    try:
        app._begin()  # engine is White, so it moves first and fails
        assert _pump_until(app, screen, lambda: app._message is not None, 10.0)
        assert app._state is AppState.GAME_OVER
        assert "simulated engine crash" in (app._message or "")
    finally:
        _stop_worker(app, worker)


def test_full_frame_renders_against_the_real_engines() -> None:
    """End-to-end: human plays 1.e4, the engine replies, the frame is captured."""
    screen = _display()
    with StockfishEvaluator() as stockfish, MaiaEvaluator(rating=1100) as maia:
        searcher = AdversarialSearcher(stockfish, maia, config=SearchConfig())
        app = ChessApp(searcher, AssetManager(), opponent_label="Opponent model: Maia-1100")
        app._build_views()
        worker = _start_worker(app)
        try:
            app._begin()
            app._handle_click(app._board_view.square_rect(chess.E2).center)
            app._handle_click(app._board_view.square_rect(chess.E4).center)
            assert _pump_until(
                app, screen, lambda: app._state is AppState.HUMAN_TURN, ENGINE_REPLY_TIMEOUT
            )
            assert app._report is not None
            assert app._evaluation is not None, "eval bar never received a score"

            # Select a piece so the frame also shows the translucent move hints.
            app._handle_click(app._board_view.square_rect(chess.G1).center)
            app._render(screen)

            SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(screen, str(SCREENSHOT_PATH))
            print(f"    engine replied {app._report.san}: {app._report.result.summary()}")
            print(f"    screenshot -> {SCREENSHOT_PATH}")
            assert SCREENSHOT_PATH.stat().st_size > 0
        finally:
            _stop_worker(app, worker)


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
    pygame.quit()
    print("all green" if not failures else f"{failures} failing test(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
