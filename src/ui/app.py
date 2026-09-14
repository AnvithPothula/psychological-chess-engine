"""Pygame application: event loop, engine threading and state machine.

**Threading model.** Exactly one worker thread touches the engines, ever.
``chess.engine.SimpleEngine`` is not reentrant -- it multiplexes one UCI process
over a background event loop -- so two threads issuing commands to the same
process interleave their protocol traffic and corrupt both. The eval bar and the
adversarial search are two separate engine consumers, so they are serialised
onto one worker through a FIFO job queue rather than given a thread each.

**Mining owns its own engine processes.** A rollout runs concurrently with the
game, which it can only do safely because it does not share the board's
Stockfish and Lc0. Sharing them would break two ways that have nothing to do
with load: ``SimpleEngine`` is not reentrant, so two threads interleave their
UCI traffic on one process; and the rollout calls ``set_rating``, which swaps
Maia's weights and forces a ``ucinewgame`` underneath whatever game is in
progress. A second pair costs about a second to start and a few hundred MB, and
buys a genuinely independent worker.

**Why there is no mutex around the board.** Every job carries its *own copy* of
the position. The worker never reads the live ``chess.Board``, so there is no
shared mutable state to guard: the queue is the synchronisation primitive.
Human input is gated by the :class:`AppState` machine, which is a UI concern
(ignore clicks while thinking), not a memory-safety one.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Final, Optional, Sequence, Tuple, Union

import chess
import pygame

from src.engine import EvaluatorError
from src.engine.search import AdversarialSearcher
from src.training.dpo_generator import (
    DPOGenerator,
    GeneratorConfig,
    RolloutUpdate,
    run_with_restarts,
)
from src.types import EngineEval, SearchConfig, SearchResult
from src.ui.asset_manager import AssetManager
from src.ui.board_view import BoardView
from src.ui.mining_view import MiningView
from src.ui.session import MiningSettings, SessionStore
from src.ui.constants import (
    ACCENT_NEGATIVE,
    ACCENT_THINKING,
    BOARD_SIZE,
    EVAL_BAR_DEPTH,
    EVAL_BAR_WIDTH,
    EVAL_BAR_X,
    FPS,
    PANEL_WIDTH,
    PANEL_X,
    SQUARE_SIZE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WINDOW_BG,
    WINDOW_HEIGHT,
    WINDOW_MARGIN,
    ROLLOUT_QUEUE_LIMIT,
    TAB_ACTIVE_BG,
    TAB_BAR_HEIGHT,
    TAB_CORNER_RADIUS,
    TAB_GAP,
    TAB_IDLE_BG,
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WORKER_JOIN_TIMEOUT,
    RGB,
    load_fonts,
)
from src.ui.telemetry_view import EvalBar, MoveReport, TelemetryView

__all__ = ["AppState", "ChessApp", "Tab"]

logger = logging.getLogger(__name__)

MOUSE_BUTTON_LEFT: Final[int] = 1


class Tab(Enum):
    """Which panel the side bar is showing."""

    PLAY = "play"
    MINE = "mine"


class AppState(Enum):
    """Whose turn it is, from the UI's point of view."""

    HUMAN_TURN = auto()
    ENGINE_THINKING = auto()
    GAME_OVER = auto()


class JobKind(Enum):
    EVALUATE = auto()
    """Refresh the eval bar. Cheap and shallow."""

    SEARCH = auto()
    """Full adversarial search. Produces the engine's move."""


@dataclass(frozen=True, slots=True)
class EngineJob:
    kind: JobKind
    board: chess.Board
    """A private copy. The worker must never see the live board."""


@dataclass(frozen=True, slots=True)
class EvaluationReady:
    evaluation: EngineEval


@dataclass(frozen=True, slots=True)
class SearchReady:
    result: SearchResult


@dataclass(frozen=True, slots=True)
class EngineFailed:
    message: str


EngineOutcome = Union[EvaluationReady, SearchReady, EngineFailed]


class ChessApp:
    """The window, the game state, and the bridge to the engine worker."""

    _board_view: BoardView
    _eval_bar: EvalBar
    _telemetry: TelemetryView
    _mining_view: MiningView
    _tab_font: pygame.font.Font

    def __init__(
        self,
        searcher: AdversarialSearcher,
        assets: AssetManager,
        *,
        human_color: chess.Color = chess.WHITE,
        opponent_label: str = "Maia",
        board: Optional[chess.Board] = None,
        session: Optional[SessionStore] = None,
    ) -> None:
        self.searcher = searcher
        self.assets = assets
        self.human_color = human_color
        self.opponent_label = opponent_label
        # No session means no disk: constructing the app must not resume a
        # game the caller never asked for.
        self.session = session if session is not None else SessionStore(path=None)

        resumed = self.session.restore_board() if board is None else None
        if resumed is not None:
            logger.info("ui: resuming the saved game after %d moves", len(resumed.move_stack))
            self.human_color = chess.WHITE if self.session.state.human_is_white else chess.BLACK
        self._board = (board.copy() if board is not None else resumed) or chess.Board()
        self._state = AppState.HUMAN_TURN
        self._selected: Optional[chess.Square] = None
        self._legal_moves: Tuple[chess.Move, ...] = ()
        self._last_move: Optional[chess.Move] = None
        self._evaluation: Optional[EngineEval] = None
        self._report: Optional[MoveReport] = None
        self._message: Optional[str] = None
        self._thinking_since = 0.0
        self._running = True

        self._tab = self._restore_tab(self.session.state.active_tab)
        self._tab_rects: Dict[Tab, pygame.Rect] = {}
        self._mining: Optional[DPOGenerator] = None
        self._mining_thread: Optional[threading.Thread] = None
        self._mining_stop = threading.Event()
        self._mining_error: Optional[str] = None
        self._mining_board = chess.Board()
        self._rollout: "queue.Queue[RolloutUpdate]" = queue.Queue(maxsize=ROLLOUT_QUEUE_LIMIT)

        self._jobs: "queue.Queue[Optional[EngineJob]]" = queue.Queue()
        self._outcomes: "queue.Queue[EngineOutcome]" = queue.Queue()

    @staticmethod
    def _restore_tab(name: str) -> Tab:
        try:
            return Tab(name)
        except ValueError:
            return Tab.PLAY

    # -- lifecycle ----------------------------------------------------------

    def run(self) -> None:
        """Open the window and run until the user closes it."""
        pygame.init()
        try:
            screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption(WINDOW_TITLE)
            self._build_views()

            worker = threading.Thread(target=self._engine_worker, name="engine-worker", daemon=True)
            worker.start()
            logger.info("ui: window %dx%d, engine worker started", WINDOW_WIDTH, WINDOW_HEIGHT)
            try:
                self._begin()
                self._main_loop(screen)
            finally:
                self._stop_worker(worker)
        finally:
            pygame.quit()

    def _build_views(self) -> None:
        fonts = load_fonts()
        pieces = self.assets.load_pieces(SQUARE_SIZE)
        self._board_view = BoardView(pieces, fonts.coordinate, orientation=self.human_color)
        self._eval_bar = EvalBar(pygame.Rect(EVAL_BAR_X, WINDOW_MARGIN, EVAL_BAR_WIDTH, BOARD_SIZE))

        bar_top = WINDOW_MARGIN
        panel_top = bar_top + TAB_BAR_HEIGHT + TAB_GAP
        panel = pygame.Rect(PANEL_X, panel_top, PANEL_WIDTH, BOARD_SIZE - TAB_BAR_HEIGHT - TAB_GAP)
        tab_width = (PANEL_WIDTH - TAB_GAP) // 2
        self._tab_rects = {
            Tab.PLAY: pygame.Rect(PANEL_X, bar_top, tab_width, TAB_BAR_HEIGHT),
            Tab.MINE: pygame.Rect(PANEL_X + tab_width + TAB_GAP, bar_top, tab_width, TAB_BAR_HEIGHT),
        }
        self._tab_font = fonts.heading
        self._telemetry = TelemetryView(panel, fonts)
        self._mining_view = MiningView(
            panel, fonts, self.session.state.mining,
            on_start=self._start_mining, on_stop=self._stop_mining,
            on_settings_changed=self.session.touch,
        )
        self._mining_view.lifetime_pairs = self.session.state.pairs_mined_total
        self._mining_view.lifetime_games = self.session.state.games_mined_total

    def _begin(self) -> None:
        self._request_evaluation()
        if self._board.is_game_over(claim_draw=True):
            self._state = AppState.GAME_OVER
        elif self._board.turn == self.human_color:
            self._state = AppState.HUMAN_TURN
        else:
            self._start_search()

    def _main_loop(self, screen: pygame.Surface) -> None:
        clock = pygame.time.Clock()
        while self._running:
            self._handle_events()
            self._drain_outcomes()
            self._drain_rollout()
            self.session.tick()
            self._render(screen)
            pygame.display.flip()
            clock.tick(FPS)

    def _stop_worker(self, worker: threading.Thread) -> None:
        self._stop_mining()
        if self._mining_thread is not None:
            self._mining_thread.join(timeout=WORKER_JOIN_TIMEOUT)
        self.session.flush()
        self._jobs.put(None)
        worker.join(timeout=WORKER_JOIN_TIMEOUT)
        if worker.is_alive():
            # The worker is blocked inside a UCI call. It is a daemon thread and
            # the caller's context manager is about to kill the engine processes,
            # which unblocks it; nothing is left to clean up here.
            logger.warning("ui: engine worker still busy after %.0fs, abandoning it", WORKER_JOIN_TIMEOUT)

    # -- engine worker (background thread) ----------------------------------

    def _engine_worker(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                self._outcomes.put(self._run_job(job))
            finally:
                self._jobs.task_done()

    def _run_job(self, job: EngineJob) -> EngineOutcome:
        try:
            if job.kind is JobKind.EVALUATE:
                return EvaluationReady(self.searcher.evaluator.evaluate(job.board, EVAL_BAR_DEPTH))
            return SearchReady(self.searcher.search(job.board))
        except (EvaluatorError, chess.engine.EngineError, ValueError) as exc:
            logger.error("engine: %s job failed: %s", job.kind.name.lower(), exc)
            return EngineFailed(f"Engine error: {exc}")
        except Exception as exc:  # noqa: BLE001
            # A worker thread that dies silently leaves the UI waiting forever,
            # so every failure is reported rather than raised.
            logger.exception("engine: unexpected failure in %s job", job.kind.name.lower())
            return EngineFailed(f"Unexpected engine failure: {exc}")

    # -- outcomes (main thread) ---------------------------------------------

    def _drain_outcomes(self) -> None:
        while True:
            try:
                outcome = self._outcomes.get_nowait()
            except queue.Empty:
                return
            if isinstance(outcome, EvaluationReady):
                self._evaluation = outcome.evaluation
            elif isinstance(outcome, SearchReady):
                self._apply_engine_move(outcome.result)
            else:
                self._fail(outcome.message)

    def _apply_engine_move(self, result: SearchResult) -> None:
        if result.move not in self._board.legal_moves:
            self._fail(f"Engine returned an illegal move: {result.move.uci()}")
            return

        san = self._board.san(result.move)
        self._board.push(result.move)
        self._last_move = result.move

        # The board now sits at the position the predicted replies belong to, so
        # this is the only moment their SAN can be resolved.
        chosen = next((c for c in result.candidates if c.move == result.move), None)
        reply_sans = tuple(self._board.san(reply.move) for reply in chosen.top_replies) if chosen else ()
        self._report = MoveReport(san=san, result=result, chosen=chosen, reply_sans=reply_sans)
        self.session.record_game(self._board, self.human_color == chess.WHITE)
        logger.info("ui: engine played %s (%s)", san, result.summary())

        if self._finish_if_over():
            return
        self._state = AppState.HUMAN_TURN
        self._request_evaluation()

    def _fail(self, message: str) -> None:
        self._message = message
        self._state = AppState.GAME_OVER

    # -- mining -------------------------------------------------------------

    def _mining_blocker(self) -> str:
        """Why a rollout cannot start right now; empty when it can."""
        if self._mining_thread is not None and self._mining_thread.is_alive():
            return "A rollout is already running."
        return ""

    def _start_mining(self) -> None:
        blocker = self._mining_blocker()
        if blocker:
            logger.info("ui: cannot start mining (%s)", blocker)
            return

        settings = self.session.state.mining
        self._mining_stop.clear()
        self._mining_error = None
        self._mining_view.latest = None
        self._mining_view.set_running(True)
        self._mining_thread = threading.Thread(
            target=self._run_mining,
            args=(replace(settings), Path(settings.output)),
            name="ui-mining",
            daemon=True,
        )
        self._mining_thread.start()
        logger.info(
            "ui: mining %d games against maia-%d into %s",
            settings.games, settings.rating, settings.output,
        )

    def _run_mining(self, settings: MiningSettings, output: Path) -> None:
        """Rollout worker, on engine processes the supervisor opens and closes.

        A ten-thousand-game run spans hours, and a child engine dying in that
        window used to discard every game still to come. The supervisor respawns
        and resumes instead; pairs are flushed per game, so a restart loses at
        most the game in flight.
        """
        try:
            def adopt(generator: DPOGenerator) -> None:
                # Each restart builds a new generator, so the stop button has to
                # be re-pointed at whichever one is currently running.
                self._mining = generator
                if self._mining_stop.is_set():
                    generator.stop()

            stats = run_with_restarts(
                GeneratorConfig(
                    games=settings.games,
                    opponent_rating=settings.rating,
                    max_plies=settings.max_plies,
                    search_config=SearchConfig(
                        root_depth=8, leaf_depth=8, max_candidates=5, max_replies=5
                    ),
                ),
                output,
                observer=self._observe_rollout,
                on_generator=adopt,
            )

            self.session.state.pairs_mined_total += stats.pairs
            self.session.state.games_mined_total += stats.games
            self.session.touch()
            if stats.aborted:
                self._mining_error = f"stopped after {stats.games}/{settings.games} games"
            logger.info(
                "ui: mining finished, %d pairs from %d games (%d engine restarts)",
                stats.pairs, stats.games, stats.restarts,
            )
        except (EvaluatorError, FileNotFoundError, OSError) as exc:
            logger.error("ui: mining failed (%s)", exc)
            self._mining_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - a dead worker must not freeze the tab
            logger.exception("ui: unexpected mining failure (%s)", exc)
            self._mining_error = str(exc)
        finally:
            self._mining = None

    def _stop_mining(self) -> None:
        # Set before checking the generator: the worker may still be starting
        # its engines, and a stop issued in that window must not be lost.
        self._mining_stop.set()
        generator = self._mining
        if generator is not None:
            logger.info("ui: stopping the rollout after the current move")
            generator.stop()

    def _observe_rollout(self, update: RolloutUpdate) -> None:
        """Called from the rollout thread; must never block it."""
        try:
            self._rollout.put_nowait(update)
        except queue.Full:
            pass  # Dropping a frame is correct; stalling the rollout is not.

    def _drain_rollout(self) -> None:
        latest: Optional[RolloutUpdate] = None
        while True:
            try:
                latest = self._rollout.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._mining_view.latest = latest
            try:
                self._mining_board.set_fen(latest.fen)
            except ValueError:
                logger.debug("ui: ignoring an unparseable rollout fen")

        alive = self._mining_thread is not None and self._mining_thread.is_alive()
        if self._mining_view.running and not alive:
            self._mining_view.set_running(False)
            self._mining = None
            self._mining_view.lifetime_pairs = self.session.state.pairs_mined_total
            self._mining_view.lifetime_games = self.session.state.games_mined_total
        if self._mining_error is not None:
            self._mining_view.set_blocked(f"Last run failed: {self._mining_error}")
        else:
            self._mining_view.set_blocked(
                self._mining_blocker() if not self._mining_view.running else ""
            )

    @property
    def _is_mining(self) -> bool:
        return self._mining_thread is not None and self._mining_thread.is_alive()

    # -- input --------------------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.type == pygame.MOUSEMOTION:
                if self._tab is Tab.MINE:
                    self._mining_view.handle_motion(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == MOUSE_BUTTON_LEFT:
                if self._handle_tab_click(event.pos):
                    continue
                if self._tab is Tab.MINE:
                    self._mining_view.handle_click(event.pos)
                    continue
                # Only the engine's own turn blocks the board; a rollout runs
                # on separate processes and does not.
                if self._state is AppState.HUMAN_TURN:
                    self._handle_click(event.pos)

    def _handle_tab_click(self, position: Tuple[int, int]) -> bool:
        for tab, rect in self._tab_rects.items():
            if rect.collidepoint(position):
                if tab is not self._tab:
                    self._tab = tab
                    self.session.state.active_tab = tab.value
                    self.session.touch()
                    self._clear_selection()
                return True
        return False

    def _handle_click(self, position: Tuple[int, int]) -> None:
        square = self._board_view.square_at(position)
        if square is None:
            self._clear_selection()
            return
        if self._selected is not None:
            move = self._resolve_move(self._selected, square)
            if move is not None:
                self._play_human_move(move)
                return
        self._select(square)

    def _select(self, square: chess.Square) -> None:
        piece = self._board.piece_at(square)
        if piece is None or piece.color != self.human_color or self._board.turn != self.human_color:
            self._clear_selection()
            return
        self._selected = square
        self._legal_moves = tuple(
            move for move in self._board.legal_moves if move.from_square == square
        )

    def _resolve_move(self, source: chess.Square, target: chess.Square) -> Optional[chess.Move]:
        """The legal move joining two squares, auto-promoting to a queen.

        Under-promotion is not reachable by click alone; it needs a promotion
        picker, which the classic two-click flow has no room for.
        """
        options = [
            move
            for move in self._board.legal_moves
            if move.from_square == source and move.to_square == target
        ]
        if not options:
            return None
        return next((move for move in options if move.promotion == chess.QUEEN), options[0])

    def _clear_selection(self) -> None:
        self._selected = None
        self._legal_moves = ()

    def _play_human_move(self, move: chess.Move) -> None:
        logger.info("ui: human played %s", self._board.san(move))
        self._board.push(move)
        self._last_move = move
        self._clear_selection()
        self.session.record_game(self._board, self.human_color == chess.WHITE)
        if self._finish_if_over():
            return
        self._request_evaluation()
        self._start_search()

    # -- engine turn --------------------------------------------------------

    def _start_search(self) -> None:
        self._state = AppState.ENGINE_THINKING
        self._thinking_since = time.perf_counter()
        self._jobs.put(EngineJob(JobKind.SEARCH, self._board.copy()))

    def _request_evaluation(self) -> None:
        if not self._board.is_game_over(claim_draw=True):
            self._jobs.put(EngineJob(JobKind.EVALUATE, self._board.copy()))

    def _finish_if_over(self) -> bool:
        if not self._board.is_game_over(claim_draw=True):
            return False
        self._state = AppState.GAME_OVER
        # A finished game moves out of the live slot into its own PGN, so the
        # next launch starts fresh rather than resuming a decided position.
        self.session.archive_pgn(self._board, self.human_color == chess.WHITE)
        return True

    # -- rendering ----------------------------------------------------------

    def _render(self, screen: pygame.Surface) -> None:
        screen.fill(WINDOW_BG)
        # The mining tab mirrors the rollout rather than the human's game, so
        # the board is always showing whatever the tab is about.
        rollout = self._mining_view.latest if self._tab is Tab.MINE else None
        showing_rollout = rollout is not None
        board = self._mining_board if showing_rollout else self._board
        last_move = rollout.last_move if rollout is not None else self._last_move
        self._board_view.draw(
            screen,
            board,
            selected=None if showing_rollout else self._selected,
            legal_moves=() if showing_rollout else self._visible_hints(),
            last_move=last_move,
        )
        self._eval_bar.draw(screen, self._evaluation, self.human_color)
        self._draw_tabs(screen)

        if self._tab is Tab.MINE:
            self._mining_view.draw(screen)
            return
        status_text, status_colour = self._status()
        self._telemetry.draw(
            screen,
            status_text=status_text,
            status_colour=status_colour,
            report=self._report,
            opponent_label=self.opponent_label,
            evaluation=self._evaluation,
            message=self._message,
        )

    def _draw_tabs(self, screen: pygame.Surface) -> None:
        labels = {Tab.PLAY: "Play", Tab.MINE: "Mine"}
        for tab, rect in self._tab_rects.items():
            active = tab is self._tab
            pygame.draw.rect(
                screen, TAB_ACTIVE_BG if active else TAB_IDLE_BG, rect,
                border_radius=TAB_CORNER_RADIUS,
            )
            caption = labels[tab]
            if tab is Tab.MINE and self._is_mining:
                caption = "Mine ●"
            label = self._tab_font.render(
                caption, True, TEXT_PRIMARY if active else TEXT_MUTED
            )
            screen.blit(label, label.get_rect(center=rect.center))

    def _visible_hints(self) -> Sequence[chess.Move]:
        return self._legal_moves if self._state is AppState.HUMAN_TURN else ()

    def _status(self) -> Tuple[str, RGB]:
        if self._message is not None:
            return "Engine unavailable", ACCENT_NEGATIVE
        if self._state is AppState.GAME_OVER:
            return self._outcome_text(), TEXT_MUTED
        if self._state is AppState.ENGINE_THINKING:
            return f"Thinking... {time.perf_counter() - self._thinking_since:.1f}s", ACCENT_THINKING
        return "Your move", TEXT_PRIMARY

    def _outcome_text(self) -> str:
        outcome = self._board.outcome(claim_draw=True)
        if outcome is None:
            return "Game over"
        reason = outcome.termination.name.replace("_", " ").title()
        if outcome.winner is None:
            return f"Draw - {reason}"
        winner = "White" if outcome.winner == chess.WHITE else "Black"
        side = "you" if outcome.winner == self.human_color else "the engine"
        return f"{winner} wins ({side}) - {reason}"
