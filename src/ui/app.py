"""Pygame application: event loop, engine threading and state machine.

**Threading model.** Exactly one worker thread touches the engines, ever.
``chess.engine.SimpleEngine`` is not reentrant -- it multiplexes one UCI process
over a background event loop -- so two threads issuing commands to the same
process interleave their protocol traffic and corrupt both. The eval bar and the
adversarial search are two separate engine consumers, so they are serialised
onto one worker through a FIFO job queue rather than given a thread each.

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
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, Optional, Sequence, Tuple, Union

import chess
import pygame

from src.engine import EvaluatorError
from src.engine.search import AdversarialSearcher
from src.types import EngineEval, SearchResult
from src.ui.asset_manager import AssetManager
from src.ui.board_view import BoardView
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
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WORKER_JOIN_TIMEOUT,
    RGB,
    load_fonts,
)
from src.ui.telemetry_view import EvalBar, MoveReport, TelemetryView

__all__ = ["AppState", "ChessApp"]

logger = logging.getLogger(__name__)

MOUSE_BUTTON_LEFT: Final[int] = 1


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

    def __init__(
        self,
        searcher: AdversarialSearcher,
        assets: AssetManager,
        *,
        human_color: chess.Color = chess.WHITE,
        opponent_label: str = "Maia",
        board: Optional[chess.Board] = None,
    ) -> None:
        self.searcher = searcher
        self.assets = assets
        self.human_color = human_color
        self.opponent_label = opponent_label

        self._board = board.copy() if board is not None else chess.Board()
        self._state = AppState.HUMAN_TURN
        self._selected: Optional[chess.Square] = None
        self._legal_moves: Tuple[chess.Move, ...] = ()
        self._last_move: Optional[chess.Move] = None
        self._evaluation: Optional[EngineEval] = None
        self._report: Optional[MoveReport] = None
        self._message: Optional[str] = None
        self._thinking_since = 0.0
        self._running = True

        self._jobs: "queue.Queue[Optional[EngineJob]]" = queue.Queue()
        self._outcomes: "queue.Queue[EngineOutcome]" = queue.Queue()

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
        self._telemetry = TelemetryView(
            pygame.Rect(PANEL_X, WINDOW_MARGIN, PANEL_WIDTH, BOARD_SIZE), fonts
        )

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
            self._render(screen)
            pygame.display.flip()
            clock.tick(FPS)

    def _stop_worker(self, worker: threading.Thread) -> None:
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
        logger.info("ui: engine played %s (%s)", san, result.summary())

        if self._finish_if_over():
            return
        self._state = AppState.HUMAN_TURN
        self._request_evaluation()

    def _fail(self, message: str) -> None:
        self._message = message
        self._state = AppState.GAME_OVER

    # -- input --------------------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == MOUSE_BUTTON_LEFT:
                # Clicks are dropped outright while the engine owns the turn.
                if self._state is AppState.HUMAN_TURN:
                    self._handle_click(event.pos)

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
        return True

    # -- rendering ----------------------------------------------------------

    def _render(self, screen: pygame.Surface) -> None:
        screen.fill(WINDOW_BG)
        self._board_view.draw(
            screen,
            self._board,
            selected=self._selected,
            legal_moves=self._visible_hints(),
            last_move=self._last_move,
        )
        self._eval_bar.draw(screen, self._evaluation, self.human_color)
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
