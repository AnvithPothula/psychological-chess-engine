"""Live viewer for self-play rollouts.

A stripped fork of the match UI: no clicks, no selection, no legal-move hints,
because there is no human to serve. The board mirrors whatever the generator is
doing and the panel carries the two numbers a rollout is actually judged on --
how strong the opponent is playing, and how much of their reply mass is walking
into something.

**The generator never waits for this.** It runs on a worker thread and pushes
snapshots into a bounded queue; when the queue is full the snapshot is dropped
rather than blocking the rollout. A dropped frame costs nothing, a stalled
rollout costs throughput. That is also what lets the whole viewer be skipped:
without an observer the generator makes no UI calls at all.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Final, List, Optional, Tuple

import chess
import pygame

from src.training.dpo_generator import DPOGenerator, GeneratorStats, RolloutUpdate
from src.ui.asset_manager import AssetManager
from src.ui.board_view import BoardView
from src.ui.constants import (
    ACCENT_NEGATIVE,
    ACCENT_POSITIVE,
    ACCENT_THINKING,
    ACCENT_TRAP,
    BOARD_SIZE,
    DIVIDER_HEIGHT,
    ELO_METER_SPAN,
    FPS,
    HAZARD_METER_SEGMENTS,
    LINE_SPACING,
    METER_CORNER_RADIUS,
    METER_HEIGHT,
    METER_TRACK,
    PANEL_BG,
    PANEL_CORNER_RADIUS,
    PANEL_DIVIDER,
    PANEL_PADDING,
    PANEL_WIDTH,
    PANEL_X,
    ROLLOUT_QUEUE_LIMIT,
    SECTION_SPACING,
    SQUARE_SIZE,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TRAINING_WINDOW_TITLE,
    WINDOW_BG,
    WINDOW_HEIGHT,
    WINDOW_MARGIN,
    WINDOW_WIDTH,
    FontSet,
    RGB,
    load_fonts,
)

__all__ = ["TrainingViewer", "run_with_viewer"]

logger = logging.getLogger(__name__)

WORKER_JOIN_TIMEOUT: Final[float] = 10.0
BAR_HEIGHT: Final[int] = 14
SEGMENT_GAP: Final[int] = 2


class TrainingViewer:
    """Renders rollout snapshots. Owns no game state of its own."""

    def __init__(self, assets: Optional[AssetManager] = None) -> None:
        self.assets = assets if assets is not None else AssetManager()
        self.updates: "queue.Queue[RolloutUpdate]" = queue.Queue(maxsize=ROLLOUT_QUEUE_LIMIT)
        self._board = chess.Board()
        self._latest: Optional[RolloutUpdate] = None
        self._running = True

    # -- generator side -----------------------------------------------------

    def observe(self, update: RolloutUpdate) -> None:
        """Observer handed to the generator. Must never block the rollout."""
        try:
            self.updates.put_nowait(update)
        except queue.Full:
            pass  # Dropping a frame is the correct trade; stalling is not.

    def stop(self) -> None:
        self._running = False

    # -- viewer side --------------------------------------------------------

    def run(self, worker: threading.Thread) -> None:
        """Pump pygame until the rollout finishes or the window is closed."""
        pygame.init()
        try:
            screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption(TRAINING_WINDOW_TITLE)
            fonts = load_fonts()
            board_view = BoardView(self.assets.load_pieces(SQUARE_SIZE), fonts.coordinate)
            panel = pygame.Rect(PANEL_X, WINDOW_MARGIN, PANEL_WIDTH, BOARD_SIZE)
            clock = pygame.time.Clock()

            while self._running:
                # Pumping every frame regardless of rollout speed is what keeps
                # the window responsive: at high speed the queue delivers many
                # snapshots per frame, at low speed none, and neither starves
                # the event loop.
                if not self._pump_events():
                    break
                self._drain()
                self._render(screen, board_view, panel, fonts)
                pygame.display.flip()
                clock.tick(FPS)

                if not worker.is_alive() and self.updates.empty():
                    self._render(screen, board_view, panel, fonts)
                    pygame.display.flip()
                    break
        finally:
            pygame.quit()

    def _pump_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def _drain(self) -> None:
        """Take every queued snapshot; only the last one is worth drawing."""
        latest = self._latest
        while True:
            try:
                latest = self.updates.get_nowait()
            except queue.Empty:
                break
        if latest is not None and latest is not self._latest:
            self._latest = latest
            try:
                self._board.set_fen(latest.fen)
            except ValueError:
                logger.debug("viewer: ignoring unparseable fen %r", latest.fen)

    # -- rendering ----------------------------------------------------------

    def _render(
        self,
        screen: pygame.Surface,
        board_view: BoardView,
        panel: pygame.Rect,
        fonts: FontSet,
    ) -> None:
        screen.fill(WINDOW_BG)
        last_move = self._latest.last_move if self._latest else None
        board_view.draw(screen, self._board, last_move=last_move)
        self._draw_panel(screen, panel, fonts)

    def _draw_panel(self, screen: pygame.Surface, panel: pygame.Rect, fonts: FontSet) -> None:
        pygame.draw.rect(screen, PANEL_BG, panel, border_radius=PANEL_CORNER_RADIUS)
        left = panel.left + PANEL_PADDING
        width = panel.width - PANEL_PADDING * 2
        cursor = panel.top + PANEL_PADDING

        cursor = self._text(screen, "SELF-PLAY TRAP MINING", fonts.title, TEXT_PRIMARY, left, cursor)
        update = self._latest
        if update is None:
            self._text(screen, "Waiting for the first rollout...", fonts.body, TEXT_DIM,
                       left, cursor + SECTION_SPACING)
            return

        cursor = self._text(screen, f"game {update.game_index + 1} · ply {update.ply} · {update.status}",
                            fonts.small, TEXT_DIM, left, cursor + LINE_SPACING)
        cursor = self._divider(screen, left, cursor + SECTION_SPACING, width)

        cursor = self._text(screen, "COGNITIVE ELO", fonts.heading, TEXT_MUTED,
                            left, cursor + SECTION_SPACING)
        drift = update.current_elo - update.starting_elo
        cursor = self._elo_meter(screen, left, cursor + LINE_SPACING, width, drift)
        cursor = self._row(screen, fonts, "estimate", f"{update.current_elo}", left, cursor + LINE_SPACING, width,
                           ACCENT_NEGATIVE if drift < 0 else ACCENT_POSITIVE if drift > 0 else TEXT_PRIMARY)
        cursor = self._row(screen, fonts, "started at", f"{update.starting_elo}", left, cursor, width, TEXT_MUTED)
        cursor = self._row(screen, fonts, "drift", f"{drift:+d}", left, cursor, width,
                           ACCENT_NEGATIVE if drift < 0 else ACCENT_POSITIVE if drift > 0 else TEXT_MUTED)

        cursor = self._divider(screen, left, cursor + SECTION_SPACING, width)
        cursor = self._text(screen, "BLUNDER HAZARD", fonts.heading, TEXT_MUTED,
                            left, cursor + SECTION_SPACING)
        cursor = self._hazard_meter(screen, left, cursor + LINE_SPACING, width, update.blunder_hazard)
        cursor = self._row(screen, fonts, "reply mass at risk", f"{update.blunder_hazard:.0%}",
                           left, cursor + LINE_SPACING, width,
                           ACCENT_TRAP if update.blunder_hazard > 0.15 else TEXT_MUTED)

        cursor = self._divider(screen, left, cursor + SECTION_SPACING, width)
        cursor = self._text(screen, "YIELD", fonts.heading, TEXT_MUTED, left, cursor + SECTION_SPACING)
        cursor = self._row(screen, fonts, "pairs mined", f"{update.pairs_total}",
                           left, cursor + LINE_SPACING, width, ACCENT_TRAP)
        cursor = self._row(screen, fonts, "pairs / minute", f"{update.pairs_per_minute:.1f}",
                           left, cursor, width, TEXT_PRIMARY)
        self._row(screen, fonts, "games finished", f"{update.games_completed}", left, cursor, width, TEXT_MUTED)

    def _elo_meter(self, screen: pygame.Surface, left: int, top: int, width: int, drift: int) -> int:
        """Centre-anchored: the opponent playing above or below their rating."""
        track = pygame.Rect(left, top, width, BAR_HEIGHT)
        pygame.draw.rect(screen, METER_TRACK, track, border_radius=METER_CORNER_RADIUS)
        clamped = max(-ELO_METER_SPAN, min(ELO_METER_SPAN, float(drift)))
        centre = left + width // 2
        span = int((width // 2) * abs(clamped) / ELO_METER_SPAN)
        if span:
            rect = pygame.Rect(centre if clamped > 0 else centre - span, top, span, BAR_HEIGHT)
            pygame.draw.rect(screen, ACCENT_POSITIVE if clamped > 0 else ACCENT_NEGATIVE,
                             rect, border_radius=METER_CORNER_RADIUS)
        pygame.draw.line(screen, TEXT_DIM, (centre, top), (centre, top + BAR_HEIGHT))
        return top + BAR_HEIGHT

    def _hazard_meter(self, screen: pygame.Surface, left: int, top: int, width: int, hazard: float) -> int:
        """Segmented bar: reads as a probability rather than a continuous value."""
        segments = HAZARD_METER_SEGMENTS
        lit = int(round(max(0.0, min(1.0, hazard)) * segments))
        segment_width = (width - SEGMENT_GAP * (segments - 1)) / segments
        for index in range(segments):
            rect = pygame.Rect(
                int(left + index * (segment_width + SEGMENT_GAP)), top,
                max(1, int(segment_width)), METER_HEIGHT,
            )
            colour: RGB = ACCENT_TRAP if index < lit else METER_TRACK
            pygame.draw.rect(screen, colour, rect, border_radius=2)
        return top + METER_HEIGHT

    def _row(
        self, screen: pygame.Surface, fonts: FontSet, label: str, value: str,
        left: int, top: int, width: int, colour: RGB,
    ) -> int:
        name = fonts.body.render(label, True, TEXT_MUTED)
        amount = fonts.body.render(value, True, colour)
        screen.blit(name, (left, top))
        screen.blit(amount, (left + width - amount.get_width(), top))
        return top + max(name.get_height(), amount.get_height()) + LINE_SPACING

    @staticmethod
    def _text(
        screen: pygame.Surface, text: str, font: pygame.font.Font, colour: RGB, left: int, top: int
    ) -> int:
        label = font.render(text, True, colour)
        screen.blit(label, (left, top))
        return top + label.get_height()

    @staticmethod
    def _divider(screen: pygame.Surface, left: int, top: int, width: int) -> int:
        pygame.draw.rect(screen, PANEL_DIVIDER, pygame.Rect(left, top, width, DIVIDER_HEIGHT))
        return top + DIVIDER_HEIGHT


def run_with_viewer(
    generator: DPOGenerator, output_path: Path, *, assets: Optional[AssetManager] = None
) -> GeneratorStats:
    """Run a rollout with the window open. Returns the generator's own stats.

    The rollout owns the worker thread and pygame owns the main thread, which is
    not negotiable on macOS. Closing the window asks the generator to stop after
    the current move rather than killing it mid-search.
    """
    viewer = TrainingViewer(assets)
    generator.observer = viewer.observe
    stats: List[GeneratorStats] = []

    def rollout() -> None:
        try:
            stats.append(generator.run(output_path))
        except Exception:  # noqa: BLE001 - a dead worker must not hang the window
            logger.exception("rollout: worker failed")
        finally:
            viewer.stop()

    worker = threading.Thread(target=rollout, name="dpo-rollout", daemon=True)
    worker.start()
    try:
        viewer.run(worker)
    finally:
        generator.stop()
        worker.join(timeout=WORKER_JOIN_TIMEOUT)
    return stats[0] if stats else generator.stats
