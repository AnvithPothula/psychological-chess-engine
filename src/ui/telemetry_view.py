"""The evaluation bar and the psychological-telemetry side panel.

The panel unpacks a :class:`~src.types.SearchResult` into four blocks: what the
engine played, whether it played it for objective or psychological reasons, the
numbers behind that decision, and the human replies it is betting on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional, Sequence, Tuple

import chess
import pygame

from src.types import CandidateStats, EngineEval, MoveSource, SearchResult
from src.ui.constants import (
    ACCENT_FALLBACK,
    ACCENT_NEGATIVE,
    ACCENT_POSITIVE,
    ACCENT_TRAP,
    BADGE_CORNER_RADIUS,
    BADGE_HEIGHT,
    BADGE_PADDING_X,
    DIVIDER_HEIGHT,
    EVAL_BAR_BLACK,
    EVAL_BAR_BORDER,
    EVAL_BAR_CLAMP_PAWNS,
    EVAL_BAR_MIDLINE,
    EVAL_BAR_WHITE,
    LINE_SPACING,
    METER_CORNER_RADIUS,
    METER_HEIGHT,
    METER_TRACK,
    PANEL_BG,
    PANEL_CORNER_RADIUS,
    PANEL_DIVIDER,
    PANEL_PADDING,
    REPLY_METER_HEIGHT,
    REPLY_ROW_HEIGHT,
    SECTION_SPACING,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    UTILITY_METER_CLAMP_CP,
    FontSet,
    RGB,
)

__all__ = ["EvalBar", "MoveReport", "TelemetryView"]

CENTIPAWNS_PER_PAWN: Final[float] = 100.0
BORDER_WIDTH: Final[int] = 2
REPLY_METER_WIDTH_RATIO: Final[float] = 0.46


@dataclass(frozen=True, slots=True)
class MoveReport:
    """A :class:`SearchResult` with the SAN strings the panel needs to render it.

    SAN is resolved when the move is applied, while the boards it refers to are
    still to hand; the view never needs a ``chess.Board``.
    """

    san: str
    result: SearchResult
    chosen: Optional[CandidateStats]
    """Telemetry for the move actually played, or ``None`` for a mate in 1."""

    reply_sans: Tuple[str, ...]
    """SAN for ``chosen.top_replies``, in the same order."""


class EvalBar:
    """Vertical Stockfish evaluation bar, oriented to match the board.

    Clamped to +/-``EVAL_BAR_CLAMP_PAWNS``, because past a few pawns the exact
    number stops meaning anything to a human reading a bar.
    """

    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect

    def draw(
        self,
        surface: pygame.Surface,
        evaluation: Optional[EngineEval],
        orientation: chess.Color = chess.WHITE,
    ) -> None:
        pawns = 0.0 if evaluation is None else evaluation.centipawns / CENTIPAWNS_PER_PAWN
        if evaluation is not None and evaluation.is_mate:
            pawns = EVAL_BAR_CLAMP_PAWNS if evaluation.centipawns > 0 else -EVAL_BAR_CLAMP_PAWNS

        clamped = max(-EVAL_BAR_CLAMP_PAWNS, min(EVAL_BAR_CLAMP_PAWNS, pawns))
        # The advantaged side's colour grows from its own end of the bar.
        white_fraction = 0.5 + clamped / (2.0 * EVAL_BAR_CLAMP_PAWNS)
        if orientation == chess.BLACK:
            white_fraction = 1.0 - white_fraction

        white_height = int(self.rect.height * white_fraction)
        near, far = (EVAL_BAR_WHITE, EVAL_BAR_BLACK) if orientation == chess.WHITE else (EVAL_BAR_BLACK, EVAL_BAR_WHITE)

        surface.fill(far, self.rect)
        surface.fill(
            near,
            pygame.Rect(
                self.rect.left,
                self.rect.bottom - white_height,
                self.rect.width,
                white_height,
            ),
        )
        pygame.draw.line(
            surface,
            EVAL_BAR_MIDLINE,
            (self.rect.left, self.rect.centery),
            (self.rect.right, self.rect.centery),
        )
        pygame.draw.rect(surface, EVAL_BAR_BORDER, self.rect, BORDER_WIDTH)


class TelemetryView:
    """Renders the side panel. Stateless: everything comes in through ``draw``."""

    def __init__(self, rect: pygame.Rect, fonts: FontSet) -> None:
        self.rect = rect
        self.fonts = fonts

    def draw(
        self,
        surface: pygame.Surface,
        *,
        status_text: str,
        status_colour: RGB,
        report: Optional[MoveReport],
        opponent_label: str,
        evaluation: Optional[EngineEval],
        message: Optional[str] = None,
    ) -> None:
        pygame.draw.rect(surface, PANEL_BG, self.rect, border_radius=PANEL_CORNER_RADIUS)

        left = self.rect.left + PANEL_PADDING
        width = self.rect.width - PANEL_PADDING * 2
        cursor = self.rect.top + PANEL_PADDING

        cursor = self._text(surface, "PSYCHOLOGICAL TELEMETRY", self.fonts.title, TEXT_PRIMARY, left, cursor)
        cursor = self._text(surface, opponent_label, self.fonts.small, TEXT_DIM, left, cursor + LINE_SPACING)
        cursor = self._divider(surface, left, cursor + SECTION_SPACING, width)

        cursor = self._text(surface, status_text, self.fonts.status, status_colour, left, cursor + SECTION_SPACING)
        if message is not None:
            cursor = self._wrapped(surface, message, self.fonts.small, ACCENT_NEGATIVE, left, cursor + LINE_SPACING, width)

        cursor = self._divider(surface, left, cursor + SECTION_SPACING, width)
        cursor = self._draw_decision(surface, report, evaluation, left, cursor + SECTION_SPACING, width)
        cursor = self._divider(surface, left, cursor + SECTION_SPACING, width)
        self._draw_replies(surface, report, left, cursor + SECTION_SPACING, width)

        self._draw_footer(surface, report, left, width)

    # -- blocks -------------------------------------------------------------

    def _draw_decision(
        self,
        surface: pygame.Surface,
        report: Optional[MoveReport],
        evaluation: Optional[EngineEval],
        left: int,
        top: int,
        width: int,
    ) -> int:
        cursor = self._text(surface, "ENGINE DECISION", self.fonts.heading, TEXT_MUTED, left, top)
        if report is None:
            objective = "--" if evaluation is None else self._format_cp(evaluation.centipawns)
            cursor = self._text(
                surface, "Waiting for the engine's first move.", self.fonts.body, TEXT_DIM, left, cursor + LINE_SPACING
            )
            return self._row(surface, "Objective (White)", objective, left, cursor + LINE_SPACING, width)

        result = report.result
        cursor = self._text(surface, report.san, self.fonts.title, TEXT_PRIMARY, left, cursor + LINE_SPACING)

        label, colour = self._badge_for(result)
        cursor = self._badge(surface, label, colour, left, cursor + LINE_SPACING)

        cursor = self._meter(
            surface, result.expected_utility, UTILITY_METER_CLAMP_CP, left, cursor + SECTION_SPACING, width
        )
        cursor = self._row(
            surface, "Psychological utility", self._format_cp(result.expected_utility), left, cursor + LINE_SPACING, width
        )
        if report.chosen is not None:
            chosen = report.chosen
            cursor = self._row(
                surface, "Objective score", self._format_cp(chosen.objective_score), left, cursor, width
            )
            cursor = self._row(
                surface,
                "Blunder-trap delta",
                self._format_cp(chosen.blunder_trap_delta),
                left,
                cursor,
                width,
                value_colour=ACCENT_POSITIVE if chosen.blunder_trap_delta > 0 else TEXT_MUTED,
            )
            cursor = self._row(
                surface,
                "Safety floor",
                self._format_cp(chosen.worst_case),
                left,
                cursor,
                width,
                value_colour=ACCENT_POSITIVE if chosen.is_safe else ACCENT_NEGATIVE,
            )
        return cursor

    def _draw_replies(
        self, surface: pygame.Surface, report: Optional[MoveReport], left: int, top: int, width: int
    ) -> int:
        cursor = self._text(surface, "PREDICTED HUMAN REPLIES", self.fonts.heading, TEXT_MUTED, left, top)
        cursor += LINE_SPACING

        if report is None or report.chosen is None or not report.chosen.top_replies:
            return self._text(surface, "No opponent model for this move.", self.fonts.body, TEXT_DIM, left, cursor)

        replies = report.chosen.top_replies
        meter_width = int(width * REPLY_METER_WIDTH_RATIO)
        for index, reply in enumerate(replies):
            san = report.reply_sans[index] if index < len(report.reply_sans) else reply.move.uci()
            row_top = cursor + index * REPLY_ROW_HEIGHT

            surface.blit(self.fonts.body.render(san, True, TEXT_PRIMARY), (left, row_top))
            share = self.fonts.body.render(f"{reply.probability:.0%}", True, TEXT_MUTED)
            surface.blit(share, (left + width - share.get_width(), row_top))

            bar_top = row_top + self.fonts.body.get_height() + 2
            track = pygame.Rect(left, bar_top, meter_width, REPLY_METER_HEIGHT)
            pygame.draw.rect(surface, METER_TRACK, track, border_radius=METER_CORNER_RADIUS)
            filled = pygame.Rect(left, bar_top, max(1, int(meter_width * reply.probability)), REPLY_METER_HEIGHT)
            pygame.draw.rect(surface, ACCENT_TRAP, filled, border_radius=METER_CORNER_RADIUS)

            leads_to = self.fonts.small.render(
                f"leads to {self._format_cp(reply.evaluation)}",
                True,
                ACCENT_POSITIVE if reply.evaluation > 0 else ACCENT_NEGATIVE,
            )
            surface.blit(leads_to, (left + width - leads_to.get_width(), bar_top - 1))

        return cursor + len(replies) * REPLY_ROW_HEIGHT

    def _draw_footer(
        self, surface: pygame.Surface, report: Optional[MoveReport], left: int, width: int
    ) -> None:
        if report is None:
            return
        result = report.result
        text = (
            f"{len(result.candidates)} candidates | {result.nodes_evaluated} nodes "
            f"| {result.duration_ms:.0f} ms"
        )
        label = self.fonts.small.render(text, True, TEXT_DIM)
        baseline = self.rect.bottom - PANEL_PADDING - label.get_height()
        pygame.draw.rect(
            surface, PANEL_DIVIDER, pygame.Rect(left, baseline - SECTION_SPACING, width, DIVIDER_HEIGHT)
        )
        surface.blit(label, (left, baseline))

    # -- primitives ---------------------------------------------------------

    @staticmethod
    def _badge_for(result: SearchResult) -> Tuple[str, RGB]:
        # Book provenance wins: a Stafford Gambit move badged "OBJECTIVE BEST"
        # would be actively misleading, and is_trap is False for every book move.
        if result.source is MoveSource.BOOK_TRAP:
            return "TRAP BOOK", ACCENT_TRAP
        if result.source is MoveSource.BOOK_STANDARD:
            return "OPENING BOOK", ACCENT_FALLBACK
        if result.source is MoveSource.MATE_IN_ONE:
            return "MATE IN 1", ACCENT_TRAP
        if result.fallback_triggered:
            return "MINIMAX FALLBACK", ACCENT_FALLBACK
        if result.is_trap:
            return "TRAP ACTIVE", ACCENT_TRAP
        return "OBJECTIVE BEST", ACCENT_FALLBACK

    def _text(
        self, surface: pygame.Surface, text: str, font: pygame.font.Font, colour: RGB, left: int, top: int
    ) -> int:
        label = font.render(text, True, colour)
        surface.blit(label, (left, top))
        return top + label.get_height()

    def _wrapped(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        colour: RGB,
        left: int,
        top: int,
        width: int,
    ) -> int:
        line = ""
        cursor = top
        for word in text.split():
            probe = f"{line} {word}".strip()
            if font.size(probe)[0] > width and line:
                cursor = self._text(surface, line, font, colour, left, cursor)
                line = word
            else:
                line = probe
        if line:
            cursor = self._text(surface, line, font, colour, left, cursor)
        return cursor

    def _row(
        self,
        surface: pygame.Surface,
        label: str,
        value: str,
        left: int,
        top: int,
        width: int,
        *,
        value_colour: RGB = TEXT_PRIMARY,
    ) -> int:
        name = self.fonts.body.render(label, True, TEXT_MUTED)
        amount = self.fonts.body.render(value, True, value_colour)
        surface.blit(name, (left, top))
        surface.blit(amount, (left + width - amount.get_width(), top))
        return top + max(name.get_height(), amount.get_height()) + LINE_SPACING

    def _meter(
        self, surface: pygame.Surface, value: float, clamp: float, left: int, top: int, width: int
    ) -> int:
        """A centre-anchored bar: fills right when the bot is better, left when worse."""
        track = pygame.Rect(left, top, width, METER_HEIGHT)
        pygame.draw.rect(surface, METER_TRACK, track, border_radius=METER_CORNER_RADIUS)

        clamped = max(-clamp, min(clamp, value))
        centre = left + width // 2
        span = int((width // 2) * abs(clamped) / clamp)
        if span:
            filled = pygame.Rect(centre if clamped > 0 else centre - span, top, span, METER_HEIGHT)
            colour = ACCENT_POSITIVE if clamped > 0 else ACCENT_NEGATIVE
            pygame.draw.rect(surface, colour, filled, border_radius=METER_CORNER_RADIUS)
        pygame.draw.line(surface, TEXT_DIM, (centre, top), (centre, top + METER_HEIGHT))
        return top + METER_HEIGHT

    def _badge(self, surface: pygame.Surface, text: str, colour: RGB, left: int, top: int) -> int:
        label = self.fonts.heading.render(text, True, TEXT_PRIMARY)
        rect = pygame.Rect(left, top, label.get_width() + BADGE_PADDING_X * 2, BADGE_HEIGHT)
        pygame.draw.rect(surface, colour, rect, border_radius=BADGE_CORNER_RADIUS)
        surface.blit(label, label.get_rect(center=rect.center))
        return rect.bottom

    def _divider(self, surface: pygame.Surface, left: int, top: int, width: int) -> int:
        pygame.draw.rect(surface, PANEL_DIVIDER, pygame.Rect(left, top, width, DIVIDER_HEIGHT))
        return top + DIVIDER_HEIGHT

    @staticmethod
    def _format_cp(centipawns: float) -> str:
        return f"{centipawns / CENTIPAWNS_PER_PAWN:+.2f}"
