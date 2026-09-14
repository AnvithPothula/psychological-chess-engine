"""The mining tab: controls for self-play rollouts, and their live progress.

pygame ships no widgets, so this module carries the two it needs -- a button and
a numeric stepper -- built to the same palette as the rest of the panel. They are
deliberately dumb: they own a rectangle, a label and a callback, and the view
owns all the state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, List, Optional, Sequence, Tuple

import pygame

from src.training.dpo_generator import RolloutUpdate
from src.ui.constants import (
    ACCENT_POSITIVE,
    ACCENT_START,
    ACCENT_STOP,
    ACCENT_THINKING,
    ACCENT_TRAP,
    CONTROL_BG,
    CONTROL_CORNER_RADIUS,
    CONTROL_DISABLED_BG,
    CONTROL_DISABLED_TEXT,
    CONTROL_GAP,
    CONTROL_HEIGHT,
    CONTROL_HOVER_BG,
    DIVIDER_HEIGHT,
    LINE_SPACING,
    METER_CORNER_RADIUS,
    METER_HEIGHT,
    METER_TRACK,
    PANEL_DIVIDER,
    PANEL_PADDING,
    SECTION_SPACING,
    STEPPER_BUTTON_WIDTH,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    FontSet,
    RGB,
)
from src.ui.session import MiningSettings

__all__ = ["Button", "MiningView", "Stepper"]

logger = logging.getLogger(__name__)

RATING_CHOICES: Final[Tuple[int, ...]] = (1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900)

# Fitted to a measured sweep (30 games per cell) so the panel can predict a run.
SECONDS_PER_GAME_AT_30: Final[float] = 1.3
PLY_COST_EXPONENT: Final[float] = 1.35
PAIRS_PER_GAME: Final[float] = 0.8
GAMES_CHOICES: Final[Tuple[int, ...]] = (
    10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10_000, 25_000, 50_000,
)
"""Roughly geometric, not linear. A ten-step ladder covers a smoke test and an
overnight run; a linear +10 stepper needed 199 clicks to cross the same range."""

PLIES_CHOICES: Final[Tuple[int, ...]] = (20, 30, 40, 50, 60, 80, 100, 140, 200)
"""30 is the sweet spot on measured yield: traps are an opening phenomenon, so
longer games cost time without producing more pairs. At rating 1100, 30 plies
yields 0.90 pairs/game at 2487 pairs/hour against 0.57 and 414 at 80 plies."""


@dataclass
class Button:
    """A labelled rectangle that calls back when clicked while enabled."""

    rect: pygame.Rect
    label: str
    on_click: Callable[[], None]
    enabled: bool = True
    colour: Optional[RGB] = None
    hovered: bool = field(default=False, init=False)

    def handle_motion(self, position: Tuple[int, int]) -> None:
        self.hovered = self.enabled and self.rect.collidepoint(position)

    def handle_click(self, position: Tuple[int, int]) -> bool:
        if not self.enabled or not self.rect.collidepoint(position):
            return False
        self.on_click()
        return True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.enabled:
            background, text_colour = CONTROL_DISABLED_BG, CONTROL_DISABLED_TEXT
        elif self.colour is not None:
            background, text_colour = self.colour, TEXT_PRIMARY
        else:
            background = CONTROL_HOVER_BG if self.hovered else CONTROL_BG
            text_colour = TEXT_PRIMARY
        pygame.draw.rect(surface, background, self.rect, border_radius=CONTROL_CORNER_RADIUS)
        label = font.render(self.label, True, text_colour)
        surface.blit(label, label.get_rect(center=self.rect.center))


@dataclass
class Stepper:
    """A ``-  value  +`` row over a fixed list of allowed values."""

    rect: pygame.Rect
    label: str
    values: Sequence[int]
    index: int
    on_change: Callable[[int], None]
    enabled: bool = True
    suffix: str = ""
    _minus: pygame.Rect = field(init=False)
    _plus: pygame.Rect = field(init=False)
    _hover: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._minus = pygame.Rect(self.rect.left, self.rect.top, STEPPER_BUTTON_WIDTH, self.rect.height)
        self._plus = pygame.Rect(
            self.rect.right - STEPPER_BUTTON_WIDTH, self.rect.top, STEPPER_BUTTON_WIDTH, self.rect.height
        )
        self.index = max(0, min(len(self.values) - 1, self.index))

    @property
    def value(self) -> int:
        return self.values[self.index]

    def handle_motion(self, position: Tuple[int, int]) -> None:
        if not self.enabled:
            self._hover = None
        elif self._minus.collidepoint(position):
            self._hover = "minus"
        elif self._plus.collidepoint(position):
            self._hover = "plus"
        else:
            self._hover = None

    def handle_click(self, position: Tuple[int, int]) -> bool:
        if not self.enabled:
            return False
        if self._minus.collidepoint(position):
            moved = max(0, self.index - 1)
        elif self._plus.collidepoint(position):
            moved = min(len(self.values) - 1, self.index + 1)
        else:
            return False
        if moved != self.index:
            self.index = moved
            self.on_change(self.value)
        return True

    def draw(self, surface: pygame.Surface, fonts: FontSet) -> None:
        caption = fonts.small.render(self.label, True, TEXT_MUTED if self.enabled else CONTROL_DISABLED_TEXT)
        surface.blit(caption, (self.rect.left, self.rect.top - caption.get_height() - 2))

        base = CONTROL_BG if self.enabled else CONTROL_DISABLED_BG
        pygame.draw.rect(surface, base, self.rect, border_radius=CONTROL_CORNER_RADIUS)
        for name, box, glyph in (("minus", self._minus, "-"), ("plus", self._plus, "+")):
            at_limit = (name == "minus" and self.index == 0) or (
                name == "plus" and self.index == len(self.values) - 1
            )
            usable = self.enabled and not at_limit
            colour = CONTROL_HOVER_BG if (self._hover == name and usable) else base
            pygame.draw.rect(surface, colour, box, border_radius=CONTROL_CORNER_RADIUS)
            mark = fonts.heading.render(glyph, True, TEXT_PRIMARY if usable else CONTROL_DISABLED_TEXT)
            surface.blit(mark, mark.get_rect(center=box.center))

        text = fonts.body.render(
            f"{self.value}{self.suffix}", True, TEXT_PRIMARY if self.enabled else CONTROL_DISABLED_TEXT
        )
        surface.blit(text, text.get_rect(center=self.rect.center))


class MiningView:
    """Panel for configuring, starting and watching a rollout."""

    def __init__(
        self,
        rect: pygame.Rect,
        fonts: FontSet,
        settings: MiningSettings,
        *,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_settings_changed: Callable[[], None],
    ) -> None:
        self.rect = rect
        self.fonts = fonts
        self.settings = settings
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_settings_changed = on_settings_changed

        self.running = False
        self.can_start = True
        self.blocked_reason = ""
        self.latest: Optional[RolloutUpdate] = None
        self.lifetime_pairs = 0
        self.lifetime_games = 0

        left = rect.left + PANEL_PADDING
        width = rect.width - PANEL_PADDING * 2
        # Room for the title, the output path beneath it, and the first
        # stepper's own caption, which is drawn above its box.
        cursor = (
            rect.top + PANEL_PADDING
            + fonts.title.get_height() + LINE_SPACING
            + fonts.small.get_height() + SECTION_SPACING
            + fonts.small.get_height() + LINE_SPACING
        )

        self.rating = Stepper(
            pygame.Rect(left, cursor, width, CONTROL_HEIGHT), "OPPONENT RATING",
            RATING_CHOICES, self._closest(RATING_CHOICES, settings.rating), self._set_rating,
        )
        cursor += CONTROL_HEIGHT + SECTION_SPACING + fonts.small.get_height() + LINE_SPACING
        self.games = Stepper(
            pygame.Rect(left, cursor, width, CONTROL_HEIGHT), "GAMES",
            GAMES_CHOICES, self._closest(GAMES_CHOICES, settings.games), self._set_games,
        )
        cursor += CONTROL_HEIGHT + SECTION_SPACING + fonts.small.get_height() + LINE_SPACING
        self.plies = Stepper(
            pygame.Rect(left, cursor, width, CONTROL_HEIGHT), "MAX PLIES PER GAME",
            PLIES_CHOICES, self._closest(PLIES_CHOICES, settings.max_plies), self._set_plies,
            suffix=" plies",
        )
        cursor += CONTROL_HEIGHT + SECTION_SPACING * 2

        self.action = Button(
            pygame.Rect(left, cursor, width, CONTROL_HEIGHT + 6), "Start mining", self._toggle,
            colour=ACCENT_START,
        )
        self._controls_bottom = self.action.rect.bottom
        # A stepper snaps to its nearest allowed value, so write that back
        # immediately. Otherwise the panel would show 10 games while the run
        # actually used the 6 that was persisted.
        self._sync_settings()

    def _sync_settings(self) -> None:
        snapped = (self.rating.value, self.games.value, self.plies.value)
        current = (self.settings.rating, self.settings.games, self.settings.max_plies)
        if snapped != current:
            self.settings.rating, self.settings.games, self.settings.max_plies = snapped
            logger.info("mining: settings snapped to %s", snapped)
            self.on_settings_changed()

    def estimate(self) -> Tuple[float, int]:
        """Rough ``(hours, pairs)`` for the current settings.

        Interpolated from a measured sweep: seconds per game scale with ply cap
        and opponent rating, and pairs per game barely move. Good enough to tell
        a coffee break from an overnight run.
        """
        plies = self.settings.max_plies
        rating = self.settings.rating
        seconds = SECONDS_PER_GAME_AT_30 * (plies / 30.0) ** PLY_COST_EXPONENT
        seconds *= 1.0 + (rating - 1100) / 1000.0
        games = self.settings.games
        return games * seconds / 3600.0, int(games * PAIRS_PER_GAME)

    @staticmethod
    def _closest(values: Sequence[int], target: int) -> int:
        return min(range(len(values)), key=lambda index: abs(values[index] - target))

    # -- settings -----------------------------------------------------------

    def _set_rating(self, value: int) -> None:
        self.settings.rating = value
        self.on_settings_changed()

    def _set_games(self, value: int) -> None:
        self.settings.games = value
        self.on_settings_changed()

    def _set_plies(self, value: int) -> None:
        self.settings.max_plies = value
        self.on_settings_changed()

    def _toggle(self) -> None:
        if self.running:
            self.on_stop()
        else:
            self.on_start()

    # -- state --------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        self.running = running
        self.action.label = "Stop mining" if running else "Start mining"
        self.action.colour = ACCENT_STOP if running else ACCENT_START
        for stepper in (self.rating, self.games, self.plies):
            stepper.enabled = not running
        self._refresh_enabled()

    def set_blocked(self, reason: str) -> None:
        """Why mining cannot start right now; empty string means it can."""
        self.blocked_reason = reason
        self.can_start = not reason
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        self.action.enabled = self.running or self.can_start

    # -- events -------------------------------------------------------------

    def handle_motion(self, position: Tuple[int, int]) -> None:
        for widget in (self.rating, self.games, self.plies):
            widget.handle_motion(position)
        self.action.handle_motion(position)

    def handle_click(self, position: Tuple[int, int]) -> bool:
        for stepper in (self.rating, self.games, self.plies):
            if stepper.handle_click(position):
                return True
        return self.action.handle_click(position)

    # -- rendering ----------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        left = self.rect.left + PANEL_PADDING
        width = self.rect.width - PANEL_PADDING * 2
        cursor = self.rect.top + PANEL_PADDING

        title = self.fonts.title.render("TRAP MINING", True, TEXT_PRIMARY)
        surface.blit(title, (left, cursor))
        cursor += title.get_height() + LINE_SPACING
        subtitle = self.fonts.small.render(self.settings.output, True, TEXT_DIM)
        surface.blit(subtitle, (left, cursor))

        for stepper in (self.rating, self.games, self.plies):
            stepper.draw(surface, self.fonts)
        self.action.draw(surface, self.fonts.heading)

        cursor = self._controls_bottom + SECTION_SPACING
        if self.blocked_reason and not self.running:
            note = self.fonts.small.render(self.blocked_reason, True, ACCENT_THINKING)
            surface.blit(note, (left, cursor))
            cursor += note.get_height() + LINE_SPACING

        cursor = self._divider(surface, left, cursor + LINE_SPACING, width)
        cursor = self._heading(surface, "PROGRESS", left, cursor + SECTION_SPACING)

        update = self.latest
        if update is None:
            if self.running:
                message = "Starting engines..."
            else:
                hours, pairs = self.estimate()
                message = f"~{pairs:,} pairs in ~{hours:.1f}h at these settings"
            surface.blit(self.fonts.body.render(message, True, TEXT_DIM), (left, cursor + LINE_SPACING))
        else:
            done = max(0, min(1.0, update.games_completed / max(1, self.settings.games)))
            cursor = self._bar(surface, left, cursor + LINE_SPACING, width, done)
            cursor = self._row(surface, "games", f"{update.games_completed} / {self.settings.games}",
                               left, cursor + LINE_SPACING, width, TEXT_PRIMARY)
            cursor = self._row(surface, "pairs this run", f"{update.pairs_total}",
                               left, cursor, width, ACCENT_TRAP)
            cursor = self._row(surface, "pairs / minute", f"{update.pairs_per_minute:.1f}",
                               left, cursor, width, TEXT_PRIMARY)
            cursor = self._row(surface, "opponent estimate", f"{update.current_elo}",
                               left, cursor, width,
                               ACCENT_POSITIVE if update.current_elo >= update.starting_elo else ACCENT_STOP)
            cursor = self._row(surface, "blunder hazard", f"{update.blunder_hazard:.0%}",
                               left, cursor, width, TEXT_MUTED)

        cursor = self._divider(surface, left, cursor + SECTION_SPACING, width)
        cursor = self._heading(surface, "ALL TIME", left, cursor + SECTION_SPACING)
        cursor = self._row(surface, "pairs mined", f"{self.lifetime_pairs}",
                           left, cursor + LINE_SPACING, width, TEXT_PRIMARY)
        self._row(surface, "games played", f"{self.lifetime_games}", left, cursor, width, TEXT_MUTED)

    def _heading(self, surface: pygame.Surface, text: str, left: int, top: int) -> int:
        label = self.fonts.heading.render(text, True, TEXT_MUTED)
        surface.blit(label, (left, top))
        return top + label.get_height()

    def _row(
        self, surface: pygame.Surface, label: str, value: str,
        left: int, top: int, width: int, colour: RGB,
    ) -> int:
        name = self.fonts.body.render(label, True, TEXT_MUTED)
        amount = self.fonts.body.render(value, True, colour)
        surface.blit(name, (left, top))
        surface.blit(amount, (left + width - amount.get_width(), top))
        return top + max(name.get_height(), amount.get_height()) + LINE_SPACING

    @staticmethod
    def _bar(surface: pygame.Surface, left: int, top: int, width: int, fraction: float) -> int:
        track = pygame.Rect(left, top, width, METER_HEIGHT)
        pygame.draw.rect(surface, METER_TRACK, track, border_radius=METER_CORNER_RADIUS)
        filled = pygame.Rect(left, top, max(1, int(width * fraction)), METER_HEIGHT)
        pygame.draw.rect(surface, ACCENT_TRAP, filled, border_radius=METER_CORNER_RADIUS)
        return top + METER_HEIGHT

    @staticmethod
    def _divider(surface: pygame.Surface, left: int, top: int, width: int) -> int:
        pygame.draw.rect(surface, PANEL_DIVIDER, pygame.Rect(left, top, width, DIVIDER_HEIGHT))
        return top + DIVIDER_HEIGHT
