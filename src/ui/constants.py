"""Every colour, dimension, font size and timing constant used by the UI.

Nothing in the view modules hardcodes a number: geometry is derived here from
``SQUARE_SIZE`` and the margins, so changing the board size relayouts the whole
window. The palette follows the ChessAI desktop client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Tuple

import pygame

RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]

# --- geometry --------------------------------------------------------------
SQUARE_SIZE: Final[int] = 84
BOARD_SQUARES: Final[int] = 8
BOARD_SIZE: Final[int] = SQUARE_SIZE * BOARD_SQUARES

WINDOW_MARGIN: Final[int] = 24
EVAL_BAR_WIDTH: Final[int] = 25
EVAL_BAR_GAP: Final[int] = 14
PANEL_GAP: Final[int] = 18
PANEL_WIDTH: Final[int] = 372

BOARD_ORIGIN: Final[Tuple[int, int]] = (WINDOW_MARGIN, WINDOW_MARGIN)
EVAL_BAR_X: Final[int] = WINDOW_MARGIN + BOARD_SIZE + EVAL_BAR_GAP
PANEL_X: Final[int] = EVAL_BAR_X + EVAL_BAR_WIDTH + PANEL_GAP

WINDOW_WIDTH: Final[int] = PANEL_X + PANEL_WIDTH + WINDOW_MARGIN
WINDOW_HEIGHT: Final[int] = WINDOW_MARGIN * 2 + BOARD_SIZE

PANEL_PADDING: Final[int] = 18
PANEL_CORNER_RADIUS: Final[int] = 8
SECTION_SPACING: Final[int] = 16
LINE_SPACING: Final[int] = 5
DIVIDER_HEIGHT: Final[int] = 1

BADGE_HEIGHT: Final[int] = 30
BADGE_CORNER_RADIUS: Final[int] = 6
BADGE_PADDING_X: Final[int] = 12

METER_HEIGHT: Final[int] = 10
METER_CORNER_RADIUS: Final[int] = 5
REPLY_ROW_HEIGHT: Final[int] = 34
REPLY_METER_HEIGHT: Final[int] = 6

COORDINATE_INSET: Final[int] = 4

# --- board palette (ChessAI) ----------------------------------------------
LIGHT_SQUARE: Final[RGB] = (0xDD, 0xB8, 0x8C)
DARK_SQUARE: Final[RGB] = (0xA6, 0x6D, 0x4F)
LIGHT_SQUARE_HIGHLIGHT: Final[RGB] = (0xF4, 0xA2, 0x61)
DARK_SQUARE_HIGHLIGHT: Final[RGB] = (0xE7, 0x6F, 0x51)
LEGAL_MOVE_DOT: Final[RGBA] = (0, 0, 0, 50)
LEGAL_CAPTURE_RING: Final[RGBA] = (0, 0, 0, 90)
CHECK_HIGHLIGHT: Final[RGBA] = (0xD6, 0x28, 0x28, 140)

LEGAL_DOT_RADIUS_RATIO: Final[float] = 0.20
LEGAL_RING_RADIUS_RATIO: Final[float] = 0.40
LEGAL_RING_WIDTH_RATIO: Final[float] = 0.08

# --- chrome palette --------------------------------------------------------
WINDOW_BG: Final[RGB] = (0x2B, 0x2B, 0x2B)
PANEL_BG: Final[RGB] = (0x40, 0x40, 0x40)
PANEL_DIVIDER: Final[RGB] = (0x55, 0x55, 0x55)
METER_TRACK: Final[RGB] = (0x33, 0x33, 0x33)
BOARD_BORDER: Final[RGB] = (0x1E, 0x1E, 0x1E)

TEXT_PRIMARY: Final[RGB] = (0xFF, 0xFF, 0xFF)
TEXT_MUTED: Final[RGB] = (0xB0, 0xB0, 0xB0)
TEXT_DIM: Final[RGB] = (0x8A, 0x8A, 0x8A)

ACCENT_TRAP: Final[RGB] = (0x2A, 0x9D, 0x8F)
ACCENT_FALLBACK: Final[RGB] = (0x6C, 0x6C, 0x6C)
ACCENT_POSITIVE: Final[RGB] = (0x8A, 0xC9, 0x26)
ACCENT_NEGATIVE: Final[RGB] = (0xE7, 0x6F, 0x51)
ACCENT_THINKING: Final[RGB] = (0xF4, 0xA2, 0x61)

EVAL_BAR_WHITE: Final[RGB] = (0xFF, 0xFF, 0xFF)
EVAL_BAR_BLACK: Final[RGB] = (0x40, 0x40, 0x40)
EVAL_BAR_BORDER: Final[RGB] = (0x7A, 0x7A, 0x7A)
"""Mid grey, not black: ChessAI's bar sat on a light Qt background, where a dark
border read fine. On this dark window the bar's extent needs the contrast."""
EVAL_BAR_MIDLINE: Final[RGB] = (0x8A, 0x8A, 0x8A)

# --- typography ------------------------------------------------------------
FONT_NAME: Final[str] = "Arial"
GLYPH_FONT_NAMES: Final[str] = "Arial Unicode MS,DejaVu Sans,Apple Symbols,Arial"
FONT_SIZE_TITLE: Final[int] = 20
FONT_SIZE_HEADING: Final[int] = 14
FONT_SIZE_BODY: Final[int] = 13
FONT_SIZE_SMALL: Final[int] = 11
FONT_SIZE_STATUS: Final[int] = 16

# --- behaviour -------------------------------------------------------------
FPS: Final[int] = 60
WINDOW_TITLE: Final[str] = "Psychological Chess Engine"
EVAL_BAR_CLAMP_PAWNS: Final[float] = 5.0
EVAL_BAR_DEPTH: Final[int] = 10
"""Depth for the eval-bar refresh. Shallow on purpose: it must not compete for
engine time with the search that decides the actual move."""

UTILITY_METER_CLAMP_CP: Final[float] = 400.0
TOP_REPLIES_SHOWN: Final[int] = 3
TRAINING_WINDOW_TITLE: Final[str] = "Self-Play Trap Mining"
ELO_METER_SPAN: Final[float] = 400.0
"""Elo either side of the starting rating that the drift meter spans."""
HAZARD_METER_SEGMENTS: Final[int] = 20
ROLLOUT_QUEUE_LIMIT: Final[int] = 256
"""Snapshots buffered for the viewer. The generator never blocks on a full
queue -- dropping a frame is correct, stalling the rollout is not."""
WORKER_JOIN_TIMEOUT: Final[float] = 5.0
ASSET_DOWNLOAD_TIMEOUT: Final[float] = 30.0
ASSET_DOWNLOAD_DELAY: Final[float] = 0.6
"""Pause between piece downloads. Wikimedia answers 429 to burst traffic."""
ASSET_DOWNLOAD_ATTEMPTS: Final[int] = 4
ASSET_DOWNLOAD_BACKOFF: Final[float] = 1.8


@dataclass(frozen=True, slots=True)
class FontSet:
    """The fonts every view shares. Built once, after ``pygame.font.init()``."""

    title: pygame.font.Font
    heading: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font
    status: pygame.font.Font
    coordinate: pygame.font.Font


def load_fonts() -> FontSet:
    """Build the shared font set. Requires ``pygame.font`` to be initialised."""
    if not pygame.font.get_init():
        pygame.font.init()
    return FontSet(
        title=pygame.font.SysFont(FONT_NAME, FONT_SIZE_TITLE, bold=True),
        heading=pygame.font.SysFont(FONT_NAME, FONT_SIZE_HEADING, bold=True),
        body=pygame.font.SysFont(FONT_NAME, FONT_SIZE_BODY),
        small=pygame.font.SysFont(FONT_NAME, FONT_SIZE_SMALL),
        status=pygame.font.SysFont(FONT_NAME, FONT_SIZE_STATUS, bold=True),
        coordinate=pygame.font.SysFont(FONT_NAME, FONT_SIZE_SMALL, bold=True),
    )
