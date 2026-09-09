"""Board rendering and click-to-move hit testing.

Square colours, highlight colours and the legal-move indicator geometry follow
the ChessAI client: a filled translucent dot for a quiet move, a translucent
ring for a capture, and the same warm highlight on the selected square and on
both squares of the last move.
"""

from __future__ import annotations

from typing import Dict, Final, Optional, Sequence, Tuple

import chess
import pygame

from src.ui.constants import (
    BOARD_BORDER,
    BOARD_ORIGIN,
    BOARD_SIZE,
    BOARD_SQUARES,
    CHECK_HIGHLIGHT,
    COORDINATE_INSET,
    DARK_SQUARE,
    DARK_SQUARE_HIGHLIGHT,
    LEGAL_CAPTURE_RING,
    LEGAL_DOT_RADIUS_RATIO,
    LEGAL_MOVE_DOT,
    LEGAL_RING_RADIUS_RATIO,
    LEGAL_RING_WIDTH_RATIO,
    LIGHT_SQUARE,
    LIGHT_SQUARE_HIGHLIGHT,
    SQUARE_SIZE,
)

__all__ = ["BoardView"]

BORDER_WIDTH: Final[int] = 2


class BoardView:
    """Draws the board and maps window coordinates back to squares.

    Holds no game state: every draw is a pure function of the board handed in.
    """

    def __init__(
        self,
        pieces: Dict[str, pygame.Surface],
        coordinate_font: pygame.font.Font,
        *,
        origin: Tuple[int, int] = BOARD_ORIGIN,
        square_size: int = SQUARE_SIZE,
        orientation: chess.Color = chess.WHITE,
    ) -> None:
        self.pieces = pieces
        self.coordinate_font = coordinate_font
        self.origin = origin
        self.square_size = square_size
        self.orientation = orientation
        self.rect = pygame.Rect(origin[0], origin[1], square_size * BOARD_SQUARES, square_size * BOARD_SQUARES)
        self._dot = self._build_dot_overlay(square_size)
        self._ring = self._build_ring_overlay(square_size)
        self._check = self._build_check_overlay(square_size)

    # -- hit testing --------------------------------------------------------

    def square_at(self, position: Tuple[int, int]) -> Optional[chess.Square]:
        """The square under a window coordinate, or ``None`` if outside the board."""
        if not self.rect.collidepoint(position):
            return None
        column = (position[0] - self.origin[0]) // self.square_size
        row = (position[1] - self.origin[1]) // self.square_size
        column = min(max(column, 0), BOARD_SQUARES - 1)
        row = min(max(row, 0), BOARD_SQUARES - 1)
        if self.orientation == chess.WHITE:
            return chess.square(column, BOARD_SQUARES - 1 - row)
        return chess.square(BOARD_SQUARES - 1 - column, row)

    def square_rect(self, square: chess.Square) -> pygame.Rect:
        """Window rectangle covering ``square`` at the current orientation."""
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if self.orientation == chess.WHITE:
            column, row = file_index, BOARD_SQUARES - 1 - rank_index
        else:
            column, row = BOARD_SQUARES - 1 - file_index, rank_index
        return pygame.Rect(
            self.origin[0] + column * self.square_size,
            self.origin[1] + row * self.square_size,
            self.square_size,
            self.square_size,
        )

    # -- rendering ----------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        board: chess.Board,
        *,
        selected: Optional[chess.Square] = None,
        legal_moves: Sequence[chess.Move] = (),
        last_move: Optional[chess.Move] = None,
    ) -> None:
        highlighted = self._highlighted_squares(selected, last_move)
        self._draw_squares(surface, highlighted)
        self._draw_check(surface, board)
        self._draw_coordinates(surface)
        self._draw_pieces(surface, board)
        self._draw_move_hints(surface, board, legal_moves)
        pygame.draw.rect(surface, BOARD_BORDER, self.rect.inflate(BORDER_WIDTH * 2, BORDER_WIDTH * 2), BORDER_WIDTH)

    @staticmethod
    def _highlighted_squares(
        selected: Optional[chess.Square], last_move: Optional[chess.Move]
    ) -> frozenset[chess.Square]:
        squares: set[chess.Square] = set()
        if selected is not None:
            squares.add(selected)
        if last_move is not None:
            squares.update((last_move.from_square, last_move.to_square))
        return frozenset(squares)

    def _draw_squares(self, surface: pygame.Surface, highlighted: frozenset[chess.Square]) -> None:
        for square in chess.SQUARES:
            is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            if square in highlighted:
                colour = LIGHT_SQUARE_HIGHLIGHT if is_light else DARK_SQUARE_HIGHLIGHT
            else:
                colour = LIGHT_SQUARE if is_light else DARK_SQUARE
            pygame.draw.rect(surface, colour, self.square_rect(square))

    def _draw_check(self, surface: pygame.Surface, board: chess.Board) -> None:
        if not board.is_check():
            return
        king_square = board.king(board.turn)
        if king_square is not None:
            surface.blit(self._check, self.square_rect(king_square))

    def _draw_coordinates(self, surface: pygame.Surface) -> None:
        """File letters along the bottom edge, rank numbers along the left."""
        bottom_rank = 0 if self.orientation == chess.WHITE else BOARD_SQUARES - 1
        left_file = 0 if self.orientation == chess.WHITE else BOARD_SQUARES - 1

        for index in range(BOARD_SQUARES):
            file_square = chess.square(index, bottom_rank)
            rect = self.square_rect(file_square)
            label = self.coordinate_font.render(
                chess.FILE_NAMES[index], True, self._label_colour(file_square)
            )
            surface.blit(
                label,
                (
                    rect.right - label.get_width() - COORDINATE_INSET,
                    rect.bottom - label.get_height() - COORDINATE_INSET,
                ),
            )

            rank_square = chess.square(left_file, index)
            rect = self.square_rect(rank_square)
            number = self.coordinate_font.render(
                chess.RANK_NAMES[index], True, self._label_colour(rank_square)
            )
            surface.blit(number, (rect.left + COORDINATE_INSET, rect.top + COORDINATE_INSET))

    @staticmethod
    def _label_colour(square: chess.Square) -> Tuple[int, int, int]:
        """Coordinates take the opposite square colour, so they always read."""
        is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        return DARK_SQUARE if is_light else LIGHT_SQUARE

    def _draw_pieces(self, surface: pygame.Surface, board: chess.Board) -> None:
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                surface.blit(self.pieces[piece.symbol()], self.square_rect(square))

    def _draw_move_hints(
        self, surface: pygame.Surface, board: chess.Board, legal_moves: Sequence[chess.Move]
    ) -> None:
        for move in legal_moves:
            overlay = self._ring if board.is_capture(move) else self._dot
            surface.blit(overlay, self.square_rect(move.to_square))

    # -- overlay surfaces ---------------------------------------------------
    # Built once at construction: SRCALPHA surfaces are cheap to blit but not
    # to allocate, and these are drawn every frame while a piece is selected.

    @staticmethod
    def _build_dot_overlay(square_size: int) -> pygame.Surface:
        overlay = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        centre = (square_size // 2, square_size // 2)
        pygame.draw.circle(overlay, LEGAL_MOVE_DOT, centre, int(square_size * LEGAL_DOT_RADIUS_RATIO))
        return overlay

    @staticmethod
    def _build_ring_overlay(square_size: int) -> pygame.Surface:
        overlay = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        centre = (square_size // 2, square_size // 2)
        pygame.draw.circle(
            overlay,
            LEGAL_CAPTURE_RING,
            centre,
            int(square_size * LEGAL_RING_RADIUS_RATIO),
            max(1, int(square_size * LEGAL_RING_WIDTH_RATIO)),
        )
        return overlay

    @staticmethod
    def _build_check_overlay(square_size: int) -> pygame.Surface:
        overlay = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        overlay.fill(CHECK_HIGHLIGHT)
        return overlay
