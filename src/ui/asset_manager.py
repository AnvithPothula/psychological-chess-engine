"""Obtains, caches and scales the piece image set.

Assets are resolved in reliability order, stopping at the first tier that works:

1. **Cached PNG** in ``assets/pieces``. The steady state after first run.
2. **Local rasterisation.** ``python-chess`` already ships the Cburnett set as
   SVG (``chess.svg.piece``) -- the same artwork the Wikimedia files render from.
   pygame rasterises it through SDL_image, so the pieces are generated offline
   with no network call at all. This is the primary generator.
3. **Wikimedia Commons download.** Only reached when SDL_image was built without
   SVG support. Two things about that endpoint are load-bearing: a ``User-Agent``
   header is mandatory (403 without one), and only a fixed set of thumbnail
   widths is served (400 otherwise). It also enforces a robot policy that answers
   429 to bursts, so requests are spaced and retried with backoff.
4. **Unicode glyphs.** Last resort, so a missing asset never takes down the app.

Tier 2 is preferred over the download the brief asked for because it cannot be
rate-limited, blocked, or broken by a missing CA bundle, and it produces byte-for
-byte the same piece set.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Final, List, Optional

import chess
import chess.svg
import pygame

from src.ui.constants import (
    ASSET_DOWNLOAD_ATTEMPTS,
    ASSET_DOWNLOAD_BACKOFF,
    ASSET_DOWNLOAD_DELAY,
    ASSET_DOWNLOAD_TIMEOUT,
    GLYPH_FONT_NAMES,
    TEXT_PRIMARY,
)

__all__ = ["AssetManager", "AssetError"]

logger = logging.getLogger(__name__)

DEFAULT_ASSETS_DIR: Final[Path] = Path(__file__).resolve().parent / "assets" / "pieces"
PIECE_SYMBOLS: Final[str] = "PNBRQKpnbrqk"
ASSET_SOURCE_WIDTH: Final[int] = 500
"""Rasterisation width. Generous, so downscaling to any square size stays crisp."""

COMMONS_THUMB_ROOT: Final[str] = "https://upload.wikimedia.org/wikipedia/commons/thumb"
COMMONS_ALLOWED_WIDTHS: Final[tuple[int, ...]] = (120, 250, 500)
USER_AGENT: Final[str] = "psychological-chess-engine/1.0 (local desktop client)"
SYSTEM_CA_BUNDLE: Final[Path] = Path("/etc/ssl/cert.pem")
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

GLYPH_SCALE: Final[float] = 0.82
GLYPH_OUTLINE: Final[int] = 2
GLYPH_DARK: Final[tuple[int, int, int]] = (0x10, 0x10, 0x10)


class AssetError(RuntimeError):
    """A piece image could not be produced by any tier."""


def _commons_file_name(symbol: str) -> str:
    """``'Q'`` -> ``'Chess_qlt45.svg'`` (l = light piece, d = dark piece)."""
    return f"Chess_{symbol.lower()}{'l' if symbol.isupper() else 'd'}t45.svg"


def _thumbnail_url(file_name: str, width: int) -> str:
    """Commons thumbnail URL. The path embeds an MD5 prefix of the file name."""
    if width not in COMMONS_ALLOWED_WIDTHS:
        raise ValueError(f"width must be one of {COMMONS_ALLOWED_WIDTHS}, got {width}")
    digest = hashlib.md5(file_name.encode("utf-8")).hexdigest()
    return f"{COMMONS_THUMB_ROOT}/{digest[0]}/{digest[:2]}/{file_name}/{width}px-{file_name}.png"


def _ssl_context() -> ssl.SSLContext:
    """A context with a usable CA store.

    Python installed from python.org ships no CA bundle, so the default context
    fails verification on a stock macOS install. Prefer ``certifi``, then the
    system bundle, then the default.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    if SYSTEM_CA_BUNDLE.exists():
        return ssl.create_default_context(cafile=str(SYSTEM_CA_BUNDLE))
    return ssl.create_default_context()


class AssetManager:
    """Owns the on-disk piece cache and the scaled surfaces handed to the views."""

    def __init__(self, assets_dir: Path = DEFAULT_ASSETS_DIR) -> None:
        self.assets_dir = assets_dir
        self._scaled: Dict[int, Dict[str, pygame.Surface]] = {}
        self._context: Optional[ssl.SSLContext] = None

    def local_path(self, symbol: str) -> Path:
        """Cache location for one piece, e.g. ``wQ.png`` / ``bN.png``."""
        return self.assets_dir / f"{'w' if symbol.isupper() else 'b'}{symbol.upper()}.png"

    def missing_symbols(self) -> tuple[str, ...]:
        """Piece symbols with no cached image on disk."""
        return tuple(symbol for symbol in PIECE_SYMBOLS if not self.local_path(symbol).exists())

    def ensure_assets(self) -> tuple[str, ...]:
        """Produce every missing piece image. Returns the symbols that failed.

        Never raises: an unobtainable asset degrades to a glyph rather than
        taking the application down.
        """
        missing = self.missing_symbols()
        if not missing:
            logger.debug("assets: %d piece images cached in %s", len(PIECE_SYMBOLS), self.assets_dir)
            return ()

        self.assets_dir.mkdir(parents=True, exist_ok=True)
        logger.info("assets: generating %d missing piece images", len(missing))

        failed: List[str] = []
        needs_download: List[str] = []
        for symbol in missing:
            try:
                self._write(symbol, self._rasterise(symbol))
            except (pygame.error, OSError) as exc:
                logger.debug("assets: local rasterisation of %s failed (%s)", symbol, exc)
                needs_download.append(symbol)

        for index, symbol in enumerate(needs_download):
            if index:
                time.sleep(ASSET_DOWNLOAD_DELAY)
            try:
                self._write(symbol, self._download(symbol))
            except (urllib.error.URLError, AssetError, OSError, ssl.SSLError) as exc:
                logger.warning("assets: could not obtain %s: %s", symbol, exc)
                failed.append(symbol)

        if failed:
            logger.warning(
                "assets: %d piece image(s) unavailable, falling back to Unicode glyphs", len(failed)
            )
        return tuple(failed)

    def load_pieces(self, square_size: int) -> Dict[str, pygame.Surface]:
        """Piece surfaces scaled to ``square_size``, keyed by ``chess`` symbol.

        Must be called after ``pygame.display.set_mode``: ``convert_alpha`` needs
        a display surface to convert against.
        """
        if square_size < 1:
            raise ValueError(f"square_size must be >= 1, got {square_size}")
        cached = self._scaled.get(square_size)
        if cached is not None:
            return cached
        if pygame.display.get_surface() is None:
            raise RuntimeError("load_pieces() requires an initialised display surface")

        unavailable = set(self.ensure_assets())
        surfaces: Dict[str, pygame.Surface] = {}
        for symbol in PIECE_SYMBOLS:
            path = self.local_path(symbol)
            if symbol in unavailable or not path.exists():
                surfaces[symbol] = self._glyph_surface(symbol, square_size)
                continue
            try:
                image = pygame.image.load(str(path)).convert_alpha()
            except pygame.error as exc:
                logger.warning("assets: %s is unreadable (%s), using a glyph instead", path, exc)
                surfaces[symbol] = self._glyph_surface(symbol, square_size)
                continue
            surfaces[symbol] = pygame.transform.smoothscale(image, (square_size, square_size))

        self._scaled[square_size] = surfaces
        return surfaces

    # -- tier 2: local rasterisation ----------------------------------------

    @staticmethod
    def _rasterise(symbol: str) -> bytes:
        """Render the bundled Cburnett SVG for ``symbol`` to PNG bytes.

        ``chess.svg.piece`` emits a 45-unit viewBox with no width/height, and
        SDL_image rasterises an SVG at its intrinsic size. Injecting explicit
        dimensions is what gets a high-resolution bitmap out of it.
        """
        markup = chess.svg.piece(chess.Piece.from_symbol(symbol))
        sized = markup.replace(
            "<svg ", f'<svg width="{ASSET_SOURCE_WIDTH}" height="{ASSET_SOURCE_WIDTH}" ', 1
        )
        surface = pygame.image.load(io.BytesIO(sized.encode("utf-8")), "piece.svg")
        buffer = io.BytesIO()
        pygame.image.save(surface, buffer, "piece.png")
        return buffer.getvalue()

    # -- tier 3: download ---------------------------------------------------

    def _download(self, symbol: str) -> bytes:
        if self._context is None:
            self._context = _ssl_context()
        url = _thumbnail_url(_commons_file_name(symbol), ASSET_SOURCE_WIDTH)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        delay = ASSET_DOWNLOAD_DELAY
        last: Optional[urllib.error.HTTPError] = None

        for attempt in range(1, ASSET_DOWNLOAD_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=ASSET_DOWNLOAD_TIMEOUT, context=self._context
                ) as response:
                    payload: bytes = response.read()
                    logger.debug("assets: downloaded %s (%d bytes)", symbol, len(payload))
                    return payload
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_STATUS or attempt == ASSET_DOWNLOAD_ATTEMPTS:
                    raise
                last = exc
                logger.debug(
                    "assets: HTTP %d on attempt %d/%d, retrying in %.1fs",
                    exc.code, attempt, ASSET_DOWNLOAD_ATTEMPTS, delay,
                )
                time.sleep(delay)
                delay *= ASSET_DOWNLOAD_BACKOFF

        raise AssetError(f"exhausted retries for {url}") from last

    def _write(self, symbol: str, payload: bytes) -> None:
        """Write via a temporary file: an interrupted write must not leave a
        truncated PNG that later loads as a corrupt surface."""
        destination = self.local_path(symbol)
        staging = destination.with_suffix(".part")
        staging.write_bytes(payload)
        os.replace(staging, destination)

    # -- tier 4: glyphs -----------------------------------------------------

    @staticmethod
    def _glyph_surface(symbol: str, square_size: int) -> pygame.Surface:
        """Last resort: a filled Unicode chess glyph, outlined for contrast."""
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont(GLYPH_FONT_NAMES, int(square_size * GLYPH_SCALE))
        # The lowercase entries are the solid glyphs; colour carries the side.
        glyph = chess.UNICODE_PIECE_SYMBOLS[symbol.lower()]
        fill = TEXT_PRIMARY if symbol.isupper() else GLYPH_DARK
        outline = GLYPH_DARK if symbol.isupper() else TEXT_PRIMARY

        surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        body = font.render(glyph, True, fill)
        edge = font.render(glyph, True, outline)
        centre = body.get_rect(center=(square_size // 2, square_size // 2))
        for dx in (-GLYPH_OUTLINE, 0, GLYPH_OUTLINE):
            for dy in (-GLYPH_OUTLINE, 0, GLYPH_OUTLINE):
                if dx or dy:
                    surface.blit(edge, centre.move(dx, dy))
        surface.blit(body, centre)
        return surface
