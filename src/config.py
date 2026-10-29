"""Binary locations, model paths and tuning constants for the evaluators.

Every path may be overridden with an environment variable so the same code runs
against Homebrew binaries on a workstation and vendored binaries in CI.
Resolution order for an engine binary: env override -> ``<repo>/bin/<name>`` -> ``$PATH``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
BIN_DIR: Final[Path] = PROJECT_ROOT / "bin"

# --- Stockfish -------------------------------------------------------------
STOCKFISH_ENV_VAR: Final[str] = "STOCKFISH_PATH"
DEFAULT_STOCKFISH_DEPTH: Final[int] = 12
STOCKFISH_THREADS: Final[int] = 1
STOCKFISH_HASH_MB: Final[int] = 128

# --- Maia ------------------------------------------------------------------
DEFAULT_MAIA_RATING: Final[int] = 1100
# Rating bands of the opening books' Polyglot ``learn`` masks: one bit per band.
AVAILABLE_MAIA_RATINGS: Final[tuple[int, ...]] = (1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900)

# Post-hoc reshaping of the extracted priors. 1.0 is the identity and preserves
# Maia's calibration; >1.0 flattens (more exploratory opponent model), <1.0
# sharpens (more deterministic opponent model).
DEFAULT_MAIA_TEMPERATURE: Final[float] = 1.0

# --- Opening books ---------------------------------------------------------
BOOKS_DIR: Final[Path] = PROJECT_ROOT / "src" / "engine" / "books"
TRAP_BOOK_ENV_VAR: Final[str] = "TRAP_BOOK_PATH"
STANDARD_BOOK_ENV_VAR: Final[str] = "STANDARD_BOOK_PATH"

# --- Lichess ---------------------------------------------------------------
LICHESS_TOKEN_ENV_VAR: Final[str] = "LICHESS_API_TOKEN"

# --- Shared ----------------------------------------------------------------
ENGINE_STARTUP_TIMEOUT: Final[float] = 30.0
WIN_PROBABILITY_SCALE: Final[float] = 400.0
MATE_SCORE_CP: Final[int] = 10_000
PROBABILITY_SUM_TOLERANCE: Final[float] = 1e-6


def _resolve_binary(env_var: str, name: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{env_var}={override!r} does not exist")
        return path

    vendored = BIN_DIR / name
    if vendored.exists():
        return vendored

    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)

    raise FileNotFoundError(
        f"Could not locate the {name!r} binary. Install it, drop it in {BIN_DIR}, "
        f"or set {env_var} to its absolute path."
    )


def stockfish_binary() -> Path:
    """Absolute path to the Stockfish executable."""
    return _resolve_binary(STOCKFISH_ENV_VAR, "stockfish")


def trap_book_path() -> Path:
    """Curated gambit book, tried first. Missing is fine; the book degrades to empty."""
    return Path(os.environ.get(TRAP_BOOK_ENV_VAR, BOOKS_DIR / "traps.bin")).expanduser()


def standard_book_path() -> Path:
    """Broad GM book, used when the trap book has nothing for the position."""
    return Path(os.environ.get(STANDARD_BOOK_ENV_VAR, BOOKS_DIR / "standard.bin")).expanduser()


def _dotenv_value(key: str) -> str:
    """``key`` from the project's ``.env``, or empty.

    Read on demand rather than loaded into ``os.environ``, so the token never
    reaches child processes that do not ask for it. A background job launched
    without sourcing ``.env`` failed on this once; the environment still wins.
    """
    try:
        lines = (PROJECT_ROOT / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() == key:
            return value.strip().strip("'\"")
    return ""


def lichess_token() -> str:
    """The Lichess bot API token from the environment.

    Never accepted as a command-line argument: tokens on an argv line leak into
    shell history and process listings.
    """
    token = os.environ.get(LICHESS_TOKEN_ENV_VAR, "").strip() or _dotenv_value(
        LICHESS_TOKEN_ENV_VAR
    )
    if not token:
        raise RuntimeError(
            f"{LICHESS_TOKEN_ENV_VAR} is unset. Create a token with the 'bot:play' scope at "
            "https://lichess.org/account/oauth/token and export it or put it in .env."
        )
    return token


def nearest_maia_rating(rating: int) -> int:
    """The closest available Maia checkpoint to an opponent rating.

    Ties break downward -- modelling an opponent as slightly weaker than they are
    costs less than assuming they will find the refutation.
    """
    return min(AVAILABLE_MAIA_RATINGS, key=lambda available: (abs(available - rating), available))
