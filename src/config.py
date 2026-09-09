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
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
BIN_DIR: Final[Path] = PROJECT_ROOT / "bin"

# --- Stockfish -------------------------------------------------------------
STOCKFISH_ENV_VAR: Final[str] = "STOCKFISH_PATH"
DEFAULT_STOCKFISH_DEPTH: Final[int] = 12
STOCKFISH_THREADS: Final[int] = 1
STOCKFISH_HASH_MB: Final[int] = 128

# --- Maia / Lc0 ------------------------------------------------------------
LC0_ENV_VAR: Final[str] = "LC0_PATH"
MAIA_WEIGHTS_ENV_VAR: Final[str] = "MAIA_WEIGHTS_DIR"
DEFAULT_MAIA_RATING: Final[int] = 1100
AVAILABLE_MAIA_RATINGS: Final[tuple[int, ...]] = (1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900)

# Maia is a pure policy network: one node expands the root and yields the
# priors. Anything above 1 starts blending in MCTS, which is exactly the
# superhuman behaviour we are trying not to model.
MAIA_SEARCH_NODES: Final[int] = 1

# Lc0 applies its own softmax over the network's policy logits before reporting
# priors. Its default (1.359) is tuned for Leela search nets and flattens Maia
# away from the human move frequencies it was trained to reproduce; 1.0 reads
# the network out as trained. Treat this as a calibration knob, not a constant:
# raise it if Maia looks too confident against real human game data.
LC0_POLICY_TEMPERATURE: Final[float] = 1.0

# Post-hoc reshaping of the extracted priors. 1.0 is the identity and preserves
# Maia's calibration; >1.0 flattens (more exploratory opponent model), <1.0
# sharpens (more deterministic opponent model).
DEFAULT_MAIA_TEMPERATURE: Final[float] = 1.0

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


def lc0_binary() -> Path:
    """Absolute path to the Lc0 executable used to host the Maia weights."""
    return _resolve_binary(LC0_ENV_VAR, "lc0")


def maia_weights(rating: int = DEFAULT_MAIA_RATING) -> Path:
    """Absolute path to the ``maia-<rating>.pb.gz`` weights file."""
    if rating not in AVAILABLE_MAIA_RATINGS:
        raise ValueError(f"No Maia weights for rating {rating}; have {AVAILABLE_MAIA_RATINGS}")

    directory = Path(os.environ.get(MAIA_WEIGHTS_ENV_VAR, MODELS_DIR)).expanduser()
    path = directory / f"maia-{rating}.pb.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Maia weights {path}. Download them into {directory} or set {MAIA_WEIGHTS_ENV_VAR}."
        )
    return path
