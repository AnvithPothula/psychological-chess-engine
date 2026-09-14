"""Autosaved UI session: settings, counters and the game in progress.

Everything the window knows that is not reconstructible from the repository
lives here and is written to disk as it changes, so closing the window -- or
losing it -- costs nothing. Writes are debounced rather than issued per frame:
at 60fps a naive save would be sixty file writes a second, and the interesting
state changes a few times a minute.

Mined preference pairs are *not* stored here. The generator already appends and
flushes them per game, which is a better durability story than anything a UI
layer could add on top.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

import chess

__all__ = ["MiningSettings", "SessionState", "SessionStore"]

logger = logging.getLogger(__name__)

DEFAULT_SESSION_PATH: Final[Path] = Path("build/session.json")
AUTOSAVE_INTERVAL_SECONDS: Final[float] = 2.0
SESSION_VERSION: Final[int] = 1
PGN_DIRECTORY: Final[Path] = Path("build/games")


@dataclass
class MiningSettings:
    """What the mining tab is configured to do. Persisted verbatim."""

    rating: int = 1300
    games: int = 50
    max_plies: int = 80
    output: str = "build/dpo_pairs.jsonl"


@dataclass
class SessionState:
    """Everything restored on the next launch."""

    version: int = SESSION_VERSION
    active_tab: str = "play"
    mining: MiningSettings = field(default_factory=MiningSettings)
    pairs_mined_total: int = 0
    games_mined_total: int = 0
    game_start_fen: str = chess.STARTING_FEN
    game_moves: List[str] = field(default_factory=list)
    human_is_white: bool = True


class SessionStore:
    """Loads, holds and debounce-writes :class:`SessionState`."""

    def __init__(self, path: Optional[Path] = DEFAULT_SESSION_PATH) -> None:
        """``path=None`` gives an in-memory store that never touches disk.

        Persistence is opt-in on purpose. A store that loaded by default would
        make constructing the app implicitly stateful -- every caller would
        silently resume whatever game was last saved, which is surprising for a
        library and wrong for a test.
        """
        self.path = path
        self.state = self._load(path) if path is not None else SessionState()
        self._dirty = False
        self._last_write = time.monotonic()

    @staticmethod
    def _load(path: Path) -> SessionState:
        if not path.exists():
            return SessionState()
        try:
            payload: Dict[str, Any] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("session: %s is unreadable (%s), starting fresh", path, exc)
            return SessionState()
        if int(payload.get("version", 0)) != SESSION_VERSION:
            logger.info("session: %s is from an older layout, starting fresh", path.name)
            return SessionState()

        mining = payload.get("mining", {})
        state = SessionState(
            active_tab=str(payload.get("active_tab", "play")),
            mining=MiningSettings(
                rating=int(mining.get("rating", 1300)),
                games=int(mining.get("games", 50)),
                max_plies=int(mining.get("max_plies", 80)),
                output=str(mining.get("output", "build/dpo_pairs.jsonl")),
            ),
            pairs_mined_total=int(payload.get("pairs_mined_total", 0)),
            games_mined_total=int(payload.get("games_mined_total", 0)),
            game_start_fen=str(payload.get("game_start_fen", chess.STARTING_FEN)),
            game_moves=[str(move) for move in payload.get("game_moves", [])],
            human_is_white=bool(payload.get("human_is_white", True)),
        )
        logger.info(
            "session: restored from %s (%d saved moves, %d pairs mined all time)",
            path.name, len(state.game_moves), state.pairs_mined_total,
        )
        return state

    def touch(self) -> None:
        """Mark the state changed; the next :meth:`tick` will write it."""
        self._dirty = True

    def record_game(self, board: chess.Board, human_is_white: bool) -> None:
        """Snapshot the game in progress so it survives a restart."""
        root = board.root()
        self.state.game_start_fen = root.fen()
        self.state.game_moves = [move.uci() for move in board.move_stack]
        self.state.human_is_white = human_is_white
        self.touch()

    def restore_board(self) -> Optional[chess.Board]:
        """The saved game, replayed. ``None`` when there is nothing to resume."""
        if not self.state.game_moves:
            return None
        try:
            board = chess.Board(self.state.game_start_fen)
            for uci in self.state.game_moves:
                board.push_uci(uci)
        except ValueError as exc:
            logger.warning("session: saved game is not replayable (%s), discarding it", exc)
            self.clear_game()
            return None
        return board

    def clear_game(self) -> None:
        self.state.game_start_fen = chess.STARTING_FEN
        self.state.game_moves = []
        self.touch()

    def tick(self) -> None:
        """Write if something changed and the debounce window has elapsed."""
        if self.path is None or not self._dirty:
            return
        if time.monotonic() - self._last_write < AUTOSAVE_INTERVAL_SECONDS:
            return
        self.flush()

    def flush(self) -> None:
        """Write immediately. Called on every exit path, clean or not."""
        if self.path is None or not self._dirty:
            return
        payload = asdict(self.state)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write-and-rename: a crash mid-write must not leave a truncated
            # session file that the next launch then refuses to load.
            staging = self.path.with_suffix(".part")
            staging.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(staging, self.path)
        except OSError as exc:
            logger.warning("session: could not save to %s (%s)", self.path, exc)
            return
        self._dirty = False
        self._last_write = time.monotonic()
        logger.debug("session: saved to %s", self.path)

    def archive_pgn(
        self, board: chess.Board, human_is_white: bool, directory: Optional[Path] = None
    ) -> Optional[Path]:
        """Write a finished game to ``build/games`` and forget the live copy."""
        target = directory if directory is not None else PGN_DIRECTORY
        if not board.move_stack:
            return None
        outcome = board.outcome(claim_draw=True)
        headers = {
            "Event": "Psychological Chess Engine",
            "Site": "local",
            "Date": time.strftime("%Y.%m.%d"),
            "White": "Human" if human_is_white else "Engine",
            "Black": "Engine" if human_is_white else "Human",
            "Result": outcome.result() if outcome else "*",
        }
        replay = chess.Board(self.state.game_start_fen)
        moves: List[str] = []
        for move in board.move_stack:
            moves.append(replay.san(move))
            replay.push(move)

        body = " ".join(
            f"{index // 2 + 1}. {san}" if index % 2 == 0 else san
            for index, san in enumerate(moves)
        )
        text = "".join(f'[{key} "{value}"]\n' for key, value in headers.items())
        text += f"\n{body} {headers['Result']}\n"

        try:
            target.mkdir(parents=True, exist_ok=True)
            path = target / f"game-{time.strftime('%Y%m%d-%H%M%S')}.pgn"
            path.write_text(text)
        except OSError as exc:
            logger.warning("session: could not archive the game (%s)", exc)
            return None
        logger.info("session: archived %s", path)
        self.clear_game()
        return path
