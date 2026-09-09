"""Lichess bot bridge: event streaming, challenge handling and the game loop.

**One game at a time, deliberately.** The bridge declines challenges while a
game is running. This is not a simplification, it is a correctness requirement:
the opponent model is per-game state (``MaiaEvaluator.set_rating``) living on a
single process-wide Lc0 instance, and Stockfish is likewise one process. Two
concurrent games would silently swap each other's Maia weights mid-search --
game A against a 1200 would start using the 1900 model the moment game B began,
with nothing in the logs to show for it. Serving one game also keeps move
latency predictable, which is what stops us flagging.

The game still runs on its own thread so the event stream keeps draining;
without that, challenge and ``gameFinish`` events would queue behind the game.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterator, Mapping, Optional, Protocol

import chess
import requests
from berserk.exceptions import ResponseError
from berserk.types.challenges import ChallengeDeclineReason

from src import config as engine_config
from src.engine import EvaluatorError
from src.engine.search import AdversarialSearcher
from src.lichess.time_manager import TimeManager

__all__ = ["BotConfig", "LichessBot", "LichessClient", "RatingAdaptiveModel", "main"]

logger = logging.getLogger(__name__)

SUPPORTED_VARIANTS: Final[frozenset[str]] = frozenset({"standard", "fromPosition"})
STARTING_POSITION: Final[str] = "startpos"
RATE_LIMIT_WAIT_SECONDS: Final[float] = 60.0
"""Lichess asks clients to back off a full minute after a 429."""

DEFAULT_MIN_INITIAL_SECONDS: Final[float] = 60.0
"""Below this the search cannot move quickly enough to be worth playing."""

DEFAULT_OPPONENT_RATING: Final[int] = 1500
"""Used when the opponent has no rating: Lichess AI, or a provisional new account."""

NETWORK_ERRORS: Final[tuple[type[BaseException], ...]] = (requests.RequestException, OSError)


class BotsApi(Protocol):
    """The slice of ``berserk.Client.bots`` this bridge uses."""

    def stream_incoming_events(self) -> Iterator[Mapping[str, Any]]: ...

    def stream_game_state(self, game_id: str) -> Iterator[Mapping[str, Any]]: ...

    def make_move(self, game_id: str, move: str) -> None: ...

    def accept_challenge(self, challenge_id: str) -> None: ...

    def decline_challenge(
        self, challenge_id: str, reason: ChallengeDeclineReason = "generic"
    ) -> None: ...

    def resign_game(self, game_id: str) -> None: ...


class AccountApi(Protocol):
    def get(self) -> Mapping[str, Any]: ...


class LichessClient(Protocol):
    """Structural view of ``berserk.Client``, so tests can supply a fake."""

    @property
    def bots(self) -> BotsApi: ...

    @property
    def account(self) -> AccountApi: ...


class RatingAdaptiveModel(Protocol):
    """A human model whose checkpoint can be swapped between games."""

    rating: int

    def set_rating(self, rating: int) -> None: ...


@dataclass(frozen=True, slots=True)
class BotConfig:
    """Challenge policy and network behaviour."""

    min_initial_seconds: float = DEFAULT_MIN_INITIAL_SECONDS
    max_initial_seconds: float = 5400.0
    accept_rated: bool = True
    accept_casual: bool = True
    reconnect_backoff_seconds: float = 2.0
    max_reconnect_backoff_seconds: float = 60.0
    move_attempts: int = 3


@dataclass(slots=True)
class GameSession:
    """Everything the game loop tracks for one game."""

    game_id: str
    my_color: chess.Color
    initial_fen: str
    board: chess.Board
    opponent_name: str
    opponent_rating: int
    maia_rating: int
    moves_played: int = field(default=0)


class LichessBot:
    """Streams Lichess events and plays the accepted game with the local engine."""

    def __init__(
        self,
        client: LichessClient,
        searcher: AdversarialSearcher,
        human_model: RatingAdaptiveModel,
        *,
        time_manager: Optional[TimeManager] = None,
        config: Optional[BotConfig] = None,
        bot_id: Optional[str] = None,
    ) -> None:
        self.client = client
        self.searcher = searcher
        self.human_model = human_model
        self.time_manager = time_manager if time_manager is not None else TimeManager()
        self.config = config if config is not None else BotConfig()

        self._bot_id = bot_id
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._reservation: Optional[str] = None
        self._game_thread: Optional[threading.Thread] = None

    # -- lifecycle ----------------------------------------------------------

    @property
    def bot_id(self) -> str:
        """This account's Lichess id, fetched once and cached."""
        if self._bot_id is None:
            self._bot_id = str(self.client.account.get()["id"]).lower()
        return self._bot_id

    def run(self) -> None:
        """Stream events until :meth:`stop` is called. Reconnects on failure."""
        logger.info("lichess: signed in as %s", self.bot_id)
        backoff = self.config.reconnect_backoff_seconds

        while not self._stop.is_set():
            try:
                logger.info("lichess: opening the incoming event stream")
                for event in self.client.bots.stream_incoming_events():
                    if self._stop.is_set():
                        return
                    self._handle_event(event)
                logger.warning("lichess: event stream closed by the server")
                backoff = self.config.reconnect_backoff_seconds
            except ResponseError as exc:
                backoff = self._after_response_error("event stream", exc, backoff)
            except NETWORK_ERRORS as exc:
                logger.warning("lichess: event stream dropped (%s), retrying in %.0fs", exc, backoff)

            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, self.config.max_reconnect_backoff_seconds)

        logger.info("lichess: event loop stopped")

    def stop(self) -> None:
        """Ask the event loop to exit. The in-flight game is left to finish."""
        self._stop.set()

    def _after_response_error(self, context: str, exc: ResponseError, backoff: float) -> float:
        if exc.status_code == 429:
            wait = self._retry_after(exc, RATE_LIMIT_WAIT_SECONDS)
            logger.warning("lichess: rate limited on %s, backing off %.0fs", context, wait)
            return wait
        logger.warning("lichess: %s failed with HTTP %s (%s)", context, exc.status_code, exc)
        return backoff

    @staticmethod
    def _retry_after(exc: ResponseError, default: float) -> float:
        raw = exc.response.headers.get("Retry-After") if exc.response is not None else None
        try:
            return max(float(raw), 0.0) if raw is not None else default
        except (TypeError, ValueError):
            return default

    # -- event routing ------------------------------------------------------

    def _handle_event(self, event: Mapping[str, Any]) -> None:
        kind = event.get("type")
        if kind == "challenge":
            challenge = event.get("challenge")
            if isinstance(challenge, Mapping):
                self._handle_challenge(challenge)
        elif kind == "gameStart":
            game = event.get("game")
            if isinstance(game, Mapping):
                self._handle_game_start(game)
        elif kind == "gameFinish":
            logger.debug("lichess: gameFinish %s", _game_id(event.get("game")))
        else:
            logger.debug("lichess: ignoring %s event", kind)

    # -- challenges ---------------------------------------------------------

    def _handle_challenge(self, challenge: Mapping[str, Any]) -> None:
        challenge_id = str(challenge.get("id", ""))
        challenger = challenge.get("challenger")
        who = str(challenger.get("name", "?")) if isinstance(challenger, Mapping) else "?"
        if not challenge_id:
            return
        if challenge.get("direction") == "out":
            return  # One of ours, not an invitation.

        reason = self._decline_reason(challenge)
        if reason is None and not self._reserve(challenge_id):
            reason = "later"

        if reason is not None:
            logger.info("challenge %s from %s: declined (%s)", challenge_id, who, reason)
            self._call_api(
                "decline", lambda: self.client.bots.decline_challenge(challenge_id, reason or "generic")
            )
            return

        logger.info("challenge %s from %s: accepted", challenge_id, who)
        if not self._call_api("accept", lambda: self.client.bots.accept_challenge(challenge_id)):
            self._release(challenge_id)

    def _decline_reason(self, challenge: Mapping[str, Any]) -> Optional[ChallengeDeclineReason]:
        """``None`` to accept, otherwise the Lichess decline reason to send."""
        variant = challenge.get("variant")
        key = variant.get("key") if isinstance(variant, Mapping) else None
        if key not in SUPPORTED_VARIANTS:
            return "variant"

        rated = bool(challenge.get("rated", False))
        if rated and not self.config.accept_rated:
            return "rated"
        if not rated and not self.config.accept_casual:
            return "casual"

        control = challenge.get("timeControl")
        if not isinstance(control, Mapping) or control.get("type") != "clock":
            return "timeControl"  # Correspondence and unlimited games have no clock to manage.

        # Challenge clocks are quoted in seconds, unlike the game stream's
        # milliseconds. Same field name, different unit, different endpoint.
        initial = float(control.get("limit", 0))
        if initial < self.config.min_initial_seconds:
            return "tooFast"
        if initial > self.config.max_initial_seconds:
            return "tooSlow"
        return None

    # -- single-game slot ---------------------------------------------------

    def _reserve(self, key: str) -> bool:
        with self._lock:
            if self._reservation is not None:
                return False
            if self._game_thread is not None and self._game_thread.is_alive():
                return False
            self._reservation = key
            return True

    def _release(self, key: str) -> None:
        with self._lock:
            if self._reservation == key:
                self._reservation = None

    def _handle_game_start(self, game: Mapping[str, Any]) -> None:
        game_id = _game_id(game)
        if not game_id:
            return
        with self._lock:
            if self._game_thread is not None and self._game_thread.is_alive():
                logger.error(
                    "game %s started while %s is still running; leaving it unplayed",
                    game_id, self._reservation,
                )
                return
            self._reservation = game_id
            thread = threading.Thread(
                target=self._run_game, args=(game_id,), name=f"lichess-game-{game_id}", daemon=True
            )
            self._game_thread = thread
        thread.start()

    def _run_game(self, game_id: str) -> None:
        try:
            self.play_game(game_id)
        finally:
            self._release(game_id)

    # -- game loop ----------------------------------------------------------

    def play_game(self, game_id: str) -> None:
        """Play one game to completion, streaming its state.

        Safe to call directly; the event loop runs it on its own thread.
        """
        logger.info("game %s: joining", game_id)
        session: Optional[GameSession] = None
        try:
            for event in self.client.bots.stream_game_state(game_id):
                if self._stop.is_set():
                    logger.info("game %s: shutting down, leaving the game", game_id)
                    return
                kind = event.get("type")
                if kind == "gameFull":
                    session = self._start_session(game_id, event)
                    state = event.get("state")
                    if isinstance(state, Mapping) and not self._advance(session, state):
                        return
                elif kind == "gameState":
                    if session is None:
                        logger.warning("game %s: gameState before gameFull, ignoring", game_id)
                        continue
                    if not self._advance(session, event):
                        return
                else:
                    logger.debug("game %s: ignoring %s event", game_id, kind)
        except ResponseError as exc:
            logger.error("game %s: stream failed with HTTP %s (%s)", game_id, exc.status_code, exc)
        except NETWORK_ERRORS as exc:
            logger.error("game %s: stream dropped (%s)", game_id, exc)
        finally:
            logger.info("game %s: loop finished", game_id)

    def _start_session(self, game_id: str, event: Mapping[str, Any]) -> GameSession:
        white = event.get("white") if isinstance(event.get("white"), Mapping) else {}
        black = event.get("black") if isinstance(event.get("black"), Mapping) else {}
        assert isinstance(white, Mapping) and isinstance(black, Mapping)

        my_color = chess.WHITE if str(white.get("id", "")).lower() == self.bot_id else chess.BLACK
        opponent = black if my_color == chess.WHITE else white
        opponent_name = str(opponent.get("name") or opponent.get("id") or "anonymous")
        opponent_rating = _rating_of(opponent)

        initial_fen = str(event.get("initialFen") or STARTING_POSITION)
        board = chess.Board() if initial_fen == STARTING_POSITION else chess.Board(initial_fen)

        maia_rating = engine_config.nearest_maia_rating(opponent_rating)
        try:
            self.human_model.set_rating(maia_rating)
        except EvaluatorError as exc:
            logger.error("game %s: could not load maia-%d (%s), keeping maia-%d",
                         game_id, maia_rating, exc, self.human_model.rating)
            maia_rating = self.human_model.rating
        # The book bands off the same rating: traps the opponent is likely to
        # see through are simply not offered.
        if self.searcher.book is not None:
            self.searcher.book.set_opponent_rating(opponent_rating)

        logger.info(
            "game %s: playing %s against %s (%d) -> opponent model maia-%d",
            game_id,
            "white" if my_color == chess.WHITE else "black",
            opponent_name,
            opponent_rating,
            maia_rating,
        )
        return GameSession(
            game_id=game_id,
            my_color=my_color,
            initial_fen=initial_fen,
            board=board,
            opponent_name=opponent_name,
            opponent_rating=opponent_rating,
            maia_rating=maia_rating,
        )

    def _advance(self, session: GameSession, state: Mapping[str, Any]) -> bool:
        """Apply one game state. Returns ``False`` once the game is over."""
        self._sync_board(session, str(state.get("moves", "")))

        status = str(state.get("status", "started"))
        if status != "started":
            logger.info(
                "game %s: finished (%s%s) after %d moves",
                session.game_id, status,
                f", {state['winner']} wins" if state.get("winner") else "",
                session.moves_played,
            )
            return False
        if session.board.is_game_over(claim_draw=True):
            logger.info("game %s: position is terminal, waiting for the server", session.game_id)
            return True
        if session.board.turn != session.my_color:
            return True

        search_config = self.time_manager.calculate_search_config(
            state.get("wtime", 0),
            state.get("btime", 0),
            state.get("winc", 0),
            state.get("binc", 0),
            is_white=session.my_color == chess.WHITE,
        )
        try:
            result = self.searcher.search(session.board, search_config)
        except EvaluatorError as exc:
            logger.error("game %s: search failed (%s), resigning", session.game_id, exc)
            self._call_api("resign", lambda: self.client.bots.resign_game(session.game_id))
            return False

        san = session.board.san(result.move)
        logger.info(
            "game %s: playing %s (depth %d/%d) %s",
            session.game_id, san, search_config.root_depth, search_config.leaf_depth, result.summary(),
        )
        if not self._submit_move(session.game_id, result.move.uci()):
            logger.error("game %s: could not submit %s, abandoning the game", session.game_id, san)
            return False
        return True

    def _sync_board(self, session: GameSession, moves_text: str) -> None:
        """Rebuild the position from the authoritative move list.

        Replayed from scratch rather than pushed incrementally: after a stream
        reconnect Lichess resends state, and a bot that assumes it only ever
        gains one move desynchronises and starts sending illegal moves.
        """
        board = session.board
        if session.initial_fen == STARTING_POSITION:
            board.reset()
        else:
            board.set_fen(session.initial_fen)
        moves = moves_text.split()
        for uci in moves:
            try:
                board.push_uci(uci)
            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                logger.error("game %s: cannot replay move %r, stopping replay", session.game_id, uci)
                break
        session.moves_played = len(moves)

    # -- API helpers --------------------------------------------------------

    def _submit_move(self, game_id: str, uci: str) -> bool:
        for attempt in range(1, self.config.move_attempts + 1):
            try:
                self.client.bots.make_move(game_id, uci)
                return True
            except ResponseError as exc:
                if exc.status_code == 429:
                    wait = self._retry_after(exc, RATE_LIMIT_WAIT_SECONDS)
                    logger.warning("game %s: rate limited, waiting %.0fs", game_id, wait)
                    time.sleep(wait)
                    continue
                # 4xx here means the server rejected the move outright -- the game
                # ended or moved on. Retrying cannot help and burns clock.
                if 400 <= exc.status_code < 500:
                    logger.error("game %s: move %s rejected (HTTP %s)", game_id, uci, exc.status_code)
                    return False
                logger.warning("game %s: move %s failed (HTTP %s), attempt %d/%d",
                               game_id, uci, exc.status_code, attempt, self.config.move_attempts)
            except NETWORK_ERRORS as exc:
                logger.warning("game %s: move %s failed (%s), attempt %d/%d",
                               game_id, uci, exc, attempt, self.config.move_attempts)
            time.sleep(min(2.0**attempt, self.config.max_reconnect_backoff_seconds))
        return False

    def _call_api(self, what: str, action: Callable[[], None]) -> bool:
        """Run a one-shot API call, logging rather than raising on failure."""
        try:
            action()
            return True
        except ResponseError as exc:
            logger.warning("lichess: %s failed with HTTP %s (%s)", what, exc.status_code, exc)
        except NETWORK_ERRORS as exc:
            logger.warning("lichess: %s failed (%s)", what, exc)
        return False


def _game_id(game: Any) -> str:
    if not isinstance(game, Mapping):
        return ""
    return str(game.get("gameId") or game.get("id") or "")


def _rating_of(player: Mapping[str, Any]) -> int:
    rating = player.get("rating")
    if isinstance(rating, (int, float)) and rating > 0:
        return int(rating)
    if "aiLevel" in player:
        logger.debug("opponent is the Lichess AI, using the default rating")
    return DEFAULT_OPPONENT_RATING


def main() -> int:
    """Entry point: ``python -m src.lichess.bot``."""
    import argparse

    import berserk

    from src.engine import MaiaEvaluator, StockfishEvaluator
    from src.engine.book import OpeningBook

    parser = argparse.ArgumentParser(prog="python -m src.lichess.bot", description="Lichess bot bridge.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--min-clock", type=float, default=DEFAULT_MIN_INITIAL_SECONDS,
                        help="Decline games with a shorter initial clock (seconds).")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("chess.engine").setLevel(logging.WARNING)

    try:
        token = engine_config.lichess_token()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)

    with (
        StockfishEvaluator() as stockfish,
        MaiaEvaluator(engine_config.DEFAULT_MAIA_RATING) as maia,
        OpeningBook() as book,
    ):
        bot = LichessBot(
            client,
            AdversarialSearcher(stockfish, maia, book=book),
            maia,
            config=BotConfig(min_initial_seconds=args.min_clock),
        )
        try:
            bot.run()
        except KeyboardInterrupt:
            logger.info("lichess: interrupted, shutting down")
            bot.stop()
    logger.info("lichess: engine processes terminated, books unmapped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
