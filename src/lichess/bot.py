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

**Each game is a coin flip between two books.** Every opponent in the accepted
band meets either the standard book or the skew repertoire, chosen at random,
and the choice is logged with the game id. A live run that played only the
repertoire would win nearly every game either way and could not say whether the
book did anything; the randomised arm is what makes it an experiment.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Final, Iterator, List, Mapping, Optional, Protocol, Tuple

import berserk
import chess
import requests
from berserk.exceptions import ApiError, ResponseError
from berserk.types.challenges import ChallengeDeclineReason

from src import config as engine_config
from src.engine import EvaluatorError
from src.engine.book import OpeningBook
from src.engine.bot_factory import SKEW, STANDARD, build_book, for_humans
from src.engine.policy_generator import load_proposer
from src.engine.search import AdversarialSearcher, TerminalPositionError
from src.lichess.time_manager import AdaptivePacingController, Cadence, TimeManager, clock_seconds
from src.types import MoveSource

__all__ = ["BotConfig", "LichessBot", "LichessClient", "RatingAdaptiveModel", "main"]

logger = logging.getLogger(__name__)

SUPPORTED_VARIANTS: Final[frozenset[str]] = frozenset({"standard", "fromPosition"})
STARTING_POSITION: Final[str] = "startpos"
RATE_LIMIT_WAIT_SECONDS: Final[float] = 60.0
"""Lichess asks clients to back off a full minute after a 429."""

STABLE_STREAM_SECONDS: Final[float] = 60.0
"""A stream the server closes sooner than this was not a healthy connection, and
reopening it does not reset the backoff. Without this a stream closed on open
was reopened every two seconds indefinitely -- seen with a second bot process
running on the same account."""

DEFAULT_MIN_INITIAL_SECONDS: Final[float] = 60.0
"""Below this the search cannot move quickly enough to be worth playing."""

DEFAULT_OPPONENT_RATING: Final[int] = 1500
"""Used when the opponent has no rating: Lichess AI, or a provisional new account."""

NETWORK_ERRORS: Final[tuple[type[BaseException], ...]] = (requests.RequestException, OSError, ApiError)
"""ApiError is how berserk wraps a dropped connection. Without it here, a server
closing an idle keep-alive connection as a move was submitted killed the game
thread, and the bot lost two games on time without moving again. Every handler
catches ResponseError -- an ApiError subclass -- first."""

READ_TIMEOUT_SECONDS: Final[float] = 20.0
"""Silence on a connection longer than this means it is dead. Lichess streams
send a keep-alive line every 7.0s (measured), so 20s is three missed. Without a
timeout a half-open connection hangs forever: twice the game stream went quiet
for ten minutes after the bot's move, the opponent's reply never arrived, and
the bot lost on time with its clock untouched."""

CONNECT_TIMEOUT_SECONDS: Final[float] = 10.0


class TimedTokenSession(berserk.TokenSession):
    """berserk's session, with a connect and read timeout on every request that
    does not set its own. berserk sets none."""

    def request(self, method: Any, url: Any, *args: Any, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS))
        return super().request(method, url, *args, **kwargs)


LOW_CLOCK_SECONDS: Final[float] = 30.0
LOW_CLOCK_BOOST: Final[float] = 2.0
"""When a human is under 30s, narrow paths count double. Human blunder rates
are flat above about 10s and jump below it (24% under 10s against under 5% at
2-3 minutes, elite blitz), and low time and an ambiguous position compound each
other; 30s is when there is still time to steer into one."""

STALL_CHECK_SECONDS: Final[float] = 15.0
"""Silence on a game stream after which Lichess is asked whether the bot owes
a move. A read timeout cannot see this failure: in KYmceMVK the stream kept
sending keep-alives for ten minutes but no game updates, until the server
reset it with the bot's clock at zero. tq8SjaTx and wczLemzW were the same."""

_STREAM_END: Final[object] = object()

GAME_STREAM_ATTEMPTS: Final[int] = 8
"""Reconnects to one game's stream before giving it up. The game goes on on
the server whether or not we are listening."""

SPEEDS: Final[tuple[str, ...]] = (
    "ultraBullet", "bullet", "blitz", "rapid", "classical", "correspondence",
)
"""Lichess speed categories, fastest first, so a declined speed maps to
``tooFast`` or ``tooSlow`` rather than a bare refusal."""


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


class GamesApi(Protocol):
    def get_ongoing(self, count: int = 10) -> List[Dict[str, Any]]: ...


class LichessClient(Protocol):
    """Structural view of ``berserk.Client``, so tests can supply a fake."""

    @property
    def bots(self) -> BotsApi: ...

    @property
    def account(self) -> AccountApi: ...

    @property
    def games(self) -> GamesApi: ...


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
    stall_check_seconds: float = STALL_CHECK_SECONDS

    min_rating: int = 1100
    max_rating: int = 1700
    """The band the skew positions were mined in. Outside it the skew was never
    measured, so a game there adds noise to the experiment, not evidence."""

    speeds: frozenset[str] = frozenset({"blitz", "rapid"})
    """Also the mined speeds, for the same reason."""

    control_share: float = 0.5
    """Chance a game is played with the standard book instead of the repertoire."""

    game_log: Optional[Path] = None
    """JSONL of each game's arm and outcome, keyed by game id. ``None`` disables it."""

    pacing: bool = True
    """Pace moves against humans with :class:`AdaptivePacingController`. Bots get
    every move at once: the daily game quota, not their feelings, is the limit."""

    pacing_share: float = 0.5
    """Chance a move's wait is actually applied. The cadence is chosen by the
    position -- snaps for forced moves, waits for quiet ones -- so comparing
    cadences compares positions. A coin flip within each cadence is what lets
    paced and unpaced replies to the same kind of position be compared."""

    psychological: bool = True
    """Play humans with ``bot_factory.HUMAN_PLAY`` -- looser floor, narrow paths,
    traps while winning -- and bots with the conservative defaults and no trap
    book. Bots are not fooled by any of it and it cost games against them."""

    allow_bots: bool = False
    """Accept BOT challengers at any rating and speed. For play, not data: their
    games are logged with ``opponent_bot`` and the analyzer leaves them out."""


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
    arm: str = ""
    front_book_moves: int = 0
    opponent_bot: bool = False
    cadence: Dict[str, int] = field(default_factory=dict)
    """Moves per cadence, logged at the finish so pacing can be judged later."""


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
        books: Optional[Mapping[str, OpeningBook]] = None,
        rng: Optional[random.Random] = None,
        pacing: Optional[AdaptivePacingController] = None,
    ) -> None:
        self.client = client
        self.pacing = pacing if pacing is not None else AdaptivePacingController()
        self.searcher = searcher
        self.books = books
        """Arm name to book. ``None`` keeps whatever book the searcher was given."""
        self._rng = rng if rng is not None else random.Random()
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
            opened = time.monotonic()
            try:
                logger.info("lichess: opening the incoming event stream")
                for event in self.client.bots.stream_incoming_events():
                    if self._stop.is_set():
                        return
                    self._handle_event(event)
                lasted = time.monotonic() - opened
                if lasted >= STABLE_STREAM_SECONDS:
                    logger.warning("lichess: event stream closed by the server")
                    backoff = self.config.reconnect_backoff_seconds
                else:
                    logger.warning(
                        "lichess: event stream closed by the server after %.0fs, retrying in %.0fs; "
                        "if this repeats, check for a second bot process on this account",
                        lasted, backoff,
                    )
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

    def is_idle(self) -> bool:
        """No game running and no challenge holding the slot."""
        with self._lock:
            running = self._game_thread is not None and self._game_thread.is_alive()
            return self._reservation is None and not running

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
        challenger_id = str(challenger.get("id", "")).lower() if isinstance(challenger, Mapping) else ""
        if challenge.get("direction") == "out" or challenger_id == self.bot_id:
            # One of ours, not an invitation. The event stream echoes outbound
            # challenges without a direction field; declining one would cancel
            # the challenge the challenger just sent.
            return

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

        challenger = challenge.get("challenger")
        challenger = challenger if isinstance(challenger, Mapping) else {}
        if challenger.get("title") == "BOT":
            return None if self.config.allow_bots else "noBot"

        speed = str(challenge.get("speed", ""))
        if speed not in self.config.speeds:
            if speed not in SPEEDS:
                return "timeControl"
            allowed = [SPEEDS.index(s) for s in self.config.speeds if s in SPEEDS]
            return "tooFast" if allowed and SPEEDS.index(speed) < min(allowed) else "tooSlow"

        rating = challenger.get("rating")
        if not isinstance(rating, (int, float)) or not (
            self.config.min_rating <= rating <= self.config.max_rating
        ):
            return "generic"  # Lichess has no rating-band reason to send.
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
        backoff = self.config.reconnect_backoff_seconds
        try:
            for attempt in range(1, GAME_STREAM_ATTEMPTS + 1):
                try:
                    events = self._game_events(game_id)
                    while True:
                        try:
                            event = events.get(timeout=self.config.stall_check_seconds)
                        except queue.Empty:
                            if self._stop.is_set():
                                return
                            if self._stream_is_stale(game_id):
                                logger.warning(
                                    "game %s: no update for %.0fs and Lichess says the move is ours; "
                                    "the stream is stale", game_id, self.config.stall_check_seconds,
                                )
                                break
                            continue
                        if event is _STREAM_END:
                            logger.warning("game %s: stream ended before the game did", game_id)
                            break
                        if isinstance(event, BaseException):
                            raise event
                        if self._stop.is_set():
                            logger.info("game %s: shutting down, leaving the game", game_id)
                            return
                        kind = event.get("type")
                        if kind == "gameFull":
                            # A reconnect resends gameFull; the session, its
                            # book draw and its log line belong to the first.
                            if session is None:
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
                    if 400 <= exc.status_code < 500 and exc.status_code != 429:
                        logger.error("game %s: stream refused with HTTP %s (%s)", game_id, exc.status_code, exc)
                        return
                    logger.warning("game %s: stream failed with HTTP %s", game_id, exc.status_code)
                except NETWORK_ERRORS as exc:
                    logger.warning("game %s: stream dropped (%s)", game_id, exc)
                logger.info("game %s: reconnecting in %.0fs (%d/%d)",
                            game_id, backoff, attempt, GAME_STREAM_ATTEMPTS)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, self.config.max_reconnect_backoff_seconds)
            logger.error("game %s: gave up after %d reconnects", game_id, GAME_STREAM_ATTEMPTS)
        finally:
            logger.info("game %s: loop finished", game_id)

    def _game_events(self, game_id: str) -> "queue.Queue[Any]":
        """The game's stream, read on its own thread, so that silence can be
        noticed. Errors and the stream's end are handed over through the queue;
        a stale reader left behind exits when the server finally closes it."""
        events: "queue.Queue[Any]" = queue.Queue()

        def read() -> None:
            try:
                for event in self.client.bots.stream_game_state(game_id):
                    events.put(event)
                events.put(_STREAM_END)
            except BaseException as exc:  # the game thread decides what each error means
                events.put(exc)

        threading.Thread(target=read, name=f"lichess-stream-{game_id}", daemon=True).start()
        return events

    def _stream_is_stale(self, game_id: str) -> bool:
        """True if Lichess says this game awaits our move, or has ended.

        Asked only while the game thread is waiting -- never while it is
        searching -- so a move owed is a move the stream failed to deliver.
        """
        try:
            ongoing = self.client.games.get_ongoing(50)
        except ResponseError as exc:
            logger.warning("game %s: could not check for a stall (HTTP %s)", game_id, exc.status_code)
            return False
        except NETWORK_ERRORS as exc:
            logger.warning("game %s: could not check for a stall (%s)", game_id, exc)
            return False
        game = next((g for g in ongoing if str(g.get("gameId")) == game_id), None)
        return game is None or bool(game.get("isMyTurn"))

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

        # Maia-3 interpolates the rating continuously, so the opponent's real
        # number goes in unchanged -- no band, no bucket, no sharpening.
        maia_rating = opponent_rating
        try:
            self.human_model.set_rating(maia_rating)
        except EvaluatorError as exc:
            logger.error("game %s: could not model rating %d (%s), keeping %d",
                         game_id, maia_rating, exc, self.human_model.rating)
            maia_rating = self.human_model.rating
        opponent_bot = opponent.get("title") == "BOT"
        arm = ""
        if self.books and opponent_bot and self.config.psychological:
            # No trap gambits against engines; the repertoire is engine-equal.
            arm = "vs-bot"
            self.searcher.book = self.books[SKEW.name]
        elif self.books:
            arm = STANDARD.name if self._rng.random() < self.config.control_share else SKEW.name
            self.searcher.book = self.books[arm]
        # The book bands off the same rating: traps the opponent is likely to
        # see through are simply not offered.
        if self.searcher.book is not None:
            self.searcher.book.set_opponent_rating(opponent_rating)
        # The policy net takes the same rating as an input plane, so it has to
        # track the real opponent rather than the searcher's default.
        self.searcher.opponent_rating = opponent_rating

        logger.info(
            "game %s: playing %s against %s (%d) -> opponent model maia3@%d%s",
            game_id,
            "white" if my_color == chess.WHITE else "black",
            opponent_name,
            opponent_rating,
            maia_rating,
            f", {arm} book" if arm else "",
        )
        # Logged at the start as well as the end: a crash mid-game still leaves
        # the arm on record, and the game id is enough to fetch the rest.
        self._record({
            "event": "start", "game": game_id, "arm": arm, "at": int(time.time()),
            "colour": "white" if my_color == chess.WHITE else "black",
            "rating": opponent_rating, "speed": str(event.get("speed", "")),
            "rated": bool(event.get("rated", False)),
            "opponent_bot": opponent.get("title") == "BOT",
        })
        return GameSession(
            arm=arm,
            opponent_bot=opponent_bot,
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
            self._record({
                "event": "finish", "game": session.game_id, "arm": session.arm,
                "status": status, "winner": str(state.get("winner", "")),
                "plies": session.moves_played, "front_book_moves": session.front_book_moves,
                "cadence": session.cadence,
            })
            return False
        # Over by the rules, not merely claimable: python-chess calls a draw
        # claimable when the *next* move would repeat a third time, which
        # Lichess does not end. Waiting there lost two games on time.
        if session.board.is_game_over():
            logger.info("game %s: position is terminal, waiting for the server", session.game_id)
            return True
        if session.board.turn != session.my_color:
            return True

        mine, theirs = ("w", "b") if session.my_color == chess.WHITE else ("b", "w")
        own_clock = clock_seconds(state.get(f"{mine}time", 0))
        their_clock = clock_seconds(state.get(f"{theirs}time", 0))
        increment = clock_seconds(state.get(f"{mine}inc", 0))
        pacing = self.config.pacing and not session.opponent_bot
        search_config = self.time_manager.calculate_search_config(
            state.get("wtime", 0),
            state.get("btime", 0),
            state.get("winc", 0),
            state.get("binc", 0),
            is_white=session.my_color == chess.WHITE,
        )
        if pacing and 0 < their_clock < self.pacing.opponent_scramble_seconds:
            # A wait cannot make a slow search fast; a cheaper profile can.
            search_config = self.time_manager.profiles[min(1, len(self.time_manager.profiles) - 1)].config
        if self.config.psychological and not session.opponent_bot:
            search_config = for_humans(search_config)
            if 0 < their_clock < LOW_CLOCK_SECONDS:
                search_config = replace(
                    search_config,
                    narrow_path_weight=search_config.narrow_path_weight * LOW_CLOCK_BOOST,
                )
        started = time.monotonic()
        try:
            result = self.searcher.search(session.board, search_config)
        except TerminalPositionError:
            # The searcher scores a repetition or fifty-move position as a draw
            # and declines to search it, but the server has not ended the game.
            # Any legal move beats the clock running out; Stockfish's best is
            # the least bad. Caught before EvaluatorError, which would resign.
            move = self._fallback_move(session.board)
            logger.warning("game %s: search declined a drawn position, playing %s",
                           session.game_id, session.board.san(move))
            return self._submit_move(session.game_id, move.uci())
        except EvaluatorError as exc:
            logger.error("game %s: search failed (%s), resigning", session.game_id, exc)
            self._call_api("resign", lambda: self.client.bots.resign_game(session.game_id))
            return False

        if result.source is MoveSource.BOOK_TRAP:
            session.front_book_moves += 1
        san = session.board.san(result.move)
        logger.info(
            "game %s: playing %s (depth %d/%d) %s",
            session.game_id, san, search_config.root_depth, search_config.leaf_depth, result.summary(),
        )
        cadence, wait = Cadence.NONE, 0.0
        if pacing:
            cadence, wait = self._pace(
                session, result, time.monotonic() - started, own_clock, increment, their_clock,
            )
        if not session.opponent_bot:
            chosen = next((c for c in result.candidates if c.move == result.move), None)
            self._record({
                "event": "move", "game": session.game_id, "ply": session.board.ply(),
                "cadence": cadence.value, "paced": wait > 0, "wait": round(wait, 2),
                "safe_replies": chosen.safe_replies if chosen else None,
                "utility": round(float(result.expected_utility), 1),
                "objective": chosen.objective_score if chosen else None,
                "trap": bool(result.is_trap), "gambit": bool(chosen and chosen.is_gambit),
                "source": result.source.value,
            })
        if not self._submit_move(session.game_id, result.move.uci()):
            logger.error("game %s: could not submit %s, abandoning the game", session.game_id, san)
            return False
        return True

    def _pace(
        self, session: GameSession, result: Any, elapsed: float,
        own_clock: float, increment: float, their_clock: float,
    ) -> Tuple[Cadence, float]:
        """Wait out the move's cadence, on the stop event so shutdown is not held
        up. Returns the cadence and the wait actually applied."""
        board, move = session.board, result.move
        chosen = next((c for c in result.candidates if c.move == move), None)
        cadence = self.pacing.classify(
            forced=board.legal_moves.count() == 1,
            book=result.source in (MoveSource.BOOK_TRAP, MoveSource.BOOK_STANDARD),
            trap=bool(result.is_trap or (chosen is not None and chosen.is_gambit)),
            quiet=not (board.is_check() or board.is_capture(move) or board.gives_check(move)),
            utility=float(result.expected_utility),
            opponent_clock=their_clock,
        )
        wait = self.pacing.delay(
            cadence, elapsed=elapsed, own_clock=own_clock,
            budget=self.time_manager.budget_seconds(own_clock, increment), rng=self._rng,
        )
        if self._rng.random() >= self.config.pacing_share:
            wait = 0.0  # the control half of the coin flip
        session.cadence[cadence.value] = session.cadence.get(cadence.value, 0) + 1
        if wait > 0:
            logger.debug("game %s: %s, waiting %.2fs", session.game_id, cadence.value, wait)
            self._stop.wait(wait)
        return cadence, wait

    def _fallback_move(self, board: chess.Board) -> chess.Move:
        try:
            best = self.searcher.evaluator.analyse_root_moves(board, depth=8, multipv=1)
            if best:
                return next(iter(best))
        except EvaluatorError:
            pass
        return next(iter(board.legal_moves))

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

    def _record(self, row: Mapping[str, Any]) -> None:
        """Append one line to the game log. Ids and ratings only, no usernames."""
        if self.config.game_log is None:
            return
        try:
            with self.config.game_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        except OSError as exc:
            logger.error("lichess: could not write the game log (%s)", exc)

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


def keep_awake(pid: int) -> Optional["subprocess.Popen[bytes]"]:
    """Hold macOS awake for as long as ``pid`` lives. ``None`` off macOS.

    The bot lost four games on time by freezing mid-game while the Mac slept
    (20:17, 23:53, 05:47 and 08:57 in ``pmset -g log``). In one the Mac woke for
    maintenance, the challenger started a game, and the Mac slept again 15s in.
    ``caffeinate -i -s`` blocks idle sleep, and system sleep on AC power, until
    ``-w`` sees the bot exit. Closing the lid still sleeps the machine.
    """
    import shutil
    import subprocess
    import sys

    if sys.platform != "darwin" or shutil.which("caffeinate") is None:
        return None
    return subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(pid)])


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
    from contextlib import ExitStack

    from src.engine import Maia3Evaluator, StockfishEvaluator

    parser = argparse.ArgumentParser(prog="python -m src.lichess.bot", description="Lichess bot bridge.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--min-clock", type=float, default=DEFAULT_MIN_INITIAL_SECONDS,
                        help="Decline games with a shorter initial clock (seconds).")
    parser.add_argument("--no-policy", action="store_true",
                        help="Skip the neural proposer and use the Stockfish scan alone.")
    defaults = BotConfig()
    parser.add_argument("--min-rating", type=int, default=defaults.min_rating)
    parser.add_argument("--max-rating", type=int, default=defaults.max_rating)
    parser.add_argument("--speeds", nargs="+", default=sorted(defaults.speeds), choices=SPEEDS)
    parser.add_argument("--control-share", type=float, default=defaults.control_share,
                        help="Chance a game uses the standard book instead of the repertoire. "
                             "0 plays every game with the repertoire and measures nothing.")
    parser.add_argument("--game-log", type=Path, default=Path("build/live_games.jsonl"))
    parser.add_argument("--log-file", type=Path, default=Path("build/bot.log"),
                        help="Where the bridge's own log is kept, besides the terminal.")
    parser.add_argument("--allow-bots", action="store_true",
                        help="Also accept BOT challengers, at any rating and speed. Not data.")
    parser.add_argument("--challenge-bots", action="store_true",
                        help="Also challenge nearby-rated online bots, one at a time, minutes apart.")
    parser.add_argument("--no-pacing", action="store_true",
                        help="Move as soon as the search finishes, against humans too.")
    parser.add_argument("--no-psych", action="store_true",
                        help="Play humans with the same conservative search as bots.")
    parser.add_argument("--arena", default=None,
                        help="Join this Arena on start. Only Arenas created with bots allowed "
                             "admit BOT accounts, and the token needs the tournament:write scope.")
    args = parser.parse_args()
    if not 0.0 <= args.control_share <= 1.0:
        parser.error("--control-share must be between 0 and 1")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("chess.engine").setLevel(logging.WARNING)
    # The terminal alone lost the log of four timeouts when its scrollback was
    # cleared; the file keeps the last ~30MB.
    from logging.handlers import RotatingFileHandler

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    file_log = RotatingFileHandler(args.log_file, maxBytes=10_000_000, backupCount=3)
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_log)
    if keep_awake(os.getpid()) is not None:
        logger.info("lichess: holding the Mac awake while the bot runs")

    try:
        token = engine_config.lichess_token()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    session = TimedTokenSession(token)
    client = berserk.Client(session=session)
    args.game_log.parent.mkdir(parents=True, exist_ok=True)
    config = BotConfig(
        min_initial_seconds=args.min_clock, min_rating=args.min_rating,
        max_rating=args.max_rating, speeds=frozenset(args.speeds),
        control_share=args.control_share, game_log=args.game_log, allow_bots=args.allow_bots,
        psychological=not args.no_psych, pacing=not args.no_pacing,
    )

    with ExitStack() as stack:
        books: Dict[str, OpeningBook] = {}
        for spec in (STANDARD, SKEW):
            try:
                book = build_book(spec, opponent_rating=DEFAULT_OPPONENT_RATING)
            except FileNotFoundError as exc:
                logger.error("%s", exc)
                return 1
            assert book is not None, f"the {spec.name} arm opens a book"
            books[spec.name] = stack.enter_context(book)
        stockfish = stack.enter_context(StockfishEvaluator())
        maia = stack.enter_context(Maia3Evaluator(engine_config.DEFAULT_MAIA_RATING))
        bot = LichessBot(
            client,
            AdversarialSearcher(
                stockfish, maia, book=books[STANDARD.name],
                proposer=None if args.no_policy else load_proposer(),
            ),
            maia,
            config=config,
            books=books,
        )
        if args.arena and not bot._call_api(
            "join arena", lambda: client.tournaments.join_arena(args.arena)
        ):
            logger.error("lichess: could not join arena %s; see the warning above", args.arena)
            return 1
        if args.challenge_bots:
            from src.lichess.challenger import Challenger

            threading.Thread(
                target=Challenger(bot, client).run, name="lichess-challenger", daemon=True
            ).start()
        try:
            bot.run()
        except KeyboardInterrupt:
            logger.info("lichess: interrupted, shutting down")
            bot.stop()
    logger.info("lichess: engine processes terminated, books unmapped")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
