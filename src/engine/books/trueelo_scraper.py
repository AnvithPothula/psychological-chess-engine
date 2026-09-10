"""ETL: empirical rating ceilings for the trap repertoire.

    python -m src.engine.books.trueelo_scraper

Milestone 5 derived every trap's rating ceiling from a Maia rollout and the
rollout could not price it -- a two-ply opening measurement cannot see a 1900
converting a bad structure over the following forty moves. This script replaces
those curated numbers with the thing that does measure it: the expected score
real players of each rating actually get from the position, over millions of
games.

Two providers, tried in order:

* :class:`LichessExplorerProvider` -- the right instrument. Addressable by FEN,
  so it can score the *exact* position our line reaches, and it reports raw
  win/draw/loss counts per rating bucket. As of this writing
  ``explorer.lichess.ovh`` answers 401 to unauthenticated clients, verified from
  two independent networks, so it usually contributes nothing today. It is kept
  first because it is correct and the outage is not ours to fix.
* :class:`TrueEloProvider` -- reachable, and reads the same Lichess games. Keyed
  by opening name rather than position, so it scores the *named opening* rather
  than our exact line: coarser, and the reason a slug that does not resolve is
  a miss rather than a guess.

Anything neither provider covers keeps the curated ceiling in ``build_trap_book``
and is logged, so a network failure degrades the calibration rather than the bot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Final, List, Optional, Protocol, Sequence, Tuple

import chess
import chess.polyglot

from src import config
from src.engine.books.build_trap_book import (
    BOOKS_DIR,
    CEILINGS_PATH,
    RATINGS_PATH,
    RATING_BANDS,
    TRAP_LINES,
    TrapLine,
)

__all__ = [
    "BandScore",
    "CurveProvider",
    "HttpFetcher",
    "LichessExplorerProvider",
    "RatingCurve",
    "TrueEloProvider",
    "calibrate",
    "passing_bands",
    "render_report",
    "main",
]

logger = logging.getLogger(__name__)

CACHE_DIR: Final[Path] = BOOKS_DIR / ".http_cache"
REPORT_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "docs" / "GAMBIT_CALIBRATION.md"

USER_AGENT: Final[str] = (
    "psychological-chess-engine/1.0 (opening calibration; +https://github.com/AnvithPothula)"
)
REQUEST_INTERVAL_SECONDS: Final[float] = 1.0
"""Minimum gap between requests to one host. Both sources are free services
reading donated data; hammering them is how they end up behind a 401."""

REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0
RETRY_ATTEMPTS: Final[int] = 3
RETRY_BACKOFF: Final[float] = 2.0
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
CACHE_TTL_SECONDS: Final[float] = 7 * 24 * 3600.0

WIN_RATE_THRESHOLD: Final[float] = 0.50
"""Expected score, from the gambit owner's side, below which a line stops paying."""

ENDPOINT_FAILURE_LIMIT: Final[int] = 3
"""Consecutive empty lines before a provider is written off for the run."""

MIN_GAMES: Final[int] = 500
"""Evidence needed before a bracket may set or veto a ceiling."""

SYSTEM_CA_BUNDLE: Final[Path] = Path("/etc/ssl/cert.pem")

# We decline bullet and faster (BotConfig.min_initial_seconds), so scoring the
# repertoire on those controls would calibrate against games we never play.
SCORED_SPEEDS: Final[Tuple[str, ...]] = ("Blitz", "Rapid", "Classical")
EXPLORER_SPEEDS: Final[str] = "blitz,rapid,classical"
EXPLORER_BUCKETS: Final[Tuple[int, ...]] = (0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500)

TRUEELO_ROOT: Final[str] = "https://trueelo.app/stats"
EXPLORER_ROOT: Final[str] = "https://explorer.lichess.ovh/lichess"

# React renders text nodes separated by empty comments; strip them before parsing.
_REACT_SEPARATOR: Final[str] = "<!-- -->"
_SR_ROW = re.compile(
    r"<li>(\d+)-(?:\d+|\+)? ?rating, ([A-Za-z]+): ([\d.]+)% expected score \(([\d,]+) games\)</li>"
)
_SR_PERSPECTIVE = re.compile(r"(White|Black) expected score")
_ANALYZE_LINK = re.compile(r'href="/\?pgn=([^"]+)"')

TRUEELO_SLUGS: Final[Dict[str, str]] = {
    "Stafford Gambit, Bg5 trap": "petrovs-defense-stafford-gambit",
    "Stafford Gambit, h3 trap": "petrovs-defense-stafford-gambit",
    "Stafford Gambit, Be7 setup": "petrovs-defense-stafford-gambit",
    "Traxler Counterattack, Kxf2": "italian-game-two-knights-defense-traxler-counterattack",
    "Traxler Counterattack, Kf1": "italian-game-two-knights-defense-traxler-counterattack",
    "Englund Gambit, Qc1 mate": "englund-gambit",
    "Elephant Gambit, Wasp": "elephant-gambit",
    "Latvian Gambit": "latvian-gambit",
    "Halloween Gambit": "four-knights-game-halloween-gambit",
    "Fishing Pole Trap": "ruy-lopez-berlin-defense-fishing-pole-variation",
    "Siberian Trap": "sicilian-defense-smith-morra-gambit-accepted-siberian-variation",
    "Lasker Trap, underpromotion": "queens-gambit-declined-albin-countergambit-lasker-trap",
    "Elephant Trap": "queens-gambit-declined-modern-variation-normal-line",
    "Budapest, smothered mate": "budapest-defense-rubinstein-variation",
    "Legal's Mate": "italian-game-hungarian-defense",
    "Cochrane Gambit": "petrovs-defense-cochrane-gambit",
    "Tennison Gambit, Brigg's Trap": "zukertort-opening-tennison-gambit",
    "Wayward Queen, punished": "kings-pawn-game-wayward-queen-attack",
    "Vienna Gambit": "vienna-game-vienna-gambit",
    "Vienna Gambit, Wurzburger Trap": "vienna-game-vienna-gambit-wurzburger-trap",
    "Fried Liver Attack": "italian-game-two-knights-defense-fried-liver-attack",
    "Lolli Attack": "italian-game-two-knights-defense-lolli-attack",
    "Danish Gambit": "danish-gambit",
    "Smith-Morra Gambit": "sicilian-defense-smith-morra-gambit",
    "Scotch Gambit accepted": "italian-game-scotch-gambit",
    "Evans Gambit": "italian-game-evans-gambit",
    "Max Lange Attack": "italian-game-two-knights-defense-max-lange-attack",
    "Urusov Gambit": "bishops-opening-urusov-gambit",
    "Albin Countergambit": "queens-gambit-declined-albin-countergambit",
}
"""Opening-name keys for TrueElo. Absent lines fall back to the curated ceiling."""


@dataclass(frozen=True, slots=True)
class BandScore:
    """Expected score for the *gambit owner* in one rating bracket."""

    band: int
    """Lower edge of the bracket."""

    expected_score: float
    """``wins + draws/2``, in ``[0.0, 1.0]``, from the owner's side."""

    games: int


@dataclass(frozen=True, slots=True)
class RatingCurve:
    line: str
    source: str
    bands: Tuple[BandScore, ...]
    detail: str = ""
    """Provenance worth keeping: the ply scored, or the opening matched."""

    def score_at(self, rating: int) -> Optional[BandScore]:
        """The bracket covering ``rating``, or ``None`` if uncovered."""
        covering = [band for band in self.bands if band.band <= rating]
        return max(covering, key=lambda band: band.band) if covering else None


class CurveProvider(Protocol):
    name: str

    def curve(self, line: TrapLine) -> Optional[RatingCurve]: ...


def _ssl_context() -> ssl.SSLContext:
    """A context with a usable CA store; python.org builds ship none on macOS."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    if SYSTEM_CA_BUNDLE.exists():
        return ssl.create_default_context(cafile=str(SYSTEM_CA_BUNDLE))
    return ssl.create_default_context()


class HttpFetcher:
    """Polite GET: on-disk cache, one request per interval, retry with backoff."""

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        *,
        interval: float = REQUEST_INTERVAL_SECONDS,
        ttl: float = CACHE_TTL_SECONDS,
        token: Optional[str] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.interval = interval
        self.ttl = ttl
        self.token = token
        self._context = _ssl_context()
        self._last_request = 0.0
        self.hits = 0
        self.misses = 0

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:32]}.json"

    def get(self, url: str) -> Optional[str]:
        """Body of ``url``, from cache when fresh. ``None`` if unreachable."""
        path = self._cache_path(url)
        if path.exists():
            try:
                payload = json.loads(path.read_text())
                if time.time() - float(payload["fetched_at"]) < self.ttl:
                    self.hits += 1
                    body: Optional[str] = payload["body"]
                    return body
            except (OSError, ValueError, KeyError):
                pass  # A damaged cache entry is just a miss.

        body = self._fetch(url)
        self.misses += 1
        if body is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "fetched_at": time.time(), "body": body}))
        return body

    def _fetch(self, url: str) -> Optional[str]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/html"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        delay = self.interval

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last_request = time.monotonic()
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(
                    request, timeout=REQUEST_TIMEOUT_SECONDS, context=self._context
                ) as response:
                    return str(response.read().decode("utf-8", "replace"))
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRYABLE_STATUS:
                    logger.debug("fetch: HTTP %d for %s", exc.code, url)
                    return None
                logger.debug("fetch: HTTP %d, attempt %d/%d", exc.code, attempt, RETRY_ATTEMPTS)
            except (urllib.error.URLError, OSError, ssl.SSLError) as exc:
                logger.debug("fetch: %s on attempt %d/%d (%s)", type(exc).__name__, attempt, RETRY_ATTEMPTS, exc)
            time.sleep(delay)
            delay *= RETRY_BACKOFF
        return None


def _owner_score(white: int, draws: int, black: int, owner: chess.Color) -> Tuple[float, int]:
    total = white + draws + black
    if total == 0:
        return 0.0, 0
    wins = white if owner == chess.WHITE else black
    return (wins + draws / 2.0) / total, total


class LichessExplorerProvider:
    """Scores the exact position our line reaches, per rating bucket.

    Walks *backwards* from the end of the line until a position has enough
    games. Scoring the final FEN -- as the obvious reading of the brief would --
    measures the wrong thing entirely: five of these lines end in checkmate, so
    the position has no continuations at all, and any game reaching it was one
    we had already won. The deepest position with real traffic is the one whose
    win rate says whether the trap is worth springing.
    """

    name = "lichess-explorer"

    def __init__(self, fetcher: HttpFetcher, *, min_games: int = MIN_GAMES) -> None:
        self.fetcher = fetcher
        self.min_games = min_games
        self.available = True
        self._consecutive_failures = 0

    def curve(self, line: TrapLine) -> Optional[RatingCurve]:
        if not self.available:
            return None
        for board in self._positions_deepest_first(line):
            bands = self._bands_for(board, line.owner)
            if bands and max(band.games for band in bands) >= self.min_games:
                self._consecutive_failures = 0
                return RatingCurve(
                    line=line.name,
                    source=self.name,
                    bands=tuple(bands),
                    detail=f"ply {board.ply()}",
                )

        self._consecutive_failures += 1
        if self._consecutive_failures >= ENDPOINT_FAILURE_LIMIT:
            # The endpoint is comprehensively unavailable rather than thin on
            # this one position. Keep asking and the run costs minutes of
            # timeouts per line for nothing.
            self.available = False
            logger.warning(
                "explorer: no data after %d lines, disabling it for this run",
                self._consecutive_failures,
            )
        return None

    @staticmethod
    def _positions_deepest_first(line: TrapLine) -> List[chess.Board]:
        board = chess.Board()
        positions: List[chess.Board] = []
        for san in line.moves.split():
            board.push(board.parse_san(san))
            # Only positions where the owner is about to move can be scored as
            # "the owner chose this"; the others are the victim's decisions.
            if board.turn == line.owner:
                positions.append(board.copy(stack=False))
        return list(reversed(positions))

    def _bands_for(self, board: chess.Board, owner: chess.Color) -> List[BandScore]:
        bands: List[BandScore] = []
        for bucket in EXPLORER_BUCKETS:
            query = urllib.parse.urlencode(
                {
                    "variant": "standard",
                    "fen": board.fen(),
                    "ratings": bucket,
                    "speeds": EXPLORER_SPEEDS,
                    "topGames": 0,
                    "recentGames": 0,
                }
            )
            body = self.fetcher.get(f"{EXPLORER_ROOT}?{query}")
            if body is None:
                return []  # Endpoint is down; do not report a partial curve.
            try:
                payload = json.loads(body)
                score, games = _owner_score(
                    int(payload["white"]), int(payload["draws"]), int(payload["black"]), owner
                )
            except (ValueError, KeyError, TypeError):
                return []
            if games:
                bands.append(BandScore(bucket, score, games))
        return bands


class TrueEloProvider:
    """Reads TrueElo's per-bracket expected scores for a named opening.

    The numbers live in a screen-reader list rather than the visible table,
    which is the more stable target: it exists for accessibility and survives
    restyling. Its perspective is fixed per page and named in the heading, and
    the ``side`` query parameter does *not* change it -- so the heading is read
    and the score flipped when it disagrees with the line's owner.
    """

    name = "trueelo"

    def __init__(
        self,
        fetcher: HttpFetcher,
        slugs: Dict[str, str] = TRUEELO_SLUGS,
        *,
        min_games: int = MIN_GAMES,
    ) -> None:
        self.fetcher = fetcher
        self.slugs = slugs
        self.min_games = min_games

    def curve(self, line: TrapLine) -> Optional[RatingCurve]:
        slug = self.slugs.get(line.name)
        if slug is None:
            return None
        body = self.fetcher.get(f"{TRUEELO_ROOT}/{slug}")
        if body is None:
            return None
        if not self.covers_line(body, line):
            logger.warning(
                "%-34s slug %r is a different opening, discarding its numbers", line.name, slug
            )
            return None
        bands = self.parse(body, line.owner)
        if not bands:
            return None
        return RatingCurve(line=line.name, source=self.name, bands=bands, detail=slug)

    @staticmethod
    def covers_line(html: str, line: TrapLine) -> bool:
        """Does the page's opening actually reach a position on our line?

        A slug resolving to HTTP 200 proves only that *some* opening lives
        there; calibrating Legal's Mate against the Hungarian Defense because
        both are Italian Game pages would silently poison the ceiling.

        The test is **positional, not textual**. Openings transpose: TrueElo
        reaches the Max Lange via 3.d4 exd4 4.Bc4 while our line plays 3.Bc4
        Bc5 4.O-O, and by move six they are the same position by different
        roads. Comparing move sequences rejects that, so the page's final
        position is looked for among the positions our line passes through.
        """
        link = _ANALYZE_LINK.search(html.replace(_REACT_SEPARATOR, ""))
        if link is None:
            return False
        page_moves = [
            re.sub(r"^\d+\.", "", token)
            for token in re.split(r"[+\s]+", urllib.parse.unquote_plus(link.group(1)))
            if token and not re.fullmatch(r"\d+\.", token)
        ]
        if not page_moves:
            return False

        ours = chess.Board()
        reached = {chess.polyglot.zobrist_hash(ours)}
        for san in line.moves.split():
            ours.push_san(san)
            reached.add(chess.polyglot.zobrist_hash(ours))

        theirs = chess.Board()
        for san in page_moves:
            try:
                theirs.push_san(san)
            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                return False
        return chess.polyglot.zobrist_hash(theirs) in reached

    @staticmethod
    def parse(html: str, owner: chess.Color) -> Tuple[BandScore, ...]:
        """Parse the screen-reader score list, normalised to the owner's side."""
        text = html.replace(_REACT_SEPARATOR, "")
        perspective = _SR_PERSPECTIVE.search(text)
        page_side = chess.WHITE if perspective and perspective.group(1) == "White" else chess.BLACK
        flip = page_side != owner

        totals: Dict[int, List[float]] = {}
        for low, speed, score, games in _SR_ROW.findall(text):
            if speed not in SCORED_SPEEDS:
                continue
            count = int(games.replace(",", ""))
            if count <= 0:
                continue
            value = float(score) / 100.0
            bucket = totals.setdefault(int(low), [0.0, 0.0])
            bucket[0] += value * count
            bucket[1] += count

        bands: List[BandScore] = []
        for band, (weighted, played) in sorted(totals.items()):
            owner_score = weighted / played
            bands.append(
                BandScore(band, 1.0 - owner_score if flip else owner_score, int(played))
            )
        return tuple(bands)


def passing_bands(
    curve: RatingCurve,
    bands: Sequence[int] = RATING_BANDS,
    *,
    threshold: float = WIN_RATE_THRESHOLD,
    min_games: int = MIN_GAMES,
) -> Tuple[int, ...]:
    """Every Maia band whose bracket scores at or above ``threshold``.

    Deliberately not "everything below the first failure". That reading assumes
    the curve falls monotonically, and real ones do not: the Smith-Morra starts
    at 49% in the low brackets and *climbs* to 56% at 2500+, because a sound
    gambit rewards the side that knows the plans and knowing plans tracks
    rating. Stopping at its first sub-threshold band would switch off one of the
    strongest lines in the repertoire. Bands with too few games are skipped
    rather than failed: no evidence is not evidence of no effect.
    """
    return tuple(
        band
        for band in bands
        if (entry := curve.score_at(band)) is not None
        and entry.games >= min_games
        and entry.expected_score >= threshold
    )


def calibrate(
    lines: Sequence[TrapLine],
    providers: Sequence[CurveProvider],
    *,
    threshold: float = WIN_RATE_THRESHOLD,
    min_games: int = MIN_GAMES,
) -> Dict[str, object]:
    """Resolve an empirical ceiling for every line it can, in provider order."""
    results: Dict[str, object] = {}
    for line in lines:
        for provider in providers:
            curve = provider.curve(line)
            if curve is None:
                continue
            live = passing_bands(curve, threshold=threshold, min_games=min_games)
            ceiling = max(live) if live else None
            results[line.name] = {
                "bands": list(live),
                "ceiling": ceiling,
                "curated_ceiling": line.ceiling,
                "source": curve.source,
                "detail": curve.detail,
                "owner": line.owner_name,
                "curve": [
                    {"band": band.band, "score": round(band.expected_score, 4), "games": band.games}
                    for band in curve.bands
                ],
            }
            logger.info(
                "%-34s %-16s %s (curated %d) via %s",
                line.name,
                curve.detail,
                "off everywhere" if not live else f"bands {min(live)}-{ceiling} ({len(live)}/9)",
                line.ceiling,
                curve.source,
            )
            break
        else:
            logger.warning("%-34s no empirical curve; keeping curated %d", line.name, line.ceiling)

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": threshold,
        "min_games": min_games,
        "speeds": list(SCORED_SPEEDS),
        "bands": list(RATING_BANDS),
        "lines": results,
    }


REPORT_PREAMBLE: Final[str] = """# Gambit Calibration

How every trap in the repertoire got its rating ceiling, and why the obvious
method does not work.

*Generated by `python -m src.engine.books.trueelo_scraper --report`. Do not edit
by hand; the numbers come from `trap_ceilings.json`.*

## Why one-ply expectimax cannot price a gambit

The searcher scores a candidate move `m` from state `s` as

```
U(m) = SUM_h  P(h | s'_m) * V(s''_{m,h})
```

`P` is Maia's reply distribution and `V` is Stockfish's value at `leaf_depth`.
This is a one-ply expectimax: it enumerates the opponent's *immediate* reply and
then evaluates statically.

A gambit's compensation is not realised on the immediate reply. It is realised
if the victim errs at some later decision node `k`, typically three to five
plies on. At node 1 essentially every reply is adequate for the victim, so

```
V(s''_{m,h}) ~ V_objective(s'_m) ~ -D     for all h
```

where `D` is the material invested. Substituting:

```
U(m) ~ -D * SUM_h P(h | s'_m) = -D
```

The reply distribution sums to one, so it cancels. **The opponent model drops
out of the estimate entirely.** Formally, with `θ` the opponent's rating,

```
dU/dθ = SUM_h  [dP(h | s'_m)/dθ] * V(s''_{m,h}) ~ -D * d/dθ SUM_h P(h) = 0
```

The estimator is *constant in the rating parameter* to first order, which is
exactly what the Milestone 5 measurement showed: per-band payoff curves came
out flat or mildly rising while real-world results decay sharply. It was not a
tuning problem. The quantity had no rating signal in it to find.

Rating signal enters at depth `2k`, and the search truncates at depth 2.

## Why empirical win rates do

The empirical estimator is

```
S(θ) = E[ game result | position, both players rated θ ]
```

taken over completed games. It integrates the entire remaining game tree,
including the part that actually separates ratings: conversion. A 1900 who
declines the gambit correctly then grinds the extra pawn out over forty moves;
a 1100 declines it correctly and loses the thread by move 25. Both decline. Only
the first shows up as a worse result for us, and only `S(θ)` can see it.

The ceiling is then the set of bands where `S(θ) >= threshold`.

## Why the ceiling is a band set, not a cut-off

Reading the breakpoint as "every band below the first sub-threshold one"
assumes `S` decreases monotonically. It does not. Sound gambits *improve* with
rating, because they reward knowing the plans and that tracks skill: the
Smith-Morra opens below 50% in the low brackets and climbs above it later.
Cutting at its first weak band would disable one of the strongest lines in the
repertoire. Each band is therefore tested independently, and brackets with too
few games are skipped rather than failed -- no evidence is not evidence of no
effect.

## Data integrity

Two guards, both of which caught real errors:

* **Position, not move order.** An opening page returning HTTP 200 is not proof
  it is the right opening. Each page states its own line and it must reach a
  position our line passes through. Comparing move *sequences* is too strict --
  openings transpose, and the Max Lange arrives by two different move orders --
  so the test compares Zobrist hashes.
* **Never the final FEN.** Five of these lines end in checkmate, so their final
  position has no continuations and any game reaching it was already won. The
  explorer provider walks *backwards* from the end of the line until a position
  has enough traffic to be worth scoring.

"""


def render_report(payload: Dict[str, object], lines: Sequence[TrapLine]) -> str:
    """Build the calibration report from the measured data."""
    records = payload.get("lines", {})
    assert isinstance(records, dict)
    threshold = float(str(payload.get("threshold", WIN_RATE_THRESHOLD)))
    min_games = int(str(payload.get("min_games", MIN_GAMES)))

    rows: List[str] = []
    for line in lines:
        record = records.get(line.name)
        if not isinstance(record, dict):
            rows.append(
                f"| {line.name} | {line.owner_name} | — | _curated_ {line.ceiling} | no curve |"
            )
            continue
        bands = [int(band) for band in record.get("bands", [])]
        span = "off everywhere" if not bands else f"{min(bands)}–{max(bands)}"
        curve = record.get("curve", [])
        assert isinstance(curve, list)
        games = sum(int(entry["games"]) for entry in curve)
        rows.append(
            f"| {line.name} | {line.owner_name} | {span} | curated {line.ceiling} | "
            f"{record.get('source')} ({games:,} games) |"
        )

    covered = sum(1 for line in lines if line.name in records)
    return (
        REPORT_PREAMBLE
        + f"\n## Results\n\nThreshold: expected score >= {threshold:.0%}, "
        f"minimum {min_games:,} games per bracket, scored on "
        f"{', '.join(SCORED_SPEEDS).lower()}.\n\n"
        f"{covered} of {len(lines)} lines carry an empirical curve; the rest keep their "
        "curated ceiling and are marked below.\n\n"
        "| Line | Side | Empirical bands | Fallback | Source |\n"
        "| --- | --- | --- | --- | --- |\n" + "\n".join(rows) + "\n\n"
        f"_Generated {payload.get('generated')}._\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.engine.books.trueelo_scraper")
    parser.add_argument("--threshold", type=float, default=WIN_RATE_THRESHOLD,
                        help="Expected score below which a line is deemed to stop paying.")
    parser.add_argument("--min-games", type=int, default=MIN_GAMES,
                        help="Games a bracket needs before it may set or veto a ceiling.")
    parser.add_argument("--refresh", action="store_true", help="Ignore the on-disk HTTP cache.")
    parser.add_argument("--report", action="store_true",
                        help=f"Also write {REPORT_PATH.name}.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    try:
        token: Optional[str] = config.lichess_token()
    except RuntimeError:
        token = None  # The explorer is public in principle; a token is a bonus.

    ttl: float = 0.0 if args.refresh else CACHE_TTL_SECONDS
    fetcher = HttpFetcher(ttl=ttl, token=token)
    providers: Tuple[CurveProvider, ...] = (
        LichessExplorerProvider(fetcher, min_games=args.min_games),
        TrueEloProvider(fetcher, min_games=args.min_games),
    )

    payload = calibrate(
        TRAP_LINES, providers, threshold=args.threshold, min_games=args.min_games
    )
    CEILINGS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    if args.report:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(render_report(payload, TRAP_LINES))
        logger.info("wrote %s", REPORT_PATH)
    resolved = len(payload["lines"]) if isinstance(payload["lines"], dict) else 0
    logger.info(
        "wrote %s: %d/%d lines calibrated (cache %d hits / %d misses)",
        CEILINGS_PATH.name, resolved, len(TRAP_LINES), fetcher.hits, fetcher.misses,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
