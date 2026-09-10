"""Compiles the curated trap repertoire into ``traps.bin``.

    python -m src.engine.books.build_trap_book

Polyglot is a 16-byte record format -- big-endian ``key, move, weight, learn``,
sorted by key -- so the book is written directly rather than shelling out to a
book compiler. ``python-chess`` reads Polyglot but cannot write it.

**Rating bands live in the Polyglot ``learn`` field.** Each entry carries a
9-bit mask over the Maia checkpoints (1100..1900); a bit is set when the trap is
worth playing against an opponent in that band. ``learn`` is a free-form 32-bit
field that other engines ignore, so the book stays a valid Polyglot file and a
book without bands (``learn == 0``) reads as "eligible everywhere".

Which bands get set is **measured, not asserted**: ``--measure`` replays every
line through the real Maia checkpoints and the real Stockfish, and records the
expectimax utility of each trap at each rating. See :func:`eligible_bands` for
why the raw per-band numbers cannot be thresholded directly.

**Only the trap owner's moves are stored.** A trap line contains the opponent's
losing replies too, and storing those would have the bot walk into its own
Stafford Gambit whenever it happened to hold the white pieces. Each line
therefore declares the colour that owns it, and only that side's moves become
book entries.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Dict, Final, List, Mapping, Optional, Sequence, Tuple

import chess
import chess.polyglot

from src import config

if TYPE_CHECKING:  # engines are imported lazily so a plain build needs none
    from src.engine.search import AdversarialSearcher
    from src.types import SearchConfig

logger = logging.getLogger(__name__)

ENTRY_STRUCT: Final[struct.Struct] = struct.Struct(">QHHI")
MAX_WEIGHT: Final[int] = 0xFFFF
DEFAULT_WEIGHT: Final[int] = 100
BOOKS_DIR: Final[Path] = Path(__file__).resolve().parent
RATINGS_PATH: Final[Path] = BOOKS_DIR / "trap_ratings.json"
CEILINGS_PATH: Final[Path] = BOOKS_DIR / "trap_ceilings.json"
"""Empirical ceilings from ``trueelo_scraper``. Defined here rather than there so
the scraper can import it without this module importing the scraper back."""

OFF_EVERYWHERE: Final[int] = 0
"""Ceiling for a line the data says never pays: below every rating band."""

RATING_BANDS: Final[Tuple[int, ...]] = config.AVAILABLE_MAIA_RATINGS
"""Shared with the Maia checkpoints: a band *is* a Maia model."""
ALL_BANDS: Final[int] = (1 << len(RATING_BANDS)) - 1

SOUND_OBJECTIVE_CP: Final[int] = -80
"""A line costing less than this is a real opening, not a swindle, and stays
enabled at every rating. Lichess data backs this: the Smith-Morra climbs from
48.4% at 0-999 to 56.6% at 2500+, while the Stafford falls from 72.3% to 41.9%."""

MIN_TRAP_PAYOFF_CP: Final[int] = 25
"""An unsound line is enabled only in bands where its measured payoff clears
this. Calibrated against the Lichess breakpoints rather than picked: the
Stafford scores 72.3% at 0-999 and 41.9% at 1800-1999 over 1.8M games, so it
should switch off in the high 1700s, and 25cp is the bar that puts it there."""

WIN_MARGIN_CP: Final[int] = 300
"""Once the position is worth this much against *best* defence, the trap has
already worked. A trap needs the victim to err once, not to keep cooperating to
the final mate, so the rollout banks its winnings here and stops."""

SMOOTHING_WINDOW: Final[int] = 3
"""Bands averaged with their neighbours before thresholding. The Maia-2 paper
reports the separate Maia-1 checkpoints are *volatile* across skill levels --
only ~1.4% of positions transition monotonically -- so a per-band threshold on
raw numbers produces on/off/on band masks that are measurement noise, not
chess. Adjacent checkpoints are trained on adjacent rating bins, so averaging
over neighbours borrows strength exactly where it is legitimate."""


@dataclass(frozen=True, slots=True)
class TrapLine:
    """One curated line, given in SAN from the initial position."""

    name: str
    owner: chess.Color
    """The side setting the trap. Only this side's moves are stored."""

    moves: str
    weight: int = DEFAULT_WEIGHT
    """Relative likelihood of choosing this line where it branches."""

    ceiling: int = 1900
    """Fallback ceiling, used only when the empirical calibration has no curve
    for this line. The real number comes from ``trap_ceilings.json``; see
    :func:`resolve_ceilings`."""

    @property
    def owner_name(self) -> str:
        return "white" if self.owner == chess.WHITE else "black"


# Practical gambits: objectively imperfect, but every one of them has a
# concrete refutation that a club player is unlikely to find at the board.
TRAP_LINES: Final[Tuple[TrapLine, ...]] = (
    # --- unsound traps: huge payoff low down, collapse by ~1800 --------------
    TrapLine("Stafford Gambit, Bg5 trap", chess.BLACK,
             "e4 e5 Nf3 Nf6 Nxe5 Nc6 Nxc6 dxc6 d3 Bc5 Bg5 Nxe4 Bxd8 Bxf2+ Ke2 Bg4#", 220, 1700),
    TrapLine("Stafford Gambit, h3 trap", chess.BLACK,
             "e4 e5 Nf3 Nf6 Nxe5 Nc6 Nxc6 dxc6 Nc3 Bc5 Bc4 Ng4 O-O Qh4 h3 Nxf2 Rxf2 Bxf2+", 180, 1700),
    TrapLine("Stafford Gambit, Be7 setup", chess.BLACK,
             "e4 e5 Nf3 Nf6 Nxe5 Nc6 Nxc6 dxc6 e5 Ne4 d3 Bc5 dxe4 Bxf2+ Ke2 Bg4+", 170, 1600),
    TrapLine("Blackburne Shilling Gambit", chess.BLACK,
             "e4 e5 Nf3 Nc6 Bc4 Nd4 Nxe5 Qg5 Nxf7 Qxg2 Rf1 Qxe4+ Be2 Nf3#", 200, 1500),
    TrapLine("Traxler Counterattack, Kxf2", chess.BLACK,
             "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 Bc5 Nxf7 Bxf2+ Kxf2 Nxe4+ Kg1 Qh4", 200, 1600),
    TrapLine("Traxler Counterattack, Kf1", chess.BLACK,
             "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 Bc5 Nxf7 Bxf2+ Kf1 Qe7 Nxh8 d5", 150, 1500),
    TrapLine("Englund Gambit, Qc1 mate", chess.BLACK,
             "d4 e5 dxe5 Nc6 Nf3 Qe7 Bf4 Qb4+ Bd2 Qxb2 Bc3 Bb4 Qd2 Bxc3 Qxc3 Qc1#", 200, 1400),
    TrapLine("Elephant Gambit, Wasp", chess.BLACK,
             "e4 e5 Nf3 d5 Nxe5 Bd6 d4 dxe4 Bc4 Qg5 Bxf7+ Ke7 Bb3 Qxg2", 120, 1300),
    TrapLine("Latvian Gambit", chess.BLACK,
             "e4 e5 Nf3 f5 Nxe5 Qf6 d4 d6 Nc4 fxe4 Nc3 Qg6", 110, 1300),
    TrapLine("Halloween Gambit", chess.WHITE,
             "e4 e5 Nc3 Nc6 Nf3 Nf6 Nxe5 Nxe5 d4 Ng6 e5 Ng8 Bc4", 150, 1400),
    TrapLine("Fishing Pole Trap", chess.BLACK,
             "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Ng4 h3 h5 hxg4 hxg4 Ne1 Qh4 f4 g3", 190, 1600),
    TrapLine("Siberian Trap", chess.BLACK,
             "e4 c5 d4 cxd4 c3 dxc3 Nxc3 Nc6 Nf3 e6 Bc4 Qc7 O-O Nf6 Qe2 Ng4 h3 Nd4", 190, 1800),
    TrapLine("Lasker Trap, underpromotion", chess.BLACK,
             "d4 d5 c4 e5 dxe5 d4 e3 Bb4+ Bd2 dxe3 Bxb4 exf2+ Ke2 fxg1=N+", 200, 1800),
    TrapLine("Elephant Trap", chess.BLACK,
             "d4 d5 c4 e6 Nc3 Nf6 Bg5 Nbd7 cxd5 exd5 Nxd5 Nxd5 Bxd8 Bb4+ Qd2 Bxd2+", 190, 1800),
    TrapLine("Budapest, Kieninger Trap", chess.BLACK,
             "d4 Nf6 c4 e5 dxe5 Ng4 Bf4 Nc6 Nf3 Bb4+ Nc3 Qe7 Qd5 Bxc3+ bxc3 Qa3", 180, 1700),
    TrapLine("Budapest, smothered mate", chess.BLACK,
             "d4 Nf6 c4 e5 dxe5 Ng4 Bf4 Nc6 Nf3 Bb4+ Nbd2 Qe7 a3 Ngxe5 axb4 Nd3#", 190, 1600),
    TrapLine("Legal's Mate", chess.WHITE,
             "e4 e5 Nf3 Nc6 Bc4 d6 Nc3 Bg4 h3 Bh5 Nxe5 Bxd1 Bxf7+ Ke7 Nd5#", 200),
    TrapLine("Cochrane Gambit", chess.WHITE,
             "e4 e5 Nf3 Nf6 Nxe5 d6 Nxf7 Kxf7 d4 c5 Bc4+ Be6 Bxe6+ Kxe6 Qg4+", 140, 1700),
    TrapLine("Tennison Gambit, Brigg's Trap", chess.WHITE,
             "e4 d5 Nf3 dxe4 Ng5 Nf6 d3 exd3 Bxd3 h6 Nxf7 Kxf7 Bg6+ Kxg6 Qxd8", 170, 1500),
    TrapLine("Wayward Queen, punished", chess.BLACK,
             "e4 e5 Qh5 Nc6 Bc4 g6 Qf3 Nf6 Qb3 Nd4", 140, 1400),

    # --- sound gambits: hold up as the opponent rating climbs ----------------
    TrapLine("Vienna Gambit", chess.WHITE,
             "e4 e5 Nc3 Nf6 f4 exf4 e5 Qe7 Qe2 Ng8 Nf3 d6 d4 dxe5 Nxe5", 220),
    TrapLine("Vienna Gambit, Wurzburger Trap", chess.WHITE,
             "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Bg4 Qe2 Nxc3 dxc3 Bxf3 Qxf3", 200),
    TrapLine("Fried Liver Attack", chess.WHITE,
             "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Nxd5 Nxf7 Kxf7 Qf3+ Ke6 Nc3", 200),
    TrapLine("Lolli Attack", chess.WHITE,
             "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Nxd5 d4 Be6 O-O Be7 Nxf7", 150),
    TrapLine("Danish Gambit", chess.WHITE,
             "e4 e5 d4 exd4 c3 dxc3 Bc4 cxb2 Bxb2 d5 Bxd5 Nf6 Bxf7+", 180),
    TrapLine("Smith-Morra Gambit", chess.WHITE,
             "e4 c5 d4 cxd4 c3 dxc3 Nxc3 Nc6 Nf3 d6 Bc4 e6 O-O", 200),
    TrapLine("Scotch Gambit accepted", chess.WHITE,
             "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bc5 c3 dxc3 Nxc3 d6 Qb3 Qd7 O-O", 180),
    TrapLine("Evans Gambit", chess.WHITE,
             "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O Nge7 cxd4", 200),
    TrapLine("Goring Gambit", chess.WHITE,
             "e4 e5 Nf3 Nc6 d4 exd4 c3 dxc3 Nxc3 Bb4 Bc4 d6 O-O Bxc3 bxc3", 160),
    TrapLine("Max Lange Attack", chess.WHITE,
             "e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6 d4 exd4 e5 d5 exf6 dxc4 Re1+ Be6 Ng5", 170),
    TrapLine("Urusov Gambit", chess.WHITE,
             "e4 e5 Bc4 Nf6 d4 exd4 Nf3 Nxe4 Qxd4 Nf6 Bg5 Be7 Nc3 c6 O-O-O", 170),
    TrapLine("Albin Countergambit", chess.BLACK,
             "d4 d5 c4 e5 dxe5 d4 Nf3 Nc6 a3 Bg4 Nbd2 Qe7 h3 Bxf3 Nxf3 O-O-O", 150, 1700),
)


def polyglot_raw_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a move the way Polyglot stores it.

    Castling is written as the king *capturing its own rook* -- e1h1 rather than
    e1g1 -- which is what ``MemoryMappedReader.find_all`` converts back when it
    normalises entries against a board.
    """
    to_square = move.to_square
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        to_square = chess.square(7 if kingside else 0, rank)

    promotion_part = (move.promotion - 1) if move.promotion else 0
    return (promotion_part << 12) | (move.from_square << 6) | to_square


def smooth(values: Sequence[float], window: int = SMOOTHING_WINDOW) -> List[float]:
    """Rolling mean over neighbouring rating bands, clamped at the edges."""
    half = window // 2
    return [
        fmean(values[max(0, i - half) : min(len(values), i + half + 1)])
        for i in range(len(values))
    ]


def eligible_bands(utilities: Sequence[float], objective_worst: float, allowed: int = ALL_BANDS) -> int:
    """Bitmask of the rating bands where a line is worth playing.

    Two instruments, each doing the job it is actually good at.

    **Measurement decides soundness and dud-detection.** The Maia rollout
    cleanly separates real openings from swindles via ``objective_worst``, and
    its payoff numbers identify lines that do not pay at *any* rating.

    **Empirical win rates decide the ceiling.** The rollout was tried for this
    and does not work: its per-band payoff curves come out flat or mildly
    *rising* with rating, while millions of Lichess games show real decay. A
    two-ply opening measurement cannot see the reason -- a 1900 converts a bad
    structure over the following forty moves and a 1100 does not -- and the
    Maia-2 paper independently reports the Maia-1 checkpoints are volatile
    across levels. The ceiling therefore comes from measured expected score by
    rating, via ``trueelo_scraper``.
    """
    if objective_worst >= SOUND_OBJECTIVE_CP:
        return allowed
    mask = 0
    for index, value in enumerate(smooth(utilities)):
        # An unsound line must at least be measurably worth something.
        if value >= MIN_TRAP_PAYOFF_CP:
            mask |= 1 << index
    return mask & allowed


def compile_lines(
    lines: Sequence[TrapLine], masks: Optional[Mapping[str, int]] = None
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Fold curated lines into ``(zobrist key, raw move) -> (weight, band mask)``.

    Masks are OR-ed across lines, so a shared prefix such as 1...e5 stays
    available to whichever bands still have some live line that needs it.
    """
    folded: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for line in lines:
        board = chess.Board()
        for san in line.moves.split():
            try:
                move = board.parse_san(san)
            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as exc:
                raise ValueError(f"{line.name}: illegal move {san!r} after {board.fen()}") from exc
            if board.turn == line.owner:
                key = (chess.polyglot.zobrist_hash(board), polyglot_raw_move(board, move))
                mask = ALL_BANDS if masks is None else masks.get(line.name, ALL_BANDS)
                weight, bands = folded.get(key, (0, 0))
                folded[key] = (min(MAX_WEIGHT, weight + line.weight), bands | mask)
            board.push(move)
    return folded


def write_book(entries: Mapping[Tuple[int, int], Tuple[int, int]], destination: Path) -> int:
    """Write a sorted Polyglot book. Returns the entry count."""
    packed: List[bytes] = [
        ENTRY_STRUCT.pack(key, raw_move, weight, bands)
        for (key, raw_move), (weight, bands) in sorted(entries.items())
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".part")
    staging.write_bytes(b"".join(packed))
    staging.replace(destination)
    return len(packed)


def measure(leaf_depth: int = 12, path_floor: float = 1e-4) -> Dict[str, object]:
    """Roll every line forward through real Maia checkpoints and real Stockfish.

    A gambit cannot be valued one ply at a time. The searcher's per-move
    expectimax looks at a single opponent reply and then evaluates statically,
    so a knight sacrificed for a trap that springs three moves later always
    scores negative -- which is why every gambit entry move measured as
    unplayable at every rating. The payoff is conditional on the victim going
    wrong at a *specific later* decision, so the line must be rolled out:

        gain_k  = V_bot(after the victim plays the line move) - V_bot(best defence)
        reach_k = product of P_maia(victim plays the line move) up to k
        value   = sum over victim nodes of  reach_(k-1) * p_k * max(gain_k, 0)

    ``gain_k`` is what the victim's *error* is worth: near zero where the line
    move is simply normal play, large at the node where the trap actually bites.
    Weighting it by ``p_k`` -- how often Maia at that rating plays it -- and by
    the chance of reaching the node gives the expected centipawn payoff.

    Two earlier framings were wrong and are worth recording. Scoring a line by
    its *best* position values the closing mate as if it were free, marking
    every trap universally good. Scoring the whole rollout to its terminal value
    prices the line against best defence, which is about -200cp for any unsound
    gambit no matter who is sitting opposite -- true, but constant across
    ratings, so it cannot band anything. Only the victim's error rate at the
    biting node varies with skill, and that is what this measures.

    Engines are imported here so a plain book build stays free of them.
    """
    from src.engine import MaiaEvaluator, StockfishEvaluator
    from src.engine.search import AdversarialSearcher
    from src.types import SearchConfig

    settings = SearchConfig(leaf_depth=leaf_depth, root_depth=leaf_depth)
    lines: Dict[str, Dict[str, object]] = {
        line.name: {"owner": line.owner_name, "objective_worst": 0.0,
                    "utility": {}, "survival": {}}
        for line in TRAP_LINES
    }

    with StockfishEvaluator() as stockfish, MaiaEvaluator(RATING_BANDS[0]) as maia:
        searcher = AdversarialSearcher(stockfish, maia, config=settings)
        for band in RATING_BANDS:
            maia.set_rating(band)
            for line in TRAP_LINES:
                value, survival, worst = _roll_out(searcher, line, settings, path_floor)
                record = lines[line.name]
                utility, survivals = record["utility"], record["survival"]
                assert isinstance(utility, dict) and isinstance(survivals, dict)
                utility[str(band)] = round(value, 1)
                survivals[str(band)] = round(survival, 4)  # P(reaching and taking the bait)
                record["objective_worst"] = round(worst, 1)
            logger.info("measured maia-%d", band)

    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bands": list(RATING_BANDS),
        "settings": {"leaf_depth": leaf_depth},
        "lines": lines,
    }


def _roll_out(
    searcher: "AdversarialSearcher", line: TrapLine, settings: "SearchConfig", path_floor: float
) -> Tuple[float, float, float]:
    """Expected payoff of one line: ``(value, biting probability, worst objective)``."""
    board = chess.Board()
    reach = 1.0
    value = 0.0
    worst_objective = 0.0
    biting = 0.0
    best_gain = 0.0

    for san in line.moves.split():
        move = board.parse_san(san)
        if board.turn != line.owner:
            defended = searcher._evaluate_bot(board, line.owner, settings.leaf_depth)
            board.push(move)
            cooperated = searcher._evaluate_bot(board, line.owner, settings.leaf_depth)
            board.pop()

            gain = cooperated - defended
            probability = searcher.human_model.predict_move_probabilities(board)[move]
            if gain > 0.0:
                value += reach * probability * gain
                if gain > best_gain:
                    # The node where the trap actually bites, and how often they fall in.
                    best_gain, biting = gain, reach * probability
            reach *= probability
            if reach < path_floor:
                return value, biting, worst_objective
        else:
            board.push(move)
            worst_objective = min(
                worst_objective, float(searcher._evaluate_bot(board, line.owner, settings.leaf_depth))
            )
            board.pop()
        board.push(move)

    return value, biting, worst_objective


def entry_id(key: int, raw_move: int) -> str:
    """Stable JSON-safe identifier for one book entry."""
    return f"{key:016x}:{raw_move:04x}"


def _mask_from_ceiling(ceiling: int) -> int:
    mask = 0
    for index, band in enumerate(RATING_BANDS):
        if band <= ceiling:
            mask |= 1 << index
    return mask


def resolve_band_masks(path: Path = CEILINGS_PATH) -> Dict[str, int]:
    """Empirical band masks where they exist, curated ceilings where they do not.

    The calibration reports the bands that actually cleared the win-rate
    threshold, not a ceiling, because real curves are not monotone -- a sound
    gambit climbs with rating. A line that was scored but cleared nothing gets
    an empty mask: the data actively says it never pays, which is a different
    statement from having no data.
    """
    curated = {line.name: _mask_from_ceiling(line.ceiling) for line in TRAP_LINES}
    if not path.exists():
        logger.warning(
            "%s missing; every ceiling falls back to the curated value. "
            "Run: python -m src.engine.books.trueelo_scraper",
            path.name,
        )
        return curated

    payload = json.loads(path.read_text())
    measured = payload.get("lines", {})
    resolved = dict(curated)
    for name, record in measured.items():
        if name not in resolved:
            logger.warning("calibration mentions unknown line %r, ignoring it", name)
            continue
        mask = 0
        for band in record.get("bands", []):
            if int(band) in RATING_BANDS:
                mask |= 1 << RATING_BANDS.index(int(band))
        resolved[name] = mask

    absent = sorted(set(curated) - set(measured))
    for name in absent:
        logger.warning("no empirical curve for %-34s keeping curated ceiling", name)
    logger.info(
        "ceilings: %d empirical, %d curated fallbacks", len(curated) - len(absent), len(absent)
    )
    return resolved


def load_masks(path: Path = RATINGS_PATH) -> Optional[Dict[str, int]]:
    """Per-line band masks: measured payoff gated by the empirical ceiling."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    allowed = resolve_band_masks()
    return {
        name: eligible_bands(
            [float(record["utility"][str(band)]) for band in RATING_BANDS],
            float(record["objective_worst"]),
            allowed.get(name, ALL_BANDS),
        )
        for name, record in payload["lines"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.engine.books.build_trap_book")
    parser.add_argument("--measure", action="store_true",
                        help="Re-measure trap value per rating band with the real engines.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.measure:
        RATINGS_PATH.write_text(json.dumps(measure(), indent=2) + "\n")
        logger.info("wrote measurements to %s", RATINGS_PATH)

    masks = load_masks()
    if masks is None:
        logger.warning("no %s found; every line will be enabled at every rating", RATINGS_PATH.name)
    destination = BOOKS_DIR / "traps.bin"
    count = write_book(compile_lines(TRAP_LINES, masks), destination)
    logger.info(
        "wrote %d entries from %d lines to %s (%d bytes)",
        count, len(TRAP_LINES), destination, destination.stat().st_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
