"""Tests for the empirical gambit calibrator.

Entirely offline: HTTP goes through a stub fetcher and the HTML fixtures are
trimmed copies of TrueElo's real markup, React comment separators included.

    python -m tests.test_calibration
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import chess

from src.engine.books.build_trap_book import (
    ALL_BANDS,
    RATING_BANDS,
    TRAP_LINES,
    TrapLine,
    resolve_band_masks,
)
from src.engine.books.trueelo_scraper import (
    BandScore,
    HttpFetcher,
    LichessExplorerProvider,
    RatingCurve,
    TrueEloProvider,
    calibrate,
    passing_bands,
)

# The real page renders scores in a screen-reader list with React's <!-- -->
# separators between every text node. Keep them: stripping them is the parse.
SR_ROW = (
    "<li>{low}-{high}<!-- --> rating, <!-- -->{speed}<!-- -->: <!-- -->{score}%<!-- --> "
    "expected score (<!-- -->{games}<!-- --> games)</li>"
)


def trueelo_html(
    perspective: str, rows: List[Dict[str, object]], *, pgn: str = "1.e4+e5+2.Nf3+Nf6"
) -> str:
    body = "".join(SR_ROW.format(**row) for row in rows)
    return (
        f'<h2>{perspective}<!-- --> expected score · <!-- -->All ratings</h2>'
        f'<a href="/?pgn={pgn}">Analyze this line move by move</a>'
        f"<div class='sr-only'><ul>{body}</ul></div>"
    )


def rows_at(score: float, games: int = 50_000) -> List[Dict[str, object]]:
    return [
        {"low": low, "high": low + 199, "speed": speed, "score": score, "games": f"{games:,}"}
        for low in (1000, 1200, 1400, 1600, 1800)
        for speed in ("Blitz", "Rapid")
    ]


class StubFetcher(HttpFetcher):
    """HttpFetcher with the network replaced by a dict."""

    def __init__(self, responses: Dict[str, Optional[str]]) -> None:
        super().__init__(cache_dir=Path(tempfile.mkdtemp()), interval=0.0)
        self.responses = responses
        self.requested: List[str] = []

    def get(self, url: str) -> Optional[str]:
        self.requested.append(url)
        for fragment, body in self.responses.items():
            if fragment in url:
                return body
        return None


def curve_of(scores: Dict[int, float], games: int = 50_000) -> RatingCurve:
    return RatingCurve(
        line="test",
        source="stub",
        bands=tuple(BandScore(band, score, games) for band, score in sorted(scores.items())),
    )


# --- breakpoint logic ------------------------------------------------------


def test_falling_curve_yields_a_ceiling() -> None:
    """The Stafford shape: strong low down, breaks even in the high bands."""
    curve = curve_of({1000: 0.62, 1200: 0.55, 1400: 0.53, 1600: 0.51, 1800: 0.47, 2000: 0.44})
    live = passing_bands(curve)
    assert live and max(live) == 1700, f"expected a ceiling at 1700, got {live}"
    assert 1800 not in live and 1900 not in live


def test_rising_curve_is_not_killed_by_its_first_weak_band() -> None:
    """Regression: sound gambits climb with rating.

    Reading the breakpoint as "everything below the first sub-50% band" switches
    the Smith-Morra off entirely -- it opens at 49% and reaches 56% at 2500+.
    """
    curve = curve_of({1000: 0.493, 1200: 0.494, 1400: 0.496, 1600: 0.507, 1800: 0.516, 2000: 0.524})
    live = passing_bands(curve)
    assert 1900 in live, "a gambit that improves with rating must stay on at the top"
    assert 1100 not in live, "and stay off in the bands where it genuinely underperforms"


def test_thin_brackets_yield_no_verdict_either_way() -> None:
    """Sparse brackets neither enable their bands nor condemn them.

    Absence of evidence resolves to "not enabled", which is the conservative
    direction for a trap book: never spring an unsound line on a strong player
    on the strength of forty games.
    """
    sparse = RatingCurve("test", "stub", (
        BandScore(1000, 0.90, 12),      # spectacular, but 12 games
        BandScore(1400, 0.55, 80_000),  # the only bracket with real evidence
        BandScore(1800, 0.20, 40),      # catastrophic, but 40 games
    ))
    live = passing_bands(sparse, min_games=500)
    assert 1100 not in live, "12 games must not enable a band"
    assert {1400, 1500, 1600, 1700} <= set(live), "the well-evidenced bracket must enable its bands"
    assert 1800 not in live and 1900 not in live, "40 games must not enable them either"


def test_no_band_clears_the_threshold() -> None:
    assert passing_bands(curve_of({1000: 0.42, 1800: 0.40})) == ()


# --- TrueElo parsing -------------------------------------------------------


def test_parse_strips_react_separators_and_weights_by_games() -> None:
    html = trueelo_html("Black", [
        {"low": 1400, "high": 1599, "speed": "Blitz", "score": 60.0, "games": "90,000"},
        {"low": 1400, "high": 1599, "speed": "Rapid", "score": 40.0, "games": "10,000"},
        {"low": 1400, "high": 1599, "speed": "Bullet", "score": 99.0, "games": "500,000"},
    ])
    bands = TrueEloProvider.parse(html, chess.BLACK)
    assert len(bands) == 1
    assert bands[0].games == 100_000, "bullet must be excluded; we decline those games"
    assert abs(bands[0].expected_score - 0.58) < 1e-6, "games-weighted, not a plain mean"


def test_parse_flips_when_the_page_perspective_differs() -> None:
    html = trueelo_html("White", [
        {"low": 1400, "high": 1599, "speed": "Blitz", "score": 70.0, "games": "10,000"},
    ])
    as_white = TrueEloProvider.parse(html, chess.WHITE)
    as_black = TrueEloProvider.parse(html, chess.BLACK)
    assert abs(as_white[0].expected_score - 0.70) < 1e-9
    assert abs(as_black[0].expected_score - 0.30) < 1e-9, "must invert for the other side"


def test_slug_pointing_at_a_different_opening_is_rejected() -> None:
    """HTTP 200 is not evidence the slug is the right opening."""
    line = TrapLine("test", chess.WHITE, "e4 e5 Nf3 Nc6 Bc4 d6 Nc3 Bg4")
    assert TrueEloProvider.covers_line(trueelo_html("White", [], pgn="1.e4+e5+2.Nf3+Nc6"), line)
    assert not TrueEloProvider.covers_line(
        trueelo_html("White", [], pgn="1.d4+d5+2.c4"), line
    ), "a Queen's Gambit page must not calibrate an Italian Game line"
    assert not TrueEloProvider.covers_line(trueelo_html("White", [], pgn=""), line)


def test_provider_discards_a_mismatched_slug_rather_than_guessing() -> None:
    line = TrapLine("Mismatched", chess.WHITE, "e4 e5 Nf3 Nc6")
    fetcher = StubFetcher({"trueelo": trueelo_html("White", rows_at(70.0), pgn="1.d4+d5")})
    assert TrueEloProvider(fetcher, {"Mismatched": "some-slug"}).curve(line) is None


# --- explorer position selection -------------------------------------------


def test_explorer_never_scores_a_terminal_final_position() -> None:
    """Five lines end in checkmate; those positions have no games at all."""
    mating = next(line for line in TRAP_LINES if line.name == "Blackburne Shilling Gambit")
    board = chess.Board()
    for san in mating.moves.split():
        board.push_san(san)
    assert board.is_checkmate(), "fixture assumption: this line ends in mate"

    positions = LichessExplorerProvider._positions_deepest_first(mating)
    assert positions, "the line must still offer scoreable positions"
    assert not any(position.is_checkmate() for position in positions)
    assert all(position.turn == mating.owner for position in positions), (
        "only positions where the trap owner chooses can be scored as their choice"
    )
    plies = [position.ply() for position in positions]
    assert plies == sorted(plies, reverse=True), "deepest first, then walk back for data"


def test_explorer_provider_disables_itself_when_the_endpoint_is_down() -> None:
    fetcher = StubFetcher({})  # every request returns None
    provider = LichessExplorerProvider(fetcher)
    for line in TRAP_LINES[:5]:
        assert provider.curve(line) is None
    assert not provider.available, "a dead endpoint must not be retried for every line"
    before = len(fetcher.requested)
    provider.curve(TRAP_LINES[6])
    assert len(fetcher.requested) == before, "no further requests once written off"


# --- end to end ------------------------------------------------------------


def test_calibrate_falls_back_and_records_provenance() -> None:
    covered = TrapLine("Covered", chess.WHITE, "e4 e5 Nf3 Nc6", ceiling=1500)
    uncovered = TrapLine("Uncovered", chess.BLACK, "e4 e5 Nf3 Nc6", ceiling=1300)
    fetcher = StubFetcher({"trueelo": trueelo_html("White", rows_at(70.0), pgn="1.e4+e5")})
    provider = TrueEloProvider(fetcher, {"Covered": "covered-slug"})

    payload = calibrate([covered, uncovered], [provider])
    lines = payload["lines"]
    assert isinstance(lines, dict)
    assert "Covered" in lines and "Uncovered" not in lines
    record = lines["Covered"]
    assert isinstance(record, dict)
    assert record["source"] == "trueelo" and record["curated_ceiling"] == 1500
    assert record["bands"], "a 70% line must clear the threshold somewhere"


def test_resolve_band_masks_prefers_data_and_falls_back_quietly() -> None:
    name = TRAP_LINES[0].name
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trap_ceilings.json"
        path.write_text(json.dumps({"lines": {name: {"bands": [1100, 1200, 1300]}}}))
        masks = resolve_band_masks(path)

    assert masks[name] == 0b111, "empirical bands must win"
    other = TRAP_LINES[1]
    assert masks[other.name] != 0, "unmeasured lines keep their curated ceiling"
    assert masks[other.name] & ~ALL_BANDS == 0

    missing = resolve_band_masks(Path("/nonexistent/trap_ceilings.json"))
    assert set(missing) == {line.name for line in TRAP_LINES}


def test_shipped_calibration_is_consistent() -> None:
    from src.engine.books.build_trap_book import CEILINGS_PATH

    if not CEILINGS_PATH.exists():
        print("    (no trap_ceilings.json; run the scraper)")
        return
    payload = json.loads(CEILINGS_PATH.read_text())
    known = {line.name: line for line in TRAP_LINES}
    for name, record in payload["lines"].items():
        assert name in known, f"calibration names an unknown line: {name}"
        assert record["owner"] == known[name].owner_name
        for band in record["bands"]:
            assert band in RATING_BANDS
    print(f"    {len(payload['lines'])}/{len(TRAP_LINES)} lines carry empirical curves")


def _main() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - standalone runner reports everything
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print("all green" if not failures else f"{failures} failing test(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
