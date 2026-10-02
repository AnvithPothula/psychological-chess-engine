"""Game review bookkeeping: error classes and per-game summaries."""

from __future__ import annotations

from src.eval.game_review import _summary, classify


def test_lichess_thresholds() -> None:
    assert [classify(x) for x in (0, 49, 50, 99, 100, 299, 300)] == [
        "", "", "inaccuracy", "inaccuracy", "mistake", "mistake", "blunder"]


def test_a_delivered_mate_is_not_a_missed_one() -> None:
    """The first run counted every game's mating move as a missed mate."""
    def ply(side: str, loss: int, had: bool, kept: bool) -> dict[str, object]:
        return {"side": side, "loss": loss, "class": classify(loss), "book": "", "think": 1.0,
                "clock": 100.0, "had_mate": had, "kept_mate": kept}

    review = {"plies": [ply("bot", 0, True, True), ply("opp", 400, False, False),
                        ply("bot", 0, True, False)], "decisive_ply": 0}
    summary = _summary(review)
    assert summary["missed_mates"] == 1 and summary["opp_errors"] == (0, 0, 1)
    assert summary["plies_to_finish"] == 3
