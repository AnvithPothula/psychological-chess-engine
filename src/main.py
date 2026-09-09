"""Entry point: wires the evaluators, the searcher and the Pygame UI together.

    python -m src.main --rating 1500 --color black

The evaluators are opened in a ``with`` block, so the ``stockfish`` and ``lc0``
child processes are terminated when the window closes -- including when the UI
raises on the way out.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

import chess

from src.config import AVAILABLE_MAIA_RATINGS, DEFAULT_MAIA_RATING
from src.engine import EvaluatorError, MaiaEvaluator, StockfishEvaluator
from src.engine.search import AdversarialSearcher
from src.types import SearchConfig
from src.ui.app import ChessApp
from src.ui.asset_manager import AssetManager

logger = logging.getLogger("psychological-chess")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.main", description="Play the adversarial chess engine."
    )
    parser.add_argument(
        "--rating",
        type=int,
        default=DEFAULT_MAIA_RATING,
        choices=AVAILABLE_MAIA_RATINGS,
        help="Maia rating used to model your moves (default: %(default)s).",
    )
    parser.add_argument(
        "--color",
        choices=("white", "black"),
        default="white",
        help="Colour you play (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # python-chess logs every UCI line at DEBUG, which drowns out our telemetry.
    logging.getLogger("chess.engine").setLevel(logging.WARNING)

    human_color = chess.WHITE if args.color == "white" else chess.BLACK
    assets = AssetManager()
    assets.ensure_assets()

    try:
        with StockfishEvaluator() as stockfish, MaiaEvaluator(rating=args.rating) as maia:
            searcher = AdversarialSearcher(stockfish, maia, config=SearchConfig())
            logger.info(
                "starting: you are %s against Maia-%d", args.color, args.rating
            )
            ChessApp(
                searcher,
                assets,
                human_color=human_color,
                opponent_label=f"Opponent model: Maia-{args.rating}",
            ).run()
    except FileNotFoundError as exc:
        logger.error("missing dependency: %s", exc)
        return 1
    except EvaluatorError as exc:
        logger.error("engine failure: %s", exc)
        return 1
    logger.info("clean shutdown: engine processes terminated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
