"""Candidate proposal by the engine-annotated late-fusion scorer.

``NeuralCandidateGenerator`` ranks moves from the policy network alone. That
network reads a board and nothing else, which is exactly the limitation the
capacity sweep ran into: preference accuracy climbed with width right up to the
point where the distilled representation started collapsing again.

This proposer gives the same frozen prior the evidence it was missing. It runs
one Stockfish MultiPV scan, turns each candidate into the four features the
scorer was trained on, and re-ranks by ``prior_logit + trap_modifier``. The scan
is the one the search is about to run anyway, and inside a search whose hash is
already warm it costs close to nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import chess
import torch

from src.engine.policy_generator import PolicyUnavailableError
from src.engine.search import PositionEvaluator, _rank_mover_relative
from src.training.annotate import MoveFeatures
from src.training.model import TrapPolicyNet, TrapScorer, legal_move_mask, move_to_index

__all__ = ["ScorerCandidateProposer", "DEFAULT_SCORER"]

logger = logging.getLogger(__name__)

DEFAULT_SCORER: Final[Path] = Path("models/trap_scorer.pth")
DEFAULT_SCAN_DEPTH: Final[int] = 6
"""Matches the annotation pass the scorer was trained against. Feeding it
features from a different depth would be a train/serve skew."""

DEFAULT_SCAN_WIDTH: Final[int] = 12
"""Candidates the scan ranks before the scorer re-orders them. The scorer can
only promote a move Stockfish already listed, which is what keeps a proposal
from being arbitrary."""


class ScorerCandidateProposer:
    """Re-ranks Stockfish candidates with the trained ``TrapScorer``."""

    def __init__(
        self,
        evaluator: PositionEvaluator,
        *,
        prior_path: Path,
        scorer_path: Path = DEFAULT_SCORER,
        device: str = "cpu",
        scan_depth: int = DEFAULT_SCAN_DEPTH,
        scan_width: int = DEFAULT_SCAN_WIDTH,
    ) -> None:
        self.evaluator = evaluator
        self.device = torch.device(device)
        self.scan_depth = scan_depth
        self.scan_width = scan_width
        self.prior = self._load_prior(prior_path, self.device)
        self.scorer = self._load_scorer(scorer_path, self.device)
        logger.info(
            "scorer-proposer: %s over %s, depth %d width %d",
            scorer_path.name, prior_path.name, scan_depth, scan_width,
        )

    # -- loading ------------------------------------------------------------

    @staticmethod
    def _load_prior(path: Path, device: torch.device) -> TrapPolicyNet:
        if not path.exists():
            raise PolicyUnavailableError(f"no distilled prior at {path}")
        payload: Dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        model = TrapPolicyNet(
            channels=int(payload.get("channels", 64)),
            blocks=int(payload.get("blocks", 4)),
            residual_hidden=int(payload.get("residual_hidden", 0)),
        )
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        if unexpected or any(not name.startswith("residual.") for name in missing):
            raise PolicyUnavailableError(f"{path} does not match this architecture")
        model.eval()
        return model.to(device)

    @staticmethod
    def _load_scorer(path: Path, device: torch.device) -> TrapScorer:
        if not path.exists():
            raise PolicyUnavailableError(
                f"no scorer at {path}; run python -m src.training.train_dpo --late-fusion"
            )
        payload: Dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        scorer = TrapScorer(hidden=int(payload.get("hidden", 96)))
        scorer.load_state_dict(payload["state_dict"])
        scorer.eval()
        return scorer.to(device)

    # -- proposal -----------------------------------------------------------

    @torch.no_grad()
    def get_candidates(
        self, board: chess.Board, rating: int, top_k: int = 5
    ) -> List[chess.Move]:
        """The ``top_k`` moves the fused score rates highest, best first."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        legal_count = board.legal_moves.count()
        if legal_count == 0:
            return []
        if legal_count <= top_k:
            return list(board.legal_moves)

        ranked = self._scan(board, legal_count)
        if not ranked:
            return []

        prior_logits = self._prior_logits(board, rating)
        best = ranked[0][1]
        scored: List[Tuple[float, str, chess.Move]] = []
        for position, (move, centipawns) in enumerate(ranked, start=1):
            features = MoveFeatures(
                centipawns=centipawns,
                loss_vs_best=max(0, best - centipawns),
                rank=position,
                is_top_choice=position == 1,
                legal_moves=legal_count,
            )
            index = move_to_index(move)
            prior = prior_logits[index]
            modifier = self.scorer(
                prior.reshape(1),
                torch.tensor([features.as_vector()], dtype=torch.float32, device=self.device),
            )
            scored.append((float(prior + modifier[0]), move.uci(), move))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [move for _score, _uci, move in scored[:top_k]]

    def _scan(self, board: chess.Board, legal_count: int) -> List[Tuple[chess.Move, int]]:
        """Top Stockfish candidates, mover-relative, best first."""
        width = min(self.scan_width, legal_count)
        try:
            scores = self.evaluator.analyse_root_moves(
                board, depth=self.scan_depth, multipv=width
            )
        except Exception as exc:  # noqa: BLE001 - a dud scan must not end the search
            logger.warning("scorer-proposer: scan failed (%s), proposing nothing", exc)
            return []
        return _rank_mover_relative(scores, board.turn)

    def _prior_logits(self, board: chess.Board, rating: int) -> torch.Tensor:
        from src.training.model import encode_board

        planes = encode_board(board, rating).unsqueeze(0).to(self.device)
        logits: torch.Tensor = self.prior(planes)[0]
        mask = legal_move_mask(board).to(self.device)
        return logits.masked_fill(~mask, float("-inf"))
