"""DPO training for the trap policy, without TRL and without a reference model.

**What this loss actually is.** The brief drops the reference-model terms, so

    L = -log sigma( beta * ( log P(chosen|x) - log P(rejected|x) ) )

and because ``log P(a) = z_a - logsumexp(z)``, the normaliser cancels in the
difference and the loss is exactly

    L = -log sigma( beta * ( z_chosen - z_rejected ) )

a pairwise Bradley-Terry ranking loss on two raw logits. Two consequences follow
and both need handling rather than hoping:

1. **It is unbounded.** Nothing anchors absolute logit scale, so the minimiser
   drives ``z_chosen`` up and ``z_rejected`` down without limit. Real DPO is
   bounded because the reference model pins both.
2. **It touches two of 4096 outputs.** The other 4094 logits receive no
   gradient, so a softmax over them is not a policy -- it is whatever the
   initialisation happened to produce. The inference wrapper's contract, mask
   then softmax then take the top K, would be reading noise.

The fix is one extra term: a legal-move-masked cross-entropy toward the chosen
move. It normalises over the whole legal set, so every legal logit gets a
gradient and the output becomes a distribution, while the pairwise term keeps
doing the job of encoding the preference over Stockfish's move. ``anchor_weight``
trades them off; setting it to zero reproduces the brief exactly, and
``--diagnose`` shows what that costs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import chess
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.training.model import (
    ENGINE_FEATURE_WIDTH,
    INPUT_PLANES,
    POLICY_SIZE,
    TrapPolicyNet,
    TrapScorer,
    encode_board,
    legal_move_mask,
    move_to_index,
)

__all__ = [
    "PreferenceDataset",
    "TrainConfig",
    "anchor_loss",
    "load_warm_start",
    "preference_loss",
    "select_device",
    "train",
    "train_scorer",
    "main",
]

logger = logging.getLogger(__name__)

DEFAULT_DATASET: Final[Path] = Path("build/dpo_dataset_clean.jsonl")
DEFAULT_CHECKPOINT: Final[Path] = Path("models/trap_policy.pth")
CHECKPOINT_VERSION: Final[int] = 1
MIN_BATCH_FOR_BATCHNORM: Final[int] = 2
"""BatchNorm cannot compute statistics from a single sample in train mode."""

DEFAULT_EPOCHS: Final[int] = 60
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_BETA: Final[float] = 0.1
DEFAULT_ANCHOR_WEIGHT: Final[float] = 1.0
DEFAULT_WARM_LEARNING_RATE: Final[float] = 1e-4
"""Warm starts get a tenth of the cold learning rate. The distilled trunk is
the only chess knowledge in the system, and 170 preference pairs at 1e-3 would
wash it out in the first few steps -- alignment is meant to nudge a policy, not
retrain it."""

WARM_CHECKPOINT: Final[Path] = Path("models/policy_warm.pth")


def select_device(preference: Optional[str] = None) -> torch.device:
    """CUDA, then Apple's MPS, then CPU. ``preference`` overrides the order."""
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = 1e-4
    beta: float = DEFAULT_BETA
    """DPO temperature on the logit gap."""

    anchor_weight: float = DEFAULT_ANCHOR_WEIGHT
    """Weight on the masked cross-entropy term. Zero reproduces the brief's
    loss exactly, and produces logits that are not a usable distribution."""

    residual_hidden: int = 0
    """Width of the residual head's hidden 3x3 layer. Zero leaves it a 1x1
    convolution, which sums with the policy head into a single 1x1 convolution
    -- a channel mixer that cannot represent a spatial pattern."""

    freeze_policy_head: bool = True
    """Freeze the distilled policy head alongside the trunk. Releasing it is
    only meaningful when ``residual_hidden`` is non-zero."""

    validation_fraction: float = 0.2
    seed: int = 20260910


@dataclass(frozen=True, slots=True)
class Sample:
    planes: Tensor
    chosen: int
    rejected: int
    legal: Tensor


class PreferenceDataset(Dataset[Sample]):
    """Preference records encoded once, up front.

    Encoding is pure Python and would otherwise dominate every epoch; these
    datasets are small enough to hold in memory by a wide margin.
    """

    def __init__(self, records: Sequence[Dict[str, Any]]) -> None:
        self.samples: List[Sample] = []
        skipped = 0
        for record in records:
            sample = self._encode(record)
            if sample is None:
                skipped += 1
                continue
            self.samples.append(sample)
        if skipped:
            logger.warning("dataset: skipped %d records whose moves were not legal", skipped)
        if not self.samples:
            raise ValueError("no usable preference records")

    @staticmethod
    def _encode(record: Dict[str, Any]) -> Optional[Sample]:
        try:
            board = chess.Board(str(record["fen"]))
            chosen = board.parse_san(str(record["chosen"]))
            rejected = board.parse_san(str(record["rejected"]))
        except (KeyError, ValueError):
            return None
        if chosen == rejected:
            return None  # A pair with no preference in it teaches nothing.
        rating = int(record.get("opponent_rating", 1500))
        return Sample(
            planes=encode_board(board, rating),
            chosen=move_to_index(chosen),
            rejected=move_to_index(rejected),
            legal=legal_move_mask(board),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


def collate(batch: Sequence[Sample]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.stack([sample.planes for sample in batch]),
        torch.tensor([sample.chosen for sample in batch], dtype=torch.long),
        torch.tensor([sample.rejected for sample in batch], dtype=torch.long),
        torch.stack([sample.legal for sample in batch]),
    )


def preference_loss(logits: Tensor, chosen: Tensor, rejected: Tensor, beta: float) -> Tensor:
    """Bradley-Terry on the logit gap -- the brief's loss, exactly.

    ``logsumexp`` cancels between the two log-probabilities, so gathering raw
    logits is not an approximation of the stated formula; it *is* the stated
    formula, computed without the cancelling term.
    """
    gap = logits.gather(1, chosen.unsqueeze(1)) - logits.gather(1, rejected.unsqueeze(1))
    return -F.logsigmoid(beta * gap.squeeze(1)).mean()


def anchor_loss(logits: Tensor, chosen: Tensor, legal: Tensor) -> Tensor:
    """Masked cross-entropy toward the chosen move over the whole legal set."""
    masked = logits.masked_fill(~legal, float("-inf"))
    return F.cross_entropy(masked, chosen)


def accuracy(logits: Tensor, chosen: Tensor, legal: Tensor) -> float:
    """How often the chosen move is the argmax over legal moves."""
    masked = logits.masked_fill(~legal, float("-inf"))
    return float((masked.argmax(dim=1) == chosen).float().mean().item())


def preference_accuracy(logits: Tensor, chosen: Tensor, rejected: Tensor) -> float:
    """How often the chosen move outranks the rejected one."""
    gap = logits.gather(1, chosen.unsqueeze(1)) - logits.gather(1, rejected.unsqueeze(1))
    return float((gap.squeeze(1) > 0).float().mean().item())


def _run_epoch(
    model: TrapPolicyNet,
    loader: DataLoader[Sample],
    config: TrainConfig,
    device: torch.device,
    optimiser: Optional[torch.optim.Optimizer],
) -> Tuple[float, float, float]:
    training = optimiser is not None
    model.train(training)
    totals = [0.0, 0.0, 0.0]
    seen = 0

    for planes, chosen, rejected, legal in loader:
        if training and planes.shape[0] < MIN_BATCH_FOR_BATCHNORM:
            continue  # BatchNorm needs at least two samples to have a variance.
        planes = planes.to(device)
        chosen, rejected, legal = chosen.to(device), rejected.to(device), legal.to(device)

        with torch.set_grad_enabled(training):
            logits = model(planes)
            loss = preference_loss(logits, chosen, rejected, config.beta)
            if config.anchor_weight:
                loss = loss + config.anchor_weight * anchor_loss(logits, chosen, legal)

        if training and optimiser is not None:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()

        count = planes.shape[0]
        totals[0] += float(loss.item()) * count
        totals[1] += preference_accuracy(logits, chosen, rejected) * count
        totals[2] += accuracy(logits, chosen, legal) * count
        seen += count

    if seen == 0:
        return math.nan, 0.0, 0.0
    return totals[0] / seen, totals[1] / seen, totals[2] / seen


def load_warm_start(
    path: Path, device: torch.device, *, residual_hidden: int = 0
) -> Tuple[TrapPolicyNet, int, int]:
    """Rebuild the distilled network from its checkpoint, shapes checked.

    The checkpoint carries its own encoding, so a corpus built under a different
    plane layout is caught here rather than training silently against garbage.
    """
    payload: Dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    for key, expected in (("input_planes", INPUT_PLANES), ("policy_size", POLICY_SIZE)):
        actual = payload.get(key)
        if actual is not None and int(actual) != expected:
            raise ValueError(f"{path} was built with {key}={actual}, this build expects {expected}")
    channels = int(payload.get("channels", 64))
    blocks = int(payload.get("blocks", 4))
    stored = payload.get("residual_hidden")
    width = residual_hidden if stored is None else int(stored)
    model = TrapPolicyNet(channels=channels, blocks=blocks, residual_hidden=width)
    # A checkpoint distilled before the residual head existed carries no
    # residual.* tensors. Leaving them at their zero initialisation is exactly
    # the intended state: the network reproduces the prior until DPO moves it.
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise ValueError(f"{path} carries unknown tensors: {sorted(unexpected)}")
    if any(not name.startswith("residual.") for name in missing):
        raise ValueError(f"{path} is missing prior tensors: {sorted(missing)}")
    logger.info(
        "train: warm start from %s (stage=%s, metrics=%s)%s",
        path.name, payload.get("stage", "?"), payload.get("metrics", {}),
        " [residual head zero-initialised]" if missing else "",
    )
    return model.to(device), channels, blocks


def train(
    dataset_path: Path,
    checkpoint_path: Path,
    config: Optional[TrainConfig] = None,
    *,
    device: Optional[torch.device] = None,
    channels: int = 64,
    blocks: int = 4,
    warm_start: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train the policy and write a checkpoint. Returns the run's metrics."""
    settings = config if config is not None else TrainConfig()
    torch.manual_seed(settings.seed)
    random.seed(settings.seed)
    target = device if device is not None else select_device()

    records = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    dataset = PreferenceDataset(records)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = max(1, int(len(indices) * settings.validation_fraction)) if len(indices) > 4 else 0
    validation_ids, training_ids = indices[:split], indices[split:]

    train_set = torch.utils.data.Subset(dataset, training_ids)
    train_loader: DataLoader[Sample] = DataLoader(
        train_set, batch_size=settings.batch_size, shuffle=True, collate_fn=collate
    )
    validation_loader: Optional[DataLoader[Sample]] = None
    if validation_ids:
        validation_loader = DataLoader(
            torch.utils.data.Subset(dataset, validation_ids),
            batch_size=settings.batch_size, shuffle=False, collate_fn=collate,
        )

    if warm_start is not None:
        model, channels, blocks = load_warm_start(
            warm_start, target, residual_hidden=settings.residual_hidden
        )
        # Matilda: the distilled representation is the only chess knowledge in
        # the system, so alignment does not get to touch it. The trunk is always
        # frozen; whether the policy head joins it is a setting, because
        # releasing it only widens the function class when the residual has a
        # non-linearity of its own.
        model.freeze_prior(include_policy=settings.freeze_policy_head)
        trainable = model.trainable_parameters
    else:
        model = TrapPolicyNet(
            channels=channels, blocks=blocks, residual_hidden=settings.residual_hidden
        ).to(target)
        trainable = list(model.parameters())
    optimiser = torch.optim.AdamW(
        trainable, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    # Cosine annealing to near-zero: the last epochs should barely move a warm
    # trunk, which is what keeps the distilled representation intact.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, settings.epochs))
    logger.info(
        "train: %s of %s parameters trainable",
        f"{sum(p.numel() for p in trainable):,}", f"{model.parameter_count:,}",
    )
    logger.info(
        "train: %d records (%d train / %d val) on %s, %s parameters",
        len(dataset), len(training_ids), len(validation_ids), target, f"{model.parameter_count:,}",
    )
    if len(dataset) < model.parameter_count / 1000:
        logger.warning(
            "train: %d examples against %s parameters -- this will memorise, not generalise",
            len(dataset), f"{model.parameter_count:,}",
        )

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Tensor]] = None
    best_pref = -1.0
    best_epoch = 0
    for epoch in range(1, settings.epochs + 1):
        loss, pref, top1 = _run_epoch(model, train_loader, settings, target, optimiser)
        schedule.step()
        entry: Dict[str, Any] = {"epoch": epoch, "loss": round(loss, 4), "pref_acc": round(pref, 4), "top1": round(top1, 4)}
        if validation_loader is not None:
            val_loss, val_pref, val_top1 = _run_epoch(model, validation_loader, settings, target, None)
            entry.update(val_loss=round(val_loss, 4), val_pref=round(val_pref, 4), val_top1=round(val_top1, 4))
            if val_pref > best_pref:
                best_pref, best_epoch = val_pref, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(entry)
        if epoch % max(1, settings.epochs // 10) == 0 or epoch == 1:
            logger.info("train: %s", entry)

    if best_state is not None and best_epoch != settings.epochs:
        logger.info("train: restoring epoch %d (val_pref=%.4f) over the final epoch", best_epoch, best_pref)
        model.load_state_dict(best_state)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "state_dict": model.state_dict(),
            "channels": channels,
            "blocks": blocks,
            # Without this a wide residual cannot be rebuilt: the state dict
            # would carry residual.0/residual.2 tensors the default 1x1 head
            # has no slots for.
            "residual_hidden": model.residual_hidden,
            "input_planes": INPUT_PLANES,
            "policy_size": POLICY_SIZE,
            "records": len(dataset),
            "stage": "aligned" if warm_start is not None else "cold",
            "warm_start": str(warm_start) if warm_start is not None else None,
            "config": asdict(settings),
        },
        checkpoint_path,
    )
    logger.info("train: wrote %s", checkpoint_path)
    return {"history": history, "records": len(dataset), "parameters": model.parameter_count}


def diagnose(dataset_path: Path, config: TrainConfig, device: torch.device) -> None:
    """Train twice, with and without the anchor, and compare the distributions.

    Demonstrates the point the module docstring makes: the brief's loss alone
    leaves the softmax unusable because 4094 of 4096 logits never move.
    """
    for anchor in (0.0, config.anchor_weight):
        settings = replace(config, anchor_weight=anchor)
        model = TrapPolicyNet().to(device)
        records = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        dataset = PreferenceDataset(records)
        loader: DataLoader[Sample] = DataLoader(
            dataset, batch_size=settings.batch_size, shuffle=True, collate_fn=collate
        )
        optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
        for _ in range(settings.epochs):
            _run_epoch(model, loader, settings, device, optimiser)

        model.eval()
        planes, chosen, rejected, legal = collate([dataset[i] for i in range(min(64, len(dataset)))])
        with torch.no_grad():
            logits = model(planes.to(device))
        masked = logits.masked_fill(~legal.to(device), float("-inf"))
        probs = F.softmax(masked, dim=1)
        top = probs.max(dim=1).values.mean().item()
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean().item())
        logger.info(
            "diagnose: anchor=%.1f  pref_acc=%.2f  top1=%.2f  mean max prob=%.3f  entropy=%.2f",
            anchor,
            preference_accuracy(logits, chosen.to(device), rejected.to(device)),
            accuracy(logits, chosen.to(device), legal.to(device)),
            top, entropy,
        )


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m src.training.train_dpo")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--warm-start", type=Path, default=None, metavar="PATH",
                        help=f"Initialise from a distilled checkpoint, e.g. {WARM_CHECKPOINT}.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--anchor-weight", type=float, default=DEFAULT_ANCHOR_WEIGHT,
                        help="0.0 reproduces the brief's loss exactly.")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available).")
    parser.add_argument("--late-fusion", action="store_true",
                        help="Train the engine-annotated TrapScorer over a frozen prior "
                             "instead of fine-tuning the convolutional network.")
    parser.add_argument("--scorer-hidden", type=int, default=96,
                        help="Hidden width of the late-fusion MLP.")
    parser.add_argument("--residual-hidden", type=int, default=0,
                        help="Hidden width of the residual head. 0 keeps it a 1x1 conv, "
                             "which adds no capacity over the policy head it sits beside.")
    parser.add_argument("--unfreeze-policy", action="store_true",
                        help="Train the distilled policy head alongside the residual.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Compare training with and without the anchor term.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    if not args.dataset.exists():
        logger.error("no dataset at %s; run python -m src.training.dedup first", args.dataset)
        return 1

    warm = args.warm_start
    if warm is not None and not warm.exists():
        logger.error("no warm checkpoint at %s; run python -m src.training.distill first", warm)
        return 1
    # A warm start defaults to the gentler rate unless the caller says otherwise.
    learning_rate = args.lr
    if warm is not None and args.lr == DEFAULT_LEARNING_RATE:
        learning_rate = DEFAULT_WARM_LEARNING_RATE

    if args.late_fusion:
        if warm is None:
            logger.error("train: --late-fusion needs a --warm-start prior to sit on")
            return 1
        summary = train_scorer(
            args.dataset, args.checkpoint, warm,
            TrainConfig(
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=learning_rate, beta=args.beta,
            ),
            device=select_device(args.device) if args.device else None,
            hidden=args.scorer_hidden,
        )
        logger.info("train: best val_pref %.4f at epoch %d",
                    summary["best_val_pref"], summary["best_epoch"])
        return 0

    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=learning_rate,
        beta=args.beta, anchor_weight=args.anchor_weight,
        residual_hidden=args.residual_hidden,
        freeze_policy_head=not args.unfreeze_policy,
    )
    device = select_device(args.device)
    if args.diagnose:
        diagnose(args.dataset, config, device)
        return 0
    train(args.dataset, args.checkpoint, config, device=device, warm_start=warm)
    return 0



# --- late fusion -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FusedSample:
    planes: Tensor
    chosen: int
    rejected: int
    chosen_features: Tensor
    rejected_features: Tensor


class AnnotatedDataset(Dataset[FusedSample]):
    """Preference pairs carrying per-move engine features.

    Records without ``chosen_features`` are skipped rather than zero-filled: an
    all-zero feature vector reads as "a balanced position the engine likes",
    which is the opposite of what a missing annotation means.
    """

    def __init__(self, records: Sequence[Dict[str, Any]]) -> None:
        self.samples: List[FusedSample] = []
        skipped = 0
        for record in records:
            sample = self._encode(record)
            if sample is None:
                skipped += 1
                continue
            self.samples.append(sample)
        if skipped:
            logger.warning("dataset: skipped %d records without engine features", skipped)
        if not self.samples:
            raise ValueError("no annotated preference records; run src.training.annotate first")

    @staticmethod
    def _encode(record: Dict[str, Any]) -> Optional[FusedSample]:
        chosen_features = record.get("chosen_features")
        rejected_features = record.get("rejected_features")
        if not isinstance(chosen_features, list) or not isinstance(rejected_features, list):
            return None
        if len(chosen_features) != ENGINE_FEATURE_WIDTH:
            return None
        if len(rejected_features) != ENGINE_FEATURE_WIDTH:
            return None
        try:
            board = chess.Board(str(record["fen"]))
            chosen = board.parse_san(str(record["chosen"]))
            rejected = board.parse_san(str(record["rejected"]))
        except (KeyError, ValueError):
            return None
        if chosen == rejected:
            return None
        return FusedSample(
            planes=encode_board(board, int(record.get("opponent_rating", 1500))),
            chosen=move_to_index(chosen),
            rejected=move_to_index(rejected),
            chosen_features=torch.tensor(chosen_features, dtype=torch.float32),
            rejected_features=torch.tensor(rejected_features, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> FusedSample:
        return self.samples[index]


def collate_fused(batch: Sequence[FusedSample]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.stack([sample.planes for sample in batch]),
        torch.tensor([sample.chosen for sample in batch], dtype=torch.long),
        torch.tensor([sample.rejected for sample in batch], dtype=torch.long),
        torch.stack([sample.chosen_features for sample in batch]),
        torch.stack([sample.rejected_features for sample in batch]),
    )


def _fused_epoch(
    prior: TrapPolicyNet,
    scorer: TrapScorer,
    loader: DataLoader[FusedSample],
    config: TrainConfig,
    device: torch.device,
    optimiser: Optional[torch.optim.Optimizer],
) -> Tuple[float, float]:
    """One pass. Returns ``(loss, preference accuracy)`` on the fused score.

    There is no anchor term here and none is needed. The anchor existed because
    a bare pairwise loss leaves 4,094 of 4,096 logits without a gradient, so the
    softmax was not a distribution. Under late fusion the distribution comes
    from the frozen prior and never moves; the scorer only re-ranks candidates.
    """
    training = optimiser is not None
    scorer.train(training)
    prior.eval()
    totals = [0.0, 0.0]
    seen = 0

    for planes, chosen, rejected, chosen_features, rejected_features in loader:
        planes = planes.to(device)
        chosen, rejected = chosen.to(device), rejected.to(device)
        chosen_features = chosen_features.to(device)
        rejected_features = rejected_features.to(device)

        with torch.no_grad():
            logits = prior(planes)
            chosen_prior = logits.gather(1, chosen.unsqueeze(1)).squeeze(1)
            rejected_prior = logits.gather(1, rejected.unsqueeze(1)).squeeze(1)

        with torch.set_grad_enabled(training):
            chosen_score = chosen_prior + scorer(chosen_prior, chosen_features)
            rejected_score = rejected_prior + scorer(rejected_prior, rejected_features)
            gap = chosen_score - rejected_score
            loss = -F.logsigmoid(config.beta * gap).mean()

        if training and optimiser is not None:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()

        count = planes.shape[0]
        totals[0] += float(loss.item()) * count
        totals[1] += float((gap > 0).float().mean().item()) * count
        seen += count

    if seen == 0:
        return 0.0, 0.0
    return totals[0] / seen, totals[1] / seen


def train_scorer(
    dataset_path: Path,
    checkpoint_path: Path,
    warm_start: Path,
    config: Optional[TrainConfig] = None,
    *,
    device: Optional[torch.device] = None,
    hidden: int = 96,
) -> Dict[str, Any]:
    """Train the late-fusion scorer over a frozen distilled prior."""
    settings = config if config is not None else TrainConfig()
    torch.manual_seed(settings.seed)
    random.seed(settings.seed)
    target = device if device is not None else select_device()

    records = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    dataset = AnnotatedDataset(records)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = max(1, int(len(indices) * settings.validation_fraction))
    validation_ids, training_ids = indices[:split], indices[split:]

    train_loader: DataLoader[FusedSample] = DataLoader(
        torch.utils.data.Subset(dataset, training_ids),
        batch_size=settings.batch_size, shuffle=True, collate_fn=collate_fused,
    )
    validation_loader: DataLoader[FusedSample] = DataLoader(
        torch.utils.data.Subset(dataset, validation_ids),
        batch_size=settings.batch_size, shuffle=False, collate_fn=collate_fused,
    )

    prior, channels, blocks = load_warm_start(warm_start, target)
    prior.freeze_prior()
    scorer = TrapScorer(hidden=hidden).to(target)
    optimiser = torch.optim.AdamW(
        scorer.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, settings.epochs))
    logger.info(
        "scorer: %d records (%d train / %d val) on %s, %s scorer parameters over a frozen %s prior",
        len(dataset), len(training_ids), len(validation_ids), target,
        f"{scorer.parameter_count:,}", f"{prior.parameter_count:,}",
    )

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Tensor]] = None
    best_pref = -1.0
    best_epoch = 0
    for epoch in range(1, settings.epochs + 1):
        loss, pref = _fused_epoch(prior, scorer, train_loader, settings, target, optimiser)
        schedule.step()
        val_loss, val_pref = _fused_epoch(prior, scorer, validation_loader, settings, target, None)
        entry = {
            "epoch": epoch, "loss": round(loss, 4), "pref_acc": round(pref, 4),
            "val_loss": round(val_loss, 4), "val_pref": round(val_pref, 4),
        }
        history.append(entry)
        if val_pref > best_pref:
            best_pref, best_epoch = val_pref, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}
        if epoch % max(1, settings.epochs // 10) == 0 or epoch == 1:
            logger.info("scorer: %s", entry)

    if best_state is not None and best_epoch != settings.epochs:
        logger.info("scorer: restoring epoch %d (val_pref=%.4f)", best_epoch, best_pref)
        scorer.load_state_dict(best_state)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "state_dict": scorer.state_dict(),
            "hidden": hidden,
            "prior": str(warm_start),
            "channels": channels,
            "blocks": blocks,
            "stage": "fused",
            "records": len(dataset),
            "config": asdict(settings),
        },
        checkpoint_path,
    )
    logger.info("scorer: wrote %s", checkpoint_path)
    return {"history": history, "best_epoch": best_epoch, "best_val_pref": best_pref}

if __name__ == "__main__":
    raise SystemExit(main())
