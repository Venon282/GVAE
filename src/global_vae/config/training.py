"""Structured config schema for the training domain (spec §9, §10 "Config management"),
plus the builder functions that turn a validated `TrainingConfig` into a real `Trainer`
with real optimizer, beta schedules, and callbacks wired in.

Every registry-backed field here (`beta_schedules[...].strategy`, `loggers[...].name`)
is resolved through this project's existing registries
(`training.beta_schedules.registry`, `training.loggers.registry`), never hardcoded, so
adding a new schedule or logger strategy elsewhere in the codebase makes it usable from
config automatically, with zero changes needed here (spec §10, §12).

`optimizer.name` and `reconstruction_loss` are the two exceptions: they select a plain
`torch.optim.Optimizer` subclass or a `torch.nn.functional` loss function, neither of
which is one of this project's own pluggable strategies (spec §10's registry pattern is
for *this framework's* extension points; wrapping every PyTorch built-in in a registry
of its own would be pure ceremony). A small name -> class/function lookup covers the
common cases; nothing stops a caller from constructing a `Trainer` directly (bypassing
config entirely) for an optimizer or loss this lookup does not cover.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from omegaconf import MISSING
from torch.optim import Optimizer

from global_vae.losses.reconstruction import LossFn
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.beta_schedules.registry import getBetaScheduleClass
from global_vae.training.callbacks import TrainerCallback
from global_vae.training.checkpoint import BestCheckpointCallback, CheckpointCallback
from global_vae.training.loggers.registry import getLoggerClass
from global_vae.training.trainer import Trainer

logger = logging.getLogger(__name__)

_OPTIMIZER_CLASSES: dict[str, type[Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

_RECONSTRUCTION_LOSS_FNS: dict[str, LossFn] = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "bce": F.binary_cross_entropy,
    "smooth_l1": F.smooth_l1_loss,
}


@dataclass
class OptimizerConfig:
    """Optimizer choice and constructor kwargs.

    Attributes:
        name: One of `listSupportedOptimizerNames()` (`"adam"`
            (default), `"adamw"`, `"sgd"`, `"rmsprop"`).
        kwargs: Forwarded to the optimizer's constructor, e.g.
            `{"lr": 1e-3}`.
    """

    name: str = "adam"
    kwargs: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})


@dataclass
class BetaScheduleConfig:
    """One latent space's beta-weighting schedule (spec §2.3).

    Attributes:
        strategy: `training.beta_schedules` registry key, e.g.
            `"constant"`, `"linear_warmup"`, `"cyclical_annealing"`.
        kwargs: Forwarded to the schedule's constructor.
    """

    strategy: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggerEntryConfig:
    """One experiment logger to attach to the `Trainer` (spec §10 "Experiment tracking").

    Attributes:
        name: `training.loggers` registry key, e.g. `"csv"`,
            `"tensorboard"`.
        kwargs: Forwarded to the logger's constructor, e.g.
            `{"path": "runs/metrics.csv"}` for `"csv"`.
    """

    name: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Checkpointing configuration (spec §10 "Reproducibility"), covering both
    `CheckpointCallback` (periodic, for resuming) and `BestCheckpointCallback`
    (metric-driven, for model selection); see
    `docs/adr/0006-reproducibility-seed-and-checkpointing.md` and
    `docs/adr/0007-best-checkpoint-callback.md` for why these are two separate
    callbacks rather than one. Either, both, or neither can be enabled.

    Attributes:
        directory: Directory for periodic checkpoints
            (`CheckpointCallback`). `None` (default) disables periodic
            checkpointing entirely.
        every_n_epochs: Forwarded to `CheckpointCallback`.
        keep_last_n: Forwarded to `CheckpointCallback`.
        best_path: File path for the single best-so-far checkpoint
            (`BestCheckpointCallback`). `None` (default) disables it.
        best_monitor: Forwarded to `BestCheckpointCallback`.
        best_mode: Forwarded to `BestCheckpointCallback` (`"min"` or
            `"max"`).
    """

    directory: str | None = None
    every_n_epochs: int = 1
    keep_last_n: int | None = None
    best_path: str | None = None
    best_monitor: str = "val/loss/total"
    best_mode: str = "min"


@dataclass
class TrainingConfig:
    """Top-level training configuration, matching `Trainer`'s own constructor almost
    field-for-field (spec §9, §10).

    Attributes:
        num_epochs: Forwarded to `Trainer.fit`.
        optimizer: See `OptimizerConfig`.
        reconstruction_loss: One of `listSupportedReconstructionLossNames()`
            (`"mse"` (default), `"l1"`, `"bce"`, `"smooth_l1"`),
            forwarded to `Trainer`'s `reconstruction_loss_fn`, shared
            across every modality. A genuinely per-modality loss choice
            (spec: e.g. `binary_cross_entropy` for a segmentation
            target alongside `mse_loss` for a continuous one) is not
            yet expressible from config; construct a `Trainer` directly
            with a `dict[str, LossFn]` for that case.
        reconstruction_weight: Forwarded to `Trainer`'s
            `reconstruction_weights`, shared across every modality (see
            `reconstruction_loss`'s own note on per-modality config).
        beta: Base regularization weight (spec §2.3), shared across
            every latent space that has no entry in `beta_schedules`.
        beta_schedules: Latent space name -> `BetaScheduleConfig`.
            Empty (default) means every latent space uses the plain
            `beta` constant, unannealed.
        modality_dropout_p: Forwarded to `Trainer` (spec §5).
        grad_clip_norm: Forwarded to `Trainer`.
        device: Forwarded to `Trainer`. `None` (default) auto-detects.
        log_every_n_steps: Forwarded to `Trainer`.
        loggers: Experiment loggers to attach (spec §10 "Experiment
            tracking"). Empty (default) means no logger; several may
            be listed at once (`docs/adr/0008-experiment-loggers.md`).
        checkpoint: See `CheckpointConfig`.
    """

    num_epochs: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    reconstruction_loss: str = "mse"
    reconstruction_weight: float = 1.0
    beta: float = 1.0
    beta_schedules: dict[str, BetaScheduleConfig] = field(default_factory=dict)
    modality_dropout_p: float = 0.0
    grad_clip_norm: float | None = None
    device: str | None = None
    log_every_n_steps: int = 50
    loggers: list[LoggerEntryConfig] = field(default_factory=list)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def listSupportedOptimizerNames() -> list[str]:
    """Return every `OptimizerConfig.name` this module knows how to resolve.

    Returns:
        Sorted list of supported optimizer names.
    """
    return sorted(_OPTIMIZER_CLASSES)


def listSupportedReconstructionLossNames() -> list[str]:
    """Return every `TrainingConfig.reconstruction_loss` this module knows how to resolve.

    Returns:
        Sorted list of supported reconstruction loss names.
    """
    return sorted(_RECONSTRUCTION_LOSS_FNS)


def resolveOptimizerClass(name: str) -> type[Optimizer]:
    """Look up a `torch.optim.Optimizer` subclass by name.

    Args:
        name: One of `listSupportedOptimizerNames()`.

    Returns:
        The optimizer class.

    Raises:
        KeyError: If `name` is not supported.
    """
    if name not in _OPTIMIZER_CLASSES:
        available = ", ".join(listSupportedOptimizerNames())
        raise KeyError(f"Unknown optimizer '{name}'. Available: {available}")
    return _OPTIMIZER_CLASSES[name]


def resolveReconstructionLossFn(name: str) -> LossFn:
    """Look up a reconstruction loss function by name.

    Args:
        name: One of `listSupportedReconstructionLossNames()`.

    Returns:
        The loss function.

    Raises:
        KeyError: If `name` is not supported.
    """
    if name not in _RECONSTRUCTION_LOSS_FNS:
        available = ", ".join(listSupportedReconstructionLossNames())
        raise KeyError(f"Unknown reconstruction_loss '{name}'. Available: {available}")
    return _RECONSTRUCTION_LOSS_FNS[name]


def buildBetaSchedules(config: TrainingConfig) -> dict[str, AbstractBetaSchedule]:
    """Instantiate every latent space's beta schedule from `config.beta_schedules`.

    Args:
        config: A `TrainingConfig`.

    Returns:
        Latent space name -> `AbstractBetaSchedule` instance, ready to
        pass as `Trainer(beta_schedules=...)`.

    Raises:
        KeyError: If any `BetaScheduleConfig.strategy` is not a
            registered `training.beta_schedules` strategy.
    """
    return {
        latent_name: getBetaScheduleClass(schedule.strategy)(**schedule.kwargs)
        for latent_name, schedule in config.beta_schedules.items()
    }


def buildCallbacksFromConfig(
    config: TrainingConfig, config_snapshot: Any = None
) -> list[TrainerCallback]:
    """Instantiate every logger and checkpoint callback described by `config`.

    Args:
        config: A `TrainingConfig`.
        config_snapshot: Forwarded to `CheckpointCallback`/
            `BestCheckpointCallback`'s own `config` parameter (spec
            §10: "config snapshotted with every run"), typically the
            full `ExperimentConfig` this training run was built from.

    Returns:
        `config.loggers` instances (in order), followed by
        `CheckpointCallback` if `config.checkpoint.directory` is set,
        followed by `BestCheckpointCallback` if
        `config.checkpoint.best_path` is set. Empty list if none of
        the above are configured.

    Raises:
        KeyError: If any `LoggerEntryConfig.name` is not a registered
            `training.loggers` strategy.
    """
    callbacks: list[TrainerCallback] = [
        getLoggerClass(entry.name)(**entry.kwargs) for entry in config.loggers
    ]

    checkpoint = config.checkpoint
    if checkpoint.directory is not None:
        callbacks.append(
            CheckpointCallback(
                checkpoint.directory,
                every_n_epochs=checkpoint.every_n_epochs,
                config=config_snapshot,
                keep_last_n=checkpoint.keep_last_n,
            )
        )
    if checkpoint.best_path is not None:
        callbacks.append(
            BestCheckpointCallback(
                checkpoint.best_path,
                monitor=checkpoint.best_monitor,
                mode=checkpoint.best_mode,
                config=config_snapshot,
            )
        )
    return callbacks


def buildTrainerFromConfig(
    model: GlobalVae, config: TrainingConfig, config_snapshot: Any = None
) -> Trainer:
    """Build a real `Trainer` from a validated `TrainingConfig`.

    Args:
        model: The model to train, typically from
            `global_vae.config.model.buildModelFromConfig`.
        config: A `TrainingConfig`, typically produced by
            `global_vae.config.experiment.loadExperimentConfig`.
        config_snapshot: Forwarded to `buildCallbacksFromConfig` for
            checkpoint snapshotting.

    Returns:
        A `Trainer` instance wired with the resolved optimizer,
        reconstruction loss, beta schedules, and callbacks. Call
        `.fit(dataloaders.train, num_epochs=config.num_epochs,
        val_dataloader=dataloaders.val)` to actually train (spec §6.1
        milestone 1's data pipeline stays the caller's own
        responsibility, see `global_vae/config/data.py`).

    Raises:
        KeyError: If `config.optimizer.name`, `config.reconstruction_loss`,
            any `beta_schedules[...].strategy`, or any `loggers[...].name`
            is not a supported/registered name.
    """
    optimizer_cls = resolveOptimizerClass(config.optimizer.name)
    reconstruction_loss_fn = resolveReconstructionLossFn(config.reconstruction_loss)
    beta_schedules = buildBetaSchedules(config)
    callbacks = buildCallbacksFromConfig(config, config_snapshot=config_snapshot)

    return Trainer(
        model,
        optimizer=optimizer_cls,
        optimizer_kwargs=config.optimizer.kwargs,
        device=config.device,
        reconstruction_weights=config.reconstruction_weight,
        reconstruction_loss_fn=reconstruction_loss_fn,
        beta=config.beta,
        beta_schedules=beta_schedules,
        modality_dropout_p=config.modality_dropout_p,
        grad_clip_norm=config.grad_clip_norm,
        callbacks=callbacks,
        log_every_n_steps=config.log_every_n_steps,
    )
