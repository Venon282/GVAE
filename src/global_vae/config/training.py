"""Structured config schema for the training domain (spec §9, §10 "Config management").

Also holds the builder functions that turn a validated `TrainingConfig` into a real
`Trainer` with real optimizer, beta schedules, and callbacks wired in.

Every registry-backed field here (`beta_schedules[...].strategy`, `loggers[...].name`,
the keys of `callbacks`) is resolved through this project's existing registries
(`training.beta_schedules.registry`, `training.loggers.registry`,
`training.callbacks.registry`), never hardcoded, so adding a new schedule, logger, or
callback strategy elsewhere in the codebase makes it usable from config automatically,
with zero changes needed here (spec §10, §12).

`callbacks` (ADR 0022) replaces what used to be a single, hardcoded `checkpoint:
CheckpointConfig` field covering both checkpoint callbacks at once: a plain
`dict[str, dict[str, Any] | None]`, registry name -> constructor kwargs, resolved
through the `training.callbacks` registry every non-logger `TrainerCallback` (built-in
or a caller's own) self-registers into. This is what makes a callback genuinely
opt-in, in any combination, by name, the same way every other pluggable strategy in
this codebase already works, rather than a fixed, hardcoded field this module has to
grow every time a new kind of callback is added. `loggers` (predating this field,
ADR 0008, unchanged by ADR 0022) stays its own, separate `list[LoggerEntryConfig]`:
a `Logger` is a journalling service, not a member of this registry, even though
`AbstractExperimentLogger` happens to subclass `TrainerCallback` for the hook plumbing;
see `docs/adr/0022-callback-registry.md` for why the two stay apart.

`optimizer.name` and `reconstruction_loss` are the two exceptions: they select a plain
`torch.optim.Optimizer` subclass or a `torch.nn.functional` loss function, neither of
which is one of this project's own pluggable strategies (spec §10's registry pattern is
for *this framework's* extension points; wrapping every PyTorch built-in in a registry
of its own would be pure ceremony). A small name -> class/function lookup covers the
common cases; nothing stops a caller from constructing a `Trainer` directly (bypassing
config entirely) for an optimizer or loss this lookup does not cover.
"""

import inspect
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
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.registry import getCallbackClass
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
class TrainingConfig:
    """Top-level training configuration, matching `Trainer`'s constructor almost field-for-field.

    Spec §9, §10.

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
            Resolved through the separate `training.loggers` registry
            (`getLoggerClass`), not through `callbacks` below: a
            `Logger` is a journalling service, not a callback
            selection (`docs/adr/0022-callback-registry.md`).
        callbacks: `training.callbacks` registry name -> constructor
            kwargs (`None`, or an empty mapping, both mean "every
            default"; ADR 0022). Empty (default) means no such
            callback at all: nothing is enabled unless explicitly
            named here. Any combination of the built-ins
            (`"checkpoint"`, `"best_checkpoint"`, `"early_stopping"`,
            `"reduce_lr_on_plateau"`) or a caller's own
            `@registerCallback(...)`-decorated class may be listed, in
            any order; callbacks are instantiated, and `Trainer` calls
            them, in this mapping's own iteration order (a plain
            `dict`, so the order given in YAML), after every logger
            from `loggers` above. Example::

                loggers:
                  - name: csv
                    kwargs:
                      path: ${output_dir}/metrics.csv
                callbacks:
                  checkpoint:
                    directory: ${output_dir}/checkpoints
                    every_n_epochs: 10
                  early_stopping:
                    monitor: val/loss/total
                    patience: 5

            Listing the same registry name twice is not expressible
            this way (a `dict` key is unique); construct a `Trainer`
            directly with an explicit `callbacks=[...]` list for that
            rare case (e.g. two `CheckpointCallback`s writing to
            different directories).

            Note on Hydra CLI overrides specifically (not a limitation
            of this field itself, nor of a config *file* adding a new
            entry, which always just works): Hydra's structured-config
            "struct mode" only lets a `dotlist` override change a key
            that already exists somewhere in the composed config.
            Adding a callback entirely absent from every composed YAML
            file (e.g. `training.callbacks.early_stopping...` when no
            `early_stopping:` key is in `configs/training/*.yaml`)
            needs Hydra's own `+` prefix from the command line:
            `+training.callbacks.early_stopping.monitor=val/loss/total`.
            Overriding a field of an *already-present* entry (e.g.
            `training.callbacks.best_checkpoint.monitor=...` when
            `best_checkpoint:` is already in `configs/training/default.yaml`)
            needs no `+`, exactly like overriding any other already-present
            config field.
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
    callbacks: dict[str, dict[str, Any] | None] = field(default_factory=dict)


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
    """Instantiate every logger and callback described by `config` (spec §10, ADR 0022).

    Two independent mechanisms are combined, in this order:

    1. `config.loggers`: each `LoggerEntryConfig` resolved through the separate
       `training.loggers` registry (`getLoggerClass`), exactly as before ADR 0022
       (`docs/adr/0008-experiment-loggers.md`). Unaffected by `config.callbacks`.
    2. `config.callbacks`: each entry (`registry name -> constructor kwargs`, `None`
       or an empty mapping both meaning "every default") resolved through the
       unified `training.callbacks` registry (`training/callbacks/registry.py`)
       exactly like every other pluggable strategy in this codebase: adding a new
       callback (built-in, or a caller's own via `@registerCallback(...)`) makes it
       selectable from config with no change needed here. Callbacks are
       instantiated in `config.callbacks`' own iteration order (a plain `dict`, so
       YAML key order), and `Trainer` then calls them in that same order, after
       every logger from step 1.

    A callback from `config.callbacks` whose constructor accepts a `config`
    parameter, and whose own kwargs did not already supply one, receives
    `config_snapshot` automatically. This is checked via `inspect.signature`
    against the constructor itself, not by special-casing
    `CheckpointCallback`/`BestCheckpointCallback` (or any other specific class) by
    name: any callback, built-in or a caller's own, opts into receiving the run's
    own config snapshot (spec §10: "config snapshotted with every run") simply by
    declaring a `config` parameter. Loggers (step 1) never receive
    `config_snapshot`: a logger's constructor has no such parameter, matching its
    behavior before ADR 0022.

    Args:
        config: A `TrainingConfig`.
        config_snapshot: Forwarded to any `config.callbacks` entry whose
            constructor accepts a `config` parameter (see above), typically the
            full `ExperimentConfig` this training run was built from.

    Returns:
        One logger instance per entry of `config.loggers` (in order), followed by
        one callback instance per entry of `config.callbacks` (in order). Empty
        list if both are empty.

    Raises:
        KeyError: If any `LoggerEntryConfig.name` is not a registered
            `training.loggers` strategy, or any key of `config.callbacks` is not a
            registered `training.callbacks` strategy.
        TypeError: If a callback's kwargs do not match its constructor
            (propagated unchanged from that callback's own `__init__`).
    """
    callbacks: list[TrainerCallback] = [
        getLoggerClass(entry.name)(**entry.kwargs) for entry in config.loggers
    ]

    for name, raw_kwargs in config.callbacks.items():
        kwargs = dict(raw_kwargs) if raw_kwargs else {}
        callback_cls = getCallbackClass(name)
        if config_snapshot is not None and "config" not in kwargs:
            constructor_params = inspect.signature(callback_cls.__init__).parameters
            if "config" in constructor_params:
                kwargs["config"] = config_snapshot
        callbacks.append(callback_cls(**kwargs))

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
            any `beta_schedules[...].strategy`, any `loggers[...].name`, or
            any key of `config.callbacks` is not a supported/registered name.
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
