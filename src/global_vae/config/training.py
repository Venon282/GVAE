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

`optimizer.name` and `reconstruction_loss` (a name, or a decoder -> name mapping, ADR 0030)
are the two exceptions: they select a plain
`torch.optim.Optimizer` subclass or a `torch.nn.functional` loss function, neither of
which is one of this project's own pluggable strategies (spec §10's registry pattern is
for *this framework's* extension points; wrapping every PyTorch built-in in a registry
of its own would be pure ceremony). A small name -> class/function lookup covers the
common cases; nothing stops a caller from constructing a `Trainer` directly (bypassing
config entirely) for an optimizer or loss this lookup does not cover.
"""

import inspect
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from omegaconf import MISSING
from torch.optim import Optimizer

from global_vae.losses.reconstruction import DEFAULT_REDUCTION, LossFn, validateReduction
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

DEFAULT_RECONSTRUCTION_LOSS = "mse"
"""The loss of a decoder that `TrainingConfig.reconstruction_loss` does not name."""

DEFAULT_RECONSTRUCTION_WEIGHT = 1.0
"""The weight of a decoder that `TrainingConfig.reconstruction_weight` does not name."""


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
            `"tensorboard"`, `"mlflow"`.
        kwargs: Forwarded to the logger's constructor, e.g.
            `{"path": "runs/metrics.csv"}` for `"csv"`, or
            `{"tracking_uri": "sqlite:///runs/mlflow.db"}` for `"mlflow"`.
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
            (`"mse"` (default), `"l1"`, `"bce"`, `"smooth_l1"`) for every
            modality, or a mapping from a decoder name to one of them
            (roadmap P2-2, ADR 0030), for example `{signal: mse, mask:
            bce}` for a continuous target next to a binary one. A decoder
            left out of the mapping keeps `"mse"`. Forwarded to
            `Trainer`'s `reconstruction_loss_fn`. Typed `Any` because a
            structured config cannot express `str | dict[str, str]`;
            `buildTrainerFromConfig` validates it and names this field in
            the error.
        reconstruction_weight: A number (at least zero) for every
            modality, or a mapping from a decoder name to one, for example
            `{signal: 0.5, image: 2.0}`. A decoder left out of the mapping
            keeps `1.0`. Forwarded to `Trainer`'s `reconstruction_weights`.
            Typed `Any` and validated like `reconstruction_loss`.
        reconstruction_reduction: How each modality's reconstruction
            loss is scaled (roadmap P2-1, ADR 0029): `"mean"` (default,
            the per-element mean, which is what every config did before
            this field existed) or `"sum"` (the per-sample sum averaged
            over the batch, the scale of the KL term). Either one name
            for every modality or a mapping from a decoder name to a
            name, for example `{signal: sum, image: mean}`; a decoder
            left out of the mapping keeps `"mean"`. Typed `Any` because
            a structured config cannot express `str | dict[str, str]`;
            `buildTrainerFromConfig` validates it and names this field
            in the error. Forwarded to `Trainer`'s
            `reconstruction_reduction`.
        beta: Base regularization weight (spec §2.3), shared across
            every latent space that has no entry in `beta_schedules`.
        beta_schedules: Latent space name -> `BetaScheduleConfig`.
            Empty (default) means every latent space uses the plain
            `beta` constant, unannealed.
        modality_dropout_p: Forwarded to `Trainer` (spec §5).
        grad_clip_norm: Forwarded to `Trainer`.
        device: Forwarded to `Trainer`. `None` (default) auto-detects.
        log_every_n_steps: Forwarded to `Trainer`.
        mixed_precision: `None` (default, full `float32`), `"bf16"` or `"fp16"`, forwarded to
            `Trainer` (roadmap P2-3, ADR 0031). `"fp16"` adds a gradient scaler. Needs a device
            whose autocast supports the dtype; `bf16` runs on a CPU as well as on recent GPUs.
        accumulate_grad_batches: Forwarded to `Trainer`: batches whose gradients are added
            before one optimizer step (default `1`, a step per batch). A step is then an
            optimizer update, which is also what `beta_schedules`, `loggers` and the schedule
            callbacks count.
        compile_model: Forwarded to `Trainer`: run the forward pass through `torch.compile`
            (default `False`).
        compile_kwargs: Forwarded to `torch.compile` when `compile_model` is `True`, for
            example `{mode: reduce-overhead}` or `{backend: eager}`.
        resume_from: Path of a checkpoint to continue from, or `None` (default) for a fresh
            run (roadmap P2-3, ADR 0032). `scripts/train.py` calls `resumeTrainerFromConfig`,
            which restores the model, the optimizer, the step and epoch counters, the history,
            the RNG state and the callbacks' state. `num_epochs` is then the **total** number
            of epochs of the run: a run saved after epoch 40 with `num_epochs: 100` trains 60
            more. (`Trainer.fit(num_epochs)` is unchanged: it trains that many more.) Loggers
            that accept `append` (`csv`) are opened in append mode, so the file of the first
            part of the run is kept.
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
            (`"checkpoint"`, `"best_checkpoint"`, `"top_k_checkpoint"`,
            `"early_stopping"`, `"reduce_lr_on_plateau"`, `"linear_warmup_lr"`,
            `"cosine_lr"`) or a caller's own
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
    reconstruction_loss: Any = DEFAULT_RECONSTRUCTION_LOSS
    reconstruction_weight: Any = DEFAULT_RECONSTRUCTION_WEIGHT
    reconstruction_reduction: Any = DEFAULT_REDUCTION
    beta: float = 1.0
    beta_schedules: dict[str, BetaScheduleConfig] = field(default_factory=dict)
    modality_dropout_p: float = 0.0
    grad_clip_norm: float | None = None
    device: str | None = None
    log_every_n_steps: int = 50
    loggers: list[LoggerEntryConfig] = field(default_factory=list)
    callbacks: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    mixed_precision: str | None = None
    accumulate_grad_batches: int = 1
    compile_model: bool = False
    compile_kwargs: dict[str, Any] = field(default_factory=dict)
    resume_from: str | None = None


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


def _checkDecoderKeys(
    field_name: str, mapping: Mapping[Any, Any], model: GlobalVae, default_text: str
) -> None:
    """Raise if a per-decoder mapping names a decoder that `model` does not have.

    Args:
        field_name: How the setting is called in the message.
        mapping: The configured mapping, decoder name -> value.
        model: The model the trainer is built for. Its `decoders` give the valid keys.
        default_text: What a decoder left out of the mapping gets, written for the message.

    Raises:
        ValueError: If a key is not a decoder of `model`. It is an error and not a warning: the
            decoder the key was meant for would silently keep its default.
    """
    unknown = sorted(str(name) for name in set(mapping) - set(model.decoders))
    if unknown:
        raise ValueError(
            f"{field_name} names decoder(s) {unknown} that the model does not have "
            f"(decoders: {sorted(model.decoders)}). A decoder left out of the mapping keeps "
            f"{default_text}, so a mistyped name would change nothing and say nothing."
        )


def _checkModalityName(name: Any, field_name: str) -> None:
    """Raise `ValueError` if a mapping key is not a string (a decoder name always is)."""
    if not isinstance(name, str):
        raise ValueError(f"{field_name}: a modality name must be a string, got {name!r}.")


def validateReconstructionLoss(
    value: Any, field_name: str = "training.reconstruction_loss"
) -> None:
    """Check that `value` is a supported loss name or a mapping decoder name -> supported name.

    This needs no model, so it cannot tell whether a mapping key is a decoder; see
    `resolveReconstructionLoss` for that.

    Args:
        value: The value to check.
        field_name: How the value is called in the error message.

    Raises:
        KeyError: If a loss name is not one of `listSupportedReconstructionLossNames()`.
        ValueError: If `value` is neither a name nor a mapping, if a mapping key is not a
            string, or if a mapping value is not a string.
    """
    available = ", ".join(listSupportedReconstructionLossNames())

    def checkName(name: Any, label: str) -> None:
        if not isinstance(name, str):
            raise ValueError(f"{label}: expected a loss name, got {name!r}.")
        if name not in _RECONSTRUCTION_LOSS_FNS:
            raise KeyError(f"{label}: unknown loss '{name}'. Available: {available}")

    if isinstance(value, Mapping):
        for modality, name in value.items():
            _checkModalityName(modality, field_name)
            checkName(name, f"{field_name}['{modality}']")
    elif isinstance(value, str):
        checkName(value, field_name)
    else:
        raise ValueError(
            f"{field_name}: expected a loss name ({available}), or a mapping from a decoder "
            f"name to one of them, got {value!r}."
        )


def validateReconstructionWeight(
    value: Any, field_name: str = "training.reconstruction_weight"
) -> None:
    """Check that `value` is a finite number >= 0, or a mapping decoder name -> such a number.

    A negative weight would make the optimizer maximize a reconstruction error, which is never
    what a config means, so it is refused here (the `Trainer` itself does not check).

    Args:
        value: The value to check.
        field_name: How the value is called in the error message.

    Raises:
        ValueError: If `value` is not a number (a `bool` is not one), is negative or not
            finite, or is a mapping with a non-string key or such a value.
    """

    def checkNumber(number: Any, label: str) -> None:
        if isinstance(number, bool) or not isinstance(number, int | float):
            raise ValueError(f"{label}: expected a number, got {number!r}.")
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{label}: must be a finite number >= 0, got {number}.")

    if isinstance(value, Mapping):
        for modality, number in value.items():
            _checkModalityName(modality, field_name)
            checkNumber(number, f"{field_name}['{modality}']")
    else:
        checkNumber(value, field_name)


def resolveReconstructionLoss(
    value: Any, model: GlobalVae, field_name: str = "training.reconstruction_loss"
) -> LossFn | dict[str, LossFn]:
    """Turn `TrainingConfig.reconstruction_loss` into `Trainer`'s `reconstruction_loss_fn`.

    Args:
        value: The configured value: a loss name, or a mapping decoder name -> loss name.
        model: The model the trainer is built for. Its `decoders` give the valid keys.
        field_name: How the setting is called in error messages.

    Returns:
        One loss function for a name. For a mapping, a `dict` with an entry for **every**
        decoder of `model`: the configured loss, or `"mse"` for a decoder the mapping leaves
        out (`Trainer` raises `KeyError` for a decoder that has no entry).

    Raises:
        KeyError: If a loss name is not supported (`validateReconstructionLoss`).
        ValueError: If `value` has the wrong shape, or a mapping key is not a decoder of
            `model`.
    """
    validateReconstructionLoss(value, field_name)
    if isinstance(value, Mapping):
        _checkDecoderKeys(field_name, value, model, repr(DEFAULT_RECONSTRUCTION_LOSS))
        return {
            decoder: _RECONSTRUCTION_LOSS_FNS[value.get(decoder, DEFAULT_RECONSTRUCTION_LOSS)]
            for decoder in model.decoders
        }
    return _RECONSTRUCTION_LOSS_FNS[value]


def resolveReconstructionWeight(
    value: Any, model: GlobalVae, field_name: str = "training.reconstruction_weight"
) -> float | dict[str, float]:
    """Turn `TrainingConfig.reconstruction_weight` into `Trainer`'s `reconstruction_weights`.

    Args:
        value: The configured value: a number, or a mapping decoder name -> number.
        model: The model the trainer is built for. Its `decoders` give the valid keys.
        field_name: How the setting is called in error messages.

    Returns:
        The weight as a `float`, or a plain `dict` of floats (never an `omegaconf` container)
        with only the entries the mapping had: `computeTotalReconstructionLoss` gives a decoder
        that has none the weight `1.0`.

    Raises:
        ValueError: If `value` is not a finite number >= 0 or a mapping of them, or a mapping
            key is not a decoder of `model`.
    """
    validateReconstructionWeight(value, field_name)
    if isinstance(value, Mapping):
        _checkDecoderKeys(field_name, value, model, repr(DEFAULT_RECONSTRUCTION_WEIGHT))
        return {str(name): float(weight) for name, weight in value.items()}
    return float(value)


def resolveReconstructionReduction(
    value: Any, model: GlobalVae, field_name: str = "training.reconstruction_reduction"
) -> str | dict[str, str]:
    """Validate `TrainingConfig.reconstruction_reduction` against a model's decoders.

    Args:
        value: The configured value: a reduction name, or a mapping decoder name -> name.
        model: The model the trainer is built for. Its `decoders` give the valid keys.
        field_name: How the setting is called in error messages.

    Returns:
        The name, or a plain `dict` copy of the mapping (never an `omegaconf` container),
        ready to pass as `Trainer(reconstruction_reduction=...)`.

    Raises:
        ValueError: If `value` is not a supported reduction or a mapping of them (the message
            starts with `field_name`), or if the mapping has a key that is not a decoder of
            `model`.
    """
    validateReduction(value, field_name)
    if isinstance(value, Mapping):
        _checkDecoderKeys(field_name, value, model, repr(DEFAULT_REDUCTION))
        return {str(name): str(reduction) for name, reduction in value.items()}
    return str(value)


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
    behavior before ADR 0022. A logger whose constructor accepts `append` and whose kwargs
    did not give it receives `append=True` when `config.resume_from` is set, so a resumed run
    adds to its metrics file instead of replacing it (ADR 0032).

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
    callbacks: list[TrainerCallback] = []
    for entry in config.loggers:
        logger_cls = getLoggerClass(entry.name)
        logger_kwargs = dict(entry.kwargs)
        if (
            config.resume_from is not None
            and "append" not in logger_kwargs
            and "append" in inspect.signature(logger_cls.__init__).parameters
        ):
            # A resumed run continues the file of the first part instead of replacing it.
            logger_kwargs["append"] = True
        callbacks.append(logger_cls(**logger_kwargs))

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
        reconstruction loss, beta schedules, and callbacks. A key of
        `config.beta_schedules` that is not a latent space of `model` is
        logged as a warning and otherwise ignored (a schedule written for
        another model's latent space names). Call
        `.fit(dataloaders.train, num_epochs=config.num_epochs,
        val_dataloader=dataloaders.val)` to actually train (spec §6.1
        milestone 1's data pipeline stays the caller's own
        responsibility, see `global_vae/config/data.py`).

    Raises:
        KeyError: If `config.optimizer.name`, a name in
            `config.reconstruction_loss`, any `beta_schedules[...].strategy`,
            any `loggers[...].name`, or any key of `config.callbacks` is not
            a supported/registered name.
        ValueError: If `config.reconstruction_loss`,
            `config.reconstruction_weight` or `config.reconstruction_reduction`
            has the wrong shape or an unsupported value (a negative weight,
            say), or is a mapping that names a decoder `model` does not
            have. A mistyped decoder name is an error and not a warning: it
            would leave that decoder on its default (`"mse"`, `1.0`,
            `"mean"`), a silent change of the objective (ADR 0029, ADR 0030).
    """
    optimizer_cls = resolveOptimizerClass(config.optimizer.name)
    reconstruction_loss_fn = resolveReconstructionLoss(config.reconstruction_loss, model)
    reconstruction_weights = resolveReconstructionWeight(config.reconstruction_weight, model)
    reconstruction_reduction = resolveReconstructionReduction(
        config.reconstruction_reduction, model
    )
    beta_schedules = buildBetaSchedules(config)
    unmatched = sorted(set(beta_schedules) - set(model.latent_spaces))
    if unmatched:
        logger.warning(
            "training.beta_schedules names latent space(s) %s that the model does not have "
            "(latent spaces: %s). Those schedules have no effect, and the spaces they were "
            "meant for are weighted by `training.beta`.",
            unmatched,
            sorted(model.latent_spaces),
        )
    callbacks = buildCallbacksFromConfig(config, config_snapshot=config_snapshot)

    return Trainer(
        model,
        optimizer=optimizer_cls,
        optimizer_kwargs=config.optimizer.kwargs,
        device=config.device,
        reconstruction_weights=reconstruction_weights,
        reconstruction_loss_fn=reconstruction_loss_fn,
        beta=config.beta,
        beta_schedules=beta_schedules,
        modality_dropout_p=config.modality_dropout_p,
        grad_clip_norm=config.grad_clip_norm,
        callbacks=callbacks,
        log_every_n_steps=config.log_every_n_steps,
        reconstruction_reduction=reconstruction_reduction,
        mixed_precision=config.mixed_precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        compile_model=config.compile_model,
        compile_kwargs=dict(config.compile_kwargs),
    )


def resumeTrainerFromConfig(trainer: Trainer, config: TrainingConfig) -> int:
    """Continue `trainer` from `config.resume_from` and say how many epochs are left to run.

    `num_epochs` of a config is the **total** number of epochs of the run. A checkpoint saved
    after `trainer.start_epoch` epochs leaves `num_epochs - start_epoch` to train, so running
    the same command line again with `training.resume_from` set completes the run it was
    interrupted from and gives the history the uninterrupted run would have (ADR 0032). With
    no `resume_from` the trainer is untouched and every epoch is left.

    Args:
        trainer: A trainer built from `config` by `buildTrainerFromConfig`, not yet fitted.
        config: The training config of the run.

    Returns:
        The number of epochs `trainer.fit` should be asked to run: `config.num_epochs` for a
        fresh run, otherwise `num_epochs - start_epoch`, or `0` when the checkpoint already
        holds `num_epochs` epochs or more (nothing is left to train, and `fit` would refuse
        `0`, so the caller skips it).

    Raises:
        FileNotFoundError: If `config.resume_from` is set and no file is there.
        pickle.UnpicklingError: If the checkpoint is not readable in the safe mode (see
            `training.checkpoint.loadCheckpoint` and `upgradeCheckpoint`).
    """
    if config.resume_from is None:
        return config.num_epochs
    path = Path(config.resume_from)
    if not path.is_file():
        raise FileNotFoundError(f"training.resume_from: no checkpoint at '{path}'.")
    trainer.loadCheckpoint(path)
    remaining = max(config.num_epochs - trainer.start_epoch, 0)
    logger.info(
        "Resumed from '%s' after %d epoch(s) (step %d): %d of %d epoch(s) left.",
        path,
        trainer.start_epoch,
        trainer.global_step,
        remaining,
        config.num_epochs,
    )
    return remaining
