"""Structured config schema for the data domain (spec §9, §10 "Config management").

This module is deliberately a **contract, not an implementation** for dataset
loading, matching/pairing, and splitting: per this project's explicit scope
decision, those stay entirely the caller's own responsibility (`data/NOTE.md`),
since the concrete format is dataset-specific (spec §6: "preprocessing is
dataset-specific... and must live outside the architecture"). `DataConfig`
describes the *shape* of that information (paths, batch size, split) so it can
be captured, validated, and snapshotted alongside the rest of an experiment's
config (spec §10: "config snapshotted with every run"), but this module never
reads a file, resamples a series, or builds a `torch.utils.data.Dataset`
itself.

Preprocessing (spec §6.2) is the one part of this boundary that *does* get
real, reusable framework code: `DataConfig.transforms` is a **per-modality**
mapping, modality name -> ordered list of `TransformConfig` entries (name +
kwargs), each resolved through the `data.transforms` registry exactly like
every other pluggable strategy in this codebase; `buildTransformPipeline`
turns it into one composed, invertible `ComposeTransform` per modality.
Per-modality, not one shared pipeline, because spec §6 is explicit that more
than one dataset can share the 1D-signal encoder/decoder family "with only
preprocessing differing, not the architecture": a second signal dataset (a
different instrument, a different sensor) genuinely needs its own `log`/
`standardize` statistics, its own choice of steps, and its own resampled
length, independently of any other modality/dataset already configured, not
a single pipeline reused (or, worse, silently misapplied) across all of them.
Nothing here calls `buildTransformPipeline` automatically: a caller's own
`loader_factory` may call it and apply the resulting per-modality pipelines
while loading data, or ignore `config.transforms` entirely and do its own
thing. Because every `data.transforms` strategy is itself dimensionality-
agnostic (`data/transforms/base.py`), no transform *implementation* needs to
know about modalities; the per-modality *mapping* lives here, one layer up,
purely so two modalities/datasets can be configured independently. See
`docs/adr/0015-per-modality-data-transforms.md`.

The other piece of indirection that makes this config actually *usable* end
to end without the framework owning any data-loading code is `loader_factory`:
a `"module.path:function_name"` reference (the same convention
`scripts/evaluate.py` already uses for `--model-factory`/`--dataloader-factory`,
shared via `global_vae.utils.imports.importCallable`) to a function the
caller writes themselves, taking this exact `DataConfig` and returning a
`DataloaderBundle`. `buildDataloadersFromConfig` does nothing but resolve and
call that reference.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from omegaconf import MISSING

from global_vae.data.transforms.compose import ComposeTransform
from global_vae.data.transforms.registry import getTransformClass
from global_vae.utils.imports import importCallable


@dataclass
class TransformConfig:
    """One preprocessing step in one modality's entry of `DataConfig.transforms`
    (spec §6.2, §9).

    Attributes:
        name: `data.transforms` registry key, e.g. `"log"`,
            `"standardize"`, `"resample"`.
        kwargs: Forwarded to the transform's constructor (e.g.
            `{"eps": 1e-6}` for `"log"`, `{"mean": ..., "std": ...}`
            for `"standardize"`).
    """

    name: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data configuration contract (spec §6: preprocessing/pairing/splitting stays
    outside this framework; this only describes it, except for the reusable
    generic-transform part covered by `transforms` below, spec §6.2).

    Attributes:
        loader_factory: `"module.path:function_name"` pointing at a
            callable the caller writes: `(DataConfig) -> DataloaderBundle`.
            Required: without it, nothing in this framework can obtain
            actual data, by design.
        train_path: Path (file, directory, glob, or any string the
            caller's own `loader_factory` knows how to interpret) to
            the training data. Required.
        val_path: Optional explicit validation data path. If `None`
            and `val_split` is set, the caller's own `loader_factory`
            is expected to carve a validation split out of
            `train_path` instead (this config only records the
            intent; carrying it out is the factory's job).
        test_path: Optional explicit test data path, same convention
            as `val_path`/`test_split`.
        batch_size: Batch size the caller's `loader_factory` is
            expected to use.
        num_workers: `torch.utils.data.DataLoader`-style worker count,
            forwarded as a plain integer for the caller's own
            `loader_factory` to use however it sees fit (this
            framework never constructs a `DataLoader` itself).
        val_split: Fraction of `train_path` to hold out for
            validation, if `val_path` is not given directly. `None`
            means no validation split.
        test_split: As `val_split`, for a test split.
        shuffle_train: Whether the training split should be shuffled
            per epoch. Recorded here so it is part of the snapshotted
            config (spec §10) even though this framework never
            shuffles anything itself.
        transforms: Modality name -> ordered list of generic,
            invertible preprocessing steps for that modality (spec
            §6.2), each resolved through the `data.transforms` registry
            (`log`, `standardize`, `resample`, or any further strategy
            registered there). `buildTransformPipeline(config)` turns
            this into one composed `ComposeTransform` per modality;
            nothing in this framework applies any of them automatically
            (see the module docstring). A modality absent from this
            dict simply has no configured pipeline here (a caller may
            still preprocess it however it likes inside its own
            `loader_factory`; see `buildTransformPipeline`'s own
            docstring for the exact "absent" contract). Keyed per
            modality, not a flat list, so two modalities/datasets --
            e.g. two different 1D-signal-family datasets, spec §6's own
            example of "only preprocessing differing" -- can each have
            their own steps and statistics without colliding. See
            `__post_init__` for a Hydra/OmegaConf-specific detail: this
            field's leaf entries are repaired into real `TransformConfig`
            instances there, since `OmegaConf.to_object()` does not do so
            itself for a dataclass nested this deep.
        sequence_length: Modality name -> target fixed length after any
            resampling that modality's own `loader_factory` pipeline
            performs, if that modality's signals are resampled to a
            common grid before being batched (typically via a
            `"resample"` entry in that modality's own `transforms` list
            above). A modality absent from this dict, or this field
            left `None` entirely, means "not applicable" for that
            modality (e.g. images, or a modality already fixed-length).
            Purely informational here: a decoder's own `output_length`
            (`configs/model/*.yaml`) must still be set to match this
            value by hand, for that same modality, since
            `config/model.py` deliberately knows nothing about the data
            domain (see that module's docstring).
        seed: Seed for any train/val/test split randomization the
            caller's own `loader_factory` performs. Kept separate from
            `ExperimentConfig.seed` (spec §10's global seed) so a data
            split can stay fixed across runs that otherwise use
            different global seeds, if desired; defaults to `0`.
    """

    loader_factory: str = MISSING
    train_path: str = MISSING
    val_path: str | None = None
    test_path: str | None = None
    batch_size: int = 32
    num_workers: int = 0
    val_split: float | None = None
    test_split: float | None = None
    shuffle_train: bool = True
    transforms: dict[str, list[TransformConfig]] = field(default_factory=dict)
    sequence_length: dict[str, int] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """Repair `self.transforms`' leaf entries after Hydra/OmegaConf composition.

        `OmegaConf.to_object()` (used by
        `global_vae.config.experiment.loadExperimentConfig`) does not
        reconstruct a real dataclass instance for a value nested two
        containers deep: `transforms`'s modality-level `dict` and each
        modality's own `list` are both correctly materialized, but each
        individual step inside that list comes back as a plain `dict`
        (`{"name": ..., "kwargs": {...}}`) instead of a `TransformConfig`.
        This is a known limitation of `OmegaConf.to_object()` with
        dataclasses nested inside `Dict[str, List[...]]` (it does correctly
        hydrate the single-nesting cases used elsewhere in this codebase,
        e.g. `TrainingConfig.beta_schedules: dict[str, BetaScheduleConfig]`),
        not a bug in the composed YAML or in this schema's own shape.

        This hook repairs exactly that: any step that is not already a
        `TransformConfig` is reconstructed as one from its own fields. A
        caller constructing `DataConfig(transforms=...)` directly in Python
        with real `TransformConfig` instances (every direct-construction
        call site in this codebase, e.g. in tests) is unaffected: the
        `isinstance` check makes this a no-op for them.
        """
        if not self.transforms:
            return
        self.transforms = {
            modality_name: [
                step if isinstance(step, TransformConfig) else TransformConfig(**step)
                for step in steps
            ]
            for modality_name, steps in self.transforms.items()
        }


@dataclass
class DataloaderBundle:
    """What a `loader_factory` (see `DataConfig.loader_factory`) must return.

    A plain data carrier, not a framework-provided dataset/loader
    implementation: every field is whatever iterable of
    `dict[str, torch.Tensor]` batches the caller's own code produces
    (the same convention `Trainer`/`GlobalVae.forward` already use
    throughout this framework), most commonly a
    `torch.utils.data.DataLoader`, but any re-iterable object works.

    Attributes:
        train: Training batches. Required.
        val: Optional validation batches, forwarded to
            `Trainer.fit`'s own `val_dataloader`.
        test: Optional test batches, for a later `evaluation.evaluate`
            pass (spec's C8 requirement); not consumed by
            `scripts/train.py` itself.
    """

    train: Iterable[dict[str, torch.Tensor]]
    val: Iterable[dict[str, torch.Tensor]] | None = None
    test: Iterable[dict[str, torch.Tensor]] | None = None


def buildDataloadersFromConfig(config: DataConfig) -> DataloaderBundle:
    """Resolve `config.loader_factory` and call it with `config`.

    The only function in this module with any behavior around
    dataset loading itself, and even this is pure indirection: it
    never loads data itself, it only finds and calls the
    caller-supplied factory function that does. See
    `buildTransformPipeline` for the one piece of data-domain
    behavior this module does own outright (spec §6.2).

    Args:
        config: A `DataConfig`, typically produced by
            `global_vae.config.experiment.loadExperimentConfig`.

    Returns:
        Whatever `DataloaderBundle` the resolved `loader_factory`
        returns.

    Raises:
        ValueError: If `config.loader_factory` is not a valid
            `"module.path:function_name"` string (delegated to
            `importCallable`).
        ModuleNotFoundError: If the factory's module cannot be
            imported.
        AttributeError: If the factory function does not exist on that
            module.
        TypeError: If the resolved factory does not accept a single
            `DataConfig` positional argument (surfaces at call time,
            from the factory itself).
    """
    loader_factory = importCallable(config.loader_factory)
    bundle: DataloaderBundle = loader_factory(config)  # type: ignore[operator]
    return bundle


def buildTransformPipeline(config: DataConfig) -> dict[str, ComposeTransform]:
    """Resolve `config.transforms` into one composed, invertible pipeline per
    modality (spec §6.2).

    Every configured modality's step list is instantiated via the
    `data.transforms` registry (`getTransformClass`, mirroring how every
    other config-driven strategy in this codebase is resolved) and
    chained, in order, into that modality's own `ComposeTransform`.

    This function is a convenience the caller's own `loader_factory`,
    or later evaluation/visualization code, *may* call; nothing in
    this framework calls it automatically (see the module docstring:
    the data pipeline stays the caller's own responsibility). Each
    returned pipeline's `.apply`/`__call__` and, in particular,
    `.inverse` is directly usable wherever this codebase already
    accepts a plain `Callable[[Tensor], Tensor]` for undoing
    preprocessing, e.g. `visualization.reconstruction_plot`'s own
    `inverse_transform` parameter (pass `pipelines["signal"].inverse`)
    or `evaluation.visual_export.exportEvaluationFigures`'s
    `inverse_transforms` dict (the returned dict is already keyed the
    same way that parameter expects: by modality name).

    Args:
        config: A `DataConfig`.

    Returns:
        Modality name -> `ComposeTransform` chaining that modality's
        configured steps in order. A modality with an empty step list
        (`config.transforms[name] == []`) gets a `ComposeTransform`
        with zero steps, whose `.apply`/`.inverse` are both the
        identity. A modality entirely absent from `config.transforms`
        is not a key of the returned dict at all, not an implicit
        identity entry: `DataConfig` has no independent notion of
        which modalities exist in the first place (that list lives in
        `ModelConfig.modalities`, a separate config domain, spec §9),
        so this function cannot invent a key for a modality it was
        never told about. Callers wanting an explicit identity
        fallback for an unconfigured modality do so themselves, e.g.
        `pipelines.get(modality_name, ComposeTransform([]))`.

    Raises:
        KeyError: If any `TransformConfig.name` is not a registered
            `data.transforms` strategy.
    """
    return {
        modality_name: ComposeTransform(
            [getTransformClass(step.name)(**step.kwargs) for step in steps]
        )
        for modality_name, steps in config.transforms.items()
    }
