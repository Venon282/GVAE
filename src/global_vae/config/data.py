"""Structured config schema for the data domain (spec §9, §10 "Config management").

This module is deliberately a **contract, not an implementation**: per this project's
explicit scope decision, dataset loading, transforms, and train/val/test splitting stay
entirely the caller's own responsibility (`data/NOTE.md`, `configs/data/NOTE.md`), since
the concrete format is dataset-specific (spec §6: "preprocessing is dataset-specific...
and must live outside the architecture"). `DataConfig` describes the *shape* of that
information (paths, batch size, split, named transforms) so it can be captured,
validated, and snapshotted alongside the rest of an experiment's config (spec §10:
"config snapshotted with every run"), but this module never reads a file, resamples a
series, or builds a `torch.utils.data.Dataset` itself.

The one piece of indirection that makes this config actually *usable* end to end
without the framework owning any data-loading code is `loader_factory`: a
`"module.path:function_name"` reference (the same convention `scripts/evaluate.py`
already uses for `--model-factory`/`--dataloader-factory`, now shared via
`global_vae.utils.imports.importCallable`) to a function the caller writes themselves,
taking this exact `DataConfig` and returning a `DataloaderBundle`. `buildDataloadersFromConfig`
is the one function here with any behavior: it does nothing but resolve and call that
reference.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from omegaconf import MISSING

from global_vae.utils.imports import importCallable


@dataclass
class DataConfig:
    """Data configuration contract (spec §6: preprocessing/pairing/splitting stays
    outside this framework; this only describes it).

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
        transforms: Named preprocessing steps (e.g. `["log_intensity",
            "standardize"]`) the caller's own data pipeline applies,
            in order. Names only: this framework has no built-in
            transform implementations to select between (spec §6, and
            `visualization.reconstruction_plot`'s own
            `inverse_transform` docstring makes the same point from
            the plotting side).
        sequence_length: Target fixed length after any resampling the
            caller's own pipeline performs, if signals are resampled
            to a common grid before being batched. `None` if not
            applicable (e.g. images, or signals already fixed-length).
            Purely informational here: nothing in this config module
            resamples anything; a decoder's own `output_length`
            (`configs/model/*.yaml`) must still be set to match this
            value by hand, since `config/model.py` deliberately knows
            nothing about the data domain (see that module's
            docstring).
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
    transforms: list[str] = field(default_factory=list)
    sequence_length: int | None = None
    seed: int = 0


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

    The only function in this module with any behavior, and even this
    is pure indirection: it never loads data itself, it only finds and
    calls the caller-supplied factory function that does.

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
