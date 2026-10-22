#!/usr/bin/env python3
r"""Reference `loader_factory` for paired (signal, image) data (roadmap P3-2, spec §6.1 and §6.2).

Spec §6.2 makes dataset loading, pairing and splitting the caller's job, for good: no
`datamodule.py` exists or is planned, and `src/global_vae` never reads a file. This module is
what that looks like in practice: one `loader_factory`, `buildPairedDataloaders`, that takes the
`DataConfig` of `configs/data/signal_image.yaml` and returns a `DataloaderBundle` whose batches
carry the two keys the signal plus image experiments read, `signal` of shape `(batch, 256)` and
`image` of shape `(batch, 64, 64)`. Copy it into your own package and change the three
adaptation points below; `scripts/train.py` then trains on your pairs unchanged.

**The pairing convention (roadmap D-5).** Taken at the default the roadmap proposes: samples are
paired by **sample ID**, in the caller's `loader_factory`, and a sample that has only one of
the two modalities is **left out** (single-modality batches are not produced). The convention
is the simplest one that matches the spec's "filename/sample-ID" hint:

    <train_path>/
        signal/<sample_id>.npy    a 1D array, the signal of that sample
        image/<sample_id>.npy     a 2D array, the image of that sample

Two files pair when their names agree up to the extension. The match is by ID, never by
position: a sample missing from one folder does not shift the pairs that follow it. The owner
has not confirmed D-5 yet, which is why the convention sits behind three small functions
instead of being spread through the code.

**Adaptation points** (the only places that know your files):

- `sampleIdFromPath(path, modality)`: how a file name becomes a sample ID. Change it when the
  two modalities name a sample differently (`sig_0007.npy` and `img_0007.npy`), or when the ID
  comes from a manifest rather than from the name.
- `loadSignal(path)` and `loadImage(path)`: how one file becomes a tensor, `(length,)` and
  `(height, width)`. Swap `np.load` for `h5py`, `tifffile`, `fabio` or whatever holds your data.

Everything else is generic: pairing, the seeded split by sample ID, the per-modality transform
pipelines of `DataConfig.transforms` (`buildTransformPipeline`, spec §6.2), the shape check
against `DataConfig.sequence_length`, and the `DataLoader` settings (`batch_size`,
`num_workers`, `shuffle_train`).

**Splits.** `val_path` and `test_path`, when set, are folders of the same layout, paired the
same way. When they are not set, `val_split` and `test_split` hold out that fraction of the
pairs of `train_path`, chosen by sample ID with `DataConfig.seed`, so a split never depends on
the global seed or on the order of the files. No sample is in two splits. A split left unset
(`None`) gives no loader for it. Only the training loader shuffles.

**Transforms.** A modality with no entry in `DataConfig.transforms` is used as stored. The
signal must have `DataConfig.sequence_length["signal"]` points once its pipeline has run, and
this is checked on the first sample before training starts: if your files are longer or shorter,
add a `resample` step to `data.transforms.signal` (spec §6.2) rather than changing the model.
Images are not resized here: their size is what the decoder's `output_shape` says, and a
mismatch is reported by the decoder.

**Run it.** With no argument this script writes a small synthetic paired dataset in the layout
above (including samples that have only one modality, to show that they are left out), builds
the loaders through `buildPairedDataloaders` from the composed `experiment/signal_image_vae`
config and prints what came out:

    python examples/04_paired_loader.py

With `--train` it then runs `scripts/train.py` on that folder, exactly as you would on your own
data. This is the command it runs, and the one to adapt to your files (the repository root must
be on `PYTHONPATH` so that `examples.04_paired_loader` imports; with a copy in your own package
use `my_project.data:buildPairedDataloaders` instead):

    PYTHONPATH=. python scripts/train.py --config-name experiment/signal_image_vae \
        data.loader_factory=examples.04_paired_loader:buildPairedDataloaders \
        data.train_path=/path/to/dataset

The other signal plus image experiments (`signal_image_independent_latents`,
`signal_image_shared_private`) read the same batches, and so do the image-only ones (`image_vae`,
`image_resnet_vae`, which read the `image` key and ignore `signal`). `signal_image_to_image` does
not: its inputs are a degraded and a clean copy of one image, which these pairs do not have.

Everything is written to `examples/outputs/04_paired_loader/` (git-ignored). On CPU the whole
run, training included, takes well under a minute.
"""

import argparse
import logging
import os
import random
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from global_vae.config.data import DataConfig, DataloaderBundle, buildTransformPipeline
from global_vae.config.experiment import loadExperimentConfig
from global_vae.data.transforms.compose import ComposeTransform

logger = logging.getLogger("global_vae.examples.paired_loader")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "04_paired_loader"

# Batch keys: what `configs/experiment/signal_image_*.yaml` read.
SIGNAL = "signal"
IMAGE = "image"

# The on-disk convention (roadmap D-5): one folder per modality, one `.npy` file per sample.
FILE_SUFFIX = ".npy"
SIGNAL_DIRECTORY = "signal"
IMAGE_DIRECTORY = "image"

# The shapes of `configs/data/signal_image.yaml` and `configs/model/default.yaml`.
SIGNAL_LENGTH = 256
IMAGE_SIZE = 64

LOADER_FACTORY = "examples.04_paired_loader:buildPairedDataloaders"
EXPERIMENT = "experiment/signal_image_vae"

# How many sample IDs a warning names before it says "and N more".
_MAX_LISTED_IDS = 5


# --- Adaptation points: the only code that knows your files --------------------------------


def sampleIdFromPath(path: Path, modality: str) -> str:
    """Return the sample ID of a modality's file: its name without the extension.

    Change this when the two modalities do not name a sample the same way (a prefix per
    modality, a zero-padded counter on one side only) or when IDs come from a manifest.

    Args:
        path: A file found in a modality folder.
        modality: `SIGNAL` or `IMAGE`, for conventions that differ per modality.

    Returns:
        The sample ID. Two files, one per modality, belong to the same pair when this returns
        the same string for both.
    """
    del modality  # The default convention is the same for both.
    return path.stem


def loadSignal(path: Path) -> torch.Tensor:
    """Read the signal of one sample.

    Swap `np.load` for the reader of your own format; the contract is the return value.

    Args:
        path: A file of the signal folder.

    Returns:
        A float32 tensor of shape `(length,)`.

    Raises:
        ValueError: If the file does not hold a 1D array.
    """
    array = np.load(path)
    if array.ndim != 1:
        raise ValueError(f"{path}: a signal must be a 1D array, got shape {array.shape}.")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def loadImage(path: Path) -> torch.Tensor:
    """Read the image of one sample.

    Swap `np.load` for the reader of your own format; the contract is the return value.

    Args:
        path: A file of the image folder.

    Returns:
        A float32 tensor of shape `(height, width)`, one grayscale channel.

    Raises:
        ValueError: If the file does not hold a 2D array.
    """
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"{path}: an image must be a 2D array, got shape {array.shape}.")
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


# --- Pairing -------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairingResult:
    """The outcome of pairing the files of two modalities by sample ID.

    Attributes:
        signal_files: Sample ID -> signal file, for the paired samples only, in sorted ID order.
        image_files: Sample ID -> image file, same keys and order as `signal_files`.
        signal_only: IDs found as a signal but not as an image, sorted. Left out.
        image_only: IDs found as an image but not as a signal, sorted. Left out.
    """

    signal_files: dict[str, Path]
    image_files: dict[str, Path]
    signal_only: list[str]
    image_only: list[str]

    @property
    def sampleIds(self) -> list[str]:
        """The IDs of the paired samples, sorted."""
        return list(self.signal_files)


def indexModality(directory: Path, modality: str) -> dict[str, Path]:
    """Map every sample ID of one modality folder to its file.

    Only files with the `FILE_SUFFIX` extension count, and subfolders are not searched, so a
    stray `.DS_Store` or a `notes.txt` is ignored.

    Args:
        directory: The folder of one modality.
        modality: `SIGNAL` or `IMAGE`, forwarded to `sampleIdFromPath`.

    Returns:
        Sample ID -> file, in sorted ID order.

    Raises:
        FileNotFoundError: If `directory` does not exist.
        ValueError: If the folder has no data file, or two files map to one sample ID.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"The {modality} folder {directory} does not exist.")
    files: dict[str, Path] = {}
    for path in sorted(directory.glob(f"*{FILE_SUFFIX}")):
        sample_id = sampleIdFromPath(path, modality)
        if sample_id in files:
            raise ValueError(
                f"Sample ID '{sample_id}' is used by {files[sample_id]} and by {path}: "
                "a sample ID must name one file per modality."
            )
        files[sample_id] = path
    if not files:
        raise ValueError(f"The {modality} folder {directory} holds no '*{FILE_SUFFIX}' file.")
    return dict(sorted(files.items()))


def pairBySampleId(
    signal_files: Mapping[str, Path], image_files: Mapping[str, Path]
) -> PairingResult:
    """Pair the samples that have both a signal and an image, by sample ID.

    Matching is by ID, never by position: a sample missing from one side does not shift the
    pairs after it. A sample with only one modality is left out of `signal_files` and
    `image_files` and reported in `signal_only` or `image_only` (roadmap D-5).

    Args:
        signal_files: Sample ID -> signal file, as from `indexModality`.
        image_files: Sample ID -> image file, as from `indexModality`.

    Returns:
        The `PairingResult`.
    """
    paired_ids = sorted(signal_files.keys() & image_files.keys())
    return PairingResult(
        signal_files={sample_id: signal_files[sample_id] for sample_id in paired_ids},
        image_files={sample_id: image_files[sample_id] for sample_id in paired_ids},
        signal_only=sorted(signal_files.keys() - image_files.keys()),
        image_only=sorted(image_files.keys() - signal_files.keys()),
    )


def _describeIds(sample_ids: Sequence[str]) -> str:
    """Name the first few sample IDs of a list, and how many more there are."""
    listed = ", ".join(sample_ids[:_MAX_LISTED_IDS])
    extra = len(sample_ids) - _MAX_LISTED_IDS
    return f"{listed} and {extra} more" if extra > 0 else listed


def pairDirectory(root: Path) -> PairingResult:
    """Pair the signals and the images found under one dataset folder.

    Logs what was left out, so that a naming mistake (two folders that share no ID at all) is
    seen at once instead of as an empty training set later.

    Args:
        root: A folder with a `signal/` and an `image/` subfolder (see the module docstring).

    Returns:
        The `PairingResult`.

    Raises:
        FileNotFoundError: If either subfolder is missing.
        ValueError: If a subfolder has no data file, or no sample has both modalities.
    """
    pairing = pairBySampleId(
        indexModality(root / SIGNAL_DIRECTORY, SIGNAL),
        indexModality(root / IMAGE_DIRECTORY, IMAGE),
    )
    if not pairing.sampleIds:
        raise ValueError(
            f"No sample of {root} has both a signal and an image: the sample IDs of the two "
            "folders do not match (check sampleIdFromPath). Signal IDs start with "
            f"{_describeIds(pairing.signal_only)}; image IDs start with "
            f"{_describeIds(pairing.image_only)}."
        )
    logger.info("%s: %d paired samples.", root, len(pairing.sampleIds))
    if pairing.signal_only:
        logger.warning(
            "%s: %d sample(s) have a signal but no image and are left out: %s.",
            root,
            len(pairing.signal_only),
            _describeIds(pairing.signal_only),
        )
    if pairing.image_only:
        logger.warning(
            "%s: %d sample(s) have an image but no signal and are left out: %s.",
            root,
            len(pairing.image_only),
            _describeIds(pairing.image_only),
        )
    return pairing


# --- Splitting -----------------------------------------------------------------------------


def _splitSize(fraction: float | None, total: int, name: str) -> int:
    """Return how many of `total` samples a split of the given fraction holds.

    Args:
        fraction: The configured fraction, or `None` for no such split.
        total: The number of paired samples.
        name: `"val_split"` or `"test_split"`, for the error message.

    Returns:
        `0` when `fraction` is `None` or `0`; otherwise `round(fraction * total)`, at least 1,
        so a small dataset still gets a non-empty split.

    Raises:
        ValueError: If `fraction` is not in `[0, 1)`.
    """
    if fraction is None or fraction == 0:
        return 0
    if not 0 < fraction < 1:
        raise ValueError(f"{name} must be in [0, 1), got {fraction}.")
    return max(1, round(fraction * total))


def splitSampleIds(
    sample_ids: Sequence[str],
    val_fraction: float | None,
    test_fraction: float | None,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split sample IDs into train, validation and test, reproducibly.

    The IDs are sorted first and then shuffled with `seed`, so the result depends on the set of
    IDs and on the seed, and on nothing else (not the order the files were listed in, not the
    global seed). The three lists are disjoint, and each is returned in sorted ID order.

    Args:
        sample_ids: The paired sample IDs.
        val_fraction: Fraction held out for validation, or `None` for no validation split.
        test_fraction: Fraction held out for the test split, or `None` for none.
        seed: The split seed (`DataConfig.seed`).

    Returns:
        `(train_ids, val_ids, test_ids)`. `val_ids` and `test_ids` are empty when their
        fraction is `None` or `0`.

    Raises:
        ValueError: If a fraction is not in `[0, 1)`, or the held-out splits leave no training
            sample.
    """
    ordered = sorted(sample_ids)
    num_val = _splitSize(val_fraction, len(ordered), "val_split")
    num_test = _splitSize(test_fraction, len(ordered), "test_split")
    if num_val + num_test >= len(ordered):
        raise ValueError(
            f"{len(ordered)} paired sample(s) cannot give {num_val} validation and {num_test} "
            "test sample(s) and keep at least one for training: lower val_split and test_split "
            "or add data."
        )
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    val_ids = sorted(shuffled[:num_val])
    test_ids = sorted(shuffled[num_val : num_val + num_test])
    train_ids = sorted(shuffled[num_val + num_test :])
    return train_ids, val_ids, test_ids


# --- Dataset and loaders -------------------------------------------------------------------


class PairedSignalImageDataset(Dataset[dict[str, torch.Tensor]]):
    """Map-style dataset of (signal, image) pairs read from files, one sample per ID.

    Files are read when a sample is asked for, so memory does not grow with the dataset and
    `DataLoader` workers can read in parallel. The per-modality transform pipelines run on the
    single, unbatched sample.

    Attributes:
        sample_ids: The sample IDs of this dataset, in the order of its indices.
        signal_files: Sample ID -> signal file.
        image_files: Sample ID -> image file.
        pipelines: Modality name -> the composed transforms applied to that modality.
    """

    def __init__(
        self,
        sample_ids: Sequence[str],
        signal_files: Mapping[str, Path],
        image_files: Mapping[str, Path],
        pipelines: Mapping[str, ComposeTransform],
    ) -> None:
        """Store what is needed to read the samples.

        Args:
            sample_ids: The IDs this dataset serves; every one must key both file mappings.
            signal_files: Sample ID -> signal file.
            image_files: Sample ID -> image file.
            pipelines: Modality name -> transforms. A modality without an entry is used as
                stored.
        """
        self.sample_ids = list(sample_ids)
        self.signal_files = dict(signal_files)
        self.image_files = dict(image_files)
        self.pipelines = dict(pipelines)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Read one sample and run each modality through its pipeline.

        Args:
            index: Position in `sample_ids`.

        Returns:
            `{"signal": (length,), "image": (height, width)}`.
        """
        sample_id = self.sample_ids[index]
        signal = loadSignal(self.signal_files[sample_id])
        image = loadImage(self.image_files[sample_id])
        identity = ComposeTransform([])
        return {
            SIGNAL: self.pipelines.get(SIGNAL, identity).apply(signal),
            IMAGE: self.pipelines.get(IMAGE, identity).apply(image),
        }


def _makeDataset(
    pairing: PairingResult, sample_ids: Sequence[str], pipelines: Mapping[str, ComposeTransform]
) -> PairedSignalImageDataset:
    """Build the dataset of some of the paired samples of one folder."""
    return PairedSignalImageDataset(
        sample_ids, pairing.signal_files, pairing.image_files, pipelines
    )


def _checkSignalLength(dataset: PairedSignalImageDataset, expected: int | None) -> None:
    """Fail early if the signals do not have the length the config promises.

    Args:
        dataset: The dataset to check, on its first sample.
        expected: `DataConfig.sequence_length["signal"]`, or `None` when it is not set.

    Raises:
        ValueError: If the first signal, after its pipeline, does not have `expected` points.
    """
    if expected is None or len(dataset) == 0:
        return
    length = dataset[0][SIGNAL].shape[-1]
    if length != expected:
        raise ValueError(
            f"The signal of sample '{dataset.sample_ids[0]}' has {length} point(s) after "
            f"data.transforms, but data.sequence_length.signal is {expected}. Add a resample "
            "step to data.transforms.signal (spec §6.2), or fix sequence_length and the "
            "decoder's output_length together."
        )


def _makeLoader(
    dataset: PairedSignalImageDataset, config: DataConfig, shuffle: bool
) -> DataLoader[dict[str, torch.Tensor]]:
    """Wrap a dataset in a `DataLoader` with the settings of `config`.

    Args:
        dataset: The dataset to serve.
        config: Supplies `batch_size`, `num_workers` and `seed`.
        shuffle: Whether to shuffle at every epoch. The shuffling is seeded, so a run is
            reproducible, and the order still differs from one epoch to the next.

    Returns:
        The loader, whose batches are `{"signal": (batch, length), "image": (batch, h, w)}`.
    """
    generator = torch.Generator().manual_seed(config.seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        generator=generator,
    )


def buildPairedDataloaders(config: DataConfig) -> DataloaderBundle:
    """Build the train, validation and test loaders of a paired signal plus image dataset.

    This is the `loader_factory` (`data.loader_factory=...:buildPairedDataloaders`). The module
    docstring describes the file layout, the pairing rule and the splits.

    Args:
        config: The data config. Read: `train_path`, `val_path`, `test_path`, `val_split`,
            `test_split`, `seed`, `batch_size`, `num_workers`, `shuffle_train`, `transforms`
            and `sequence_length`.

    Returns:
        A `DataloaderBundle`: `train` always, `val` and `test` when their `*_path` is set or
        their `*_split` is positive, otherwise `None`.

    Raises:
        FileNotFoundError: If a dataset folder or one of its modality subfolders is missing.
        ValueError: If a folder has no paired sample, a split fraction is invalid or leaves no
            training sample, or the signal length differs from `sequence_length["signal"]`.
    """
    pipelines = buildTransformPipeline(config)
    pairing = pairDirectory(Path(config.train_path))
    train_ids, val_ids, test_ids = splitSampleIds(
        pairing.sampleIds,
        None if config.val_path is not None else config.val_split,
        None if config.test_path is not None else config.test_split,
        config.seed,
    )

    def datasetFor(
        extra_path: str | None, carved_ids: Sequence[str]
    ) -> PairedSignalImageDataset | None:
        if extra_path is not None:
            extra_pairing = pairDirectory(Path(extra_path))
            return _makeDataset(extra_pairing, extra_pairing.sampleIds, pipelines)
        return _makeDataset(pairing, carved_ids, pipelines) if carved_ids else None

    train_dataset = _makeDataset(pairing, train_ids, pipelines)
    val_dataset = datasetFor(config.val_path, val_ids)
    test_dataset = datasetFor(config.test_path, test_ids)

    expected_length = (config.sequence_length or {}).get(SIGNAL)
    _checkSignalLength(train_dataset, expected_length)

    logger.info(
        "Splits: %d train, %d validation, %d test sample(s).",
        len(train_dataset),
        len(val_dataset) if val_dataset is not None else 0,
        len(test_dataset) if test_dataset is not None else 0,
    )
    return DataloaderBundle(
        train=_makeLoader(train_dataset, config, shuffle=config.shuffle_train),
        val=_makeLoader(val_dataset, config, shuffle=False) if val_dataset is not None else None,
        test=_makeLoader(test_dataset, config, shuffle=False) if test_dataset is not None else None,
    )


# --- Synthetic paired data: stands in for your dataset in this example ---------------------


def writeSyntheticPairedDataset(
    directory: Path,
    num_pairs: int,
    num_signal_only: int = 0,
    num_image_only: int = 0,
    seed: int = 0,
) -> list[str]:
    """Write a synthetic paired dataset in the layout `buildPairedDataloaders` reads.

    Each sample has a hidden size and centre. Its signal is a positive decaying curve whose
    decay rate depends on the size (it does not see the centre). Its image is a blob of that
    size at that centre (it sees both). The two views therefore share information, which is
    what fusion needs, and the signal is positive so that the `log` step of
    `configs/data/signal_image.yaml` is defined.

    Samples that have only one modality are written too, with IDs mixed among the paired ones
    (not all at the end), so that matching by position would pair the wrong files.

    Args:
        directory: Dataset root; `signal/` and `image/` are created under it.
        num_pairs: Samples that get both files.
        num_signal_only: Extra samples that get a signal file only.
        num_image_only: Extra samples that get an image file only.
        seed: Seed of the random generator.

    Returns:
        The IDs of the paired samples, sorted.
    """
    rng = np.random.default_rng(seed)
    total = num_pairs + num_signal_only + num_image_only
    ids = [f"s{index:05d}" for index in range(total)]
    roles = np.array(
        ["both"] * num_pairs + ["signal"] * num_signal_only + ["image"] * num_image_only
    )
    rng.shuffle(roles)

    (directory / SIGNAL_DIRECTORY).mkdir(parents=True, exist_ok=True)
    (directory / IMAGE_DIRECTORY).mkdir(parents=True, exist_ok=True)
    positions = (np.arange(IMAGE_SIZE) + 0.5) / IMAGE_SIZE
    momentum = np.linspace(0.02, 1.0, SIGNAL_LENGTH)
    paired_ids: list[str] = []
    for sample_id, role in zip(ids, roles, strict=True):
        size = rng.uniform(0.05, 0.2)
        centre_x, centre_y = rng.uniform(0.3, 0.7, size=2)
        if role in ("both", "signal"):
            decay = np.exp(-((momentum * size / 0.05) ** 2) / 3.0)
            signal = decay * rng.lognormal(0.0, 0.02, size=SIGNAL_LENGTH) + 1e-3
            np.save(
                directory / SIGNAL_DIRECTORY / f"{sample_id}{FILE_SUFFIX}",
                signal.astype(np.float32),
            )
        if role in ("both", "image"):
            squared = (positions[None, :] - centre_x) ** 2 + (positions[:, None] - centre_y) ** 2
            image = np.exp(-squared / (2.0 * size**2)) + rng.normal(0.0, 0.01, squared.shape)
            np.save(
                directory / IMAGE_DIRECTORY / f"{sample_id}{FILE_SUFFIX}", image.astype(np.float32)
            )
        if role == "both":
            paired_ids.append(sample_id)
    return sorted(paired_ids)


# --- Command line --------------------------------------------------------------------------


def parseArguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments without the program name; `None` reads `sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Write a synthetic paired dataset, load it through buildPairedDataloaders, "
        "and optionally train on it with scripts/train.py."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-pairs", type=int, default=240)
    parser.add_argument("--num-signal-only", type=int, default=10)
    parser.add_argument("--num-image-only", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train", action="store_true", help="also run scripts/train.py on the dataset"
    )
    parser.add_argument("--num-epochs", type=int, default=10, help="epochs of --train")
    return parser.parse_args(argv)


def buildTrainCommand(
    data_dir: Path, output_dir: Path, num_epochs: int, warmup_steps: int, seed: int
) -> list[str]:
    """Build the `scripts/train.py` command that trains on the dataset.

    Args:
        data_dir: The dataset root (`data.train_path`).
        output_dir: Where the run writes its checkpoints and logs (`output_dir`).
        num_epochs: `training.num_epochs`.
        warmup_steps: Length of the beta warm-up, in steps; the shipped 1000 is longer than a
            run this short.
        seed: `data.seed`, the split seed.

    Returns:
        The command, ready for `subprocess.run`.
    """
    return [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "train.py"),
        "--config-name",
        EXPERIMENT,
        f"data.loader_factory={LOADER_FACTORY}",
        f"data.train_path={data_dir}",
        f"data.seed={seed}",
        f"output_dir={output_dir / 'run'}",
        f"training.num_epochs={num_epochs}",
        f"training.beta_schedules.z_fused.kwargs.warmup_steps={warmup_steps}",
        f"hydra.run.dir={output_dir / 'hydra'}",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Write the demo dataset, load it, and optionally train on it.

    Args:
        argv: Command line arguments; `None` reads `sys.argv`.

    Returns:
        The exit status: `0`, or the status of `scripts/train.py` with `--train`.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parseArguments(argv)
    data_dir = args.output_dir / "data"
    paired_ids = writeSyntheticPairedDataset(
        data_dir, args.num_pairs, args.num_signal_only, args.num_image_only, args.seed
    )

    # The very config `scripts/train.py` composes for this experiment, not a hand-built copy.
    experiment = loadExperimentConfig(
        EXPERIMENT,
        overrides=[
            f"data.loader_factory={LOADER_FACTORY}",
            f"data.train_path={data_dir}",
            f"data.seed={args.seed}",
        ],
    )
    bundle = buildPairedDataloaders(experiment.data)
    batch = next(iter(bundle.train))
    num_train_batches = len(cast(Sized, bundle.train))
    print(
        f"Wrote {len(paired_ids)} pairs, {args.num_signal_only} signal-only and "
        f"{args.num_image_only} image-only samples to {data_dir}. First training batch: "
        f"signal {tuple(batch[SIGNAL].shape)}, image {tuple(batch[IMAGE].shape)}."
    )

    warmup_steps = max(1, num_train_batches * args.num_epochs // 2)
    command = buildTrainCommand(data_dir, args.output_dir, args.num_epochs, warmup_steps, args.seed)
    print("Train on it with (the repository root must be on PYTHONPATH):")
    print("  " + shlex.join(["python", *command[1:]]))
    if not args.train:
        return 0

    python_path = [str(_REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(python_path)}
    return subprocess.run(command, cwd=_REPO_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
