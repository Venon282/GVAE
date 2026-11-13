"""Shared pieces of the warm start tests (roadmap P6-2, ADR 0052).

Named without a `test_` prefix so that pytest never collects it, and importable as
`tests.integration._warm_start_fixtures`, which is how `scripts/train.py` finds the loader factory
in the subprocess test (the repository root must be on `PYTHONPATH`).

It makes what a warm start is tested on:

- structured data (a sine of a random frequency and phase for the signal, a Gaussian blob at a
  random place for the image), because a model trained on noise collapses and the guard of ADR
  0052 would refuse it;
- two real solo checkpoints, each written by the real recipe (`experiment/pair_solo_signal` and
  `experiment/pair_solo_image`, with their stage manifest), after a few seconds of training;
- the pair model of `configs/model/signal_image_pair.yaml`.
"""

from pathlib import Path

import torch

from global_vae.config.data import DataConfig, DataloaderBundle
from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import buildTrainerFromConfig
from global_vae.models.global_vae import GlobalVae
from global_vae.utils.seed import setGlobalSeed

SIGNAL_LENGTH = 256
IMAGE_SIZE = 64
BATCH_SIZE = 16
NUM_TRAIN_BATCHES = 4
SOLO_EPOCHS = 15
LOADER_FACTORY = "tests.integration._warm_start_fixtures:buildStructuredPairDataloaders"

SOLO_EXPERIMENTS = {
    "signal": "experiment/pair_solo_signal",
    "image": "experiment/pair_solo_image",
}

# The signal stage of the shipped recipe averages its reconstruction over 256 samples, which makes
# a model that has just started collapse (docs/how-to/choose-beta.md); the image stage already
# sums, as `pair_solo_image.yaml` says. A few epochs on a handful of batches need the sum for
# both, so that the solo encoders are active.
SOLO_OVERRIDES = {
    "signal": ["training.reconstruction_reduction=sum"],
    "image": [],
}


def makeSignals(count: int, generator: torch.Generator) -> torch.Tensor:
    """Return `count` sines of a random frequency and phase, shape `(count, 256)`."""
    time = torch.linspace(0.0, 1.0, SIGNAL_LENGTH)
    frequency = torch.rand(count, 1, generator=generator) * 8.0 + 1.0
    phase = torch.rand(count, 1, generator=generator) * 6.28
    return torch.sin(2.0 * 3.14159 * frequency * time + phase)


def makeImages(count: int, generator: torch.Generator) -> torch.Tensor:
    """Return `count` Gaussian blobs at a random place, shape `(count, 64, 64)`."""
    axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE)
    center_x = torch.rand(count, 1, 1, generator=generator) * 1.6 - 0.8
    center_y = torch.rand(count, 1, 1, generator=generator) * 1.6 - 0.8
    squared = (axis[None, None, :] - center_x) ** 2 + (axis[None, :, None] - center_y) ** 2
    return torch.exp(-squared / 0.05)


def makePairedBatches(num_batches: int, seed: int) -> list[dict[str, torch.Tensor]]:
    """Return batches of paired samples: a `signal` and an `image` per row."""
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "signal": makeSignals(BATCH_SIZE, generator),
            "image": makeImages(BATCH_SIZE, generator),
        }
        for _ in range(num_batches)
    ]


def buildStructuredPairDataloaders(config: DataConfig) -> DataloaderBundle:
    """A `loader_factory` for `scripts/train.py`: 4 train, 1 validation and 1 test batch.

    Args:
        config: The data config; `seed` is read.

    Returns:
        The bundle, with structured paired batches that never touch the disk.
    """
    return DataloaderBundle(
        train=makePairedBatches(NUM_TRAIN_BATCHES, config.seed),
        val=makePairedBatches(1, config.seed + 1),
        test=makePairedBatches(1, config.seed + 2),
    )


def loadExperiment(
    name: str, output_dir: Path, overrides: list[str] | None = None
) -> ExperimentConfig:
    """Compose an experiment with the structured loader, writing under `output_dir`."""
    return loadExperimentConfig(
        name,
        overrides=[
            f"data.loader_factory={LOADER_FACTORY}",
            "data.train_path=unused",
            f"output_dir={output_dir}",
            "training.loggers=[]",
            *(overrides or []),
        ],
    )


def trainSoloCheckpoint(modality: str, output_dir: Path) -> Path:
    """Run the solo stage of `modality` in this process and return its `best.pt`.

    Args:
        modality: `"signal"` or `"image"`.
        output_dir: Where the run writes its checkpoints.

    Returns:
        The path of `best.pt`, a checkpoint with the stage manifest of the solo recipe.
    """
    experiment = loadExperiment(SOLO_EXPERIMENTS[modality], output_dir, SOLO_OVERRIDES[modality])
    setGlobalSeed(experiment.seed)
    model = buildModelFromConfig(experiment.model)
    trainer = buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)
    trainer.fit(
        makePairedBatches(NUM_TRAIN_BATCHES, 0),
        num_epochs=SOLO_EPOCHS,
        val_dataloader=makePairedBatches(1, 1),
    )
    return output_dir / "checkpoints" / "best.pt"


def loadPairExperiment(output_dir: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Compose the pair stage: the pair model, the several-latent training group, stage `pair`."""
    return loadExperiment(
        SOLO_EXPERIMENTS["signal"],
        output_dir,
        ["model=signal_image_pair", "training=several_latents", "training.stage=pair"]
        + (overrides or []),
    )


def buildPairModel(overrides: list[str] | None = None) -> tuple[GlobalVae, ExperimentConfig]:
    """Build a freshly initialized pair model, and return it with the config that built it."""
    experiment = loadPairExperiment(Path("unused"), overrides)
    setGlobalSeed(7)
    return buildModelFromConfig(experiment.model), experiment
