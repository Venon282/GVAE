"""Data and measurements of the end-to-end staged-training test (roadmap P6-7).

Named without a `test_` prefix so that pytest never collects it, and importable as
`tests.integration._staged_training_fixtures`, which is how `scripts/train.py` finds the loader
factory in the subprocesses of `test_staged_training.py` (the repository root must be on
`PYTHONPATH`).

The data is the linked synthetic pair of `_warm_start_fixtures.makeLinkedBatches`: the image is a
function of the signal's frequency and phase, so the two modalities of a row describe the same two
numbers and there is something for stage 3 to align. A pair of independent modalities would give
retrieval at chance whatever the stage did. It is synthetic, and what it shows is that the stages
chain and what each guard sees, not how well the recipe does on real pairs (roadmap I-15).

Three things come from here:

- the `loader_factory` of the three stages, which yields the same linked pairs to all of them (the
  solo stages ignore the other modality's key, as the recipe says);
- the held-out pairs and the training pairs the measurements are taken on, generated again here
  from the same seeds, so that the test measures on data the runs did not see;
- `measurePair`, which reads one pair model the way the evaluation of P6-6 does, on posterior
  means: the latent distance, the retrieval of each direction and its chance level, the active
  units of each space, the self and the cross reconstruction error of each modality, and the error
  of the constant "mean sample" predictor that a cross reconstruction has to beat;
- the three factories `scripts/evaluate.py --latent-pair` takes (`buildPairModelForEvaluation`,
  `buildHeldoutPairLoader`, `buildTrainPairLoader`), so that the test can evaluate the checkpoint
  of stage 3 through the script too.
"""

from dataclasses import dataclass

import torch

from global_vae.config.data import DataConfig, DataloaderBundle
from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.evaluation.latent_alignment import evaluateLatentAlignment
from global_vae.models.global_vae import GlobalVae
from tests.integration._warm_start_fixtures import makeLinkedBatches

LOADER_FACTORY = "tests.integration._staged_training_fixtures:buildLinkedPairDataloaders"
NUM_TRAIN_BATCHES = 4
NUM_HELDOUT_BATCHES = 4
HELDOUT_SEED = 99
BATCH_SIZE = 16
SPACES = ("z_signal", "z_image")
RETRIEVAL_K = 10


def buildLinkedPairDataloaders(config: DataConfig) -> DataloaderBundle:
    """A `loader_factory` for `scripts/train.py`: linked pairs, 4 train, 1 validation, 1 test batch.

    Args:
        config: The data config; `seed` is read.

    Returns:
        The bundle. Every batch carries a `signal` and an `image`, row `i` of each describing the
        same two numbers, and nothing touches the disk.
    """
    return DataloaderBundle(
        train=makeLinkedBatches(NUM_TRAIN_BATCHES, config.seed),
        val=makeLinkedBatches(1, config.seed + 1),
        test=makeLinkedBatches(1, config.seed + 2),
    )


def trainPairs(seed: int = 0) -> list[dict[str, torch.Tensor]]:
    """The training pairs the loader yields for `data.seed = seed`, for the baselines below."""
    return makeLinkedBatches(NUM_TRAIN_BATCHES, seed)


def heldoutPairs() -> list[dict[str, torch.Tensor]]:
    """Held-out pairs no stage trained or validated on: 64 samples from their own seed."""
    return makeLinkedBatches(NUM_HELDOUT_BATCHES, HELDOUT_SEED)


@dataclass(frozen=True)
class PairMeasures:
    """What one pair model is judged by, on held-out pairs and the posterior means.

    Attributes:
        distance: The mean latent distance between the two posteriors of one sample, for the
            distance stage 3 trained with.
        recall: Direction (`"z_signal->z_image"`) -> the recall at `RETRIEVAL_K`.
        chance_recall: The recall at `RETRIEVAL_K` of a random ranking.
        active_units: Space -> how many of its latent units are active on the held-out pairs.
        self_error: Modality -> the mean squared error of the modality rebuilt from its own space.
        cross_error: Modality -> the mean squared error of the modality rebuilt from the other
            space's code.
        baseline_error: Modality -> the mean squared error of the constant prediction (the mean of
            the training samples of that modality).
        num_heldout: The number of held-out pairs.
    """

    distance: float
    recall: dict[str, float]
    chance_recall: float
    active_units: dict[str, int]
    self_error: dict[str, float]
    cross_error: dict[str, float]
    baseline_error: dict[str, float]
    num_heldout: int


def _meanSquaredError(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(((prediction - target) ** 2).mean().item())


def constantPredictorErrors(
    train: list[dict[str, torch.Tensor]], heldout: list[dict[str, torch.Tensor]]
) -> dict[str, float]:
    """The error of predicting, for every held-out sample, the mean training sample.

    This is the "mean-image baseline" of the roadmap, for each modality: what a decoder that
    ignored the code and knew only the average sample would score. A cross reconstruction that does
    not beat it carries no information about the partner.

    Args:
        train: Training batches, each with a `signal` and an `image`.
        heldout: Held-out batches.

    Returns:
        Modality -> mean squared error per element.
    """
    errors: dict[str, float] = {}
    for modality in ("signal", "image"):
        mean_sample = torch.cat([batch[modality] for batch in train]).mean(dim=0, keepdim=True)
        target = torch.cat([batch[modality] for batch in heldout])
        errors[modality] = _meanSquaredError(mean_sample.expand_as(target), target)
    return errors


def measurePair(
    model: GlobalVae,
    distance: str,
    train: list[dict[str, torch.Tensor]],
    heldout: list[dict[str, torch.Tensor]],
) -> PairMeasures:
    """Read a pair model the way `scripts/evaluate.py --latent-pair` does.

    Args:
        model: A pair model with the spaces `z_signal` and `z_image`.
        distance: The registered latent distance to report (the one stage 3 trained with).
        train: Training batches, only for the constant predictor.
        heldout: Held-out batches.

    Returns:
        The `PairMeasures`.
    """
    report = evaluateLatentAlignment(
        model, SPACES[0], SPACES[1], heldout, distances=[distance], ks=[RETRIEVAL_K]
    )
    forward, backward = f"{SPACES[0]}->{SPACES[1]}", f"{SPACES[1]}->{SPACES[0]}"
    metric = f"recall@{RETRIEVAL_K}"
    # The first view's label is the name of the space and its self reconstruction is the signal;
    # the second's is the image. `cross_reconstruction["a->b"]` holds the modality of `b`.
    return PairMeasures(
        distance=report.distances[distance].paired,
        recall={name: report.retrieval[name][metric] for name in (forward, backward)},
        chance_recall=report.retrieval[forward][f"chance_{metric}"],
        active_units={name: report.diagnostics[name].active_units for name in SPACES},
        self_error={
            "signal": report.self_reconstruction[SPACES[0]]["signal"]["mse"],
            "image": report.self_reconstruction[SPACES[1]]["image"]["mse"],
        },
        cross_error={
            "image": report.cross_reconstruction[forward]["image"]["mse"],
            "signal": report.cross_reconstruction[backward]["signal"]["mse"],
        },
        baseline_error=constantPredictorErrors(train, heldout),
        num_heldout=report.num_heldout,
    )


def buildPairModelForEvaluation() -> GlobalVae:
    """The pair model of `experiment/pair_align`, which stage 3 saves its checkpoint from.

    A `--model-factory` for `scripts/evaluate.py`: the same architecture the checkpoint was saved
    from, built from the experiment file that trained it.
    """
    experiment = loadExperimentConfig(
        "experiment/pair_align",
        overrides=[f"data.loader_factory={LOADER_FACTORY}", "data.train_path=unused"],
    )
    return buildModelFromConfig(experiment.model)


def buildHeldoutPairLoader() -> list[dict[str, torch.Tensor]]:
    """A `--dataloader-factory` for `scripts/evaluate.py`: the held-out pairs."""
    return heldoutPairs()


def buildTrainPairLoader() -> list[dict[str, torch.Tensor]]:
    """A `--train-dataloader-factory` for `scripts/evaluate.py`: the training pairs."""
    return trainPairs()
