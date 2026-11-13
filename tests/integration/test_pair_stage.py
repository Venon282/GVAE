"""Stage 3 with the pair-stage loss terms, on a linked synthetic pair (roadmap P6-4, ADR 0054).

The stage is run the way the recipe runs it: the pair model of
`configs/model/signal_image_pair.yaml` warm-started from two real solo checkpoints (ADR 0052),
`training=several_latents`, modality dropout off, the sum reduction, and the two terms,
`latent_alignment` and `cross_reconstruction`, with a ramp from zero (a ramp protects the solo
solutions at the start). The data is a signal and an image that describe the same two numbers
(`makeLinkedBatches`): on independent modalities there would be nothing for the terms to align.
It is synthetic, and the numbers are those of a small pair, not of a real one (roadmap I-15).

What the roadmap's "done when" asks, and the tests that pin it:

- stage 3 lowers the latent distance over the epochs: it ends under a tenth of where it started
  and under half of where the same stage without the terms ends (the control's distance falls too,
  because both spaces move toward the prior under the KL term, so a fall alone proves little);
- without raising a self reconstruction loss by more than a stated tolerance: neither modality
  ends above `SELF_TOLERANCE` times its starting value;
- the cross decoding, which is what "aligned" means (a signal's code decoded as an image gives the
  image), is better with the terms than without, in both directions;
- each term shows in the logs: in the history, in the validation metrics and in the CSV file;
- stage 3 resumes under the guarantee of ADR 0032: a run stopped and resumed from its checkpoint
  gives the loss curve and the weights of the uninterrupted one, for a deterministic distance and
  for `mixture_jsd`, whose estimate draws random numbers, so the random-number state matters.

The rules of the stage are also pinned here, on the real model: a batch that lacks a modality
stops the run by name, modality dropout is refused when the trainer is built, and so are two
spaces of different widths.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from global_vae.config.experiment import ExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import (
    BetaScheduleConfig,
    LoggerEntryConfig,
    LossTermConfig,
    WarmStartEntryConfig,
    applyWarmStartFromConfig,
    buildTrainerFromConfig,
    resumeTrainerFromConfig,
)
from global_vae.losses.distances.registry import getLatentDistanceClass
from global_vae.models.global_vae import GlobalVae
from global_vae.training.trainer import Trainer
from global_vae.utils.seed import setGlobalSeed
from tests.integration._warm_start_fixtures import (
    loadPairExperiment,
    makeLinkedBatches,
    makePairedBatches,
    trainSoloCheckpoint,
)

MODALITIES = ("signal", "image")
SEED = 11
EPOCHS = 100
BETA = 0.1
RAMP_STEPS = 100
RAMP_END = 0.3
TRAIN_BATCHES = 4

SELF_TOLERANCE = 1.10
"""A self reconstruction may end at most this many times its starting value (10% above)."""
DISTANCE_FRACTION_OF_START = 0.1
DISTANCE_FRACTION_OF_CONTROL = 0.5

REL_TOLERANCE = 1e-6
ABS_TOLERANCE = 1e-9

TERM_NAMES = ("latent_alignment", "cross_reconstruction")


@pytest.fixture(scope="module")
def solo_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The `best.pt` of the signal and of the image solo stage, trained once for the module."""
    root = tmp_path_factory.mktemp("solo")
    return {modality: trainSoloCheckpoint(modality, root / modality) for modality in MODALITIES}


def _terms(distance: str = "wasserstein2_diag") -> list[LossTermConfig]:
    """Both terms, each with a ramp from zero over `RAMP_STEPS` steps to `RAMP_END`."""

    def ramp() -> BetaScheduleConfig:
        return BetaScheduleConfig(
            strategy="linear_warmup",
            kwargs={"warmup_steps": RAMP_STEPS, "end_value": RAMP_END},
        )

    return [
        LossTermConfig(
            name="latent_alignment",
            schedule=ramp(),
            kwargs={"pairs": [["z_signal", "z_image"]], "distance": distance},
        ),
        LossTermConfig(
            name="cross_reconstruction",
            schedule=ramp(),
            kwargs={"pairs": [["z_signal", "z_image"], ["z_image", "z_signal"]]},
        ),
    ]


def _experiment(
    output_dir: Path,
    solo_checkpoints: dict[str, Path],
    terms: list[LossTermConfig],
    num_epochs: int,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Compose stage 3: warm-started from the two solo files, dropout off, sum reduction."""
    experiment = loadPairExperiment(
        output_dir,
        [
            "training.device=cpu",
            "training.modality_dropout_p=0.0",
            "training.reconstruction_reduction=sum",
            f"training.beta={BETA}",
            f"training.num_epochs={num_epochs}",
            *(overrides or []),
        ],
    )
    experiment.training.warm_start = [
        WarmStartEntryConfig(checkpoint=str(solo_checkpoints[modality]), modules=[modality])
        for modality in MODALITIES
    ]
    experiment.training.loss_terms = terms
    return experiment


def _buildStage(experiment: ExperimentConfig, seed: int) -> tuple[Trainer, GlobalVae]:
    """Build the pair model, warm-start it (a no-op on a resumed run) and build its trainer."""
    setGlobalSeed(seed)
    model = buildModelFromConfig(experiment.model)
    applyWarmStartFromConfig(
        model,
        experiment.training,
        experiment.model,
        probe_dataloader=makePairedBatches(2, 5),
    )
    trainer = buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)
    return trainer, model


def _fit(trainer: Trainer, num_epochs: int) -> list[dict[str, float]]:
    return trainer.fit(
        makeLinkedBatches(TRAIN_BATCHES, 0),
        num_epochs=num_epochs,
        val_dataloader=makeLinkedBatches(1, 1),
    )


@dataclass(frozen=True)
class Measures:
    """What a stage is judged by, read on the posterior means of held-out batches.

    The distance is the squared 2-Wasserstein distance between the two posteriors, whatever the
    stage trained with, and each reconstruction figure is the sum of squared errors of one sample,
    averaged over the batch. The means (no sampling) keep the training noise out of them.
    """

    distance: float
    self_signal: float
    self_image: float
    cross_signal_to_image: float
    cross_image_to_signal: float


def _measure(model: GlobalVae) -> Measures:
    """Measure the model on two held-out linked batches, then put it back in training mode."""
    distance = getLatentDistanceClass("wasserstein2_diag")()
    totals = dict.fromkeys(Measures.__dataclass_fields__, 0.0)
    batches = makeLinkedBatches(2, 99)

    def squared(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
        return float(((reconstruction - target) ** 2).flatten(1).sum(1).mean().item())

    model.eval()
    with torch.no_grad():
        for batch in batches:
            encoded = model.encode(batch, use_mean=True)
            params, samples = encoded["latent_params"], encoded["latent_samples"]
            decoded = model.decode(samples)
            totals["distance"] += float(
                distance(params["z_signal"], params["z_image"]).mean().item()
            )
            totals["self_signal"] += squared(decoded["signal"], batch["signal"])
            totals["self_image"] += squared(decoded["image"], batch["image"])
            totals["cross_signal_to_image"] += squared(
                model.decode({"z_image": samples["z_signal"]})["image"], batch["image"]
            )
            totals["cross_image_to_signal"] += squared(
                model.decode({"z_signal": samples["z_image"]})["signal"], batch["signal"]
            )
    model.train()
    return Measures(**{name: total / len(batches) for name, total in totals.items()})


@dataclass
class Stage3Run:
    """One run of stage 3: where it started, where it ended, and what it logged."""

    trainer: Trainer
    history: list[dict[str, float]]
    start: Measures
    end: Measures
    csv_path: Path


def _runStage3(
    output_dir: Path, solo_checkpoints: dict[str, Path], terms: list[LossTermConfig]
) -> Stage3Run:
    csv_path = output_dir / "metrics.csv"
    experiment = _experiment(output_dir, solo_checkpoints, terms, EPOCHS)
    experiment.training.loggers = [LoggerEntryConfig(name="csv", kwargs={"path": str(csv_path)})]
    trainer, model = _buildStage(experiment, SEED)
    start = _measure(model)
    history = _fit(trainer, EPOCHS)
    return Stage3Run(trainer, history, start, _measure(model), csv_path)


@pytest.fixture(scope="module")
def control(
    tmp_path_factory: pytest.TempPathFactory, solo_checkpoints: dict[str, Path]
) -> Stage3Run:
    """Stage 3 with no term: what the stage does to the pair on its own."""
    return _runStage3(tmp_path_factory.mktemp("control"), solo_checkpoints, [])


@pytest.fixture(scope="module")
def aligned(
    tmp_path_factory: pytest.TempPathFactory, solo_checkpoints: dict[str, Path]
) -> Stage3Run:
    """Stage 3 with `latent_alignment` and `cross_reconstruction`, ramped from zero."""
    return _runStage3(tmp_path_factory.mktemp("aligned"), solo_checkpoints, _terms())


class TestTheTermsAlignThePair:
    def test_both_runs_start_from_the_same_warm_started_pair(
        self, control: Stage3Run, aligned: Stage3Run
    ) -> None:
        assert aligned.start == control.start
        # The solo models were trained on their own: the two spaces start far apart.
        assert aligned.start.distance > 50.0

    def test_the_latent_distance_falls_to_a_tenth_of_its_start_and_half_of_the_controls(
        self, control: Stage3Run, aligned: Stage3Run
    ) -> None:
        assert aligned.end.distance < DISTANCE_FRACTION_OF_START * aligned.start.distance
        assert aligned.end.distance < DISTANCE_FRACTION_OF_CONTROL * control.end.distance

    @pytest.mark.parametrize("measure", ["self_signal", "self_image"])
    def test_no_self_reconstruction_ends_above_its_start_by_more_than_the_tolerance(
        self, aligned: Stage3Run, measure: str
    ) -> None:
        start, end = getattr(aligned.start, measure), getattr(aligned.end, measure)
        assert start > 0.0
        assert end <= SELF_TOLERANCE * start, (measure, start, end)

    @pytest.mark.parametrize("measure", ["cross_signal_to_image", "cross_image_to_signal"])
    def test_the_cross_decoding_is_better_with_the_terms_than_without(
        self, control: Stage3Run, aligned: Stage3Run, measure: str
    ) -> None:
        assert getattr(aligned.end, measure) < getattr(control.end, measure), measure

    def test_the_cross_decoding_of_the_signal_as_an_image_is_far_better(
        self, control: Stage3Run, aligned: Stage3Run
    ) -> None:
        # The control drifts: its image decoder is fed a code that means something else.
        assert aligned.end.cross_signal_to_image < 0.5 * control.end.cross_signal_to_image


class TestEachTermShowsInTheLogs:
    @pytest.mark.parametrize("name", TERM_NAMES)
    def test_the_term_is_in_every_epoch_of_the_history_for_both_splits(
        self, aligned: Stage3Run, name: str
    ) -> None:
        assert len(aligned.history) == EPOCHS
        for split in ("train", "val"):
            values = [entry[f"{split}/loss/{name}"] for entry in aligned.history]
            assert all(value > 0.0 and value == value for value in values), (split, name)

    @pytest.mark.parametrize("name", TERM_NAMES)
    def test_the_logged_value_is_the_weighted_one_so_the_ramp_starts_it_near_zero(
        self, aligned: Stage3Run, name: str
    ) -> None:
        # ADR 0054: the metric is weight times value, and the weight starts at 0, so in the
        # first epoch (weights 0 to 0.009) a term is a small part of what the stage minimizes.
        first = aligned.history[0]
        assert first[f"train/loss/{name}"] < 0.05 * first["train/loss/total"]

    def test_the_terms_are_in_the_total_with_the_reconstruction_and_the_regularization(
        self, aligned: Stage3Run
    ) -> None:
        last = aligned.history[-1]
        parts = sum(last[f"train/loss/{name}"] for name in TERM_NAMES)
        assert parts > 0.0
        assert last["train/loss/total"] > parts

    @pytest.mark.parametrize("name", TERM_NAMES)
    def test_the_csv_file_has_the_term_for_the_train_and_the_validation_metrics(
        self, aligned: Stage3Run, name: str
    ) -> None:
        with aligned.csv_path.open() as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == ["x", "tag", "metric", "value"]
        written = {row[2] for row in rows[1:]}
        assert f"train/loss/{name}" in written
        assert f"val/loss/{name}" in written

    def test_a_run_without_the_terms_logs_neither(self, control: Stage3Run) -> None:
        assert not any(name in key for key in control.history[-1] for name in TERM_NAMES)
        assert "loss_terms_state_dict" not in torch.load(
            control.csv_path.parent / "checkpoints" / "best.pt", weights_only=True
        )


class TestTheRulesOfTheStage:
    def _stage(self, tmp_path: Path, solo: dict[str, Path], overrides: list[str]) -> Trainer:
        experiment = _experiment(tmp_path, solo, _terms(), 1, overrides)
        return _buildStage(experiment, SEED)[0]

    def test_a_loader_that_does_not_pair_its_samples_stops_the_run_by_name(
        self, tmp_path: Path, solo_checkpoints: dict[str, Path]
    ) -> None:
        trainer = self._stage(tmp_path, solo_checkpoints, [])
        unpaired = [{"signal": batch["signal"]} for batch in makeLinkedBatches(2, 0)]
        with pytest.raises(ValueError, match=r"needs paired batches.*'image'"):
            trainer.fit(unpaired, num_epochs=1)

    def test_a_validation_loader_that_does_not_pair_its_samples_is_refused_too(
        self, tmp_path: Path, solo_checkpoints: dict[str, Path]
    ) -> None:
        trainer = self._stage(tmp_path, solo_checkpoints, [])
        unpaired = [{"image": batch["image"]} for batch in makeLinkedBatches(1, 1)]
        with pytest.raises(ValueError, match=r"needs paired batches.*'signal'"):
            trainer.fit(makeLinkedBatches(1, 0), num_epochs=1, val_dataloader=unpaired)

    @pytest.mark.parametrize("name", TERM_NAMES)
    def test_modality_dropout_is_refused_when_the_trainer_is_built(
        self, tmp_path: Path, solo_checkpoints: dict[str, Path], name: str
    ) -> None:
        terms = [term for term in _terms() if term.name == name]
        experiment = _experiment(
            tmp_path, solo_checkpoints, terms, 1, ["training.modality_dropout_p=0.3"]
        )
        with pytest.raises(ValueError, match=rf"'{name}'.*modality dropout must be off"):
            _buildStage(experiment, SEED)

    @pytest.mark.parametrize("name", TERM_NAMES)
    def test_two_spaces_of_different_widths_are_refused_when_the_trainer_is_built(
        self, tmp_path: Path, solo_checkpoints: dict[str, Path], name: str
    ) -> None:
        terms = [term for term in _terms() if term.name == name]
        experiment = _experiment(tmp_path, solo_checkpoints, terms, 1)
        experiment.training.warm_start = []  # the solo files have the width of the other model
        assert experiment.model.latent is not None
        experiment.model.latent.spaces["z_image"].dim = 8
        setGlobalSeed(SEED)
        model = buildModelFromConfig(experiment.model)
        with pytest.raises(ValueError, match=r"latent widths 16 and 8.*roadmap D-16"):
            buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)

    def test_a_pair_that_names_a_space_the_model_does_not_have_is_refused(
        self, tmp_path: Path, solo_checkpoints: dict[str, Path]
    ) -> None:
        terms = [
            LossTermConfig(
                name="latent_alignment",
                kwargs={"pairs": [["z_signal", "z_picture"]], "distance": "cosine"},
            )
        ]
        experiment = _experiment(tmp_path, solo_checkpoints, terms, 1)
        with pytest.raises(ValueError, match=r"'z_picture', which is not a latent space"):
            _buildStage(experiment, SEED)

    def test_the_distance_has_no_default(
        self, tmp_path: Path, solo_checkpoints: dict[str, Path]
    ) -> None:
        terms = [
            LossTermConfig(name="latent_alignment", kwargs={"pairs": [["z_signal", "z_image"]]})
        ]
        experiment = _experiment(tmp_path, solo_checkpoints, terms, 1)
        with pytest.raises(TypeError, match="distance"):
            _buildStage(experiment, SEED)


RESUME_EPOCHS = 6
INTERRUPT_AFTER = 3


def _resumeExperiment(
    output_dir: Path,
    solo_checkpoints: dict[str, Path],
    distance: str,
    resume_from: Path | None = None,
) -> ExperimentConfig:
    experiment = _experiment(
        output_dir,
        solo_checkpoints,
        [
            LossTermConfig(
                name=term.name,
                schedule=BetaScheduleConfig(
                    strategy="linear_warmup", kwargs={"warmup_steps": 8, "end_value": 0.3}
                ),
                kwargs=term.kwargs,
            )
            for term in _terms(distance)
        ],
        RESUME_EPOCHS,
    )
    experiment.training.callbacks = {"checkpoint": {"directory": str(output_dir / "checkpoints")}}
    experiment.training.log_every_n_steps = 1000
    experiment.training.resume_from = str(resume_from) if resume_from is not None else None
    return experiment


def _interruptAndResume(
    tmp_path: Path, solo_checkpoints: dict[str, Path], distance: str, drop_rng: bool = False
) -> tuple[Trainer, Trainer]:
    """Return (uninterrupted trainer, resumed trainer), both having run `RESUME_EPOCHS` epochs.

    The resumed trainer is built with another seed and no warm start that matters: nothing
    survives from the first run but its checkpoint file.
    """
    whole_experiment = _resumeExperiment(tmp_path / "whole", solo_checkpoints, distance)
    whole, _ = _buildStage(whole_experiment, SEED)
    _fit(whole, resumeTrainerFromConfig(whole, whole_experiment.training))

    first_experiment = _resumeExperiment(tmp_path / "first", solo_checkpoints, distance)
    first, _ = _buildStage(first_experiment, SEED)
    _fit(first, INTERRUPT_AFTER)  # the run is "killed" here
    checkpoint = (
        tmp_path / "first" / "checkpoints" / f"checkpoint_epoch_{INTERRUPT_AFTER - 1:04d}.pt"
    )
    assert checkpoint.is_file()
    if drop_rng:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        del raw["rng_state"]
        torch.save(raw, checkpoint)

    resumed_experiment = _resumeExperiment(
        tmp_path / "resumed", solo_checkpoints, distance, resume_from=checkpoint
    )
    resumed, _ = _buildStage(resumed_experiment, 999)
    remaining = resumeTrainerFromConfig(resumed, resumed_experiment.training)
    assert remaining == RESUME_EPOCHS - INTERRUPT_AFTER
    _fit(resumed, remaining)
    return whole, resumed


def _assertSameCurve(observed: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    assert len(observed) == len(expected) == RESUME_EPOCHS
    for epoch, (seen, wanted) in enumerate(zip(observed, expected, strict=True)):
        assert set(seen) == set(wanted), f"epoch {epoch}"
        for key, value in wanted.items():
            assert seen[key] == pytest.approx(value, rel=REL_TOLERANCE, abs=ABS_TOLERANCE), (
                f"epoch {epoch}, {key}"
            )


class TestStage3ResumesUnderTheGuaranteeOfAdr0032:
    @pytest.mark.parametrize("distance", ["wasserstein2_diag", "mixture_jsd"])
    def test_a_resumed_stage_gives_the_curve_and_the_weights_of_the_uninterrupted_one(
        self, tmp_path: Path, solo_checkpoints: dict[str, Path], distance: str
    ) -> None:
        whole, resumed = _interruptAndResume(tmp_path, solo_checkpoints, distance)

        _assertSameCurve(resumed.history, whole.history)
        assert resumed.global_step == whole.global_step == RESUME_EPOCHS * TRAIN_BATCHES
        for name, tensor in whole.model.state_dict().items():
            torch.testing.assert_close(
                resumed.model.state_dict()[name], tensor, rtol=REL_TOLERANCE, atol=1e-6
            )

    def test_the_terms_are_really_in_the_resumed_curve(
        self, tmp_path: Path, solo_checkpoints: dict[str, Path]
    ) -> None:
        _, resumed = _interruptAndResume(tmp_path, solo_checkpoints, "wasserstein2_diag")
        for name in TERM_NAMES:
            assert all(entry[f"train/loss/{name}"] > 0.0 for entry in resumed.history)

    def test_a_resumed_stage_that_forgot_the_random_number_state_would_differ(
        self, tmp_path: Path, solo_checkpoints: dict[str, Path]
    ) -> None:
        """The check above bites: `mixture_jsd` draws random numbers, so a lost state shows."""
        whole, resumed = _interruptAndResume(
            tmp_path, solo_checkpoints, "mixture_jsd", drop_rng=True
        )
        differences = [
            abs(a["train/loss/total"] - b["train/loss/total"])
            for a, b in zip(resumed.history, whole.history, strict=True)
        ]
        assert max(differences) > 1e-4
