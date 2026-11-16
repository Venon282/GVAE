"""The three stages of the staged recipe, end to end through `scripts/train.py` (roadmap P6-7).

Each stage is a subprocess, run the way `docs/how-to/train-in-stages.md` prints the commands: from
one working directory, with no `output_dir` and no `training.warm_start` on the command line, so
that stage 3 finds the two solo files at the folders the two solo experiments write to by default.
What the command line adds is only what a toy needs (the loader and the number of epochs, a few
settings that the comment of each constant gives the reason for). The data is the linked synthetic
pair of `_staged_training_fixtures.py`, in which the image is a function of the signal's frequency
and phase, so there is something to align; it is synthetic, and the numbers say that the stages
chain and what each guard sees, not how the recipe does on real pairs (roadmap I-15).

What the roadmap asks of the test, and where it is pinned:

- (a) the warm start is exact: the script copies both solo models, and in evaluation mode the pair
  model's `use_mean=True` reconstruction of each modality from its own input is the solo model's,
  bit for bit (`TestTheWarmStartIsExact`);
- (b) the latent distance is lower after stage 3 than before, by a stated factor;
- (c) `recall@10` of each direction is at chance before stage 3 and above it after, by a stated
  margin;
- (d) no self reconstruction ends above its start by more than a stated tolerance;
- (e) each cross reconstruction beats the constant "mean sample" predictor after stage 3, by a
  stated factor, and does not before it (`TestStage3AlignsThePair`: (b) to (e), and the guards of
  the evaluation of P6-6 on the checkpoint it wrote);
- (f) stage 3 resumes: a run stopped inside the ramp of its loss terms and resumed from its periodic
  checkpoint ends with the weights of the uninterrupted one (`TestStage3Resumes`);
- (g) every existing suite is unchanged: no other shipped experiment turns the stage, the warm
  start, the loss terms or the learning-rate groups on (`TestNothingElseChanged`), and this item
  changes no line of `src/`.

The distance is measured on posterior means of 64 held-out pairs no stage trained or validated on,
and the recall at 10 of a random ranking is 10 / 64.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import applyWarmStartFromConfig, buildLossTerms
from global_vae.losses.distances.mean_mse import MeanMseDistance
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint, readCheckpointWeights
from global_vae.utils.seed import setGlobalSeed
from tests.integration._staged_training_fixtures import (
    LOADER_FACTORY,
    NUM_TRAIN_BATCHES,
    RETRIEVAL_K,
    SPACES,
    PairMeasures,
    heldoutPairs,
    measurePair,
    trainPairs,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"
_EVALUATE_SCRIPT = _REPO_ROOT / "scripts" / "evaluate.py"
_EXPERIMENTS = _REPO_ROOT / "configs" / "experiment"
_FIXTURES = "tests.integration._staged_training_fixtures"

MODALITIES = ("signal", "image")
DIRECTIONS = (f"{SPACES[0]}->{SPACES[1]}", f"{SPACES[1]}->{SPACES[0]}")
DISTANCE = "wasserstein2_diag"
"""The distance `pair_align` trains with, and the one the test reads."""

SOLO_EPOCHS = 20
PAIR_EPOCHS = 100
BETA = 0.1
"""The constant beta of stage 3: the end value of the solo warm-up below, as the how-to says."""
SOLO_WARMUP_STEPS = 40
"""A warm-up that ends inside the run of 80 steps (the shipped 1000 would leave beta at 0.08)."""

# The margins of the assertions. They sit well inside what the stage reaches (the last column of the
# comment is what a run measured, on Python 3.11 and 3.13, for seeds 0 to 2 of stage 3), so that a
# change of platform does not decide the test, and well outside what an unaligned pair gives.
DISTANCE_FRACTION_OF_START = 0.1
"""After stage 3 the distance is under a tenth of its start (measured: 0.035 to 0.039)."""
CHANCE_MARGIN_BEFORE = 0.15
"""Before stage 3 the recall is within 0.15 of chance: three standard deviations of a random
ranking of 64 queries (measured: 0.00 to 0.06 above)."""
RECALL_MARGIN_AFTER = 0.5
"""After stage 3 the recall is at least 0.5 above chance (measured: 0.73 to 0.81 above)."""
SELF_TOLERANCE = 1.10
"""No self reconstruction ends more than 10 % above its start (measured: 0.37 to 0.51 times its
start: the solo stages are short, and stage 3 goes on training both models)."""
CROSS_FRACTION_OF_BASELINE = 0.8
"""A cross reconstruction ends under 0.8 times the error of the constant predictor (measured: 0.44
to 0.62, against 1.09 to 1.28 before stage 3)."""

RESUME_FROM_EPOCH = 19
"""The periodic checkpoint to resume from: 20 epochs of 4 steps is step 80, inside the 100 steps of
the ramp of the two loss terms, so the schedules are restored, and not only the weights."""
RESUME_TOLERANCE = 1e-6
"""The tolerance of the guarantee of ADR 0032."""


def _env() -> dict[str, str]:
    python_path = [str(_REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(python_path), "HYDRA_FULL_ERROR": "1"}


def _run(
    config_name: str,
    overrides: list[str],
    cwd: Path,
    *,
    name: str,
    expect_success: bool = True,
) -> str:
    """Run `scripts/train.py` on one experiment file from `cwd`, and return its output.

    No `output_dir` is given unless the caller does: the run writes where the experiment file says,
    under `cwd`, as the commands of the how-to do.
    """
    completed = subprocess.run(
        [
            sys.executable,
            str(_TRAIN_SCRIPT),
            "--config-name",
            config_name,
            f"data.loader_factory={LOADER_FACTORY}",
            "data.train_path=unused",
            "training.loggers=[]",
            "training.device=cpu",
            f"hydra.run.dir={cwd / 'hydra' / name}",
            *overrides,
        ],
        cwd=cwd,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    text = completed.stdout + completed.stderr
    if expect_success:
        assert completed.returncode == 0, text
        assert "Training complete" in text or "Nothing to train" in text, text
    else:
        assert completed.returncode != 0, text
    return text


def _readCheckpoint(path: Path) -> dict[str, Any]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint


def _soloOverrides() -> list[str]:
    """What a toy needs on top of a solo experiment file.

    The shipped solo experiments are for 200 epochs of real-sized data, and their beta warms up over
    1000 steps, so 80 steps of a toy would end with beta at 0.08. The warm-up is shortened to end
    inside the run, at the beta that stage 3 then holds. The loss scale is the file's.
    """
    return [
        f"training.num_epochs={SOLO_EPOCHS}",
        f"training.beta_schedules.z_fused.kwargs.warmup_steps={SOLO_WARMUP_STEPS}",
        f"training.beta_schedules.z_fused.kwargs.end_value={BETA}",
    ]


def _pairOverrides() -> list[str]:
    """What a toy needs on top of `pair_align`.

    Nothing about the loss terms, the ramps, the distance, the loss scales or the model: those are
    the experiment file's. Beta is the one the solo stages ended at, and the periodic checkpoints
    are all kept, so that one inside the ramp is there to resume from.
    """
    return [
        f"training.beta={BETA}",
        f"training.num_epochs={PAIR_EPOCHS}",
        "training.callbacks.checkpoint.keep_last_n=20",
    ]


@dataclass(frozen=True)
class Stages:
    """The three stages, run once for the module.

    Attributes:
        root: The working directory of every command.
        solo: Modality -> the `best.pt` of its solo stage.
        pair_dir: The folder stage 3 wrote to.
        pair_log: What `scripts/train.py` printed in stage 3.
    """

    root: Path
    solo: dict[str, Path]
    pair_dir: Path
    pair_log: str

    @property
    def pair(self) -> Path:
        return self.pair_dir / "checkpoints" / "best.pt"


@pytest.fixture(scope="module")
def stages(tmp_path_factory: pytest.TempPathFactory) -> Stages:
    """Stage 1 on the signal, stage 2 on the image, stage 3: three runs, chained by files."""
    root = tmp_path_factory.mktemp("recipe")
    solo = {}
    for modality in MODALITIES:
        _run(
            f"experiment/pair_solo_{modality}",
            _soloOverrides(),
            root,
            name=f"solo_{modality}",
        )
        solo[modality] = root / "outputs" / f"pair_solo_{modality}" / "checkpoints" / "best.pt"
        assert solo[modality].is_file()
    pair_log = _run("experiment/pair_align", _pairOverrides(), root, name="pair")
    return Stages(root, solo, root / "outputs" / "pair_align", pair_log)


def _pairExperiment(stages: Stages, extra: list[str] | None = None) -> ExperimentConfig:
    """The `pair_align` config of stage 3, composed in this process with the same overrides."""
    entries = ",".join(
        f'{{checkpoint: "{stages.solo[modality]}", modules: [{modality}]}}'
        for modality in MODALITIES
    )
    return loadExperimentConfig(
        "experiment/pair_align",
        overrides=[
            f"data.loader_factory={LOADER_FACTORY}",
            "data.train_path=unused",
            f"training.warm_start=[{entries}]",
            *_pairOverrides(),
            *(extra or []),
        ],
    )


def _warmStartedPair(stages: Stages) -> GlobalVae:
    """The pair model as stage 3 starts: built from the file, warm-started from the solo files."""
    experiment = _pairExperiment(stages)
    setGlobalSeed(experiment.seed)
    model = buildModelFromConfig(experiment.model)
    applyWarmStartFromConfig(
        model, experiment.training, experiment.model, probe_dataloader=trainPairs()
    )
    return model


def _stage3Model(stages: Stages) -> GlobalVae:
    model = buildModelFromConfig(_pairExperiment(stages).model)
    loadCheckpoint(stages.pair, model)
    return model


@dataclass(frozen=True)
class Measured:
    """The pair model before and after stage 3, read by the same function on the same pairs."""

    before: PairMeasures
    after: PairMeasures


@pytest.fixture(scope="module")
def measured(stages: Stages) -> Measured:
    train, heldout = trainPairs(), heldoutPairs()
    return Measured(
        before=measurePair(_warmStartedPair(stages), DISTANCE, train, heldout),
        after=measurePair(_stage3Model(stages), DISTANCE, train, heldout),
    )


@pytest.mark.slow
class TestTheStagesChain:
    def test_each_stage_writes_its_own_manifest(self, stages: Stages) -> None:
        expected = {
            "solo_signal": (["signal"], {"z_fused": 16}),
            "solo_image": (["image"], {"z_fused": 16}),
        }
        for modality in MODALITIES:
            manifest = _readCheckpoint(stages.solo[modality])["stage_manifest"]
            modalities, spaces = expected[f"solo_{modality}"]
            assert manifest["stage"] == f"solo_{modality}"
            assert manifest["modalities"] == modalities
            assert manifest["latent_spaces"] == spaces
        pair = _readCheckpoint(stages.pair)["stage_manifest"]
        assert pair["stage"] == "pair"
        assert pair["modalities"] == list(MODALITIES)
        assert pair["latent_spaces"] == {SPACES[0]: 16, SPACES[1]: 16}

    def test_stage_3_is_a_new_run_with_its_own_step_count(self, stages: Stages) -> None:
        checkpoint = _readCheckpoint(stages.pair_dir / "checkpoints" / "checkpoint_epoch_0099.pt")
        assert checkpoint["global_step"] == PAIR_EPOCHS * NUM_TRAIN_BATCHES

    def test_the_loss_terms_of_the_experiment_file_reach_the_trainer(self, stages: Stages) -> None:
        history = _readCheckpoint(stages.pair)["history"]
        assert history
        for epoch in history:
            for name in ("latent_alignment", "cross_reconstruction"):
                assert f"train/loss/{name}" in epoch, name
                assert f"val/loss/{name}" in epoch, name
        # The ramp of both terms starts at 0: the first step logs them at exactly nothing, which is
        # what protects the solo solutions from a distance of 50 at the start.
        first_step = re.search(r"epoch 0 step 0: (.*)", stages.pair_log)
        assert first_step is not None
        assert "loss/latent_alignment=0.0000" in first_step.group(1)
        assert "loss/cross_reconstruction=0.0000" in first_step.group(1)

    def test_without_the_solo_stages_stage_3_stops_before_its_first_step(
        self, tmp_path: Path
    ) -> None:
        # From a folder where stage 1 has not been run, the default paths of the file name nothing.
        text = _run("experiment/pair_align", [], tmp_path, name="alone", expect_success=False)
        assert "No checkpoint found at 'outputs/pair_solo_signal/checkpoints/best.pt'" in text
        assert not (tmp_path / "outputs" / "pair_align" / "checkpoints").exists()


@pytest.mark.slow
class TestTheWarmStartIsExact:
    def test_the_script_copied_both_solo_models(self, stages: Stages) -> None:
        assert "Warm start: copied encoders.signal, decoders.signal" in stages.pair_log
        assert "Warm start: copied encoders.image, decoders.image" in stages.pair_log

    def test_each_modality_is_rebuilt_bit_for_bit_from_its_own_input(self, stages: Stages) -> None:
        pair = _warmStartedPair(stages).eval()
        batch = heldoutPairs()[0]
        for modality in MODALITIES:
            experiment = loadExperimentConfig(
                f"experiment/pair_solo_{modality}",
                overrides=[f"data.loader_factory={LOADER_FACTORY}", "data.train_path=unused"],
            )
            solo = buildModelFromConfig(experiment.model)
            solo.load_state_dict(readCheckpointWeights(stages.solo[modality]).state_dict)
            solo.eval()
            with torch.no_grad():
                expected = solo({modality: batch[modality]}, use_mean=True)["reconstructions"]
                alone = pair({modality: batch[modality]}, use_mean=True)["reconstructions"]
                both = pair(batch, use_mean=True)["reconstructions"]
            assert torch.equal(alone[modality], expected[modality]), modality
            assert torch.equal(both[modality], expected[modality]), modality


@pytest.mark.slow
class TestStage3AlignsThePair:
    def test_both_spaces_start_apart_and_the_distance_falls(self, measured: Measured) -> None:
        assert measured.before.distance > 10.0
        assert measured.after.distance < DISTANCE_FRACTION_OF_START * measured.before.distance

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_retrieval_is_at_chance_before_stage_3(
        self, measured: Measured, direction: str
    ) -> None:
        before = measured.before
        assert before.chance_recall == pytest.approx(RETRIEVAL_K / before.num_heldout)
        assert before.recall[direction] <= before.chance_recall + CHANCE_MARGIN_BEFORE

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_retrieval_is_above_chance_after_stage_3(
        self, measured: Measured, direction: str
    ) -> None:
        after = measured.after
        assert after.recall[direction] >= after.chance_recall + RECALL_MARGIN_AFTER

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_no_self_reconstruction_degrades_beyond_the_tolerance(
        self, measured: Measured, modality: str
    ) -> None:
        before, after = measured.before.self_error[modality], measured.after.self_error[modality]
        assert after <= SELF_TOLERANCE * before

    @pytest.mark.parametrize("modality", MODALITIES)
    def test_the_cross_reconstruction_beats_the_mean_sample_only_after_stage_3(
        self, measured: Measured, modality: str
    ) -> None:
        baseline = measured.after.baseline_error[modality]
        assert measured.before.cross_error[modality] >= baseline
        assert measured.after.cross_error[modality] < CROSS_FRACTION_OF_BASELINE * baseline

    @pytest.mark.parametrize("space", SPACES)
    def test_no_space_ends_with_fewer_active_units_than_its_solo_stage(
        self, measured: Measured, space: str
    ) -> None:
        assert measured.after.active_units[space] >= measured.before.active_units[space]

    def test_the_evaluation_script_reports_the_checkpoint_and_flags_nothing(
        self, stages: Stages, measured: Measured, tmp_path: Path
    ) -> None:
        reference = " ".join(f"{name}={measured.before.active_units[name]}" for name in SPACES)
        completed = subprocess.run(
            [
                sys.executable,
                str(_EVALUATE_SCRIPT),
                "--checkpoint",
                str(stages.pair),
                "--model-factory",
                f"{_FIXTURES}:buildPairModelForEvaluation",
                "--dataloader-factory",
                f"{_FIXTURES}:buildHeldoutPairLoader",
                "--train-dataloader-factory",
                f"{_FIXTURES}:buildTrainPairLoader",
                "--latent-pair",
                *SPACES,
                "--retrieval-k",
                str(RETRIEVAL_K),
                "--solo-active-units",
                *reference.split(),
                "--output-dir",
                str(tmp_path),
                "--no-figures",
                "--device",
                "cpu",
            ],
            cwd=tmp_path,
            env=_env(),
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        report = _readJson(tmp_path / "latent_alignment.json")
        assert (tmp_path / "results.json").is_file()
        assert (tmp_path / "latent_alignment.md").is_file()
        assert report["flags"] == []
        assert report["pair_integrity"]["status"] == "ok"
        retrieval = report["retrieval"][DIRECTIONS[0]]
        assert retrieval[f"recall@{RETRIEVAL_K}"] >= retrieval[f"chance_recall@{RETRIEVAL_K}"] + (
            RECALL_MARGIN_AFTER
        )
        assert report["procrustes"] is not None


def _readJson(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


@pytest.mark.slow
class TestStage3Resumes:
    def test_a_run_stopped_inside_the_ramp_ends_where_the_uninterrupted_one_does(
        self, stages: Stages, tmp_path: Path
    ) -> None:
        interrupted = (
            stages.pair_dir / "checkpoints" / f"checkpoint_epoch_{RESUME_FROM_EPOCH:04d}.pt"
        )
        assert interrupted.is_file()
        text = _run(
            "experiment/pair_align",
            [
                *_pairOverrides(),
                f"output_dir={tmp_path / 'resumed'}",
                f"training.resume_from={interrupted}",
            ],
            stages.root,
            name="resumed",
        )
        # The solo files are still there, and the run does not read them again: its weights are
        # those of the checkpoint it resumes.
        assert "training.warm_start is not applied" in text
        assert f"Starting training for {PAIR_EPOCHS - RESUME_FROM_EPOCH - 1} epoch(s)" in text

        last = f"checkpoint_epoch_{PAIR_EPOCHS - 1:04d}.pt"
        resumed = _readCheckpoint(tmp_path / "resumed" / "checkpoints" / last)
        uninterrupted = _readCheckpoint(stages.pair_dir / "checkpoints" / last)
        assert resumed["global_step"] == uninterrupted["global_step"]
        assert resumed["model_state_dict"].keys() == uninterrupted["model_state_dict"].keys()
        for name, tensor in uninterrupted["model_state_dict"].items():
            torch.testing.assert_close(
                resumed["model_state_dict"][name],
                tensor,
                rtol=RESUME_TOLERANCE,
                atol=RESUME_TOLERANCE,
                msg=lambda message, key=name: f"{key}: {message}",  # type: ignore[misc]
            )
        for name in ("latent_alignment", "cross_reconstruction"):
            key = f"train/loss/{name}"
            assert resumed["history"][-1][key] == pytest.approx(
                uninterrupted["history"][-1][key], rel=1e-4
            ), key


def _composeAll() -> dict[str, ExperimentConfig]:
    composed = {}
    for path in sorted(_EXPERIMENTS.glob("*.yaml")):
        composed[path.stem] = loadExperimentConfig(
            f"experiment/{path.stem}",
            overrides=[f"data.loader_factory={LOADER_FACTORY}", "data.train_path=unused"],
        )
    return composed


class TestNothingElseChanged:
    """(g): the recipe is opt-in. No experiment but its three files asks for any of it."""

    RECIPE = ("pair_solo_signal", "pair_solo_image", "pair_align")

    def test_the_three_experiment_files_of_the_recipe_exist(self) -> None:
        for name in self.RECIPE:
            assert (_EXPERIMENTS / f"{name}.yaml").is_file(), name

    def test_no_other_experiment_turns_on_a_stage_a_warm_start_terms_or_groups(self) -> None:
        for name, experiment in _composeAll().items():
            if name in self.RECIPE:
                continue
            training = experiment.training
            assert training.stage is None, name
            assert training.warm_start == [], name
            assert training.loss_terms == [], name
            assert training.param_groups == [], name

    def test_only_the_pair_experiment_lists_loss_terms_and_a_warm_start(self) -> None:
        composed = _composeAll()
        assert [term.name for term in composed["pair_align"].training.loss_terms] == [
            "latent_alignment",
            "cross_reconstruction",
        ]
        assert len(composed["pair_align"].training.warm_start) == 2
        for name in ("pair_solo_signal", "pair_solo_image"):
            assert composed[name].training.loss_terms == [], name
            assert composed[name].training.warm_start == [], name

    def test_the_stage_labels_are_the_ones_the_how_to_names(self) -> None:
        composed = _composeAll()
        assert composed["pair_solo_signal"].training.stage == "solo_signal"
        assert composed["pair_solo_image"].training.stage == "solo_image"
        assert composed["pair_align"].training.stage == "pair"

    def test_the_distance_of_the_pair_stage_is_swapped_from_the_command_line(self) -> None:
        experiment = loadExperimentConfig(
            "experiment/pair_align",
            overrides=[
                f"data.loader_factory={LOADER_FACTORY}",
                "data.train_path=unused",
                "training.loss_terms.0.kwargs.distance=mean_mse",
            ],
        )
        terms, _ = buildLossTerms(experiment.training)
        distance = getattr(terms["latent_alignment"], "distance")  # noqa: B009
        assert isinstance(distance, MeanMseDistance)

    def test_beta_is_constant_in_both_spaces_and_follows_training_beta(self) -> None:
        """`pair_align` names a constant schedule for each space, and `training.beta` moves both."""
        for overrides, expected in (([], 1.0), ([f"training.beta={BETA}"], BETA)):
            experiment = loadExperimentConfig(
                "experiment/pair_align",
                overrides=[
                    f"data.loader_factory={LOADER_FACTORY}",
                    "data.train_path=unused",
                    *overrides,
                ],
            )
            schedules = experiment.training.beta_schedules
            assert set(schedules) == set(SPACES)
            for space, schedule in schedules.items():
                assert schedule.strategy == "constant", space
                assert schedule.kwargs == {"value": expected}, space
