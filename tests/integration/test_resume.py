"""Resuming a run from a checkpoint (roadmap P2-3, ADR 0032).

See `docs/adr/0032-resume-from-the-training-script.md`.

The stated guarantee, which the first class below pins: **a resumed deterministic run
reproduces the loss curve of the uninterrupted run to a relative tolerance of 1e-6** (and an
absolute one of 1e-9 for values near zero), for every metric of every epoch, and ends with the
same weights to `1e-6`. On the CPU, with the same PyTorch, the two runs execute the same
floating point operations in the same order, so in practice they agree exactly; the tolerance
is the margin the guarantee promises, not the margin the tests need.

A run is "interrupted" by fitting only the first epochs and building a brand new trainer, with
another seed and so another initial model, from the same config with `resume_from` set: nothing
survives from the first trainer but the checkpoint file. `num_epochs` is the total of the run
(`resumeTrainerFromConfig`), so both configs say the same `num_epochs`.
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from global_vae.config.training import (
    BetaScheduleConfig,
    LoggerEntryConfig,
    TrainingConfig,
    buildCallbacksFromConfig,
    buildTrainerFromConfig,
    resumeTrainerFromConfig,
)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.best_checkpoint import BestCheckpointCallback
from global_vae.training.callbacks.checkpoint import CheckpointCallback
from global_vae.training.callbacks.early_stopping import EarlyStopping
from global_vae.training.callbacks.reduce_lr_on_plateau import ReduceLrOnPlateau
from global_vae.training.callbacks.top_k_checkpoint import TopKCheckpointCallback
from global_vae.training.loggers.csv_logger import CsvLogger
from global_vae.training.trainer import Trainer
from global_vae.utils.seed import setGlobalSeed

INPUT_DIM = 12
LATENT_DIM = 3
HIDDEN_DIM = 10
BATCH_SIZE = 6
NUM_EPOCHS = 6
INTERRUPT_AFTER = 3
REL_TOLERANCE = 1e-6
ABS_TOLERANCE = 1e-9


@registerEncoder("dummy_signal_encoder_resume_test")
class _DummySignalEncoder(AbstractEncoder):
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self.hidden = nn.Linear(input_dim, HIDDEN_DIM)
        self.to_mu = nn.Linear(HIDDEN_DIM, latent_dim)
        self.to_logvar = nn.Linear(HIDDEN_DIM, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.hidden(x))
        return self.to_mu(h), self.to_logvar(h)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        return "signal"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_signal_decoder_resume_test")
class _DummySignalDecoder(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.hidden = nn.Linear(latent_dim, HIDDEN_DIM)
        self.project = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(torch.tanh(self.hidden(z)))
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "signal"


@registerEncoder("dummy_other_encoder_resume_test")
class _DummyOtherEncoder(_DummySignalEncoder):
    @property
    def modality_name(self) -> str:
        return "other"


@registerDecoder("dummy_other_decoder_resume_test")
class _DummyOtherDecoder(_DummySignalDecoder):
    @property
    def modality_name(self) -> str:
        return "other"


def _buildModel(two_modalities: bool) -> GlobalVae:
    modality_configs = {
        "signal": {
            "encoder": "dummy_signal_encoder_resume_test",
            "decoder": "dummy_signal_decoder_resume_test",
        }
    }
    if not two_modalities:
        return GlobalVae.createSingleLatent(
            modality_configs=modality_configs, latent_dim=LATENT_DIM
        )
    modality_configs["other"] = {
        "encoder": "dummy_other_encoder_resume_test",
        "decoder": "dummy_other_decoder_resume_test",
    }
    return GlobalVae.createSingleLatent(
        modality_configs=modality_configs, latent_dim=LATENT_DIM, fusion_strategy="poe"
    )


def _batches(two_modalities: bool, num_batches: int, seed: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    names = ["signal", "other"] if two_modalities else ["signal"]
    return [
        {name: torch.randn(BATCH_SIZE, INPUT_DIM, generator=generator) for name in names}
        for _ in range(num_batches)
    ]


@dataclass
class Scenario:
    """One way of configuring a run, to be interrupted and resumed."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    callbacks: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    two_modalities: bool = False
    with_validation: bool = True
    beta_warmup_steps: int | None = None


SCENARIOS = [
    Scenario("plain", with_validation=False),
    Scenario(
        "validation_and_plateau_reduction",
        callbacks={
            "reduce_lr_on_plateau": {
                "monitor": "val/loss/total",
                "factor": 0.5,
                "patience": 1,
                "threshold": 0.5,
            }
        },
    ),
    Scenario(
        "beta_schedule_and_cosine_lr",
        callbacks={"cosine_lr": {"total_steps": NUM_EPOCHS * 3, "warmup_steps": 2}},
        beta_warmup_steps=8,
    ),
    Scenario("modality_dropout", overrides={"modality_dropout_p": 0.5}, two_modalities=True),
    Scenario(
        "accumulation_bf16_and_clipping",
        overrides={
            "accumulate_grad_batches": 2,
            "mixed_precision": "bf16",
            "grad_clip_norm": 1.0,
        },
    ),
    Scenario(
        "everything_but_precision",
        overrides={
            "modality_dropout_p": 0.3,
            "accumulate_grad_batches": 2,
            "grad_clip_norm": 1.0,
        },
        callbacks={
            "linear_warmup_lr": {"warmup_steps": 3},
            "reduce_lr_on_plateau": {"factor": 0.5, "patience": 1, "threshold": 0.5},
            "early_stopping": {"patience": 50},
        },
        two_modalities=True,
        beta_warmup_steps=6,
    ),
]


def _trainingConfig(
    scenario: Scenario,
    checkpoint_dir: Path,
    num_epochs: int = NUM_EPOCHS,
    resume_from: Path | None = None,
) -> TrainingConfig:
    callbacks: dict[str, dict[str, Any] | None] = {
        name: dict(kwargs or {}) for name, kwargs in scenario.callbacks.items()
    }
    # The checkpoint callback goes last: it saves the state of the callbacks before it.
    callbacks["checkpoint"] = {"directory": str(checkpoint_dir)}
    config = TrainingConfig(
        num_epochs=num_epochs,
        callbacks=callbacks,
        resume_from=str(resume_from) if resume_from is not None else None,
        log_every_n_steps=1000,
        **scenario.overrides,
    )
    config.optimizer.kwargs = {"lr": 0.02}
    return config


def _buildTrainer(scenario: Scenario, config: TrainingConfig, seed: int) -> Trainer:
    setGlobalSeed(seed)
    model = _buildModel(scenario.two_modalities)
    if scenario.beta_warmup_steps is not None:
        for latent_name in model.latent_spaces:
            config.beta_schedules[latent_name] = BetaScheduleConfig(
                strategy="linear_warmup", kwargs={"warmup_steps": scenario.beta_warmup_steps}
            )
    return buildTrainerFromConfig(model, config)


def _fit(trainer: Trainer, scenario: Scenario, num_epochs: int) -> list[dict[str, float]]:
    train = _batches(scenario.two_modalities, num_batches=3, seed=11)
    val = _batches(scenario.two_modalities, num_batches=2, seed=12)
    return trainer.fit(
        train, num_epochs=num_epochs, val_dataloader=val if scenario.with_validation else None
    )


def _assertSameCurve(observed: list[dict[str, float]], expected: list[dict[str, float]]) -> None:
    assert len(observed) == len(expected)
    for epoch, (seen, wanted) in enumerate(zip(observed, expected, strict=True)):
        assert set(seen) == set(wanted), f"epoch {epoch}"
        for key, value in wanted.items():
            assert seen[key] == pytest.approx(value, rel=REL_TOLERANCE, abs=ABS_TOLERANCE), (
                f"epoch {epoch}, {key}"
            )


def _interruptAndResume(scenario: Scenario, tmp_path: Path) -> tuple[Trainer, Trainer]:
    """Return (uninterrupted trainer, resumed trainer), both having run `NUM_EPOCHS` epochs."""
    uninterrupted_config = _trainingConfig(scenario, tmp_path / "uninterrupted")
    uninterrupted = _buildTrainer(scenario, uninterrupted_config, seed=0)
    _fit(
        uninterrupted,
        scenario,
        num_epochs=resumeTrainerFromConfig(uninterrupted, uninterrupted_config),
    )

    first_config = _trainingConfig(scenario, tmp_path / "first")
    first = _buildTrainer(scenario, first_config, seed=0)
    _fit(first, scenario, num_epochs=INTERRUPT_AFTER)  # the run is "killed" here
    checkpoint = tmp_path / "first" / f"checkpoint_epoch_{INTERRUPT_AFTER - 1:04d}.pt"
    assert checkpoint.is_file()

    resumed_config = _trainingConfig(scenario, tmp_path / "resumed", resume_from=checkpoint)
    resumed = _buildTrainer(scenario, resumed_config, seed=999)
    remaining = resumeTrainerFromConfig(resumed, resumed_config)
    assert remaining == NUM_EPOCHS - INTERRUPT_AFTER
    _fit(resumed, scenario, num_epochs=remaining)
    return uninterrupted, resumed


class TestAResumedRunReproducesTheUninterruptedOne:
    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
    def test_the_loss_curve_and_the_weights_match(self, scenario: Scenario, tmp_path: Path) -> None:
        uninterrupted, resumed = _interruptAndResume(scenario, tmp_path)

        _assertSameCurve(resumed.history, uninterrupted.history)
        assert resumed.global_step == uninterrupted.global_step
        assert resumed.start_epoch == uninterrupted.start_epoch == NUM_EPOCHS
        for name, tensor in uninterrupted.model.state_dict().items():
            torch.testing.assert_close(
                resumed.model.state_dict()[name], tensor, rtol=REL_TOLERANCE, atol=1e-6
            )

    def test_the_scenarios_are_not_vacuous(self, tmp_path: Path) -> None:
        """The learning rate really moves in the scenarios that schedule it."""
        scenario = next(s for s in SCENARIOS if s.name == "beta_schedule_and_cosine_lr")
        uninterrupted, _ = _interruptAndResume(scenario, tmp_path)
        learning_rates = [
            entry["train/lr"] for entry in uninterrupted.history if "train/lr" in entry
        ]
        assert len(set(learning_rates)) > 1

    def test_a_resumed_run_that_forgot_the_rng_would_differ(self, tmp_path: Path) -> None:
        """The check above bites: dropping the checkpoint's RNG state changes the curve."""
        scenario = next(s for s in SCENARIOS if s.name == "modality_dropout")
        uninterrupted, _ = _interruptAndResume(scenario, tmp_path)

        first_config = _trainingConfig(scenario, tmp_path / "other_first")
        first = _buildTrainer(scenario, first_config, seed=0)
        _fit(first, scenario, num_epochs=INTERRUPT_AFTER)
        checkpoint = tmp_path / "other_first" / f"checkpoint_epoch_{INTERRUPT_AFTER - 1:04d}.pt"
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        del raw["rng_state"]
        torch.save(raw, checkpoint)

        config = _trainingConfig(scenario, tmp_path / "other_resumed", resume_from=checkpoint)
        resumed = _buildTrainer(scenario, config, seed=999)
        _fit(resumed, scenario, num_epochs=resumeTrainerFromConfig(resumed, config))

        differences = [
            abs(a["train/loss/total"] - b["train/loss/total"])
            for a, b in zip(resumed.history, uninterrupted.history, strict=True)
        ]
        assert max(differences) > 1e-4


class TestResumeTrainerFromConfig:
    def _checkpoint(self, tmp_path: Path, num_epochs: int = 2) -> Path:
        scenario = Scenario("plain", with_validation=False)
        config = _trainingConfig(scenario, tmp_path / "run")
        trainer = _buildTrainer(scenario, config, seed=0)
        _fit(trainer, scenario, num_epochs=num_epochs)
        return tmp_path / "run" / f"checkpoint_epoch_{num_epochs - 1:04d}.pt"

    def test_without_resume_from_the_trainer_is_untouched_and_every_epoch_is_left(
        self, tmp_path: Path
    ) -> None:
        scenario = Scenario("plain")
        config = _trainingConfig(scenario, tmp_path, num_epochs=7)
        trainer = _buildTrainer(scenario, config, seed=0)
        assert resumeTrainerFromConfig(trainer, config) == 7
        assert trainer.global_step == 0
        assert trainer.start_epoch == 0
        assert trainer.history == []

    def test_num_epochs_is_the_total_of_the_run(self, tmp_path: Path) -> None:
        scenario = Scenario("plain", with_validation=False)
        checkpoint = self._checkpoint(tmp_path, num_epochs=2)
        config = _trainingConfig(scenario, tmp_path / "again", num_epochs=5, resume_from=checkpoint)
        trainer = _buildTrainer(scenario, config, seed=3)

        remaining = resumeTrainerFromConfig(trainer, config)
        assert remaining == 3
        assert trainer.start_epoch == 2
        assert len(trainer.history) == 2
        _fit(trainer, scenario, num_epochs=remaining)
        assert len(trainer.history) == 5
        assert trainer.start_epoch == 5

    @pytest.mark.parametrize("num_epochs", [2, 1])
    def test_nothing_is_left_when_the_checkpoint_already_holds_the_epochs(
        self, tmp_path: Path, num_epochs: int
    ) -> None:
        scenario = Scenario("plain", with_validation=False)
        checkpoint = self._checkpoint(tmp_path, num_epochs=2)
        config = _trainingConfig(
            scenario, tmp_path / "again", num_epochs=num_epochs, resume_from=checkpoint
        )
        trainer = _buildTrainer(scenario, config, seed=3)
        assert resumeTrainerFromConfig(trainer, config) == 0

    def test_a_missing_file_is_a_clear_error(self, tmp_path: Path) -> None:
        scenario = Scenario("plain")
        config = _trainingConfig(scenario, tmp_path, resume_from=tmp_path / "nope.pt")
        trainer = _buildTrainer(scenario, config, seed=0)
        with pytest.raises(FileNotFoundError, match=r"training\.resume_from: no checkpoint at"):
            resumeTrainerFromConfig(trainer, config)

    def test_a_directory_is_not_a_checkpoint(self, tmp_path: Path) -> None:
        scenario = Scenario("plain")
        config = _trainingConfig(scenario, tmp_path, resume_from=tmp_path)
        trainer = _buildTrainer(scenario, config, seed=0)
        with pytest.raises(FileNotFoundError, match="no checkpoint"):
            resumeTrainerFromConfig(trainer, config)

    def test_it_logs_what_was_resumed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        scenario = Scenario("plain", with_validation=False)
        checkpoint = self._checkpoint(tmp_path, num_epochs=2)
        config = _trainingConfig(scenario, tmp_path / "again", num_epochs=4, resume_from=checkpoint)
        trainer = _buildTrainer(scenario, config, seed=3)
        with caplog.at_level(logging.INFO, logger="global_vae.config.training"):
            resumeTrainerFromConfig(trainer, config)
        assert "2 of 4 epoch(s) left" in caplog.text

    def test_the_fields_default_to_off(self) -> None:
        config = TrainingConfig()
        assert config.resume_from is None
        assert config.mixed_precision is None
        assert config.accumulate_grad_batches == 1
        assert config.compile_model is False
        assert config.compile_kwargs == {}


class _Forced(TrainerCallback):
    """Overwrites the monitored metric with a hand-picked sequence (see test_reduce_lr...)."""

    def __init__(self, key: str, values: list[float]) -> None:
        self.key = key
        self._values = iter(values)

    def onEpochEnd(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        metrics[self.key] = next(self._values)


class TestCallbackStateSurvivesAResume:
    def _trainer(self, callbacks: list[TrainerCallback]) -> Trainer:
        setGlobalSeed(0)
        return Trainer(
            _buildModel(False), device="cpu", callbacks=callbacks, log_every_n_steps=1000
        )

    def _data(self) -> list[dict[str, torch.Tensor]]:
        return _batches(False, num_batches=2, seed=1)

    def test_early_stopping_stops_on_the_same_epoch(self, tmp_path: Path) -> None:
        values = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

        def callbacks(directory: Path, forced: list[float]) -> list[TrainerCallback]:
            return [
                _Forced("forced", forced),
                EarlyStopping(monitor="forced", patience=3),
                CheckpointCallback(directory),
            ]

        uninterrupted = self._trainer(callbacks(tmp_path / "u", values))
        uninterrupted.fit(self._data(), num_epochs=8)
        stopped_at = len(uninterrupted.history)
        assert stopped_at < 8, "early stopping must fire for this test to mean anything"

        first = self._trainer(callbacks(tmp_path / "a", values))
        first.fit(self._data(), num_epochs=2)
        resumed = self._trainer(callbacks(tmp_path / "b", values[2:]))
        resumed.loadCheckpoint(tmp_path / "a" / "checkpoint_epoch_0001.pt")
        resumed.fit(self._data(), num_epochs=8 - resumed.start_epoch)

        assert len(resumed.history) == stopped_at

    def test_reduce_lr_on_plateau_counts_the_bad_epochs_of_the_first_part(
        self, tmp_path: Path
    ) -> None:
        values = [5.0] * 6

        def callbacks(directory: Path, forced: list[float]) -> list[TrainerCallback]:
            return [
                _Forced("forced", forced),
                ReduceLrOnPlateau(monitor="forced", factor=0.5, patience=2),
                CheckpointCallback(directory),
            ]

        uninterrupted = self._trainer(callbacks(tmp_path / "u", values))
        uninterrupted.fit(self._data(), num_epochs=6)
        first = self._trainer(callbacks(tmp_path / "a", values))
        first.fit(self._data(), num_epochs=2)
        resumed = self._trainer(callbacks(tmp_path / "b", values[2:]))
        resumed.loadCheckpoint(tmp_path / "a" / "checkpoint_epoch_0001.pt")
        resumed.fit(self._data(), num_epochs=4)

        assert (
            resumed.optimizer.param_groups[0]["lr"] == uninterrupted.optimizer.param_groups[0]["lr"]
        )
        assert resumed.optimizer.param_groups[0]["lr"] < 1e-3

    def test_best_checkpoint_does_not_overwrite_a_better_file_after_a_resume(
        self, tmp_path: Path
    ) -> None:
        """Without its state a resumed `BestCheckpointCallback` would start from `inf`."""
        best = tmp_path / "best.pt"
        values = [5.0, 1.0, 3.0, 4.0, 2.0]

        def callbacks(directory: Path, forced: list[float]) -> list[TrainerCallback]:
            return [
                _Forced("forced", forced),
                BestCheckpointCallback(best, monitor="forced"),
                CheckpointCallback(directory),
            ]

        first = self._trainer(callbacks(tmp_path / "a", values))
        first.fit(self._data(), num_epochs=3)
        assert torch.load(best, map_location="cpu", weights_only=True)["start_epoch"] == 2

        resumed = self._trainer(callbacks(tmp_path / "b", values[3:]))
        resumed.loadCheckpoint(tmp_path / "a" / "checkpoint_epoch_0002.pt")
        resumed.fit(self._data(), num_epochs=2)

        # Epochs 3 (4.0) and 4 (2.0) are both worse than the 1.0 of epoch 1.
        assert torch.load(best, map_location="cpu", weights_only=True)["start_epoch"] == 2

    def test_top_k_keeps_the_same_files_after_a_resume(self, tmp_path: Path) -> None:
        values = [4.0, 2.0, 3.0, 1.0, 5.0, 0.5]

        def callbacks(directory: Path, forced: list[float], top: Path) -> list[TrainerCallback]:
            return [
                _Forced("forced", forced),
                TopKCheckpointCallback(top, k=2, monitor="forced"),
                CheckpointCallback(directory),
            ]

        uninterrupted_top = tmp_path / "top_u"
        uninterrupted = self._trainer(callbacks(tmp_path / "u", values, uninterrupted_top))
        uninterrupted.fit(self._data(), num_epochs=6)

        resumed_top = tmp_path / "top_r"
        first = self._trainer(callbacks(tmp_path / "a", values, resumed_top))
        first.fit(self._data(), num_epochs=3)
        resumed = self._trainer(callbacks(tmp_path / "b", values[3:], resumed_top))
        resumed.loadCheckpoint(tmp_path / "a" / "checkpoint_epoch_0002.pt")
        resumed.fit(self._data(), num_epochs=3)

        assert sorted(p.name for p in resumed_top.iterdir()) == sorted(
            p.name for p in uninterrupted_top.iterdir()
        )
        assert sorted(p.name for p in resumed_top.iterdir()) == [
            "top_k_epoch_0003.pt",
            "top_k_epoch_0005.pt",
        ]

    def test_a_saved_state_without_a_matching_callback_is_dropped_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = self._trainer(
            [
                _Forced("forced", [5.0, 5.0]),
                EarlyStopping(monitor="forced", patience=3),
                CheckpointCallback(tmp_path / "a"),
            ]
        )
        first.fit(self._data(), num_epochs=2)

        plain = self._trainer([])
        with caplog.at_level(logging.WARNING, logger="global_vae.training.trainer"):
            plain.loadCheckpoint(tmp_path / "a" / "checkpoint_epoch_0001.pt")
        assert "EarlyStopping" in caplog.text
        assert plain.start_epoch == 2

    def test_a_checkpoint_without_callback_state_loads_into_a_trainer_with_callbacks(
        self, tmp_path: Path
    ) -> None:
        first = self._trainer([])
        first.fit(self._data(), num_epochs=1)
        path = tmp_path / "plain.pt"
        first.saveCheckpoint(path)

        callback = EarlyStopping(monitor="train/loss/total", patience=3)
        second = self._trainer([callback])
        second.loadCheckpoint(path)
        assert callback.stateDict() == EarlyStopping(monitor="train/loss/total").stateDict()


class TestStateDictsOfTheBuiltInCallbacks:
    def test_the_base_class_has_no_state(self) -> None:
        callback = TrainerCallback()
        assert callback.stateDict() == {}
        callback.loadStateDict({"anything": 1})  # a no-op

    def test_early_stopping_round_trips(self) -> None:
        original = EarlyStopping(monitor="m", patience=5)
        trainer = Trainer(_buildModel(False), device="cpu")
        for epoch, value in enumerate([3.0, 2.0, 2.0, 2.0]):
            original.onEpochEnd(trainer, epoch, {"m": value})
        restored = EarlyStopping(monitor="m", patience=5)
        restored.loadStateDict(original.stateDict())
        assert restored.stateDict() == original.stateDict()
        assert restored.stateDict()["best"] == 2.0

    def test_reduce_lr_on_plateau_round_trips(self) -> None:
        original = ReduceLrOnPlateau(monitor="m", patience=5, cooldown=2)
        trainer = Trainer(_buildModel(False), device="cpu")
        for epoch, value in enumerate([3.0, 3.0, 3.0]):
            original.onEpochEnd(trainer, epoch, {"m": value})
        restored = ReduceLrOnPlateau(monitor="m", patience=5, cooldown=2)
        restored.loadStateDict(original.stateDict())
        assert restored.stateDict() == original.stateDict()

    def test_the_states_are_plain_values(self) -> None:
        """So that a checkpoint with them loads with `weights_only=True`."""
        for callback in (
            EarlyStopping(),
            ReduceLrOnPlateau(),
            BestCheckpointCallback("unused.pt"),
            CheckpointCallback("unused"),
            TopKCheckpointCallback("unused"),
        ):
            state = callback.stateDict()
            assert isinstance(state, dict)
            for value in state.values():
                assert isinstance(value, (int, float, str, list, tuple, dict, bool, type(None)))


class TestLoggersContinueTheirFileAfterAResume:
    def test_append_continues_a_file_without_a_second_header(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.csv"
        first = CsvLogger(path)
        first.logScalar("train/loss", 1.0, x=0)
        first.close()

        second = CsvLogger(path, append=True)
        second.logScalar("train/loss", 0.5, x=1)
        second.close()

        with path.open() as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == ["x", "tag", "metric", "value"]
        assert [row[0] for row in rows[1:]] == ["0", "1"]
        assert sum(1 for row in rows if row[0] == "x") == 1

    def test_append_on_a_missing_or_empty_file_writes_the_header(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.csv"
        CsvLogger(missing, append=True).close()
        assert missing.read_text().splitlines() == ["x,tag,metric,value"]

        empty = tmp_path / "empty.csv"
        empty.write_text("")
        CsvLogger(empty, append=True).close()
        assert empty.read_text().splitlines() == ["x,tag,metric,value"]

    def test_the_default_still_replaces_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.csv"
        first = CsvLogger(path)
        first.logScalar("train/loss", 1.0, x=0)
        first.close()
        CsvLogger(path).close()
        assert path.read_text().splitlines() == ["x,tag,metric,value"]

    def test_a_config_with_resume_from_builds_its_csv_logger_in_append_mode(
        self, tmp_path: Path
    ) -> None:
        metrics = tmp_path / "metrics.csv"
        metrics.write_text("x,tag,metric,value\n0,train,loss/total,1.0\n")
        config = TrainingConfig(
            loggers=[LoggerEntryConfig(name="csv", kwargs={"path": str(metrics)})],
            resume_from=str(tmp_path / "checkpoint.pt"),
        )
        (logger_instance,) = buildCallbacksFromConfig(config)
        logger_instance.close()  # type: ignore[attr-defined]
        assert metrics.read_text().count("x,tag,metric,value") == 1
        assert "0,train,loss/total,1.0" in metrics.read_text()

    def test_a_config_without_resume_from_keeps_replacing_it(self, tmp_path: Path) -> None:
        metrics = tmp_path / "metrics.csv"
        metrics.write_text("x,tag,metric,value\n0,train,loss/total,1.0\n")
        config = TrainingConfig(
            loggers=[LoggerEntryConfig(name="csv", kwargs={"path": str(metrics)})]
        )
        (logger_instance,) = buildCallbacksFromConfig(config)
        logger_instance.close()  # type: ignore[attr-defined]
        assert metrics.read_text().splitlines() == ["x,tag,metric,value"]

    def test_an_explicit_append_in_the_kwargs_wins(self, tmp_path: Path) -> None:
        metrics = tmp_path / "metrics.csv"
        metrics.write_text("x,tag,metric,value\n0,train,loss/total,1.0\n")
        config = TrainingConfig(
            loggers=[LoggerEntryConfig(name="csv", kwargs={"path": str(metrics), "append": False})],
            resume_from=str(tmp_path / "checkpoint.pt"),
        )
        (logger_instance,) = buildCallbacksFromConfig(config)
        logger_instance.close()  # type: ignore[attr-defined]
        assert metrics.read_text().splitlines() == ["x,tag,metric,value"]
