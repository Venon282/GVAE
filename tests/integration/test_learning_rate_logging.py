"""Integration tests: the learning rate in `Trainer`'s metrics (ADR 0028).

When a callback acts on the learning rate (`TrainerCallback.acts_on_learning_rate`), `Trainer`
records the rate each step trained with as `"lr"`, and every epoch carries its mean as
`"train/lr"`. These tests drive a real `Trainer.fit` and check what that gives: the epoch log
line, `Trainer.history`, the step metrics every callback receives, a CSV logger, the checkpoint
history that `scripts/visualize_latent.py` plots, parameter groups, a callback that changes the
rate every step, and a run built from a config file. A run with no such callback must look exactly
as it did before.

`ReduceLrOnPlateau` is fed a hand-picked, constant metric (`_ForceMetric`, placed before it in
`Trainer.callbacks`) so the epoch of each reduction is known in advance and does not depend on the
dummy model's loss: with `patience=1` and a constant metric the rate is divided at the end of
epochs 2 and 4, so six epochs train with `1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5`.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from global_vae.config.training import TrainingConfig, buildTrainerFromConfig
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.reduce_lr_on_plateau import ReduceLrOnPlateau
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.training.loggers.csv_logger import CsvLogger
from global_vae.training.trainer import Trainer
from tests.integration._script_fixtures import buildDataloaderForScript, buildModelForScript

BASE_LR = 1e-3
NUM_EPOCHS = 6
STEPS_PER_EPOCH = 2  # `buildDataloaderForScript` yields two batches
REDUCED_LRS = [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5]
TRAINER_LOGGER = "global_vae.training.trainer"


class _ForceMetric(TrainerCallback):
    """Overwrites one epoch-metrics key with a constant, before `ReduceLrOnPlateau` reads it."""

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value

    def onEpochEnd(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        metrics[self.key] = self.value


class _RecordSteps(TrainerCallback):
    """Keeps the metrics dict `Trainer` hands to `onStepEnd`, one per step."""

    def __init__(self) -> None:
        self.steps: list[dict[str, float]] = []

    def onStepEnd(self, trainer: Trainer, step: int, metrics: dict[str, float]) -> None:
        self.steps.append(dict(metrics))


class _DecayEveryStep(TrainerCallback):
    """Lowers every parameter group's rate by a fixed amount after each step."""

    acts_on_learning_rate = True

    def __init__(self, delta: float) -> None:
        self.delta = delta

    def onStepEnd(self, trainer: Trainer, step: int, metrics: dict[str, float]) -> None:
        for group in trainer.optimizer.param_groups:
            group["lr"] -= self.delta


def _plateauCallbacks() -> list[TrainerCallback]:
    return [_ForceMetric("val/loss/total", 1.0), ReduceLrOnPlateau(patience=1, factor=0.1)]


def _trainer(callbacks: list[TrainerCallback] | None = None, **kwargs: Any) -> Trainer:
    kwargs.setdefault("optimizer_kwargs", {"lr": BASE_LR})
    return Trainer(
        buildModelForScript(), device="cpu", callbacks=callbacks, log_every_n_steps=1, **kwargs
    )


def _fit(trainer: Trainer, num_epochs: int = NUM_EPOCHS) -> list[dict[str, float]]:
    return trainer.fit(buildDataloaderForScript(), num_epochs=num_epochs)


@dataclass
class _PlateauRun:
    """Six epochs with `ReduceLrOnPlateau`, and what they left behind.

    Attributes:
        trainer: The trainer after `fit`.
        recorder: The step metrics every callback received.
        messages: Every log message of the run, in order.
    """

    trainer: Trainer
    recorder: _RecordSteps
    messages: list[str]

    @property
    def epoch_lines(self) -> list[str]:
        return [m for m in self.messages if m.startswith("epoch ") and "complete" in m]

    @property
    def step_lines(self) -> list[str]:
        return [m for m in self.messages if m.startswith("epoch ") and " step " in m]


@pytest.fixture
def plateau_run(caplog: pytest.LogCaptureFixture) -> _PlateauRun:
    """Six epochs with `ReduceLrOnPlateau` (a reduction at the end of epochs 2 and 4)."""
    recorder = _RecordSteps()
    trainer = _trainer([*_plateauCallbacks(), recorder])
    with caplog.at_level(logging.INFO):
        _fit(trainer)
    return _PlateauRun(trainer, recorder, list(caplog.messages))


class TestWithoutALearningRateCallback:
    """No callback acts on the rate: nothing is added, so such a run is as before."""

    def test_the_trainer_does_not_record_it(self) -> None:
        assert _trainer().logs_learning_rate is False
        assert _trainer([_RecordSteps()]).logs_learning_rate is False

    def test_history_and_step_metrics_have_no_learning_rate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder = _RecordSteps()
        trainer = _trainer([recorder])
        with caplog.at_level(logging.INFO, logger=TRAINER_LOGGER):
            history = _fit(trainer, num_epochs=2)
        assert [set(entry) for entry in history] == [
            {"train/loss/total", "train/loss/reconstruction", "train/loss/regularization"}
        ] * 2
        assert all(
            set(step) == {"loss/total", "loss/reconstruction", "loss/regularization"}
            for step in recorder.steps
        )
        assert "lr" not in caplog.text


class TestReduceLrOnPlateauRun:
    """`ReduceLrOnPlateau` acts on the rate, so the run records it every epoch."""

    def test_the_trainer_records_it(self, plateau_run: _PlateauRun) -> None:
        assert plateau_run.trainer.logs_learning_rate is True

    def test_every_epoch_of_the_history_has_the_rate_it_trained_with(
        self, plateau_run: _PlateauRun
    ) -> None:
        """A reduction decided at the end of epoch `n` first shows in epoch `n + 1`."""
        trainer = plateau_run.trainer
        assert [entry["train/lr"] for entry in trainer.history] == pytest.approx(REDUCED_LRS)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    def test_the_rate_is_the_only_new_key(self, plateau_run: _PlateauRun) -> None:
        trainer = plateau_run.trainer
        for entry in trainer.history:
            assert set(entry) == {
                "train/loss/total",
                "train/loss/reconstruction",
                "train/loss/regularization",
                "train/lr",
                "val/loss/total",
            }

    def test_every_step_has_the_rate_it_trained_with(self, plateau_run: _PlateauRun) -> None:
        recorder = plateau_run.recorder
        assert len(recorder.steps) == NUM_EPOCHS * STEPS_PER_EPOCH
        expected = [lr for lr in REDUCED_LRS for _ in range(STEPS_PER_EPOCH)]
        assert [step["lr"] for step in recorder.steps] == pytest.approx(expected)

    def test_the_epoch_log_line_shows_the_rate(self, plateau_run: _PlateauRun) -> None:
        lines = plateau_run.epoch_lines
        assert len(lines) == NUM_EPOCHS
        assert [line.rsplit("train/lr=", 1)[1].split(",")[0] for line in lines] == [
            "0.001",
            "0.001",
            "0.001",
            "0.0001",
            "0.0001",
            "1e-05",
        ]

    def test_a_small_rate_is_not_rounded_to_zero_in_the_log(self, plateau_run: _PlateauRun) -> None:
        """Losses print with four decimals, which would turn `1e-05` into `0.0000`."""
        text = "\n".join(plateau_run.messages)
        assert "train/lr=1e-05" in text
        assert "train/lr=0.0000" not in text
        assert "train/loss/total=" in text

    def test_the_step_log_line_shows_the_rate(self, plateau_run: _PlateauRun) -> None:
        step_lines = plateau_run.step_lines
        assert len(step_lines) == NUM_EPOCHS * STEPS_PER_EPOCH
        assert step_lines[0].endswith("lr=0.001")
        assert step_lines[-1].endswith("lr=1e-05")

    def test_the_reduction_itself_is_still_logged(self, plateau_run: _PlateauRun) -> None:
        assert any("reducing learning rate from 0.001 to 0.0001" in m for m in plateau_run.messages)

    def test_the_monitored_metric_and_the_other_losses_are_untouched(
        self, plateau_run: _PlateauRun
    ) -> None:
        trainer = plateau_run.trainer
        assert all(entry["val/loss/total"] == 1.0 for entry in trainer.history)
        assert all(entry["train/loss/total"] > 0 for entry in trainer.history)


class TestWhereTheRateGoes:
    """It is an ordinary metric, so every consumer of the metrics gets it."""

    def test_a_csv_logger_writes_a_row_per_step_and_per_epoch(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.csv"
        trainer = _trainer([*_plateauCallbacks(), CsvLogger(path)])
        _fit(trainer)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        epoch_rows = [r for r in rows if r["tag"] == "epoch" and r["metric"] == "train/lr"]
        step_rows = [r for r in rows if r["tag"] == "step" and r["metric"] == "lr"]
        assert [float(r["value"]) for r in epoch_rows] == pytest.approx(REDUCED_LRS)
        assert [int(r["x"]) for r in epoch_rows] == list(range(NUM_EPOCHS))
        assert len(step_rows) == NUM_EPOCHS * STEPS_PER_EPOCH

    def test_a_checkpoint_keeps_it_in_its_history(self, tmp_path: Path) -> None:
        """This is what `scripts/visualize_latent.py` reads to draw the learning-rate plot."""
        trainer = _trainer(_plateauCallbacks())
        _fit(trainer)
        path = tmp_path / "model.pt"
        trainer.saveCheckpoint(path)
        metadata = loadCheckpoint(path, model=buildModelForScript(), map_location="cpu")
        assert [entry["train/lr"] for entry in metadata.history] == pytest.approx(REDUCED_LRS)

    def test_resuming_keeps_recording_it(self) -> None:
        trainer = _trainer(_plateauCallbacks())
        _fit(trainer, num_epochs=3)
        _fit(trainer, num_epochs=3)
        assert len(trainer.history) == 6
        assert "train/lr" in trainer.history[-1]


class TestOtherWaysToActOnTheRate:
    """The declaration is on the callback class, so a callback of your own can make it."""

    def test_a_custom_callback_that_declares_it_turns_the_recording_on(self) -> None:
        assert _trainer([_DecayEveryStep(1e-5)]).logs_learning_rate is True

    def test_a_rate_changed_every_step_is_recorded_per_step_and_averaged_per_epoch(self) -> None:
        recorder = _RecordSteps()
        trainer = _trainer([_DecayEveryStep(1e-4), recorder])
        history = _fit(trainer, num_epochs=2)
        step_rates = [step["lr"] for step in recorder.steps]
        assert step_rates == pytest.approx([1e-3, 9e-4, 8e-4, 7e-4])
        assert [entry["train/lr"] for entry in history] == pytest.approx([9.5e-4, 7.5e-4])

    def test_the_declaration_is_read_when_the_epoch_starts(self) -> None:
        trainer = _trainer()
        trainer.callbacks.append(_DecayEveryStep(0.0))
        assert trainer.logs_learning_rate is True
        assert "train/lr" in _fit(trainer, num_epochs=1)[0]

    def test_several_parameter_groups_get_one_entry_each(self) -> None:
        model = buildModelForScript()
        encoder_params = list(model.encoders.parameters())
        other_params = [
            p for name, p in model.named_parameters() if not name.startswith("encoders.")
        ]
        optimizer = torch.optim.Adam(
            [{"params": encoder_params, "lr": 1e-2}, {"params": other_params, "lr": 1e-3}]
        )
        trainer = Trainer(model, optimizer=optimizer, device="cpu", callbacks=_plateauCallbacks())
        history = trainer.fit(buildDataloaderForScript(), num_epochs=NUM_EPOCHS)
        assert "train/lr" not in history[0]
        assert [e["train/lr/group0"] for e in history] == pytest.approx(
            [lr * 10 for lr in REDUCED_LRS]
        )
        assert [e["train/lr/group1"] for e in history] == pytest.approx(REDUCED_LRS)


class TestBuiltFromConfig:
    """A `reduce_lr_on_plateau` entry in `TrainingConfig.callbacks` is enough."""

    def test_the_trainer_built_from_a_config_records_the_rate(self) -> None:
        config = TrainingConfig(
            num_epochs=3, callbacks={"reduce_lr_on_plateau": {"monitor": "train/loss/total"}}
        )
        trainer = buildTrainerFromConfig(buildModelForScript(), config)
        assert trainer.logs_learning_rate is True
        history = trainer.fit(buildDataloaderForScript(), num_epochs=2)
        assert all("train/lr" in entry for entry in history)

    def test_a_config_without_it_does_not(self) -> None:
        trainer = buildTrainerFromConfig(buildModelForScript(), TrainingConfig(num_epochs=3))
        assert trainer.logs_learning_rate is False
