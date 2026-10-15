"""Integration tests for `visualization.loss_curves` and `visualization.history_callback`.

Uses its own trivial linear dummy encoder/decoder registered under a
`_loss_curves_test` suffix for the end-to-end `Trainer` tests (see
`test_trainer.py`'s module docstring for why test files do not import
each other's fixtures).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedules.constant import ConstantBetaSchedule
from global_vae.training.beta_schedules.linear_warmup import LinearWarmupBetaSchedule
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.reduce_lr_on_plateau import ReduceLrOnPlateau
from global_vae.training.trainer import Trainer
from global_vae.visualization.history_callback import HistoryCallback
from global_vae.visualization.loss_curves import (
    plotBetaSchedule,
    plotLearningRate,
    plotLossCurves,
    plotStepCurves,
)

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 6


@registerEncoder("dummy_signal_encoder_loss_curves_test")
class _DummySignalEncoder(AbstractEncoder):
    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self.to_mu = nn.Linear(input_dim, latent_dim)
        self.to_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.to_mu(x), self.to_logvar(x)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        return "signal"

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerDecoder("dummy_signal_decoder_loss_curves_test")
class _DummySignalDecoder(AbstractDecoder):
    def __init__(self, output_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.project = nn.Linear(latent_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return "signal"


def _buildModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_loss_curves_test",
                "decoder": "dummy_signal_decoder_loss_curves_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedDataset(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


class TestPlotLossCurves:
    def test_default_metrics_selects_every_train_and_val_loss_key(self) -> None:
        history = [
            {
                "train/loss/total": 1.0,
                "train/loss/reconstruction": 0.6,
                "train/loss/regularization": 0.4,
            },
            {
                "train/loss/total": 0.8,
                "train/loss/reconstruction": 0.5,
                "train/loss/regularization": 0.3,
            },
        ]
        fig = plotLossCurves(history)
        assert len(fig.axes[0].lines) == 3
        plt.close(fig)

    def test_explicit_metrics_are_used(self) -> None:
        history = [{"train/loss/total": 1.0, "other": 5.0}, {"train/loss/total": 0.5, "other": 4.0}]
        fig = plotLossCurves(history, metrics=["other"])
        line = fig.axes[0].lines[0]
        assert np.asarray(line.get_ydata()).tolist() == [5.0, 4.0]
        plt.close(fig)

    def test_metric_present_only_in_later_epochs_only_plots_those_points(self) -> None:
        history = [
            {"train/loss/total": 1.0},
            {"train/loss/total": 0.8, "val/loss/total": 0.9},
            {"train/loss/total": 0.6, "val/loss/total": 0.7},
        ]
        fig = plotLossCurves(history)
        val_line = next(line for line in fig.axes[0].lines if line.get_label() == "val/loss/total")
        assert np.asarray(val_line.get_xdata()).tolist() == [1, 2]
        plt.close(fig)

    def test_log_scale_does_not_raise(self) -> None:
        history = [{"train/loss/total": 1.0}, {"train/loss/total": 0.5}]
        fig = plotLossCurves(history, log_scale=True)
        assert fig.axes[0].get_yscale() == "log"
        plt.close(fig)

    def test_empty_history_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plotLossCurves([])

    def test_no_matching_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="None of the requested"):
            plotLossCurves([{"unrelated_key": 1.0}])

    def test_end_to_end_with_trainer_history(self) -> None:
        model = _buildModel()
        trainer = Trainer(model, device="cpu")
        trainer.fit(_fixedDataset(2), num_epochs=3)
        fig = plotLossCurves(trainer.history)
        assert fig is not None
        plt.close(fig)


class TestPlotLearningRate:
    """`plotLearningRate` draws the `train/lr` entries a run records when a callback acts on it."""

    HISTORY = [
        {"train/loss/total": 1.0, "train/lr": 1e-3},
        {"train/loss/total": 0.9, "train/lr": 1e-3},
        {"train/loss/total": 0.8, "train/lr": 1e-4},
    ]

    def test_plots_the_rate_of_every_epoch(self) -> None:
        fig = plotLearningRate(self.HISTORY)
        (line,) = fig.axes[0].lines
        assert line.get_label() == "train/lr"
        assert np.asarray(line.get_xdata()).tolist() == [0, 1, 2]
        assert np.asarray(line.get_ydata()).tolist() == [1e-3, 1e-3, 1e-4]
        assert fig.axes[0].get_xlabel() == "epoch"
        assert fig.axes[0].get_ylabel() == "learning rate"
        plt.close(fig)

    def test_the_loss_keys_are_not_drawn(self) -> None:
        fig = plotLearningRate(self.HISTORY)
        assert [line.get_label() for line in fig.axes[0].lines] == ["train/lr"]
        plt.close(fig)

    def test_the_axis_is_logarithmic_by_default_and_can_be_linear(self) -> None:
        log_fig = plotLearningRate(self.HISTORY)
        linear_fig = plotLearningRate(self.HISTORY, log_scale=False)
        assert log_fig.axes[0].get_yscale() == "log"
        assert linear_fig.axes[0].get_yscale() == "linear"
        plt.close(log_fig)
        plt.close(linear_fig)

    def test_every_parameter_group_gets_a_line(self) -> None:
        history = [
            {"train/lr/group1": 1e-3, "train/lr/group0": 1e-2},
            {"train/lr/group1": 1e-4, "train/lr/group0": 1e-3},
        ]
        fig = plotLearningRate(history)
        assert [line.get_label() for line in fig.axes[0].lines] == [
            "train/lr/group0",
            "train/lr/group1",
        ]
        plt.close(fig)

    def test_explicit_keys_are_used(self) -> None:
        history = [{"train/lr/group0": 1e-2, "train/lr/group1": 1e-3}]
        fig = plotLearningRate(history, keys=["train/lr/group1"])
        assert [line.get_label() for line in fig.axes[0].lines] == ["train/lr/group1"]
        plt.close(fig)

    def test_a_key_present_only_in_later_epochs_only_plots_those_points(self) -> None:
        history = [{"train/loss/total": 1.0}, {"train/loss/total": 0.9, "train/lr": 1e-3}]
        fig = plotLearningRate(history)
        assert np.asarray(fig.axes[0].lines[0].get_xdata()).tolist() == [1]
        plt.close(fig)

    def test_empty_history_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plotLearningRate([])

    def test_a_history_without_a_learning_rate_says_why(self) -> None:
        """A run with no callback acting on the rate records none, so there is nothing to draw."""
        with pytest.raises(ValueError, match="acts_on_learning_rate"):
            plotLearningRate([{"train/loss/total": 1.0}])

    def test_the_loss_plot_does_not_draw_the_rate(self) -> None:
        """The rate is on a scale of its own: `plotLossCurves` keeps to the loss keys."""
        fig = plotLossCurves(self.HISTORY)
        assert [line.get_label() for line in fig.axes[0].lines] == ["train/loss/total"]
        plt.close(fig)

    def test_end_to_end_with_a_run_that_reduces_the_rate(self) -> None:
        class _Flat(TrainerCallback):
            def onEpochEnd(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
                metrics["val/loss/total"] = 1.0

        trainer = Trainer(
            _buildModel(),
            device="cpu",
            optimizer_kwargs={"lr": 1e-3},
            callbacks=[_Flat(), ReduceLrOnPlateau(patience=1, factor=0.1)],
        )
        trainer.fit(_fixedDataset(2), num_epochs=4)
        fig = plotLearningRate(trainer.history)
        (line,) = fig.axes[0].lines
        assert np.asarray(line.get_ydata()).tolist() == pytest.approx([1e-3, 1e-3, 1e-3, 1e-4])
        plt.close(fig)


class TestPlotStepCurves:
    def test_uses_the_step_key_not_the_list_index(self) -> None:
        step_history = [{"step": 10, "loss/total": 1.0}, {"step": 20, "loss/total": 0.5}]
        fig = plotStepCurves(step_history)
        line = fig.axes[0].lines[0]
        assert np.asarray(line.get_xdata()).tolist() == [10, 20]
        plt.close(fig)

    def test_default_metrics_excludes_the_step_key_itself(self) -> None:
        step_history = [{"step": 0, "a": 1.0, "b": 2.0}]
        fig = plotStepCurves(step_history)
        labels = {line.get_label() for line in fig.axes[0].lines}
        assert labels == {"a", "b"}
        plt.close(fig)

    def test_missing_step_key_raises(self) -> None:
        with pytest.raises(ValueError, match="step"):
            plotStepCurves([{"loss/total": 1.0}])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plotStepCurves([])

    def test_no_matching_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="None of the requested"):
            plotStepCurves([{"step": 0}], metrics=["does_not_exist"])


class TestPlotBetaSchedule:
    def test_single_schedule_values_match_direct_calls(self) -> None:
        schedule = LinearWarmupBetaSchedule(warmup_steps=10, start_value=0.0, end_value=1.0)
        fig = plotBetaSchedule(schedule, num_steps=20)
        line = fig.axes[0].lines[0]
        expected = [schedule(step) for step in range(20)]
        assert np.asarray(line.get_ydata()).tolist() == pytest.approx(expected)
        plt.close(fig)

    def test_dict_of_schedules_produces_one_line_each(self) -> None:
        schedules = {
            "z_shared": LinearWarmupBetaSchedule(warmup_steps=10, start_value=0.0, end_value=1.0),
            "z_private": ConstantBetaSchedule(value=0.1),
        }
        fig = plotBetaSchedule(schedules, num_steps=15)
        assert len(fig.axes[0].lines) == 2
        plt.close(fig)

    def test_non_positive_num_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="num_steps"):
            plotBetaSchedule(ConstantBetaSchedule(), num_steps=0)


class TestHistoryCallback:
    def test_records_step_and_epoch_metrics_with_the_right_keys(self) -> None:
        model = _buildModel()
        history_callback = HistoryCallback()
        trainer = Trainer(model, device="cpu", callbacks=[history_callback])
        trainer.fit(_fixedDataset(2), num_epochs=2)

        assert len(history_callback.step_history) == 4  # 2 batches * 2 epochs
        assert len(history_callback.epoch_history) == 2
        for entry in history_callback.step_history:
            assert "step" in entry and "loss/total" in entry
        for entry in history_callback.epoch_history:
            assert "epoch" in entry and "train/loss/total" in entry

    def test_step_history_feeds_plot_step_curves_end_to_end(self) -> None:
        model = _buildModel()
        history_callback = HistoryCallback()
        trainer = Trainer(model, device="cpu", callbacks=[history_callback])
        trainer.fit(_fixedDataset(2), num_epochs=2)

        fig = plotStepCurves(history_callback.step_history)
        assert fig is not None
        plt.close(fig)
