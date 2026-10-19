"""Integration tests for the learning-rate schedules `linear_warmup_lr` and `cosine_lr`.

Roadmap P2-3, `docs/adr/0033-top-k-checkpoints-and-learning-rate-schedules.md`. Both are
registered callbacks that set the rate of every parameter group at each optimizer step as
`initial_lr * factor(step)`. The value tests drive the callbacks by hand against a real
optimizer; the run tests check the effect through `Trainer.fit` (the rate each step trained
with is recorded by `_RecordLr`), through the metrics of ADR 0028 and across a resume.
"""

import logging
import math
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

from global_vae.config.training import TrainingConfig, buildCallbacksFromConfig
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.callbacks._lr_schedule import AbstractLrSchedule
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.checkpoint import CheckpointCallback
from global_vae.training.callbacks.cosine_lr import CosineLr
from global_vae.training.callbacks.linear_warmup_lr import LinearWarmupLr
from global_vae.training.callbacks.reduce_lr_on_plateau import ReduceLrOnPlateau
from global_vae.training.trainer import Trainer

INPUT_DIM = 8
LATENT_DIM = 2
BATCH_SIZE = 4
BASE_LR = 0.1


@registerEncoder("dummy_signal_encoder_lr_schedule_test")
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


@registerDecoder("dummy_signal_decoder_lr_schedule_test")
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


class _RecordLr(TrainerCallback):
    """Records the learning rate each optimizer step trained with.

    `onStepEnd` is called after the step, and the schedules (which sit before this callback in
    the list) set the rate of the *next* step in their own `onStepEnd`. So the rate seen here
    is the next one; the one the step used is the value seen on the previous call, and the
    first one is read in `onTrainBegin`.
    """

    def __init__(self) -> None:
        self.rates: list[float] = []

    def onTrainBegin(self, trainer: Trainer) -> None:
        self.rates.append(trainer.optimizer.param_groups[0]["lr"])

    def onStepEnd(self, trainer: Trainer, step: int, metrics: dict[str, float]) -> None:
        self.rates.append(trainer.optimizer.param_groups[0]["lr"])


def _trainer(callbacks: list[TrainerCallback] | None = None, **kwargs: object) -> Trainer:
    torch.manual_seed(0)
    model = GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_lr_schedule_test",
                "decoder": "dummy_signal_decoder_lr_schedule_test",
            }
        },
        latent_dim=LATENT_DIM,
    )
    return Trainer(
        model,
        device="cpu",
        optimizer_kwargs={"lr": BASE_LR},
        callbacks=callbacks,
        log_every_n_steps=1000,
        **kwargs,  # type: ignore[arg-type]
    )


def _data(num_batches: int = 3) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(1)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


class TestLinearWarmupFactor:
    def test_it_ramps_linearly_then_leaves_the_rate_alone(self) -> None:
        schedule = LinearWarmupLr(warmup_steps=4)
        assert [schedule.factor(step) for step in range(6)] == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
        assert [schedule.isActive(step) for step in range(6)] == [True] * 4 + [False] * 2

    @pytest.mark.parametrize("warmup_steps", [0, -1])
    def test_the_warmup_must_be_positive(self, warmup_steps: int) -> None:
        with pytest.raises(ValueError, match="warmup_steps"):
            LinearWarmupLr(warmup_steps=warmup_steps)


class TestCosineFactor:
    def test_it_decays_from_one_to_zero_along_a_half_cosine(self) -> None:
        schedule = CosineLr(total_steps=10)
        assert schedule.factor(0) == pytest.approx(1.0)
        assert schedule.factor(5) == pytest.approx(0.5)
        assert schedule.factor(10) == pytest.approx(0.0, abs=1e-12)
        for step in range(10):
            assert schedule.factor(step) == pytest.approx((1 + math.cos(math.pi * step / 10)) / 2)

    def test_it_stays_at_the_floor_after_the_last_step(self) -> None:
        schedule = CosineLr(total_steps=10, min_lr_ratio=0.2)
        assert schedule.factor(10) == pytest.approx(0.2)
        assert schedule.factor(500) == pytest.approx(0.2)

    def test_the_floor_is_a_fraction_of_the_initial_rate(self) -> None:
        schedule = CosineLr(total_steps=10, min_lr_ratio=0.25)
        assert schedule.factor(5) == pytest.approx(0.25 + 0.75 * 0.5)

    def test_the_warmup_is_linear_and_reaches_the_peak_where_the_cosine_starts(self) -> None:
        schedule = CosineLr(total_steps=10, warmup_steps=4)
        assert [schedule.factor(step) for step in range(4)] == [0.25, 0.5, 0.75, 1.0]
        assert schedule.factor(4) == pytest.approx(1.0)
        assert schedule.factor(7) == pytest.approx((1 + math.cos(math.pi * 3 / 6)) / 2)
        assert schedule.factor(10) == pytest.approx(0.0, abs=1e-12)

    def test_it_never_goes_up_after_the_warmup(self) -> None:
        schedule = CosineLr(total_steps=30, warmup_steps=5, min_lr_ratio=0.1)
        factors = [schedule.factor(step) for step in range(5, 40)]
        assert all(
            later <= earlier + 1e-12 for earlier, later in zip(factors, factors[1:], strict=False)
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"total_steps": 0}, "total_steps"),
            ({"total_steps": -3}, "total_steps"),
            ({"total_steps": 10, "warmup_steps": -1}, "warmup_steps"),
            ({"total_steps": 10, "warmup_steps": 10}, "warmup_steps"),
            ({"total_steps": 10, "min_lr_ratio": -0.1}, "min_lr_ratio"),
            ({"total_steps": 10, "min_lr_ratio": 1.1}, "min_lr_ratio"),
        ],
    )
    def test_invalid_arguments_are_refused(
        self, kwargs: dict[str, int | float], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            CosineLr(**kwargs)  # type: ignore[arg-type]


class TestInATrainingRun:
    def test_the_warmup_sets_the_rate_of_each_step_then_stops_touching_it(self) -> None:
        recorder = _RecordLr()
        trainer = _trainer([LinearWarmupLr(warmup_steps=4), recorder])
        trainer.fit(_data(3), num_epochs=2)

        # Rates of steps 0..5, then the (untouched) rate the step after the last would use.
        assert recorder.rates == pytest.approx(
            [0.25 * BASE_LR, 0.5 * BASE_LR, 0.75 * BASE_LR, BASE_LR, BASE_LR, BASE_LR, BASE_LR]
        )

    def test_the_cosine_ends_at_the_floor(self) -> None:
        recorder = _RecordLr()
        trainer = _trainer([CosineLr(total_steps=6, min_lr_ratio=0.1), recorder])
        trainer.fit(_data(3), num_epochs=2)
        assert recorder.rates[0] == pytest.approx(BASE_LR)
        assert recorder.rates[-1] == pytest.approx(0.1 * BASE_LR)
        assert recorder.rates == sorted(recorder.rates, reverse=True)

    def test_the_rate_is_a_function_of_the_optimizer_step_not_of_the_batch(self) -> None:
        """With accumulation over 2 batches, 4 batches make 2 steps, so 2 distinct rates."""
        recorder = _RecordLr()
        trainer = _trainer([CosineLr(total_steps=2), recorder], accumulate_grad_batches=2)
        trainer.fit(_data(4), num_epochs=1)
        assert recorder.rates == pytest.approx([BASE_LR, 0.5 * BASE_LR, 0.0], abs=1e-12)

    def test_every_parameter_group_is_scaled_from_its_own_initial_rate(self) -> None:
        torch.manual_seed(0)
        trainer = _trainer([CosineLr(total_steps=4)])
        encoder_parameters = list(trainer.model.encoders.parameters())
        decoder_parameters = list(trainer.model.decoders.parameters())
        trainer.optimizer = torch.optim.SGD(
            [
                {"params": encoder_parameters, "lr": 0.1},
                {"params": decoder_parameters, "lr": 0.01},
            ]
        )
        trainer.fit(_data(2), num_epochs=1)
        # Two steps of 4: the next step to run is the third, factor (1 + cos(pi * 2/4)) / 2.
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5)
        assert trainer.optimizer.param_groups[1]["lr"] == pytest.approx(0.01 * 0.5)

    def test_the_learning_rate_is_recorded_in_the_epoch_metrics(self) -> None:
        trainer = _trainer([CosineLr(total_steps=6)])
        history = trainer.fit(_data(3), num_epochs=2)
        assert "train/lr" in history[0]
        assert history[1]["train/lr"] < history[0]["train/lr"]

    def test_without_a_schedule_no_learning_rate_is_recorded(self) -> None:
        history = _trainer().fit(_data(2), num_epochs=1)
        assert "train/lr" not in history[0]

    def test_the_initial_rate_is_kept_in_the_optimizer_groups(self) -> None:
        trainer = _trainer([CosineLr(total_steps=6)])
        trainer.fit(_data(3), num_epochs=1)
        assert trainer.optimizer.param_groups[0]["initial_lr"] == BASE_LR

    def test_a_warmup_leaves_the_rate_to_a_plateau_reduction_afterwards(self) -> None:
        reducer = ReduceLrOnPlateau(
            monitor="train/loss/total", factor=0.5, patience=0, threshold=10.0
        )
        trainer = _trainer([LinearWarmupLr(warmup_steps=2), reducer])
        trainer.fit(_data(1), num_epochs=6)
        # The warm-up is over after 2 steps, and the plateau reduction has lowered the rate since.
        assert trainer.optimizer.param_groups[0]["lr"] < BASE_LR


class TestCombiningWithAnotherLrCallback:
    def test_a_cosine_warns_that_another_lr_callback_will_be_overwritten(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        trainer = _trainer([ReduceLrOnPlateau(monitor="train/loss/total"), CosineLr(total_steps=6)])
        with caplog.at_level(logging.WARNING):
            trainer.fit(_data(1), num_epochs=1)
        assert "ReduceLrOnPlateau" in caplog.text
        assert "overwritten" in caplog.text

    def test_a_cosine_alone_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        trainer = _trainer([CosineLr(total_steps=6)])
        with caplog.at_level(logging.WARNING):
            trainer.fit(_data(1), num_epochs=1)
        assert "overwritten" not in caplog.text

    def test_two_warmups_are_not_a_problem_for_the_cosine_check(self) -> None:
        assert CosineLr.acts_on_learning_rate is True
        assert LinearWarmupLr.acts_on_learning_rate is True
        assert ReduceLrOnPlateau.acts_on_learning_rate is True


class TestResume:
    @pytest.mark.parametrize(
        "make_schedule",
        [lambda: CosineLr(total_steps=12, warmup_steps=3), lambda: LinearWarmupLr(warmup_steps=7)],
        ids=["cosine", "warmup"],
    )
    def test_a_resumed_run_continues_the_schedule_mid_way(
        self, tmp_path: Path, make_schedule: Callable[[], AbstractLrSchedule]
    ) -> None:
        data = _data(3)
        full_recorder = _RecordLr()
        full = _trainer([make_schedule(), full_recorder])
        full.fit(data, num_epochs=4)

        first = _trainer([make_schedule(), CheckpointCallback(tmp_path)])
        first.fit(data, num_epochs=2)

        recorder = _RecordLr()
        resumed = _trainer([make_schedule(), recorder])
        resumed.loadCheckpoint(tmp_path / "checkpoint_epoch_0001.pt")
        # The checkpoint holds the rate of its own last step: the schedule must not scale that.
        resumed.fit(data, num_epochs=2)

        assert recorder.rates == pytest.approx(full_recorder.rates[6:])
        assert resumed.optimizer.param_groups[0]["initial_lr"] == BASE_LR

    def test_the_schedule_scales_the_initial_rate_not_the_rate_in_the_checkpoint(
        self, tmp_path: Path
    ) -> None:
        first = _trainer([CosineLr(total_steps=8), CheckpointCallback(tmp_path)])
        first.fit(_data(2), num_epochs=2)
        saved = torch.load(
            tmp_path / "checkpoint_epoch_0001.pt", map_location="cpu", weights_only=True
        )
        assert saved["optimizer_state_dict"]["param_groups"][0]["initial_lr"] == BASE_LR
        assert saved["optimizer_state_dict"]["param_groups"][0]["lr"] < BASE_LR


class TestBuiltFromTheConfig:
    def test_both_are_selected_by_name(self) -> None:
        config = TrainingConfig(
            callbacks={
                "cosine_lr": {"total_steps": 10, "warmup_steps": 2, "min_lr_ratio": 0.1},
            }
        )
        (schedule,) = buildCallbacksFromConfig(config)
        assert isinstance(schedule, CosineLr)
        assert (schedule.total_steps, schedule.warmup_steps, schedule.min_lr_ratio) == (10, 2, 0.1)

        config = TrainingConfig(callbacks={"linear_warmup_lr": {"warmup_steps": 5}})
        (warmup,) = buildCallbacksFromConfig(config)
        assert isinstance(warmup, LinearWarmupLr)
        assert warmup.warmup_steps == 5
