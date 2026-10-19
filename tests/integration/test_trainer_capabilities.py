"""Smoke tests of the opt-in trainer capabilities (roadmap P2-3, ADRs 0031 and 0032).

Mixed precision, gradient accumulation and `torch.compile`: each is off by default, and each
gets a few optimizer steps on dummy data here, in the style of `test_trainer_smoke.py` (the loss
goes down, no gradient is `None`), plus the property that makes the option safe to turn on:

- the default path is unchanged (an explicit `None`/`1`/`False` is the same run as no argument);
- accumulation over `N` batches is one optimizer step on the mean of their gradients;
- a bf16/fp16 run keeps a float32 loss and float32 reconstructions for the loss functions;
- a compiled model trains the same and saves checkpoints with the same keys.

Only the CPU is assumed: `torch.autocast("cpu", dtype=torch.bfloat16/float16)` and
`torch.amp.GradScaler("cpu")` exist from PyTorch 2.3, and the package requires 2.2 or later,
so a test that needs the scaler skips itself on an older PyTorch. Uses its own dummy
encoder/decoder, registered under a `_capabilities_test` suffix (see `test_trainer.py` for why
sibling test files do not share fixtures).
"""

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.checkpoint import saveCheckpoint
from global_vae.training.trainer import SUPPORTED_MIXED_PRECISIONS, Trainer
from global_vae.utils.autograd import backward

INPUT_DIM = 20
LATENT_DIM = 4
HIDDEN_DIM = 16
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_capabilities_test")
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


@registerDecoder("dummy_signal_decoder_capabilities_test")
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


@registerEncoder("dummy_other_encoder_capabilities_test")
class _DummyOtherEncoder(_DummySignalEncoder):
    @property
    def modality_name(self) -> str:
        return "other"


@registerDecoder("dummy_other_decoder_capabilities_test")
class _DummyOtherDecoder(_DummySignalDecoder):
    @property
    def modality_name(self) -> str:
        return "other"


def _buildModel(seed: int = 0) -> GlobalVae:
    torch.manual_seed(seed)
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_capabilities_test",
                "decoder": "dummy_signal_decoder_capabilities_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _buildTwoModalityModel(seed: int = 0) -> GlobalVae:
    torch.manual_seed(seed)
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_capabilities_test",
                "decoder": "dummy_signal_decoder_capabilities_test",
            },
            "other": {
                "encoder": "dummy_other_encoder_capabilities_test",
                "decoder": "dummy_other_decoder_capabilities_test",
            },
        },
        latent_dim=LATENT_DIM,
        fusion_strategy="poe",
    )


def _fixedBatches(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


def _fixedTwoModalityBatches(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [
        {"signal": torch.randn(BATCH_SIZE, INPUT_DIM), "other": torch.randn(BATCH_SIZE, INPUT_DIM)}
        for _ in range(num_batches)
    ]


def _needsCpuGradScaler() -> None:
    try:
        torch.amp.GradScaler("cpu")
    except (AttributeError, RuntimeError, ValueError):  # pragma: no cover - old PyTorch
        pytest.skip("torch.amp.GradScaler does not support the CPU in this PyTorch")


def _lossesDecrease(history: list[dict[str, float]]) -> bool:
    first = sum(entry["train/loss/total"] for entry in history[:3]) / 3
    last = sum(entry["train/loss/total"] for entry in history[-3:]) / 3
    return last < first


class _RecordingBeta(AbstractBetaSchedule):
    """A beta schedule that records every step it is asked about."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def __call__(self, step: int) -> float:
        self.steps.append(step)
        return 1.0


class _RecordStepMetrics(TrainerCallback):
    """Records (step, metrics) of every optimizer step."""

    def __init__(self) -> None:
        self.steps: list[tuple[int, dict[str, float]]] = []

    def onStepEnd(self, trainer: Trainer, step: int, metrics: dict[str, float]) -> None:
        self.steps.append((step, dict(metrics)))


class TestOffByDefault:
    def test_the_defaults_are_the_documented_ones(self) -> None:
        trainer = Trainer(_buildModel(), device="cpu")
        assert trainer.mixed_precision is None
        assert trainer.accumulate_grad_batches == 1
        assert trainer.scaler is None
        assert trainer.model is trainer._forward

    def test_naming_the_defaults_is_the_same_run_as_not_naming_them(self) -> None:
        batches = _fixedBatches(6)
        plain = Trainer(_buildModel(), device="cpu", optimizer_kwargs={"lr": 0.05})
        named = Trainer(
            _buildModel(),
            device="cpu",
            optimizer_kwargs={"lr": 0.05},
            mixed_precision=None,
            accumulate_grad_batches=1,
            compile_model=False,
            compile_kwargs=None,
        )
        torch.manual_seed(1)
        history_plain = plain.fit(batches, num_epochs=3)
        torch.manual_seed(1)
        history_named = named.fit(batches, num_epochs=3)
        assert history_plain == history_named
        for name, parameter in plain.model.state_dict().items():
            assert torch.equal(parameter, named.model.state_dict()[name])

    def test_the_supported_precisions_are_listed(self) -> None:
        assert SUPPORTED_MIXED_PRECISIONS == ("bf16", "fp16")


class TestConstructorValidation:
    @pytest.mark.parametrize("value", ["fp32", "BF16", "half", "", True, 16])
    def test_an_unsupported_precision_is_refused_and_the_supported_ones_are_named(
        self, value: Any
    ) -> None:
        with pytest.raises(ValueError, match=r"mixed_precision.*bf16.*fp16"):
            Trainer(_buildModel(), device="cpu", mixed_precision=value)

    @pytest.mark.parametrize("value", [0, -1])
    def test_accumulate_grad_batches_must_be_positive(self, value: int) -> None:
        with pytest.raises(ValueError, match="accumulate_grad_batches"):
            Trainer(_buildModel(), device="cpu", accumulate_grad_batches=value)

    def test_an_empty_dataloader_is_a_clear_error(self) -> None:
        trainer = Trainer(_buildModel(), device="cpu")
        with pytest.raises(ValueError, match="empty dataloader"):
            trainer.fit([], num_epochs=1)


class TestMixedPrecision:
    @pytest.mark.parametrize("precision", SUPPORTED_MIXED_PRECISIONS)
    def test_the_loss_decreases_and_no_gradient_is_none(self, precision: str) -> None:
        if precision == "fp16":
            _needsCpuGradScaler()
        trainer = Trainer(
            _buildModel(), device="cpu", optimizer_kwargs={"lr": 0.05}, mixed_precision=precision
        )
        history = trainer.fit(_fixedBatches(8), num_epochs=15)

        assert all(math.isfinite(entry["train/loss/total"]) for entry in history)
        assert _lossesDecrease(history)
        for name, parameter in trainer.model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    @pytest.mark.parametrize("precision", SUPPORTED_MIXED_PRECISIONS)
    def test_the_losses_are_computed_in_float32_and_close_to_the_full_precision_ones(
        self, precision: str
    ) -> None:
        if precision == "fp16":
            _needsCpuGradScaler()
        batch = _fixedBatches(1)[0]
        reference = Trainer(_buildModel(), device="cpu")
        lowered = Trainer(_buildModel(), device="cpu", mixed_precision=precision)

        expected = reference.computeLosses(batch, step=0)
        observed = lowered.computeLosses(batch, step=0)

        assert observed.total.dtype == torch.float32
        assert observed.reconstruction.dtype == torch.float32
        assert observed.regularization.dtype == torch.float32
        assert observed.total.item() == pytest.approx(expected.total.item(), rel=5e-2)

    def test_the_parameters_stay_in_float32(self) -> None:
        trainer = Trainer(_buildModel(), device="cpu", mixed_precision="bf16")
        trainer.fit(_fixedBatches(2), num_epochs=1)
        assert {p.dtype for p in trainer.model.parameters()} == {torch.float32}

    def test_only_fp16_has_a_scaler(self) -> None:
        assert Trainer(_buildModel(), device="cpu", mixed_precision="bf16").scaler is None
        _needsCpuGradScaler()
        assert Trainer(_buildModel(), device="cpu", mixed_precision="fp16").scaler is not None

    def test_the_scaler_state_is_saved_and_restored(self, tmp_path: Path) -> None:
        _needsCpuGradScaler()
        trainer = Trainer(_buildModel(), device="cpu", mixed_precision="fp16")
        trainer.fit(_fixedBatches(2), num_epochs=1)
        assert trainer.scaler is not None
        trainer.scaler.update(new_scale=1234.0)
        path = tmp_path / "fp16.pt"
        trainer.saveCheckpoint(path)

        restored = Trainer(_buildModel(1), device="cpu", mixed_precision="fp16")
        restored.loadCheckpoint(path)
        assert restored.scaler is not None
        assert restored.scaler.get_scale() == 1234.0

    def test_a_bf16_checkpoint_loads_into_a_full_precision_trainer(self, tmp_path: Path) -> None:
        trainer = Trainer(_buildModel(), device="cpu", mixed_precision="bf16")
        trainer.fit(_fixedBatches(2), num_epochs=1)
        path = tmp_path / "bf16.pt"
        trainer.saveCheckpoint(path)

        plain = Trainer(_buildModel(1), device="cpu")
        plain.loadCheckpoint(path)
        for name, tensor in trainer.model.state_dict().items():
            assert torch.equal(tensor, plain.model.state_dict()[name]), name

    def test_it_works_with_clipping_and_modality_dropout(self) -> None:
        trainer = Trainer(
            _buildTwoModalityModel(),
            device="cpu",
            optimizer_kwargs={"lr": 0.05},
            mixed_precision="bf16",
            grad_clip_norm=1.0,
            modality_dropout_p=0.5,
        )
        history = trainer.fit(_fixedTwoModalityBatches(4), num_epochs=3)
        assert all(math.isfinite(entry["train/loss/total"]) for entry in history)


class TestGradientAccumulation:
    def test_a_group_of_n_batches_is_one_step_on_the_mean_gradient(self) -> None:
        """N=2 over batches (a, b) equals one plain step on the mean of both gradients."""
        a, b = _fixedBatches(2)
        accumulating = Trainer(
            _buildModel(), device="cpu", optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.1}
        )
        manual = Trainer(
            _buildModel(), device="cpu", optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.1}
        )
        accumulating.accumulate_grad_batches = 2

        # The reparameterization noise is drawn per batch: both runs draw the same sequence.
        torch.manual_seed(5)
        accumulating.fit([a, b], num_epochs=1)

        torch.manual_seed(5)
        manual.optimizer.zero_grad()
        for batch in (a, b):
            backward(manual.computeLosses(batch, step=0).total / 2)
        manual.optimizer.step()

        for name, tensor in accumulating.model.state_dict().items():
            torch.testing.assert_close(
                tensor, manual.model.state_dict()[name], atol=1e-6, rtol=1e-5
            )

    def test_the_counters_count_optimizer_steps(self) -> None:
        recorder = _RecordStepMetrics()
        trainer = Trainer(
            _buildModel(), device="cpu", accumulate_grad_batches=3, callbacks=[recorder]
        )
        trainer.fit(_fixedBatches(7), num_epochs=2)

        # 7 batches in groups of 3 -> 3 optimizer steps per epoch (3 + 3 + a partial 1).
        assert trainer.global_step == 6
        assert [step for step, _ in recorder.steps] == [0, 1, 2, 3, 4, 5]

    def test_the_epoch_metric_is_the_mean_over_the_optimizer_steps(self) -> None:
        recorder = _RecordStepMetrics()
        trainer = Trainer(
            _buildModel(), device="cpu", accumulate_grad_batches=2, callbacks=[recorder]
        )
        history = trainer.fit(_fixedBatches(4), num_epochs=1)

        step_losses = [metrics["loss/total"] for _, metrics in recorder.steps]
        assert len(step_losses) == 2
        assert history[0]["train/loss/total"] == pytest.approx(sum(step_losses) / 2)

    def test_a_partial_last_group_is_averaged_over_its_own_batches(self) -> None:
        """3 batches with N=2: the second step uses batch c alone, not c / 2."""
        a, b, c = _fixedBatches(3)
        accumulating = Trainer(
            _buildModel(),
            device="cpu",
            optimizer=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
            accumulate_grad_batches=2,
        )
        manual = Trainer(
            _buildModel(), device="cpu", optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.1}
        )

        torch.manual_seed(5)
        accumulating.fit([a, b, c], num_epochs=1)

        torch.manual_seed(5)
        manual.optimizer.zero_grad()
        for batch in (a, b):
            backward(manual.computeLosses(batch, step=0).total / 2)
        manual.optimizer.step()
        manual.optimizer.zero_grad()
        backward(manual.computeLosses(c, step=1).total)
        manual.optimizer.step()

        for name, tensor in accumulating.model.state_dict().items():
            torch.testing.assert_close(
                tensor, manual.model.state_dict()[name], atol=1e-6, rtol=1e-5
            )

    def test_one_is_the_unchanged_per_batch_step(self) -> None:
        recorder = _RecordStepMetrics()
        trainer = Trainer(_buildModel(), device="cpu", callbacks=[recorder])
        trainer.fit(_fixedBatches(5), num_epochs=1)
        assert trainer.global_step == 5
        assert len(recorder.steps) == 5

    def test_the_beta_schedule_is_asked_about_the_optimizer_step_not_the_batch(self) -> None:
        schedule = _RecordingBeta()
        trainer = Trainer(
            _buildModel(),
            device="cpu",
            beta_schedules={"z": schedule},
            accumulate_grad_batches=2,
        )
        trainer.fit(_fixedBatches(4), num_epochs=1)
        assert trainer.global_step == 2
        assert schedule.steps == [0, 0, 1, 1]

    def test_clipping_applies_to_the_accumulated_gradient(self) -> None:
        trainer = Trainer(
            _buildModel(),
            device="cpu",
            optimizer=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.0},
            accumulate_grad_batches=2,
            grad_clip_norm=1e-3,
        )
        trainer.fit(_fixedBatches(2), num_epochs=1)
        gradients = [p.grad.flatten() for p in trainer.model.parameters() if p.grad is not None]
        total = torch.linalg.vector_norm(torch.cat(gradients))
        assert total.item() <= 1e-3 * (1 + 1e-4)

    def test_it_trains(self) -> None:
        trainer = Trainer(
            _buildModel(),
            device="cpu",
            optimizer_kwargs={"lr": 0.05},
            accumulate_grad_batches=4,
        )
        history = trainer.fit(_fixedBatches(8), num_epochs=20)
        assert _lossesDecrease(history)
        for name, parameter in trainer.model.named_parameters():
            assert parameter.grad is not None, name

    def test_it_composes_with_mixed_precision(self) -> None:
        trainer = Trainer(
            _buildModel(),
            device="cpu",
            optimizer_kwargs={"lr": 0.05},
            accumulate_grad_batches=2,
            mixed_precision="bf16",
        )
        history = trainer.fit(_fixedBatches(8), num_epochs=10)
        assert all(math.isfinite(entry["train/loss/total"]) for entry in history)

    def test_it_composes_with_fp16_and_its_scaler(self) -> None:
        _needsCpuGradScaler()
        trainer = Trainer(
            _buildModel(),
            device="cpu",
            optimizer_kwargs={"lr": 0.05},
            accumulate_grad_batches=2,
            mixed_precision="fp16",
            grad_clip_norm=1.0,
        )
        history = trainer.fit(_fixedBatches(8), num_epochs=10)
        assert all(math.isfinite(entry["train/loss/total"]) for entry in history)


class TestCompile:
    """`torch.compile(backend="eager")`: the wrapper machinery without a C++ toolchain."""

    def _compiled(self, **kwargs: Any) -> Trainer:
        return Trainer(
            _buildModel(),
            device="cpu",
            compile_model=True,
            compile_kwargs={"backend": "eager"},
            **kwargs,
        )

    def test_it_trains_like_the_eager_model(self) -> None:
        batches = _fixedBatches(6)
        eager = Trainer(_buildModel(), device="cpu", optimizer_kwargs={"lr": 0.05})
        compiled = self._compiled(optimizer_kwargs={"lr": 0.05})
        torch.manual_seed(2)
        history_eager = eager.fit(batches, num_epochs=3)
        torch.manual_seed(2)
        history_compiled = compiled.fit(batches, num_epochs=3)
        for expected, observed in zip(history_eager, history_compiled, strict=True):
            assert observed["train/loss/total"] == pytest.approx(
                expected["train/loss/total"], rel=1e-5
            )

    def test_checkpoints_keep_the_keys_of_the_uncompiled_model(self, tmp_path: Path) -> None:
        compiled = self._compiled()
        compiled.fit(_fixedBatches(2), num_epochs=1)
        path = tmp_path / "compiled.pt"
        compiled.saveCheckpoint(path)

        plain = Trainer(_buildModel(1), device="cpu")
        plain.loadCheckpoint(path)
        assert not any(key.startswith("_orig_mod") for key in plain.model.state_dict())
        for name, tensor in compiled.model.state_dict().items():
            assert torch.equal(tensor, plain.model.state_dict()[name]), name

    def test_it_works_with_modality_dropout(self) -> None:
        trainer = Trainer(
            _buildTwoModalityModel(),
            device="cpu",
            compile_model=True,
            compile_kwargs={"backend": "eager"},
            modality_dropout_p=0.5,
        )
        history = trainer.fit(_fixedTwoModalityBatches(4), num_epochs=2)
        assert all(math.isfinite(entry["train/loss/total"]) for entry in history)

    def test_evaluation_uses_the_model_too(self) -> None:
        trainer = self._compiled()
        metrics = trainer.evaluate(_fixedBatches(2))
        assert math.isfinite(metrics["val/loss/total"])


class TestCheckpointFileOfATrainerWithCapabilities:
    def test_the_file_is_readable_in_the_safe_mode(self, tmp_path: Path) -> None:
        trainer = Trainer(
            _buildModel(), device="cpu", mixed_precision="bf16", accumulate_grad_batches=2
        )
        trainer.fit(_fixedBatches(4), num_epochs=1)
        path = tmp_path / "ckpt.pt"
        trainer.saveCheckpoint(path)

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["global_step"] == 2
        assert "scaler_state_dict" not in checkpoint

    def test_saving_without_a_scaler_or_callback_state_adds_no_new_keys(
        self, tmp_path: Path
    ) -> None:
        trainer = Trainer(_buildModel(), device="cpu")
        path = tmp_path / "ckpt.pt"
        trainer.saveCheckpoint(path)
        reference = tmp_path / "reference.pt"
        saveCheckpoint(reference, model=trainer.model, optimizer=trainer.optimizer)

        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert set(saved) == set(torch.load(reference, map_location="cpu", weights_only=True))
