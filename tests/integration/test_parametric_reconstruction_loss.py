"""A reconstruction loss that owns parameters, in the trainer and the checkpoint (roadmap P5-3).

`AbstractReconstructionLoss` is an `nn.Module`, so a loss may learn something (a noise scale, a
class weight). ADR 0048 says it reaches the optimizer and the checkpoint through the mechanism of
the loss terms (ADR 0047), under a key of its own. This file pins that with a Gaussian likelihood
whose log-scale is learned, registered here under a unique name:

- its parameter is in the one optimizer group, moves, is clipped with the model's, follows the
  train and eval modes, and works under mixed precision and gradient accumulation;
- it is saved as `reconstruction_loss_state_dict`, restored with its optimizer state, and a resumed
  run gives the curve and the weights of the uninterrupted one;
- **a loss with no parameter changes nothing**: the checkpoint has no new key, and a run with the
  `MseLoss` class gives, bit for bit, the history and weights of the run with the plain function.
"""

import logging
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from global_vae.config.training import (
    TrainingConfig,
    buildTrainerFromConfig,
    resumeTrainerFromConfig,
)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.evaluation.evaluate import evaluate
from global_vae.losses.reconstruction import (
    AbstractReconstructionLoss,
    registerReconstructionLoss,
)
from global_vae.losses.reconstruction.mse import MseLoss
from global_vae.models.global_vae import GlobalVae
from global_vae.training.trainer import Trainer

INPUT_DIM = 12
LATENT_DIM = 3
HIDDEN_DIM = 10
BATCH_SIZE = 6
LOSS = "learned_scale_reconstruction_loss_test"
PARAMETER = "default.log_scale"


@registerEncoder("dummy_signal_encoder_parametric_loss_test")
class _DummyEncoder(AbstractEncoder):
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


@registerDecoder("dummy_signal_decoder_parametric_loss_test")
class _DummyDecoder(AbstractDecoder):
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


@registerReconstructionLoss(LOSS)
class _LearnedScaleLoss(AbstractReconstructionLoss):
    """A Gaussian negative log-likelihood whose log-scale is a parameter: a mean over the values."""

    def __init__(self, initial: float = 0.0) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(float(initial)))

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared = (reconstruction - target).pow(2)
        value: torch.Tensor = (0.5 * (squared * torch.exp(-self.log_scale) + self.log_scale)).mean()
        return value


def _buildModel(seed: int = 0) -> GlobalVae:
    torch.manual_seed(seed)
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_parametric_loss_test",
                "decoder": "dummy_signal_decoder_parametric_loss_test",
            }
        },
        latent_dim=LATENT_DIM,
    )


def _batches(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return [
        {"signal": torch.randn(BATCH_SIZE, INPUT_DIM, generator=generator)}
        for _ in range(num_batches)
    ]


def _trainer(loss: Any | None = None, model: GlobalVae | None = None, **kwargs: Any) -> Trainer:
    kwargs.setdefault("optimizer_kwargs", {"lr": 0.05})
    return Trainer(
        model if model is not None else _buildModel(),
        device="cpu",
        reconstruction_loss_fn=loss if loss is not None else _LearnedScaleLoss(),
        **kwargs,
    )


def _scale(trainer: Trainer) -> torch.Tensor:
    loss = trainer.reconstruction_losses["default"]
    assert isinstance(loss, _LearnedScaleLoss)
    return loss.log_scale


class TestTheTrainerHoldsTheLoss:
    def test_a_loss_module_is_held_under_default(self) -> None:
        trainer = _trainer()
        assert list(trainer.reconstruction_losses) == ["default"]
        assert trainer.reconstruction_losses["default"] is trainer.reconstruction_loss_fn

    def test_a_plain_function_leaves_the_holder_empty(self) -> None:
        trainer = _trainer(torch.nn.functional.mse_loss)
        assert len(trainer.reconstruction_losses) == 0

    def test_a_loss_without_parameters_is_held_but_adds_no_parameter(self) -> None:
        with_class = _trainer(MseLoss())
        assert list(with_class.reconstruction_losses) == ["default"]
        assert list(with_class.reconstruction_losses.parameters()) == []

    def test_a_mapping_holds_each_module_once(self) -> None:
        shared = _LearnedScaleLoss()
        trainer = _trainer({"signal": shared})
        assert list(trainer.reconstruction_losses) == ["signal"]
        assert trainer.reconstruction_losses["signal"] is shared


class TestAParameterTrains:
    def test_it_is_in_the_one_optimizer_group_and_moves(self) -> None:
        trainer = _trainer()
        parameters = [p for group in trainer.optimizer.param_groups for p in group["params"]]
        assert len(trainer.optimizer.param_groups) == 1
        assert any(p is _scale(trainer) for p in parameters)
        assert len(parameters) == len(list(trainer.model.parameters())) + 1
        before = _scale(trainer).item()
        history = trainer.fit(_batches(4), num_epochs=3)
        assert _scale(trainer).item() != before
        assert _scale(trainer).grad is not None
        assert all(math.isfinite(entry["train/loss/reconstruction"]) for entry in history)

    def test_an_optimizer_instance_gets_the_missing_parameter_as_a_group(self) -> None:
        model = _buildModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = _LearnedScaleLoss()
        Trainer(model, optimizer=optimizer, device="cpu", reconstruction_loss_fn=loss)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[1]["params"] == [loss.log_scale]

    def test_a_parameter_the_optimizer_already_holds_is_not_added_again(self) -> None:
        model = _buildModel()
        loss = _LearnedScaleLoss()
        optimizer = torch.optim.SGD([*model.parameters(), *loss.parameters()], lr=0.1)
        Trainer(model, optimizer=optimizer, device="cpu", reconstruction_loss_fn=loss)
        assert len(optimizer.param_groups) == 1

    def test_clipping_covers_it(self) -> None:
        trainer = _trainer(
            optimizer=torch.optim.SGD, optimizer_kwargs={"lr": 0.0}, grad_clip_norm=1e-3
        )
        trainer.fit(_batches(2), num_epochs=1)
        gradients = [
            p.grad.flatten()
            for p in [*trainer.model.parameters(), *trainer.reconstruction_losses.parameters()]
            if p.grad is not None
        ]
        assert torch.cat(gradients).norm().item() <= 1e-3 * (1 + 1e-4)
        assert _scale(trainer).grad is not None

    def test_it_follows_the_train_and_eval_modes_of_the_trainer(self) -> None:
        trainer = _trainer()
        trainer.fitEpoch(_batches(2), epoch=0)
        assert trainer.reconstruction_losses.training
        trainer.evaluate(_batches(2, seed=9))
        assert not trainer.reconstruction_losses.training

    def test_it_trains_under_bf16_and_accumulation_and_stays_float32(self) -> None:
        trainer = _trainer(mixed_precision="bf16", accumulate_grad_batches=2)
        before = _scale(trainer).item()
        history = trainer.fit(_batches(4), num_epochs=2)
        assert _scale(trainer).dtype == torch.float32
        assert _scale(trainer).item() != before
        assert all(math.isfinite(entry["train/loss/reconstruction"]) for entry in history)

    def test_it_is_the_reconstruction_term_of_the_total_times_the_weight(self) -> None:
        trainer = _trainer(reconstruction_weights=3.0, modality_dropout_p=0.0)
        batch = _batches(1)[0]
        torch.manual_seed(1)
        losses = trainer.computeLosses(batch, step=0)
        torch.manual_seed(1)
        reconstruction = trainer.model(batch)["reconstructions"]["signal"]
        loss_fn = trainer.reconstruction_loss_fn
        assert callable(loss_fn)
        expected = 3.0 * loss_fn(reconstruction, batch["signal"])
        assert losses.reconstruction.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_evaluate_runs_the_loss_in_eval_mode(self) -> None:
        trainer = _trainer()
        trainer.fit(_batches(2), num_epochs=1)
        results = evaluate(
            trainer.model,
            _batches(2, seed=9),
            reconstruction_loss_fn=trainer.reconstruction_loss_fn,
        )
        assert math.isfinite(results.total_reconstruction_loss)


class TestCheckpointAndResume:
    def _saved(self, path: Path, **kwargs: Any) -> Trainer:
        trainer = _trainer(**kwargs)
        trainer.fit(_batches(3), num_epochs=2)
        trainer.saveCheckpoint(path)
        return trainer

    def test_the_state_is_in_a_key_of_its_own_and_readable_in_the_safe_mode(
        self, tmp_path: Path
    ) -> None:
        trainer = self._saved(tmp_path / "c.pt")
        checkpoint = torch.load(tmp_path / "c.pt", weights_only=True)
        assert list(checkpoint["reconstruction_loss_state_dict"]) == [PARAMETER]
        assert torch.equal(
            checkpoint["reconstruction_loss_state_dict"][PARAMETER], _scale(trainer).detach()
        )
        assert "loss_terms_state_dict" not in checkpoint
        assert not any("log_scale" in name for name in checkpoint["model_state_dict"])

    def test_it_is_restored_with_its_optimizer_state(self, tmp_path: Path) -> None:
        saved = self._saved(tmp_path / "c.pt", optimizer=torch.optim.Adam)
        fresh = _trainer(_LearnedScaleLoss(initial=9.0), _buildModel(1), optimizer=torch.optim.Adam)
        fresh.loadCheckpoint(tmp_path / "c.pt")
        assert torch.equal(_scale(fresh), _scale(saved))
        saved_state = saved.optimizer.state[_scale(saved)]
        fresh_state = fresh.optimizer.state[_scale(fresh)]
        assert torch.equal(saved_state["exp_avg"], fresh_state["exp_avg"])
        assert torch.equal(saved_state["exp_avg_sq"], fresh_state["exp_avg_sq"])

    @pytest.mark.parametrize(
        "extras", [{}, {"mixed_precision": "bf16", "accumulate_grad_batches": 2}]
    )
    def test_a_resumed_run_gives_the_curve_and_the_weights_of_the_uninterrupted_one(
        self, tmp_path: Path, extras: dict[str, Any]
    ) -> None:
        batches = _batches(4)

        def build(seed: int, initial: float) -> Trainer:
            return _trainer(
                _LearnedScaleLoss(initial=initial),
                _buildModel(seed),
                optimizer=torch.optim.Adam,
                **extras,
            )

        torch.manual_seed(7)
        whole = build(0, 0.0)
        history_whole = whole.fit(batches, num_epochs=4)

        torch.manual_seed(7)
        first = build(0, 0.0)
        first.fit(batches, num_epochs=2)
        first.saveCheckpoint(tmp_path / "half.pt")

        second = build(5, 9.0)  # another model and another initial value: only the file survives
        second.loadCheckpoint(tmp_path / "half.pt")
        history_resumed = second.fit(batches, num_epochs=2)

        assert len(history_resumed) == len(history_whole) == 4
        for expected, observed in zip(history_whole, history_resumed, strict=True):
            assert expected.keys() == observed.keys()
            for key, value in expected.items():
                assert observed[key] == pytest.approx(value, rel=1e-6, abs=1e-9), key
        assert _scale(second).item() == pytest.approx(_scale(whole).item(), rel=1e-6)

    def test_the_resume_helper_of_the_config_restores_it(self, tmp_path: Path) -> None:
        config = TrainingConfig(num_epochs=4, reconstruction_loss={"name": LOSS})
        first = buildTrainerFromConfig(_buildModel(), config)
        first.fit(_batches(3), num_epochs=2)
        first.saveCheckpoint(tmp_path / "c.pt")
        config.resume_from = str(tmp_path / "c.pt")

        second = buildTrainerFromConfig(_buildModel(3), config)
        assert resumeTrainerFromConfig(second, config) == 2
        assert torch.equal(_scale(second), _scale(first))

    def test_the_config_builds_the_loss_with_its_arguments(self) -> None:
        config = TrainingConfig(
            num_epochs=1, reconstruction_loss={"name": LOSS, "kwargs": {"initial": 2.5}}
        )
        trainer = buildTrainerFromConfig(_buildModel(), config)
        assert isinstance(trainer.reconstruction_loss_fn, _LearnedScaleLoss)
        assert _scale(trainer).item() == 2.5

    def test_a_loss_with_no_parameter_writes_no_key(self, tmp_path: Path) -> None:
        for name, loss in (("class", MseLoss()), ("function", torch.nn.functional.mse_loss)):
            trainer = _trainer(loss)
            trainer.fit(_batches(2), num_epochs=1)
            trainer.saveCheckpoint(tmp_path / f"{name}.pt")
            checkpoint = torch.load(tmp_path / f"{name}.pt", weights_only=True)
            assert "reconstruction_loss_state_dict" not in checkpoint

    def test_a_checkpoint_of_a_run_without_the_parameter_is_refused_with_its_optimizer(
        self, tmp_path: Path
    ) -> None:
        plain = _trainer(MseLoss())
        plain.fit(_batches(2), num_epochs=1)
        plain.saveCheckpoint(tmp_path / "plain.pt")
        with pytest.raises(
            ValueError, match=r"no state for the reconstruction loss parameter.*log_scale"
        ):
            _trainer().loadCheckpoint(tmp_path / "plain.pt")

    def test_the_weights_alone_load_and_the_parameter_keeps_its_initial_value(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        plain = _trainer(MseLoss())
        plain.fit(_batches(2), num_epochs=1)
        plain.saveCheckpoint(tmp_path / "plain.pt")
        fresh = _trainer(_LearnedScaleLoss(initial=4.0))
        with caplog.at_level(logging.WARNING):
            fresh.loadCheckpoint(tmp_path / "plain.pt", restore_optimizer=False)
        assert _scale(fresh).item() == 4.0
        assert "keep their initial values" in caplog.text

    def test_a_checkpoint_with_the_parameter_is_refused_by_a_trainer_without_it(
        self, tmp_path: Path
    ) -> None:
        self._saved(tmp_path / "c.pt")
        with pytest.raises(
            ValueError, match=r"holds reconstruction loss state \['default.log_scale'\]"
        ):
            _trainer(MseLoss()).loadCheckpoint(tmp_path / "c.pt")

    def test_a_non_strict_load_drops_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._saved(tmp_path / "c.pt")
        with caplog.at_level(logging.WARNING):
            _trainer(MseLoss()).loadCheckpoint(
                tmp_path / "c.pt", restore_optimizer=False, strict=False
            )
        assert "Dropped" in caplog.text

    def test_the_message_names_the_config_field_that_chooses_the_loss(self, tmp_path: Path) -> None:
        self._saved(tmp_path / "c.pt")
        with pytest.raises(ValueError, match=r"`training.reconstruction_loss`"):
            _trainer(MseLoss()).loadCheckpoint(tmp_path / "c.pt")


class TestALossWithNoParameterIsTheRunOfBefore:
    """`MseLoss()` is `torch.nn.functional.mse_loss` for the trainer, bit for bit."""

    @pytest.mark.parametrize(
        "extras",
        [
            {},
            {"mixed_precision": "bf16", "accumulate_grad_batches": 2, "grad_clip_norm": 1.0},
            {"modality_dropout_p": 0.5},
        ],
    )
    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_the_history_and_the_weights_are_those_of_the_function(
        self, extras: dict[str, Any], reduction: str
    ) -> None:
        def run(loss: Any) -> tuple[list[dict[str, float]], dict[str, torch.Tensor]]:
            torch.manual_seed(3)
            trainer = _trainer(loss, reconstruction_reduction=reduction, **extras)
            history = trainer.fit(_batches(4), num_epochs=3, val_dataloader=_batches(2, seed=9))
            return history, trainer.model.state_dict()

        history_class, weights_class = run(MseLoss())
        history_function, weights_function = run(torch.nn.functional.mse_loss)
        assert history_class == history_function
        for name, tensor in weights_function.items():
            assert torch.equal(weights_class[name], tensor), name
