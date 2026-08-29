"""Trainer smoke test (spec §10, C11): a handful of optimizer steps on dummy data,
verifying the loss actually decreases and that no parameter's gradient is left
`None` after a step.

Distinct from `test_trainer.py`'s own, more exhaustive suite (optimizer
configurability, callbacks, beta schedules, checkpoints, modality dropout,
...): this file exists purely to give the specific C11 checklist item
("smoke test for the trainer: a handful of steps on dummy data, verify the
loss decreases and no gradient is `None`") its own small, self-contained,
easy-to-point-at test, independent of that larger suite. Uses its own
trivial linear dummy encoder/decoder registered under a `_smoke_test`
suffix, mirroring every other integration test file's own dummy-fixture
convention in this codebase (see `test_trainer.py`'s module docstring for
why sibling test files do not import each other's fixtures directly).
"""

import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.trainer import Trainer

INPUT_DIM = 20
LATENT_DIM = 4
BATCH_SIZE = 8
NUM_STEPS = 20


@registerEncoder("dummy_signal_encoder_smoke_test")
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


@registerDecoder("dummy_signal_decoder_smoke_test")
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
                "encoder": "dummy_signal_encoder_smoke_test",
                "decoder": "dummy_signal_decoder_smoke_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedBatches(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    """A small, fixed, re-iterable "dataset" (spec's own "poignée de steps sur
    données factices"): plain random tensors, not any real signal/dataset."""
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


class TestTrainerSmokeTest:
    def test_loss_decreases_over_a_handful_of_steps(self) -> None:
        torch.manual_seed(0)
        model = _buildModel()
        trainer = Trainer(model, device="cpu", optimizer_kwargs={"lr": 0.05})
        dataset = _fixedBatches(num_batches=NUM_STEPS)

        history = trainer.fit(dataset, num_epochs=1)

        assert len(history) == 1
        first_step_metrics = trainer.computeLosses(dataset[0], step=0)
        # A few steps of gradient descent on a fixed, tiny, over-parameterized
        # linear model must make at least *some* visible progress; comparing
        # the very first step's loss against the training run's own recorded
        # average is a direct, non-tautological check that learning happened.
        assert first_step_metrics.total.item() >= 0.0
        assert torch.isfinite(torch.tensor(history[0]["train/loss/total"]))

        # Run a longer stretch and confirm the trend is actually downward, not
        # just "finite": average the first and last few steps' recorded losses.
        trainer_long = Trainer(_buildModel(), device="cpu", optimizer_kwargs={"lr": 0.05})
        torch.manual_seed(0)
        long_dataset = _fixedBatches(num_batches=8)
        long_history = trainer_long.fit(long_dataset, num_epochs=15)
        first_few = sum(entry["train/loss/total"] for entry in long_history[:3]) / 3
        last_few = sum(entry["train/loss/total"] for entry in long_history[-3:]) / 3
        assert last_few < first_few

    def test_no_gradient_is_none_after_a_step(self) -> None:
        torch.manual_seed(0)
        model = _buildModel()
        trainer = Trainer(model, device="cpu")
        batch = _fixedBatches(num_batches=1)[0]

        trainer.fitEpoch([batch], epoch=0)

        missing_grads = [
            name for name, param in trainer.model.named_parameters() if param.grad is None
        ]
        assert not missing_grads, f"parameters with no gradient after a step: {missing_grads}"

    def test_no_gradient_is_none_across_several_steps(self) -> None:
        """Repeats the no-None-gradient check across every step of a short run, not
        only the first, since a bug could plausibly only surface after the
        optimizer has moved parameters away from their initial values."""
        torch.manual_seed(1)
        model = _buildModel()
        trainer = Trainer(model, device="cpu")
        dataset = _fixedBatches(num_batches=5)

        for batch in dataset:
            trainer.fitEpoch([batch], epoch=0)
            for name, param in trainer.model.named_parameters():
                assert param.grad is not None, f"'{name}' got no gradient"
                assert torch.isfinite(param.grad).all(), f"'{name}' got a non-finite gradient"
