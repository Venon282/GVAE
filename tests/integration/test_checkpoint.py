"""Integration tests for `training.checkpoint` (spec §10: config snapshotted with every
run; re-run eval/visualization without retraining).

Covers the plain, `Trainer`-independent checkpoint *format* only
(`saveCheckpoint`/`loadCheckpoint` roundtrips, RNG state, error paths).
`CheckpointCallback`/`BestCheckpointCallback` (the two `TrainerCallback`s that call
into this format on a schedule or on metric improvement) live in
`training/callbacks/checkpoint.py` and `training/callbacks/best_checkpoint.py`
respectively; their own tests are in `test_checkpoint_callbacks.py` (ADR 0022).

Uses its own trivial linear dummy encoder/decoder (mirroring
`test_trainer.py`'s pattern), registered under a `_checkpoint_test`
suffix so it cannot collide with dummy fixtures in sibling integration
test files: importing a sibling test module directly (rather than
duplicating its small dummy fixture) would make pytest import it twice
under two different module names during full-suite collection,
re-running its `@registerEncoder(...)` decorators and raising
"already registered".
"""

import random

import numpy as np
import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import loadCheckpoint, saveCheckpoint
from global_vae.training.trainer import Trainer

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_checkpoint_test")
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


@registerDecoder("dummy_signal_decoder_checkpoint_test")
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


def _buildSingleModalityModel() -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_checkpoint_test",
                "decoder": "dummy_signal_decoder_checkpoint_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedDataset(
    num_batches: int, input_dim: int = INPUT_DIM, seed: int = 0
) -> list[dict[str, torch.Tensor]]:
    """A small, deterministic, re-iterable "dataset" (a plain list of batches)."""
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, input_dim)} for _ in range(num_batches)]


class TestSaveAndLoadRoundtrip:
    def test_model_weights_are_restored_exactly(self, tmp_path) -> None:
        trained_model = _buildSingleModalityModel()
        trainer = Trainer(trained_model, device="cpu", optimizer_kwargs={"lr": 0.1})
        trainer.fit(_fixedDataset(num_batches=3), num_epochs=5)  # move weights away from init

        checkpoint_path = tmp_path / "model.pt"
        saveCheckpoint(checkpoint_path, model=trainer.model)

        fresh_model = _buildSingleModalityModel()  # different random init
        assert not torch.equal(
            fresh_model.encoders["signal"].to_mu.weight,
            trainer.model.encoders["signal"].to_mu.weight,
        )

        loadCheckpoint(checkpoint_path, model=fresh_model)
        assert torch.equal(
            fresh_model.encoders["signal"].to_mu.weight,
            trainer.model.encoders["signal"].to_mu.weight,
        )

    def test_optimizer_state_is_restored(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        batch = {"signal": torch.randn(BATCH_SIZE, INPUT_DIM)}
        loss = sum(p.sum() for p in model(batch)["reconstructions"].values())
        loss.backward()
        optimizer.step()  # populate Adam's internal moment-estimate state

        checkpoint_path = tmp_path / "with_optimizer.pt"
        saveCheckpoint(checkpoint_path, model=model, optimizer=optimizer)

        fresh_model = _buildSingleModalityModel()
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=0.01)
        loadCheckpoint(checkpoint_path, model=fresh_model, optimizer=fresh_optimizer)

        original_state = optimizer.state_dict()["state"]
        restored_state = fresh_optimizer.state_dict()["state"]
        assert original_state.keys() == restored_state.keys()
        for key in original_state:
            assert torch.equal(original_state[key]["exp_avg"], restored_state[key]["exp_avg"])

    def test_config_is_restored_unchanged(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        config = {"latent_dim": 4, "modalities": ["signal"], "notes": "first run"}
        checkpoint_path = tmp_path / "with_config.pt"
        saveCheckpoint(checkpoint_path, model=model, config=config)

        fresh_model = _buildSingleModalityModel()
        metadata = loadCheckpoint(checkpoint_path, model=fresh_model)
        assert metadata.config == config

    def test_step_epoch_and_history_roundtrip_through_trainer_methods(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        trainer.fit(_fixedDataset(num_batches=2), num_epochs=3)

        checkpoint_path = tmp_path / "trainer_state.pt"
        trainer.saveCheckpoint(checkpoint_path, config={"note": "checkpoint"})

        resumed_model = _buildSingleModalityModel()
        resumed_trainer = Trainer(resumed_model, device="cpu")
        returned_config = resumed_trainer.loadCheckpoint(checkpoint_path)

        assert resumed_trainer.global_step == trainer.global_step == 6
        assert resumed_trainer.start_epoch == trainer.start_epoch == 3
        assert len(resumed_trainer.history) == 3
        assert returned_config == {"note": "checkpoint"}

    def test_resumed_training_continues_epoch_numbering(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu")
        trainer.fit(_fixedDataset(num_batches=2), num_epochs=2)
        checkpoint_path = tmp_path / "resume.pt"
        trainer.saveCheckpoint(checkpoint_path)

        resumed_model = _buildSingleModalityModel()
        resumed_trainer = Trainer(resumed_model, device="cpu")
        resumed_trainer.loadCheckpoint(checkpoint_path)
        resumed_trainer.fit(_fixedDataset(num_batches=2), num_epochs=1)

        assert resumed_trainer.start_epoch == 3
        assert len(resumed_trainer.history) == 3


class TestRngStateRoundtrip:
    def test_restoring_rng_state_reproduces_the_next_draws(self, tmp_path) -> None:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        model = _buildSingleModalityModel()
        checkpoint_path = tmp_path / "rng.pt"
        saveCheckpoint(checkpoint_path, model=model, include_rng_state=True)

        expected_python = [random.random() for _ in range(3)]
        expected_numpy = np.random.rand(3)
        expected_torch = torch.randn(3)

        # Move the RNGs somewhere else entirely, then restore from the checkpoint.
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)

        fresh_model = _buildSingleModalityModel()
        metadata = loadCheckpoint(checkpoint_path, model=fresh_model, restore_rng_state=True)
        assert metadata.rng_state_restored

        assert [random.random() for _ in range(3)] == expected_python
        assert np.array_equal(np.random.rand(3), expected_numpy)
        assert torch.equal(torch.randn(3), expected_torch)

    def test_restore_rng_state_false_leaves_current_rng_untouched(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        checkpoint_path = tmp_path / "rng_skip.pt"
        saveCheckpoint(checkpoint_path, model=model, include_rng_state=True)
        fresh_model = _buildSingleModalityModel()  # build before the seed dance below

        torch.manual_seed(555)
        expected = torch.randn(3)

        torch.manual_seed(555)
        metadata = loadCheckpoint(checkpoint_path, model=fresh_model, restore_rng_state=False)
        assert not metadata.rng_state_restored
        assert torch.equal(torch.randn(3), expected)

    def test_no_rng_state_saved_when_disabled(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        checkpoint_path = tmp_path / "no_rng.pt"
        saveCheckpoint(checkpoint_path, model=model, include_rng_state=False)

        fresh_model = _buildSingleModalityModel()
        metadata = loadCheckpoint(checkpoint_path, model=fresh_model)
        assert not metadata.rng_state_restored


class TestErrorPaths:
    def test_missing_file_raises_file_not_found(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        with pytest.raises(FileNotFoundError):
            loadCheckpoint(tmp_path / "does_not_exist.pt", model=model)

    def test_requesting_optimizer_restore_without_saved_optimizer_raises(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        checkpoint_path = tmp_path / "no_optimizer.pt"
        saveCheckpoint(checkpoint_path, model=model)  # no optimizer passed in

        fresh_model = _buildSingleModalityModel()
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters())
        with pytest.raises(ValueError, match="optimizer"):
            loadCheckpoint(checkpoint_path, model=fresh_model, optimizer=fresh_optimizer)
