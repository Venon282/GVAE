"""Integration tests for `training.checkpoint` (spec §10: config snapshotted with every
run; re-run eval/visualization without retraining).

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
from global_vae.training.checkpoint import (
    BestCheckpointCallback,
    CheckpointCallback,
    loadCheckpoint,
    saveCheckpoint,
)
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


class TestCheckpointCallback:
    def test_saves_every_n_epochs(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        callback = CheckpointCallback(directory=tmp_path, every_n_epochs=2)
        trainer = Trainer(model, device="cpu", callbacks=[callback])

        trainer.fit(_fixedDataset(num_batches=2), num_epochs=4)

        saved_files = sorted(tmp_path.glob("checkpoint_epoch_*.pt"))
        assert len(saved_files) == 2  # after epoch 1 (0-based) and epoch 3

    def test_last_checkpoint_can_be_loaded_and_matches_final_state(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        callback = CheckpointCallback(directory=tmp_path, every_n_epochs=1)
        trainer = Trainer(model, device="cpu")
        trainer.callbacks = [callback]

        trainer.fit(_fixedDataset(num_batches=2), num_epochs=3)

        last_checkpoint = sorted(tmp_path.glob("checkpoint_epoch_*.pt"))[-1]
        reloaded_model = _buildSingleModalityModel()
        metadata = loadCheckpoint(last_checkpoint, model=reloaded_model)
        assert metadata.start_epoch == 3
        assert torch.equal(
            reloaded_model.encoders["signal"].to_mu.weight,
            trainer.model.encoders["signal"].to_mu.weight,
        )

    def test_keep_last_n_deletes_older_checkpoints(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        callback = CheckpointCallback(directory=tmp_path, every_n_epochs=1, keep_last_n=2)
        trainer = Trainer(model, device="cpu", callbacks=[callback])

        trainer.fit(_fixedDataset(num_batches=2), num_epochs=4)

        saved_files = sorted(tmp_path.glob("checkpoint_epoch_*.pt"))
        assert len(saved_files) == 2
        assert saved_files[-1].name == "checkpoint_epoch_0003.pt"

    def test_invalid_every_n_epochs_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="every_n_epochs"):
            CheckpointCallback(directory=tmp_path, every_n_epochs=0)

    def test_invalid_keep_last_n_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="keep_last_n"):
            CheckpointCallback(directory=tmp_path, keep_last_n=0)


class TestBestCheckpointCallback:
    def test_only_saves_on_improvement_min_mode(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        checkpoint_path = tmp_path / "best.pt"
        callback = BestCheckpointCallback(path=checkpoint_path, monitor="train/loss/total")
        trainer = Trainer(model, device="cpu", callbacks=[callback])

        trainer.fit(_fixedDataset(num_batches=2), num_epochs=5)

        assert checkpoint_path.exists()
        best_values = [
            metrics["train/loss/total"]
            for metrics in trainer.history
            if metrics["train/loss/total"] == callback.best_value
        ]
        assert callback.best_value == min(entry["train/loss/total"] for entry in trainer.history)
        assert best_values  # sanity: the recorded best actually occurred in history

    def test_always_overwrites_the_same_single_file(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        checkpoint_path = tmp_path / "best.pt"
        callback = BestCheckpointCallback(path=checkpoint_path, monitor="train/loss/total")
        trainer = Trainer(model, device="cpu", callbacks=[callback])

        trainer.fit(_fixedDataset(num_batches=2), num_epochs=5)

        assert checkpoint_path.exists()
        assert len(list(tmp_path.glob("*.pt"))) == 1

    def test_max_mode_keeps_the_highest_value(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        checkpoint_path = tmp_path / "best.pt"
        callback = BestCheckpointCallback(
            path=checkpoint_path, monitor="train/loss/total", mode="max"
        )
        trainer = Trainer(model, device="cpu", callbacks=[callback])

        trainer.fit(_fixedDataset(num_batches=2), num_epochs=5)

        assert callback.best_value == max(entry["train/loss/total"] for entry in trainer.history)

    def test_best_checkpoint_can_be_loaded_and_matches_the_best_epoch_weights(
        self, tmp_path
    ) -> None:
        """The actual point: loading `path` at any time gives the best model, not the last one."""
        model = _buildSingleModalityModel()
        checkpoint_path = tmp_path / "best.pt"
        callback = BestCheckpointCallback(path=checkpoint_path, monitor="train/loss/total")
        trainer = Trainer(model, device="cpu", optimizer_kwargs={"lr": 0.2}, callbacks=[callback])

        trainer.fit(_fixedDataset(num_batches=2), num_epochs=8)

        reloaded_model = _buildSingleModalityModel()
        metadata = loadCheckpoint(checkpoint_path, model=reloaded_model)

        # The checkpoint's own reconstruction loss (deterministic reparameterization aside via
        # eval mode not being enforced here) should match the best epoch's recorded metric,
        # not necessarily the final epoch's.
        best_epoch_metrics = min(trainer.history, key=lambda entry: entry["train/loss/total"])
        assert metadata.history[-1] == best_epoch_metrics

    def test_missing_monitor_key_raises_key_error(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        callback = BestCheckpointCallback(path=tmp_path / "best.pt", monitor="val/loss/total")
        trainer = Trainer(model, device="cpu", callbacks=[callback])

        with pytest.raises(KeyError, match="val/loss/total"):
            trainer.fit(_fixedDataset(num_batches=2), num_epochs=1)  # no val_dataloader given

    def test_works_with_validation_dataloader(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        checkpoint_path = tmp_path / "best.pt"
        callback = BestCheckpointCallback(path=checkpoint_path, monitor="val/loss/total")
        trainer = Trainer(model, device="cpu", callbacks=[callback])

        trainer.fit(
            _fixedDataset(num_batches=2), num_epochs=3, val_dataloader=_fixedDataset(num_batches=1)
        )

        assert checkpoint_path.exists()

    def test_invalid_mode_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="mode"):
            BestCheckpointCallback(path=tmp_path / "best.pt", mode="sideways")
