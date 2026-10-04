"""Integration tests for `training.callbacks.checkpoint.CheckpointCallback` and
`training.callbacks.best_checkpoint.BestCheckpointCallback` (spec §10
"Reproducibility", ADR 0022), one class per file per spec §10's "Modularity" rule.

Checkpoint *format* correctness (`saveCheckpoint`/`loadCheckpoint` roundtrips, RNG
state, error paths) is covered by `test_checkpoint.py`; this file only covers the two
`TrainerCallback`s that call into that format on a schedule or on metric improvement.

Uses its own trivial linear dummy encoder/decoder (mirroring `test_trainer.py`'s and
`test_checkpoint.py`'s pattern), registered under a `_checkpoint_callbacks_test` suffix
so it cannot collide with dummy fixtures in sibling integration test files: importing a
sibling test module directly (rather than duplicating its small dummy fixture) would
make pytest import it twice under two different module names during full-suite
collection, re-running its `@registerEncoder(...)` decorators and raising "already
registered".
"""

import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.callbacks.best_checkpoint import BestCheckpointCallback
from global_vae.training.callbacks.checkpoint import CheckpointCallback
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.training.trainer import Trainer

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_checkpoint_callbacks_test")
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


@registerDecoder("dummy_signal_decoder_checkpoint_callbacks_test")
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
                "encoder": "dummy_signal_encoder_checkpoint_callbacks_test",
                "decoder": "dummy_signal_decoder_checkpoint_callbacks_test",
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
