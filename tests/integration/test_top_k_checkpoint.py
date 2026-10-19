"""Integration tests for `TopKCheckpointCallback` (roadmap P2-3, ADR 0033).

The callback ranks epochs by a monitored metric and keeps the files of the best `k`. As in
`test_reduce_lr_on_plateau.py`, the metric is overwritten by a small callback placed before it
with a hand-picked sequence, so that which epochs are the best is known in advance and does
not depend on the real loss of a dummy model.
"""

import logging
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.callbacks.registry import getCallbackClass
from global_vae.training.callbacks.top_k_checkpoint import TopKCheckpointCallback
from global_vae.training.checkpoint import loadCheckpoint
from global_vae.training.trainer import Trainer

INPUT_DIM = 8
LATENT_DIM = 2
BATCH_SIZE = 4


@registerEncoder("dummy_signal_encoder_top_k_test")
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


@registerDecoder("dummy_signal_decoder_top_k_test")
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


class _Forced(TrainerCallback):
    def __init__(self, key: str, values: list[float]) -> None:
        self.key = key
        self._values = iter(values)

    def onEpochEnd(self, trainer: Trainer, epoch: int, metrics: dict[str, float]) -> None:
        metrics[self.key] = next(self._values)


def _trainer(callbacks: list[TrainerCallback] | None = None) -> Trainer:
    torch.manual_seed(0)
    model = GlobalVae.createSingleLatent(
        modality_configs={
            "signal": {
                "encoder": "dummy_signal_encoder_top_k_test",
                "decoder": "dummy_signal_decoder_top_k_test",
            }
        },
        latent_dim=LATENT_DIM,
    )
    return Trainer(model, device="cpu", callbacks=callbacks, log_every_n_steps=1000)


def _data() -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(1)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)}]


def _run(
    tmp_path: Path, values: list[float], **kwargs: object
) -> tuple[TopKCheckpointCallback, Trainer]:
    callback = TopKCheckpointCallback(tmp_path, monitor="forced", **kwargs)  # type: ignore[arg-type]
    trainer = _trainer([_Forced("forced", values), callback])
    trainer.fit(_data(), num_epochs=len(values))
    return callback, trainer


def _names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


class TestRanking:
    def test_it_keeps_the_k_lowest_values_in_min_mode(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [4.0, 2.0, 5.0, 1.0, 3.0], k=2)
        assert [value for value, _ in callback.entries] == [1.0, 2.0]
        assert _names(tmp_path) == ["top_k_epoch_0001.pt", "top_k_epoch_0003.pt"]

    def test_it_keeps_the_k_highest_values_in_max_mode(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [4.0, 2.0, 5.0, 1.0, 3.0], k=2, mode="max")
        assert [value for value, _ in callback.entries] == [5.0, 4.0]
        assert _names(tmp_path) == ["top_k_epoch_0000.pt", "top_k_epoch_0002.pt"]

    def test_the_entries_and_paths_are_listed_best_first(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [3.0, 1.0, 2.0], k=3)
        assert [path.name for path in callback.paths] == [
            "top_k_epoch_0001.pt",
            "top_k_epoch_0002.pt",
            "top_k_epoch_0000.pt",
        ]

    def test_fewer_epochs_than_k_keeps_every_epoch(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [3.0, 1.0], k=5)
        assert len(callback.paths) == 2
        assert len(_names(tmp_path)) == 2

    def test_a_value_equal_to_the_worst_kept_does_not_displace_it(self, tmp_path: Path) -> None:
        _run(tmp_path, [2.0, 1.0, 2.0, 2.0], k=2)
        assert _names(tmp_path) == ["top_k_epoch_0000.pt", "top_k_epoch_0001.pt"]

    def test_a_tie_with_a_better_kept_value_ranks_the_earlier_epoch_first(
        self, tmp_path: Path
    ) -> None:
        callback, _ = _run(tmp_path, [1.0, 1.0, 1.0], k=3)
        assert [path.name for path in callback.paths] == [
            "top_k_epoch_0000.pt",
            "top_k_epoch_0001.pt",
            "top_k_epoch_0002.pt",
        ]

    def test_a_tie_ranks_the_earlier_epoch_first_in_max_mode_too(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [1.0, 1.0], k=2, mode="max")
        assert [path.name for path in callback.paths] == [
            "top_k_epoch_0000.pt",
            "top_k_epoch_0001.pt",
        ]

    def test_k_equal_to_one_behaves_like_a_best_checkpoint_with_one_file_per_best_epoch(
        self, tmp_path: Path
    ) -> None:
        _run(tmp_path, [3.0, 2.0, 4.0, 1.0], k=1)
        assert _names(tmp_path) == ["top_k_epoch_0003.pt"]

    def test_the_evicted_file_is_deleted(self, tmp_path: Path) -> None:
        _run(tmp_path, [3.0, 2.0, 1.0], k=2)
        assert not (tmp_path / "top_k_epoch_0000.pt").exists()


class TestTheKeptFiles:
    def test_each_file_is_a_full_checkpoint_of_its_epoch(self, tmp_path: Path) -> None:
        callback, trainer = _run(tmp_path, [3.0, 1.0, 2.0], k=2)
        best = callback.paths[0]
        restored = _trainer()
        restored.loadCheckpoint(best)
        assert restored.start_epoch == 2  # saved at the end of the epoch with index 1
        assert len(restored.history) == 2

    def test_a_kept_checkpoint_loads_for_evaluation_only(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [1.0], k=1)
        model = _trainer().model
        metadata = loadCheckpoint(callback.paths[0], model=model)
        assert metadata.start_epoch == 1

    def test_the_file_name_pattern_and_directory_are_used(self, tmp_path: Path) -> None:
        directory = tmp_path / "nested" / "dir"
        callback = TopKCheckpointCallback(
            directory, k=2, monitor="forced", filename_pattern="ep{epoch}.pt"
        )
        trainer = _trainer([_Forced("forced", [2.0, 1.0]), callback])
        trainer.fit(_data(), num_epochs=2)
        assert _names(directory) == ["ep0.pt", "ep1.pt"]

    def test_the_config_snapshot_is_stored_in_each_file(self, tmp_path: Path) -> None:
        callback = TopKCheckpointCallback(
            tmp_path, k=1, monitor="forced", config={"experiment": "demo", "lr": 0.1}
        )
        trainer = _trainer([_Forced("forced", [1.0]), callback])
        trainer.fit(_data(), num_epochs=1)
        raw = torch.load(callback.paths[0], map_location="cpu", weights_only=True)
        assert raw["config"] == {"experiment": "demo", "lr": 0.1}

    def test_a_kept_file_lists_the_callback_in_its_own_state(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [3.0, 1.0], k=2)
        raw = torch.load(callback.paths[0], map_location="cpu", weights_only=True)
        states = {entry["class"]: entry["state"] for entry in raw["callbacks_state"]}
        assert len(states["TopKCheckpointCallback"]["entries"]) == 2


class TestEdgeCases:
    def test_a_non_finite_value_is_never_kept_and_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            callback, _ = _run(tmp_path, [math.nan, 2.0, math.inf, -math.inf], k=3)
        assert [value for value, _ in callback.entries] == [2.0]
        assert "not keeping a checkpoint" in caplog.text

    def test_a_missing_monitor_is_a_key_error_naming_the_keys(self, tmp_path: Path) -> None:
        callback = TopKCheckpointCallback(tmp_path, monitor="val/loss/total")
        trainer = _trainer([callback])
        with pytest.raises(KeyError, match=r"val/loss/total.*val_dataloader"):
            trainer.fit(_data(), num_epochs=1)

    def test_it_monitors_the_validation_loss_by_default(self, tmp_path: Path) -> None:
        callback = TopKCheckpointCallback(tmp_path, k=2)
        trainer = _trainer([callback])
        trainer.fit(_data(), num_epochs=3, val_dataloader=_data())
        assert len(callback.entries) == 2
        assert all(math.isfinite(value) for value, _ in callback.entries)

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_must_be_positive(self, tmp_path: Path, k: int) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            TopKCheckpointCallback(tmp_path, k=k)

    def test_the_mode_must_be_min_or_max(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mode"):
            TopKCheckpointCallback(tmp_path, mode="sideways")

    def test_the_file_name_must_depend_on_the_epoch(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="epoch"):
            TopKCheckpointCallback(tmp_path, filename_pattern="best.pt")

    def test_a_directory_that_does_not_exist_is_created(self, tmp_path: Path) -> None:
        directory = tmp_path / "a" / "b"
        _run(directory, [1.0], k=1)
        assert directory.is_dir()


class TestState:
    def test_the_state_round_trips(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [3.0, 1.0, 2.0], k=2)
        restored = TopKCheckpointCallback(tmp_path, k=2, monitor="forced")
        restored.loadStateDict(callback.stateDict())
        assert restored.entries == callback.entries

    def test_the_state_is_made_of_plain_values(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [3.0, 1.0], k=2)
        state = callback.stateDict()
        assert state == {
            "entries": [
                [1.0, str(tmp_path / "top_k_epoch_0001.pt")],
                [3.0, str(tmp_path / "top_k_epoch_0000.pt")],
            ]
        }

    def test_the_entries_property_is_a_copy(self, tmp_path: Path) -> None:
        callback, _ = _run(tmp_path, [3.0, 1.0], k=2)
        callback.entries.clear()
        assert len(callback.entries) == 2


class TestRegistry:
    def test_it_is_registered_under_its_config_name(self) -> None:
        assert getCallbackClass("top_k_checkpoint") is TopKCheckpointCallback
