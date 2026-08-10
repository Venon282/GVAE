"""Integration tests for `training.loggers` (spec §10 experiment tracking).

`TensorBoardLogger` is exercised against the real `tensorboard`
package (a dev/test dependency, `pyproject.toml`'s `dev` extra), not a
mock, matching this codebase's existing testing style. Uses its own
trivial linear dummy encoder/decoder registered under a `_logger_test`
suffix, mirroring `test_trainer.py`'s and `test_checkpoint.py`'s
pattern (a shared import across test files would make pytest import
the same registrations twice under full-suite collection; see
`test_checkpoint.py`'s module docstring for why).
"""

import csv
import logging
import sys

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display in this environment
import matplotlib.pyplot as plt
import pytest
import torch
from torch import nn

from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.loggers.base import AbstractExperimentLogger
from global_vae.training.loggers.csv_logger import CsvLogger
from global_vae.training.loggers.registry import (
    getLoggerClass,
    listRegisteredLoggers,
    registerLogger,
)
from global_vae.training.loggers.tensorboard_logger import TensorBoardLogger
from global_vae.training.trainer import Trainer

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8


@registerEncoder("dummy_signal_encoder_logger_test")
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


@registerDecoder("dummy_signal_decoder_logger_test")
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
                "encoder": "dummy_signal_encoder_logger_test",
                "decoder": "dummy_signal_decoder_logger_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedDataset(
    num_batches: int, input_dim: int = INPUT_DIM, seed: int = 0
) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, input_dim)} for _ in range(num_batches)]


class TestRegistry:
    def test_csv_is_registered_by_default(self) -> None:
        assert "csv" in listRegisteredLoggers()
        assert getLoggerClass("csv") is CsvLogger

    def test_tensorboard_is_registered_by_default(self) -> None:
        assert "tensorboard" in listRegisteredLoggers()
        assert getLoggerClass("tensorboard") is TensorBoardLogger

    def test_unknown_logger_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="does_not_exist"):
            getLoggerClass("does_not_exist")

    def test_duplicate_registration_raises_value_error(self) -> None:
        @registerLogger("dummy_logger_duplicate_check")
        class _First(AbstractExperimentLogger):
            def logScalar(self, name: str, value: float, x: int, tag: str = "step") -> None:
                pass

        with pytest.raises(ValueError, match="already registered"):

            @registerLogger("dummy_logger_duplicate_check")
            class _Second(AbstractExperimentLogger):
                def logScalar(self, name: str, value: float, x: int, tag: str = "step") -> None:
                    pass


class TestCsvLogger:
    def test_writes_header(self, tmp_path) -> None:
        path = tmp_path / "metrics.csv"
        csv_logger = CsvLogger(path)
        csv_logger.close()
        with path.open() as f:
            header = next(csv.reader(f))
        assert header == ["x", "tag", "metric", "value"]

    def test_log_scalar_writes_a_row(self, tmp_path) -> None:
        path = tmp_path / "metrics.csv"
        csv_logger = CsvLogger(path)
        csv_logger.logScalar("loss/total", 1.23, 5, tag="step")
        csv_logger.close()
        rows = list(csv.DictReader(path.open()))
        assert rows == [{"x": "5", "tag": "step", "metric": "loss/total", "value": "1.23"}]

    def test_log_scalars_writes_one_row_per_metric(self, tmp_path) -> None:
        path = tmp_path / "metrics.csv"
        csv_logger = CsvLogger(path)
        csv_logger.logScalars({"a": 1.0, "b": 2.0}, 3, tag="epoch")
        csv_logger.close()
        rows = list(csv.DictReader(path.open()))
        assert {(row["metric"], row["value"]) for row in rows} == {("a", "1.0"), ("b", "2.0")}
        assert all(row["tag"] == "epoch" and row["x"] == "3" for row in rows)

    def test_on_step_end_tags_step(self, tmp_path) -> None:
        path = tmp_path / "metrics.csv"
        csv_logger = CsvLogger(path)
        csv_logger.onStepEnd(None, 7, {"loss/total": 0.5})
        csv_logger.close()
        row = next(csv.DictReader(path.open()))
        assert row["tag"] == "step"
        assert row["x"] == "7"

    def test_on_epoch_end_tags_epoch(self, tmp_path) -> None:
        path = tmp_path / "metrics.csv"
        csv_logger = CsvLogger(path)
        csv_logger.onEpochEnd(None, 2, {"train/loss/total": 0.5})
        csv_logger.close()
        row = next(csv.DictReader(path.open()))
        assert row["tag"] == "epoch"
        assert row["x"] == "2"

    def test_on_train_end_closes_the_file(self, tmp_path) -> None:
        path = tmp_path / "metrics.csv"
        csv_logger = CsvLogger(path)
        csv_logger.onTrainEnd(None)
        assert csv_logger._file.closed

    def test_creates_parent_directory(self, tmp_path) -> None:
        path = tmp_path / "nested" / "dir" / "metrics.csv"
        csv_logger = CsvLogger(path)
        csv_logger.close()
        assert path.exists()

    def test_log_image_and_log_figure_warn_instead_of_raising(
        self, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "metrics.csv"
        csv_logger = CsvLogger(path)
        with caplog.at_level(logging.WARNING):
            csv_logger.logImage("recon", torch.zeros(3, 4, 4), 0)
            csv_logger.logFigure("latent", object(), 0)
        csv_logger.close()
        assert sum("does not support" in record.message for record in caplog.records) == 2

    def test_invalid_flush_every_n_rows_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="flush_every_n_rows"):
            CsvLogger(tmp_path / "metrics.csv", flush_every_n_rows=0)

    def test_full_training_run_produces_expected_row_counts(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        path = tmp_path / "metrics.csv"
        trainer = Trainer(model, device="cpu", callbacks=[CsvLogger(path)])
        trainer.fit(_fixedDataset(num_batches=2), num_epochs=3)

        rows = list(csv.DictReader(path.open()))
        step_rows = [row for row in rows if row["tag"] == "step"]
        epoch_rows = [row for row in rows if row["tag"] == "epoch"]
        assert len(step_rows) == 2 * 3 * 3  # batches * epochs * 3 step-level metrics
        assert len(epoch_rows) == 3 * 3  # epochs * 3 train-level metrics (no val_dataloader)


class TestTensorBoardLogger:
    def test_missing_tensorboard_package_raises_clear_import_error(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
        with pytest.raises(ImportError, match="tensorboard"):
            TensorBoardLogger(tmp_path / "tb_logs")

    def test_construction_creates_log_dir_and_an_event_file(self, tmp_path) -> None:
        log_dir = tmp_path / "tb_logs"
        tb_logger = TensorBoardLogger(log_dir)
        tb_logger.logScalar("loss/total", 1.0, 0)
        tb_logger.close()
        assert log_dir.exists()
        assert any(log_dir.iterdir())

    def test_log_image_does_not_raise(self, tmp_path) -> None:
        tb_logger = TensorBoardLogger(tmp_path / "tb_logs")
        tb_logger.logImage("recon", torch.rand(3, 8, 8), 0)
        tb_logger.close()

    def test_log_figure_does_not_raise(self, tmp_path) -> None:
        figure = plt.figure()
        plt.plot([0, 1], [0, 1])
        tb_logger = TensorBoardLogger(tmp_path / "tb_logs")
        try:
            tb_logger.logFigure("latent", figure, 0)
            tb_logger.close()
        finally:
            plt.close(figure)

    def test_invalid_flush_every_n_scalars_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="flush_every_n_scalars"):
            TensorBoardLogger(tmp_path / "tb_logs", flush_every_n_scalars=0)

    def test_full_training_run_does_not_raise(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        log_dir = tmp_path / "tb_logs"
        trainer = Trainer(model, device="cpu", callbacks=[TensorBoardLogger(log_dir)])
        trainer.fit(_fixedDataset(num_batches=2), num_epochs=2)
        assert log_dir.exists()


class TestComposability:
    """No dedicated "composite logger" is needed: Trainer.callbacks is already a list."""

    def test_multiple_loggers_are_both_active_at_once(self, tmp_path) -> None:
        model = _buildSingleModalityModel()
        csv_path = tmp_path / "metrics.csv"
        tb_dir = tmp_path / "tb_logs"
        trainer = Trainer(
            model,
            device="cpu",
            callbacks=[CsvLogger(csv_path), TensorBoardLogger(tb_dir)],
        )
        trainer.fit(_fixedDataset(num_batches=2), num_epochs=2)

        assert csv_path.exists()
        assert tb_dir.exists()
        assert len(list(csv.DictReader(csv_path.open()))) > 0
        assert any(tb_dir.iterdir())
