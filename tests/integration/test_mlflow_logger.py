"""Integration tests for `MlflowLogger` (spec section 10 experiment tracking, roadmap P2-4).

Everything runs in MLflow's offline mode: a SQLite file as the tracking store
(`sqlite:///.../mlflow.db`) and a local directory as the artifact location, both inside pytest's
`tmp_path`, so no server is needed and nothing is written outside the test's directory. The real
`mlflow` package is used (a dev dependency, `pyproject.toml`'s `dev` extra), not a mock, as
`test_loggers.py` does for TensorBoard. The runs are read back with a second `MlflowClient` on the
same store, which checks what MLflow holds and not what the logger believes it sent.

Uses its own trivial linear dummy encoder and decoder registered under a `_mlflow_logger_test`
suffix, following `test_loggers.py` (a shared import across test files would make pytest import the
same registrations twice under full-suite collection).
"""

import math
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display in this environment
import matplotlib.pyplot as plt
import pytest
import torch
from mlflow import MlflowClient
from torch import nn

from global_vae.config.training import LoggerEntryConfig, TrainingConfig, buildCallbacksFromConfig
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.models.global_vae import GlobalVae
from global_vae.training.loggers.csv_logger import CsvLogger
from global_vae.training.loggers.mlflow_logger import MlflowLogger, _imageToArray, _safeName
from global_vae.training.loggers.registry import getLoggerClass, listRegisteredLoggers
from global_vae.training.trainer import Trainer
from global_vae.visualization.latent_plot import collectLatentParams, plotLatentSpace
from global_vae.visualization.loss_curves import plotLossCurves

INPUT_DIM = 16
LATENT_DIM = 4
BATCH_SIZE = 8
# `MlflowLogger` never reads the trainer passed to its hooks, so the hook tests pass no real one.
_NO_TRAINER = cast(Trainer, None)


@registerEncoder("dummy_signal_encoder_mlflow_logger_test")
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


@registerDecoder("dummy_signal_decoder_mlflow_logger_test")
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
                "encoder": "dummy_signal_encoder_mlflow_logger_test",
                "decoder": "dummy_signal_decoder_mlflow_logger_test",
            },
        },
        latent_dim=LATENT_DIM,
    )


def _fixedDataset(num_batches: int, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    return [{"signal": torch.randn(BATCH_SIZE, INPUT_DIM)} for _ in range(num_batches)]


class _Store:
    """An offline MLflow store: a SQLite tracking database plus a folder for the artifacts."""

    def __init__(self, root: Path, database: Path, experiment: str) -> None:
        self.root = root
        self.uri = f"sqlite:///{database}"
        self.artifacts = root / "artifacts"
        self.experiment = experiment
        self.client = MlflowClient(tracking_uri=self.uri)

    def logger(self, **kwargs: Any) -> MlflowLogger:
        """Build a logger on this store; keyword arguments override the defaults."""
        settings: dict[str, Any] = {
            "tracking_uri": self.uri,
            "experiment_name": self.experiment,
            "artifact_location": self.artifacts,
        }
        settings.update(kwargs)
        return MlflowLogger(**settings)

    def metric(self, logger: MlflowLogger, name: str) -> list[tuple[int, float]]:
        """The `(step, value)` pairs MLflow holds for `name`, in step order."""
        history = self.client.get_metric_history(logger.run_id, name)
        return sorted((metric.step, metric.value) for metric in history)

    def artifactPaths(self, logger: MlflowLogger, folder: str | None = None) -> list[str]:
        """Artifact paths directly under `folder` (the run's root when `None`), sorted."""
        return sorted(item.path for item in self.client.list_artifacts(logger.run_id, folder))

    def status(self, logger: MlflowLogger) -> str:
        """Status MLflow holds for the logger's run."""
        return str(self.client.get_run(logger.run_id).info.status)


@pytest.fixture(scope="module")
def shared_database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One SQLite database for the module: creating the schema costs seconds, so do it once.

    Tests stay independent anyway: each one has its own experiment (named after its `tmp_path`),
    its own artifact folder, and reads its run back by id.
    """
    return tmp_path_factory.mktemp("mlflow_store") / "mlflow.db"


@pytest.fixture
def store(shared_database: Path, tmp_path: Path) -> Iterator[_Store]:
    yield _Store(tmp_path, shared_database, experiment=f"test_{tmp_path.name}")
    plt.close("all")


class TestRegistry:
    def test_mlflow_is_registered_by_default(self) -> None:
        assert "mlflow" in listRegisteredLoggers()
        assert getLoggerClass("mlflow") is MlflowLogger

    def test_the_other_builtin_loggers_are_still_registered(self) -> None:
        assert {"csv", "mlflow", "tensorboard"} <= set(listRegisteredLoggers())


class TestDeferredImport:
    @pytest.mark.slow
    def test_importing_the_loggers_package_does_not_import_mlflow(self) -> None:
        """A fresh interpreter: `import global_vae.training.loggers` must not load `mlflow`."""
        program = (
            "import sys; import global_vae.training.loggers; "
            "assert 'mlflow' not in sys.modules, 'mlflow was imported'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=120
        )
        assert completed.returncode == 0, completed.stderr

    def test_missing_mlflow_package_raises_clear_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "mlflow", None)
        with pytest.raises(ImportError, match="mlflow"):
            MlflowLogger()


class TestConstruction:
    def test_starts_a_running_run_in_the_default_experiment(self, tmp_path: Path) -> None:
        own = _Store(tmp_path, tmp_path / "own.db", experiment="unused")
        logger = MlflowLogger(tracking_uri=own.uri, artifact_location=own.artifacts)
        run = own.client.get_run(logger.run_id)
        assert run.info.status == "RUNNING"
        assert own.client.get_experiment(run.info.experiment_id).name == "global_vae"
        logger.close()

    def test_experiment_name_and_run_name_are_used(self, store: _Store) -> None:
        logger = store.logger(run_name="baseline")
        run = store.client.get_run(logger.run_id)
        assert store.client.get_experiment(run.info.experiment_id).name == store.experiment
        assert run.info.run_name == "baseline"
        logger.close()

    def test_a_second_logger_joins_the_existing_experiment_with_its_own_run(
        self, store: _Store
    ) -> None:
        first = store.logger()
        second = store.logger()
        runs = [store.client.get_run(item.run_id).info for item in (first, second)]
        assert runs[0].experiment_id == runs[1].experiment_id
        assert first.run_id != second.run_id
        first.close()
        second.close()

    def test_tags_and_params_are_recorded_as_strings(self, store: _Store) -> None:
        logger = store.logger(tags={"dataset": "saxs_v2", "fold": 3}, params={"beta": 0.5, "n": 7})
        data = store.client.get_run(logger.run_id).data
        assert data.tags["dataset"] == "saxs_v2"
        assert data.tags["fold"] == "3"
        assert data.params == {"beta": "0.5", "n": "7"}
        logger.close()

    def test_a_path_artifact_location_becomes_an_absolute_uri(self, store: _Store) -> None:
        logger = store.logger()
        run = store.client.get_run(logger.run_id)
        experiment = store.client.get_experiment(run.info.experiment_id)
        assert experiment.artifact_location == store.artifacts.resolve().as_uri()
        logger.close()

    def test_a_string_artifact_location_is_passed_through(self, store: _Store) -> None:
        location = (store.root / "plain_string_location").resolve().as_uri()
        logger = store.logger(artifact_location=location)
        run = store.client.get_run(logger.run_id)
        assert store.client.get_experiment(run.info.experiment_id).artifact_location == location
        logger.close()

    def test_an_experiment_created_by_another_process_in_the_meantime_is_reused(
        self, store: _Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lookup misses, the creation then fails (it exists), and the second lookup wins."""
        existing = store.client.create_experiment(store.experiment)
        original = MlflowClient.get_experiment_by_name
        calls: list[str] = []

        def flakyLookup(self: MlflowClient, name: str) -> Any:
            calls.append(name)
            return None if len(calls) == 1 else original(self, name)

        monkeypatch.setattr(MlflowClient, "get_experiment_by_name", flakyLookup)
        logger = store.logger()
        assert store.client.get_run(logger.run_id).info.experiment_id == existing
        assert len(calls) == 2
        logger.close()

    def test_a_creation_failure_that_is_not_a_race_is_raised(
        self, store: _Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failingCreation(self: MlflowClient, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("the store is read-only")

        monkeypatch.setattr(MlflowClient, "create_experiment", failingCreation)
        with pytest.raises(RuntimeError, match="read-only"):
            store.logger()


class TestScalars:
    def test_log_scalar_stores_the_value_at_the_step(self, store: _Store) -> None:
        logger = store.logger()
        logger.logScalar("loss/total", 1.5, 3)
        logger.logScalar("loss/total", 1.25, 4)
        assert store.metric(logger, "loss/total") == [(3, 1.5), (4, 1.25)]
        logger.close()

    def test_tag_does_not_change_the_metric_name(self, store: _Store) -> None:
        logger = store.logger()
        logger.logScalar("train/loss/total", 2.0, 1, tag="epoch")
        assert store.metric(logger, "train/loss/total") == [(1, 2.0)]
        logger.close()

    def test_log_scalars_sends_every_metric_in_one_request(
        self, store: _Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logger = store.logger()
        requests: list[int] = []
        original = MlflowClient.log_batch

        def countingBatch(self: MlflowClient, run_id: str, **kwargs: Any) -> Any:
            requests.append(len(kwargs.get("metrics", ())))
            return original(self, run_id, **kwargs)

        monkeypatch.setattr(MlflowClient, "log_batch", countingBatch)
        logger.logScalars({"loss/total": 3.0, "loss/reconstruction": 2.0, "loss/kl": 1.0}, 5)
        assert requests == [3]
        assert store.metric(logger, "loss/reconstruction") == [(5, 2.0)]
        logger.close()

    def test_a_large_dict_is_split_into_requests_of_the_mlflow_limit(
        self, store: _Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("global_vae.training.loggers.mlflow_logger._MAX_METRICS_PER_BATCH", 2)
        logger = store.logger()
        requests: list[int] = []
        original = MlflowClient.log_batch

        def countingBatch(self: MlflowClient, run_id: str, **kwargs: Any) -> Any:
            requests.append(len(kwargs.get("metrics", ())))
            return original(self, run_id, **kwargs)

        monkeypatch.setattr(MlflowClient, "log_batch", countingBatch)
        logger.logScalars({f"m{index}": float(index) for index in range(5)}, 0)
        assert requests == [2, 2, 1]
        assert store.metric(logger, "m4") == [(0, 4.0)]
        logger.close()

    def test_an_empty_dict_sends_no_request(
        self, store: _Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logger = store.logger()

        def forbiddenBatch(self: MlflowClient, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("log_batch must not be called for an empty dict")

        monkeypatch.setattr(MlflowClient, "log_batch", forbiddenBatch)
        logger.logScalars({}, 0)
        logger.close()

    def test_nan_and_infinity_are_stored_without_error(self, store: _Store) -> None:
        logger = store.logger()
        logger.logScalars({"nan": float("nan"), "inf": float("inf")}, 0)
        assert math.isnan(store.metric(logger, "nan")[0][1])
        # MLflow stores an infinite value as the largest float.
        assert store.metric(logger, "inf")[0][1] > 1e300
        logger.close()

    def test_on_step_end_and_on_epoch_end_log_their_metrics(self, store: _Store) -> None:
        logger = store.logger()
        logger.onStepEnd(_NO_TRAINER, 7, {"loss/total": 0.5})
        logger.onEpochEnd(_NO_TRAINER, 2, {"train/loss/total": 0.75})
        assert store.metric(logger, "loss/total") == [(7, 0.5)]
        assert store.metric(logger, "train/loss/total") == [(2, 0.75)]
        logger.close()


class TestImages:
    @pytest.mark.parametrize("shape", [(3, 8, 8), (1, 8, 8), (4, 8, 8), (8, 8)])
    def test_log_image_stores_a_png_for_each_supported_shape(
        self, store: _Store, shape: tuple[int, ...]
    ) -> None:
        logger = store.logger()
        logger.logImage("recon/signal", torch.rand(*shape), 3)
        paths = store.artifactPaths(logger, "images")
        # MLflow spells the step into the file name with a separator that changed between
        # versions (`step+3`, `step%3`), so the pattern accepts any non-word character.
        assert any(path.endswith(".png") and re.search(r"step\W3\W", path) for path in paths), paths
        logger.close()

    def test_one_image_per_step_accumulates(self, store: _Store) -> None:
        logger = store.logger()
        for step in range(3):
            logger.logImage("recon", torch.rand(3, 8, 8), step)
        pngs = [path for path in store.artifactPaths(logger, "images") if path.endswith(".png")]
        assert len(pngs) == 3
        logger.close()

    def test_a_uint8_image_and_a_tensor_that_requires_grad_are_accepted(
        self, store: _Store
    ) -> None:
        logger = store.logger()
        logger.logImage("raw", torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8), 0)
        logger.logImage("grad", torch.rand(3, 8, 8, requires_grad=True), 0)
        assert len(store.artifactPaths(logger, "images")) >= 2
        logger.close()

    def test_a_name_with_characters_mlflow_refuses_in_a_key_is_sanitized(
        self, store: _Store
    ) -> None:
        logger = store.logger()
        logger.logImage("recon:signal (e)", torch.rand(3, 4, 4), 0)
        assert store.artifactPaths(logger, "images")
        logger.close()

    @pytest.mark.parametrize("shape", [(2, 4, 4), (5, 4, 4), (4,), (1, 3, 4, 4)])
    def test_an_unsupported_shape_raises_value_error(self, store: _Store, shape: Any) -> None:
        logger = store.logger()
        with pytest.raises(ValueError, match="logImage expects"):
            logger.logImage("bad", torch.rand(*shape), 0)
        logger.close()

    def test_an_integer_dtype_other_than_uint8_raises_type_error(self, store: _Store) -> None:
        logger = store.logger()
        with pytest.raises(TypeError, match="uint8 or floating-point"):
            logger.logImage("bad", torch.zeros(3, 4, 4, dtype=torch.int64), 0)
        logger.close()


class TestImageConversion:
    def test_channels_move_last(self) -> None:
        array = _imageToArray(torch.rand(3, 5, 7))
        assert array.shape == (5, 7, 3)

    def test_a_single_channel_becomes_a_two_dimensional_array(self) -> None:
        assert _imageToArray(torch.rand(1, 5, 7)).shape == (5, 7)
        assert _imageToArray(torch.rand(5, 7)).shape == (5, 7)

    def test_floats_outside_zero_one_are_clipped_not_rescaled(self) -> None:
        image = torch.tensor([[[-1.0, 0.25], [0.75, 2.0]]])
        array = _imageToArray(image)
        assert array.min() == 0.0
        assert array.max() == 1.0
        assert array[0, 1] == pytest.approx(0.25)
        assert array[1, 0] == pytest.approx(0.75)

    def test_uint8_values_are_kept(self) -> None:
        image = torch.tensor([[[0, 128], [200, 255]]], dtype=torch.uint8)
        array = _imageToArray(image)
        assert array.dtype.name == "uint8"
        assert array.tolist() == [[0, 128], [200, 255]]

    def test_the_input_is_not_modified(self) -> None:
        image = torch.tensor([[[-1.0, 2.0]]])
        _imageToArray(image)
        assert image.tolist() == [[[-1.0, 2.0]]]


class TestFigures:
    def test_log_figure_stores_a_png_named_after_tag_and_step(self, store: _Store) -> None:
        logger = store.logger()
        figure = plt.figure()
        plt.plot([0, 1], [0, 1])
        logger.logFigure("latent/pca", figure, 7)
        assert store.artifactPaths(logger, "figures/latent/pca") == [
            "figures/latent/pca/step_00000007.png"
        ]
        logger.close()

    def test_step_and_epoch_of_one_figure_do_not_overwrite_each_other(self, store: _Store) -> None:
        logger = store.logger()
        figure = plt.figure()
        logger.logFigure("latent", figure, 3, tag="step")
        logger.logFigure("latent", figure, 3, tag="epoch")
        assert store.artifactPaths(logger, "figures/latent") == [
            "figures/latent/epoch_00000003.png",
            "figures/latent/step_00000003.png",
        ]
        logger.close()

    def test_the_figure_is_left_open_for_the_caller(self, store: _Store) -> None:
        logger = store.logger()
        figure = plt.figure()
        logger.logFigure("latent", figure, 0)
        assert plt.fignum_exists(figure.number)
        logger.close()

    def test_a_name_cannot_climb_out_of_the_artifact_directory(self, store: _Store) -> None:
        logger = store.logger()
        figure = plt.figure()
        logger.logFigure("../../escape", figure, 0)
        assert store.artifactPaths(logger, "figures/escape") == ["figures/escape/step_00000000.png"]
        assert not (store.root / "escape").exists()
        logger.close()


class TestSafeName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("recon/signal", "recon/signal"),
            ("loss total-1.5", "loss total-1.5"),
            ("a:b", "a_b"),
            ("../../x", "x"),
            ("a/./b", "a/b"),
            ("a\\b", "a/b"),
            ("/leading/slash/", "leading/slash"),
            ("", "unnamed"),
            ("..", "unnamed"),
            ("   ", "unnamed"),
            ("é", "_"),
        ],
    )
    def test_names(self, name: str, expected: str) -> None:
        assert _safeName(name) == expected


class TestLifecycle:
    def test_close_ends_the_run_as_finished(self, store: _Store) -> None:
        logger = store.logger()
        logger.close()
        run = store.client.get_run(logger.run_id)
        assert run.info.status == "FINISHED"
        assert run.info.end_time is not None

    def test_close_twice_is_harmless(self, store: _Store) -> None:
        logger = store.logger()
        logger.close()
        logger.close()
        assert store.status(logger) == "FINISHED"

    def test_on_train_end_closes_the_run(self, store: _Store) -> None:
        logger = store.logger()
        logger.onTrainEnd(_NO_TRAINER)
        assert store.status(logger) == "FINISHED"

    def test_a_finished_run_still_accepts_scalars_images_and_figures(self, store: _Store) -> None:
        """MLflow does not lock a finished run: the final model's figures can follow `fit`."""
        logger = store.logger()
        logger.close()
        logger.logScalar("loss/total", 1.0, 0)
        logger.logImage("recon", torch.rand(3, 4, 4), 0)
        logger.logFigure("latent", plt.figure(), 0)
        assert store.metric(logger, "loss/total") == [(0, 1.0)]
        assert store.artifactPaths(logger, "images")
        assert store.artifactPaths(logger, "figures/latent")
        assert store.status(logger) == "FINISHED"

    def test_on_train_end_inside_an_error_ends_the_run_as_failed(self, store: _Store) -> None:
        logger = store.logger()
        try:
            try:
                raise ValueError("training broke")
            finally:
                logger.onTrainEnd(_NO_TRAINER)
        except ValueError:
            pass
        assert store.status(logger) == "FAILED"

    def test_on_train_end_inside_a_keyboard_interrupt_ends_the_run_as_killed(
        self, store: _Store
    ) -> None:
        logger = store.logger()
        try:
            try:
                raise KeyboardInterrupt
            finally:
                logger.onTrainEnd(_NO_TRAINER)
        except KeyboardInterrupt:
            pass
        assert store.status(logger) == "KILLED"

    def test_an_explicit_close_is_finished_even_inside_an_error(self, store: _Store) -> None:
        """Only `onTrainEnd` reads the exception: a direct `close()` always means finished."""
        logger = store.logger()
        try:
            try:
                raise ValueError("unrelated")
            finally:
                logger.close()
        except ValueError:
            pass
        assert store.status(logger) == "FINISHED"


class _FailingDataloader:
    """Yields one batch, then fails: a training run that dies in its first epoch."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield {"signal": torch.randn(BATCH_SIZE, INPUT_DIM)}
        raise self._error


class TestTrainerRun:
    def test_a_full_fit_logs_steps_epochs_and_validation_and_finishes_the_run(
        self, store: _Store
    ) -> None:
        logger = store.logger(run_name="fit")
        trainer = Trainer(_buildSingleModalityModel(), device="cpu", callbacks=[logger])
        trainer.fit(_fixedDataset(num_batches=2), num_epochs=3, val_dataloader=_fixedDataset(1, 1))

        assert len(store.metric(logger, "loss/total")) == 2 * 3  # batches * epochs
        assert [step for step, _ in store.metric(logger, "train/loss/total")] == [0, 1, 2]
        assert [step for step, _ in store.metric(logger, "val/loss/total")] == [0, 1, 2]
        assert store.status(logger) == "FINISHED"

    def test_metric_values_match_the_trainer_history(self, store: _Store) -> None:
        logger = store.logger()
        trainer = Trainer(_buildSingleModalityModel(), device="cpu", callbacks=[logger])
        history = trainer.fit(_fixedDataset(num_batches=2), num_epochs=2)
        logged = dict(store.metric(logger, "train/loss/total"))
        for epoch, metrics in enumerate(history):
            assert logged[epoch] == pytest.approx(metrics["train/loss/total"])

    def test_a_run_that_raises_is_marked_failed_and_the_error_propagates(
        self, store: _Store
    ) -> None:
        logger = store.logger()
        trainer = Trainer(_buildSingleModalityModel(), device="cpu", callbacks=[logger])
        with pytest.raises(RuntimeError, match="out of memory"):
            trainer.fit(_FailingDataloader(RuntimeError("out of memory")), num_epochs=2)
        assert store.status(logger) == "FAILED"

    def test_an_interrupted_run_is_marked_killed(self, store: _Store) -> None:
        logger = store.logger()
        trainer = Trainer(_buildSingleModalityModel(), device="cpu", callbacks=[logger])
        with pytest.raises(KeyboardInterrupt):
            trainer.fit(_FailingDataloader(KeyboardInterrupt()), num_epochs=2)
        assert store.status(logger) == "KILLED"

    def test_it_runs_next_to_another_logger(self, store: _Store, tmp_path: Path) -> None:
        mlflow_logger = store.logger()
        csv_path = tmp_path / "metrics.csv"
        trainer = Trainer(
            _buildSingleModalityModel(),
            device="cpu",
            callbacks=[CsvLogger(csv_path), mlflow_logger],
        )
        trainer.fit(_fixedDataset(num_batches=2), num_epochs=2)
        assert len(csv_path.read_text().splitlines()) > 1
        assert len(store.metric(mlflow_logger, "loss/total")) == 4

    def test_a_second_fit_continues_the_same_run_and_sets_its_own_status(
        self, store: _Store
    ) -> None:
        logger = store.logger()
        trainer = Trainer(_buildSingleModalityModel(), device="cpu", callbacks=[logger])
        trainer.fit(_fixedDataset(num_batches=1), num_epochs=1)
        assert store.status(logger) == "FINISHED"
        trainer.fit(_fixedDataset(num_batches=1), num_epochs=2)
        assert [step for step, _ in store.metric(logger, "train/loss/total")] == [0, 1, 2]
        with pytest.raises(RuntimeError, match="out of memory"):
            trainer.fit(_FailingDataloader(RuntimeError("out of memory")), num_epochs=1)
        assert store.status(logger) == "FAILED"

    def test_the_figures_of_the_final_model_are_logged_after_fit(self, store: _Store) -> None:
        """The pattern of `how-to/train.md`: plots from `global_vae.visualization` after `fit`."""
        logger = store.logger()
        model = _buildSingleModalityModel()
        trainer = Trainer(model, device="cpu", callbacks=[logger])
        data = _fixedDataset(num_batches=2)
        trainer.fit(data, num_epochs=2)

        mu, _ = collectLatentParams(model, data, "z_fused")
        logger.logFigure("latent/pca", plotLatentSpace(mu), trainer.global_step)
        logger.logFigure("loss_curves", plotLossCurves(trainer.history), trainer.global_step)
        assert store.artifactPaths(logger, "figures/latent/pca") == [
            f"figures/latent/pca/step_{trainer.global_step:08d}.png"
        ]
        assert store.artifactPaths(logger, "figures/loss_curves")


class TestBuiltFromConfig:
    def test_the_registry_name_and_kwargs_reach_the_logger(self, store: _Store) -> None:
        config = TrainingConfig(
            loggers=[
                LoggerEntryConfig(
                    name="mlflow",
                    kwargs={
                        "tracking_uri": store.uri,
                        "artifact_location": str(store.artifacts),
                        "experiment_name": store.experiment,
                        "run_name": "r1",
                        "params": {"beta": 0.1},
                    },
                )
            ]
        )
        (callback,) = buildCallbacksFromConfig(config)
        assert isinstance(callback, MlflowLogger)
        run = store.client.get_run(callback.run_id)
        assert run.info.run_name == "r1"
        assert run.data.params == {"beta": "0.1"}
        callback.close()

    def test_a_resumed_run_still_builds_the_logger(self, store: _Store) -> None:
        """`append` goes only to loggers that accept it (ADR 0032): this one starts a new run."""
        config = TrainingConfig(
            resume_from="unused.pt",
            loggers=[
                LoggerEntryConfig(
                    name="mlflow",
                    kwargs={
                        "tracking_uri": store.uri,
                        "experiment_name": store.experiment,
                        "artifact_location": str(store.artifacts),
                    },
                )
            ],
        )
        (callback,) = buildCallbacksFromConfig(config)
        assert isinstance(callback, MlflowLogger)
        callback.close()
