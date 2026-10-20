"""MLflow experiment logger, wrapping `mlflow.MlflowClient` (roadmap P2-4, decision D-8).

The `mlflow` PyPI package is a soft dependency of this module: importing `training.loggers`
(and hence this file, for its `@registerLogger` side effect) never requires it, since the
imports of `MlflowClient` and of the entity classes are deferred to `MlflowLogger.__init__`.
The reason is the one given in `tensorboard_logger.py`: `training.loggers.__init__` imports every
concrete logger to populate the registry, so a module-level `import mlflow` would break
`getLoggerClass("csv")` in an environment without MLflow. (`import mlflow` is also slow, which is
a second reason not to pay for it when no MLflow logger is built.) `pyproject.toml`'s `mlflow`
extra installs it: `pip install -e ".[mlflow]"`.

What is logged, per run (spec section 10, "losses, latent-space visualizations, and
reconstructions"):

- scalars (`logScalar`, `logScalars`, hence every `Trainer` step and epoch): MLflow metrics,
  with `x` as the metric's `step`. All the metrics of one call travel in a single `log_batch`
  request, so one `Trainer` step costs one request to a tracking server, not one per metric.
  `tag` does not change the metric name (a step-level `"loss/total"` and an epoch-level
  `"train/loss/total"` are different names already), exactly as in `TensorBoardLogger`.
- figures (`logFigure`): a matplotlib figure, saved as the artifact
  `figures/<name>/<tag>_<x>.png`. `global_vae.visualization` returns such figures (latent
  projections, loss curves, reconstructions). The tag is part of the file name because a
  figure is a file: epoch 3 and step 3 of one name must not overwrite each other.
- images (`logImage`): a `(C, H, W)` tensor, logged with `MlflowClient.log_image(key=name,
  step=x)`, which the MLflow UI shows in its image grid, one image per step.

Offline mode: MLflow needs no server. A `tracking_uri` of the form `sqlite:///path/to/mlflow.db`
keeps runs in a local SQLite file, and `artifact_location` keeps figures and images in a local
directory; nothing leaves the machine, and `mlflow ui --backend-store-uri sqlite:///...` reads
the result later. The tests of this module run that way. MLflow's older local backend, a
`./mlruns` directory of files (`file:` URIs), is in maintenance mode and refused by MLflow 3
unless `MLFLOW_ALLOW_FILE_STORE=true` is set, so it is not the documented route.

The logger talks to an explicit `MlflowClient` and a run id, never to MLflow's fluent API
(`mlflow.start_run`, `mlflow.log_metric`), which keeps a process-wide "active run": two loggers,
or a logger and the user's own `mlflow` calls, would otherwise step on each other.
"""

import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from global_vae.training.loggers.base import AbstractExperimentLogger
from global_vae.training.loggers.registry import registerLogger

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.entities import Metric

    from global_vae.training.trainer import Trainer

# MLflow refuses a `log_batch` request of more than 1000 metrics.
_MAX_METRICS_PER_BATCH = 1000
# Characters MLflow accepts in an image key, minus the ones that make little sense in a name.
_UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_.\- ]")
# Channel counts that `logImage` can turn into a grayscale, RGB or RGBA picture.
_IMAGE_CHANNELS = (1, 3, 4)


def _safeName(name: str) -> str:
    """Turn a metric-like name into a relative path that MLflow accepts as a key or artifact path.

    `/` stays a separator (`"recon/signal"` becomes a folder), any other character outside
    letters, digits, `_`, `.`, `-` and space becomes `_`, and empty or dot-only segments
    (`""`, `"."`, `".."`) are dropped, so a name can never climb out of the run's artifact
    directory. A name with nothing left becomes `"unnamed"`.
    """
    segments = []
    for segment in name.replace("\\", "/").split("/"):
        cleaned = _UNSAFE_NAME_CHARACTERS.sub("_", segment).strip()
        if cleaned and set(cleaned) != {"."}:
            segments.append(cleaned)
    return "/".join(segments) or "unnamed"


def _imageToArray(image: torch.Tensor) -> NDArray[Any]:
    """Convert a `(C, H, W)` (or `(H, W)`) tensor into the array `MlflowClient.log_image` takes.

    Args:
        image: A `uint8` tensor, or a floating-point tensor whose values are meant to lie in
            `[0, 1]`. `C` is 1 (grayscale), 3 (RGB) or 4 (RGBA).

    Returns:
        An `(H, W)` array for one channel, an `(H, W, C)` array otherwise. A floating-point
        image is clipped to `[0, 1]` first, since an image file cannot hold anything else (and
        MLflow would otherwise rescale it by its own min and max, with a warning, so two
        reconstructions of one run would end up on different scales).

    Raises:
        ValueError: If the shape is not `(C, H, W)` with `C` in `(1, 3, 4)` or `(H, W)`.
        TypeError: If the dtype is neither `uint8` nor floating point.
    """
    tensor = image.detach().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[0] not in _IMAGE_CHANNELS:
        raise ValueError(
            "logImage expects a (C, H, W) tensor with C in (1, 3, 4), or an (H, W) tensor, "
            f"got shape {tuple(image.shape)}."
        )
    if tensor.dtype != torch.uint8:
        if not tensor.is_floating_point():
            raise TypeError(
                f"logImage expects a uint8 or floating-point tensor, got dtype {tensor.dtype}."
            )
        tensor = tensor.to(torch.float32).clamp(0.0, 1.0)

    array: NDArray[Any] = np.asarray(tensor.permute(1, 2, 0).contiguous().numpy())
    return array[:, :, 0] if array.shape[2] == 1 else array


@registerLogger("mlflow")
class MlflowLogger(AbstractExperimentLogger):
    """Logs scalars, images and figures of one training run to MLflow.

    The run is created when the logger is built, and its status is set when training ends:
    `Trainer.fit` calls `onTrainEnd` (even when training raises). A finished run is not locked,
    in MLflow or here: the logger keeps accepting scalars, images and figures afterwards, so
    the figures of the final model can be logged once `fit` has returned, and a second `fit`
    on the same trainer continues the same run. `logImage` and `logFigure` are both supported,
    unlike `CsvLogger`.

    Not covered: a resumed run (`training.resume_from`, ADR 0032) starts a new MLflow run in the
    same experiment, since there is no run to continue unless the caller names one; the
    `append` argument of `CsvLogger` has no counterpart here.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "global_vae",
        run_name: str | None = None,
        artifact_location: str | Path | None = None,
        tags: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Connect to a tracking store, find or create the experiment, and start a run.

        Args:
            tracking_uri: Where runs are stored: `"http://host:5000"` for a tracking server, or
                `"sqlite:///path/to/mlflow.db"` for a local file and no server (the offline
                mode; the parent directory is created by MLflow). `None` (default) leaves the
                choice to MLflow: the `MLFLOW_TRACKING_URI` environment variable, then
                `mlflow.set_tracking_uri`, then MLflow's own default (a `mlflow.db` file in
                the working directory in MLflow 3).
            experiment_name: Experiment the run belongs to, created if it does not exist.
            run_name: Name shown in the MLflow UI. `None` (default) lets MLflow pick one.
            artifact_location: Where figures and images are written, for an experiment this
                call creates: a directory or an artifact store URI (`"s3://bucket/path"`). A
                `Path` is resolved to an absolute `file:` URI. Ignored when the experiment
                already exists (MLflow fixes it at creation). `None` leaves MLflow's default,
                a `mlruns` directory in the working directory.
            tags: Tags set on the run (for example `{"dataset": "saxs_v2"}`). Values are
                converted with `str`.
            params: Hyperparameters recorded once, at the start of the run (MLflow parameters
                cannot change afterwards). Values are converted with `str`; MLflow accepts at
                most 100 parameters here, and nested values should be flattened by the caller.

        Raises:
            ImportError: If the `mlflow` package is not installed. Install it with
                `pip install mlflow` or `pip install -e ".[mlflow]"`.
        """
        try:
            from mlflow import MlflowClient as _MlflowClient
            from mlflow.entities import Metric as _Metric
            from mlflow.entities import Param as _Param
        except ImportError as error:
            raise ImportError(
                "MlflowLogger requires the 'mlflow' package, which is not installed. "
                'Install it with `pip install mlflow` or `pip install -e ".[mlflow]"`.'
            ) from error

        self._client: MlflowClient = _MlflowClient(tracking_uri=tracking_uri)
        self._metric_class: type[Metric] = _Metric

        if isinstance(artifact_location, Path):
            artifact_location = artifact_location.resolve().as_uri()
        experiment_id = self._getOrCreateExperiment(experiment_name, artifact_location)

        run = self._client.create_run(
            experiment_id,
            run_name=run_name,
            tags={key: str(value) for key, value in (tags or {}).items()},
        )
        self._run_id: str = run.info.run_id
        if params:
            self._client.log_batch(
                self._run_id,
                params=[_Param(key, str(value)) for key, value in params.items()],
            )

    @property
    def run_id(self) -> str:
        """MLflow id of the run this logger writes to (to query it or attach more artifacts)."""
        return self._run_id

    def _getOrCreateExperiment(self, name: str, artifact_location: str | None) -> str:
        """Return the id of experiment `name`, creating it when it does not exist yet."""
        experiment = self._client.get_experiment_by_name(name)
        if experiment is not None:
            return str(experiment.experiment_id)
        try:
            return str(self._client.create_experiment(name, artifact_location=artifact_location))
        except Exception:
            # Another process created it between the lookup and the creation (parallel runs
            # of one sweep share an experiment name); anything else is still an error.
            experiment = self._client.get_experiment_by_name(name)
            if experiment is None:
                raise
            return str(experiment.experiment_id)

    def logScalar(self, name: str, value: float, x: int, tag: str = "step") -> None:
        """Log one metric. See `AbstractExperimentLogger.logScalar`."""
        self.logScalars({name: value}, x, tag=tag)

    def logScalars(self, metrics: dict[str, float], x: int, tag: str = "step") -> None:
        """Log every entry of `metrics` as MLflow metrics at step `x`, in one batch request.

        Overrides the one-call-per-metric default of the base class: with a tracking server each
        request is a network round trip, and `Trainer` logs several metrics at every step. MLflow
        accepts at most 1000 metrics per request, so a larger dict is split.

        Args:
            metrics: Metric name -> value. A `NaN` or infinite value is stored as MLflow stores
                it (infinity is clipped to the largest float).
            x: The metric's step (a global step or an epoch index, see `tag`).
            tag: As in `AbstractExperimentLogger.logScalar`; it does not change the name.
        """
        timestamp = int(time.time() * 1000)
        entries = [
            self._metric_class(name, float(value), timestamp, x) for name, value in metrics.items()
        ]
        for start in range(0, len(entries), _MAX_METRICS_PER_BATCH):
            self._client.log_batch(
                self._run_id, metrics=entries[start : start + _MAX_METRICS_PER_BATCH]
            )

    def logImage(self, name: str, image: torch.Tensor, x: int, tag: str = "step") -> None:
        """Log an image with `MlflowClient.log_image`, shown in the MLflow UI image grid.

        Args:
            name: Image key. Characters MLflow does not accept in a key become `_`; `/` groups
                images (`"recon/signal"`).
            image: `(C, H, W)` with `C` in 1, 3 or 4 (or `(H, W)`), `uint8` or floating point
                in `[0, 1]`; see `_imageToArray` for the conversion (floats are clipped).
            x: The step the image is logged at.
            tag: As in `AbstractExperimentLogger.logScalar`; it does not change the key, as in
                `TensorBoardLogger`.

        Raises:
            ValueError: If `image` has an unsupported shape.
            TypeError: If `image` has an unsupported dtype.
        """
        self._client.log_image(self._run_id, _imageToArray(image), key=_safeName(name), step=x)

    def logFigure(self, name: str, figure: Any, x: int, tag: str = "step") -> None:
        """Save a matplotlib figure as the artifact `figures/<name>/<tag>_<x>.png`.

        Args:
            name: Figure name; `/` makes folders (`"latent/pca"`), characters that are unsafe in
                a path become `_`, and `..` segments are dropped.
            figure: A `matplotlib.figure.Figure` (typed `Any` since matplotlib is not a
                dependency of this module). It is not closed: that stays with the caller.
            x: A global step or an epoch index, zero-padded to eight digits so the files sort.
            tag: `"step"` or `"epoch"` (or any caller-chosen granularity). It is part of the file
                name, so that epoch 3 and step 3 of one figure do not overwrite each other.
        """
        artifact_file = f"figures/{_safeName(name)}/{_safeName(tag).replace('/', '_')}_{x:08d}.png"
        self._client.log_figure(self._run_id, figure, artifact_file)

    def onTrainEnd(self, trainer: "Trainer") -> None:
        """Set the run's status from how training ended.

        `Trainer.fit` calls this from a `finally` block, so `sys.exc_info()` holds the
        exception that is ending training, if there is one: `FAILED` for an error, `KILLED` for
        a `KeyboardInterrupt`, `FINISHED` otherwise. (A `fit` call made inside the caller's own
        `except` block, after an earlier failure, would be read as failed: a rare case.)
        See `TrainerCallback.onTrainEnd`.
        """
        error = sys.exc_info()[1]
        if error is None:
            status = "FINISHED"
        elif isinstance(error, KeyboardInterrupt):
            status = "KILLED"
        else:
            status = "FAILED"
        self._client.set_terminated(self._run_id, status=status)

    def close(self) -> None:
        """Mark the run `FINISHED` (and stamp its end time).

        Calling it again is harmless, and logging still works afterwards: MLflow does not lock a
        finished run.
        """
        self._client.set_terminated(self._run_id, status="FINISHED")
