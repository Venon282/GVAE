"""Abstract interface for experiment-tracking backends (spec §10: "Weights & Biases or
MLflow, logging losses, latent-space visualizations, and reconstructions per run").

An `AbstractExperimentLogger` is a `TrainerCallback` (`training/callbacks.py`) specialized
for exactly this job: it translates `Trainer`'s generic per-step/per-epoch events into the
handful of calls a concrete tracking backend actually needs to implement (`logScalar`, and
optionally `logImage`/`logFigure`/`close`), instead of every backend reimplementing the
callback wiring itself. This is the seam `docs/adr/0005-training-loop.md` and
`training/checkpoint.py`'s `CheckpointCallback` already anticipated ("a metrics logger...
`TrainerCallback` is the seam this plugs into").

Concrete backends self-register via `@registerLogger(name)` (see `registry.py`), the same
pattern as every other pluggable strategy in this codebase (encoders, fusion, regularizers,
beta schedules, ...): a caller picks one by name from config rather than importing a
specific class directly. This is a different shape of extension point from
`TrainerCallback` itself, whose composability is a plain list, not a registry
(`docs/adr/0005-training-loop.md`): combining several loggers at once needs no dedicated
"composite logger" class, since `Trainer.callbacks` is already a list and every logger is
already a `TrainerCallback` (`callbacks=[CsvLogger(...), TensorBoardLogger(...)]` just
works). The registry exists for the separate, complementary need of picking *which* logger
class a config name refers to.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from global_vae.training.callbacks import TrainerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

logger = logging.getLogger(__name__)


class AbstractExperimentLogger(TrainerCallback, ABC):
    """Base class for every experiment-tracking backend.

    `onStepEnd` and `onEpochEnd` are already wired to call `logScalars`
    (tagged `"step"` and `"epoch"` respectively: the two carry
    differently-shaped metric dicts, since `onStepEnd`'s are the three
    raw `"loss/..."` keys from `StepLosses.asMetrics()` while
    `onEpochEnd`'s are epoch-averaged and prefixed `"train/"`/`"val/"`).
    `onTrainEnd` closes the backend. Concrete subclasses do not need to
    override any of these three: implement `logScalar` (the one
    mandatory method) and, optionally, `logImage`/`logFigure`/`close`
    for backends that support them.
    """

    def onStepEnd(self, trainer: "Trainer", step: int, metrics: dict[str, float]) -> None:
        """Log this step's metrics, tagged `"step"`. See `TrainerCallback.onStepEnd`."""
        self.logScalars(metrics, step, tag="step")

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Log this epoch's metrics, tagged `"epoch"`. See `TrainerCallback.onEpochEnd`."""
        self.logScalars(metrics, epoch, tag="epoch")

    def onTrainEnd(self, trainer: "Trainer") -> None:
        """Close the backend. See `TrainerCallback.onTrainEnd`."""
        self.close()

    @abstractmethod
    def logScalar(self, name: str, value: float, x: int, tag: str = "step") -> None:
        """Log a single named scalar value.

        Args:
            name: Metric name (e.g. `"loss/total"`,
                `"train/loss/reconstruction"`).
            value: The scalar value.
            x: The x-axis position: a global step when `tag="step"`, an
                epoch index when `tag="epoch"`.
            tag: `"step"` or `"epoch"` (or any other caller-chosen
                granularity, for direct use outside `Trainer`'s own
                calls): distinguishes *which* x-axis `x` is measured
                against, since a backend may need to route these
                differently (e.g. `CsvLogger` records it as a plain
                column; a backend with a rigid per-granularity schema
                could route to different files/tables instead).

        Raises:
            NotImplementedError: If called on the abstract base class.
        """
        raise NotImplementedError

    def logScalars(self, metrics: dict[str, float], x: int, tag: str = "step") -> None:
        """Log every entry of `metrics` via `logScalar`.

        Args:
            metrics: Metric name -> value.
            x: As in `logScalar`.
            tag: As in `logScalar`.
        """
        for name, value in metrics.items():
            self.logScalar(name, value, x, tag=tag)

    def logImage(self, name: str, image: torch.Tensor, x: int, tag: str = "step") -> None:
        """Log an image tensor (spec §10: "reconstructions").

        Default: not supported, logs a warning and does nothing.
        Override in a backend that can store visual content (e.g.
        `TensorBoardLogger`); a purely tabular backend (e.g.
        `CsvLogger`) has no honest way to represent an image and is
        not required to.

        Args:
            name: Image name/tag.
            image: Image tensor. Shape convention matches whatever the
                overriding backend expects (`TensorBoardLogger` follows
                `SummaryWriter.add_image`'s own `(C, H, W)` convention).
            x: As in `logScalar`.
            tag: As in `logScalar`.
        """
        logger.warning(
            "%s does not support image logging; skipping logImage('%s', x=%d).",
            type(self).__name__,
            name,
            x,
        )

    def logFigure(self, name: str, figure: Any, x: int, tag: str = "step") -> None:
        """Log a plotting-library figure (spec §10: "latent-space visualizations").

        Default: not supported, logs a warning and does nothing (see
        `logImage`). `figure` is typed `Any` rather than e.g.
        `matplotlib.figure.Figure` so this module never requires a
        plotting library as a hard dependency; a backend that
        overrides this documents which figure type it expects.

        Args:
            name: Figure name/tag.
            figure: The figure object.
            x: As in `logScalar`.
            tag: As in `logScalar`.
        """
        logger.warning(
            "%s does not support figure logging; skipping logFigure('%s', x=%d).",
            type(self).__name__,
            name,
            x,
        )

    def close(self) -> None:
        """Release any resources this backend holds (e.g. an open file handle).

        Default: a no-op, correct for a backend with nothing to
        release. Overridden by e.g. `CsvLogger` (closes its file) and
        `TensorBoardLogger` (closes its `SummaryWriter`).
        """
