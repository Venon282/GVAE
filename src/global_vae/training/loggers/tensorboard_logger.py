"""TensorBoard experiment logger, wrapping `torch.utils.tensorboard.SummaryWriter`.

The `tensorboard` PyPI package (`torch.utils.tensorboard`'s own
runtime dependency) is a soft dependency of this module: importing
`training.loggers` (and hence this file, for its `@registerLogger`
side effect) never requires it, since the actual `SummaryWriter`
import is deferred to `TensorBoardLogger.__init__`. This matters
because `training.loggers.__init__` imports every concrete logger
implementation to populate the registry; if the `SummaryWriter` import
were at module level here, merely importing the loggers subpackage
(or `getLoggerClass("csv")`, needing nothing to do with TensorBoard at
all) would fail in an environment without `tensorboard` installed.
`pyproject.toml`'s `tensorboard` extra installs it: `pip install
-e ".[tensorboard]"`.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from global_vae.training.loggers.base import AbstractExperimentLogger
from global_vae.training.loggers.registry import registerLogger

if TYPE_CHECKING:
    import torch
    from torch.utils.tensorboard import SummaryWriter


@registerLogger("tensorboard")
class TensorBoardLogger(AbstractExperimentLogger):
    """Logs scalars, images, and figures to a TensorBoard event file.

    `logImage`/`logFigure` are both supported (overriding the base
    class's warn-and-skip default), unlike `CsvLogger`.
    """

    def __init__(self, log_dir: str | Path, flush_every_n_scalars: int = 100) -> None:
        """Open a `SummaryWriter` at `log_dir`.

        Args:
            log_dir: Directory TensorBoard event files are written
                into (created if missing, matching `SummaryWriter`'s
                own behavior).
            flush_every_n_scalars: Explicitly flush the writer every
                `N` scalars logged (`100` by default), independent of
                `SummaryWriter`'s own internal flushing, so a live
                `tensorboard --logdir=...` view updates reasonably
                promptly during a long run without flushing on every
                single call (unlike `CsvLogger`'s per-row default,
                since TensorBoard event files are a less cheap target
                to flush very frequently).

        Raises:
            ImportError: If the `tensorboard` package is not
                installed. Install it with `pip install tensorboard`
                or `pip install -e ".[tensorboard]"`.
            ValueError: If `flush_every_n_scalars` is not positive.
        """
        if flush_every_n_scalars <= 0:
            raise ValueError(
                f"flush_every_n_scalars must be positive, got {flush_every_n_scalars}."
            )

        try:
            from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
        except ImportError as error:
            raise ImportError(
                "TensorBoardLogger requires the 'tensorboard' package, which is not "
                "installed. Install it with `pip install tensorboard` or "
                '`pip install -e ".[tensorboard]"`.'
            ) from error

        self._writer: SummaryWriter = _SummaryWriter(log_dir=str(log_dir))
        self._flush_every_n_scalars = flush_every_n_scalars
        self._scalars_since_flush = 0

    def logScalar(self, name: str, value: float, x: int, tag: str = "step") -> None:
        """Write one scalar via `SummaryWriter.add_scalar`. `tag` is not used as a namespace
        prefix (TensorBoard already groups series by `name`, e.g. `"train/loss/total"` reads
        as the `"train"` group); it only affects the flush counter's bookkeeping here, not
        which series a point lands on.

        See `AbstractExperimentLogger.logScalar` for the full argument contract.
        """
        self._writer.add_scalar(name, value, global_step=x)
        self._scalars_since_flush += 1
        if self._scalars_since_flush >= self._flush_every_n_scalars:
            self._writer.flush()
            self._scalars_since_flush = 0

    def logImage(self, name: str, image: "torch.Tensor", x: int, tag: str = "step") -> None:
        """Write an image via `SummaryWriter.add_image` (`(C, H, W)` tensor convention)."""
        self._writer.add_image(name, image, global_step=x)

    def logFigure(self, name: str, figure: Any, x: int, tag: str = "step") -> None:
        """Write a matplotlib figure via `SummaryWriter.add_figure`.

        Args:
            figure: A `matplotlib.figure.Figure` (or a list of them,
                `SummaryWriter.add_figure`'s own accepted input); typed
                `Any` here since matplotlib is not a dependency of this
                module (see `AbstractExperimentLogger.logFigure`).
        """
        self._writer.add_figure(name, figure, global_step=x)

    def close(self) -> None:
        """Flush and close the underlying `SummaryWriter`."""
        self._writer.close()
