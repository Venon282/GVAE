"""CSV experiment logger: no extra dependency, human-readable, trivial to load with pandas.

Uses a "long" (tidy) format, one row per `(x, tag, metric, value)`,
instead of one wide row per step with one column per metric. A wide
format has a real problem here: the set of metric keys differs between
`"step"` and `"epoch"` (three raw `"loss/..."` keys vs. epoch-averaged,
`"train/"`/`"val/"`-prefixed ones), and can even change between one
`Trainer.fit` call and the next (e.g. adding a validation dataloader
partway through a notebook session). CSV has no clean way to add a
column to a file that has already been started; a fixed 4-column long
format never runs into this. It is also trivial to pivot back into a
wide table for analysis, e.g. with pandas:
`df.pivot(index=["x", "tag"], columns="metric", values="value")`.
"""

import csv
from pathlib import Path

from global_vae.training.loggers.base import AbstractExperimentLogger
from global_vae.training.loggers.registry import registerLogger


@registerLogger("csv")
class CsvLogger(AbstractExperimentLogger):
    """Appends every logged scalar as one row to a single CSV file.

    Does not support `logImage`/`logFigure` (inherits the base class's
    warn-and-skip default): a CSV has no honest way to represent
    visual content.
    """

    def __init__(self, path: str | Path, flush_every_n_rows: int = 1, append: bool = False) -> None:
        """Open `path` for writing and write the header row.

        Args:
            path: Destination CSV file path. Parent directories are
                created if they do not already exist. Opened in write
                mode (`"w"`) by default: an existing file at this path is
                overwritten, matching `TensorBoardLogger`'s own
                start-a-fresh-run behavior for a given `log_dir`.
            flush_every_n_rows: Flush to disk every `N` rows (`1`, the
                default, flushes every row: safe if the process is
                killed mid-run, at some I/O cost for very frequent
                step-level logging). Raise this if step-level logging
                volume makes flushing every row a measurable
                bottleneck.
            append: If `True` and a non-empty file is already at `path`, add rows to it,
                with no second header, instead of replacing it. This is how a resumed run
                (`training.resume_from`, ADR 0032) continues the file of the first part:
                `buildTrainerFromConfig` passes it for you. Rows the interrupted run wrote
                after its last checkpoint stay in the file, so a step can appear twice; the
                later row is the one of the resumed run.

        Raises:
            ValueError: If `flush_every_n_rows` is not positive.
        """
        if flush_every_n_rows <= 0:
            raise ValueError(f"flush_every_n_rows must be positive, got {flush_every_n_rows}.")

        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        continuing = append and resolved_path.is_file() and resolved_path.stat().st_size > 0
        self._file = resolved_path.open("a" if continuing else "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not continuing:
            self._writer.writerow(["x", "tag", "metric", "value"])
        self._flush_every_n_rows = flush_every_n_rows
        self._rows_since_flush = 0

    def logScalar(self, name: str, value: float, x: int, tag: str = "step") -> None:
        """Write one `(x, tag, name, value)` row. See `AbstractExperimentLogger.logScalar`."""
        self._writer.writerow([x, tag, name, value])
        self._rows_since_flush += 1
        if self._rows_since_flush >= self._flush_every_n_rows:
            self._file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._file.close()
