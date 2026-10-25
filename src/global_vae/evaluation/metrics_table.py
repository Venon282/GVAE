"""A per-modality metrics table: one row per modality, one column per metric (roadmap P3-6).

`EvaluationResults.reconstruction_metrics` is a nested dict, and the printed summary writes each
modality as one line of `name=value` text. Neither pastes into a report or opens in a
spreadsheet, and neither puts two modalities side by side. `MetricsTable` is that view: the rows
are the modalities, the columns are the metrics, and it is written as Markdown (for the docs and
a pull request) or CSV (for a spreadsheet) or JSON.

A model that can be queried with a subset of its inputs has more than one number per modality,
one for each input subset (spec §5): "image, read from the signal alone" is a different row from
"image, read from everything". The first column, `inputs`, says which: `all` for the ordinary
evaluation pass, and the modalities joined by `+` for a cross-modal one
(`evaluation.cross_modal.computeCrossModalReconstructionMetrics`).
"""

import csv
import io
import json
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from global_vae.evaluation.evaluate import EvaluationResults

ALL_INPUTS = "all"
"""The `inputs` label of the rows that come from an ordinary evaluation pass."""

_LABEL_COLUMNS = ("inputs", "modality")


@dataclass
class MetricsTableRow:
    """One row of a `MetricsTable`.

    Attributes:
        inputs: Which inputs the model was given: `"all"`, or the modalities of a subset joined by
            `+` in alphabetical order, such as `"signal"` or `"image+signal"`.
        modality: The decoder or modality whose reconstruction was measured.
        metrics: Metric name -> value. A metric a row does not have is absent, and its cell is
            empty.
    """

    inputs: str
    modality: str
    metrics: dict[str, float] = field(default_factory=dict)


def _formatValue(value: float | None, precision: int) -> str:
    """Write one cell: empty for a missing metric, `nan` for an undefined one."""
    if value is None:
        return ""
    return f"{value:.{precision}g}"


@dataclass
class MetricsTable:
    """Reconstruction metrics laid out as rows of modalities and columns of metrics.

    Attributes:
        columns: The metric names, in the order they are written.
        rows: The rows, in the order they are written.
    """

    columns: list[str] = field(default_factory=list)
    rows: list[MetricsTableRow] = field(default_factory=list)

    def _cells(self, row: MetricsTableRow, precision: int) -> list[str]:
        """The label cells then one cell per column of `row`."""
        return [row.inputs, row.modality] + [
            _formatValue(row.metrics.get(column), precision) for column in self.columns
        ]

    def toMarkdown(self, precision: int = 6) -> str:
        """Write the table as a GitHub-flavoured Markdown table.

        Args:
            precision: Significant digits of each value.

        Returns:
            The header, the separator and one line per row, ending with a newline.
        """
        header = [*_LABEL_COLUMNS, *self.columns]
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(_LABEL_COLUMNS) + ["---:"] * len(self.columns)) + "|",
        ]
        for row in self.rows:
            cells = [cell.replace("|", "\\|") for cell in self._cells(row, precision)]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    def toCsv(self, precision: int = 10) -> str:
        """Write the table as CSV.

        Args:
            precision: Significant digits of each value.

        Returns:
            The header and one line per row, with Unix line endings.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow([*_LABEL_COLUMNS, *self.columns])
        for row in self.rows:
            writer.writerow(self._cells(row, precision))
        return buffer.getvalue()

    def toDict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable dict (`dataclasses.asdict(self)`)."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the table to `path`, in the format its suffix names.

        Args:
            path: A file ending in `.md` (Markdown), `.csv` or `.json`. The parent directories
                are created if they do not exist.

        Raises:
            ValueError: If the suffix is none of those.
        """
        resolved_path = Path(path)
        suffix = resolved_path.suffix.lower()
        if suffix == ".md":
            text = self.toMarkdown()
        elif suffix == ".csv":
            text = self.toCsv()
        elif suffix == ".json":
            text = json.dumps(self.toDict(), indent=2)
        else:
            raise ValueError(
                f"Cannot tell the format of '{resolved_path.name}': use a .md, .csv or .json file."
            )
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(text, encoding="utf-8")


def _subsetLabel(subset: Collection[str]) -> str:
    """Write an input subset as its modalities, sorted, joined by `+`."""
    return "+".join(sorted(subset))


def buildMetricsTable(
    evaluation: EvaluationResults | None = None,
    cross_modal: Mapping[frozenset[str], Mapping[str, Mapping[str, float]]] | None = None,
    omit_subsets: Collection[frozenset[str]] = (),
) -> MetricsTable:
    """Build the per-modality table of an evaluation, with the cross-modal cells if there are any.

    Args:
        evaluation: The result of `evaluate`. Its rows come first, with `inputs` set to `"all"`.
        cross_modal: As returned by `evaluation.cross_modal.computeCrossModalReconstructionMetrics`
            (input subset -> decoder name -> metric name -> value). Its rows follow, the smaller
            subsets first and the subsets of one size in alphabetical order.
        omit_subsets: Input subsets of `cross_modal` to leave out. Pass the full set of input
            modalities when `evaluation` is given, since the ordinary pass already is that
            subset and the two would write the same numbers twice.

    Returns:
        The `MetricsTable`. Its columns are the metric names of the rows, in order of first
        appearance (`mse`, `rmse`, `mae`, `r2`, `pearson_r` for the defaults).

    Raises:
        ValueError: If neither argument gives a row.
    """
    rows: list[MetricsTableRow] = []
    if evaluation is not None:
        rows.extend(
            MetricsTableRow(ALL_INPUTS, modality, dict(metrics))
            for modality, metrics in evaluation.reconstruction_metrics.items()
        )
    if cross_modal is not None:
        omitted = {frozenset(subset) for subset in omit_subsets}
        for subset in sorted(cross_modal, key=lambda s: (len(s), sorted(s))):
            if subset in omitted:
                continue
            rows.extend(
                MetricsTableRow(_subsetLabel(subset), modality, dict(metrics))
                for modality, metrics in cross_modal[subset].items()
            )
    if not rows:
        raise ValueError("There is no metric to put in a table: give an evaluation or cross_modal.")
    columns = list(dict.fromkeys(name for row in rows for name in row.metrics))
    return MetricsTable(columns=columns, rows=rows)
