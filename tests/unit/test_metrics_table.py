"""Unit tests for `evaluation.metrics_table` (roadmap P3-6)."""

import csv
import io
import json
import math
from pathlib import Path

import pytest

from global_vae.evaluation.evaluate import EvaluationResults
from global_vae.evaluation.metrics_table import (
    ALL_INPUTS,
    MetricsTable,
    MetricsTableRow,
    buildMetricsTable,
)


def _evaluation() -> EvaluationResults:
    return EvaluationResults(
        num_samples=10,
        total_reconstruction_loss=1.0,
        total_regularization_loss=0.5,
        reconstruction_metrics={
            "signal": {"mse": 0.25, "rmse": 0.5, "r2": 0.9},
            "image": {"mse": 0.04, "rmse": 0.2, "r2": 0.6},
        },
    )


_CROSS_MODAL = {
    frozenset({"signal"}): {"signal": {"mse": 0.3, "rmse": 0.55}, "image": {"mse": 0.09}},
    frozenset({"image", "signal"}): {"signal": {"mse": 0.25}, "image": {"mse": 0.04}},
    frozenset({"image"}): {"signal": {"mse": 0.4}, "image": {"mse": 0.05, "rmse": 0.22}},
}


class TestBuild:
    def test_one_row_per_modality_of_an_evaluation(self) -> None:
        table = buildMetricsTable(_evaluation())
        assert [(row.inputs, row.modality) for row in table.rows] == [
            (ALL_INPUTS, "signal"),
            (ALL_INPUTS, "image"),
        ]
        assert table.columns == ["mse", "rmse", "r2"]
        assert table.rows[0].metrics == {"mse": 0.25, "rmse": 0.5, "r2": 0.9}

    def test_cross_modal_rows_carry_the_input_subset_smaller_subsets_first(self) -> None:
        table = buildMetricsTable(cross_modal=_CROSS_MODAL)
        assert [(row.inputs, row.modality) for row in table.rows] == [
            ("image", "signal"),
            ("image", "image"),
            ("signal", "signal"),
            ("signal", "image"),
            ("image+signal", "signal"),
            ("image+signal", "image"),
        ]

    def test_an_evaluation_comes_before_the_cross_modal_rows(self) -> None:
        table = buildMetricsTable(_evaluation(), _CROSS_MODAL)
        labels = [row.inputs for row in table.rows]
        assert labels[:2] == [ALL_INPUTS, ALL_INPUTS]
        assert len(table.rows) == 2 + 6

    def test_the_full_subset_can_be_left_out_since_the_evaluation_is_it(self) -> None:
        table = buildMetricsTable(
            _evaluation(), _CROSS_MODAL, omit_subsets=[frozenset({"signal", "image"})]
        )
        assert "image+signal" not in {row.inputs for row in table.rows}
        assert len(table.rows) == 2 + 4

    def test_columns_are_the_union_of_the_metrics_in_order_of_appearance(self) -> None:
        table = buildMetricsTable(_evaluation(), _CROSS_MODAL)
        assert table.columns == ["mse", "rmse", "r2"]
        reordered = buildMetricsTable(
            cross_modal={frozenset({"a"}): {"x": {"mae": 1.0}, "y": {"mse": 2.0, "mae": 3.0}}}
        )
        assert reordered.columns == ["mae", "mse"]

    def test_the_rows_are_copies_and_leave_the_inputs_alone(self) -> None:
        evaluation = _evaluation()
        table = buildMetricsTable(evaluation)
        table.rows[0].metrics["mse"] = 99.0
        assert evaluation.reconstruction_metrics["signal"]["mse"] == 0.25

    def test_nothing_to_tabulate_raises(self) -> None:
        with pytest.raises(ValueError, match="no metric to put in a table"):
            buildMetricsTable()
        with pytest.raises(ValueError, match="no metric to put in a table"):
            buildMetricsTable(
                EvaluationResults(
                    num_samples=1, total_reconstruction_loss=0.0, total_regularization_loss=0.0
                )
            )
        with pytest.raises(ValueError, match="no metric to put in a table"):
            buildMetricsTable(cross_modal={})


class TestMarkdown:
    def test_writes_a_header_a_separator_and_a_line_per_row(self) -> None:
        text = buildMetricsTable(_evaluation()).toMarkdown()
        assert text == (
            "| inputs | modality | mse | rmse | r2 |\n"
            "|---|---|---:|---:|---:|\n"
            "| all | signal | 0.25 | 0.5 | 0.9 |\n"
            "| all | image | 0.04 | 0.2 | 0.6 |\n"
        )

    def test_a_missing_metric_is_an_empty_cell_and_an_undefined_one_is_nan(self) -> None:
        table = MetricsTable(
            columns=["mse", "r2"],
            rows=[
                MetricsTableRow("all", "a", {"mse": 1.0}),
                MetricsTableRow("all", "b", {"mse": 2.0, "r2": math.nan}),
            ],
        )
        lines = table.toMarkdown().splitlines()
        assert lines[2] == "| all | a | 1 |  |"
        assert lines[3] == "| all | b | 2 | nan |"

    def test_precision_sets_the_significant_digits(self) -> None:
        table = MetricsTable(columns=["mse"], rows=[MetricsTableRow("all", "a", {"mse": 1 / 3})])
        assert "0.333333" in table.toMarkdown()
        assert "0.33 " in table.toMarkdown(precision=2)

    def test_a_pipe_in_a_name_does_not_break_the_row(self) -> None:
        table = MetricsTable(columns=["mse"], rows=[MetricsTableRow("a|b", "c", {"mse": 1.0})])
        assert "| a\\|b | c | 1 |" in table.toMarkdown()

    def test_every_line_has_the_same_number_of_cells(self) -> None:
        text = buildMetricsTable(_evaluation(), _CROSS_MODAL).toMarkdown()
        widths = {line.count("|") for line in text.splitlines()}
        assert widths == {len(["inputs", "modality", "mse", "rmse", "r2"]) + 1}


class TestCsv:
    def test_round_trips_through_a_csv_reader(self) -> None:
        table = buildMetricsTable(_evaluation(), _CROSS_MODAL)
        rows = list(csv.reader(io.StringIO(table.toCsv())))
        assert rows[0] == ["inputs", "modality", "mse", "rmse", "r2"]
        assert rows[1] == ["all", "signal", "0.25", "0.5", "0.9"]
        assert len(rows) == 1 + len(table.rows)

    def test_a_missing_metric_is_an_empty_field(self) -> None:
        rows = list(csv.reader(io.StringIO(buildMetricsTable(cross_modal=_CROSS_MODAL).toCsv())))
        by_label = {(row[0], row[1]): row for row in rows[1:]}
        assert by_label[("image", "signal")][3] == ""  # no rmse for this cell

    def test_a_comma_in_a_name_is_quoted(self) -> None:
        table = MetricsTable(columns=["mse"], rows=[MetricsTableRow("a,b", "c", {"mse": 1.0})])
        assert next(csv.reader(io.StringIO(table.toCsv().splitlines()[1])))[0] == "a,b"

    def test_precision_is_higher_than_the_markdown_default(self) -> None:
        table = MetricsTable(columns=["mse"], rows=[MetricsTableRow("all", "a", {"mse": 1 / 3})])
        assert table.toCsv().splitlines()[1] == "all,a,0.3333333333"


class TestSave:
    def test_the_suffix_picks_the_format(self, tmp_path: Path) -> None:
        table = buildMetricsTable(_evaluation())
        table.save(tmp_path / "table.md")
        table.save(tmp_path / "table.csv")
        table.save(tmp_path / "table.json")
        assert (tmp_path / "table.md").read_text() == table.toMarkdown()
        assert (tmp_path / "table.csv").read_text() == table.toCsv()
        assert json.loads((tmp_path / "table.json").read_text()) == table.toDict()

    def test_the_suffix_is_case_insensitive_and_missing_folders_are_made(
        self, tmp_path: Path
    ) -> None:
        buildMetricsTable(_evaluation()).save(tmp_path / "deep" / "er" / "TABLE.MD")
        assert (tmp_path / "deep" / "er" / "TABLE.MD").read_text().startswith("| inputs")

    def test_an_unknown_suffix_raises_and_writes_nothing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"\.md, \.csv or \.json"):
            buildMetricsTable(_evaluation()).save(tmp_path / "table.txt")
        assert list(tmp_path.iterdir()) == []

    def test_the_json_form_has_the_rows(self) -> None:
        document = buildMetricsTable(_evaluation()).toDict()
        assert document["columns"] == ["mse", "rmse", "r2"]
        assert document["rows"][1] == {
            "inputs": "all",
            "modality": "image",
            "metrics": {"mse": 0.04, "rmse": 0.2, "r2": 0.6},
        }
