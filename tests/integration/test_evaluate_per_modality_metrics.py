"""`evaluate()` on series with gaps, point clouds and texts (roadmap P5-3, ADR 0048).

Each shipped experiment names the loss that fits its data. `evaluate()` reads that loss and reports
the metrics that fit the same data: the five default ones in their masked form for a series with
gaps, the cloud scores for a cloud, the token accuracy for a text. These tests build the models of
the experiments and run `evaluate()` as `scripts/evaluate.py` does.
"""

import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as functional

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import resolveReconstructionLoss
from global_vae.evaluation.evaluate import evaluate
from global_vae.evaluation.metrics import getMetric
from global_vae.losses.reconstruction.chamfer import ChamferLoss
from global_vae.losses.reconstruction.masked_mse import MaskedMseLoss
from global_vae.losses.reconstruction.token_cross_entropy import TokenCrossEntropyLoss
from global_vae.models.global_vae import GlobalVae
from global_vae.utils.autograd import backward
from global_vae.utils.point_cloud_metrics import computeCloudScores
from tests.integration._time_series_corpus import makeSeries

_DATA_OVERRIDES = ["data.loader_factory=unused.module:unused", "data.train_path=unused"]
FIVE = ["mse", "rmse", "mae", "r2", "pearson_r"]
CLOUD_SCORES = ["chamfer", "precision", "recall", "f_score"]
VOCAB = 32
LENGTH = 8


def _build(experiment: str, *overrides: str) -> tuple[GlobalVae, Any]:
    """The model of a shipped experiment and the loss its YAML names."""
    config = loadExperimentConfig(
        f"experiment/{experiment}", overrides=[*_DATA_OVERRIDES, *overrides]
    )
    model = buildModelFromConfig(config.model)
    return model, resolveReconstructionLoss(config.training.reconstruction_loss, model)


def _gappy(rows: int = 6) -> dict[str, torch.Tensor]:
    observed, _, _ = makeSeries(rows, seed=7)
    assert torch.isnan(observed).any()
    return {"time_series": observed}


def _clouds(rows: int = 6, points: int = 64, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    clouds = torch.randn(rows, points, 3, generator=generator)
    inside: torch.Tensor = clouds / clouds.norm(dim=-1, keepdim=True).clamp_min(1.0)
    return inside


def _everyValue(values: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in values.values())


class TestASeriesWithGaps:
    def test_the_default_report_has_no_nan(self) -> None:
        model, loss = _build("time_series_vae")
        assert isinstance(loss, MaskedMseLoss)
        results = evaluate(
            model, [_gappy()], reconstruction_loss_fn=loss, reconstruction_reduction="sum"
        )
        metrics = results.reconstruction_metrics["time_series"]
        assert list(metrics) == FIVE
        assert _everyValue(metrics)
        assert math.isfinite(results.total_reconstruction_loss)

    def test_the_numbers_are_the_masked_metrics_of_the_pooled_data(self) -> None:
        model, loss = _build("time_series_vae")
        batch = _gappy()
        results = evaluate(model, [batch], reconstruction_loss_fn=loss)
        model.eval()
        with torch.no_grad():
            reconstruction = model(batch, use_mean=True)["reconstructions"]["time_series"]
        for name in FIVE:
            expected = getMetric(f"masked_{name}")(reconstruction, batch["time_series"])
            assert results.reconstruction_metrics["time_series"][name] == pytest.approx(
                expected, rel=1e-5
            )

    def test_the_plain_mse_still_reports_nan_on_the_same_data(self) -> None:
        """Why the experiment no longer names `mse`: a gap makes every plain metric `nan`."""
        model, _ = _build("time_series_vae")
        results = evaluate(model, [_gappy()], reconstruction_loss_fn=functional.mse_loss)
        assert all(
            math.isnan(value) for value in results.reconstruction_metrics["time_series"].values()
        )

    def test_on_complete_series_the_masked_report_is_the_plain_one(self) -> None:
        model, loss = _build("time_series_vae")
        batch = {"time_series": torch.randn(5, 3, 96)}
        masked = evaluate(model, [batch], reconstruction_loss_fn=loss)
        plain = evaluate(model, [batch], reconstruction_loss_fn=functional.mse_loss)
        for name in FIVE:
            assert masked.reconstruction_metrics["time_series"][name] == pytest.approx(
                plain.reconstruction_metrics["time_series"][name], rel=1e-6
            )

    def test_streaming_gives_the_same_numbers_without_pooling(self) -> None:
        model, loss = _build("time_series_vae")
        batches = [_gappy(3), _gappy(4)]
        pooled = evaluate(model, batches, reconstruction_loss_fn=loss)
        streamed = evaluate(model, batches, reconstruction_loss_fn=loss, streaming=True)
        for name in FIVE:
            assert streamed.reconstruction_metrics["time_series"][name] == pytest.approx(
                pooled.reconstruction_metrics["time_series"][name], rel=1e-4, abs=1e-6
            )

    def test_a_list_of_names_picks_the_metrics(self) -> None:
        model, loss = _build("time_series_vae")
        results = evaluate(
            model,
            [_gappy()],
            reconstruction_loss_fn=loss,
            reconstruction_metrics=["masked_mae", "masked_rmse"],
        )
        assert list(results.reconstruction_metrics["time_series"]) == ["masked_mae", "masked_rmse"]
        assert _everyValue(results.reconstruction_metrics["time_series"])

    def test_a_mapping_names_the_modality_it_changes(self) -> None:
        model, loss = _build("time_series_vae")
        results = evaluate(
            model,
            [_gappy()],
            reconstruction_loss_fn=loss,
            reconstruction_metrics={"time_series": ["masked_mse"]},
        )
        assert list(results.reconstruction_metrics["time_series"]) == ["masked_mse"]

    def test_a_dict_of_functions_still_works_as_it_always_did(self) -> None:
        model, loss = _build("time_series_vae")

        def spread(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
            return float(reconstruction.std())

        results = evaluate(
            model,
            [_gappy()],
            reconstruction_loss_fn=loss,
            reconstruction_metrics={"spread": spread},
        )
        assert list(results.reconstruction_metrics["time_series"]) == ["spread"]

    def test_streaming_refuses_a_function_and_a_score_it_cannot_accumulate(self) -> None:
        model, loss = _build("time_series_vae")
        with pytest.raises(ValueError, match="cannot take custom"):
            evaluate(
                model,
                [_gappy()],
                reconstruction_loss_fn=loss,
                reconstruction_metrics={"mine": lambda reconstruction, target: 0.0},
                streaming=True,
            )
        with pytest.raises(ValueError, match="No streaming form"):
            evaluate(
                model,
                [_gappy()],
                reconstruction_loss_fn=loss,
                reconstruction_metrics=["token_accuracy"],
                streaming=True,
            )

    def test_an_unknown_name_is_refused_before_any_batch_is_read(self) -> None:
        model, loss = _build("time_series_vae")

        def batches() -> object:
            raise AssertionError("the data was read")
            yield  # pragma: no cover

        with pytest.raises(KeyError, match="Unknown metric 'nope'"):
            evaluate(
                model,
                batches(),  # type: ignore[arg-type]
                reconstruction_loss_fn=loss,
                reconstruction_metrics=["nope"],
            )

    def test_a_mapping_that_names_no_reconstruction_of_the_model_is_refused(self) -> None:
        model, loss = _build("time_series_vae")
        with pytest.raises(ValueError, match="does not produce"):
            evaluate(
                model,
                [_gappy()],
                reconstruction_loss_fn=loss,
                reconstruction_metrics={"series": ["mse"]},
            )


class TestAPointCloud:
    def test_the_default_report_is_the_cloud_scores(self) -> None:
        model, loss = _build("point_cloud_vae")
        assert isinstance(loss, ChamferLoss)
        results = evaluate(
            model,
            [{"point_cloud": _clouds()}],
            reconstruction_loss_fn=loss,
            reconstruction_reduction="sum",
        )
        metrics = results.reconstruction_metrics["point_cloud"]
        assert list(metrics) == CLOUD_SCORES
        assert _everyValue(metrics)
        assert 0.0 <= metrics["f_score"] <= 1.0

    def test_the_numbers_are_the_scores_of_the_pooled_clouds(self) -> None:
        model, loss = _build("point_cloud_vae")
        clouds = _clouds()
        results = evaluate(model, [{"point_cloud": clouds}], reconstruction_loss_fn=loss)
        model.eval()
        with torch.no_grad():
            reconstruction = model({"point_cloud": clouds}, use_mean=True)["reconstructions"][
                "point_cloud"
            ]
        expected = computeCloudScores(reconstruction, clouds)
        for name in CLOUD_SCORES:
            assert results.reconstruction_metrics["point_cloud"][name] == pytest.approx(
                expected[name], rel=1e-5
            )

    def test_a_decoder_of_another_size_is_scored_too(self) -> None:
        model, loss = _build(
            "point_cloud_vae", "model.modalities.point_cloud.decoder.kwargs.num_points=48"
        )
        results = evaluate(
            model, [{"point_cloud": _clouds(points=64)}], reconstruction_loss_fn=loss
        )
        assert _everyValue(results.reconstruction_metrics["point_cloud"])

    def test_padded_clouds_are_scored_on_their_real_points(self) -> None:
        model, loss = _build("point_cloud_vae")
        clouds = _clouds(points=40)
        padded = torch.cat([clouds, torch.full((6, 24, 3), float("nan"))], dim=1)
        results = evaluate(model, [{"point_cloud": padded}], reconstruction_loss_fn=loss)
        assert _everyValue(results.reconstruction_metrics["point_cloud"])
        assert math.isfinite(results.total_reconstruction_loss)

    def test_the_scores_say_something_a_trained_model_beats_an_untrained_one(self) -> None:
        """Meaningful, not just finite: fitting a batch lowers the chamfer and raises the recall."""
        torch.manual_seed(0)
        model, loss = _build("point_cloud_vae")
        batch = {"point_cloud": _clouds(8, seed=1)}
        before = evaluate(model, [batch], reconstruction_loss_fn=loss).reconstruction_metrics[
            "point_cloud"
        ]
        optimizer = torch.optim.Adam([*model.parameters(), *loss.parameters()], lr=3e-3)
        model.train()
        for _ in range(150):
            optimizer.zero_grad()
            reconstruction = model(batch)["reconstructions"]["point_cloud"]
            backward(loss(reconstruction, batch["point_cloud"]))
            optimizer.step()
        after = evaluate(model, [batch], reconstruction_loss_fn=loss).reconstruction_metrics[
            "point_cloud"
        ]
        assert after["chamfer"] < 0.5 * before["chamfer"]
        assert after["recall"] > before["recall"]

    def test_a_metric_that_is_not_for_clouds_is_still_a_choice_of_the_caller(self) -> None:
        model, loss = _build("point_cloud_vae")
        results = evaluate(
            model,
            [{"point_cloud": _clouds()}],
            reconstruction_loss_fn=loss,
            reconstruction_metrics=["chamfer"],
        )
        assert list(results.reconstruction_metrics["point_cloud"]) == ["chamfer"]

    def test_the_loss_of_another_size_is_the_only_one_that_can_be_evaluated(
        self, tmp_path: Path
    ) -> None:
        """`mse` compares point by point, so two sizes are refused where `chamfer` scores them."""
        model, _ = _build(
            "point_cloud_vae", "model.modalities.point_cloud.decoder.kwargs.num_points=48"
        )
        with pytest.raises(ValueError, match="shape"):
            evaluate(
                model,
                [{"point_cloud": _clouds(points=64)}],
                reconstruction_loss_fn=functional.mse_loss,
            )


class TestAText:
    def _ids(self, rows: int = 6) -> torch.Tensor:
        return torch.randint(1, VOCAB, (rows, LENGTH))

    def test_the_default_report_is_the_token_accuracy(self) -> None:
        model, loss = _build("text_vae")
        assert isinstance(loss, TokenCrossEntropyLoss)
        batch = {"text": functional.one_hot(self._ids(), VOCAB).float()}
        results = evaluate(
            model, [batch], reconstruction_loss_fn=loss, reconstruction_reduction="sum"
        )
        metrics = results.reconstruction_metrics["text"]
        assert list(metrics) == ["token_accuracy"]
        assert 0.0 <= metrics["token_accuracy"] <= 1.0
        assert math.isfinite(results.total_reconstruction_loss)

    def test_ids_and_one_hot_rows_give_the_same_report(self) -> None:
        model, loss = _build("text_vae")
        ids = self._ids()
        by_ids = evaluate(model, [{"text": ids}], reconstruction_loss_fn=loss)
        by_rows = evaluate(
            model, [{"text": functional.one_hot(ids, VOCAB).float()}], reconstruction_loss_fn=loss
        )
        assert by_ids.reconstruction_metrics == by_rows.reconstruction_metrics
        assert by_ids.total_reconstruction_loss == pytest.approx(by_rows.total_reconstruction_loss)

    def test_a_model_that_has_fitted_a_batch_has_a_higher_accuracy(self) -> None:
        torch.manual_seed(0)
        model, loss = _build("text_vae")
        batch = {"text": functional.one_hot(self._ids(8), VOCAB).float()}
        before = evaluate(model, [batch], reconstruction_loss_fn=loss).reconstruction_metrics
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        model.train()
        for _ in range(120):
            optimizer.zero_grad()
            reconstruction = model(batch)["reconstructions"]["text"]
            backward(loss(reconstruction, batch["text"]))
            optimizer.step()
        after = evaluate(model, [batch], reconstruction_loss_fn=loss).reconstruction_metrics
        assert after["text"]["token_accuracy"] > before["text"]["token_accuracy"] + 0.3

    def test_streaming_refuses_the_accuracy_with_the_reason(self) -> None:
        model, loss = _build("text_vae")
        batch = {"text": self._ids()}
        with pytest.raises(ValueError, match="No streaming form"):
            evaluate(model, [batch], reconstruction_loss_fn=loss, streaming=True)
