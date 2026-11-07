"""Unit tests for the metric registry and the per-modality metric sets (roadmap P5-3, ADR 0048)."""

import math
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as functional

import global_vae.evaluation  # noqa: F401  (registers the modality metrics)
from global_vae.evaluation.metric_sets import MetricSpec, defaultMetricSpecs, resolveMetricSets
from global_vae.evaluation.metrics import (
    DEFAULT_METRIC_NAMES,
    DEFAULT_RECONSTRUCTION_METRICS,
    computeMse,
    getMetric,
    listRegisteredMetrics,
    maskUnobserved,
    registerMetric,
)
from global_vae.evaluation.streaming_metrics import (
    STREAMING_METRICS,
    STREAMING_RECONSTRUCTION_METRICS,
    buildStreamingMetrics,
)
from global_vae.losses.reconstruction.chamfer import ChamferLoss
from global_vae.losses.reconstruction.masked_mse import MaskedMseLoss
from global_vae.losses.reconstruction.mse import MseLoss
from global_vae.losses.reconstruction.token_cross_entropy import TokenCrossEntropyLoss
from global_vae.utils.graph_metrics import computeEdgeScores
from global_vae.utils.point_cloud_metrics import computeCloudScores
from global_vae.utils.text_blocks import computeTokenAccuracy

NAN = float("nan")
FIVE = list(DEFAULT_METRIC_NAMES)
MASKED_FIVE = [f"masked_{name}" for name in FIVE]
CLOUD_SCORES = ["chamfer", "precision", "recall", "f_score"]


def _pair(*shape: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator), torch.randn(*shape, generator=generator)


def _gaps(target: torch.Tensor, every: int = 3) -> torch.Tensor:
    """A copy of `target` with every `every`-th value turned into `NaN`."""
    gapped = target.clone()
    gapped.view(-1)[::every] = NAN
    return gapped


def _defaults() -> dict[str, Callable[[torch.Tensor, torch.Tensor], float]]:
    return dict(DEFAULT_RECONSTRUCTION_METRICS)


class TestTheRegistry:
    def test_the_metrics_of_the_package_are_listed(self) -> None:
        names = listRegisteredMetrics()
        assert names == sorted(names)
        for expected in [*FIVE, *MASKED_FIVE, *CLOUD_SCORES, "edge_auc", "token_accuracy"]:
            assert expected in names

    def test_the_default_set_is_the_five_it_always_was(self) -> None:
        assert DEFAULT_METRIC_NAMES == ("mse", "rmse", "mae", "r2", "pearson_r")
        assert list(DEFAULT_RECONSTRUCTION_METRICS) == FIVE
        for name in FIVE:
            assert DEFAULT_RECONSTRUCTION_METRICS[name] is getMetric(name)

    def test_the_default_dict_keeps_the_function_the_module_exports(self) -> None:
        assert DEFAULT_RECONSTRUCTION_METRICS["mse"] is computeMse

    def test_an_unknown_name_lists_the_available_ones(self) -> None:
        with pytest.raises(KeyError, match="Unknown metric 'nope'.*Available: .*mse"):
            getMetric("nope")

    def test_a_name_cannot_be_registered_twice(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            registerMetric("mse")(computeMse)

    def test_a_registered_function_is_returned_unchanged(self) -> None:
        def metric(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
            return 0.0

        try:
            assert registerMetric("test_only_zero")(metric) is metric
            assert getMetric("test_only_zero") is metric
            assert "test_only_zero" in listRegisteredMetrics()
        finally:
            from global_vae.evaluation import metrics

            metrics._METRIC_REGISTRY.pop("test_only_zero", None)


class TestMaskedMetrics:
    @pytest.mark.parametrize("name", FIVE)
    def test_on_complete_data_a_masked_metric_is_the_plain_one(self, name: str) -> None:
        reconstruction, target = _pair(4, 6, 3)
        assert getMetric(f"masked_{name}")(reconstruction, target) == pytest.approx(
            getMetric(name)(reconstruction, target), rel=1e-6
        )

    @pytest.mark.parametrize("name", FIVE)
    def test_it_is_the_plain_metric_of_the_observed_values(self, name: str) -> None:
        reconstruction, target = _pair(4, 6, 3)
        gapped = _gaps(target)
        observed = ~torch.isnan(gapped)
        expected = getMetric(name)(reconstruction[observed], gapped[observed])
        assert getMetric(f"masked_{name}")(reconstruction, gapped) == pytest.approx(
            expected, rel=1e-6
        )

    @pytest.mark.parametrize("name", FIVE)
    def test_it_is_finite_where_the_plain_metric_is_nan(self, name: str) -> None:
        reconstruction, target = _pair(4, 6, 3)
        gapped = _gaps(target)
        assert math.isnan(getMetric(name)(reconstruction, gapped))
        assert math.isfinite(getMetric(f"masked_{name}")(reconstruction, gapped))

    def test_a_reconstruction_that_is_nan_where_the_target_is_observed_stays_nan(self) -> None:
        """Only the target says what is observed: a broken reconstruction must not look fine."""
        reconstruction, target = _pair(2, 5)
        reconstruction[0, 0] = NAN
        assert math.isnan(getMetric("masked_mse")(reconstruction, target))

    def test_a_target_that_is_all_nan_gives_nan(self) -> None:
        reconstruction, _ = _pair(2, 5)
        assert math.isnan(getMetric("masked_mae")(reconstruction, torch.full((2, 5), NAN)))

    def test_two_shapes_are_refused_instead_of_broadcast(self) -> None:
        with pytest.raises(ValueError, match="one shape"):
            getMetric("masked_mse")(torch.zeros(2, 5), torch.zeros(2, 1))

    def test_the_wrapper_works_on_a_function_of_the_caller(self) -> None:
        reconstruction, target = _pair(3, 4)
        gapped = _gaps(target, every=2)

        def maxError(a: torch.Tensor, b: torch.Tensor) -> float:
            return float((a - b).abs().max())

        observed = ~torch.isnan(gapped)
        masked = maskUnobserved(maxError)
        assert masked(reconstruction, gapped) == pytest.approx(
            maxError(reconstruction[observed], gapped[observed])
        )

    def test_the_docstring_says_what_the_wrapper_changed(self) -> None:
        assert "not NaN" in (getMetric("masked_r2").__doc__ or "")


class TestModalityMetrics:
    def test_the_cloud_scores_are_the_ones_of_computeCloudScores(self) -> None:
        generator = torch.Generator().manual_seed(0)
        target = torch.randn(3, 20, 3, generator=generator)
        reconstruction = target[:, :16] + 0.02 * torch.randn(3, 16, 3, generator=generator)
        scores = computeCloudScores(reconstruction, target)
        for name in CLOUD_SCORES:
            assert getMetric(name)(reconstruction, target) == pytest.approx(scores[name])

    def test_a_perfect_cloud_scores_zero_and_one(self) -> None:
        cloud = torch.randn(2, 12, 3)
        assert getMetric("chamfer")(cloud, cloud) == pytest.approx(0.0, abs=1e-6)
        assert getMetric("precision")(cloud, cloud) == 1.0
        assert getMetric("recall")(cloud, cloud) == 1.0
        assert getMetric("f_score")(cloud, cloud) == 1.0

    def test_a_cloud_far_away_is_worse_on_every_score(self) -> None:
        cloud = torch.randn(2, 12, 3)
        far = cloud + 10.0
        assert getMetric("chamfer")(far, cloud) > getMetric("chamfer")(cloud, cloud)
        assert getMetric("f_score")(far, cloud) == 0.0

    def test_a_padded_target_is_scored_on_its_real_points(self) -> None:
        """The padding is rows of `NaN`, and the scores that count are the real points'."""
        cloud = torch.randn(2, 10, 3)
        padded = torch.cat([cloud, torch.full((2, 6, 3), NAN)], dim=1)
        assert getMetric("chamfer")(cloud, padded) == pytest.approx(0.0, abs=1e-6)
        assert getMetric("recall")(cloud, padded) == 1.0

    def test_the_edge_auc_is_the_one_of_computeEdgeScores(self) -> None:
        generator = torch.Generator().manual_seed(1)
        raw = torch.rand(4, 6, 6, generator=generator)
        adjacency = (raw + raw.transpose(1, 2) > 1.0).float()
        probabilities = torch.rand(4, 6, 6, generator=generator)
        assert getMetric("edge_auc")(probabilities, adjacency) == pytest.approx(
            computeEdgeScores(probabilities, adjacency)["auc"]
        )

    def test_a_perfect_edge_predictor_has_an_auc_of_one(self) -> None:
        raw = torch.rand(4, 6, 6)
        adjacency = (raw + raw.transpose(1, 2) > 1.0).float()
        assert getMetric("edge_auc")(adjacency, adjacency) == pytest.approx(1.0)

    def test_the_token_accuracy_reads_ids_and_one_hot_targets(self) -> None:
        ids = torch.randint(0, 7, (3, 5))
        scores = functional.one_hot(ids, 7).float()
        scores[0, 0] = torch.eye(7)[(ids[0, 0] + 1) % 7]
        accuracy = getMetric("token_accuracy")
        assert accuracy(scores, ids) == pytest.approx(14 / 15)
        assert accuracy(scores, functional.one_hot(ids, 7).float()) == pytest.approx(14 / 15)
        assert accuracy is not computeTokenAccuracy or accuracy(scores, ids) == pytest.approx(
            computeTokenAccuracy(scores, ids)
        )


class TestTheDefaultSetOfAModality:
    def test_a_plain_function_or_nothing_gets_the_five(self) -> None:
        for loss in (None, functional.mse_loss, MseLoss()):
            specs = defaultMetricSpecs(loss, _defaults())
            assert list(specs) == FIVE
            assert [spec.name for spec in specs.values()] == FIVE

    def test_a_loss_that_reads_nan_as_unobserved_gets_the_masked_five_under_plain_names(
        self,
    ) -> None:
        specs = defaultMetricSpecs(MaskedMseLoss(), _defaults())
        assert list(specs) == FIVE
        assert [spec.name for spec in specs.values()] == MASKED_FIVE
        assert specs["mse"].function is getMetric("masked_mse")

    def test_a_loss_that_declares_its_metrics_gets_them(self) -> None:
        cloud = defaultMetricSpecs(ChamferLoss(), _defaults())
        assert list(cloud) == CLOUD_SCORES
        text = defaultMetricSpecs(TokenCrossEntropyLoss(), _defaults())
        assert list(text) == ["token_accuracy"]
        assert text["token_accuracy"].function is getMetric("token_accuracy")

    def test_what_a_loss_declares_comes_before_its_nan_reading(self) -> None:
        """`chamfer` reads `NaN` as padding and still reports cloud scores, not masked errors."""
        assert ChamferLoss.reads_nan_as_unobserved
        assert list(defaultMetricSpecs(ChamferLoss(), _defaults())) == CLOUD_SCORES


class TestResolvingWhatEvaluateIsGiven:
    MODALITIES = ("series", "cloud")

    def _resolve(self, setting: object, loss: object = None) -> dict[str, dict[str, MetricSpec]]:
        return resolveMetricSets(
            setting,  # type: ignore[arg-type]
            loss if loss is not None else functional.mse_loss,  # type: ignore[arg-type]
            list(self.MODALITIES),
            _defaults(),
        )

    @pytest.mark.parametrize("setting", [None, [], {}])
    def test_nothing_is_the_default_of_each_modality(self, setting: object) -> None:
        resolved = self._resolve(setting, {"series": MaskedMseLoss(), "cloud": ChamferLoss()})
        assert [spec.name for spec in resolved["series"].values()] == MASKED_FIVE
        assert list(resolved["cloud"]) == CLOUD_SCORES

    def test_a_mapping_of_losses_missing_a_modality_gives_it_the_five(self) -> None:
        resolved = self._resolve(None, {"cloud": ChamferLoss()})
        assert list(resolved["series"]) == FIVE
        assert list(resolved["cloud"]) == CLOUD_SCORES

    def test_a_dict_of_functions_is_shared_and_has_no_name(self) -> None:
        def metric(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
            return 1.0

        resolved = self._resolve({"mine": metric})
        for modality in self.MODALITIES:
            assert list(resolved[modality]) == ["mine"]
            assert resolved[modality]["mine"] == MetricSpec(None, metric)

    def test_a_list_of_names_is_shared_in_its_order(self) -> None:
        resolved = self._resolve(["mae", "token_accuracy"])
        for modality in self.MODALITIES:
            assert list(resolved[modality]) == ["mae", "token_accuracy"]
            assert resolved[modality]["mae"].name == "mae"

    def test_a_mapping_of_modalities_changes_only_the_ones_it_names(self) -> None:
        resolved = self._resolve({"cloud": ["chamfer", "f_score"]})
        assert list(resolved["cloud"]) == ["chamfer", "f_score"]
        assert list(resolved["series"]) == FIVE

    def test_a_modality_can_have_a_dict_of_functions(self) -> None:
        def metric(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
            return 2.0

        resolved = self._resolve({"series": {"two": metric}})
        assert resolved["series"] == {"two": MetricSpec(None, metric)}
        assert list(resolved["cloud"]) == FIVE

    def test_a_name_that_is_not_registered_is_refused_with_the_available_ones(self) -> None:
        with pytest.raises(KeyError, match="reconstruction_metrics.*Unknown metric 'rmsee'"):
            self._resolve(["rmsee"])
        with pytest.raises(KeyError, match=r"reconstruction_metrics\['cloud'\].*Unknown metric"):
            self._resolve({"cloud": ["nope"]})

    def test_a_name_listed_twice_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'mse' is listed twice"):
            self._resolve(["mse", "mse"])

    def test_a_string_is_not_read_one_letter_at_a_time(self) -> None:
        with pytest.raises(ValueError, match="pass a list of metric names"):
            self._resolve("mse")

    def test_a_reconstruction_the_model_does_not_produce_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"names reconstruction\(s\) \['clouds'\]"):
            self._resolve({"clouds": ["chamfer"]})

    def test_a_mapping_that_mixes_functions_and_modalities_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a mix"):
            self._resolve({"series": ["mse"], "mine": computeMse})

    def test_a_modality_set_that_is_neither_names_nor_functions_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"expected a list of metric names"):
            self._resolve({"series": 3})
        with pytest.raises(ValueError, match=r"expected a metric name"):
            self._resolve([3])

    def test_the_sets_of_two_modalities_do_not_share_a_dict(self) -> None:
        resolved = self._resolve(["mse"])
        resolved["series"]["extra"] = MetricSpec(None, computeMse)
        assert list(resolved["cloud"]) == ["mse"]


class TestStreamingMaskedMetrics:
    def test_the_ten_streamable_names_are_the_five_and_their_masked_forms(self) -> None:
        assert list(STREAMING_METRICS) == [*FIVE, *MASKED_FIVE]
        assert set(STREAMING_RECONSTRUCTION_METRICS) == set(FIVE)

    @pytest.mark.parametrize("name", FIVE)
    def test_a_masked_stream_is_the_masked_pooled_metric_whatever_the_batches(
        self, name: str
    ) -> None:
        reconstruction, target = _pair(12, 8, seed=3)
        gapped = _gaps(target)
        meter = STREAMING_METRICS[f"masked_{name}"]()
        for start in range(0, 12, 5):
            meter.update(reconstruction[start : start + 5], gapped[start : start + 5])
        expected = getMetric(f"masked_{name}")(reconstruction.double(), gapped.double())
        assert meter.compute() == pytest.approx(expected, rel=1e-9, abs=1e-12)

    def test_a_batch_with_no_observed_value_adds_nothing(self) -> None:
        reconstruction, target = _pair(6, 4)
        meter = STREAMING_METRICS["masked_mse"]()
        meter.update(reconstruction[:3], target[:3])
        before = meter.compute()
        meter.update(reconstruction[3:], torch.full_like(target[3:], NAN))
        assert meter.compute() == before

    def test_a_plain_stream_still_takes_every_value(self) -> None:
        reconstruction, target = _pair(6, 4)
        meter = STREAMING_METRICS["mse"]()
        meter.update(reconstruction, _gaps(target))
        assert math.isnan(meter.compute())

    def test_the_masked_form_can_be_built_under_the_plain_key(self) -> None:
        meters = buildStreamingMetrics({"mse": "masked_mse", "mae": "masked_mae"})
        assert list(meters) == ["mse", "mae"]
        reconstruction, target = _pair(4, 4)
        for meter in meters.values():
            meter.update(reconstruction, _gaps(target))
        assert math.isfinite(meters["mse"].compute())

    def test_by_default_it_builds_the_five(self) -> None:
        assert list(buildStreamingMetrics()) == FIVE

    def test_a_metric_with_no_streaming_form_is_refused_with_the_streamable_ones(self) -> None:
        with pytest.raises(
            ValueError, match=r"No streaming form for the metric\(s\) \['chamfer'\]"
        ):
            buildStreamingMetrics({"chamfer": "chamfer"})
