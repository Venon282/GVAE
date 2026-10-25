"""Integration tests for the paired-data evaluation of roadmap P3-6.

`evaluation.paired_alignment` (retrieval and alignment between the posterior means of the
modalities of one sample), the streaming mode of `evaluation.evaluate.evaluate`, and
`scripts/evaluate.py --paired --streaming`. The pure functions are in
`tests/unit/test_paired_alignment.py` and `test_streaming_metrics.py`.

The models here use encoders that return their input as the posterior mean (`mu = x`), so a test
controls exactly what each modality says about the latent space. The dummy modules are registered
under a `_paired_eval_test` suffix, as the sibling test files do for theirs.
"""

import importlib.util
import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import global_vae.config  # noqa: F401  (registers structured configs with Hydra's ConfigStore)
import global_vae.evaluation.evaluate as evaluate_module
from global_vae.config.model import ModelConfig, buildModelFromConfig
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import registerDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.evaluation.evaluate import evaluate
from global_vae.evaluation.metrics import DEFAULT_RECONSTRUCTION_METRICS
from global_vae.evaluation.paired_alignment import (
    PairedAlignmentResults,
    collectModalityPosteriorMeans,
    evaluatePairedAlignment,
)
from global_vae.models.global_vae import GlobalVae
from global_vae.training.checkpoint import saveCheckpoint
from tests.integration._script_fixtures import (
    buildModelForScript,
    buildTwoModalityModelForScript,
)

LATENT = 4
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"


class _IdentityEncoder(AbstractEncoder):
    """`mu = x` and `logvar = 0`: the posterior mean of the modality is its input."""

    def __init__(self, latent_dim: int = LATENT, name: str = "a") -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._name = name

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.zeros_like(x)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        return self._name

    @property
    def minimal_input_length(self) -> int:
        return 1


@registerEncoder("identity_a_encoder_paired_eval_test")
class _EncoderA(_IdentityEncoder):
    def __init__(self, latent_dim: int = LATENT) -> None:
        super().__init__(latent_dim, "a")


@registerEncoder("identity_b_encoder_paired_eval_test")
class _EncoderB(_IdentityEncoder):
    def __init__(self, latent_dim: int = LATENT) -> None:
        super().__init__(latent_dim, "b")


@registerEncoder("identity_c_encoder_paired_eval_test")
class _EncoderC(_IdentityEncoder):
    def __init__(self, latent_dim: int = LATENT) -> None:
        super().__init__(latent_dim, "c")


class _LinearDecoder(AbstractDecoder):
    def __init__(self, latent_dim: int = LATENT, name: str = "a") -> None:
        super().__init__()
        self._name = name
        self.project = nn.Linear(latent_dim, LATENT)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        reconstruction: torch.Tensor = self.project(z)
        return reconstruction

    @property
    def modality_name(self) -> str:
        return self._name


@registerDecoder("linear_a_decoder_paired_eval_test")
class _DecoderA(_LinearDecoder):
    def __init__(self, latent_dim: int = LATENT) -> None:
        super().__init__(latent_dim, "a")


@registerDecoder("linear_b_decoder_paired_eval_test")
class _DecoderB(_LinearDecoder):
    def __init__(self, latent_dim: int = LATENT) -> None:
        super().__init__(latent_dim, "b")


@registerDecoder("linear_c_decoder_paired_eval_test")
class _DecoderC(_LinearDecoder):
    def __init__(self, latent_dim: int = LATENT) -> None:
        super().__init__(latent_dim, "c")


def _model(modalities: tuple[str, ...] = ("a", "b")) -> GlobalVae:
    return GlobalVae.createSingleLatent(
        modality_configs={
            name: {
                "encoder": f"identity_{name}_encoder_paired_eval_test",
                "decoder": f"linear_{name}_decoder_paired_eval_test",
            }
            for name in modalities
        },
        latent_dim=LATENT,
        fusion_strategy="poe" if len(modalities) > 1 else None,
    )


def _views(
    samples: int, noise: float, seed: int = 0, modalities: tuple[str, ...] = ("a", "b")
) -> dict[str, torch.Tensor]:
    """One noisy view of the same latent points per modality."""
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(samples, LATENT, generator=generator)
    return {
        name: latent + noise * torch.randn(samples, LATENT, generator=generator)
        for name in modalities
    }


def _batches(
    views: dict[str, torch.Tensor], batch_size: int = 8, **extra: torch.Tensor
) -> list[dict[str, torch.Tensor]]:
    samples = next(iter(views.values())).shape[0]
    return [
        {
            **{name: tensor[start : start + batch_size] for name, tensor in views.items()},
            **{name: tensor[start : start + batch_size] for name, tensor in extra.items()},
        }
        for start in range(0, samples, batch_size)
    ]


class TestCollectModalityPosteriorMeans:
    def test_each_modality_gives_its_own_mean_and_not_the_fused_one(self) -> None:
        views = _views(16, noise=0.5)
        model = _model()
        records = collectModalityPosteriorMeans(model, _batches(views, 8))
        assert len(records) == 2
        for index, record in enumerate(records):
            rows = slice(8 * index, 8 * index + 8)
            # With one modality in the query, PoE leaves the encoder's own mean untouched.
            assert torch.equal(record["a"]["z_fused"], views["a"][rows])
            assert torch.equal(record["b"]["z_fused"], views["b"][rows])
        fused_mu = model({"a": views["a"][:8], "b": views["b"][:8]}, use_mean=True)[
            "latent_params"
        ]["z_fused"][0]
        assert not torch.allclose(fused_mu, records[0]["a"]["z_fused"])

    def test_the_means_do_not_require_grad_and_the_model_is_left_in_eval_mode(self) -> None:
        model = _model().train()
        records = collectModalityPosteriorMeans(model, _batches(_views(8, 0.1), 8))
        assert not model.training
        assert not records[0]["a"]["z_fused"].requires_grad

    def test_a_batch_with_a_single_modality_holds_no_pair_and_is_skipped(self) -> None:
        views = _views(16, noise=0.1, modalities=("a", "b"))
        batches = _batches(views, 8)
        batches.insert(1, {"a": torch.randn(5, LATENT)})
        records = collectModalityPosteriorMeans(_model(), batches)
        assert len(records) == 2
        assert all(set(record) == {"a", "b"} for record in records)

    def test_a_key_that_is_not_an_input_modality_is_ignored(self) -> None:
        views = _views(8, noise=0.1)
        batches = _batches(views, 8, label=torch.arange(8))
        records = collectModalityPosteriorMeans(_model(), batches)
        assert set(records[0]) == {"a", "b"}

    def test_a_batch_with_no_input_modality_raises(self) -> None:
        with pytest.raises(ValueError, match="None of this batch's keys"):
            collectModalityPosteriorMeans(_model(), [{"label": torch.arange(4)}])

    def test_a_modality_missing_from_a_batch_is_missing_from_its_record(self) -> None:
        views = _views(8, noise=0.1, modalities=("a", "b", "c"))
        batch = {"a": views["a"], "c": views["c"]}
        record = collectModalityPosteriorMeans(_model(("a", "b", "c")), [batch])[0]
        assert set(record) == {"a", "c"}

    def test_max_samples_trims_the_last_batch_and_stops_reading(self) -> None:
        batches = _batches(_views(40, noise=0.1), 8)
        read: list[int] = []

        def loader() -> Iterator[dict[str, torch.Tensor]]:
            for index, batch in enumerate(batches):
                read.append(index)
                yield batch

        records = collectModalityPosteriorMeans(_model(), loader(), max_samples=10)
        assert [record["a"]["z_fused"].shape[0] for record in records] == [8, 2]
        assert read == [0, 1]

    def test_a_dataloader_that_is_a_single_use_generator_is_read_once(self) -> None:
        records = collectModalityPosteriorMeans(
            _model(), (batch for batch in _batches(_views(16, 0.1), 8))
        )
        assert len(records) == 2

    def test_rejects_a_max_samples_below_one(self) -> None:
        with pytest.raises(ValueError, match="max_samples must be at least 1"):
            collectModalityPosteriorMeans(_model(), [], max_samples=0)


class TestEvaluatePairedAlignment:
    def test_aligned_views_are_retrieved_far_above_chance(self) -> None:
        results = evaluatePairedAlignment(_model(), _batches(_views(200, noise=0.15)))
        assert isinstance(results, PairedAlignmentResults)
        assert results.num_pairs == {"a|b": 200}
        assert set(results.retrieval) == set(results.alignment) == {"z_fused"}
        assert set(results.retrieval["z_fused"]) == {"a->b", "b->a"}
        assert set(results.alignment["z_fused"]) == {"a|b"}
        for direction in ("a->b", "b->a"):
            metrics = results.retrieval["z_fused"][direction]
            assert metrics["recall@1"] > 0.5
            assert metrics["recall@1"] > 50 * metrics["chance_recall@1"]
            assert metrics["median_rank"] <= 2
        alignment = results.alignment["z_fused"]["a|b"]
        assert alignment["cosine_gap"] > 0.9
        assert alignment["squared_distance_ratio"] < 0.1
        assert alignment["dimension_correlation"] > 0.9

    def test_a_broken_pairing_is_seen_as_chance(self) -> None:
        """The same views with the rows of one modality shuffled: nothing pairs any more."""
        views = _views(300, noise=0.3)
        views["b"] = views["b"][torch.randperm(300, generator=torch.Generator().manual_seed(1))]
        results = evaluatePairedAlignment(_model(), _batches(views))
        metrics = results.retrieval["z_fused"]["a->b"]
        assert metrics["recall@10"] < 3 * metrics["chance_recall@10"]
        assert abs(results.alignment["z_fused"]["a|b"]["cosine_gap"]) < 0.15

    def test_identical_views_are_a_perfect_retrieval(self) -> None:
        views = _views(50, noise=0.0)
        results = evaluatePairedAlignment(_model(), _batches(views))
        for direction in ("a->b", "b->a"):
            assert results.retrieval["z_fused"][direction]["recall@1"] == 1.0
            assert results.retrieval["z_fused"][direction]["mrr"] == pytest.approx(1.0)
        assert results.alignment["z_fused"]["a|b"]["squared_distance_ratio"] == pytest.approx(0.0)

    def test_a_collapsed_latent_space_is_chance_and_not_perfect(self) -> None:
        """Both encoders return one constant for every sample: every candidate ties."""
        constant = {name: torch.full((40, LATENT), 0.5) for name in ("a", "b")}
        results = evaluatePairedAlignment(_model(), _batches(constant), ks=(1, 5))
        for direction in ("a->b", "b->a"):
            metrics = results.retrieval["z_fused"][direction]
            assert metrics["recall@1"] == pytest.approx(1 / 40)
            assert metrics["recall@5"] == pytest.approx(5 / 40)
            assert metrics["mean_rank"] == pytest.approx(20.5)

    def test_ks_and_similarity_are_recorded_and_used(self) -> None:
        views = _views(60, noise=0.5)
        cosine = evaluatePairedAlignment(_model(), _batches(views), ks=(5, 1, 5))
        l2 = evaluatePairedAlignment(_model(), _batches(views), ks=(2,), similarity="l2")
        assert cosine.ks == [1, 5] and cosine.similarity == "cosine"
        assert l2.ks == [2] and l2.similarity == "l2"
        assert "recall@2" in l2.retrieval["z_fused"]["a->b"]
        assert "recall@2" not in cosine.retrieval["z_fused"]["a->b"]

    def test_max_samples_limits_the_pairs(self) -> None:
        results = evaluatePairedAlignment(
            _model(), _batches(_views(100, noise=0.3), 8), max_samples=30
        )
        assert results.num_pairs == {"a|b": 30}

    def test_every_pair_of_three_modalities_is_compared_in_both_directions(self) -> None:
        views = _views(80, noise=0.3, modalities=("a", "b", "c"))
        results = evaluatePairedAlignment(_model(("a", "b", "c")), _batches(views))
        assert set(results.alignment["z_fused"]) == {"a|b", "a|c", "b|c"}
        assert set(results.retrieval["z_fused"]) == {
            "a->b",
            "b->a",
            "a->c",
            "c->a",
            "b->c",
            "c->b",
        }

    def test_only_the_pairs_that_the_data_carries_are_compared(self) -> None:
        views = _views(40, noise=0.3, modalities=("a", "b", "c"))
        batches = [{"a": batch["a"], "b": batch["b"]} for batch in _batches(views)]
        results = evaluatePairedAlignment(_model(("a", "b", "c")), batches)
        assert set(results.alignment["z_fused"]) == {"a|b"}
        assert set(results.num_pairs) == {"a|b"}

    def test_a_pair_is_counted_over_the_batches_that_carry_both(self) -> None:
        views = _views(40, noise=0.3, modalities=("a", "b", "c"))
        batches = _batches(views, 10)
        batches[1] = {"a": batches[1]["a"], "b": batches[1]["b"]}  # no c in this batch
        results = evaluatePairedAlignment(_model(("a", "b", "c")), batches)
        assert results.num_pairs == {"a|b": 40, "a|c": 30, "b|c": 30}

    def test_a_single_modality_model_has_nothing_to_compare_and_reads_no_data(self) -> None:
        def explode() -> Iterator[dict[str, torch.Tensor]]:
            raise AssertionError("the dataloader must not be read")
            yield {}

        results = evaluatePairedAlignment(_model(("a",)), explode())
        assert results.retrieval == {} and results.alignment == {} and results.num_pairs == {}
        assert "fewer than two input modalities" in results.summary()

    def test_data_that_never_pairs_the_modalities_raises(self) -> None:
        batches = [{"a": torch.randn(6, LATENT)}, {"b": torch.randn(6, LATENT)}]
        with pytest.raises(ValueError, match="No pair of input modalities"):
            evaluatePairedAlignment(_model(), batches)

    def test_a_single_paired_sample_is_not_enough(self) -> None:
        batch = {"a": torch.randn(1, LATENT), "b": torch.randn(1, LATENT)}
        with pytest.raises(ValueError, match="at least 2 samples"):
            evaluatePairedAlignment(_model(), [batch])

    def test_the_settings_are_checked_before_the_data_is_read(self) -> None:
        with pytest.raises(ValueError, match="Unknown similarity"):
            evaluatePairedAlignment(_model(), [], similarity="euclid")
        with pytest.raises(ValueError, match="at least one value of k"):
            evaluatePairedAlignment(_model(), [], ks=())

    def test_the_summary_names_the_pairs_the_directions_and_chance(self) -> None:
        summary = evaluatePairedAlignment(_model(), _batches(_views(40, 0.3)), ks=(1, 5)).summary()
        assert "[z_fused] a|b over 40 pairs" in summary
        assert "a->b: recall@1=" in summary and "b->a: recall@1=" in summary
        assert "recall@5=" in summary
        assert "chance 0.0250" in summary
        assert "cosine_paired=" in summary and "dimension_correlation=" in summary

    def test_the_results_survive_a_round_trip_through_json(self, tmp_path: Path) -> None:
        results = evaluatePairedAlignment(_model(), _batches(_views(40, 0.3)))
        results.save(tmp_path / "deep" / "paired.json")
        loaded = json.loads((tmp_path / "deep" / "paired.json").read_text())
        assert loaded == json.loads(json.dumps(results.toDict()))
        assert loaded["retrieval"]["z_fused"]["a->b"]["recall@1"] == pytest.approx(
            results.retrieval["z_fused"]["a->b"]["recall@1"]
        )

    def test_evaluation_leaves_the_model_untouched(self) -> None:
        model = _model()
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        evaluatePairedAlignment(model, _batches(_views(16, 0.3)))
        assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
        assert all(parameter.grad is None for parameter in model.parameters())


def _modelFromYaml(name: str) -> GlobalVae:
    text = (_CONFIG_DIR / "model" / f"{name}.yaml").read_text()
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(text))
    )
    assert isinstance(config, ModelConfig)
    return buildModelFromConfig(config)


def _signalImageBatches(samples: int = 16) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    signal = torch.randn(samples, 256, generator=generator)
    image = torch.rand(samples, 64, 64, generator=generator)
    return [
        {"signal": signal[start : start + 8], "image": image[start : start + 8]}
        for start in range(0, samples, 8)
    ]


class TestRealModels:
    def test_a_latent_space_only_one_modality_reaches_is_not_compared(self) -> None:
        """Shared plus private: the private spaces are fed by one modality each."""
        model = _modelFromYaml("signal_image_shared_private")
        results = evaluatePairedAlignment(model, _signalImageBatches())
        assert set(results.retrieval) == set(results.alignment) == {"z_shared"}
        assert results.num_pairs == {"signal|image": 16}
        assert set(results.retrieval["z_shared"]) == {"signal->image", "image->signal"}

    def test_a_shared_encoder_is_queried_with_each_modality_alone(self) -> None:
        """The shared transformer takes a dict: a missing modality is a shorter stream."""
        model = _modelFromYaml("signal_image_shared_transformer")
        results = evaluatePairedAlignment(model, _signalImageBatches(), ks=(1, 3))
        assert results.num_pairs == {"signal|image": 16}
        metrics = results.retrieval["z_fused"]["signal->image"]
        assert 0.0 <= metrics["recall@1"] <= metrics["recall@3"] <= 1.0
        assert math.isfinite(results.alignment["z_fused"]["signal|image"]["cosine_paired"])

    def test_the_cross_attention_model_is_compared_through_its_fusion(self) -> None:
        model = _modelFromYaml("signal_image_transformer_cross_attention")
        results = evaluatePairedAlignment(model, _signalImageBatches(), ks=(1,))
        assert set(results.retrieval["z_fused"]) == {"signal->image", "image->signal"}


def _meanBatches(sizes: tuple[int, ...]) -> list[dict[str, torch.Tensor]]:
    views = _views(sum(sizes), noise=0.4, seed=3)
    batches, start = [], 0
    for size in sizes:
        batches.append({name: tensor[start : start + size] for name, tensor in views.items()})
        start += size
    return batches


class TestStreamingEvaluate:
    def test_the_streamed_metrics_equal_the_pooled_ones(self) -> None:
        model = _model()
        batches = _meanBatches((8, 8, 5))
        pooled = evaluate(model, batches)
        streamed = evaluate(model, batches, streaming=True)
        assert streamed.reconstruction_metrics.keys() == pooled.reconstruction_metrics.keys()
        for modality, metrics in pooled.reconstruction_metrics.items():
            assert list(streamed.reconstruction_metrics[modality]) == list(metrics)
            for name, value in metrics.items():
                assert streamed.reconstruction_metrics[modality][name] == pytest.approx(
                    value, rel=1e-5, abs=1e-9
                ), (modality, name)

    def test_everything_else_in_the_report_is_unchanged(self) -> None:
        model = _model()
        batches = _meanBatches((8, 8, 5))
        pooled, streamed = evaluate(model, batches), evaluate(model, batches, streaming=True)
        assert streamed.num_samples == pooled.num_samples == 21
        assert streamed.total_reconstruction_loss == pooled.total_reconstruction_loss
        assert streamed.regularization_metrics == pooled.regularization_metrics
        assert streamed.latent_diagnostics == pooled.latent_diagnostics

    def test_max_samples_keeps_the_first_samples_as_the_pooled_path_does(self) -> None:
        model = _model()
        batches = _meanBatches((8, 8, 5))
        pooled = evaluate(model, batches, max_samples=11)
        streamed = evaluate(model, batches, max_samples=11, streaming=True)
        assert streamed.num_samples == pooled.num_samples == 11
        for modality, metrics in pooled.reconstruction_metrics.items():
            for name, value in metrics.items():
                assert streamed.reconstruction_metrics[modality][name] == pytest.approx(
                    value, rel=1e-5, abs=1e-9
                ), (modality, name)

    def test_the_report_says_the_metrics_were_streamed(self) -> None:
        model = _model()
        batches = _meanBatches((8,))
        pooled, streamed = evaluate(model, batches), evaluate(model, batches, streaming=True)
        assert pooled.metrics_streaming is False and streamed.metrics_streaming is True
        assert streamed.toDict()["metrics_streaming"] is True
        assert "reconstruction metrics (streamed):" in streamed.summary()
        assert "reconstruction metrics:" in pooled.summary()
        assert "(streamed)" not in pooled.summary()

    def test_streaming_never_computes_a_pooled_metric(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
            raise AssertionError("a pooled metric was computed")

        monkeypatch.setattr(
            evaluate_module,
            "DEFAULT_RECONSTRUCTION_METRICS",
            {name: forbidden for name in DEFAULT_RECONSTRUCTION_METRICS},
        )
        model = _model()
        batches = _meanBatches((8, 8))
        assert evaluate(model, batches, streaming=True).reconstruction_metrics
        with pytest.raises(AssertionError, match="pooled metric"):
            evaluate(model, batches)

    def test_custom_metrics_need_the_pool(self) -> None:
        with pytest.raises(ValueError, match="cannot take custom `reconstruction_metrics`"):
            evaluate(
                _model(),
                _meanBatches((8,)),
                reconstruction_metrics={"mse": DEFAULT_RECONSTRUCTION_METRICS["mse"]},
                streaming=True,
            )

    def test_an_empty_dataloader_still_raises(self) -> None:
        with pytest.raises(ValueError, match="empty dataloader"):
            evaluate(_model(), [], streaming=True)

    def test_a_decoder_only_target_is_streamed_too(self) -> None:
        """A key with no encoder is a reconstruction target only (ADR 0019): nothing special."""
        model = _model()
        batches = [
            {**batch, "label": torch.arange(batch["a"].shape[0])} for batch in _meanBatches((8,))
        ]
        assert set(evaluate(model, batches, streaming=True).reconstruction_metrics) == {"a", "b"}

    def test_a_single_modality_model_streams_as_well(self) -> None:
        model = buildModelForScript()
        generator = torch.Generator().manual_seed(1)
        batches = [{"signal": torch.randn(8, 16, generator=generator)} for _ in range(3)]
        pooled, streamed = evaluate(model, batches), evaluate(model, batches, streaming=True)
        assert streamed.reconstruction_metrics["signal"]["r2"] == pytest.approx(
            pooled.reconstruction_metrics["signal"]["r2"], rel=1e-5
        )


def _loadScript() -> ModuleType:
    spec = importlib.util.spec_from_file_location("global_vae_evaluate_script_paired", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    return _loadScript()


_TWO_MODEL = "tests.integration._script_fixtures:buildTwoModalityModelForScript"
_TWO_DATA = "tests.integration._script_fixtures:buildTwoModalityDataloaderForScript"
_TWO_GENERATOR = "tests.integration._script_fixtures:buildTwoModalityGeneratorForScript"
_ONE_MODEL = "tests.integration._script_fixtures:buildModelForScript"
_ONE_DATA = "tests.integration._script_fixtures:buildDataloaderForScript"


def _run(
    script: ModuleType,
    tmp_path: Path,
    model_factory: str,
    data_factory: str,
    *flags: str,
    two_modalities: bool = True,
) -> Path:
    model = buildTwoModalityModelForScript() if two_modalities else buildModelForScript()
    checkpoint = tmp_path / "model.pt"
    saveCheckpoint(checkpoint, model=model)
    output_dir = tmp_path / "results"
    script.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--model-factory",
            model_factory,
            "--dataloader-factory",
            data_factory,
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--no-figures",
            *flags,
        ]
    )
    return output_dir


class TestScriptTable:
    def test_a_table_is_written_for_every_run_with_an_output_dir(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        output_dir = _run(script, tmp_path, _ONE_MODEL, _ONE_DATA, two_modalities=False)
        markdown = (output_dir / "metrics_table.md").read_text()
        assert (
            markdown.splitlines()[0] == "| inputs | modality | mse | rmse | mae | r2 | pearson_r |"
        )
        assert "| all | signal |" in markdown
        header = (output_dir / "metrics_table.csv").read_text().splitlines()[0]
        assert header == "inputs,modality,mse,rmse,mae,r2,pearson_r"

    def test_no_paired_file_is_written_without_the_flag(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        output_dir = _run(script, tmp_path, _TWO_MODEL, _TWO_DATA)
        assert not (output_dir / "paired_results.json").exists()
        labels = {
            line.split(",")[0]
            for line in (output_dir / "metrics_table.csv").read_text().splitlines()[1:]
        }
        assert labels == {"all"}


class TestScriptStreaming:
    def test_streaming_flag_gives_the_same_metrics_and_says_so(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        pooled = json.loads(
            (_run(script, tmp_path / "pooled", _TWO_MODEL, _TWO_DATA) / "results.json").read_text()
        )
        streamed = json.loads(
            (
                _run(script, tmp_path / "streamed", _TWO_MODEL, _TWO_DATA, "--streaming")
                / "results.json"
            ).read_text()
        )
        assert pooled["metrics_streaming"] is False and streamed["metrics_streaming"] is True
        for modality, metrics in pooled["reconstruction_metrics"].items():
            for name, value in metrics.items():
                assert streamed["reconstruction_metrics"][modality][name] == pytest.approx(
                    value, rel=1e-4, abs=1e-8
                )


class TestScriptPaired:
    def test_paired_writes_the_retrieval_report_and_the_cross_modal_rows(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        output_dir = _run(script, tmp_path, _TWO_MODEL, _TWO_DATA, "--paired")
        paired = json.loads((output_dir / "paired_results.json").read_text())
        assert paired["similarity"] == "cosine" and paired["ks"] == [1, 5, 10]
        assert set(paired["retrieval"]["z_fused"]) == {"x->y", "y->x"}
        assert paired["num_pairs"] == {"x|y": 16}
        printed = capsys.readouterr().out
        assert "Paired evaluation (retrieval by cosine similarity)" in printed
        assert "| inputs | modality |" in printed
        csv_lines = (output_dir / "metrics_table.csv").read_text().splitlines()[1:]
        assert {line.split(",")[0] for line in csv_lines} == {"all", "x", "y"}
        assert len(csv_lines) == 2 + 2 * 2  # "all" plus the two single-input subsets

    def test_the_retrieval_flags_are_forwarded(self, script: ModuleType, tmp_path: Path) -> None:
        output_dir = _run(
            script,
            tmp_path,
            _TWO_MODEL,
            _TWO_DATA,
            "--paired",
            "--retrieval-k",
            "2",
            "1",
            "--retrieval-similarity",
            "l2",
        )
        paired = json.loads((output_dir / "paired_results.json").read_text())
        assert paired["similarity"] == "l2" and paired["ks"] == [1, 2]
        assert "recall@2" in paired["retrieval"]["z_fused"]["x->y"]

    def test_max_samples_limits_the_pairs(self, script: ModuleType, tmp_path: Path) -> None:
        output_dir = _run(
            script, tmp_path, _TWO_MODEL, _TWO_DATA, "--paired", "--max-samples", "10"
        )
        assert json.loads((output_dir / "paired_results.json").read_text())["num_pairs"] == {
            "x|y": 10
        }

    def test_a_single_use_generator_serves_every_pass(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """Evaluation, retrieval and the cross-modal metrics each ask the factory for data."""
        output_dir = _run(script, tmp_path, _TWO_MODEL, _TWO_GENERATOR, "--paired")
        paired = json.loads((output_dir / "paired_results.json").read_text())
        assert paired["num_pairs"] == {"x|y": 16}
        csv_lines = (output_dir / "metrics_table.csv").read_text().splitlines()[1:]
        assert {line.split(",")[0] for line in csv_lines} == {"all", "x", "y"}

    def test_a_single_modality_model_writes_no_paired_report(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        output_dir = _run(script, tmp_path, _ONE_MODEL, _ONE_DATA, "--paired", two_modalities=False)
        assert not (output_dir / "paired_results.json").exists()
        assert (output_dir / "results.json").exists()

    def test_an_invalid_k_is_a_usage_error(
        self, script: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        checkpoint = tmp_path / "model.pt"
        saveCheckpoint(checkpoint, model=buildTwoModalityModelForScript())
        with pytest.raises(SystemExit) as raised:
            script.main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--model-factory",
                    _TWO_MODEL,
                    "--dataloader-factory",
                    _TWO_DATA,
                    "--paired",
                    "--retrieval-k",
                    "0",
                ]
            )
        assert raised.value.code == 2
        assert "--retrieval-k" in capsys.readouterr().err
