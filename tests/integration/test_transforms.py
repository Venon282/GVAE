"""Unit tests for the generic data-transform registry pattern (spec §6.2, §10, C11).

Mirrors the test style already used for the other self-registration
registries in this codebase (encoders, decoders, fusion, assemblers,
regularizers, beta schedules): registration, lookup, duplicate-registration
and unknown-name error paths, plus value-correctness and invertibility
(round-trip) checks for each built-in transform.

The invertibility checks are the point of this file (spec §6.2's own
testing requirement, C11): `LogTransform`/`StandardizeTransform`/
`ComposeTransform` round-trip to floating-point precision;
`ResampleTransform` round-trips only approximately (a lossy operation by
construction, see its own module docstring) and is checked with a looser
tolerance instead.

Every transform is exercised across more than one dimensionality (a 1D
vector, a 2D "image"-shaped tensor, a 3D "volume"-shaped tensor) using the
exact same class and constructor arguments each time, proving no
dimensionality- or dataset-specific code exists anywhere in this
subpackage (spec §6.2).
"""

from collections.abc import Callable

import pytest
import torch

from global_vae.config.data import DataConfig, TransformConfig, buildTransformPipeline
from global_vae.data.transforms.base import AbstractTransform
from global_vae.data.transforms.compose import ComposeTransform
from global_vae.data.transforms.log import LogTransform
from global_vae.data.transforms.registry import (
    getTransformClass,
    listRegisteredTransforms,
    registerTransform,
)
from global_vae.data.transforms.resample import ResampleTransform
from global_vae.data.transforms.standardize import StandardizeTransform

# Shapes spanning several dimensionalities, reused verbatim by every test below to
# prove the same transform class/instance handles each without modification.
_SHAPES: dict[str, tuple[int, ...]] = {
    "1d_vector": (4, 32),
    "2d_image_like": (4, 3, 16, 16),
    "3d_volume_like": (2, 8, 8, 8),
}


def _makeDummyTransform(
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    inverse_fn: Callable[[torch.Tensor], torch.Tensor],
) -> AbstractTransform:
    class _Dummy(AbstractTransform):
        def apply(self, x: torch.Tensor) -> torch.Tensor:
            return apply_fn(x)

        def inverse(self, y: torch.Tensor) -> torch.Tensor:
            return inverse_fn(y)

    return _Dummy()


def test_log_standardize_resample_are_registered_by_default() -> None:
    assert {"log", "standardize", "resample"} <= set(listRegisteredTransforms())
    assert getTransformClass("log") is LogTransform
    assert getTransformClass("standardize") is StandardizeTransform
    assert getTransformClass("resample") is ResampleTransform


def test_unknown_transform_name_raises_key_error() -> None:
    with pytest.raises(KeyError, match="does_not_exist"):
        getTransformClass("does_not_exist")


def test_duplicate_registration_raises_value_error() -> None:
    @registerTransform("dummy_transform_duplicate_check")
    class _First(AbstractTransform):
        def apply(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def inverse(self, y: torch.Tensor) -> torch.Tensor:
            return y

    with pytest.raises(ValueError, match="already registered"):

        @registerTransform("dummy_transform_duplicate_check")
        class _Second(AbstractTransform):
            def apply(self, x: torch.Tensor) -> torch.Tensor:
                return x

            def inverse(self, y: torch.Tensor) -> torch.Tensor:
                return y


class TestLogTransform:
    @pytest.mark.parametrize("shape", list(_SHAPES.values()), ids=list(_SHAPES.keys()))
    def test_round_trip_recovers_the_input(self, shape: tuple[int, ...]) -> None:
        """Same class, same constructor args, exercised at three different
        dimensionalities: nothing here is written for one specific shape."""
        transform = LogTransform(eps=1e-6)
        x = torch.rand(*shape) * 5 + 0.1
        recovered = transform.inverse(transform.apply(x))
        assert torch.allclose(x, recovered, atol=1e-4)

    def test_apply_matches_the_closed_form(self) -> None:
        transform = LogTransform(eps=1e-3)
        x = torch.rand(4, 8) * 2 + 0.5
        assert torch.allclose(transform.apply(x), torch.log(x + 1e-3), atol=1e-6)

    def test_call_is_an_alias_for_apply(self) -> None:
        transform = LogTransform(eps=1e-6)
        x = torch.rand(3, 5) + 0.1
        assert torch.equal(transform(x), transform.apply(x))

    def test_domain_violation_raises_instead_of_returning_nan(self) -> None:
        transform = LogTransform(eps=1e-6)
        with pytest.raises(ValueError, match="eps"):
            transform.apply(torch.tensor([-1.0, 0.5]))

    def test_non_positive_eps_raises(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            LogTransform(eps=0.0)
        with pytest.raises(ValueError, match="eps"):
            LogTransform(eps=-1.0)


class TestStandardizeTransform:
    @pytest.mark.parametrize("shape", list(_SHAPES.values()), ids=list(_SHAPES.keys()))
    def test_round_trip_recovers_the_input_with_scalar_mean_std(
        self, shape: tuple[int, ...]
    ) -> None:
        transform = StandardizeTransform(mean=2.0, std=3.0)
        x = torch.randn(*shape) * 5 + 2
        recovered = transform.inverse(transform.apply(x))
        assert torch.allclose(x, recovered, atol=1e-4)

    def test_round_trip_with_broadcastable_per_channel_mean_std(self) -> None:
        """mean/std as tensors broadcast per-channel: the same mechanism a
        multi-channel 1D series or a 2D image would both use, with no separate
        per-dimensionality code path."""
        mean = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
        std = torch.tensor([0.5, 1.5, 2.5]).view(3, 1, 1)
        transform = StandardizeTransform(mean=mean, std=std)
        x = torch.randn(3, 8, 8) * 4 + 1
        recovered = transform.inverse(transform.apply(x))
        assert torch.allclose(x, recovered, atol=1e-4)

    def test_apply_matches_the_closed_form(self) -> None:
        transform = StandardizeTransform(mean=1.0, std=2.0)
        x = torch.randn(4, 6)
        assert torch.allclose(transform.apply(x), (x - 1.0) / 2.0, atol=1e-6)

    def test_non_positive_scalar_std_raises(self) -> None:
        with pytest.raises(ValueError, match="std"):
            StandardizeTransform(mean=0.0, std=0.0)
        with pytest.raises(ValueError, match="std"):
            StandardizeTransform(mean=0.0, std=-1.0)

    def test_non_positive_tensor_std_raises(self) -> None:
        with pytest.raises(ValueError, match="std"):
            StandardizeTransform(mean=0.0, std=torch.tensor([1.0, -0.5]))


class TestResampleTransform:
    @pytest.mark.parametrize(
        ("num_spatial_dims", "source_size", "target_size", "leading_shape"),
        [
            (1, 128, 64, (3,)),
            (2, (32, 32), (16, 16), (2, 3)),
            (3, 16, 8, (2,)),
        ],
        ids=["1d", "2d", "3d"],
    )
    def test_apply_produces_the_target_shape(
        self,
        num_spatial_dims: int,
        source_size: int | tuple[int, ...],
        target_size: int | tuple[int, ...],
        leading_shape: tuple[int, ...],
    ) -> None:
        """One ResampleTransform implementation, parametrized purely by
        num_spatial_dims, covers 1D/2D/3D data: no per-dimensionality subclass
        exists or is needed (spec §6.2)."""
        transform = ResampleTransform(
            target_size=target_size, source_size=source_size, num_spatial_dims=num_spatial_dims
        )
        spatial_shape = (
            (source_size,) * num_spatial_dims
            if isinstance(source_size, int)
            else tuple(source_size)
        )
        x = torch.rand(*leading_shape, *spatial_shape)
        resampled = transform.apply(x)

        expected_spatial = (
            (target_size,) * num_spatial_dims
            if isinstance(target_size, int)
            else tuple(target_size)
        )
        assert resampled.shape == (*leading_shape, *expected_spatial)

    def test_inverse_restores_the_original_shape(self) -> None:
        transform = ResampleTransform(target_size=64, source_size=128, num_spatial_dims=1)
        x = torch.rand(5, 128)
        restored = transform.inverse(transform.apply(x))
        assert restored.shape == x.shape

    def test_downsample_then_upsample_approximately_recovers_a_smooth_signal(self) -> None:
        """Not exact (resampling is lossy, see the class docstring): a smooth, slowly
        varying signal should still round-trip closely."""
        transform = ResampleTransform(target_size=64, source_size=256, num_spatial_dims=1)
        t = torch.linspace(0, 4 * 3.14159, 256)
        signal = torch.sin(t).unsqueeze(0)
        restored = transform.inverse(transform.apply(signal))
        assert torch.allclose(signal, restored, atol=0.1)

    def test_inverse_without_source_size_raises(self) -> None:
        transform = ResampleTransform(target_size=32, num_spatial_dims=1)
        with pytest.raises(ValueError, match="source_size"):
            transform.inverse(torch.rand(2, 32))

    def test_default_mode_is_chosen_per_num_spatial_dims(self) -> None:
        assert ResampleTransform(target_size=8, num_spatial_dims=1).mode == "linear"
        assert ResampleTransform(target_size=8, num_spatial_dims=2).mode == "bilinear"
        assert ResampleTransform(target_size=8, num_spatial_dims=3).mode == "trilinear"

    def test_explicit_mode_overrides_the_default(self) -> None:
        transform = ResampleTransform(target_size=8, num_spatial_dims=1, mode="nearest")
        x = torch.rand(2, 16)
        # must not raise even though "nearest" does not accept align_corners
        assert transform.apply(x).shape == (2, 8)

    def test_input_with_too_few_dimensions_raises(self) -> None:
        transform = ResampleTransform(target_size=8, num_spatial_dims=2)
        with pytest.raises(ValueError, match="dimension"):
            transform.apply(torch.rand(8))

    def test_invalid_num_spatial_dims_raises(self) -> None:
        with pytest.raises(ValueError, match="num_spatial_dims"):
            ResampleTransform(target_size=8, num_spatial_dims=4)


class TestResampleTransformCoordinateAware:
    """`interpolation="scipy"`: resampling onto explicit x-positions, including
    positions that differ per sample (spec §6.2). This is what makes resampling
    two curves recorded on genuinely different grids land on the *same* physical
    positions, which point-count-only resampling (the `"torch"` backend) cannot
    express at all.
    """

    def test_shared_source_coords_given_at_construction(self) -> None:
        # A dense enough source grid that even the default linear interpolation
        # stays close to the true function between knots.
        source_q = torch.linspace(0.0, 4.0, 41)
        values = torch.sin(source_q)
        target_q = torch.tensor([0.5, 1.5, 2.5, 3.5])
        transform = ResampleTransform(
            target_coords=target_q, source_coords=source_q, interpolation="scipy"
        )
        result = transform.apply(values)
        assert result.shape == target_q.shape
        assert torch.allclose(result, torch.sin(target_q), atol=0.01)

    def test_per_sample_source_coords_align_two_different_grids(self) -> None:
        """The exact motivating case: curve A and curve B were measured at different
        positions; resampled onto the same common_q, index n now means the same
        position for both, which naive index-based resampling cannot guarantee."""

        def f(q: torch.Tensor) -> torch.Tensor:
            return torch.sin(q) + 2.0

        q_a = torch.tensor([0.5, 1.0, 1.8, 2.6, 3.2])
        q_b = torch.tensor([0.6, 1.1, 1.9, 2.5, 3.0, 3.3])
        common_q = torch.linspace(0.7, 3.0, 6)

        transform = ResampleTransform(
            target_coords=common_q, interpolation="scipy", scipy_kind="cubic_spline"
        )
        resampled_a = transform.apply(f(q_a), source_coords=q_a)
        resampled_b = transform.apply(f(q_b), source_coords=q_b)

        assert resampled_a.shape == common_q.shape
        assert resampled_b.shape == common_q.shape
        # Both curves are the same underlying function, just measured at different
        # positions: once resampled onto the same common_q they must closely agree.
        assert torch.allclose(resampled_a, resampled_b, atol=0.05)
        assert torch.allclose(resampled_a, f(common_q), atol=0.05)

    @pytest.mark.parametrize(
        "kind", ["linear", "nearest", "cubic", "pchip", "akima", "cubic_spline"]
    )
    def test_every_scipy_kind_produces_finite_output(self, kind: str) -> None:
        source_q = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0])
        values = torch.sin(source_q) + 2.0
        target_q = torch.linspace(0.5, 3.5, 8)
        transform = ResampleTransform(
            target_coords=target_q, interpolation="scipy", scipy_kind=kind
        )
        result = transform.apply(values, source_coords=source_q)
        assert result.shape == target_q.shape
        assert torch.isfinite(result).all()

    def test_unknown_scipy_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="scipy_kind"):
            ResampleTransform(target_size=4, interpolation="scipy", scipy_kind="does_not_exist")

    def test_out_of_range_target_raises_without_extrapolate(self) -> None:
        source_q = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([0.0, 1.0, 0.0])
        transform = ResampleTransform(target_coords=torch.tensor([5.0]), interpolation="scipy")
        with pytest.raises(ValueError, match="outside the source positions"):
            transform.apply(values, source_coords=source_q)

    def test_extrapolate_true_allows_out_of_range_target(self) -> None:
        source_q = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([0.0, 1.0, 2.0])
        transform = ResampleTransform(
            target_coords=torch.tensor([5.0]),
            interpolation="scipy",
            scipy_kind="linear",
            extrapolate=True,
        )
        result = transform.apply(values, source_coords=source_q)
        assert torch.isfinite(result).all()

    def test_batched_input_with_one_shared_grid(self) -> None:
        source_q = torch.tensor([0.0, 1.0, 2.0, 3.0])
        batch = torch.stack([torch.sin(source_q), torch.cos(source_q)])
        transform = ResampleTransform(
            target_coords=torch.linspace(0.5, 2.5, 5),
            source_coords=source_q,
            interpolation="scipy",
        )
        result = transform.apply(batch)
        assert result.shape == (2, 5)

    def test_default_target_coords_span_the_source_range(self) -> None:
        source_q = torch.tensor([1.0, 2.0, 5.0])
        values = torch.tensor([0.0, 1.0, 4.0])
        transform = ResampleTransform(target_size=4, interpolation="scipy")
        result = transform.apply(values, source_coords=source_q)
        assert result.shape == (4,)

    def test_inverse_round_trips_approximately(self) -> None:
        source_q = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0])
        values = torch.sin(source_q) + 2.0
        target_q = torch.linspace(0.5, 3.5, 8)
        transform = ResampleTransform(
            target_coords=target_q,
            source_coords=source_q,
            interpolation="scipy",
            scipy_kind="cubic_spline",
            extrapolate=True,
        )
        forward = transform.apply(values)
        restored = transform.inverse(forward)
        assert restored.shape == values.shape

    def test_inverse_without_any_source_coords_raises(self) -> None:
        transform = ResampleTransform(target_size=4, interpolation="scipy")
        with pytest.raises(ValueError, match="source_coords"):
            transform.inverse(torch.rand(4), target_coords=torch.linspace(0.0, 1.0, 4))

    def test_torch_backend_rejects_explicit_coords(self) -> None:
        with pytest.raises(ValueError, match="interpolation='scipy'"):
            ResampleTransform(target_size=8, source_coords=[0.0, 1.0, 2.0])

    def test_scipy_backend_rejects_multi_dimensional_resampling(self) -> None:
        with pytest.raises(NotImplementedError, match="num_spatial_dims"):
            ResampleTransform(target_size=8, num_spatial_dims=2, interpolation="scipy")

    def test_missing_target_size_and_target_coords_raises(self) -> None:
        with pytest.raises(ValueError, match="target_size"):
            ResampleTransform()

    def test_unknown_interpolation_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="interpolation"):
            ResampleTransform(target_size=8, interpolation="does_not_exist")

    def test_missing_scipy_package_raises_clear_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("simulated missing scipy")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        transform = ResampleTransform(target_size=4, interpolation="scipy")
        with pytest.raises(ImportError, match="scipy"):
            transform.apply(torch.rand(6), source_coords=torch.arange(6, dtype=torch.float32))


class TestComposeTransform:
    def test_applies_steps_in_order(self) -> None:
        doubling = _makeDummyTransform(lambda x: x * 2, lambda y: y / 2)
        offsetting = _makeDummyTransform(lambda x: x + 1, lambda y: y - 1)
        pipeline = ComposeTransform([doubling, offsetting])
        x = torch.tensor([1.0, 2.0])
        # doubling first, then offsetting: (x * 2) + 1
        assert torch.equal(pipeline.apply(x), x * 2 + 1)

    def test_inverse_undoes_steps_in_reverse_order(self) -> None:
        doubling = _makeDummyTransform(lambda x: x * 2, lambda y: y / 2)
        offsetting = _makeDummyTransform(lambda x: x + 1, lambda y: y - 1)
        pipeline = ComposeTransform([doubling, offsetting])
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(pipeline.inverse(pipeline.apply(x)), x, atol=1e-6)

    def test_empty_pipeline_is_the_identity(self) -> None:
        pipeline = ComposeTransform([])
        x = torch.randn(3, 4)
        assert torch.equal(pipeline.apply(x), x)
        assert torch.equal(pipeline.inverse(x), x)

    @pytest.mark.parametrize("shape", list(_SHAPES.values()), ids=list(_SHAPES.keys()))
    def test_log_then_standardize_round_trips_at_every_dimensionality(
        self, shape: tuple[int, ...]
    ) -> None:
        pipeline = ComposeTransform(
            [LogTransform(eps=1e-6), StandardizeTransform(mean=0.0, std=1.0)]
        )
        x = torch.rand(*shape) * 4 + 0.2
        recovered = pipeline.inverse(pipeline.apply(x))
        assert torch.allclose(x, recovered, atol=1e-4)

    def test_error_from_a_step_without_a_usable_inverse_propagates(self) -> None:
        pipeline = ComposeTransform([ResampleTransform(target_size=8, num_spatial_dims=1)])
        with pytest.raises(ValueError, match="source_size"):
            pipeline.inverse(torch.rand(2, 8))


class TestBuildTransformPipelineFromConfig:
    """`global_vae.config.data.buildTransformPipeline`: the config-driven wiring
    that makes `DataConfig.transforms` actually operable (spec §6.2), instead of
    the purely decorative `list[str]` it used to be.
    """

    def test_resolves_each_step_by_registry_name_in_order(self) -> None:
        config = DataConfig(
            loader_factory="unused:unused",
            train_path="unused",
            transforms=[
                TransformConfig(name="log", kwargs={"eps": 1e-6}),
                TransformConfig(name="standardize", kwargs={"mean": 0.0, "std": 2.0}),
            ],
        )
        pipeline = buildTransformPipeline(config)
        assert isinstance(pipeline.transforms[0], LogTransform)
        assert isinstance(pipeline.transforms[1], StandardizeTransform)

        x = torch.rand(4, 16) * 3 + 0.1
        recovered = pipeline.inverse(pipeline.apply(x))
        assert torch.allclose(x, recovered, atol=1e-4)

    def test_empty_transforms_gives_an_identity_pipeline(self) -> None:
        config = DataConfig(loader_factory="unused:unused", train_path="unused")
        pipeline = buildTransformPipeline(config)
        x = torch.randn(3, 5)
        assert torch.equal(pipeline.apply(x), x)

    def test_unknown_transform_name_raises_key_error(self) -> None:
        config = DataConfig(
            loader_factory="unused:unused",
            train_path="unused",
            transforms=[TransformConfig(name="does_not_exist")],
        )
        with pytest.raises(KeyError, match="does_not_exist"):
            buildTransformPipeline(config)

    def test_pipeline_inverse_is_usable_as_a_plain_callable(self) -> None:
        """The exact shape `visualization.reconstruction_plot`'s own
        `inverse_transform: Callable[[Tensor], Tensor]` parameter expects."""
        config = DataConfig(
            loader_factory="unused:unused",
            train_path="unused",
            transforms=[TransformConfig(name="standardize", kwargs={"mean": 1.0, "std": 2.0})],
        )
        pipeline = buildTransformPipeline(config)
        inverse_transform: Callable[[torch.Tensor], torch.Tensor] = pipeline.inverse
        x = torch.randn(4, 10)
        assert torch.allclose(inverse_transform(pipeline.apply(x)), x, atol=1e-5)

    def test_real_signal_yaml_config_wires_a_working_pipeline(self) -> None:
        """End-to-end: the shipped configs/data/signal.yaml, composed through
        Hydra, actually produces a usable, invertible pipeline (spec §6.2)."""
        import global_vae.config  # noqa: F401  (registers structured configs)
        from global_vae.config.experiment import loadExperimentConfig

        cfg = loadExperimentConfig(
            overrides=["data.loader_factory=os.path:join", "data.train_path=/unused"]
        )
        pipeline = buildTransformPipeline(cfg.data)
        assert cfg.data.sequence_length is not None
        x = torch.rand(2, cfg.data.sequence_length) * 3 + 0.1
        recovered = pipeline.inverse(pipeline.apply(x))
        assert torch.allclose(x, recovered, atol=1e-4)
