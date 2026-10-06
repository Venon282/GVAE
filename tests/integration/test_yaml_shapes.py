"""Shapes written as lists, as a YAML/Hydra config produces them (spec §9, §10).

YAML has no tuple type, so `output_shape: [64, 64]` reaches a constructor as a `list`.
A `list` never equals a `tuple` in Python, which used to make `TwoDCnnDecoder`'s exact-shape
check reject a configuration whose computed shape was actually correct, and made
`resolveSpatialShape` silently turn `[64, 64]` into `([64, 64], [64, 64])`.

These tests pin the fix at every level it lives at: the shared helpers in
`utils.stage_config`, both 2D decoders, `ResampleTransform` (which resolves its
`target_size` through the same helper), and the real path a config takes, from YAML text
through `OmegaConf` and `buildModelFromConfig`.
"""

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.config.model import ModelConfig, buildModelFromConfig
from global_vae.data.transforms.resample import ResampleTransform
from global_vae.decoders.TwoDCnnDecoder import TwoDCnnDecoder
from global_vae.decoders.TwoDCnnResidualDecoder import TwoDCnnResidualDecoder
from global_vae.utils.stage_config import broadcastPerStageShape, resolveSpatialShape

# Each decoder with the extra kwargs its own defaults need to reach an exact doubling per
# transition (8 -> 16 -> 32 -> 64): the plain decoder's `kernel_sizes=4` default is tuned for
# conv_transpose, the residual decoder's `kernel_sizes=3` default for interpolate_conv.
_DECODERS: list[tuple[type[Any], dict[str, Any]]] = [
    (TwoDCnnDecoder, {"upsample_modes": "conv_transpose"}),
    (TwoDCnnResidualDecoder, {}),
]


def _yamlShapeKwargs() -> dict[str, Any]:
    """Return shape kwargs exactly as OmegaConf materializes them from YAML: plain lists."""
    container = OmegaConf.to_container(
        OmegaConf.create({"output_shape": [64, 64], "seed_shape": [8, 8]}), resolve=True
    )
    assert isinstance(container, dict)
    assert isinstance(container["output_shape"], list)
    return container


class TestResolveSpatialShape:
    """`resolveSpatialShape` treats a `list` and a `tuple` as the same single shape."""

    def test_list_becomes_a_tuple(self) -> None:
        """A list shape is returned as a plain tuple, never as the list itself."""
        resolved = resolveSpatialShape([64, 48], 2, "shape")
        assert resolved == (64, 48)
        assert isinstance(resolved, tuple)

    def test_list_and_tuple_resolve_identically(self) -> None:
        """The same numbers give the same result whichever sequence type carries them."""
        assert resolveSpatialShape([3, 5], 2, "k") == resolveSpatialShape((3, 5), 2, "k")

    def test_int_still_broadcasts_to_every_dimension(self) -> None:
        """A bare int is still a square/cubic shape (unchanged behavior)."""
        assert resolveSpatialShape(4, 3, "k") == (4, 4, 4)

    def test_list_of_wrong_length_raises(self) -> None:
        """A list is length-checked exactly like a tuple."""
        with pytest.raises(ValueError, match="3-dimensional"):
            resolveSpatialShape([1, 2, 3], 2, "shape")


class TestBroadcastPerStageShapeWithListEntries:
    """A per-stage entry may be a list (what YAML gives); a top-level list stays per-stage."""

    def test_list_entries_resolve_to_tuples(self) -> None:
        """`[[3, 5], [3, 5]]` is two stages of one `(3, 5)` kernel, not nested lists."""
        resolved = broadcastPerStageShape([[3, 5], [3, 5]], 2, 2, "kernel_sizes")
        assert resolved == ((3, 5), (3, 5))
        assert all(isinstance(entry, tuple) for entry in resolved)

    def test_mixed_int_tuple_and_list_entries(self) -> None:
        """Entries of different kinds can coexist in one per-stage list."""
        resolved = broadcastPerStageShape([3, (3, 5), [5, 3]], 3, 2, "kernel_sizes")
        assert resolved == ((3, 3), (3, 5), (5, 3))

    def test_top_level_list_is_still_the_per_stage_wrapper(self) -> None:
        """The top-level `list` rule is unchanged: its length must equal the stage count."""
        with pytest.raises(ValueError, match="3 stage"):
            broadcastPerStageShape([1, 2], 3, 2, "strides")

    def test_top_level_tuple_is_still_one_shared_shape(self) -> None:
        """The top-level `tuple` rule is unchanged: one shape shared by every stage."""
        assert broadcastPerStageShape((3, 5), 2, 2, "k") == ((3, 5), (3, 5))


@pytest.mark.parametrize(("decoder_cls", "extra_kwargs"), _DECODERS)
class TestTwoDDecodersAcceptListShapes:
    """Both 2D decoders accept `output_shape`/`seed_shape` as lists."""

    def test_yaml_style_list_shapes_build_and_reconstruct(
        self, decoder_cls: type[Any], extra_kwargs: dict[str, Any]
    ) -> None:
        """The reported bug: a list `output_shape` no longer fails the exact-shape check."""
        decoder = decoder_cls(latent_dim=8, **_yamlShapeKwargs(), **extra_kwargs)
        assert decoder(torch.randn(2, 8)).shape == (2, 64, 64)

    def test_list_and_tuple_build_the_same_decoder(
        self, decoder_cls: type[Any], extra_kwargs: dict[str, Any]
    ) -> None:
        """A list config and a tuple config produce interchangeable modules."""
        from_lists = decoder_cls(latent_dim=8, **_yamlShapeKwargs(), **extra_kwargs)
        from_tuples = decoder_cls(
            latent_dim=8, output_shape=(64, 64), seed_shape=(8, 8), **extra_kwargs
        )
        assert from_lists.state_dict().keys() == from_tuples.state_dict().keys()

    def test_output_shape_is_stored_as_a_tuple(
        self, decoder_cls: type[Any], extra_kwargs: dict[str, Any]
    ) -> None:
        """The normalized shape is a tuple, so later comparisons against tuples hold."""
        decoder = decoder_cls(latent_dim=8, **_yamlShapeKwargs(), **extra_kwargs)
        assert decoder._output_shape == (64, 64)
        assert isinstance(decoder._output_shape, tuple)

    def test_list_of_wrong_length_raises_a_clear_error(
        self, decoder_cls: type[Any], extra_kwargs: dict[str, Any]
    ) -> None:
        """A 3-element list is rejected by name, not by an obscure failure downstream."""
        with pytest.raises(ValueError, match="output_shape"):
            decoder_cls(latent_dim=8, output_shape=[64, 64, 64], **extra_kwargs)

    def test_a_genuinely_unreachable_list_shape_is_still_rejected(
        self, decoder_cls: type[Any], extra_kwargs: dict[str, Any]
    ) -> None:
        """The fix must not weaken the exact-shape guarantee: a wrong shape still raises."""
        with pytest.raises(ValueError, match="output_shape"):
            decoder_cls(latent_dim=8, output_shape=[50, 50], seed_shape=[8, 8], **extra_kwargs)


def test_compute_output_shape_accepts_a_list_seed_shape() -> None:
    """The pre-construction check accepts the same YAML-style shape the constructor does."""
    computed = TwoDCnnDecoder.computeOutputShape(
        seed_shape=[8, 8], hidden_channels=(128, 64, 32), upsample_modes="conv_transpose"
    )
    assert computed == (64, 64)
    computed_residual = TwoDCnnResidualDecoder.computeOutputShape(
        seed_shape=[8, 8], hidden_channels=(128, 64, 32)
    )
    assert computed_residual == (64, 64)


def test_resample_transform_accepts_a_list_target_size() -> None:
    """`DataConfig.transforms` kwargs come from YAML too, so `target_size: [16, 16]` must work."""
    transform = ResampleTransform(target_size=[16, 16], num_spatial_dims=2)
    assert transform.apply(torch.randn(3, 32, 32)).shape == (3, 16, 16)


def test_yaml_text_to_built_model_with_list_output_shape() -> None:
    """The real path end to end: YAML text, `OmegaConf`, `buildModelFromConfig`, a forward pass."""
    yaml_text = """
    name: image_vae
    modalities:
      image:
        encoder: {name: 2d_cnn_encoder_v1}
        decoder:
          name: 2d_cnn_decoder_v1
          kwargs:
            output_shape: [64, 64]
            seed_shape: [8, 8]
            upsample_modes: conv_transpose
    latent_mode: single
    single_latent: {dim: 16}
    """
    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ModelConfig), OmegaConf.create(yaml_text))
    )
    assert isinstance(config, ModelConfig)

    model = buildModelFromConfig(config)
    outputs = model({"image": torch.randn(2, 64, 64)})
    assert outputs["reconstructions"]["image"].shape == (2, 64, 64)
