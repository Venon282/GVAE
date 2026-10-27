"""Unit tests for the audio waveform encoder, decoder and building blocks (roadmap P3-8)."""

import math
from typing import Any

import pytest
import torch
from torch import nn

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
import global_vae.encoders  # noqa: F401  (registers the built-in encoders)
from global_vae.decoders.audio_waveform_decoder import AudioWaveformDecoder
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders
from global_vae.encoders.audio_waveform_encoder import AudioWaveformEncoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders
from global_vae.utils.activations import getActivationFactory, listSupportedActivations
from global_vae.utils.autograd import backward
from global_vae.utils.waveform_blocks import (
    buildDilatedResidualUnits,
    buildDownsampleConv,
    buildUpsampleStage,
    computeDownsampledLength,
    computeSeedLength,
    getNormalizationFactory,
    resolveWaveformStages,
)

LATENT = 4
SMALL: dict[str, Any] = {"hidden_channels": (4, 8), "strides": 2, "dilations": (1, 2)}
LENGTH = 64


def _encoder(**kwargs: Any) -> AudioWaveformEncoder:
    return AudioWaveformEncoder(latent_dim=LATENT, **{**SMALL, **kwargs})


def _decoder(output_length: int = LENGTH, **kwargs: Any) -> AudioWaveformDecoder:
    options = {
        "hidden_channels": (8, 4),
        **{k: v for k, v in SMALL.items() if k != "hidden_channels"},
    }
    return AudioWaveformDecoder(
        latent_dim=LATENT, output_length=output_length, **{**options, **kwargs}
    )


class TestActivations:
    def test_the_factory_gives_a_fresh_module_each_time(self) -> None:
        factory = getActivationFactory("elu")
        assert isinstance(factory(), nn.ELU)
        assert factory() is not factory()

    def test_elu_is_one_of_the_listed_names(self) -> None:
        assert "elu" in listSupportedActivations()

    def test_an_unknown_name_is_refused_and_the_known_ones_listed(self) -> None:
        with pytest.raises(ValueError, match="swish.*elu"):
            getActivationFactory("swish")


class TestRegistration:
    def test_the_encoder_is_registered_under_its_name(self) -> None:
        assert getEncoderClass("audio_waveform_encoder_v1") is AudioWaveformEncoder
        assert "audio_waveform_encoder_v1" in listRegisteredEncoders()

    def test_the_decoder_is_registered_under_its_name(self) -> None:
        assert getDecoderClass("audio_waveform_decoder_v1") is AudioWaveformDecoder
        assert "audio_waveform_decoder_v1" in listRegisteredDecoders()

    def test_they_implement_the_one_modality_interfaces(self) -> None:
        assert issubclass(AudioWaveformEncoder, AbstractEncoder)
        assert issubclass(AudioWaveformDecoder, AbstractDecoder)

    def test_the_modality_name_defaults_to_audio_and_can_be_changed(self) -> None:
        assert _encoder().modality_name == _decoder().modality_name == "audio"
        assert _encoder(modality_name="speech").modality_name == "speech"
        assert _decoder(modality_name="speech").modality_name == "speech"


class TestWaveformBlocks:
    def test_the_normalization_names_resolve(self) -> None:
        assert getNormalizationFactory(None) is None
        group = getNormalizationFactory("group")
        batch = getNormalizationFactory("batch")
        assert group is not None and batch is not None
        assert isinstance(group(8), nn.GroupNorm) and group(8).num_groups == 1
        assert isinstance(batch(8), nn.BatchNorm1d)

    def test_an_unknown_normalization_is_refused(self) -> None:
        with pytest.raises(ValueError, match="normalization 'layer'"):
            getNormalizationFactory("layer")

    def test_a_shared_stride_is_given_to_every_stage(self) -> None:
        assert resolveWaveformStages([4, 8, 16], 3, "Owner") == ((4, 8, 16), (3, 3, 3))

    def test_one_stride_per_stage_is_kept(self) -> None:
        assert resolveWaveformStages([4, 8], [5, 2], "Owner") == ((4, 8), (5, 2))

    @pytest.mark.parametrize(
        ("channels", "strides", "message"),
        [
            ([], 2, "hidden_channels needs at least one"),
            ([4, 0], 2, "hidden_channels must all be at least 1"),
            ([4, 8], 0, "strides must all be at least 1"),
            ([4, 8], [2, 2, 2], "strides"),
        ],
    )
    def test_a_bad_stack_is_refused_and_the_argument_named(
        self, channels: list[int], strides: Any, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            resolveWaveformStages(channels, strides, "Owner")

    @pytest.mark.parametrize("stride", [1, 2, 3, 4, 5, 8])
    @pytest.mark.parametrize("length", [1, 2, 7, 64, 100, 1001])
    def test_a_downsampling_convolution_divides_the_length_rounding_up(
        self, stride: int, length: int
    ) -> None:
        out = buildDownsampleConv(2, 3, stride)(torch.randn(1, 2, length))
        assert out.shape == (1, 3, math.ceil(length / stride))

    @pytest.mark.parametrize("stride", [1, 2, 3, 5])
    def test_an_upsampling_stage_multiplies_the_length_exactly(self, stride: int) -> None:
        out = buildUpsampleStage(2, 3, stride)(torch.randn(1, 2, 11))
        assert out.shape == (1, 3, 11 * stride)

    def test_the_dilated_units_keep_the_length_and_the_width(self) -> None:
        units = buildDilatedResidualUnits(6, (1, 3, 9), 3, "elu", "group")
        assert units(torch.randn(2, 6, 50)).shape == (2, 6, 50)
        assert len(units) == 3

    def test_a_larger_dilation_widens_the_receptive_field(self) -> None:
        """A change at the first sample reaches farther along the series with a dilation of 9."""

        def reach(dilations: tuple[int, ...]) -> int:
            torch.manual_seed(0)
            units = buildDilatedResidualUnits(1, dilations, 3, "elu", None).eval()
            base = torch.zeros(1, 1, 200)
            poked = base.clone()
            poked[0, 0, 100] = 1.0
            with torch.no_grad():
                difference = (units(poked) - units(base)).abs()[0, 0]
            return int((difference > 0).nonzero().max() - (difference > 0).nonzero().min())

        assert reach((1, 9)) > reach((1,))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"dilations": ()}, "dilations"),
            ({"dilations": (1, 0)}, "dilations"),
            ({"kernel_size": 4}, "kernel_size must be odd"),
            ({"activation": "swish"}, "activation"),
            ({"normalization": "layer"}, "normalization"),
        ],
    )
    def test_a_bad_unit_is_refused_and_the_argument_named(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        arguments: dict[str, Any] = {
            "channels": 4,
            "dilations": (1,),
            "kernel_size": 3,
            "activation": "elu",
            "normalization": "group",
            **kwargs,
        }
        with pytest.raises(ValueError, match=message):
            buildDilatedResidualUnits(**arguments)

    def test_the_frame_count_is_the_length_rounded_up_at_each_stage(self) -> None:
        assert computeDownsampledLength(1000, (4, 4, 4)) == 16
        assert computeDownsampledLength(1, (4, 4)) == 1
        assert computeDownsampledLength(10, (5, 1, 3)) == 1

    def test_the_seed_is_the_smallest_one_that_reaches_the_length(self) -> None:
        assert computeSeedLength(16000, (4, 4, 4, 4)) == 63  # 63 * 256 = 16128 >= 16000
        assert computeSeedLength(256, (4, 4, 4, 4)) == 1
        assert computeSeedLength(257, (4, 4, 4, 4)) == 2
        assert computeSeedLength(1, (4,)) == 1


class TestEncoder:
    def test_output_shapes_from_a_plain_waveform(self) -> None:
        mu, logvar = _encoder()(torch.randn(3, LENGTH))
        assert mu.shape == logvar.shape == (3, LATENT)

    def test_an_explicit_channel_dimension_gives_the_same_result(self) -> None:
        encoder = _encoder().eval()
        x = torch.randn(3, LENGTH)
        assert torch.allclose(encoder(x)[0], encoder(x.unsqueeze(1))[0])

    def test_stereo(self) -> None:
        mu, _ = _encoder(in_channels=2)(torch.randn(3, 2, LENGTH))
        assert mu.shape == (3, LATENT)

    @pytest.mark.parametrize("length", [1, 2, 3, 17, 64, 1000])
    def test_a_clip_of_any_length_is_encoded_to_the_same_shape(self, length: int) -> None:
        mu, logvar = _encoder()(torch.randn(2, length))
        assert mu.shape == logvar.shape == (2, LATENT)

    def test_it_reports_its_latent_width_and_a_minimal_length_of_one_sample(self) -> None:
        encoder = _encoder()
        assert encoder.latent_dim == LATENT
        assert encoder.minimal_input_length == 1

    def test_the_stack_leaves_the_frames_the_blocks_compute(self) -> None:
        encoder = _encoder(hidden_channels=(4, 4, 8), strides=(2, 3, 5))
        frames = encoder.body(torch.randn(1, 1, 1000)).shape[2]
        assert frames == computeDownsampledLength(1000, (2, 3, 5))

    def test_a_max_over_time_differs_from_the_mean(self) -> None:
        torch.manual_seed(0)
        average = _encoder(global_pool="avg").eval()
        maximum = _encoder(global_pool="max").eval()
        maximum.load_state_dict(average.state_dict())
        x = torch.randn(2, LENGTH)
        assert not torch.allclose(average(x)[0], maximum(x)[0])

    def test_a_head_with_hidden_layers_adds_parameters(self) -> None:
        plain = sum(p.numel() for p in _encoder().parameters())
        deeper = sum(p.numel() for p in _encoder(head_hidden_dims=(16,)).parameters())
        assert deeper > plain

    def test_group_normalization_does_not_depend_on_the_rest_of_the_batch(self) -> None:
        encoder = _encoder(normalization="group").train()
        x = torch.randn(4, LENGTH)
        alone = encoder(x[:1])[0]
        together = encoder(x)[0][:1]
        assert torch.allclose(alone, together, atol=1e-5)

    def test_batch_normalization_builds_and_a_single_value_per_channel_is_refused(self) -> None:
        """One clip of many samples has enough values per channel; one sample of one clip, not."""
        encoder = _encoder(normalization="batch")
        assert encoder(torch.randn(1, LENGTH))[0].shape == (1, LATENT)
        with pytest.raises(ValueError, match="more than 1 value per channel"):
            encoder(torch.randn(1, 1))
        assert encoder.eval()(torch.randn(1, 1))[0].shape == (1, LATENT)

    @pytest.mark.parametrize("shape", [(LENGTH,), (2, 3, 1, LENGTH), (2, 2, LENGTH), (2, 0)])
    def test_a_tensor_that_is_not_a_batch_of_mono_waveforms_is_refused(
        self, shape: tuple[int, ...]
    ) -> None:
        with pytest.raises(ValueError, match="AudioWaveformEncoder"):
            _encoder()(torch.randn(*shape))

    def test_a_channel_count_other_than_the_built_one_is_refused_and_named(self) -> None:
        with pytest.raises(ValueError, match="in_channels=2, got 1"):
            _encoder(in_channels=2)(torch.randn(3, 1, LENGTH))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"latent_dim": 0}, "latent_dim"),
            ({"in_channels": 0}, "in_channels"),
            ({"global_pool": "median"}, "global_pool"),
            ({"stem_kernel_size": 4}, "stem_kernel_size"),
            ({"hidden_channels": ()}, "hidden_channels"),
            ({"strides": (2, 2, 2)}, "strides"),
            ({"dilations": ()}, "dilations"),
            ({"kernel_size": 2}, "kernel_size"),
            ({"activation": "swish"}, "activation"),
            ({"normalization": "layer"}, "normalization"),
        ],
    )
    def test_a_bad_constructor_argument_is_refused_and_named(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        arguments: dict[str, Any] = {"latent_dim": LATENT, **SMALL, **kwargs}
        with pytest.raises(ValueError, match=message):
            AudioWaveformEncoder(**arguments)

    def test_every_parameter_receives_a_gradient(self) -> None:
        encoder = _encoder(head_hidden_dims=(8,))
        mu, logvar = encoder(torch.randn(3, LENGTH))
        backward(mu.sum() + logvar.sum())
        assert [n for n, p in encoder.named_parameters() if p.grad is None] == []


class TestDecoder:
    def test_output_shape_is_the_length_it_was_built_for(self) -> None:
        assert _decoder()(torch.randn(3, LATENT)).shape == (3, LENGTH)

    @pytest.mark.parametrize("length", [1, 7, 64, 100, 777, 1000])
    def test_any_number_of_samples_is_reached_whatever_the_strides(self, length: int) -> None:
        decoder = _decoder(length, strides=(3, 2))
        assert decoder(torch.randn(2, LATENT)).shape == (2, length)

    def test_a_length_that_no_stack_of_doublings_reaches(self) -> None:
        decoder = AudioWaveformDecoder(
            latent_dim=LATENT,
            output_length=16000,
            hidden_channels=(4, 4),
            strides=(5, 4),
            dilations=(1,),
        )
        assert decoder(torch.randn(1, LATENT)).shape == (1, 16000)

    def test_stereo(self) -> None:
        assert _decoder(out_channels=2)(torch.randn(3, LATENT)).shape == (3, 2, LENGTH)

    def test_the_waveform_is_bounded_by_the_default_tanh(self) -> None:
        out = _decoder()(torch.randn(8, LATENT) * 50)
        assert out.abs().max() <= 1.0

    def test_without_an_output_activation_the_waveform_is_not_bounded(self) -> None:
        torch.manual_seed(0)
        out = _decoder(output_activation=None)(torch.randn(8, LATENT) * 50)
        assert out.abs().max() > 1.0

    def test_every_parameter_receives_a_gradient(self) -> None:
        decoder = _decoder()
        backward(decoder(torch.randn(3, LATENT)).sum())
        assert [n for n, p in decoder.named_parameters() if p.grad is None] == []

    def test_a_latent_of_another_width_is_refused_and_named(self) -> None:
        with pytest.raises(ValueError, match=rf"\(batch, {LATENT}\)"):
            _decoder()(torch.randn(3, LATENT + 1))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"latent_dim": 0}, "latent_dim"),
            ({"output_length": 0}, "output_length"),
            ({"out_channels": 0}, "out_channels"),
            ({"output_kernel_size": 2}, "output_kernel_size"),
            ({"hidden_channels": ()}, "hidden_channels"),
            ({"strides": (2, 2, 2)}, "strides"),
            ({"dilations": (0,)}, "dilations"),
            ({"activation": "swish"}, "activation"),
            ({"normalization": "layer"}, "normalization"),
            ({"output_activation": "swish"}, "activation"),
        ],
    )
    def test_a_bad_constructor_argument_is_refused_and_named(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        arguments: dict[str, Any] = {
            "latent_dim": LATENT,
            "output_length": LENGTH,
            "hidden_channels": (8, 4),
            "strides": 2,
            "dilations": (1,),
            **kwargs,
        }
        with pytest.raises(ValueError, match=message):
            AudioWaveformDecoder(**arguments)


class TestEncoderAndDecoderTogether:
    def test_a_clip_goes_through_both_and_comes_back_at_its_length(self) -> None:
        x = torch.randn(3, LENGTH)
        mu, _ = _encoder()(x)
        assert _decoder()(mu).shape == x.shape

    def test_a_tiny_model_learns_to_reproduce_one_tone(self) -> None:
        torch.manual_seed(0)
        t = torch.arange(LENGTH).float()
        clip = (0.8 * torch.sin(2 * math.pi * t / 16)).repeat(4, 1)
        encoder, decoder = _encoder(), _decoder()
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=3e-3)

        def loss() -> torch.Tensor:
            return torch.nn.functional.mse_loss(decoder(encoder(clip)[0]), clip)

        first = loss().item()
        for _ in range(60):
            optimizer.zero_grad()
            backward(loss())
            optimizer.step()
        assert loss().item() < first / 2
