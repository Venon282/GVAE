"""The encode and decode seams of `GlobalVae` (roadmap P5-5, ADR 0046).

`GlobalVae.forward` used to be the only public path, and it always ran encode, fuse, sample,
assemble and decode. It is now `encode` followed by `decode`, and each half is public. What this
file pins:

* **Bit for bit with 2.0.0.** `_legacyEncode` and `_legacyDecode` are the body of the 2.0.0
  `forward`, copied without a change and cut at the one place where the sampling ends and the
  decoding starts. `forward`, `encode` and `decode` must give exactly those tensors, under a fixed
  seed, on every row of the configuration matrix, on every subset of the input modalities and of
  the latent spaces (the rules for a missing one), and on a model with the real shared modules.
* **`encode` followed by `decode` equals `forward`**, and neither half runs the other's modules.
* **Cross decoding** needs no method of its own: the sample of one latent space is passed under the
  name of another. It needs equal widths, and a clear error says so when they differ.
* The errors of `encode`, `decode` and `forward` name the method that was called.
"""

import itertools
from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest
import torch
from torch import nn

from global_vae.assemblers.base import AbstractAssembler
from global_vae.decoders.shared_base import AbstractSharedDecoder
from global_vae.encoders.shared_base import AbstractSharedEncoder
from global_vae.latent.base import LatentSpace, RoutingGraph
from global_vae.models.global_vae import GlobalVae
from tests.integration import test_transformer_model_configs as transformer_configs
from tests.integration.test_configuration_matrix import (
    BATCH_SIZE,
    IMAGE_DECODER,
    IMAGE_ENCODER,
    IMAGE_SHAPE,
    MATRIX,
    MATRIX_CASES,
    SIGNAL_DECODER,
    SIGNAL_ENCODER,
    SIGNAL_LENGTH,
    _decoderKwargs,
    _encoderKwargs,
)

Params = dict[str, tuple[torch.Tensor, torch.Tensor]]
Samples = dict[str, torch.Tensor]


# ---------------------------------------------------------------------------------------------
# The 2.0.0 forward, copied without a change.
# ---------------------------------------------------------------------------------------------


def _legacyEncode(
    model: GlobalVae, inputs: dict[str, torch.Tensor], use_mean: bool
) -> tuple[Params, Samples]:
    """The first half of the 2.0.0 `GlobalVae.forward`: encode, fuse, sample."""
    inputs_by_encoder: dict[str, dict[str, torch.Tensor]] = {}
    for modality, x in inputs.items():
        inputs_by_encoder.setdefault(model._modality_owner[modality], {})[modality] = x
    encoder_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for encoder_name, modality_inputs in inputs_by_encoder.items():
        encoder = model.encoders[encoder_name]
        if isinstance(encoder, AbstractSharedEncoder):
            ordered = {
                modality: modality_inputs[modality]
                for modality in model._encoder_modalities[encoder_name]
                if modality in modality_inputs
            }
            encoder_outputs[encoder_name] = encoder(ordered)
        else:
            encoder_outputs[encoder_name] = encoder(modality_inputs[encoder_name])

    latent_params: Params = {}
    latent_samples: Samples = {}
    for latent_name, encoder_names in model._feeding_encoders.items():
        active = [name for name in encoder_names if name in encoder_outputs]
        if not active:
            continue
        edge_params = {
            name: model._adaptThroughHead(name, latent_name, encoder_outputs[name])
            for name in active
        }
        if len(active) == 1:
            mu, logvar = edge_params[active[0]]
        else:
            mu, logvar = model.fusions[latent_name](edge_params)
        latent_params[latent_name] = (mu, logvar)
        latent_samples[latent_name] = (
            mu if use_mean else model.latent_spaces[latent_name].reparameterize(mu, logvar)
        )
    return latent_params, latent_samples


def _legacyDecode(model: GlobalVae, latent_samples: Samples) -> dict[str, torch.Tensor]:
    """The second half of the 2.0.0 `GlobalVae.forward`: assemble and decode."""
    reconstructions: dict[str, torch.Tensor] = {}
    for decoder_name, decoder in model.decoders.items():
        required = model._decoder_inputs.get(decoder_name, [])
        consumed = [latent_name for latent_name in required if latent_name in latent_samples]
        if not consumed:
            continue
        if decoder_name not in model.assemblers:
            z_in = latent_samples[consumed[0]]
        else:
            assembler = cast(AbstractAssembler, model.assemblers[decoder_name])
            if len(consumed) < len(required) and not assembler.handlesMissingLatents:
                continue
            present = (
                None
                if len(consumed) == len(required)
                else [required.index(name) for name in consumed]
            )
            z_in = assembler([latent_samples[name] for name in consumed], present)
        if isinstance(decoder, AbstractSharedDecoder):
            produced = decoder(z_in)
            for modality in model._decoder_modalities[decoder_name]:
                reconstructions[modality] = produced[modality]
        else:
            reconstructions[decoder_name] = decoder(z_in)
    return reconstructions


def _legacyForward(
    model: GlobalVae, inputs: dict[str, torch.Tensor], use_mean: bool = False
) -> dict[str, Any]:
    """The 2.0.0 `GlobalVae.forward` as one pass, with its three keys in their order."""
    latent_params, latent_samples = _legacyEncode(model, inputs, use_mean)
    reconstructions = _legacyDecode(model, latent_samples)
    return {
        "reconstructions": reconstructions,
        "latent_params": latent_params,
        "latent_samples": latent_samples,
    }


# ---------------------------------------------------------------------------------------------
# Cases: every row of the matrix, and a model with the real shared modules.
# ---------------------------------------------------------------------------------------------


def _matrixCase(row_code: str, assembler: str | None) -> tuple[GlobalVae, dict[str, torch.Tensor]]:
    """A model of one row of the matrix and the encoder inputs of a fixed batch."""
    torch.manual_seed(0)
    row = MATRIX[row_code]
    model = row.build(assembler)
    batch = {
        "signal": torch.randn(BATCH_SIZE, SIGNAL_LENGTH),
        "image": torch.randn(BATCH_SIZE, *IMAGE_SHAPE),
    }
    return model, model.selectEncoderInputs(batch)


def _sharedTransformerCase() -> tuple[GlobalVae, dict[str, torch.Tensor]]:
    """The shipped signal plus image model with one shared transformer encoder and decoder."""
    torch.manual_seed(0)
    return transformer_configs._model("signal_image_shared_transformer"), (
        transformer_configs._batch("signal", "image")
    )


CASES: list[Any] = [
    *MATRIX_CASES,
    pytest.param("shared_transformer", None, id="shared_transformer"),
]


def _build(row_code: str, assembler: str | None) -> tuple[GlobalVae, dict[str, torch.Tensor]]:
    if row_code == "shared_transformer":
        return _sharedTransformerCase()
    return _matrixCase(row_code, assembler)


def _subsets(names: list[str]) -> Iterator[tuple[str, ...]]:
    """Every non-empty subset of `names`, smallest first."""
    for size in range(1, len(names) + 1):
        yield from itertools.combinations(names, size)


def _assertSameTensors(actual: Any, expected: Any, where: str = "") -> None:
    """Equal structure, equal key order and `torch.equal` on every tensor, recursively."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), where
        assert torch.equal(actual, expected), where
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), where
        assert list(actual) == list(expected), f"{where}: {list(actual)} != {list(expected)}"
        for key in expected:
            _assertSameTensors(actual[key], expected[key], f"{where}/{key}")
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected), where
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assertSameTensors(a, e, f"{where}[{index}]")
    else:  # pragma: no cover - the outputs hold tensors, tuples and dicts only
        raise AssertionError(f"{where}: unexpected type {type(expected).__name__}")


@pytest.fixture(autouse=True)
def _restoreRandomState() -> Iterator[None]:
    """Leave the global random state as the test found it: most tests here seed it.

    Without this, the seeds set below would change the random initialization that the tests run
    after this file (in the same process) draw their model weights from.
    """
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


def _countCalls(model: GlobalVae, *kinds: str) -> Callable[[], dict[str, int]]:
    """Count the calls of every module of `kinds` (`encoders`, `decoders`, `assemblers`, ...).

    Returns:
        A function that gives the calls so far, by `kind/name`.
    """
    calls: dict[str, int] = {}
    for kind in kinds:
        modules = cast(nn.ModuleDict, getattr(model, kind))
        for name, module in modules.items():
            key = f"{kind}/{name}"
            calls[key] = 0

            def hook(_: nn.Module, __: Any, ___: Any, key: str = key) -> None:
                calls[key] += 1

            module.register_forward_hook(hook)
    return lambda: dict(calls)


# ---------------------------------------------------------------------------------------------
# Bit for bit with 2.0.0.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("row_code", "assembler"), CASES)
class TestAgainstTheLegacyForward:
    """`forward`, `encode` and `decode` give the tensors of the 2.0.0 loop, to the last bit."""

    @pytest.mark.parametrize("use_mean", [False, True])
    def test_forward_equals_the_legacy_forward(
        self, row_code: str, assembler: str | None, use_mean: bool
    ) -> None:
        model, inputs = _build(row_code, assembler)
        torch.manual_seed(1)
        expected = _legacyForward(model, inputs, use_mean)
        torch.manual_seed(1)
        actual = model(inputs, use_mean=use_mean)
        _assertSameTensors(actual, expected)

    def test_forward_equals_the_legacy_forward_on_every_subset_of_the_inputs(
        self, row_code: str, assembler: str | None
    ) -> None:
        model, inputs = _build(row_code, assembler)
        for names in _subsets(list(inputs)):
            subset = {name: inputs[name] for name in names}
            torch.manual_seed(2)
            expected = _legacyForward(model, subset)
            torch.manual_seed(2)
            _assertSameTensors(model(subset), expected, where=str(names))

    @pytest.mark.parametrize("use_mean", [False, True])
    def test_encode_then_decode_equals_forward(
        self, row_code: str, assembler: str | None, use_mean: bool
    ) -> None:
        model, inputs = _build(row_code, assembler)
        torch.manual_seed(3)
        expected = model(inputs, use_mean=use_mean)
        torch.manual_seed(3)
        encoded = model.encode(inputs, use_mean=use_mean)
        reconstructions = model.decode(encoded["latent_samples"])
        _assertSameTensors(
            {
                "reconstructions": reconstructions,
                "latent_params": encoded["latent_params"],
                "latent_samples": encoded["latent_samples"],
            },
            expected,
        )

    @pytest.mark.parametrize("use_mean", [False, True])
    def test_encode_equals_the_first_half_of_the_legacy_forward(
        self, row_code: str, assembler: str | None, use_mean: bool
    ) -> None:
        model, inputs = _build(row_code, assembler)
        torch.manual_seed(4)
        params, samples = _legacyEncode(model, inputs, use_mean)
        torch.manual_seed(4)
        encoded = model.encode(inputs, use_mean=use_mean)
        assert list(encoded) == ["latent_params", "latent_samples"]
        _assertSameTensors(encoded["latent_params"], params)
        _assertSameTensors(encoded["latent_samples"], samples)

    def test_decode_equals_the_second_half_on_every_subset_of_the_latent_spaces(
        self, row_code: str, assembler: str | None
    ) -> None:
        """The rules for a missing latent space are the ones `forward` had, subset by subset."""
        model, inputs = _build(row_code, assembler)
        samples = model.encode(inputs, use_mean=True)["latent_samples"]
        for names in _subsets(list(samples)):
            subset = {name: samples[name] for name in names}
            _assertSameTensors(model.decode(subset), _legacyDecode(model, subset), str(names))

    def test_encode_runs_no_decoder_and_decode_runs_no_encoder(
        self, row_code: str, assembler: str | None
    ) -> None:
        model, inputs = _build(row_code, assembler)
        calls = _countCalls(model, "encoders", "decoders", "assemblers")
        encoded = model.encode(inputs)
        after_encode = calls()
        encoders = {key for key in after_encode if key.startswith("encoders/")}
        assert any(after_encode[key] > 0 for key in encoders)
        assert all(after_encode[key] == 0 for key in set(after_encode) - encoders)
        model.decode(encoded["latent_samples"])
        after_decode = calls()
        assert {key: after_decode[key] for key in encoders} == {
            key: after_encode[key] for key in encoders
        }
        assert any(after_decode[key] > 0 for key in after_decode if key.startswith("decoders/"))


# ---------------------------------------------------------------------------------------------
# Cross decoding.
# ---------------------------------------------------------------------------------------------


def _independentModel(signal_dim: int, image_dim: int) -> GlobalVae:
    """A signal and an image, each with its own latent space and its own decoder, no assembler."""
    graph = RoutingGraph(
        latent_specs={
            "z_signal": LatentSpace("z_signal", signal_dim),
            "z_image": LatentSpace("z_image", image_dim),
        },
        encoder_to_latents={"signal": ["z_signal"], "image": ["z_image"]},
        latent_to_decoders={"z_signal": ["signal"], "z_image": ["image"]},
    )
    return GlobalVae(
        encoder_configs={"signal": SIGNAL_ENCODER, "image": IMAGE_ENCODER},
        decoder_configs={"signal": SIGNAL_DECODER, "image": IMAGE_DECODER},
        routing_graph=graph,
        encoder_kwargs={
            "signal": _encoderKwargs(signal_dim),
            "image": _encoderKwargs(image_dim),
        },
        decoder_kwargs={
            "signal": _decoderKwargs(SIGNAL_DECODER, signal_dim),
            "image": _decoderKwargs(IMAGE_DECODER, image_dim),
        },
    )


def _independentInputs() -> dict[str, torch.Tensor]:
    return {
        "signal": torch.randn(BATCH_SIZE, SIGNAL_LENGTH),
        "image": torch.randn(BATCH_SIZE, *IMAGE_SHAPE),
    }


class TestCrossDecoding:
    """The sample of one latent space, passed under the name of another."""

    def test_the_sample_of_one_space_is_decoded_by_the_decoder_of_another(self) -> None:
        torch.manual_seed(0)
        model = _independentModel(8, 8)
        samples = model.encode(_independentInputs(), use_mean=True)["latent_samples"]
        crossed = model.decode({"z_image": samples["z_signal"]})
        assert list(crossed) == ["image"]
        assert crossed["image"].shape == (BATCH_SIZE, *IMAGE_SHAPE)
        assert torch.equal(crossed["image"], model.decoders["image"](samples["z_signal"]))
        straight = model.decode({"z_image": samples["z_image"]})
        assert not torch.equal(crossed["image"], straight["image"])

    def test_both_directions_at_once(self) -> None:
        torch.manual_seed(0)
        model = _independentModel(8, 8)
        samples = model.encode(_independentInputs(), use_mean=True)["latent_samples"]
        swapped = model.decode({"z_signal": samples["z_image"], "z_image": samples["z_signal"]})
        assert list(swapped) == ["signal", "image"]
        assert torch.equal(swapped["signal"], model.decoders["signal"](samples["z_image"]))
        assert torch.equal(swapped["image"], model.decoders["image"](samples["z_signal"]))

    def test_different_widths_are_refused_with_the_names_and_the_widths(self) -> None:
        torch.manual_seed(0)
        model = _independentModel(6, 10)
        samples = model.encode(_independentInputs(), use_mean=True)["latent_samples"]
        with pytest.raises(ValueError, match=r"z_image.*10.*6") as excinfo:
            model.decode({"z_image": samples["z_signal"]})
        assert "equal latent widths" in str(excinfo.value)
        assert "decode()" in str(excinfo.value)

    def test_a_sample_of_the_right_width_is_accepted_whatever_its_origin(self) -> None:
        model = _independentModel(6, 10)
        reconstructions = model.decode({"z_image": torch.randn(BATCH_SIZE, 10)})
        assert reconstructions["image"].shape == (BATCH_SIZE, *IMAGE_SHAPE)


# ---------------------------------------------------------------------------------------------
# Errors and signatures.
# ---------------------------------------------------------------------------------------------


class TestErrors:
    """Each method names itself in its own errors, and `forward` keeps the messages it had."""

    def setup_method(self) -> None:
        torch.manual_seed(0)
        self.model = _independentModel(8, 8)
        self.inputs = _independentInputs()

    def test_encode_refuses_an_empty_input(self) -> None:
        with pytest.raises(ValueError, match=r"GlobalVae\.encode\(\) requires at least one"):
            self.model.encode({})

    def test_forward_still_says_forward_for_an_empty_input(self) -> None:
        with pytest.raises(ValueError, match=r"GlobalVae\.forward\(\) requires at least one"):
            self.model({})

    def test_encode_and_forward_refuse_an_unknown_input_key_in_their_own_name(self) -> None:
        bad = {**self.inputs, "label": torch.zeros(BATCH_SIZE)}
        with pytest.raises(KeyError, match=r"GlobalVae\.encode\(\) received input key\(s\)"):
            self.model.encode(bad)
        with pytest.raises(KeyError, match=r"GlobalVae\.forward\(\) received input key\(s\)"):
            self.model(bad)

    def test_decode_refuses_an_empty_mapping(self) -> None:
        with pytest.raises(ValueError, match=r"GlobalVae\.decode\(\) requires at least one"):
            self.model.decode({})

    def test_decode_refuses_a_name_that_is_not_a_latent_space_and_lists_the_names(self) -> None:
        samples = self.model.encode(self.inputs)["latent_samples"]
        with pytest.raises(KeyError, match=r"'z_other'.*\['z_image', 'z_signal'\]"):
            self.model.decode({"z_other": samples["z_signal"]})

    def test_decode_refuses_a_sample_that_is_not_a_tensor(self) -> None:
        with pytest.raises(TypeError, match=r"z_signal"):
            self.model.decode({"z_signal": [0.0] * 8})  # type: ignore[dict-item]

    def test_the_forward_signature_and_return_keys_are_unchanged(self) -> None:
        outputs = self.model(self.inputs, use_mean=True)
        assert list(outputs) == ["reconstructions", "latent_params", "latent_samples"]
