"""Unit tests for the shared encoder and decoder interfaces (spec §2.1, ADR 0027).

`AbstractSharedEncoder` and `AbstractSharedDecoder` are siblings of `AbstractEncoder` and
`AbstractDecoder`, registered in the same registries. These tests state what that promises, one
component at a time and without a `GlobalVae`: the interfaces are abstract and complete, a shared
class registers and looks up like any other, the two kinds share one namespace, and a dummy module
honours the contract (any non-empty subset in, a dict out). The wiring around them is tested in
`tests/integration/test_shared_encoder_decoder.py` and `test_configuration_matrix.py`.
"""

from collections.abc import Mapping
from typing import Any

import pytest
import torch

import global_vae.decoders.registry as decoder_registry
import global_vae.encoders.registry as encoder_registry
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders, registerDecoder
from global_vae.decoders.shared_base import AbstractSharedDecoder
from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import getEncoderClass, listRegisteredEncoders, registerEncoder
from global_vae.encoders.shared_base import AbstractSharedEncoder
from global_vae.utils.autograd import backward
from tests.integration._shared_modules import (
    SHARED_DECODER,
    SHARED_ENCODER,
    DummySharedDecoder,
    DummySharedEncoder,
)

BATCH = 5
INPUT_DIMS = {"signal": 6, "image": 12}
OUTPUT_SHAPES = {"signal": (6,), "image": (3, 4)}


class _MinimalSharedEncoder(AbstractSharedEncoder):
    def forward(self, inputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        first = next(iter(inputs.values()))
        return first, first

    @property
    def input_modalities(self) -> tuple[str, ...]:
        return ("a",)

    @property
    def latent_dim(self) -> int:
        return 1


class _MinimalSharedDecoder(AbstractSharedDecoder):
    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"a": z}

    @property
    def output_modalities(self) -> tuple[str, ...]:
        return ("a",)


@pytest.fixture
def encoder_registry_copy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A private copy of the encoder registry, swapped in for the test."""
    copy = dict(encoder_registry._ENCODER_REGISTRY)
    monkeypatch.setattr(encoder_registry, "_ENCODER_REGISTRY", copy)
    return copy


@pytest.fixture
def decoder_registry_copy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A private copy of the decoder registry, swapped in for the test."""
    copy = dict(decoder_registry._DECODER_REGISTRY)
    monkeypatch.setattr(decoder_registry, "_DECODER_REGISTRY", copy)
    return copy


class TestInterfaces:
    """The two abstract classes are complete contracts."""

    def test_the_abstract_shared_encoder_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractSharedEncoder()  # type: ignore[abstract]

    def test_the_abstract_shared_decoder_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractSharedDecoder()  # type: ignore[abstract]

    def test_a_shared_encoder_must_implement_its_three_members(self) -> None:
        """`forward`, `input_modalities` and `latent_dim` are all abstract."""
        assert AbstractSharedEncoder.__abstractmethods__ == {
            "forward",
            "input_modalities",
            "latent_dim",
        }

    def test_a_shared_decoder_must_implement_its_two_members(self) -> None:
        """`forward` and `output_modalities` are abstract."""
        assert AbstractSharedDecoder.__abstractmethods__ == {"forward", "output_modalities"}

    def test_a_minimal_subclass_can_be_instantiated(self) -> None:
        assert _MinimalSharedEncoder().input_modalities == ("a",)
        assert _MinimalSharedDecoder().output_modalities == ("a",)

    def test_the_shared_classes_are_siblings_of_the_one_modality_classes(self) -> None:
        """Different `forward` signatures, so neither kind is a subclass of the other."""
        assert not issubclass(AbstractSharedEncoder, AbstractEncoder)
        assert not issubclass(AbstractEncoder, AbstractSharedEncoder)
        assert not issubclass(AbstractSharedDecoder, AbstractDecoder)
        assert not issubclass(AbstractDecoder, AbstractSharedDecoder)


class TestRegistries:
    """A shared class registers in the encoder or decoder registry like any other."""

    def test_a_shared_encoder_registers_and_is_returned_unchanged(
        self, encoder_registry_copy: dict[str, Any]
    ) -> None:
        registered = registerEncoder("minimal_shared_encoder")(_MinimalSharedEncoder)
        assert registered is _MinimalSharedEncoder
        assert getEncoderClass("minimal_shared_encoder") is _MinimalSharedEncoder
        assert "minimal_shared_encoder" in listRegisteredEncoders()

    def test_a_shared_decoder_registers_and_is_returned_unchanged(
        self, decoder_registry_copy: dict[str, Any]
    ) -> None:
        registered = registerDecoder("minimal_shared_decoder")(_MinimalSharedDecoder)
        assert registered is _MinimalSharedDecoder
        assert getDecoderClass("minimal_shared_decoder") is _MinimalSharedDecoder
        assert "minimal_shared_decoder" in listRegisteredDecoders()

    def test_the_two_kinds_of_encoder_share_one_namespace(
        self, encoder_registry_copy: dict[str, Any]
    ) -> None:
        """A shared encoder cannot take the name of a built-in one-modality encoder."""
        with pytest.raises(ValueError, match="already registered"):
            registerEncoder("1d_cnn_encoder_v1")(_MinimalSharedEncoder)

    def test_the_two_kinds_of_decoder_share_one_namespace(
        self, decoder_registry_copy: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="already registered"):
            registerDecoder("1d_cnn_decoder_v1")(_MinimalSharedDecoder)

    def test_a_duplicate_shared_name_keeps_the_first_class(
        self, encoder_registry_copy: dict[str, Any]
    ) -> None:
        registerEncoder("twice")(_MinimalSharedEncoder)
        with pytest.raises(ValueError, match="already registered"):
            registerEncoder("twice")(DummySharedEncoder)
        assert getEncoderClass("twice") is _MinimalSharedEncoder

    def test_the_dummy_modules_are_registered_under_their_names(self) -> None:
        assert getEncoderClass(SHARED_ENCODER) is DummySharedEncoder
        assert getDecoderClass(SHARED_DECODER) is DummySharedDecoder

    def test_an_unknown_name_still_lists_the_available_ones(self) -> None:
        with pytest.raises(KeyError, match=SHARED_ENCODER):
            getEncoderClass("no_such_encoder")


class TestDummyModulesHonourTheContract:
    """The dummies used across the suite do what the interfaces ask."""

    @staticmethod
    def _inputs() -> dict[str, torch.Tensor]:
        return {"signal": torch.randn(BATCH, 6), "image": torch.randn(BATCH, 3, 4)}

    def test_the_encoder_declares_its_modalities_in_construction_order(self) -> None:
        encoder = DummySharedEncoder(INPUT_DIMS, latent_dim=4)
        assert encoder.input_modalities == ("signal", "image")
        assert encoder.latent_dim == 4

    @pytest.mark.parametrize("present", [("signal", "image"), ("signal",), ("image",)])
    def test_the_encoder_accepts_every_non_empty_subset(self, present: tuple[str, ...]) -> None:
        encoder = DummySharedEncoder(INPUT_DIMS, latent_dim=4)
        inputs = {name: tensor for name, tensor in self._inputs().items() if name in present}
        mu, logvar = encoder(inputs)
        assert mu.shape == logvar.shape == (BATCH, 4)

    def test_the_encoder_rejects_an_empty_dict(self) -> None:
        with pytest.raises(ValueError, match="no modality"):
            DummySharedEncoder(INPUT_DIMS, latent_dim=4)({})

    def test_an_absent_modality_changes_the_posterior(self) -> None:
        """Omitting a modality is not the same as feeding zeros: the pooled stream differs."""
        torch.manual_seed(0)
        encoder = DummySharedEncoder(INPUT_DIMS, latent_dim=4)
        inputs = self._inputs()
        both, _ = encoder(inputs)
        signal_only, _ = encoder({"signal": inputs["signal"]})
        assert not torch.allclose(both, signal_only)

    def test_the_decoder_returns_exactly_its_declared_modalities(self) -> None:
        decoder = DummySharedDecoder(OUTPUT_SHAPES, latent_dim=4)
        outputs = decoder(torch.randn(BATCH, 4))
        assert decoder.output_modalities == ("signal", "image")
        assert set(outputs) == set(decoder.output_modalities)
        assert outputs["signal"].shape == (BATCH, 6)
        assert outputs["image"].shape == (BATCH, 3, 4)

    def test_gradients_reach_both_modules(self) -> None:
        encoder = DummySharedEncoder(INPUT_DIMS, latent_dim=4)
        decoder = DummySharedDecoder(OUTPUT_SHAPES, latent_dim=4)
        mu, _ = encoder(self._inputs())
        loss = sum(tensor.pow(2).mean() for tensor in decoder(mu).values())
        assert isinstance(loss, torch.Tensor)
        backward(loss)
        for module in (encoder, decoder):
            for name, parameter in module.named_parameters():
                if name.endswith("to_logvar.weight") or name.endswith("to_logvar.bias"):
                    continue  # the decoder reads `mu` only in this toy loss
                assert parameter.grad is not None, name
