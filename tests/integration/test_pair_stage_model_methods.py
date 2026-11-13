"""The two read-only methods of `GlobalVae` that the pair-stage terms ask (roadmap P6-4, ADR 0054).

`modalitiesEncodedInto(latent)` says which batch keys the encoders that feed a latent space ingest,
and `modalitiesDecodedFrom(latents)` says which reconstructions `decode` gives when only those
latent spaces are passed. The second applies the rules of `decode` (ADR 0046) without calling a
decoder, so it is checked against the real `decode` on every subset of the latent spaces of every
row of the configuration matrix and of the shared transformer model: if the rules of `decode` ever
change, this is the test that says the method has to change with them.
"""

import pytest
import torch

from global_vae.models.global_vae import GlobalVae
from tests.integration.test_encode_decode_seams import (
    CASES,
    _build,
    _independentInputs,
    _independentModel,
    _subsets,
)


class TestModalitiesDecodedFrom:
    @pytest.mark.parametrize(("row_code", "assembler"), CASES)
    def test_it_names_what_decode_returns_on_every_subset_of_the_latent_spaces(
        self, row_code: str, assembler: str | None
    ) -> None:
        model, inputs = _build(row_code, assembler)
        model.eval()
        with torch.no_grad():
            samples = model.encode(inputs, use_mean=True)["latent_samples"]
            for subset in _subsets(sorted(samples)):
                decoded = model.decode({name: samples[name] for name in subset})
                assert set(model.modalitiesDecodedFrom(subset)) == set(decoded), subset

    def test_a_concat_assembler_needs_every_space_and_leaves_a_subset_out(self) -> None:
        model, _ = _build("E1-LN-D1", "concat")
        first = sorted(model.latent_spaces)[0]
        assert model.modalitiesDecodedFrom([first]) == ()
        assert set(model.modalitiesDecodedFrom(sorted(model.latent_spaces))) == {"signal", "image"}

    def test_an_average_assembler_runs_from_one_space(self) -> None:
        model, _ = _build("E1-LN-D1", "average")
        first = sorted(model.latent_spaces)[0]
        assert set(model.modalitiesDecodedFrom([first])) == {"signal", "image"}

    def test_independent_spaces_each_decode_their_own_modality(self) -> None:
        model = _independentModel(8, 8)
        assert model.modalitiesDecodedFrom(["z_signal"]) == ("signal",)
        assert model.modalitiesDecodedFrom(["z_image"]) == ("image",)
        assert set(model.modalitiesDecodedFrom(["z_signal", "z_image"])) == {"signal", "image"}

    def test_the_order_is_the_order_of_the_decoders(self) -> None:
        model = _independentModel(8, 8)
        assert model.modalitiesDecodedFrom(["z_image", "z_signal"]) == ("signal", "image")

    def test_no_names_decode_nothing(self) -> None:
        assert _independentModel(8, 8).modalitiesDecodedFrom([]) == ()

    def test_a_name_that_is_not_a_latent_space_is_a_key_error_that_lists_the_real_ones(
        self,
    ) -> None:
        with pytest.raises(KeyError, match=r"\['z_oops'\] are not latent spaces.*'z_image'"):
            _independentModel(8, 8).modalitiesDecodedFrom(["z_signal", "z_oops"])

    def test_it_runs_nothing(self) -> None:
        model = _independentModel(8, 8)
        calls: list[str] = []
        for name, decoder in model.decoders.items():
            decoder.register_forward_hook(lambda *_a, _n=name, **_k: calls.append(_n))
        model.modalitiesDecodedFrom(["z_signal", "z_image"])
        assert calls == []


class TestModalitiesEncodedInto:
    @pytest.mark.parametrize(("row_code", "assembler"), CASES)
    def test_the_named_modalities_are_enough_to_encode_the_space(
        self, row_code: str, assembler: str | None
    ) -> None:
        model, inputs = _build(row_code, assembler)
        model.eval()
        with torch.no_grad():
            for latent in sorted(model.latent_spaces):
                modalities = model.modalitiesEncodedInto(latent)
                assert modalities, latent
                assert set(modalities) <= set(inputs)
                encoded = model.encode({name: inputs[name] for name in modalities})
                assert latent in encoded["latent_samples"], latent

    def test_independent_spaces_are_fed_by_their_own_modality(self) -> None:
        model = _independentModel(8, 8)
        assert model.modalitiesEncodedInto("z_signal") == ("signal",)
        assert model.modalitiesEncodedInto("z_image") == ("image",)

    def test_a_fused_space_names_every_feeding_modality_once(self) -> None:
        # The single-latent rows of the matrix fuse signal and image into one space.
        model, _ = _build("E1-L1-D1", None)
        (latent,) = model.latent_spaces
        assert model.modalitiesEncodedInto(latent) == ("signal", "image")

    def test_a_name_that_is_not_a_latent_space_is_a_key_error_that_lists_the_real_ones(
        self,
    ) -> None:
        with pytest.raises(KeyError, match=r"'z_oops' is not a latent space.*'z_signal'"):
            _independentModel(8, 8).modalitiesEncodedInto("z_oops")

    def test_it_reads_the_graph_and_encodes_nothing(self) -> None:
        model: GlobalVae = _independentModel(8, 8)
        calls: list[str] = []
        for name, encoder in model.encoders.items():
            encoder.register_forward_hook(lambda *_a, _n=name, **_k: calls.append(_n))
        model.modalitiesEncodedInto("z_signal")
        assert calls == []
        # The inputs are only needed to show that the model is built and usable.
        assert set(_independentInputs()) == {"signal", "image"}
