"""Unit tests for the graph decoder (roadmap P3-8, ADR 0043)."""

import pytest
import torch
import torch.nn.functional as functional

import global_vae.decoders  # noqa: F401  (registers the built-in decoders)
from global_vae.decoders.base import AbstractDecoder
from global_vae.decoders.graph_cnn_decoder import GraphCnnDecoder
from global_vae.decoders.registry import getDecoderClass, listRegisteredDecoders
from global_vae.decoders.two_d_cnn_decoder import TwoDCnnDecoder
from global_vae.utils.autograd import backward

LATENT = 4
NODES = 16
SMALL = {"hidden_channels": (16, 8), "seed_shape": (4, 4), "kernel_sizes": 3, "paddings": 1}


def _decoder(**kwargs: object) -> GraphCnnDecoder:
    settings = {"latent_dim": LATENT, "output_shape": (NODES, NODES), **SMALL, **kwargs}
    return GraphCnnDecoder(**settings).eval()  # type: ignore[arg-type]


def _z(rows: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(rows, LATENT, generator=generator)


class TestRegistration:
    def test_it_is_registered_under_its_name(self) -> None:
        assert getDecoderClass("graph_cnn_decoder_v1") is GraphCnnDecoder
        assert "graph_cnn_decoder_v1" in listRegisteredDecoders()

    def test_it_is_a_2d_cnn_decoder_and_a_decoder(self) -> None:
        assert issubclass(GraphCnnDecoder, TwoDCnnDecoder)
        assert issubclass(GraphCnnDecoder, AbstractDecoder)

    def test_the_parent_class_stays_registered_under_its_own_name(self) -> None:
        assert getDecoderClass("2d_cnn_decoder_v1") is TwoDCnnDecoder

    def test_the_modality_name_defaults_to_graph_and_can_be_changed(self) -> None:
        assert _decoder().modality_name == "graph"
        assert _decoder(modality_name="molecule").modality_name == "molecule"


class TestOutput:
    def test_it_gives_one_square_matrix_per_graph(self) -> None:
        assert _decoder()(_z()).shape == (3, NODES, NODES)

    def test_every_value_is_a_probability(self) -> None:
        probabilities = _decoder()(_z())
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()

    def test_the_graph_is_undirected_exactly(self) -> None:
        probabilities = _decoder()(_z())
        assert torch.equal(probabilities, probabilities.transpose(1, 2))

    def test_a_node_has_no_edge_to_itself_exactly(self) -> None:
        probabilities = _decoder()(_z())
        assert (probabilities.diagonal(dim1=1, dim2=2) == 0).all()

    def test_it_is_the_parent_output_averaged_with_its_transpose_through_a_sigmoid(self) -> None:
        decoder = _decoder()
        z = _z()
        with torch.no_grad():
            raw = TwoDCnnDecoder.forward(decoder, z)
            expected = torch.sigmoid((raw + raw.transpose(1, 2)) / 2)
            expected = expected * (1 - torch.eye(NODES))
            assert torch.allclose(decoder(z), expected)

    def test_a_directed_graph_is_not_forced_to_be_symmetric(self) -> None:
        probabilities = _decoder(symmetric=False)(_z())
        assert not torch.allclose(probabilities, probabilities.transpose(1, 2))
        assert (probabilities.diagonal(dim1=1, dim2=2) == 0).all()

    def test_self_loops_can_be_allowed(self) -> None:
        probabilities = _decoder(zero_diagonal=False)(_z())
        assert (probabilities.diagonal(dim1=1, dim2=2) > 0).all()
        assert torch.equal(probabilities, probabilities.transpose(1, 2))

    def test_several_kinds_of_edge_are_one_matrix_each(self) -> None:
        probabilities = _decoder(out_channels=3)(_z())
        assert probabilities.shape == (3, 3, NODES, NODES)
        assert torch.equal(probabilities, probabilities.transpose(-1, -2))
        assert (probabilities.diagonal(dim1=-2, dim2=-1) == 0).all()

    def test_it_is_deterministic_in_eval_mode(self) -> None:
        decoder = _decoder()
        z = _z()
        assert torch.equal(decoder(z), decoder(z))


class TestArguments:
    def test_a_matrix_that_is_not_square_is_refused(self) -> None:
        with pytest.raises(ValueError, match="square"):
            _decoder(output_shape=(16, 8))

    def test_a_configuration_that_does_not_land_on_the_size_is_refused_by_the_parent(self) -> None:
        with pytest.raises(ValueError, match="output_shape"):
            _decoder(output_shape=(20, 20))

    def test_the_arguments_of_the_parent_are_passed_on(self) -> None:
        small = sum(p.numel() for p in _decoder().parameters())
        wide = sum(p.numel() for p in _decoder(hidden_channels=(32, 16)).parameters())
        assert wide > small

    def test_another_size_is_reached_by_changing_the_seed(self) -> None:
        decoder = _decoder(output_shape=(32, 32), seed_shape=(8, 8))
        assert decoder(_z()).shape == (3, 32, 32)

    def test_a_list_for_the_shape_is_accepted_as_a_yaml_file_gives_it(self) -> None:
        decoder = _decoder(output_shape=[NODES, NODES], seed_shape=[4, 4])
        assert decoder(_z()).shape == (3, NODES, NODES)


class TestTraining:
    def test_every_parameter_receives_a_finite_gradient_from_the_bernoulli_loss(self) -> None:
        decoder = GraphCnnDecoder(latent_dim=LATENT, output_shape=(NODES, NODES), **SMALL)  # type: ignore[arg-type]
        target = (torch.rand(3, NODES, NODES) < 0.3).float()
        target = torch.triu(target, diagonal=1)
        target = target + target.transpose(1, 2)
        loss = functional.binary_cross_entropy(decoder(_z()), target)
        backward(loss)
        gradients = {name: p.grad for name, p in decoder.named_parameters()}
        assert [name for name, grad in gradients.items() if grad is None] == []
        assert all(torch.isfinite(grad).all() for grad in gradients.values() if grad is not None)
