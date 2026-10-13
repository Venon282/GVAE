"""Unit tests for the learned assemblers `weighted_sum` and `attention` (spec §2.2, roadmap P1-4).

Unlike `concat`, `sum` and `average`, these two have parameters, so beyond exact values they are
checked for what a learned module owes the training loop: every parameter receives a gradient, the
parameters move the output, and a subset of the latent spaces (a modality left out of a pass,
spec §5) uses the parameters of the spaces that are present and leaves the others alone.
"""

from typing import Any

import pytest
import torch
from torch import nn

import global_vae.assemblers  # noqa: F401  (registers the built-in assemblers)
from global_vae.assemblers.attention import AttentionAssembler
from global_vae.assemblers.registry import getAssemblerClass
from global_vae.assemblers.weighted_sum import WeightedSumAssembler
from global_vae.utils.autograd import backward

BATCH_SIZE = 5


def _latent(dim: int, seed: int) -> torch.Tensor:
    """A reproducible `(BATCH_SIZE, dim)` latent whose entries differ from one seed to another."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH_SIZE, dim, generator=generator)


class TestWeightedSumAssembler:
    """`weighted_sum` scales each latent vector by a learned per-space weight and adds them."""

    def test_is_registered_with_the_dimension_rule_of_sum(self) -> None:
        """Same dimension rule as `sum`, and it accepts a subset of its latent spaces."""
        assert getAssemblerClass("weighted_sum") is WeightedSumAssembler
        assert WeightedSumAssembler.requires_matching_dims is True
        assert WeightedSumAssembler([4, 4]).handlesMissingLatents is True

    def test_different_dimensions_are_refused_at_construction(self) -> None:
        """The same check as `sum`, with the same message."""
        with pytest.raises(
            ValueError, match="WeightedSumAssembler requires matching dimensionality"
        ):
            WeightedSumAssembler([4, 3])

    def test_output_dim_is_the_shared_width(self) -> None:
        """The decoder is sized with the common width of the inputs."""
        assembler = WeightedSumAssembler([6, 6, 6])
        assert assembler.output_dim == 6
        assert assembler([_latent(6, i) for i in range(3)]).shape == (BATCH_SIZE, 6)

    def test_one_parameter_per_latent_space_and_nothing_else(self) -> None:
        """The module adds `len(input_dims)` values to the model."""
        assembler = WeightedSumAssembler([4, 4, 4])
        assert [name for name, _ in assembler.named_parameters()] == ["raw_weights"]
        assert assembler.raw_weights.shape == (3,)
        assert sum(parameter.numel() for parameter in assembler.parameters()) == 3

    def test_normalized_weights_start_as_the_average(self) -> None:
        """With `normalize=True` every logit is 0, so the softmax is uniform: `average`."""
        first, second, third = (_latent(4, seed) for seed in range(3))
        out = WeightedSumAssembler([4, 4, 4])([first, second, third])
        assert torch.allclose(out, (first + second + third) / 3, atol=1e-6)

    def test_unnormalized_weights_start_as_the_sum(self) -> None:
        """With `normalize=False` every weight starts at 1, so the assembler starts as `sum`."""
        first, second = _latent(4, 0), _latent(4, 1)
        out = WeightedSumAssembler([4, 4], normalize=False)([first, second])
        assert torch.equal(out, first + second)

    def test_normalized_weights_are_a_softmax_of_the_stored_values(self) -> None:
        """Stored logits `[0, log 3]` give the weights `[0.25, 0.75]`."""
        assembler = WeightedSumAssembler([4, 4])
        with torch.no_grad():
            assembler.raw_weights.copy_(torch.tensor([1.0, 3.0]).log())
        first, second = _latent(4, 0), _latent(4, 1)
        assert torch.allclose(assembler.computeWeights(), torch.tensor([0.25, 0.75]))
        assert torch.allclose(assembler([first, second]), 0.25 * first + 0.75 * second, atol=1e-6)

    def test_unnormalized_weights_are_used_as_they_are(self) -> None:
        """Stored weights `[2, -1]` give `2 * first - second`: no softmax, negative is fine."""
        assembler = WeightedSumAssembler([4, 4], normalize=False)
        with torch.no_grad():
            assembler.raw_weights.copy_(torch.tensor([2.0, -1.0]))
        first, second = _latent(4, 0), _latent(4, 1)
        assert torch.allclose(assembler([first, second]), 2.0 * first - second)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_a_single_latent_space_is_returned_unchanged_in_value(self, normalize: bool) -> None:
        """One space has weight 1 under both parametrisations at initialisation."""
        latent = _latent(4, 0)
        assert torch.allclose(WeightedSumAssembler([4], normalize=normalize)([latent]), latent)

    @pytest.mark.parametrize("normalize", [True, False])
    def test_gradient_reaches_the_weights_and_every_input(self, normalize: bool) -> None:
        """Backward gives the weights and each latent a nonzero gradient."""
        assembler = WeightedSumAssembler([4, 4, 4], normalize=normalize)
        latents = [_latent(4, seed).requires_grad_() for seed in range(3)]
        assembler(latents).pow(2).sum().backward()
        assert assembler.raw_weights.grad is not None
        assert torch.isfinite(assembler.raw_weights.grad).all()
        assert assembler.raw_weights.grad.abs().sum() > 0
        for latent in latents:
            assert latent.grad is not None
            assert latent.grad.abs().sum() > 0

    def test_the_input_gradient_is_the_weight(self) -> None:
        """For `out = sum(w_i * z_i)` and a loss `out.sum()`, `d loss / d z_i` is `w_i`."""
        assembler = WeightedSumAssembler([4, 4], normalize=False)
        with torch.no_grad():
            assembler.raw_weights.copy_(torch.tensor([2.0, -1.0]))
        first, second = _latent(4, 0).requires_grad_(), _latent(4, 1).requires_grad_()
        assembler([first, second]).sum().backward()
        assert first.grad is not None and second.grad is not None
        assert torch.equal(first.grad, torch.full_like(first, 2.0))
        assert torch.equal(second.grad, torch.full_like(second, -1.0))

    def test_an_optimizer_step_moves_the_weights_toward_the_better_latent(self) -> None:
        """Learning works: pushing the output toward `first` raises the weight of `first`."""
        assembler = WeightedSumAssembler([4, 4])
        first, second = _latent(4, 0), _latent(4, 1)
        optimizer = torch.optim.SGD(assembler.parameters(), lr=0.5)
        for _ in range(20):
            optimizer.zero_grad()
            backward(torch.nn.functional.mse_loss(assembler([first, second]), first))
            optimizer.step()
        weights = assembler.computeWeights()
        assert weights[0] > 0.5 > weights[1]
        assert torch.isclose(weights.sum(), torch.tensor(1.0))

    def test_normalized_weights_are_renormalized_over_a_subset(self) -> None:
        """Missing a space, the softmax runs over the spaces present and still sums to 1."""
        assembler = WeightedSumAssembler([4, 4, 4])
        with torch.no_grad():
            assembler.raw_weights.copy_(torch.tensor([0.0, 1.0, 2.0]))
        weights = assembler.computeWeights([0, 2])
        assert torch.allclose(weights, torch.softmax(torch.tensor([0.0, 2.0]), dim=0))
        assert torch.isclose(weights.sum(), torch.tensor(1.0))
        first, third = _latent(4, 0), _latent(4, 2)
        out = assembler([first, third], present=[0, 2])
        assert torch.allclose(out, weights[0] * first + weights[1] * third, atol=1e-6)

    def test_unnormalized_weights_are_not_renormalized_over_a_subset(self) -> None:
        """Without softmax, a subset is the weighted sum of the spaces present."""
        assembler = WeightedSumAssembler([4, 4, 4], normalize=False)
        with torch.no_grad():
            assembler.raw_weights.copy_(torch.tensor([2.0, 3.0, 5.0]))
        first, third = _latent(4, 0), _latent(4, 2)
        out = assembler([first, third], present=[0, 2])
        assert torch.allclose(out, 2.0 * first + 5.0 * third)

    def test_a_subset_leaves_the_weight_of_an_absent_space_without_gradient(self) -> None:
        """Only the weights of the spaces present are trained by that pass."""
        assembler = WeightedSumAssembler([4, 4, 4])
        assembler([_latent(4, 0), _latent(4, 2)], present=[0, 2]).pow(2).sum().backward()
        grad = assembler.raw_weights.grad
        assert grad is not None
        assert grad[1] == 0
        assert grad[0] != 0 and grad[2] != 0

    def test_the_present_order_follows_the_latents_given(self) -> None:
        """`present=[2, 0]` pairs the first tensor with the weight of space 2."""
        assembler = WeightedSumAssembler([4, 4, 4], normalize=False)
        with torch.no_grad():
            assembler.raw_weights.copy_(torch.tensor([2.0, 3.0, 5.0]))
        a, b = _latent(4, 0), _latent(4, 1)
        assert torch.allclose(assembler([a, b], present=[2, 0]), 5.0 * a + 2.0 * b)

    def test_a_latent_of_the_wrong_width_names_both_widths(self) -> None:
        """A width that does not match `input_dims` fails before the weights are applied."""
        with pytest.raises(ValueError, match=r"expects latent vector 1 of shape \(batch, 4\)"):
            WeightedSumAssembler([4, 4])([_latent(4, 0), _latent(3, 1)])

    def test_the_weights_survive_a_state_dict_round_trip(self) -> None:
        """The learned weights are in the state dict, so checkpoints keep them."""
        trained = WeightedSumAssembler([4, 4])
        with torch.no_grad():
            trained.raw_weights.copy_(torch.tensor([0.3, -0.7]))
        fresh = WeightedSumAssembler([4, 4])
        fresh.load_state_dict(trained.state_dict())
        assert torch.equal(fresh.raw_weights, trained.raw_weights)


class TestAttentionAssembler:
    """`attention` pools latent vectors of any widths with a learned cross-attention query."""

    def test_is_registered_without_a_dimension_rule(self) -> None:
        """No fixed-dimensionality requirement, and it accepts a subset of its latent spaces."""
        assert getAssemblerClass("attention") is AttentionAssembler
        assert AttentionAssembler.requires_matching_dims is False
        assert AttentionAssembler([4, 9]).handlesMissingLatents is True

    def test_accepts_latent_spaces_of_different_widths(self) -> None:
        """Three spaces of three widths give one `(batch, output_dim)` tensor."""
        assembler = AttentionAssembler([3, 8, 5], output_dim=12, num_heads=3)
        out = assembler([_latent(3, 0), _latent(8, 1), _latent(5, 2)])
        assert out.shape == (BATCH_SIZE, 12)
        assert assembler.output_dim == 12

    def test_output_dim_defaults_to_the_widest_input(self) -> None:
        """Without `output_dim`, the output is as wide as the widest latent space."""
        assembler = AttentionAssembler([3, 8, 5])
        assert assembler.output_dim == 8
        assert assembler([_latent(3, 0), _latent(8, 1), _latent(5, 2)]).shape == (BATCH_SIZE, 8)

    def test_output_dim_may_be_smaller_than_every_input(self) -> None:
        """The width is a free choice, not tied to the inputs."""
        assembler = AttentionAssembler([16, 16], output_dim=4)
        assert assembler([_latent(16, 0), _latent(16, 1)]).shape == (BATCH_SIZE, 4)

    def test_default_is_one_head_which_divides_any_width(self) -> None:
        """No kwarg besides `input_dims` is ever needed, whatever the widths are."""
        assert AttentionAssembler([7, 11]).num_heads == 1
        assert AttentionAssembler([7, 11]).output_dim == 11

    def test_several_heads(self) -> None:
        """`num_heads` reaches the attention block."""
        assembler = AttentionAssembler([6, 6], output_dim=8, num_heads=4)
        assert assembler.num_heads == 4
        assert assembler.attention.num_heads == 4
        assert assembler([_latent(6, 0), _latent(6, 1)]).shape == (BATCH_SIZE, 8)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {"output_dim": 10, "num_heads": 4},
                r"`output_dim` \(10\) must be a multiple of `num_heads` \(4\)",
            ),
            ({"num_heads": 3}, r"`output_dim` \(8\) must be a multiple of `num_heads` \(3\)"),
            ({"output_dim": 0}, "`output_dim` must be a positive integer"),
            ({"output_dim": -8}, "`output_dim` must be a positive integer"),
            ({"output_dim": 8.0}, "`output_dim` must be a positive integer"),
            ({"output_dim": True}, "`output_dim` must be a positive integer"),
            ({"num_heads": 0}, "`num_heads` must be a positive integer"),
            ({"num_heads": 1.5}, "`num_heads` must be a positive integer"),
        ],
    )
    def test_invalid_arguments_are_refused_with_a_clear_message(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        """A width that heads cannot divide, or a non-positive value, fails at construction."""
        with pytest.raises(ValueError, match=message):
            AttentionAssembler([4, 8], **kwargs)

    def test_parameters_are_one_projection_per_space_a_query_and_the_attention_block(self) -> None:
        """The structure the class docstring describes."""
        assembler = AttentionAssembler([3, 8, 5], output_dim=12, num_heads=2)
        assert len(assembler.projections) == 3
        for projection, width in zip(assembler.projections, [3, 8, 5], strict=True):
            assert isinstance(projection, nn.Linear)
            assert (projection.in_features, projection.out_features) == (width, 12)
        assert assembler.query.shape == (1, 1, 12)
        top_level = {name.split(".")[0] for name, _ in assembler.named_parameters()}
        assert top_level == {"projections", "query", "attention"}

    def test_every_parameter_receives_a_finite_nonzero_gradient(self) -> None:
        """With every space present, nothing in the module is left untrained."""
        torch.manual_seed(0)
        assembler = AttentionAssembler([3, 8, 5], output_dim=12, num_heads=2)
        latents = [_latent(width, seed).requires_grad_() for seed, width in enumerate([3, 8, 5])]
        assembler(latents).pow(2).sum().backward()
        for name, parameter in assembler.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        for latent in latents:
            assert latent.grad is not None
            assert latent.grad.abs().sum() > 0

    def test_the_output_depends_on_which_space_a_latent_comes_from(self) -> None:
        """Two spaces of the same width are told apart by their own projections."""
        torch.manual_seed(0)
        assembler = AttentionAssembler([4, 4], output_dim=8)
        first, second = _latent(4, 0), _latent(4, 1)
        assert not torch.allclose(assembler([first, second]), assembler([second, first]))

    def test_rows_of_the_batch_are_independent(self) -> None:
        """Attention runs over the latent spaces of one sample, never across samples."""
        torch.manual_seed(0)
        assembler = AttentionAssembler([3, 8], output_dim=6, num_heads=2).eval()
        first, second = _latent(3, 0), _latent(8, 1)
        full = assembler([first, second])
        for row in range(BATCH_SIZE):
            alone = assembler([first[row : row + 1], second[row : row + 1]])
            assert torch.allclose(full[row : row + 1], alone, atol=1e-6)

    def test_pooling_is_a_weighted_average_of_the_tokens(self) -> None:
        """Two spaces that give the very same token pool to that token, whatever the query is.

        The projections of both spaces are made identical, so a latent fed to both yields two
        equal tokens. Softmax weights sum to 1, so the pooled value equals the one-token result.
        """
        torch.manual_seed(0)
        assembler = AttentionAssembler([4, 4], output_dim=8, num_heads=2).eval()
        assembler.projections[1].load_state_dict(assembler.projections[0].state_dict())
        latent = _latent(4, 0)
        both = assembler([latent, latent])
        alone = assembler([latent], present=[0])
        assert torch.allclose(both, alone, atol=1e-6)

    def test_a_subset_runs_on_the_tokens_present_with_the_right_projections(self) -> None:
        """`present=[0, 2]` pairs each tensor with its own space's projection, same output width."""
        torch.manual_seed(0)
        assembler = AttentionAssembler([3, 8, 5], output_dim=12, num_heads=2).eval()
        first, third = _latent(3, 0), _latent(5, 2)
        subset = assembler([first, third], present=[0, 2])
        assert subset.shape == (BATCH_SIZE, 12)
        assert torch.isfinite(subset).all()
        full = assembler([first, _latent(8, 1), third])
        assert not torch.allclose(subset, full)

    def test_a_subset_trains_only_the_projections_of_the_spaces_present(self) -> None:
        """The absent space's projection gets no gradient in that pass; the shared parts do."""
        torch.manual_seed(0)
        assembler = AttentionAssembler([3, 8, 5], output_dim=12, num_heads=2)
        assembler([_latent(3, 0), _latent(5, 2)], present=[0, 2]).pow(2).sum().backward()
        for parameter in assembler.projections[1].parameters():
            assert parameter.grad is None
        for index in (0, 2):
            for parameter in assembler.projections[index].parameters():
                assert parameter.grad is not None
        assert assembler.query.grad is not None
        assert assembler.query.grad.abs().sum() > 0

    @pytest.mark.parametrize("present", [[0], [1], [2]])
    def test_a_single_remaining_space_still_gives_the_full_output_width(
        self, present: list[int]
    ) -> None:
        """A decoder sized for `output_dim` must get `output_dim` features even from one space."""
        assembler = AttentionAssembler([3, 8, 5], output_dim=12, num_heads=2)
        widths = [3, 8, 5]
        out = assembler([_latent(widths[present[0]], 0)], present=present)
        assert out.shape == (BATCH_SIZE, 12)

    def test_a_latent_of_the_wrong_width_names_both_widths(self) -> None:
        """A width that does not match `input_dims` fails before any projection runs."""
        with pytest.raises(ValueError, match=r"expects latent vector 1 of shape \(batch, 8\)"):
            AttentionAssembler([3, 8])([_latent(3, 0), _latent(5, 1)])

    def test_the_parameters_survive_a_state_dict_round_trip(self) -> None:
        """A checkpoint restores the same function."""
        torch.manual_seed(0)
        trained = AttentionAssembler([3, 8], output_dim=6, num_heads=2).eval()
        torch.manual_seed(1)
        fresh = AttentionAssembler([3, 8], output_dim=6, num_heads=2).eval()
        latents = [_latent(3, 0), _latent(8, 1)]
        assert not torch.allclose(trained(latents), fresh(latents))
        fresh.load_state_dict(trained.state_dict())
        assert torch.allclose(trained(latents), fresh(latents))
