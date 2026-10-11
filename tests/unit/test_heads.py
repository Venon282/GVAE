"""Unit tests for latent heads (spec §2.2, §10, roadmap P1-1).

Covers the registry (built-ins, error paths), the abstract base (width validation), and the
two built-in heads: `identity` is a true no-op, `linear` projects `mu` and `logvar` through
separate learned layers with a documented initialisation. The behaviour every registry shares
(duplicate and unknown names, sorted listing) is checked for heads, with the other nine, in
`test_registry_contract.py`.
"""

import math

import pytest
import torch

from global_vae.heads.base import AbstractLatentHead
from global_vae.heads.identity import IdentityHead
from global_vae.heads.linear import LinearHead
from global_vae.heads.registry import getHeadClass, listRegisteredHeads, registerHead

_BUILT_IN_NAMES = ["identity", "linear"]


def _params(batch: int, dim: int, requires_grad: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Make a random `(mu, logvar)` pair."""
    mu = torch.randn(batch, dim, requires_grad=requires_grad)
    logvar = torch.randn(batch, dim, requires_grad=requires_grad)
    return mu, logvar


class TestRegistry:
    def test_built_ins_are_registered_by_default(self) -> None:
        assert listRegisteredHeads() == _BUILT_IN_NAMES

    def test_names_resolve_to_their_classes(self) -> None:
        assert getHeadClass("identity") is IdentityHead
        assert getHeadClass("linear") is LinearHead

    def test_unknown_name_lists_the_built_ins(self) -> None:
        with pytest.raises(KeyError, match="identity, linear"):
            getHeadClass("does_not_exist")

    def test_registering_through_the_decorator_makes_a_head_resolvable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import global_vae.heads.registry as registry_module

        monkeypatch.setattr(registry_module, "_HEAD_REGISTRY", dict(registry_module._HEAD_REGISTRY))

        @registerHead("scale_by_two")
        class _ScaleByTwo(AbstractLatentHead):
            def forward(
                self, mu: torch.Tensor, logvar: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return 2 * mu, 2 * logvar

        head = getHeadClass("scale_by_two")(in_dim=3, out_dim=3)
        out_mu, out_logvar = head(torch.ones(2, 3), torch.ones(2, 3))
        assert torch.equal(out_mu, torch.full((2, 3), 2.0))
        assert torch.equal(out_logvar, torch.full((2, 3), 2.0))

    @pytest.mark.parametrize("name", _BUILT_IN_NAMES)
    def test_every_built_in_is_buildable_from_in_dim_and_out_dim_keywords(self, name: str) -> None:
        """The model will build heads as `getHeadClass(name)(in_dim=..., out_dim=...)`."""
        head = getHeadClass(name)(in_dim=6, out_dim=6)
        assert isinstance(head, AbstractLatentHead)
        assert (head.in_dim, head.out_dim) == (6, 6)


class TestAbstractBase:
    def test_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            AbstractLatentHead(4, 4)  # type: ignore[abstract]

    def test_forward_is_abstract(self) -> None:
        class _NoForward(AbstractLatentHead):
            pass

        with pytest.raises(TypeError):
            _NoForward(4, 4)  # type: ignore[abstract]

    def test_widths_are_exposed_as_read_only_properties(self) -> None:
        head = LinearHead(in_dim=5, out_dim=3)
        assert head.in_dim == 5
        assert head.out_dim == 3
        with pytest.raises(AttributeError):
            head.in_dim = 7  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4"])
    @pytest.mark.parametrize("head_class", [IdentityHead, LinearHead])
    def test_non_positive_or_non_integer_in_dim_raises(
        self, head_class: type[AbstractLatentHead], bad: object
    ) -> None:
        with pytest.raises(ValueError, match="in_dim"):
            head_class(bad, 4)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -3, 1.0, False, None])
    @pytest.mark.parametrize("head_class", [IdentityHead, LinearHead])
    def test_non_positive_or_non_integer_out_dim_raises(
        self, head_class: type[AbstractLatentHead], bad: object
    ) -> None:
        with pytest.raises(ValueError, match="out_dim"):
            head_class(4, bad)  # type: ignore[arg-type]


class TestIdentityHead:
    def test_requires_equal_widths(self) -> None:
        with pytest.raises(ValueError, match="in_dim == out_dim"):
            IdentityHead(in_dim=4, out_dim=8)

    def test_error_message_names_both_widths_and_the_alternative(self) -> None:
        with pytest.raises(ValueError, match=r"in_dim=4.*out_dim=8.*'linear'"):
            IdentityHead(in_dim=4, out_dim=8)

    def test_is_a_no_op_returning_the_same_tensor_objects(self) -> None:
        head = IdentityHead(in_dim=6, out_dim=6)
        mu, logvar = _params(3, 6)
        out_mu, out_logvar = head(mu, logvar)
        assert out_mu is mu
        assert out_logvar is logvar

    def test_does_not_modify_its_inputs(self) -> None:
        head = IdentityHead(in_dim=6, out_dim=6)
        mu, logvar = _params(3, 6)
        mu_before, logvar_before = mu.clone(), logvar.clone()
        head(mu, logvar)
        assert torch.equal(mu, mu_before)
        assert torch.equal(logvar, logvar_before)

    def test_has_no_parameters_and_no_buffers(self) -> None:
        head = IdentityHead(in_dim=6, out_dim=6)
        assert list(head.parameters()) == []
        assert list(head.buffers()) == []
        assert head.state_dict() == {}

    def test_gradient_passes_through_unchanged(self) -> None:
        head = IdentityHead(in_dim=4, out_dim=4)
        mu, logvar = _params(3, 4, requires_grad=True)
        out_mu, out_logvar = head(mu, logvar)
        (out_mu.sum() + 2 * out_logvar.sum()).backward()
        assert mu.grad is not None and logvar.grad is not None
        assert torch.equal(mu.grad, torch.ones_like(mu))
        assert torch.equal(logvar.grad, torch.full_like(logvar, 2.0))

    def test_is_the_same_in_eval_and_train_mode(self) -> None:
        head = IdentityHead(in_dim=4, out_dim=4)
        mu, logvar = _params(3, 4)
        head.eval()
        assert head(mu, logvar)[0] is mu
        head.train()
        assert head(mu, logvar)[0] is mu

    @pytest.mark.parametrize(
        ("mu_shape", "logvar_shape"),
        [
            ((3, 5), (3, 5)),
            ((3, 4), (3, 5)),
            ((3, 4), (2, 4)),
            ((4,), (4,)),
            ((2, 3, 4), (2, 3, 4)),
        ],
    )
    def test_wrong_shapes_raise_value_error(
        self, mu_shape: tuple[int, ...], logvar_shape: tuple[int, ...]
    ) -> None:
        head = IdentityHead(in_dim=4, out_dim=4)
        with pytest.raises(ValueError, match="IdentityHead expects"):
            head(torch.zeros(mu_shape), torch.zeros(logvar_shape))


class TestLinearHeadShapes:
    @pytest.mark.parametrize(("in_dim", "out_dim"), [(8, 3), (3, 8), (6, 6), (1, 1), (1, 5)])
    @pytest.mark.parametrize("batch", [1, 7])
    def test_output_shape_is_batch_by_out_dim(self, in_dim: int, out_dim: int, batch: int) -> None:
        head = LinearHead(in_dim=in_dim, out_dim=out_dim)
        out_mu, out_logvar = head(*_params(batch, in_dim))
        assert out_mu.shape == (batch, out_dim)
        assert out_logvar.shape == (batch, out_dim)

    def test_output_is_finite(self) -> None:
        head = LinearHead(in_dim=8, out_dim=3)
        out_mu, out_logvar = head(*_params(5, 8))
        assert torch.isfinite(out_mu).all()
        assert torch.isfinite(out_logvar).all()

    def test_keeps_the_input_dtype(self) -> None:
        head = LinearHead(in_dim=4, out_dim=2).double()
        mu, logvar = _params(3, 4)
        out_mu, out_logvar = head(mu.double(), logvar.double())
        assert out_mu.dtype == torch.float64
        assert out_logvar.dtype == torch.float64

    @pytest.mark.parametrize(
        ("mu_shape", "logvar_shape"),
        [
            ((3, 5), (3, 5)),
            ((3, 4), (3, 5)),
            ((3, 4), (2, 4)),
            ((4,), (4,)),
            ((2, 3, 4), (2, 3, 4)),
        ],
    )
    def test_wrong_shapes_raise_value_error(
        self, mu_shape: tuple[int, ...], logvar_shape: tuple[int, ...]
    ) -> None:
        head = LinearHead(in_dim=4, out_dim=2)
        with pytest.raises(ValueError, match="LinearHead expects"):
            head(torch.zeros(mu_shape), torch.zeros(logvar_shape))

    def test_wrong_width_message_names_the_expected_width(self) -> None:
        head = LinearHead(in_dim=4, out_dim=2)
        with pytest.raises(ValueError, match=r"\(batch, 4\).*\(3, 5\)"):
            head(torch.zeros(3, 5), torch.zeros(3, 5))


class TestLinearHeadGradients:
    def test_every_parameter_receives_a_nonzero_gradient(self) -> None:
        head = LinearHead(in_dim=6, out_dim=4)
        out_mu, out_logvar = head(*_params(5, 6))
        (out_mu.pow(2).sum() + out_logvar.pow(2).sum()).backward()
        named = dict(head.named_parameters())
        assert set(named) == {
            "to_mu.weight",
            "to_mu.bias",
            "to_logvar.weight",
            "to_logvar.bias",
        }
        for name, parameter in named.items():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name

    def test_gradient_reaches_both_inputs(self) -> None:
        head = LinearHead(in_dim=6, out_dim=4)
        mu, logvar = _params(5, 6, requires_grad=True)
        out_mu, out_logvar = head(mu, logvar)
        (out_mu.sum() + out_logvar.sum()).backward()
        assert mu.grad is not None and mu.grad.abs().sum() > 0
        assert logvar.grad is not None and logvar.grad.abs().sum() > 0

    def test_mu_output_depends_only_on_mu_and_logvar_output_only_on_logvar(self) -> None:
        head = LinearHead(in_dim=6, out_dim=4)
        mu, logvar = _params(5, 6, requires_grad=True)
        out_mu, out_logvar = head(mu, logvar)
        out_mu.sum().backward(retain_graph=True)
        assert logvar.grad is None
        mu.grad = None
        out_logvar.sum().backward()
        assert mu.grad is None

    def test_a_training_step_changes_the_output(self) -> None:
        head = LinearHead(in_dim=6, out_dim=4)
        mu, logvar = _params(5, 6)
        before = head(mu, logvar)[0].detach().clone()
        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
        out_mu, out_logvar = head(mu, logvar)
        (out_mu.pow(2).sum() + out_logvar.pow(2).sum()).backward()
        optimizer.step()
        assert not torch.allclose(head(mu, logvar)[0], before)


class TestLinearHeadInitialisation:
    """The documented initialisation: Xavier uniform weights, zero biases, separate layers."""

    @pytest.mark.parametrize(("in_dim", "out_dim"), [(8, 3), (3, 8), (16, 16)])
    def test_biases_are_zero(self, in_dim: int, out_dim: int) -> None:
        head = LinearHead(in_dim=in_dim, out_dim=out_dim)
        assert torch.equal(head.to_mu.bias, torch.zeros(out_dim))
        assert torch.equal(head.to_logvar.bias, torch.zeros(out_dim))

    @pytest.mark.parametrize(("in_dim", "out_dim"), [(8, 3), (3, 8), (16, 16)])
    def test_weights_lie_within_the_xavier_uniform_bound(self, in_dim: int, out_dim: int) -> None:
        torch.manual_seed(0)
        head = LinearHead(in_dim=in_dim, out_dim=out_dim)
        bound = math.sqrt(6.0 / (in_dim + out_dim))
        for projection in (head.to_mu, head.to_logvar):
            assert projection.weight.shape == (out_dim, in_dim)
            assert projection.weight.abs().max() <= bound

    def test_weights_are_spread_over_the_xavier_range_not_collapsed(self) -> None:
        """A large head fills the Xavier range: its std is close to `bound / sqrt(3)`."""
        torch.manual_seed(0)
        head = LinearHead(in_dim=128, out_dim=128)
        expected_std = math.sqrt(6.0 / 256) / math.sqrt(3.0)
        assert head.to_mu.weight.std().item() == pytest.approx(expected_std, rel=0.1)

    def test_the_prior_is_mapped_to_the_prior(self) -> None:
        """`(mu, logvar) = (0, 0)` is the standard normal and stays there at initialisation."""
        head = LinearHead(in_dim=8, out_dim=3)
        out_mu, out_logvar = head(torch.zeros(4, 8), torch.zeros(4, 8))
        assert torch.equal(out_mu, torch.zeros(4, 3))
        assert torch.equal(out_logvar, torch.zeros(4, 3))

    def test_output_scale_is_of_the_order_of_the_input_scale(self) -> None:
        torch.manual_seed(0)
        head = LinearHead(in_dim=64, out_dim=64)
        out_mu, _ = head(*_params(2000, 64))
        assert out_mu.std().item() == pytest.approx(1.0, rel=0.25)

    def test_mu_and_logvar_projections_are_separate_layers(self) -> None:
        torch.manual_seed(0)
        head = LinearHead(in_dim=8, out_dim=8)
        assert head.to_mu is not head.to_logvar
        assert head.to_mu.weight is not head.to_logvar.weight
        assert not torch.equal(head.to_mu.weight, head.to_logvar.weight)

    def test_two_heads_on_the_same_widths_start_different(self) -> None:
        torch.manual_seed(0)
        first, second = LinearHead(8, 4), LinearHead(8, 4)
        assert not torch.equal(first.to_mu.weight, second.to_mu.weight)
        mu, logvar = _params(3, 8)
        assert not torch.allclose(first(mu, logvar)[0], second(mu, logvar)[0])

    def test_initialisation_is_reproducible_under_a_seed(self) -> None:
        torch.manual_seed(123)
        first = LinearHead(8, 4)
        torch.manual_seed(123)
        second = LinearHead(8, 4)
        assert torch.equal(first.to_mu.weight, second.to_mu.weight)
        assert torch.equal(first.to_logvar.weight, second.to_logvar.weight)


class TestLinearHeadState:
    def test_state_dict_round_trips(self) -> None:
        source, target = LinearHead(8, 3), LinearHead(8, 3)
        target.load_state_dict(source.state_dict())
        mu, logvar = _params(4, 8)
        for expected, actual in zip(source(mu, logvar), target(mu, logvar), strict=True):
            assert torch.equal(expected, actual)

    def test_state_dict_keys_are_stable(self) -> None:
        """Checkpoints name the projections `to_mu` and `to_logvar`, like `ConcatMlpFusion`."""
        assert sorted(LinearHead(8, 3).state_dict()) == [
            "to_logvar.bias",
            "to_logvar.weight",
            "to_mu.bias",
            "to_mu.weight",
        ]
