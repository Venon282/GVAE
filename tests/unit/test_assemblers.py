"""Unit tests for the assemblers (spec §2.2, roadmap P0-4(b)).

An assembler merges latent vectors that were already sampled into one decoder input, so these
tests are plain tensor arithmetic: exact values for `concat`, `sum` and `average`, shape and
gradient behaviour, and what happens when the inputs do not line up. The registry error paths
(unknown name, duplicate name) are covered here for assemblers specifically, and generically for
every registry by `test_registry_contract.py`.
"""

import pytest
import torch

import global_vae.assemblers  # noqa: F401  (registers the built-in assemblers)
from global_vae.assemblers.average import AverageAssembler
from global_vae.assemblers.base import AbstractAssembler
from global_vae.assemblers.concat import ConcatAssembler
from global_vae.assemblers.registry import (
    getAssemblerClass,
    listRegisteredAssemblers,
    registerAssembler,
)
from global_vae.assemblers.sum import SumAssembler

BATCH_SIZE = 3


def _latent(dim: int, offset: float = 0.0) -> torch.Tensor:
    """A `(BATCH_SIZE, dim)` tensor whose values are distinct and easy to read.

    Args:
        dim: Feature dimension.
        offset: Added to every value, so two latents never share an entry.

    Returns:
        Row `i` is `offset + i * 10 + [0, 1, ..., dim - 1]`.
    """
    rows = torch.arange(BATCH_SIZE, dtype=torch.float32).unsqueeze(1) * 10.0
    columns = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
    return rows + columns + offset


class TestConcatAssembler:
    """`concat` joins latent vectors along the feature axis."""

    def test_values_and_order(self) -> None:
        """The output is exactly the inputs side by side, first input first."""
        first, second = _latent(2), _latent(3, offset=100.0)
        expected = torch.tensor(
            [
                [0.0, 1.0, 100.0, 101.0, 102.0],
                [10.0, 11.0, 110.0, 111.0, 112.0],
                [20.0, 21.0, 120.0, 121.0, 122.0],
            ]
        )
        assert torch.equal(ConcatAssembler()([first, second]), expected)

    def test_order_follows_the_input_list(self) -> None:
        """Swapping the inputs swaps the blocks: concatenation is not commutative."""
        first, second = _latent(2), _latent(3, offset=100.0)
        forward = ConcatAssembler()([first, second])
        backward = ConcatAssembler()([second, first])
        assert torch.equal(forward[:, :2], first)
        assert torch.equal(backward[:, :3], second)
        assert not torch.equal(forward, backward)

    def test_accepts_different_dimensions(self) -> None:
        """No dimensionality restriction: the output width is the sum of the input widths."""
        out = ConcatAssembler()([_latent(6), _latent(10), _latent(1)])
        assert out.shape == (BATCH_SIZE, 17)

    def test_single_input_is_returned_unchanged_in_value(self) -> None:
        """One latent in, the same values out."""
        latent = _latent(4)
        assert torch.equal(ConcatAssembler()([latent]), latent)

    def test_gradient_reaches_every_input(self) -> None:
        """Backward through the output gives each input a gradient of ones."""
        first = _latent(2).requires_grad_()
        second = _latent(3).requires_grad_()
        ConcatAssembler()([first, second]).sum().backward()
        assert first.grad is not None and second.grad is not None
        assert torch.equal(first.grad, torch.ones_like(first))
        assert torch.equal(second.grad, torch.ones_like(second))

    def test_mismatched_batch_sizes_fail_loudly(self) -> None:
        """Latents from different batches cannot be joined."""
        with pytest.raises(RuntimeError):
            ConcatAssembler()([torch.zeros(3, 2), torch.zeros(4, 2)])


class TestSumAssembler:
    """`sum` adds latent vectors elementwise."""

    def test_values(self) -> None:
        """Two inputs, summed entry by entry."""
        first, second = _latent(4), _latent(4, offset=100.0)
        assert torch.equal(SumAssembler()([first, second]), first + second)

    def test_three_inputs(self) -> None:
        """The reduction runs over every input, not only the first two."""
        parts = [_latent(4), _latent(4, offset=100.0), _latent(4, offset=1000.0)]
        out = SumAssembler()(parts)
        assert out.shape == (BATCH_SIZE, 4)
        assert torch.equal(out[0], torch.tensor([1100.0, 1103.0, 1106.0, 1109.0]))
        assert torch.equal(out, parts[0] + parts[1] + parts[2])

    def test_is_commutative(self) -> None:
        """Order of the inputs does not matter."""
        first, second = _latent(4), _latent(4, offset=100.0)
        assert torch.equal(SumAssembler()([first, second]), SumAssembler()([second, first]))

    def test_single_input_is_returned_unchanged_in_value(self) -> None:
        """One latent in, the same values out."""
        latent = _latent(4)
        assert torch.equal(SumAssembler()([latent]), latent)

    def test_gradient_reaches_every_input(self) -> None:
        """Each addend receives a gradient of one."""
        first = _latent(4).requires_grad_()
        second = _latent(4).requires_grad_()
        SumAssembler()([first, second]).sum().backward()
        assert first.grad is not None and second.grad is not None
        assert torch.equal(first.grad, torch.ones_like(first))
        assert torch.equal(second.grad, torch.ones_like(second))

    def test_different_dimensions_fail_loudly(self) -> None:
        """Mismatched widths raise instead of broadcasting into a wrong answer.

        `validateRoutingGraph` rejects this wiring at construction; this is the second line of
        defence if an assembler is ever called directly with latents that do not line up.
        """
        with pytest.raises(RuntimeError):
            SumAssembler()([_latent(4), _latent(3)])


class TestAverageAssembler:
    """`average` takes the elementwise mean of latent vectors."""

    def test_values(self) -> None:
        """Two inputs, averaged entry by entry."""
        first, second = _latent(4), _latent(4, offset=100.0)
        assert torch.equal(AverageAssembler()([first, second]), (first + second) / 2)

    def test_three_inputs_divide_by_three(self) -> None:
        """The divisor is the number of inputs."""
        zeros, three, six = torch.zeros(2, 5), torch.full((2, 5), 3.0), torch.full((2, 5), 6.0)
        out = AverageAssembler()([zeros, three, six])
        assert torch.allclose(out, torch.full((2, 5), 3.0))

    def test_average_of_identical_inputs_is_that_input(self) -> None:
        """Averaging a latent with itself changes nothing."""
        latent = _latent(4)
        assert torch.allclose(AverageAssembler()([latent, latent, latent]), latent)

    def test_single_input_is_returned_unchanged_in_value(self) -> None:
        """One latent in, the same values out."""
        latent = _latent(4)
        assert torch.equal(AverageAssembler()([latent]), latent)

    def test_gradient_is_one_over_the_number_of_inputs(self) -> None:
        """Each of `n` inputs receives `1 / n` of the output gradient."""
        parts = [_latent(4).requires_grad_() for _ in range(4)]
        AverageAssembler()(parts).sum().backward()
        for part in parts:
            assert part.grad is not None
            assert torch.allclose(part.grad, torch.full_like(part, 0.25))

    def test_different_dimensions_fail_loudly(self) -> None:
        """Mismatched widths raise instead of broadcasting into a wrong answer."""
        with pytest.raises(RuntimeError):
            AverageAssembler()([_latent(4), _latent(3)])


class TestAssemblersAreParameterFree:
    """The built-in assemblers merge tensors and learn nothing."""

    @pytest.mark.parametrize("name", ["concat", "sum", "average"])
    def test_no_parameters(self, name: str) -> None:
        """No parameters means optimizers and checkpoints have nothing to track for them."""
        assert list(getAssemblerClass(name)().parameters()) == []


class TestAbstractAssembler:
    """`AbstractAssembler` is an interface, not something to instantiate."""

    def test_cannot_be_instantiated(self) -> None:
        """The abstract `forward` blocks direct construction."""
        with pytest.raises(TypeError):
            AbstractAssembler()  # type: ignore[abstract]

    def test_super_forward_raises_not_implemented(self) -> None:
        """A subclass that defers to the base `forward` gets `NotImplementedError`."""

        class _Deferring(AbstractAssembler):
            def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
                return super().forward(latents)

        with pytest.raises(NotImplementedError):
            _Deferring()([torch.zeros(1, 1)])


class TestAssemblerRegistry:
    """Registration, lookup, and the error paths of the assembler registry."""

    @pytest.mark.parametrize(
        ("name", "expected_cls"),
        [("concat", ConcatAssembler), ("sum", SumAssembler), ("average", AverageAssembler)],
    )
    def test_built_ins_are_registered_under_their_documented_names(
        self, name: str, expected_cls: type[AbstractAssembler]
    ) -> None:
        """`concat`, `sum` and `average` resolve to their classes."""
        assert getAssemblerClass(name) is expected_cls

    def test_unknown_name_raises_key_error_listing_the_available_names(self) -> None:
        """The error names the unknown key and every registered assembler."""
        with pytest.raises(KeyError) as excinfo:
            getAssemblerClass("does_not_exist")
        message = str(excinfo.value)
        assert "does_not_exist" in message
        for name in listRegisteredAssemblers():
            assert name in message
        assert {"concat", "sum", "average"} <= set(listRegisteredAssemblers())

    def test_duplicate_registration_raises_value_error(self) -> None:
        """Registering over an existing name is refused, and the original stays."""
        with pytest.raises(ValueError, match="Assembler 'concat' is already registered"):

            @registerAssembler("concat")
            class _Impostor(AbstractAssembler):
                def forward(self, latents: list[torch.Tensor]) -> torch.Tensor:
                    return latents[0]

        assert getAssemblerClass("concat") is ConcatAssembler

    def test_listing_is_sorted_and_contains_the_built_ins(self) -> None:
        """`listRegisteredAssemblers` returns names in alphabetical order."""
        names = listRegisteredAssemblers()
        assert names == sorted(names)
        assert {"concat", "sum", "average"} <= set(names)
