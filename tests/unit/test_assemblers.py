"""Unit tests for the assembler contract and the parameter-free assemblers (spec §2.2).

An assembler merges latent vectors that were already sampled into one decoder input. These tests
cover the contract every assembler follows (roadmap P1-4): it is built from `input_dims`, announces
an `output_dim`, says through `requires_matching_dims` whether its inputs must share one width, and
checks what `forward` receives. They also hold the exact values for `concat`, `sum` and `average`
(roadmap P0-4(b)) and the registry error paths for assemblers specifically; the registry contract
for every registry is in `test_registry_contract.py`. The learned assemblers, `weighted_sum` and
`attention`, are in `test_learned_assemblers.py`.
"""

from collections.abc import Sequence

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
BUILT_IN_NAMES = ["attention", "average", "concat", "sum", "weighted_sum"]


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


class _Probe(AbstractAssembler):
    """Smallest concrete assembler: returns its first input. Used to test the base class."""

    @property
    def output_dim(self) -> int:
        return self.input_dims[0]

    def forward(
        self, latents: list[torch.Tensor], present: Sequence[int] | None = None
    ) -> torch.Tensor:
        self._resolvePresent(latents, present)
        return latents[0]


class _TolerantProbe(_Probe):
    """`_Probe` that accepts a subset of its latent spaces."""

    @property
    def handlesMissingLatents(self) -> bool:
        return True


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
        assert torch.equal(ConcatAssembler([2, 3])([first, second]), expected)

    def test_order_follows_the_input_list(self) -> None:
        """Swapping the inputs swaps the blocks: concatenation is not commutative."""
        first, second = _latent(2), _latent(3, offset=100.0)
        forward = ConcatAssembler([2, 3])([first, second])
        backward = ConcatAssembler([3, 2])([second, first])
        assert torch.equal(forward[:, :2], first)
        assert torch.equal(backward[:, :3], second)
        assert not torch.equal(forward, backward)

    def test_accepts_different_dimensions(self) -> None:
        """No dimensionality restriction: the output width is the sum of the input widths."""
        out = ConcatAssembler([6, 10, 1])([_latent(6), _latent(10), _latent(1)])
        assert out.shape == (BATCH_SIZE, 17)

    def test_output_dim_is_the_sum_of_the_input_dims(self) -> None:
        """The announced width is the width `forward` returns."""
        assembler = ConcatAssembler([6, 10, 1])
        assert assembler.output_dim == 17
        assert assembler([_latent(6), _latent(10), _latent(1)]).shape[-1] == assembler.output_dim

    def test_single_input_is_returned_unchanged_in_value(self) -> None:
        """One latent in, the same values out."""
        latent = _latent(4)
        assert torch.equal(ConcatAssembler([4])([latent]), latent)

    def test_gradient_reaches_every_input(self) -> None:
        """Backward through the output gives each input a gradient of ones."""
        first = _latent(2).requires_grad_()
        second = _latent(3).requires_grad_()
        ConcatAssembler([2, 3])([first, second]).sum().backward()
        assert first.grad is not None and second.grad is not None
        assert torch.equal(first.grad, torch.ones_like(first))
        assert torch.equal(second.grad, torch.ones_like(second))

    def test_mismatched_batch_sizes_fail_loudly(self) -> None:
        """Latents from different batches cannot be joined."""
        with pytest.raises(RuntimeError):
            ConcatAssembler([2, 2])([torch.zeros(3, 2), torch.zeros(4, 2)])

    def test_a_latent_of_the_wrong_width_names_both_widths(self) -> None:
        """A latent that does not match `input_dims` is refused before it is concatenated."""
        with pytest.raises(ValueError, match=r"latent vector 1 of shape \(batch, 3\).*\(3, 5\)"):
            ConcatAssembler([2, 3])([_latent(2), _latent(5)])

    def test_a_subset_is_refused_instead_of_returning_a_narrower_tensor(self) -> None:
        """The reason `concat` says it cannot handle missing latents: the width would be wrong."""
        assembler = ConcatAssembler([4, 4])
        with pytest.raises(ValueError, match="needs all 2 latent vectors"):
            assembler([_latent(4)], present=[0])
        with pytest.raises(ValueError, match="was built for 2 latent vectors"):
            assembler([_latent(4)])


class TestSumAssembler:
    """`sum` adds latent vectors elementwise."""

    def test_values(self) -> None:
        """Two inputs, summed entry by entry."""
        first, second = _latent(4), _latent(4, offset=100.0)
        assert torch.equal(SumAssembler([4, 4])([first, second]), first + second)

    def test_three_inputs(self) -> None:
        """The reduction runs over every input, not only the first two."""
        parts = [_latent(4), _latent(4, offset=100.0), _latent(4, offset=1000.0)]
        out = SumAssembler([4, 4, 4])(parts)
        assert out.shape == (BATCH_SIZE, 4)
        assert torch.equal(out[0], torch.tensor([1100.0, 1103.0, 1106.0, 1109.0]))
        assert torch.equal(out, parts[0] + parts[1] + parts[2])

    def test_is_commutative(self) -> None:
        """Order of the inputs does not matter."""
        first, second = _latent(4), _latent(4, offset=100.0)
        assembler = SumAssembler([4, 4])
        assert torch.equal(assembler([first, second]), assembler([second, first]))

    def test_single_input_is_returned_unchanged_in_value(self) -> None:
        """One latent in, the same values out."""
        latent = _latent(4)
        assert torch.equal(SumAssembler([4])([latent]), latent)

    def test_gradient_reaches_every_input(self) -> None:
        """Each addend receives a gradient of one."""
        first = _latent(4).requires_grad_()
        second = _latent(4).requires_grad_()
        SumAssembler([4, 4])([first, second]).sum().backward()
        assert first.grad is not None and second.grad is not None
        assert torch.equal(first.grad, torch.ones_like(first))
        assert torch.equal(second.grad, torch.ones_like(second))

    def test_different_dimensions_are_refused_at_construction(self) -> None:
        """Mismatched widths raise when the assembler is built, not when it first runs.

        `validateRoutingGraph` rejects this wiring earlier still; this is the second line of
        defence if an assembler is built directly with widths that do not line up.
        """
        with pytest.raises(
            ValueError, match=r"SumAssembler requires matching dimensionality.*\[3, 4\]"
        ):
            SumAssembler([4, 3])

    def test_output_dim_is_the_shared_width(self) -> None:
        """The announced width is the common width of the inputs."""
        assert SumAssembler([6, 6, 6]).output_dim == 6

    def test_a_subset_sums_the_latents_it_is_given(self) -> None:
        """With `present`, the sum runs over the given latents, whichever spaces they are."""
        first, third = _latent(4), _latent(4, offset=1000.0)
        out = SumAssembler([4, 4, 4])([first, third], present=[0, 2])
        assert torch.equal(out, first + third)

    def test_a_latent_of_the_wrong_width_names_both_widths(self) -> None:
        """A latent whose width differs from `input_dims` is refused instead of broadcast."""
        with pytest.raises(ValueError, match=r"expects latent vector 1 of shape \(batch, 4\)"):
            SumAssembler([4, 4])([_latent(4), _latent(3)])


class TestAverageAssembler:
    """`average` takes the elementwise mean of latent vectors."""

    def test_values(self) -> None:
        """Two inputs, averaged entry by entry."""
        first, second = _latent(4), _latent(4, offset=100.0)
        assert torch.equal(AverageAssembler([4, 4])([first, second]), (first + second) / 2)

    def test_three_inputs_divide_by_three(self) -> None:
        """The divisor is the number of inputs."""
        zeros, three, six = torch.zeros(2, 5), torch.full((2, 5), 3.0), torch.full((2, 5), 6.0)
        out = AverageAssembler([5, 5, 5])([zeros, three, six])
        assert torch.allclose(out, torch.full((2, 5), 3.0))

    def test_average_of_identical_inputs_is_that_input(self) -> None:
        """Averaging a latent with itself changes nothing."""
        latent = _latent(4)
        assert torch.allclose(AverageAssembler([4, 4, 4])([latent, latent, latent]), latent)

    def test_single_input_is_returned_unchanged_in_value(self) -> None:
        """One latent in, the same values out."""
        latent = _latent(4)
        assert torch.equal(AverageAssembler([4])([latent]), latent)

    def test_gradient_is_one_over_the_number_of_inputs(self) -> None:
        """Each of `n` inputs receives `1 / n` of the output gradient."""
        parts = [_latent(4).requires_grad_() for _ in range(4)]
        AverageAssembler([4, 4, 4, 4])(parts).sum().backward()
        for part in parts:
            assert part.grad is not None
            assert torch.allclose(part.grad, torch.full_like(part, 0.25))

    def test_different_dimensions_are_refused_at_construction(self) -> None:
        """Mismatched widths raise when the assembler is built."""
        with pytest.raises(ValueError, match="AverageAssembler requires matching dimensionality"):
            AverageAssembler([4, 3])

    def test_output_dim_is_the_shared_width(self) -> None:
        """The announced width is the common width of the inputs."""
        assert AverageAssembler([6, 6]).output_dim == 6

    def test_a_subset_averages_the_latents_it_is_given(self) -> None:
        """With `present`, the divisor is the number of latents given, not of latent spaces."""
        first, third = _latent(4), _latent(4, offset=1000.0)
        out = AverageAssembler([4, 4, 4])([first, third], present=[0, 2])
        assert torch.equal(out, (first + third) / 2)


class TestParameterFreeAssemblers:
    """`concat`, `sum` and `average` merge tensors and learn nothing."""

    @pytest.mark.parametrize("name", ["concat", "sum", "average"])
    def test_no_parameters(self, name: str) -> None:
        """No parameters means optimizers and checkpoints have nothing to track for them."""
        assert list(getAssemblerClass(name)([4, 4]).parameters()) == []


class TestConstructorContract:
    """Every assembler is built from `input_dims`, and the base class checks it."""

    def test_input_dims_are_stored_as_a_tuple(self) -> None:
        """A list or any other sequence of widths comes back as a tuple, in order."""
        assert _Probe([3, 5, 7]).input_dims == (3, 5, 7)
        assert _Probe((3, 5)).input_dims == (3, 5)

    def test_empty_input_dims_are_refused(self) -> None:
        """An assembler with nothing to assemble is a wiring mistake."""
        with pytest.raises(ValueError, match="at least one latent vector"):
            _Probe([])

    @pytest.mark.parametrize("bad", [0, -4, 2.0, "4", True, None])
    def test_a_width_that_is_not_a_positive_integer_is_refused(self, bad: object) -> None:
        """Zero, negative, float, string, bool and `None` widths all fail, naming the position."""
        with pytest.raises(ValueError, match=r"`input_dims\[1\]` must be a positive integer"):
            _Probe([4, bad])  # type: ignore[list-item]

    def test_the_matching_dims_flag_defaults_to_false(self) -> None:
        """A strategy that says nothing accepts any widths."""
        assert AbstractAssembler.requires_matching_dims is False
        assert _Probe([3, 5]).input_dims == (3, 5)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("concat", False),
            ("attention", False),
            ("sum", True),
            ("average", True),
            ("weighted_sum", True),
        ],
    )
    def test_built_in_flag_values(self, name: str, expected: bool) -> None:
        """`validateRoutingGraph` reads this flag from the registry, class by class."""
        assert getAssemblerClass(name).requires_matching_dims is expected

    def test_a_subclass_that_sets_the_flag_is_checked_by_the_base_constructor(self) -> None:
        """A new strategy gets the equal-widths check by setting one class attribute."""

        class _Locked(_Probe):
            requires_matching_dims = True

        _Locked([4, 4])
        with pytest.raises(ValueError, match=r"_Locked requires matching dimensionality.*\[3, 4\]"):
            _Locked([4, 3])

    def test_output_dim_is_part_of_the_contract(self) -> None:
        """A subclass that does not announce its output width cannot be built."""

        class _NoOutputDim(AbstractAssembler):
            def forward(
                self, latents: list[torch.Tensor], present: Sequence[int] | None = None
            ) -> torch.Tensor:
                return latents[0]

        with pytest.raises(TypeError, match="output_dim"):
            _NoOutputDim([4])  # type: ignore[abstract]

    @pytest.mark.parametrize("name", BUILT_IN_NAMES)
    def test_every_built_in_is_built_from_input_dims_alone(self, name: str) -> None:
        """All five registered strategies work with no argument besides `input_dims`."""
        assembler = getAssemblerClass(name)([4, 4])
        assert assembler.input_dims == (4, 4)
        assert assembler.output_dim >= 1

    @pytest.mark.parametrize("name", BUILT_IN_NAMES)
    def test_the_announced_output_dim_is_the_width_forward_returns(self, name: str) -> None:
        """`output_dim` is what the decoder is sized with, so `forward` must honour it."""
        assembler = getAssemblerClass(name)([4, 4])
        out = assembler([_latent(4), _latent(4, offset=100.0)])
        assert out.shape == (BATCH_SIZE, assembler.output_dim)


class TestResolvePresent:
    """The shared check of `forward`'s inputs, used by every concrete strategy."""

    def test_no_present_means_every_latent_space_in_order(self) -> None:
        """`present=None` names positions `0..n-1`."""
        assert _Probe([4, 4])._resolvePresent([_latent(4), _latent(4)], None) == (0, 1)

    def test_present_names_the_positions_of_a_subset(self) -> None:
        """The positions come back as given, and each tensor is checked against its own width."""
        probe = _TolerantProbe([4, 6, 8])
        assert probe._resolvePresent([_latent(4), _latent(8)], [0, 2]) == (0, 2)
        with pytest.raises(ValueError, match=r"latent vector 2 of shape \(batch, 8\)"):
            probe._resolvePresent([_latent(4), _latent(6)], [0, 2])

    def test_a_subset_is_refused_when_the_strategy_cannot_take_one(self) -> None:
        """`handlesMissingLatents` is `False` by default, and the check says so."""
        with pytest.raises(ValueError, match=r"_Probe needs all 2 latent vectors"):
            _Probe([4, 4])._resolvePresent([_latent(4)], [0])

    @pytest.mark.parametrize(
        ("present", "message"),
        [
            ([0], "one position per latent tensor"),
            ([], "one position per latent tensor"),
            ([0, 0], "distinct positions"),
            ([0, 3], "distinct positions"),
            ([-1, 1], "distinct positions"),
        ],
    )
    def test_an_inconsistent_present_is_refused(self, present: list[int], message: str) -> None:
        """Wrong length, repeated or out-of-range positions all fail with a clear message."""
        probe = _TolerantProbe([4, 4, 4])
        with pytest.raises(ValueError, match=message):
            probe._resolvePresent([_latent(4), _latent(4)], present)

    def test_a_tensor_that_is_not_two_dimensional_is_refused(self) -> None:
        """Latents are `(batch, dim)` vectors."""
        with pytest.raises(ValueError, match=r"expects latent vector 0 of shape \(batch, 4\)"):
            _Probe([4])._resolvePresent([torch.zeros(4)], None)


class TestHandlesMissingLatents:
    """`handlesMissingLatents` says whether a decoder may run on some of its latent spaces.

    `GlobalVae.forward` reads it when a modality is left out of a pass (spec §5, ADR 0024).
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("concat", False),
            ("sum", True),
            ("average", True),
            ("weighted_sum", True),
            ("attention", True),
        ],
    )
    def test_built_in_values(self, name: str, expected: bool) -> None:
        """Everything but `concat` is defined for any non-empty subset."""
        assert getAssemblerClass(name)([4, 4]).handlesMissingLatents is expected

    def test_an_assembler_that_says_nothing_is_not_assumed_to_tolerate_a_subset(self) -> None:
        """The base default is `False`, so an unknown strategy is never fed a partial input."""
        assert _Probe([4, 4]).handlesMissingLatents is False

    def test_a_subclass_can_override_it(self) -> None:
        """A custom assembler that accepts any non-empty list says so by overriding."""
        assert _TolerantProbe([4, 4]).handlesMissingLatents is True

    @pytest.mark.parametrize("name", ["sum", "average", "weighted_sum", "attention"])
    def test_tolerant_assemblers_do_accept_a_subset(self, name: str) -> None:
        """The flag is honest: one input out of two still gives the full output width."""
        assembler = getAssemblerClass(name)([4, 4])
        assert assembler([_latent(4)], present=[1]).shape == (BATCH_SIZE, assembler.output_dim)

    def test_concat_with_a_subset_is_refused(self) -> None:
        """The reason `concat` says `False`: one of two inputs gives half the expected width."""
        with pytest.raises(ValueError, match="needs all 2 latent vectors"):
            ConcatAssembler([4, 4])([_latent(4)], present=[0])


class TestAbstractAssembler:
    """`AbstractAssembler` is an interface, not something to instantiate."""

    def test_cannot_be_instantiated(self) -> None:
        """The abstract `output_dim` and `forward` block direct construction."""
        with pytest.raises(TypeError):
            AbstractAssembler([4])  # type: ignore[abstract]

    def test_super_forward_raises_not_implemented(self) -> None:
        """A subclass that defers to the base `forward` gets `NotImplementedError`."""

        class _Deferring(AbstractAssembler):
            @property
            def output_dim(self) -> int:
                return self.input_dims[0]

            def forward(
                self, latents: list[torch.Tensor], present: Sequence[int] | None = None
            ) -> torch.Tensor:
                # The abstract body is called on purpose: that is what this test checks.
                return super().forward(latents, present)  # type: ignore[safe-super]

        with pytest.raises(NotImplementedError):
            _Deferring([1])([torch.zeros(1, 1)])

    def test_super_output_dim_raises_not_implemented(self) -> None:
        """A subclass that defers to the base `output_dim` gets `NotImplementedError`."""

        class _Deferring(AbstractAssembler):
            @property
            def output_dim(self) -> int:
                # The abstract body is called on purpose: that is what this test checks.
                return super().output_dim  # type: ignore[safe-super]

            def forward(
                self, latents: list[torch.Tensor], present: Sequence[int] | None = None
            ) -> torch.Tensor:
                return latents[0]

        with pytest.raises(NotImplementedError):
            _ = _Deferring([1]).output_dim


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
        assert set(BUILT_IN_NAMES) <= set(listRegisteredAssemblers())

    def test_duplicate_registration_raises_value_error(self) -> None:
        """Registering over an existing name is refused, and the original stays."""
        with pytest.raises(ValueError, match="Assembler 'concat' is already registered"):

            @registerAssembler("concat")
            class _Impostor(_Probe):
                pass

        assert getAssemblerClass("concat") is ConcatAssembler

    def test_listing_is_sorted_and_contains_the_built_ins(self) -> None:
        """`listRegisteredAssemblers` returns names in alphabetical order."""
        names = listRegisteredAssemblers()
        assert names == sorted(names)
        assert set(BUILT_IN_NAMES) <= set(names)
