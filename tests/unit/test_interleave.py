"""`interleaveBatches`: paired batches, each followed by unpaired ones (roadmap P6-8, ADR 0061)."""

from collections.abc import Iterator

import pytest
import torch

from global_vae.data.interleave import InterleavedBatches, interleaveBatches


def _batch(tag: str, index: int) -> dict[str, torch.Tensor]:
    return {tag: torch.full((2, 1), float(index))}


def _loader(tag: str, count: int) -> list[dict[str, torch.Tensor]]:
    return [_batch(tag, index) for index in range(count)]


def _tags(stream: InterleavedBatches) -> list[tuple[str, int]]:
    return [(next(iter(batch)), int(next(iter(batch.values()))[0, 0])) for batch in stream]


class TestOrder:
    def test_each_paired_batch_is_followed_by_one_unpaired_batch_by_default(self) -> None:
        stream = interleaveBatches(_loader("p", 3), {"a": _loader("a", 10)})
        assert _tags(stream) == [("p", 0), ("a", 0), ("p", 1), ("a", 1), ("p", 2), ("a", 2)]

    def test_the_unpaired_loaders_are_used_in_turn(self) -> None:
        stream = interleaveBatches(
            _loader("p", 2), {"a": _loader("a", 10), "b": _loader("b", 10)}, unpaired_per_paired=2
        )
        assert [tag for tag, _ in _tags(stream)] == ["p", "a", "b", "p", "a", "b"]

    def test_a_fraction_puts_an_unpaired_batch_after_every_second_paired_batch(self) -> None:
        stream = interleaveBatches(
            _loader("p", 4), {"a": _loader("a", 10)}, unpaired_per_paired=0.5
        )
        assert [tag for tag, _ in _tags(stream)] == ["p", "p", "a", "p", "p", "a"]

    def test_zero_gives_the_paired_loader_back(self) -> None:
        stream = interleaveBatches(_loader("p", 3), {"a": _loader("a", 10)}, unpaired_per_paired=0)
        assert _tags(stream) == [("p", 0), ("p", 1), ("p", 2)]

    def test_a_single_loader_can_be_given_without_a_name(self) -> None:
        stream = interleaveBatches(_loader("p", 2), _loader("a", 5))
        assert [tag for tag, _ in _tags(stream)] == ["p", "a", "p", "a"]

    def test_more_unpaired_batches_per_paired_than_the_loader_holds_restarts_it(self) -> None:
        stream = interleaveBatches(_loader("p", 2), {"a": _loader("a", 3)}, unpaired_per_paired=3)
        unpaired = [index for tag, index in _tags(stream) if tag == "a"]
        assert unpaired == [0, 1, 2, 0, 1, 2]


class TestAcrossEpochs:
    def test_the_unpaired_loaders_keep_their_place_from_one_epoch_to_the_next(self) -> None:
        stream = interleaveBatches(_loader("p", 2), {"a": _loader("a", 10)})
        first = [index for tag, index in _tags(stream) if tag == "a"]
        second = [index for tag, index in _tags(stream) if tag == "a"]
        assert (first, second) == ([0, 1], [2, 3])

    def test_every_unpaired_batch_is_seen_over_enough_epochs(self) -> None:
        stream = interleaveBatches(_loader("p", 2), {"a": _loader("a", 6)})
        seen = {index for _ in range(3) for tag, index in _tags(stream) if tag == "a"}
        assert seen == set(range(6))

    def test_the_epoch_is_always_one_pass_over_the_paired_loader(self) -> None:
        stream = interleaveBatches(_loader("p", 3), {"a": _loader("a", 4)}, unpaired_per_paired=2)
        for _ in range(3):
            assert [tag for tag, _ in _tags(stream)].count("p") == 3

    def test_a_loader_that_shuffles_on_every_pass_is_asked_again_when_it_runs_out(self) -> None:
        class Counting:
            passes = 0

            def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
                type(self).passes += 1
                yield from _loader("a", 2)

        stream = interleaveBatches(_loader("p", 5), {"a": Counting()}, unpaired_per_paired=1)
        list(stream)
        assert Counting.passes == 3


class TestLength:
    @pytest.mark.parametrize(("ratio", "expected"), [(0, 4), (0.5, 6), (1, 8), (2.5, 14)])
    def test_the_length_is_what_one_epoch_yields(self, ratio: float, expected: int) -> None:
        stream = interleaveBatches(_loader("p", 4), {"a": _loader("a", 3)}, ratio)
        assert len(stream) == expected == len(list(stream))

    def test_a_paired_loader_without_a_length_has_none(self) -> None:
        stream = interleaveBatches(iter(_loader("p", 2)), {"a": _loader("a", 2)})
        with pytest.raises(TypeError):
            len(stream)


class TestRefusals:
    def test_no_unpaired_loader_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one unpaired loader"):
            interleaveBatches(_loader("p", 1), {})

    @pytest.mark.parametrize("ratio", [-1, float("nan"), float("inf")])
    def test_a_ratio_that_is_not_a_finite_non_negative_number_is_refused(
        self, ratio: float
    ) -> None:
        with pytest.raises(ValueError, match="unpaired_per_paired"):
            interleaveBatches(_loader("p", 1), {"a": _loader("a", 1)}, ratio)

    def test_an_empty_unpaired_loader_is_refused_by_name_when_it_is_needed(self) -> None:
        stream = interleaveBatches(_loader("p", 1), {"empty": []})
        with pytest.raises(ValueError, match="'empty' gave no batch"):
            list(stream)
