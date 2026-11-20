"""Optional sample ids in a batch (roadmap P6-8, ADR 0058).

The ids are how a loader that knows where each row came from says so, and how the pair terms and
the pair evaluation fail loudly on a loader that pairs its samples wrongly. What is pinned here:

- the key of a modality's ids, and that an id entry is not mistaken for a modality;
- `hashSampleIds`: the same name gives the same id in every process (a fixed value is pinned, so a
  change of the hash is a visible change), integers and their decimal text are one sample, and a
  name that is neither a string nor an integer is refused;
- `checkSampleIds`: nothing is checked and `None` is returned when no modality has ids, the rows
  compared are counted, ids for some modalities and not others are refused by name, a malformed
  entry is refused, and a mismatch names the first rows with the id each modality gave them.
"""

import pytest
import torch

from global_vae.data.sample_ids import (
    SAMPLE_ID_PREFIX,
    checkSampleIds,
    hashSampleIds,
    isSampleIdKey,
    sampleIdKey,
)

ROWS = 5


def _batch(**ids: torch.Tensor) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        "signal": torch.zeros(ROWS, 3),
        "image": torch.zeros(ROWS, 2),
    }
    for modality, values in ids.items():
        batch[sampleIdKey(modality)] = values
    return batch


class TestKeys:
    def test_the_key_of_a_modality_is_the_prefix_and_its_name(self) -> None:
        assert SAMPLE_ID_PREFIX == "sample_id."
        assert sampleIdKey("signal") == "sample_id.signal"

    def test_an_empty_modality_has_no_key(self) -> None:
        with pytest.raises(ValueError, match="needs the name of a modality"):
            sampleIdKey("")

    def test_an_id_entry_is_told_from_a_modality(self) -> None:
        assert isSampleIdKey("sample_id.image")
        assert not isSampleIdKey("image")
        assert not isSampleIdKey("sample_id.")
        assert not isSampleIdKey("sample_id")


class TestHashSampleIds:
    def test_the_ids_are_int64_with_one_per_name(self) -> None:
        ids = hashSampleIds(["a", "b", "c"])
        assert ids.dtype == torch.int64
        assert ids.shape == (3,)
        assert (ids >= 0).all()

    def test_the_same_name_gives_the_same_id_and_two_names_give_two(self) -> None:
        first = hashSampleIds(["file_001.npy", "file_002.npy"])
        again = hashSampleIds(["file_002.npy", "file_001.npy"])
        assert first[0] == again[1] and first[1] == again[0]
        assert first[0] != first[1]

    def test_the_id_of_a_name_does_not_depend_on_the_process(self) -> None:
        # Pinned: Python's own hash() of a string changes from run to run, this must not.
        assert hashSampleIds(["file_001.npy"]).tolist() == [212308146065094374]

    def test_an_integer_and_its_decimal_text_are_one_sample(self) -> None:
        assert hashSampleIds([3]).tolist() == hashSampleIds(["3"]).tolist()

    @pytest.mark.parametrize("bad", [1.5, None, True, b"x"])
    def test_a_name_that_is_not_a_string_or_an_integer_is_refused(self, bad: object) -> None:
        with pytest.raises(TypeError, match="string or an integer"):
            hashSampleIds([bad])  # type: ignore[list-item]

    def test_no_names_give_an_empty_tensor(self) -> None:
        assert hashSampleIds([]).shape == (0,)


class TestCheckSampleIds:
    def test_a_batch_without_ids_is_trusted_and_nothing_is_counted(self) -> None:
        assert checkSampleIds(_batch(), ["signal", "image"], "term") is None

    def test_matching_ids_pass_and_the_rows_are_counted(self) -> None:
        ids = torch.arange(ROWS)
        assert (
            checkSampleIds(_batch(signal=ids, image=ids.clone()), ["signal", "image"], "t") == ROWS
        )

    def test_ids_of_another_integer_type_still_compare_by_value(self) -> None:
        ids = torch.arange(ROWS)
        batch = _batch(signal=ids.to(torch.int32), image=ids.to(torch.int64))
        assert checkSampleIds(batch, ["signal", "image"], "t") == ROWS

    def test_a_modality_listed_twice_counts_once(self) -> None:
        ids = torch.arange(ROWS)
        batch = _batch(signal=ids, image=ids)
        assert checkSampleIds(batch, ["signal", "image", "signal"], "t") == ROWS

    def test_ids_of_a_modality_that_is_not_asked_for_are_not_read(self) -> None:
        ids = torch.arange(ROWS)
        batch = _batch(signal=ids, image=ids + 1)
        assert checkSampleIds(batch, ["signal"], "t") == ROWS

    def test_ids_for_some_modalities_and_not_others_are_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"ids for \['signal'\] but none for \['image'\]"):
            checkSampleIds(_batch(signal=torch.arange(ROWS)), ["signal", "image"], "Loss term 'x'")

    def test_the_message_says_who_checked(self) -> None:
        with pytest.raises(ValueError, match="^Loss term 'x': "):
            checkSampleIds(_batch(signal=torch.arange(ROWS)), ["signal", "image"], "Loss term 'x'")

    def test_a_mismatch_names_the_rows_and_the_id_each_modality_gave(self) -> None:
        signal = torch.arange(ROWS)
        image = signal.clone()
        image[1], image[3] = 30, 10
        with pytest.raises(ValueError) as caught:
            checkSampleIds(_batch(signal=signal, image=image), ["signal", "image"], "t")
        message = str(caught.value)
        assert "not the same sample" in message
        assert "differ at 2 of 5 rows" in message
        assert "row 1 (signal 1, image 30)" in message
        assert "row 3 (signal 3, image 10)" in message
        assert "strangers" in message

    def test_a_long_list_of_mismatches_is_cut_after_three_rows(self) -> None:
        signal = torch.arange(ROWS)
        with pytest.raises(ValueError, match=r"differ at 5 of 5 rows.*and 2 more"):
            checkSampleIds(_batch(signal=signal, image=signal + 100), ["signal", "image"], "t")

    def test_three_modalities_are_all_compared_with_the_first(self) -> None:
        ids = torch.arange(ROWS)
        batch = {
            "a": torch.zeros(ROWS, 1),
            "b": torch.zeros(ROWS, 1),
            "c": torch.zeros(ROWS, 1),
            "sample_id.a": ids,
            "sample_id.b": ids,
            "sample_id.c": ids.flip(0),
        }
        with pytest.raises(ValueError, match="row i of 'a' and row i of 'c'"):
            checkSampleIds(batch, ["a", "b", "c"], "t")

    @pytest.mark.parametrize(
        "bad",
        [torch.zeros(ROWS, dtype=torch.float32), torch.zeros(ROWS, dtype=torch.bool)],
    )
    def test_ids_that_are_not_integers_are_refused(self, bad: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="must be an integer tensor of shape"):
            checkSampleIds(_batch(signal=bad, image=torch.arange(ROWS)), ["signal", "image"], "t")

    def test_ids_that_are_not_one_dimensional_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must be an integer tensor of shape"):
            checkSampleIds(
                _batch(signal=torch.zeros(ROWS, 1, dtype=torch.int64), image=torch.arange(ROWS)),
                ["signal", "image"],
                "t",
            )

    def test_one_id_per_row_of_the_modality_is_needed(self) -> None:
        with pytest.raises(ValueError, match="has 4 ids for a 'signal' tensor of 5 rows"):
            checkSampleIds(
                _batch(signal=torch.arange(4), image=torch.arange(ROWS)), ["signal", "image"], "t"
            )

    def test_two_modalities_with_different_numbers_of_rows_are_refused(self) -> None:
        batch = {
            "signal": torch.zeros(4, 3),
            "image": torch.zeros(5, 2),
            "sample_id.signal": torch.arange(4),
            "sample_id.image": torch.arange(5),
        }
        with pytest.raises(ValueError, match="same number of rows"):
            checkSampleIds(batch, ["signal", "image"], "t")

    def test_the_names_of_the_ids_of_a_modality_the_batch_lacks_are_still_checked(self) -> None:
        ids = torch.arange(ROWS)
        batch = {"sample_id.signal": ids, "sample_id.image": ids + 1}
        with pytest.raises(ValueError, match="not the same sample"):
            checkSampleIds(batch, ["signal", "image"], "t")
