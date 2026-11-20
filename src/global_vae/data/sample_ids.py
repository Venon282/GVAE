"""Optional sample ids in a batch, so that a mismatched pair fails loudly (roadmap P6-8, ADR 0058).

A paired batch is a dict in which row `i` of every modality is the same sample. Nothing in the
tensors says so, and a loader that joins two datasets can break it quietly: each side sorted
differently, one side filtered, one side shuffled on its own. A pair loss then trains on strangers
and the evaluation reads the result as chance (roadmap finding I-8).

A loader that knows where each row came from can say so, by adding one entry per modality to the
batch: `sample_id.<modality>`, an integer tensor of shape `(batch,)` holding the id that **that
modality's own source** gave to row `i`. The pair-stage loss terms and the pair evaluation compare
the entries of the modalities they need and raise on the first batch where they differ. The entries
are optional: a batch without them is trusted as before. A batch with them for some of the
modalities a pair needs and not for the others is refused, since that cannot be checked.

```python
batch = {
    "signal": signal[rows],
    "image": image[rows],
    "sample_id.signal": hashSampleIds([signal_names[i] for i in rows]),
    "sample_id.image": hashSampleIds([image_names[i] for i in rows]),
}
```

The ids are tensors because a batch is moved to the device tensor by tensor, and integers because
they are compared for equality. `hashSampleIds` turns names (file names, record keys) into ids that
are the same in every process.
"""

import hashlib
from collections.abc import Mapping, Sequence

import torch

SAMPLE_ID_PREFIX = "sample_id."
"""The batch key of a modality's ids is this prefix followed by the modality's name."""

_MAX_ROWS_SHOWN = 3
_ID_BYTES = 8
_ID_MASK = (1 << 63) - 1


def sampleIdKey(modality: str) -> str:
    """The batch key under which a loader puts the ids of one modality's rows.

    Args:
        modality: The modality's name, as in the batch (`signal`).

    Returns:
        `sample_id.<modality>`.

    Raises:
        ValueError: If `modality` is empty.
    """
    if not modality:
        raise ValueError("A sample id key needs the name of a modality, got an empty string.")
    return f"{SAMPLE_ID_PREFIX}{modality}"


def isSampleIdKey(key: str) -> bool:
    """Whether a batch key is one of the id entries (and so is not a modality).

    Args:
        key: A key of a batch.

    Returns:
        `True` for `sample_id.<something>`.
    """
    return key.startswith(SAMPLE_ID_PREFIX) and len(key) > len(SAMPLE_ID_PREFIX)


def hashSampleIds(names: Sequence[str | int]) -> torch.Tensor:
    """Turn the names of samples (file names, record keys, integers) into ids for a batch.

    The id is a 63-bit integer taken from a BLAKE2b digest of the name, so the same name gives the
    same id in every process and on every machine (Python's own `hash` of a string changes from one
    run to the next), and two different names collide with a probability that no dataset reaches.
    An integer is hashed as its decimal text, so `3` and `"3"` are one sample.

    Args:
        names: One name per row, strings or integers.

    Returns:
        An `int64` tensor of shape `(len(names),)`.

    Raises:
        TypeError: If a name is not a string or an integer (a `bool` is not accepted either).
    """
    ids: list[int] = []
    for name in names:
        if isinstance(name, bool) or not isinstance(name, str | int):
            raise TypeError(
                f"A sample name must be a string or an integer, got {type(name).__name__}: "
                f"{name!r}."
            )
        digest = hashlib.blake2b(str(name).encode("utf-8"), digest_size=_ID_BYTES).digest()
        ids.append(int.from_bytes(digest, "big") & _ID_MASK)
    return torch.tensor(ids, dtype=torch.int64)


def _describeRows(
    ids: Mapping[str, torch.Tensor], first: str, others: Sequence[str], bad: torch.Tensor
) -> str:
    """Name the first rows where the ids of the modalities differ, with each modality's id."""
    shown = []
    for row in bad[:_MAX_ROWS_SHOWN].tolist():
        values = ", ".join(f"{name} {int(ids[name][row])}" for name in (first, *others))
        shown.append(f"row {row} ({values})")
    more = "" if bad.numel() <= _MAX_ROWS_SHOWN else f", and {bad.numel() - _MAX_ROWS_SHOWN} more"
    return "; ".join(shown) + more


def checkSampleIds(
    batch: Mapping[str, torch.Tensor], modalities: Sequence[str], where: str
) -> int | None:
    """Check that the rows of a paired batch carry the same sample ids in every modality.

    The ids of the modalities in `modalities` are compared with each other. Nothing is compared
    when no modality has an entry: the ids are optional, and a batch without them is trusted.

    Args:
        batch: Modality name -> tensor, with the `sample_id.<modality>` entries if the loader gives
            them.
        modalities: The modalities that must be the same samples (the ones a pair needs).
        where: Who is checking, for the error message (`Loss term 'latent_alignment'`).

    Returns:
        The number of rows that were compared, or `None` when the batch carries no ids for these
        modalities (nothing was checked).

    Raises:
        ValueError: If only some of the modalities have ids (the message says which), if an id
            entry is not a one-dimensional integer tensor with one id per row of its modality, or
            if the ids of two modalities differ (the message gives the first rows, with the id each
            modality gave them, and what to look at in the loader).
    """
    names = list(dict.fromkeys(modalities))
    present = [name for name in names if sampleIdKey(name) in batch]
    if not present:
        return None
    absent = [name for name in names if name not in present]
    if absent:
        raise ValueError(
            f"{where}: the batch has sample ids for {present} but none for {absent} "
            f"(`{sampleIdKey(absent[0])}`), so the rows of the modalities cannot be checked "
            f"against each other. Give every modality of a pair an id entry, or none."
        )
    ids: dict[str, torch.Tensor] = {}
    for name in present:
        entry = batch[sampleIdKey(name)]
        if entry.dim() != 1 or entry.is_floating_point() or entry.dtype == torch.bool:
            raise ValueError(
                f"{where}: `{sampleIdKey(name)}` must be an integer tensor of shape (batch,), got "
                f"dtype {entry.dtype} and shape {tuple(entry.shape)}. `hashSampleIds` makes one "
                f"from names."
            )
        if name in batch and entry.shape[0] != batch[name].shape[0]:
            raise ValueError(
                f"{where}: `{sampleIdKey(name)}` has {entry.shape[0]} ids for a '{name}' tensor of "
                f"{batch[name].shape[0]} rows: it needs one id per row."
            )
        ids[name] = entry
    first, *others = present
    for other in others:
        if ids[other].shape != ids[first].shape:
            raise ValueError(
                f"{where}: `{sampleIdKey(first)}` has {ids[first].shape[0]} ids and "
                f"`{sampleIdKey(other)}` has {ids[other].shape[0]}: the modalities of a paired "
                f"batch have the same number of rows."
            )
        bad = (ids[first] != ids[other].to(ids[first].device)).nonzero().flatten()
        if bad.numel() > 0:
            raise ValueError(
                f"{where}: row i of '{first}' and row i of '{other}' are not the same sample: "
                f"their ids differ at {bad.numel()} of {ids[first].shape[0]} rows "
                f"({_describeRows(ids, first, [other], bad.cpu())}). A pair loss on this batch "
                f"would train on strangers. Check how the loader pairs its sources: a sort, a "
                f"filter or a shuffle applied to one modality only."
            )
    return int(ids[first].shape[0])
