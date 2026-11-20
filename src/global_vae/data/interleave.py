"""A stream that mixes paired batches with unpaired ones (roadmap P6-8, ADR 0061).

The scarce thing in the staged recipe is the pair: a few dozen samples that carry every modality,
next to thousands that carry one. The solo stages already use every unpaired sample. A pair stage
that wants them too, so that the reconstruction of each VAE keeps seeing the whole of its own data
while the pair loss sees the pairs, needs one stream in which the two kinds of batch alternate.
`interleaveBatches` builds it from loaders the caller already has:

```python
stream = interleaveBatches(
    paired_loader,                                   # batches with signal and image
    {"signal": signal_loader, "image": image_loader},  # batches with one modality each
    unpaired_per_paired=2,
)
trainer.fit(stream, num_epochs=50)
```

An epoch is one pass over the paired loader, and each of its batches is followed by
`unpaired_per_paired` unpaired ones, taken from the unpaired loaders in turn. The unpaired loaders
are **not** restarted at the start of an epoch: their position is kept, and a loader that runs out
starts again, so that every unpaired sample is seen over the epochs even when one epoch uses a
small part of them.

The pair terms must be told to expect such a stream (`on_unpaired: skip`), or they refuse the
first unpaired batch by name (ADR 0061). The loaders themselves, and what they put in a batch, stay
the caller's (spec section 11).
"""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

Batch = dict[str, torch.Tensor]


class InterleavedBatches:
    """A re-iterable stream of paired batches, each followed by some unpaired ones.

    Attributes:
        paired: The loader of paired batches. One pass over it is one epoch of the stream.
        unpaired: Name -> loader of unpaired batches, in the order they are used in turn.
        unpaired_per_paired: How many unpaired batches follow each paired one, on average.
    """

    def __init__(
        self,
        paired: Iterable[Batch],
        unpaired: Mapping[str, Iterable[Batch]],
        unpaired_per_paired: float = 1.0,
    ) -> None:
        """Build the stream.

        Args:
            paired: Yields the paired batches. It must be re-iterable (a `DataLoader`, a list),
                since each epoch iterates it again.
            unpaired: Name -> a re-iterable loader of unpaired batches. The name is for messages.
                At least one.
            unpaired_per_paired: A number of unpaired batches per paired batch, `0` or above. It may
                be a fraction: `0.5` puts one unpaired batch after every second paired batch. The
                count of an epoch is `floor(paired batches * unpaired_per_paired)`, fixed.

        Raises:
            ValueError: If `unpaired` is empty, or `unpaired_per_paired` is negative or not finite.
        """
        if len(unpaired) == 0:
            raise ValueError(
                "interleaveBatches needs at least one unpaired loader, as a mapping name -> loader."
            )
        ratio = float(unpaired_per_paired)
        if not (ratio >= 0.0) or ratio == float("inf"):
            raise ValueError(
                f"unpaired_per_paired must be a finite number of 0 or more, "
                f"got {unpaired_per_paired!r}."
            )
        self.paired = paired
        self.unpaired: dict[str, Iterable[Batch]] = dict(unpaired)
        self.unpaired_per_paired = ratio
        self._names = list(self.unpaired)
        self._iterators: dict[str, Iterator[Batch]] = {}
        self._turn = 0

    def _nextUnpaired(self) -> Batch:
        """The next unpaired batch, from the loaders in turn, restarting a loader that ran out."""
        name = self._names[self._turn % len(self._names)]
        self._turn += 1
        iterator = self._iterators.get(name)
        if iterator is None:
            iterator = iter(self.unpaired[name])
            self._iterators[name] = iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(self.unpaired[name])
            self._iterators[name] = iterator
            try:
                return next(iterator)
            except StopIteration:
                raise ValueError(
                    f"The unpaired loader '{name}' gave no batch: it is empty, so the stream has "
                    f"nothing to put after the paired batches."
                ) from None

    def __iter__(self) -> Iterator[Batch]:
        """One epoch: each paired batch, then the unpaired ones that follow it."""
        credit = 0.0
        for batch in self.paired:
            yield batch
            credit += self.unpaired_per_paired
            while credit >= 1.0:
                credit -= 1.0
                yield self._nextUnpaired()

    def __len__(self) -> int:
        """Batches per epoch: the paired ones and the unpaired ones that follow them.

        Raises:
            TypeError: If the paired loader has no length.
        """
        paired_batches = len(self.paired)  # type: ignore[arg-type]
        credit = 0.0
        unpaired_batches = 0
        for _ in range(paired_batches):
            credit += self.unpaired_per_paired
            while credit >= 1.0:
                credit -= 1.0
                unpaired_batches += 1
        return paired_batches + unpaired_batches


def interleaveBatches(
    paired: Iterable[Batch],
    unpaired: Mapping[str, Iterable[Batch]] | Iterable[Batch],
    unpaired_per_paired: float = 1.0,
) -> InterleavedBatches:
    """Mix a loader of paired batches with loaders of unpaired ones (see the module docstring).

    Args:
        paired: The loader of paired batches, re-iterable.
        unpaired: Name -> re-iterable loader of unpaired batches, or one loader (named `unpaired`).
        unpaired_per_paired: Unpaired batches after each paired one, on average (a fraction
            is fine).

    Returns:
        The `InterleavedBatches` stream, to pass to `Trainer.fit` in place of a loader.

    Raises:
        ValueError: As `InterleavedBatches`.
    """
    sources: Any = unpaired if isinstance(unpaired, Mapping) else {"unpaired": unpaired}
    return InterleavedBatches(paired, sources, unpaired_per_paired)
