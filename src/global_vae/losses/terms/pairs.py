"""What the pair-stage loss terms share (roadmap P6-4, ADR 0054).

`latent_alignment` and `cross_reconstruction` both act on pairs of latent spaces of one model, in
a stage where every sample has every modality. The checks that follow from that are written once
here, so that the two terms refuse the same mistakes in the same words:

- `parsePairs`: the `pairs` argument is a non-empty list of pairs of two different names;
- `resolvePairs`: the pairs of a term, given either as `pairs` or in anchor mode, `anchor` with
  `others` (roadmap P6-8, ADR 0059): every one of several spaces aligned to one, expanded into pairs
  here so that the terms and every check below see an ordinary list of pairs;
- `checkStopGradient`: the `stop_gradient` argument is one of `STOP_GRADIENT_CHOICES`;
- `validatePairsAgainstModel`: every name is a latent space of the model, and the two spaces of a
  pair have the same width (roadmap D-16, or the width of the adapters of ADR 0060), both at build
  time and not at the first step;
- `refuseModalityDropout`: a trainer that hides modalities from the encoders is refused when it
  is built, since there is no fusion in this stage to be robust for;
- `requireModalitiesInBatch`: a batch that lacks a modality of a pair is refused, by name,
  because a skipped term would hide a loader that does not pair its samples;
- `checkOnUnpaired`, `batchHasModalities` and `zeroLoss`: what a term does with a batch that lacks
  a modality of a pair, `raise` (default) or `skip` it with a zero (roadmap P6-8, ADR 0061), for a
  stage whose stream mixes paired batches with unpaired ones;
- `requireSameSamples`: when the batch carries `sample_id.<modality>` entries (ADR 0058), the rows
  of the modalities a pair needs must have the same ids, so that a loader that pairs its samples
  wrongly fails at the first step and not as a retrieval at chance.

This module registers nothing: it is not a term.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from global_vae.data.sample_ids import checkSampleIds
from global_vae.losses.terms.base import TrainerSettings
from global_vae.models.global_vae import GlobalVae

STOP_GRADIENT_CHOICES: tuple[str, ...] = ("none", "source", "target")
"""Where a pair-stage term cuts the gradient: nowhere, or on the first or second name of a pair."""

Pair = tuple[str, str]

ON_UNPAIRED_CHOICES: tuple[str, ...] = ("raise", "skip")
"""What a pair term does with a batch that lacks a modality of a pair (ADR 0061)."""

ANCHOR_DIRECTIONS: tuple[str, ...] = ("to_anchor", "from_anchor", "both")
"""Which way an anchor mode pair runs: other to anchor, anchor to other, or one of each."""


def parsePairs(pairs: Any, term_name: str, allow_reversed_duplicates: bool) -> tuple[Pair, ...]:
    """Check the `pairs` argument of a pair-stage term and return it as a tuple of tuples.

    Args:
        pairs: What the caller passed: a sequence of sequences of two latent space names. A YAML
            list arrives as a list, or as an OmegaConf list, and both are accepted.
        term_name: The term's name, for the error messages.
        allow_reversed_duplicates: Whether `[a, b]` and `[b, a]` may both be listed. `True` for a
            term where the order of a pair means something (a cross reconstruction goes from the
            first to the second), `False` where the two orders are the same thing.

    Returns:
        The pairs, in the order given.

    Raises:
        ValueError: If `pairs` is not a non-empty sequence, if an entry is not a pair of two
            non-empty strings, if the two names of a pair are the same, or if a pair is listed
            twice. The message names the term and the entry.
    """
    if isinstance(pairs, str) or not isinstance(pairs, Sequence):
        raise ValueError(
            f"{term_name}: `pairs` must be a list of pairs of latent space names, "
            f"e.g. [[z_signal, z_image]], got {pairs!r}."
        )
    if len(pairs) == 0:
        raise ValueError(
            f"{term_name}: `pairs` is empty. List at least one pair of latent space names, "
            f"e.g. [[z_signal, z_image]]."
        )
    parsed: list[Pair] = []
    seen: set[Pair] = set()
    for index, entry in enumerate(pairs):
        where = f"{term_name}: pairs[{index}]"
        if (
            isinstance(entry, str)
            or not isinstance(entry, Sequence)
            or len(entry) != 2
            or not all(isinstance(name, str) and name for name in entry)
        ):
            raise ValueError(
                f"{where} must be a pair of two latent space names, e.g. [z_signal, z_image], "
                f"got {entry!r}."
            )
        first, second = str(entry[0]), str(entry[1])
        if first == second:
            raise ValueError(
                f"{where} pairs the latent space '{first}' with itself. A pair is two different "
                f"latent spaces."
            )
        key: Pair = (
            (first, second)
            if allow_reversed_duplicates
            else (min(first, second), max(first, second))
        )
        if key in seen:
            raise ValueError(
                f"{where} lists the pair [{first}, {second}] a second time"
                f"{'' if allow_reversed_duplicates else ' (in either order)'}. "
                f"Each pair counts once."
            )
        seen.add(key)
        parsed.append((first, second))
    return tuple(parsed)


def resolvePairs(
    pairs: Any,
    anchor: Any,
    others: Any,
    term_name: str,
    allow_reversed_duplicates: bool,
    directions: str = "to_anchor",
) -> tuple[Pair, ...]:
    """Return the pairs of a term, from `pairs` or from an anchor and the spaces aligned to it.

    Anchor mode is for more than two latent spaces (roadmap P6-8, ADR 0059): `anchor` names the one
    that every space in `others` is aligned to, which needs `len(others)` pairs and not one for
    each pair of spaces. Each space of `others` makes the pair `[other, anchor]` (the anchor is the
    second name, so `stop_gradient: target` is the setting that leaves it where it is), or
    `[anchor, other]`, or both, as `directions` says.

    Args:
        pairs: The `pairs` argument, or `None` in anchor mode.
        anchor: The anchor's latent space name, or `None` when `pairs` is given.
        others: The names of the spaces aligned to the anchor, or `None` when `pairs` is given.
        term_name: The term's name, for the error messages.
        allow_reversed_duplicates: As in `parsePairs`.
        directions: One of `ANCHOR_DIRECTIONS`. Only read in anchor mode.

    Returns:
        The pairs, as `parsePairs` returns them.

    Raises:
        ValueError: If neither `pairs` nor `anchor` is given, if `pairs` is given with `anchor` or
            `others`, if only one of `anchor` and `others` is given, if `others` is not a non-empty
            list of different names that does not include the anchor, or if `directions` is not
            one of `ANCHOR_DIRECTIONS`. Anything wrong with `pairs` itself is `parsePairs`'s.
    """
    if anchor is None and others is None:
        if pairs is None:
            raise ValueError(
                f"{term_name}: give `pairs`, a list of pairs of latent space names such as "
                f"[[z_signal, z_image]], or `anchor` with `others` to align several spaces to one."
            )
        return parsePairs(pairs, term_name, allow_reversed_duplicates)
    if pairs is not None:
        raise ValueError(
            f"{term_name}: give `pairs` or `anchor` with `others`, not both: anchor mode builds "
            f"the pairs itself."
        )
    if anchor is None or others is None:
        raise ValueError(
            f"{term_name}: `anchor` and `others` go together, got anchor={anchor!r} and "
            f"others={others!r}. `anchor` is the latent space the others are aligned to."
        )
    if directions not in ANCHOR_DIRECTIONS:
        raise ValueError(
            f"{term_name}: directions must be one of {list(ANCHOR_DIRECTIONS)}, got {directions!r}."
        )
    if not isinstance(anchor, str) or not anchor:
        raise ValueError(f"{term_name}: `anchor` must be a latent space name, got {anchor!r}.")
    if isinstance(others, str) or not isinstance(others, Sequence) or len(others) == 0:
        raise ValueError(
            f"{term_name}: `others` must be a non-empty list of latent space names, got {others!r}."
        )
    names = [str(other) for other in others]
    if not all(isinstance(other, str) and other for other in others):
        raise ValueError(f"{term_name}: `others` must hold latent space names, got {others!r}.")
    if anchor in names:
        raise ValueError(
            f"{term_name}: the anchor '{anchor}' is also listed in `others`. A space is not "
            f"aligned to itself."
        )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{term_name}: `others` lists {repeated} more than once.")
    resolved: list[Pair] = []
    for name in names:
        if directions in ("to_anchor", "both"):
            resolved.append((name, anchor))
        if directions in ("from_anchor", "both"):
            resolved.append((anchor, name))
    return tuple(resolved)


def checkStopGradient(value: Any, term_name: str) -> str:
    """Check the `stop_gradient` argument of a pair-stage term.

    Args:
        value: What the caller passed.
        term_name: The term's name, for the error message.

    Returns:
        `value`, one of `STOP_GRADIENT_CHOICES`.

    Raises:
        ValueError: If it is not one of them. The message lists them.
    """
    if value not in STOP_GRADIENT_CHOICES:
        raise ValueError(
            f"{term_name}: stop_gradient must be one of {list(STOP_GRADIENT_CHOICES)}, "
            f"got {value!r}."
        )
    return str(value)


def validatePairsAgainstModel(
    model: GlobalVae,
    pairs: Sequence[Pair],
    term_name: str,
    adapted_widths: Mapping[str, int] | None = None,
) -> None:
    """Check, once, that every pair names latent spaces of `model` that have the same width.

    Equal widths are roadmap decision D-16: a cross decoding passes the sample of one space to the
    decoder of the other, and a distance compares the two posteriors coordinate by coordinate, and
    neither can do that between spaces of different widths. A term that reads a space through an
    adapter (`latent_alignment`, ADR 0060) gives the width of the adapted posterior, and that is
    the width compared; a term that cannot adapt (`cross_reconstruction`) gives nothing.

    Args:
        model: The model the term will run on.
        pairs: The pairs, as `parsePairs` returned them.
        term_name: The term's name, for the error messages.
        adapted_widths: Space name -> the width the term reads that space at, for the spaces it
            has an adapter for (empty when it has none). `None` for a term that cannot read a
            space through an adapter.

    Raises:
        ValueError: If a name is not a latent space of `model` (the message lists the real ones),
            or if the two spaces of a pair have different widths as the term reads them (the
            message gives both, and what to change).
    """
    available = sorted(model.latent_spaces)
    adapted = dict(adapted_widths or {})
    for first, second in pairs:
        for name in (first, second):
            if name not in model.latent_spaces:
                raise ValueError(
                    f"{term_name}: the pair [{first}, {second}] names '{name}', which is not a "
                    f"latent space of this model (latent spaces: {available})."
                )
        first_width = adapted.get(first, model.latent_spaces[first].dim)
        second_width = adapted.get(second, model.latent_spaces[second].dim)
        if first_width != second_width:
            if adapted_widths is None:
                advice = (
                    "set the same `dim` for both in the model's `latent.spaces`. A cross "
                    "reconstruction decodes one space's sample with the other's decoders, which "
                    "no adapter of `latent_alignment` can change (ADR 0060)"
                )
            elif first in adapted or second in adapted:
                advice = (
                    "the adapters leave them at different widths: set the `out_dim` of the "
                    "adapters so that both spaces of a pair are read at one width"
                )
            else:
                advice = (
                    "set the same `dim` for both in the model's `latent.spaces`, or, for an "
                    "alignment that leaves the spaces as they are, give `adapters` (ADR 0060): it "
                    "reads a space through a head of the width of the other"
                )
            raise ValueError(
                f"{term_name}: the pair [{first}, {second}] has latent widths {first_width} and "
                f"{second_width}. The two spaces of a pair must have the same width (roadmap "
                f"D-16): {advice}."
            )


def refuseModalityDropout(settings: TrainerSettings, term_name: str) -> None:
    """Raise if the trainer hides modalities from the encoders.

    Modality dropout makes a model robust to a missing modality by training it with some missing,
    which is what a fusion needs. A pair stage has no fusion: each modality has a latent space of
    its own, and a term that compares two of them needs both encoded in every step.

    Args:
        settings: The settings the trainer gave the term (`bindTrainerSettings`).
        term_name: The term's name, for the error message.

    Raises:
        ValueError: If `settings.modality_dropout_p` is above zero. The message says which
            setting to change.
    """
    if settings.modality_dropout_p > 0.0:
        raise ValueError(
            f"Loss term '{term_name}' needs every modality of a pair encoded in every step, so "
            f"modality dropout must be off in this stage (there is no fusion to be robust for), "
            f"but modality_dropout_p is {settings.modality_dropout_p}. Set "
            f"`training.modality_dropout_p=0`."
        )


def requireModalitiesInBatch(
    batch: Mapping[str, torch.Tensor], needs: Sequence[tuple[str, str]], term_name: str
) -> None:
    """Raise if `batch` lacks a modality that a pair needs.

    The term refuses the batch instead of leaving the pair out. A pair that is left out of a loss
    changes the objective without a trace in the numbers, and the likeliest cause is a loader that
    does not pair its samples, which is what the person running the stage has to hear about.

    Args:
        batch: Modality name -> tensor, as the trainer gives it to a term.
        needs: `(modality, why)` for every modality that the pairs of the term need, in order. The
            reason says which pair it belongs to and what it is needed for.
        term_name: The term's name, for the error message.

    Raises:
        ValueError: If a modality in `needs` is not a key of `batch`. The message says that the
            stage needs paired batches, names every missing modality with its reason, and lists
            what the batch has.
    """
    missing = [(modality, why) for modality, why in needs if modality not in batch]
    if not missing:
        return
    details = "; ".join(f"'{modality}' ({why})" for modality, why in missing)
    raise ValueError(
        f"Loss term '{term_name}' needs paired batches: every batch must carry every modality of a "
        f"pair, and this one has no {details}. The batch has {sorted(batch)}. The term raises "
        f"instead of skipping the pair, because a skipped term would hide a loader that does not "
        f"pair its samples. A stage that feeds unpaired batches on purpose says "
        f"`on_unpaired: skip` (ADR 0061)."
    )


def checkOnUnpaired(value: Any, term_name: str) -> str:
    """Check the `on_unpaired` argument of a pair-stage term.

    Args:
        value: What the caller passed.
        term_name: The term's name, for the error message.

    Returns:
        `value`, one of `ON_UNPAIRED_CHOICES`.

    Raises:
        ValueError: If it is not one of them. The message lists them.
    """
    if value not in ON_UNPAIRED_CHOICES:
        raise ValueError(
            f"{term_name}: on_unpaired must be one of {list(ON_UNPAIRED_CHOICES)}, got {value!r}."
        )
    return str(value)


def batchHasModalities(batch: Mapping[str, torch.Tensor], needs: Sequence[tuple[str, str]]) -> bool:
    """Whether `batch` carries every modality that the pairs of a term need.

    Args:
        batch: Modality name -> tensor, as the trainer gives it to a term.
        needs: `(modality, why)` for every modality the pairs need.

    Returns:
        `True` if every modality is a key of `batch`.
    """
    return all(modality in batch for modality, _ in needs)


def zeroLoss(model: GlobalVae) -> torch.Tensor:
    """The value of a term that skipped a batch: a scalar zero on the model's device.

    It is not attached to the graph: the trainer's total still holds the reconstruction and the
    regularization of the batch, which are, so a skipped term adds nothing and the step trains the
    rest as it would without the term.

    Args:
        model: The model the term runs on.

    Returns:
        A `float32` scalar tensor, `0.0`.
    """
    return torch.zeros((), device=next(model.parameters()).device)


def requireSameSamples(
    batch: Mapping[str, torch.Tensor], needs: Sequence[tuple[str, str]], term_name: str
) -> None:
    """Raise if the sample ids of a batch say that its modalities are not the same samples.

    Nothing happens when the batch has no `sample_id.<modality>` entry for the modalities the pairs
    need: the ids are optional (ADR 0058). Call it after `requireModalitiesInBatch`.

    Args:
        batch: Modality name -> tensor, with the id entries if the loader gives them.
        needs: `(modality, why)` for every modality that the pairs of the term need; only the
            modalities are read.
        term_name: The term's name, for the error message.

    Raises:
        ValueError: As `checkSampleIds`: ids for some modalities and not others, a malformed id
            entry, or ids that differ between two modalities.
    """
    checkSampleIds(batch, [modality for modality, _ in needs], f"Loss term '{term_name}'")
