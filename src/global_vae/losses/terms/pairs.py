"""What the pair-stage loss terms share (roadmap P6-4, ADR 0054).

`latent_alignment` and `cross_reconstruction` both act on pairs of latent spaces of one model, in
a stage where every sample has every modality. The checks that follow from that are written once
here, so that the two terms refuse the same mistakes in the same words:

- `parsePairs`: the `pairs` argument is a non-empty list of pairs of two different names;
- `checkStopGradient`: the `stop_gradient` argument is one of `STOP_GRADIENT_CHOICES`;
- `validatePairsAgainstModel`: every name is a latent space of the model, and the two spaces of a
  pair have the same width (roadmap D-16), both at build time and not at the first step;
- `refuseModalityDropout`: a trainer that hides modalities from the encoders is refused when it
  is built, since there is no fusion in this stage to be robust for;
- `requireModalitiesInBatch`: a batch that lacks a modality of a pair is refused, by name,
  because a skipped term would hide a loader that does not pair its samples.

This module registers nothing: it is not a term.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from global_vae.losses.terms.base import TrainerSettings
from global_vae.models.global_vae import GlobalVae

STOP_GRADIENT_CHOICES: tuple[str, ...] = ("none", "source", "target")
"""Where a pair-stage term cuts the gradient: nowhere, or on the first or second name of a pair."""

Pair = tuple[str, str]


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


def validatePairsAgainstModel(model: GlobalVae, pairs: Sequence[Pair], term_name: str) -> None:
    """Check, once, that every pair names latent spaces of `model` that have the same width.

    Equal widths are roadmap decision D-16: a cross decoding passes the sample of one space to the
    decoder of the other, and a distance compares the two posteriors coordinate by coordinate, and
    neither can do that between spaces of different widths.

    Args:
        model: The model the term will run on.
        pairs: The pairs, as `parsePairs` returned them.
        term_name: The term's name, for the error messages.

    Raises:
        ValueError: If a name is not a latent space of `model` (the message lists the real ones),
            or if the two spaces of a pair have different widths (the message gives both).
    """
    available = sorted(model.latent_spaces)
    for first, second in pairs:
        for name in (first, second):
            if name not in model.latent_spaces:
                raise ValueError(
                    f"{term_name}: the pair [{first}, {second}] names '{name}', which is not a "
                    f"latent space of this model (latent spaces: {available})."
                )
        first_width = model.latent_spaces[first].dim
        second_width = model.latent_spaces[second].dim
        if first_width != second_width:
            raise ValueError(
                f"{term_name}: the pair [{first}, {second}] has latent widths {first_width} and "
                f"{second_width}. The two spaces of a pair must have the same width (roadmap "
                f"D-16): set the same `dim` for both in the model's `latent.spaces`."
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
        f"pair its samples."
    )
