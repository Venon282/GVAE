"""Per-modality reconstruction loss aggregation (spec §2.2, §11, §12).

Scale convention (roadmap P2-1, `docs/adr/0029-loss-scale-convention.md`): a reconstruction loss
function returns a mean over every element of the batch, while the regularizers return a sum over
the latent dimensions of each sample. The `reduction` of `computeTotalReconstructionLoss` chooses,
per modality, whether the reconstruction term stays that per-element mean (`"mean"`, the default)
or becomes the per-sample sum averaged over the batch (`"sum"`), which is the scale of the
regularization term.
"""

import math
from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

DEFAULT_REDUCTION = "mean"
"""The reduction used when none is given: the per-element mean `torch.nn.functional` returns."""

SUPPORTED_REDUCTIONS: tuple[str, ...] = ("mean", "sum")
"""Every value a reconstruction reduction can take, see `computeTotalReconstructionLoss`."""


def listSupportedReductions() -> list[str]:
    """Return every reconstruction reduction name this module knows how to apply.

    Returns:
        `["mean", "sum"]`.
    """
    return list(SUPPORTED_REDUCTIONS)


def validateReduction(reduction: object, field_name: str = "reduction") -> None:
    """Check that `reduction` is a supported name or a mapping modality -> supported name.

    Args:
        reduction: The value to check.
        field_name: How the value is called in the error message, so that a caller can name the
            field the user actually wrote (for example `"training.reconstruction_reduction"`).

    Raises:
        ValueError: If `reduction` is neither one of `listSupportedReductions()` nor a mapping
            whose keys are strings and whose values are all one of them. The message starts
            with `field_name` (and the offending key, for a mapping).
    """
    available = ", ".join(SUPPORTED_REDUCTIONS)
    if isinstance(reduction, str):
        if reduction not in SUPPORTED_REDUCTIONS:
            raise ValueError(
                f"{field_name}: '{reduction}' is not a supported reconstruction reduction. "
                f"Available: {available}."
            )
        return
    if isinstance(reduction, Mapping):
        for modality, value in reduction.items():
            if not isinstance(modality, str):
                raise ValueError(
                    f"{field_name}: a modality name must be a string, got {modality!r}."
                )
            if not isinstance(value, str) or value not in SUPPORTED_REDUCTIONS:
                raise ValueError(
                    f"{field_name}['{modality}']: {value!r} is not a supported reconstruction "
                    f"reduction. Available: {available}."
                )
        return
    raise ValueError(
        f"{field_name}: expected one of {available}, or a mapping from a modality name to one of "
        f"them, got {reduction!r} ({type(reduction).__name__})."
    )


def resolveReduction(reduction: str | Mapping[str, str], name: str) -> str:
    """Return the reduction that applies to one modality.

    Args:
        reduction: A single reduction shared by every modality, or a mapping modality name ->
            reduction.
        name: The decoder (modality) name.

    Returns:
        The mapping's entry for `name`, `DEFAULT_REDUCTION` when the mapping has none, or
        `reduction` itself when it is a single name. This is how the weights behave: a modality
        left out of the mapping keeps the default.
    """
    if isinstance(reduction, str):
        return reduction
    return reduction.get(name, DEFAULT_REDUCTION)


def computeTotalReconstructionLoss(
    reconstructions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | float = 1.0,
    loss_fn: LossFn | dict[str, LossFn] = F.mse_loss,
    reduction: str | Mapping[str, str] = DEFAULT_REDUCTION,
) -> torch.Tensor:
    """Sum the batch-averaged reconstruction loss across active decoders.

    Args:
        reconstructions: Decoder name -> reconstruction tensor, as
            returned by `GlobalVae.forward()["reconstructions"]`. A
            subset of the model's decoders is expected whenever a
            subset of modalities was available this pass (spec §5).
        targets: Decoder name -> ground-truth tensor. Must contain at
            least every key present in `reconstructions`.
        weights: A single weight applied to every modality, or a
            per-modality weight dict. Defaults to `1.0` (an unweighted
            sum). The exact weighting scheme is still open (spec §11).
        loss_fn: A single loss function shared by every modality, or a
            per-modality dict of loss functions. Each function takes
            `(reconstruction, target)` and returns a scalar,
            batch-averaged loss, matching how `torch.nn.functional`'s
            own loss functions behave with their default
            `reduction="mean"`. Defaults to
            `torch.nn.functional.mse_loss` (equivalent, up to an
            additive constant, to a Gaussian decoder likelihood with
            fixed unit variance), but any modality whose
            reconstruction target isn't well modeled by that
            assumption (e.g. a binary/segmentation target, better
            served by `binary_cross_entropy`) can override it
            per-modality.
        reduction: How each modality's loss is scaled, a single name for
            every modality or a mapping modality name -> name (a modality
            left out of the mapping uses `"mean"`). `"mean"` (default) keeps
            what `loss_fn` returns: the mean over every element of the
            batch, so a modality's term does not grow with the number of
            values it reconstructs. `"sum"` multiplies it by `N`, the number
            of values one sample reconstructs (the product of every axis
            after the batch axis), which is the sum over each sample's
            values averaged over the batch. It needs a `loss_fn` that
            returns an element-wise mean, as every `torch.nn.functional`
            loss does by default. The regularization term is a sum over the
            latent dimensions, so `"sum"` puts the two terms on the same
            footing. See `docs/adr/0029-loss-scale-convention.md`. Applied
            before `weights`.

    Returns:
        Scalar reconstruction loss, summed across active decoders and
        averaged over the batch.

    Raises:
        ValueError: If `reconstructions` is empty, if `reduction` is not a
            supported name (or a mapping of supported names), or if a
            reconstruction and its target differ in shape. `torch.nn.functional`'s own
            losses only warn about a mismatch and then broadcast, so a
            `(B, 1, H, W)` target against a `(B, H, W)` reconstruction
            would silently become a `(B, B, H, W)` comparison of every
            reconstruction with every target in the batch. Squeeze or
            unsqueeze the target to the reconstruction's shape instead.
        KeyError: If `reconstructions` references a name absent from
            `targets`, or (when `loss_fn` is a per-modality dict) absent
            from `loss_fn`.
    """
    if not reconstructions:
        raise ValueError("computeTotalReconstructionLoss received an empty `reconstructions`.")
    validateReduction(reduction)

    total: torch.Tensor | None = None
    for name, reconstruction in reconstructions.items():
        if name not in targets:
            raise KeyError(f"No target provided for reconstruction '{name}'.")
        weight = weights.get(name, 1.0) if isinstance(weights, dict) else weights
        if isinstance(loss_fn, dict):
            if name not in loss_fn:
                raise KeyError(f"No loss function provided for reconstruction '{name}'.")
            modality_loss_fn = loss_fn[name]
        else:
            modality_loss_fn = loss_fn
        target = targets[name]
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"Reconstruction '{name}' has shape {tuple(reconstruction.shape)} but its "
                f"target has shape {tuple(target.shape)}. A shape mismatch is rejected "
                f"because the loss would otherwise broadcast the two tensors against each "
                f"other and compare the wrong elements. Reshape the target (or the decoder "
                f"output) so that both have the same shape."
            )
        modality_loss = modality_loss_fn(reconstruction, target)
        if resolveReduction(reduction, name) == "sum":
            modality_loss = modality_loss * math.prod(reconstruction.shape[1:])
        term = weight * modality_loss
        total = term if total is None else total + term

    assert total is not None  # guaranteed by the emptiness check above
    return total
