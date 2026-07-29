"""Per-modality reconstruction loss aggregation (spec §2.2, §11, §12).
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def computeTotalReconstructionLoss(
    reconstructions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | float = 1.0,
    loss_fn: LossFn | dict[str, LossFn] = F.mse_loss,
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

    Returns:
        Scalar reconstruction loss, summed across active decoders and
        averaged over the batch.

    Raises:
        ValueError: If `reconstructions` is empty.
        KeyError: If `reconstructions` references a name absent from
            `targets`, or (when `loss_fn` is a per-modality dict) absent
            from `loss_fn`.
    """
    if not reconstructions:
        raise ValueError("computeTotalReconstructionLoss received an empty `reconstructions`.")

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
        term = weight * modality_loss_fn(reconstruction, targets[name])
        total = term if total is None else total + term

    assert total is not None  # guaranteed by the emptiness check above
    return total
