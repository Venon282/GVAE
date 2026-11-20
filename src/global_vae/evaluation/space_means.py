"""The posterior means of one latent space over a loader of unpaired batches (ADR 0061).

The target of a flow prior is what one space does with the samples that only its modality has:
thousands of them, in batches that carry that modality and nothing else. `collectSpaceMeans`
encodes them and keeps the posterior mean, as `collectViewMeans` does for a set of views over
paired batches. A batch that lacks an input of the view is **skipped**, so that a mixed stream (the
output of `interleaveBatches`) can be given as it is, and a loader that never carries the
modality is refused, since an empty target is not a prior.
"""

from collections.abc import Iterable

import torch

from global_vae.evaluation.latent_alignment import LatentView, _encodeView, resolveLatentView
from global_vae.models.global_vae import GlobalVae


def collectSpaceMeans(
    model: GlobalVae,
    view: LatentView | str,
    dataloader: Iterable[dict[str, torch.Tensor]],
    device: str | torch.device | None = None,
    max_samples: int | None = None,
) -> torch.Tensor:
    """Encode the batches that carry a view's inputs and keep the posterior mean of its space.

    Calls `model.eval()`; no gradient is tracked. The view is encoded from its own inputs alone,
    with `use_mean=True`.

    Args:
        model: The model to query.
        view: A `LatentView` or its text (`z_signal`, or `z_fused@signal`).
        dataloader: Yields batches. Those that lack an input modality of the view are skipped.
        device: Batches are moved here. Defaults to `model`'s own device.
        max_samples: Stop after this many samples, trimming the last batch. `None` reads it all.

    Returns:
        An `(N, width)` `float32` tensor on the CPU.

    Raises:
        ValueError: If `max_samples` is below 1, the view does not fit the model, or no batch
            carried the view's inputs.
        KeyError: If the space is not a latent space of the model.
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be at least 1, got {max_samples}.")
    resolved = resolveLatentView(model, view)
    resolved_device = device if device is not None else next(model.parameters()).device
    model.eval()
    chunks: list[torch.Tensor] = []
    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            if not all(name in batch for name in resolved.inputs):
                continue
            rows = int(batch[resolved.inputs[0]].shape[0])
            keep = rows if max_samples is None else min(rows, max_samples - seen)
            mu, _ = _encodeView(model, batch, resolved, resolved_device, keep)
            chunks.append(mu.float().cpu())
            seen += keep
            if max_samples is not None and seen >= max_samples:
                break
    if not chunks:
        raise ValueError(
            f"No batch of the dataloader carried the input modalities {list(resolved.inputs)} of "
            f"the view '{resolved.toText()}': there is nothing to fit a prior to."
        )
    return torch.cat(chunks)
