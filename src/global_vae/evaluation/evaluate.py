"""Standalone evaluation pass: reconstruction metrics, regularization/KL values, distinct
from `Trainer.evaluate` (spec: "un script/mode d'éval distinct de l'entraînement").

`Trainer.evaluate` (spec §10, `docs/adr/0005-training-loop.md`) already runs a
no-gradient pass and reports the same three aggregate losses `fitEpoch` does
(`"val/loss/total"` etc.), for periodic validation *during* training. `evaluate` here
serves a different purpose: a thorough, standalone report (per-modality reconstruction
metrics beyond MSE, per-latent-space regularization values including a KL number that
stays comparable regardless of which regularizer strategy actually trained the model),
usable on any checkpoint independent of ever having built a `Trainer`.

Metrics are computed once over the *entire pooled dataset* (every batch concatenated
first), not batch-averaged, since some metrics (R^2, Pearson r) are not correctly
computable as a naive average of per-batch values, and a naive average would also be
subtly wrong for any metric once batch sizes differ (e.g. a shorter final batch). This
trades memory (the whole test set's reconstructions/latents held at once) for
correctness; `max_samples` bounds it for a very large test set.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)

from global_vae.evaluation.metrics import DEFAULT_RECONSTRUCTION_METRICS, MetricFn
from global_vae.losses.reconstruction import LossFn, computeTotalReconstructionLoss
from global_vae.losses.regularizers.kl_standard_normal import KlStandardNormalRegularizer
from global_vae.models.global_vae import GlobalVae


@dataclass
class EvaluationResults:
    """Everything one `evaluate()` call produces.

    Attributes:
        num_samples: Number of samples the metrics below were computed
            over (after any `max_samples` truncation).
        total_reconstruction_loss: The same aggregate
            `computeTotalReconstructionLoss` value `Trainer` itself
            reports (batch-averaged), for direct comparability with
            training/validation curves.
        total_regularization_loss: The same aggregate
            `GlobalVae.computeRegularizationLoss` value (batch-averaged,
            `beta` as passed to `evaluate`), for the same reason.
        reconstruction_metrics: Decoder/modality name -> metric name ->
            value, computed once over the full pooled dataset (see
            module docstring).
        regularization_metrics: Latent space name -> `{"configured":
            ..., "kl_standard_normal": ...}`. `"configured"` is that
            latent space's own registered `AbstractLatentRegularizer`
            (spec §2.3, whatever strategy the model was built with,
            unweighted: `beta=1.0` regardless of what `evaluate` was
            called with for the aggregate losses above).
            `"kl_standard_normal"` is always the plain KL-to-standard-
            normal value regardless of the configured strategy, so two
            runs trained with different regularizers (e.g. one with
            `mmd`, one with `free_bits_kl`) still have one directly
            comparable number.
    """

    num_samples: int
    total_reconstruction_loss: float
    total_regularization_loss: float
    reconstruction_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    regularization_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def toDict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable dict.

        Returns:
            `dataclasses.asdict(self)`.
        """
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save as indented JSON.

        Args:
            path: Destination file path. Parent directories are
                created if they do not already exist.
        """
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("w", encoding="utf-8") as handle:
            json.dump(self.toDict(), handle, indent=2)

    def summary(self) -> str:
        """A short, human-readable multi-line summary.

        Returns:
            A string suitable for printing to the console or writing
            to a plain-text report alongside `save`'s JSON.
        """
        lines = [
            f"Evaluation over {self.num_samples} samples",
            f"  total reconstruction loss: {self.total_reconstruction_loss:.6f}",
            f"  total regularization loss: {self.total_regularization_loss:.6f}",
        ]
        if self.reconstruction_metrics:
            lines.append("  reconstruction metrics:")
            for name, metrics in self.reconstruction_metrics.items():
                formatted = ", ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                lines.append(f"    [{name}] {formatted}")
        if self.regularization_metrics:
            lines.append("  regularization metrics:")
            for name, metrics in self.regularization_metrics.items():
                formatted = ", ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                lines.append(f"    [{name}] {formatted}")
        return "\n".join(lines)


def evaluate(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    reconstruction_weights: dict[str, float] | float = 1.0,
    reconstruction_loss_fn: LossFn | dict[str, LossFn] = F.mse_loss,
    reconstruction_metrics: dict[str, MetricFn] | None = None,
    beta: dict[str, float] | float = 1.0,
    use_mean: bool = True,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
) -> EvaluationResults:
    """Run a full evaluation pass over `dataloader` and report reconstruction/regularization
    metrics.

    Calls `model.eval()` (unlike the collection helpers in
    `visualization/`, which leave the mode to the caller: a standalone
    evaluation report should always reflect eval-mode behavior, e.g.
    disabling dropout if a future encoder/decoder uses it).

    Args:
        model: The model to evaluate.
        dataloader: Yields `dict[str, torch.Tensor]` batches, the same
            convention `Trainer` uses.
        reconstruction_weights: Forwarded to
            `computeTotalReconstructionLoss` for
            `total_reconstruction_loss` only (per-modality
            `reconstruction_metrics` are always unweighted, since
            weighting is a training-loss concept, not a reporting one).
        reconstruction_loss_fn: Forwarded to
            `computeTotalReconstructionLoss` for
            `total_reconstruction_loss` only.
        reconstruction_metrics: Metric name -> `(reconstruction,
            target) -> float`. Defaults to
            `evaluation.metrics.DEFAULT_RECONSTRUCTION_METRICS` (mse,
            rmse, mae, r2, pearson_r). Pass your own dict to add,
            remove, or replace metrics without needing a registry.
        beta: Forwarded to `GlobalVae.computeRegularizationLoss` for
            `total_regularization_loss` only (`regularization_metrics`
            are always unweighted: see `EvaluationResults`).
        use_mean: Forwarded to `GlobalVae.forward`. `True` (default,
            unlike `Trainer`, which always samples): deterministic
            reconstructions from the posterior mean, the standard
            choice for reporting reconstruction quality, removing
            sampling noise as a source of run-to-run variance.
        device: Batches are moved here before the forward pass.
            Defaults to `model`'s own device.
        max_samples: Stop accumulating pooled data after at least this
            many samples (the last batch may slightly overshoot before
            being trimmed). `None` (default) uses the entire
            `dataloader`. Bounds memory for a very large test set; the
            aggregate `total_*_loss` values are unaffected either way
            (accumulated as running batch averages, not pooled).

    Returns:
        `EvaluationResults`.

    Raises:
        ValueError: If `dataloader` yields no batches.
    """
    resolved_device = device if device is not None else next(model.parameters()).device
    resolved_metrics = reconstruction_metrics or DEFAULT_RECONSTRUCTION_METRICS
    kl_regularizer = KlStandardNormalRegularizer()

    model.eval()

    collected_originals: dict[str, list[torch.Tensor]] = {}
    collected_reconstructions: dict[str, list[torch.Tensor]] = {}
    collected_mu: dict[str, list[torch.Tensor]] = {}
    collected_logvar: dict[str, list[torch.Tensor]] = {}
    total_reconstruction_loss_sum = 0.0
    total_regularization_loss_sum = 0.0
    num_batches = 0
    num_samples = 0

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = {name: tensor.to(resolved_device) for name, tensor in raw_batch.items()}
            outputs = model(batch, use_mean=use_mean)

            total_reconstruction_loss_sum += computeTotalReconstructionLoss(
                outputs["reconstructions"],
                batch,
                weights=reconstruction_weights,
                loss_fn=reconstruction_loss_fn,
            ).item()
            total_regularization_loss_sum += model.computeRegularizationLoss(
                outputs["latent_params"], beta=beta
            ).item()
            num_batches += 1

            if max_samples is None or num_samples < max_samples:
                for name, reconstruction in outputs["reconstructions"].items():
                    collected_originals.setdefault(name, []).append(batch[name].cpu())
                    collected_reconstructions.setdefault(name, []).append(reconstruction.cpu())
                for name, (mu, logvar) in outputs["latent_params"].items():
                    collected_mu.setdefault(name, []).append(mu.cpu())
                    collected_logvar.setdefault(name, []).append(logvar.cpu())
            num_samples += next(iter(batch.values())).shape[0]

    if num_batches == 0:
        raise ValueError("evaluate received an empty dataloader: at least one batch is required.")

    reconstruction_metric_results: dict[str, dict[str, float]] = {}
    for name, original_chunks in collected_originals.items():
        originals = torch.cat(original_chunks, dim=0)
        reconstructions = torch.cat(collected_reconstructions[name], dim=0)
        if max_samples is not None:
            originals, reconstructions = originals[:max_samples], reconstructions[:max_samples]
        reconstruction_metric_results[name] = {
            metric_name: metric_fn(reconstructions, originals)
            for metric_name, metric_fn in resolved_metrics.items()
        }

    regularization_metric_results: dict[str, dict[str, float]] = {}
    for name, mu_chunks in collected_mu.items():
        mu = torch.cat(mu_chunks, dim=0)
        logvar = torch.cat(collected_logvar[name], dim=0)
        if max_samples is not None:
            mu, logvar = mu[:max_samples], logvar[:max_samples]
        regularization_metric_results[name] = {
            "configured": model.regularizers[name](mu, logvar).mean().item(),
            "kl_standard_normal": kl_regularizer(mu, logvar).mean().item(),
        }

    return EvaluationResults(
        num_samples=min(num_samples, max_samples) if max_samples is not None else num_samples,
        total_reconstruction_loss=total_reconstruction_loss_sum / num_batches,
        total_regularization_loss=total_regularization_loss_sum / num_batches,
        reconstruction_metrics=reconstruction_metric_results,
        regularization_metrics=regularization_metric_results,
    )
