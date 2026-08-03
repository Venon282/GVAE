"""Raw PyTorch training loop for a `GlobalVae` instance (spec §10, §6.1).

Spec §10 resolves the raw-loop-vs-Lightning open question in favor of
a raw PyTorch loop for now, migrating to Lightning (or Lightning
Fabric as an intermediate step) once the architecture stabilizes; see
`docs/adr/0004-pluggable-beta-schedules.md`'s own note on this and
`training/NOTE.md`'s prior history. `Trainer` is that raw loop.

Data pipeline responsibilities (datasets, transforms, batching,
train/val/test splitting) are explicitly out of this framework's
scope: `Trainer` only ever consumes an already-built `Iterable`
yielding batches shaped `dict[str, torch.Tensor]` (modality name ->
raw tensor for the whole batch), the same per-modality convention
`GlobalVae.forward` already uses. For the single-modality
`signal -> z -> signal` case (spec §6.1 milestone 1) that dict has one
key; for a paired signal+image setup (milestone 2) it has one key per
modality. Whatever the caller's dataloader yields is used both as the
encoder input (after modality dropout, if enabled) and as the decoder
reconstruction target: this is what makes plain autoencoding batches
trivial to feed in (a single dict, no separate "target" the caller has
to construct) while still supporting spec §5's modality dropout
without any extra plumbing on the caller's side.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from torch import nn
from torch.optim import Optimizer

from global_vae.losses.reconstruction import LossFn, computeTotalReconstructionLoss
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedule_resolution import resolveBetaSchedules
from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.callbacks import TrainerCallback
from global_vae.utils.autograd import backward

logger = logging.getLogger(__name__)


@dataclass
class StepLosses:
    """The three loss tensors produced by one `Trainer.computeLosses` call.

    Kept as a small dataclass rather than a plain dict of tensors so
    call sites (`fitEpoch`, `evaluate`) get named, typed access instead
    of magic string keys; `asMetrics()` is the one place those string
    keys are defined, shared by every consumer (console logging,
    `TrainerCallback.onStepEnd`, `self.history`).

    Attributes:
        total: `reconstruction + regularization`, the tensor
            `fitEpoch` actually calls `.backward()` on.
        reconstruction: Summed, batch-averaged reconstruction loss
            across every active decoder this step
            (`losses.reconstruction.computeTotalReconstructionLoss`).
        regularization: Summed, batch-averaged regularization loss
            across every active latent space this step, already
            weighted by beta (`GlobalVae.computeRegularizationLoss`).
    """

    total: torch.Tensor
    reconstruction: torch.Tensor
    regularization: torch.Tensor

    def asMetrics(self) -> dict[str, float]:
        """Convert to a flat, JSON/logging-friendly `dict[str, float]`.

        Returns:
            `{"loss/total": ..., "loss/reconstruction": ...,
            "loss/regularization": ...}`, detached from the autograd
            graph (`.item()`).
        """
        return {
            "loss/total": self.total.item(),
            "loss/reconstruction": self.reconstruction.item(),
            "loss/regularization": self.regularization.item(),
        }


class Trainer:
    """Raw PyTorch training loop: forward, reconstruction + regularization loss, backward, step.

    Owns exactly the concerns spec §10 assigns to the training loop
    for now (model optimization, device placement, per-step/per-epoch
    metrics) and nothing else: no data loading (the caller's
    `Iterable`), no experiment tracking (`TrainerCallback`, spec §10's
    "Experiment tracking" item, deferred to a dedicated logger
    subpackage that plugs into the same callback seam), no
    checkpointing (a separate, not-yet-built milestone).
    """

    def __init__(
        self,
        model: GlobalVae,
        optimizer: Optimizer | type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
        reconstruction_weights: dict[str, float] | float = 1.0,
        reconstruction_loss_fn: LossFn | dict[str, LossFn] = F.mse_loss,
        beta: dict[str, float] | float = 1.0,
        beta_schedules: dict[str, AbstractBetaSchedule] | None = None,
        modality_dropout_p: float = 0.0,
        grad_clip_norm: float | None = None,
        callbacks: list[TrainerCallback] | None = None,
        log_every_n_steps: int = 50,
    ) -> None:
        """Build a `Trainer` around an already-constructed `GlobalVae`.

        Args:
            model: The model to train. Moved to `device` immediately.
            optimizer: Either an already-constructed `torch.optim.Optimizer`
                (full control, e.g. custom per-parameter-group learning
                rates set up by the caller), or an optimizer class
                (e.g. `torch.optim.Adam`, the default, or
                `torch.optim.SGD`), which `Trainer` instantiates itself
                as `optimizer(model.parameters(), **optimizer_kwargs)`.
                Mirrors the "value or factory" flexibility pattern
                already used elsewhere in this codebase (e.g.
                `OneDCnnEncoder`'s `activations`/`normalizations`).
            optimizer_kwargs: Constructor kwargs used only when
                `optimizer` is a class (e.g. `{"lr": 1e-3}`). Ignored
                (with nothing to warn about; there is nothing to pass
                them to) when `optimizer` is already an instance.
            device: `"cpu"`, `"cuda"`, `"cuda:0"`, a `torch.device`, or
                `None` (default) to auto-select `"cuda"` if available,
                else `"cpu"`.
            reconstruction_weights: Forwarded to
                `computeTotalReconstructionLoss` unchanged: a single
                weight shared by every modality, or a per-modality
                weight dict.
            reconstruction_loss_fn: Forwarded to
                `computeTotalReconstructionLoss` unchanged: a single
                loss function shared by every modality, or a
                per-modality dict of loss functions. Defaults to
                `torch.nn.functional.mse_loss`, matching that
                function's own default.
            beta: Base regularization weight (spec §2.3): a single
                value shared by every latent space, or a per-latent-space
                dict, used for any latent space that has no entry in
                `beta_schedules`.
            beta_schedules: Latent space name -> `AbstractBetaSchedule`
                instance (`training/beta_schedules/`). Resolved fresh
                every training step via `resolveBetaSchedules(...)` and
                takes priority over `beta` for exactly the latent
                spaces it covers; any latent space in neither `beta`
                (as a dict) nor `beta_schedules` falls back to
                `computeTotalRegularizationLoss`'s own default weight
                of `1.0`. `None` (default) means no latent space has a
                schedule: `beta` alone applies throughout training,
                unannealed.
            modality_dropout_p: Per-modality probability of hiding that
                modality's input from the encoders this step (spec
                §5's recommended technique for missing-modality
                robustness), independently per modality, never
                dropping every modality in a batch (at least one is
                always kept). `0.0` (default) disables it entirely,
                the correct value for a single-modality model (spec
                §6.1 milestone 1), where there is nothing to drop.
                Dropped modalities are still used as reconstruction
                targets: only the encoder side is affected.
            grad_clip_norm: If given, `torch.nn.utils.clip_grad_norm_`
                is applied with this max norm before every optimizer
                step. `None` (default) disables clipping.
            callbacks: `TrainerCallback` instances to notify at every
                training-loop event (`training/callbacks.py`). Several
                may be active at once (e.g. a metrics logger and a
                checkpointer). `None` (default) means no callbacks.
            log_every_n_steps: How often (in global steps) `Trainer`
                logs step-level progress via the standard `logging`
                module (never `print`, spec §10). Every epoch boundary
                is always logged regardless of this value.

        Raises:
            ValueError: If `modality_dropout_p` is not in `[0, 1]`, if
                `grad_clip_norm` is given and not positive, or if
                `log_every_n_steps` is not positive.
        """
        if not (0.0 <= modality_dropout_p <= 1.0):
            raise ValueError(f"modality_dropout_p must be in [0, 1], got {modality_dropout_p}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive when given, got {grad_clip_norm}.")
        if log_every_n_steps <= 0:
            raise ValueError(f"log_every_n_steps must be positive, got {log_every_n_steps}.")

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)

        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer(self.model.parameters(), **(optimizer_kwargs or {}))

        self.reconstruction_weights = reconstruction_weights
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.beta = beta
        self.beta_schedules = beta_schedules or {}
        self.modality_dropout_p = modality_dropout_p
        self.grad_clip_norm = grad_clip_norm
        self.callbacks = list(callbacks) if callbacks else []
        self.log_every_n_steps = log_every_n_steps

        self.global_step = 0
        self.start_epoch = 0
        self.history: list[dict[str, float]] = []

        logger.info("Trainer initialized on device '%s'.", self.device)

    def computeLosses(self, batch: dict[str, torch.Tensor], step: int) -> StepLosses:
        """Run one forward pass and compute the weighted total loss for `batch`.

        Shared by `fitEpoch` (which calls `.backward()` on the result)
        and `evaluate` (which does not), so the forward/loss logic
        exists exactly once.

        Args:
            batch: Modality name -> raw tensor, already moved to
                `self.device`. Used as the reconstruction target
                unchanged, and (after modality dropout, if enabled) as
                the encoder input.
            step: Global step index used to resolve any per-latent-space
                beta schedule for this call. `evaluate` passes the
                current `self.global_step` without advancing it, so a
                validation pass never itself moves a schedule forward.

        Returns:
            `StepLosses` still attached to the autograd graph.

        Raises:
            ValueError: If `batch` is empty, or (via
                `computeTotalReconstructionLoss`) if it is missing a
                target for a decoder that produced a reconstruction.
        """
        if not batch:
            raise ValueError("Trainer.computeLosses received an empty batch.")

        inputs = self._applyModalityDropout(batch)
        outputs = self.model(inputs)

        reconstruction_loss = computeTotalReconstructionLoss(
            outputs["reconstructions"],
            batch,
            weights=self.reconstruction_weights,
            loss_fn=self.reconstruction_loss_fn,
        )
        resolved_beta = self._computeBeta(step)
        regularization_loss = self.model.computeRegularizationLoss(
            outputs["latent_params"], beta=resolved_beta
        )
        total_loss = reconstruction_loss + regularization_loss
        return StepLosses(
            total=total_loss, reconstruction=reconstruction_loss, regularization=regularization_loss
        )

    def fitEpoch(
        self, dataloader: Iterable[dict[str, torch.Tensor]], epoch: int
    ) -> dict[str, float]:
        """Run one training epoch: forward, backward, optimizer step, for every batch.

        Args:
            dataloader: Yields one `dict[str, torch.Tensor]` batch at a
                time; any `Iterable` works, not only
                `torch.utils.data.DataLoader`.
            epoch: Index of this epoch (0-based), forwarded to
                callbacks and used only for logging/bookkeeping;
                `fitEpoch` itself does not track epoch state (`fit`
                does).

        Returns:
            This epoch's metrics, averaged over every step, keys
            prefixed `"train/"` (e.g. `"train/loss/total"`).

        Raises:
            ValueError: If `dataloader` yields no batches.
        """
        self.model.train()
        running_totals: dict[str, float] = {}
        num_batches = 0

        for callback in self.callbacks:
            callback.onEpochBegin(self, epoch)

        for raw_batch in dataloader:
            batch = self._moveBatchToDevice(raw_batch)
            self.optimizer.zero_grad()
            losses = self.computeLosses(batch, self.global_step)
            backward(losses.total)
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            step_metrics = losses.asMetrics()
            for key, value in step_metrics.items():
                running_totals[key] = running_totals.get(key, 0.0) + value
            num_batches += 1

            if self.global_step % self.log_every_n_steps == 0:
                logger.info(
                    "epoch %d step %d: %s",
                    epoch,
                    self.global_step,
                    ", ".join(f"{key}={value:.4f}" for key, value in step_metrics.items()),
                )
            for callback in self.callbacks:
                callback.onStepEnd(self, self.global_step, step_metrics)

            self.global_step += 1

        if num_batches == 0:
            raise ValueError(
                "fitEpoch received an empty dataloader: at least one batch is required."
            )

        return {f"train/{key}": value / num_batches for key, value in running_totals.items()}

    def evaluate(self, dataloader: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Run one pass over `dataloader` without gradient updates.

        Usable both for periodic validation during `fit` and for a
        standalone evaluation pass after training.

        Args:
            dataloader: Yields one `dict[str, torch.Tensor]` batch at a
                time, same contract as `fitEpoch`.

        Returns:
            Metrics averaged over every batch, keys prefixed `"val/"`.

        Raises:
            ValueError: If `dataloader` yields no batches.
        """
        self.model.eval()
        running_totals: dict[str, float] = {}
        num_batches = 0

        with torch.no_grad():
            for raw_batch in dataloader:
                batch = self._moveBatchToDevice(raw_batch)
                losses = self.computeLosses(batch, self.global_step)
                for key, value in losses.asMetrics().items():
                    running_totals[key] = running_totals.get(key, 0.0) + value
                num_batches += 1

        if num_batches == 0:
            raise ValueError(
                "evaluate received an empty dataloader: at least one batch is required."
            )

        return {f"val/{key}": value / num_batches for key, value in running_totals.items()}

    def fit(
        self,
        train_dataloader: Iterable[dict[str, torch.Tensor]],
        num_epochs: int,
        val_dataloader: Iterable[dict[str, torch.Tensor]] | None = None,
    ) -> list[dict[str, float]]:
        """Run the full training loop for `num_epochs` epochs.

        Callable more than once on the same `Trainer`: `self.global_step`
        and epoch numbering both persist across calls (e.g. "train 5
        more epochs" after inspecting `self.history`), which is also
        what a future checkpoint-resume feature is expected to build
        on, without any change needed here.

        Args:
            train_dataloader: Yields training batches; re-iterated
                fresh for every epoch. A `torch.utils.data.DataLoader`
                does this automatically; a plain generator does not
                and must be recreated per epoch by the caller (e.g.
                wrap it in a re-iterable object, not a single-use
                generator).
            num_epochs: Number of epochs to run.
            val_dataloader: Optional; if given, `evaluate()` runs after
                every training epoch and its metrics are merged into
                that epoch's entry in `self.history`.

        Returns:
            `self.history` (also retained on the instance for later
            inspection): one metrics dict per epoch run so far across
            every `fit()` call on this `Trainer`.

        Raises:
            ValueError: If `num_epochs` is not positive.
        """
        if num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {num_epochs}.")

        for callback in self.callbacks:
            callback.onTrainBegin(self)

        try:
            for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
                epoch_metrics = self.fitEpoch(train_dataloader, epoch)
                if val_dataloader is not None:
                    epoch_metrics.update(self.evaluate(val_dataloader))

                self.history.append(epoch_metrics)
                logger.info(
                    "epoch %d complete: %s",
                    epoch,
                    ", ".join(f"{key}={value:.4f}" for key, value in epoch_metrics.items()),
                )
                for callback in self.callbacks:
                    callback.onEpochEnd(self, epoch, epoch_metrics)
                self.start_epoch = epoch + 1
        finally:
            for callback in self.callbacks:
                callback.onTrainEnd(self)

        return self.history

    def _moveBatchToDevice(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move every tensor in `batch` to `self.device`.

        Args:
            batch: Modality name -> raw tensor, on any device.

        Returns:
            The same dict shape, every tensor on `self.device`.
        """
        return {name: tensor.to(self.device) for name, tensor in batch.items()}

    def _applyModalityDropout(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Randomly hide a subset of modalities from the encoders this step (spec §5).

        Each modality is dropped independently with probability
        `self.modality_dropout_p`; at least one modality is always
        kept, since dropping every modality would leave the model
        nothing to encode.

        Args:
            batch: Modality name -> raw tensor, the full batch (used
                unchanged as the reconstruction target regardless of
                what this method returns).

        Returns:
            `batch` unchanged if `self.modality_dropout_p == 0.0` or
            `batch` has only one modality (nothing meaningful to drop);
            otherwise a subset of `batch` with at least one modality
            kept.
        """
        if self.modality_dropout_p <= 0.0 or len(batch) <= 1:
            return batch

        kept = {
            name: tensor
            for name, tensor in batch.items()
            if torch.rand(()).item() >= self.modality_dropout_p
        }
        if not kept:
            fallback_name = next(iter(batch))
            kept = {fallback_name: batch[fallback_name]}
        return kept

    def _computeBeta(self, step: int) -> dict[str, float]:
        """Resolve this step's beta weight for every latent space in `self.model`.

        Args:
            step: Global step index to resolve `self.beta_schedules`
                against.

        Returns:
            Latent space name -> resolved beta value: the schedule's
            value for latent spaces in `self.beta_schedules`, `self.beta`
            (broadcast if it is a single float) otherwise.
        """
        latent_names = self.model.latent_spaces.keys()
        base = (
            dict(self.beta)
            if isinstance(self.beta, dict)
            else dict.fromkeys(latent_names, self.beta)
        )
        if not self.beta_schedules:
            return base
        resolved_schedule_betas = resolveBetaSchedules(self.beta_schedules, step)
        return {**base, **resolved_schedule_betas}
