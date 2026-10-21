"""Raw PyTorch training loop for a `GlobalVae` instance (spec §10, §6.1).

Spec §10 resolves the raw-loop-vs-Lightning open question in favor of
a raw PyTorch loop for now, migrating to Lightning (or Lightning
Fabric as an intermediate step) once the architecture stabilizes; see
`docs/adr/0004-pluggable-beta-schedules.md`'s own note on this and
`training/NOTE.md`'s prior history. `Trainer` is that raw loop. The triggers for the migration
are in `docs/adr/0036-lightning-migration-triggers.md`.

Data pipeline responsibilities (datasets, transforms, batching,
train/val/test splitting) are explicitly out of this framework's
scope: `Trainer` only ever consumes an already-built `Iterable`
yielding batches shaped `dict[str, torch.Tensor]` (modality name ->
raw tensor for the whole batch), the same per-modality convention
`GlobalVae.forward` already uses. For the single-modality
`signal -> z -> signal` case (spec §6.1 milestone 1) that dict has one
key; for a paired signal+image setup (milestone 2) it has one key per
modality.

The *whole* batch is always used as the reconstruction target (matched
by decoder name), but only the keys that name a registered encoder
(`GlobalVae.selectEncoderInputs`, ADR 0019) are ever used as encoder
input, after modality dropout if enabled. For plain autoencoding this
is the same dict either way (one key per modality, no separate
"target" the caller has to construct), so nothing changes for spec
§6.1's milestones. It is also what makes a translation-style model
work with no special-casing on the caller's side: a batch
`{"image_in": ..., "image_out": ...}` for a model whose only encoder
is `image_in` and whose only decoder is `image_out` trains and
evaluates exactly like any other batch, since `image_out` is simply
never selected as encoder input, only ever matched as `image_out`'s
own decoder target.
"""

import contextlib
import logging
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 (torch convention)
from torch import nn
from torch.optim import Optimizer

from global_vae.losses.reconstruction import (
    DEFAULT_REDUCTION,
    LossFn,
    computeTotalReconstructionLoss,
    validateReduction,
)
from global_vae.models.global_vae import GlobalVae
from global_vae.training.beta_schedule_resolution import resolveBetaSchedules
from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.callbacks.base import TrainerCallback
from global_vae.training.checkpoint import loadCheckpoint as loadCheckpointFile
from global_vae.training.checkpoint import saveCheckpoint as saveCheckpointFile
from global_vae.training.learning_rate import isLearningRateKey, learningRateMetrics
from global_vae.utils.autograd import backward

logger = logging.getLogger(__name__)

SUPPORTED_MIXED_PRECISIONS = ("bf16", "fp16")
"""The values of `Trainer`'s `mixed_precision` besides `None` (full precision)."""

_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _castOutputsToFloat32(outputs: dict[str, Any]) -> dict[str, Any]:
    """Cast the parts of a forward pass that feed the losses back to `float32`.

    Under autocast the model returns reduced-precision tensors. The reconstruction loss and
    the regularizer (an exponential of a log-variance, a sum over dimensions) are computed
    from them, and they are the terms that lose accuracy in `bf16` or `fp16`, so the losses
    are computed in `float32` from the model's outputs.

    Args:
        outputs: What `GlobalVae.forward` returned.

    Returns:
        `outputs` with `"reconstructions"` and `"latent_params"` cast to `float32`.
    """
    return {
        **outputs,
        "reconstructions": {
            name: tensor.float() for name, tensor in outputs["reconstructions"].items()
        },
        "latent_params": {
            name: (mu.float(), logvar.float())
            for name, (mu, logvar) in outputs["latent_params"].items()
        },
    }


def _formatMetrics(metrics: dict[str, float]) -> str:
    """Join `key=value` pairs for a log line.

    Args:
        metrics: Metric key -> value.

    Returns:
        The pairs separated by `", "`: four decimals for a loss, and the shortest general form
        for a learning rate (`1e-05`, not `0.0000`), which four decimals would round away.
    """
    return ", ".join(
        f"{key}={value:.6g}" if isLearningRateKey(key) else f"{key}={value:.4f}"
        for key, value in metrics.items()
    )


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
    itself (model optimization, device placement, per-step/per-epoch
    metrics) and nothing else: no data loading (the caller's
    `Iterable`), no experiment tracking, checkpointing, learning-rate
    scheduling, or early stopping inline in this class. Every one of
    those is a `TrainerCallback` instead (`training/callbacks/`, spec
    §10, ADR 0022), each self-registered by name so a training run
    opts into exactly the callbacks it wants, in whatever combination,
    from `TrainingConfig.callbacks` (`config/training.py`) or by
    constructing them directly and passing `callbacks=[...]` here.

    A *step* is one optimizer update. With `accumulate_grad_batches=1` (the default) that is
    one batch; with `N > 1` it is `N` batches, and `global_step`, the beta schedules, the
    callbacks' `step` argument, the step metrics and `log_every_n_steps` all count updates
    (roadmap P2-3, `docs/adr/0031-mixed-precision-gradient-accumulation-and-compile.md`).

    Attributes:
        should_stop: `False` by default (and reset to `False` at the
            start of every `fit()` call); a callback's `onEpochEnd`
            (e.g. `training.callbacks.early_stopping.EarlyStopping`)
            sets this to `True` to end the run after the current
            epoch's callbacks have all finished, rather than raising
            or returning a sentinel `fit()` would have to interpret.
        scaler: The `torch.amp.GradScaler` of a run with `mixed_precision="fp16"`, else
            `None` (`bf16` has the range of `float32` and needs none).
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
        reconstruction_reduction: str | Mapping[str, str] = DEFAULT_REDUCTION,
        mixed_precision: str | None = None,
        accumulate_grad_batches: int = 1,
        compile_model: bool = False,
        compile_kwargs: dict[str, Any] | None = None,
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
                training-loop event (`training/callbacks/`). Several
                may be active at once (e.g. a metrics logger, a
                checkpointer, early stopping, and a learning-rate
                scheduler, all together). `None` (default) means no
                callbacks. `global_vae.config.training.buildCallbacksFromConfig`
                builds this list from `TrainingConfig.callbacks` (and
                `TrainingConfig.loggers`) for a config-driven run.
            log_every_n_steps: How often (in global steps) `Trainer`
                logs step-level progress via the standard `logging`
                module (never `print`, spec §10). Every epoch boundary
                is always logged regardless of this value.
            reconstruction_reduction: Forwarded to
                `computeTotalReconstructionLoss` as `reduction`: `"mean"`
                (default, the per-element mean, as before this argument
                existed) or `"sum"` (the per-sample sum, averaged over the
                batch), as one name for every modality or a per-modality
                mapping. It applies to the training loss and to the
                validation loss of `evaluate`, so both curves keep one scale.
                See `docs/adr/0029-loss-scale-convention.md`.
            mixed_precision: `None` (default, full `float32`: nothing changes), `"bf16"` or
                `"fp16"`. The forward pass of the model runs under `torch.autocast` on
                `device` in that dtype, and the losses are computed in `float32` from its
                outputs (`_castOutputsToFloat32`). `"fp16"` also uses a `GradScaler`
                (`self.scaler`), whose state is saved in the checkpoint. Applies to the
                validation pass too, so both curves keep one scale.
            accumulate_grad_batches: Number of batches whose gradients are added before one
                optimizer step (default `1`: a step per batch, as before). Each batch's loss
                is divided by this number, so the step uses the mean gradient, which is the
                gradient of a batch `N` times larger when the loss is a mean over samples.
                A last group shorter than `N` at the end of an epoch is rescaled to the mean
                of its own batches and stepped. Gradient clipping applies to the accumulated
                gradient.
            compile_model: If `True`, the forward pass goes through `torch.compile(model)`.
                The module in `self.model` is the original one, so `state_dict` keys and
                checkpoints are the same with and without it. A modality-dropout pass with
                another set of keys compiles another graph.
            compile_kwargs: Keyword arguments for `torch.compile` (for example
                `{"mode": "reduce-overhead"}`, or `{"backend": "eager"}` to trace without a
                C++ compiler). Ignored when `compile_model` is `False`.

        Raises:
            ValueError: If `modality_dropout_p` is not in `[0, 1]`, if
                `grad_clip_norm` is given and not positive, if
                `log_every_n_steps` is not positive, if
                `reconstruction_reduction` is not a supported reduction
                (`global_vae.losses.reconstruction.validateReduction`), if
                `mixed_precision` is not `None`, `"bf16"` or `"fp16"`, or if
                `accumulate_grad_batches` is below 1.
        """
        if not (0.0 <= modality_dropout_p <= 1.0):
            raise ValueError(f"modality_dropout_p must be in [0, 1], got {modality_dropout_p}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive when given, got {grad_clip_norm}.")
        if log_every_n_steps <= 0:
            raise ValueError(f"log_every_n_steps must be positive, got {log_every_n_steps}.")
        validateReduction(reconstruction_reduction, "reconstruction_reduction")
        if mixed_precision is not None and mixed_precision not in _AUTOCAST_DTYPES:
            raise ValueError(
                f"mixed_precision must be None or one of {list(SUPPORTED_MIXED_PRECISIONS)}, "
                f"got {mixed_precision!r}."
            )
        if accumulate_grad_batches < 1:
            raise ValueError(
                f"accumulate_grad_batches must be at least 1, got {accumulate_grad_batches}."
            )

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
        self.reconstruction_reduction = reconstruction_reduction
        self.beta = beta
        self.beta_schedules = beta_schedules or {}
        self.modality_dropout_p = modality_dropout_p
        self.grad_clip_norm = grad_clip_norm
        self.callbacks = list(callbacks) if callbacks else []
        self.log_every_n_steps = log_every_n_steps
        self.mixed_precision = mixed_precision
        self.accumulate_grad_batches = accumulate_grad_batches
        self.compile_model = compile_model
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler(self.device.type) if mixed_precision == "fp16" else None
        )
        self._forward: Callable[..., dict[str, Any]] = (
            torch.compile(self.model, **(compile_kwargs or {})) if compile_model else self.model
        )

        self.global_step = 0
        self.start_epoch = 0
        self.history: list[dict[str, float]] = []
        self.should_stop = False

        logger.info("Trainer initialized on device '%s'.", self.device)

    @property
    def logs_learning_rate(self) -> bool:
        """Whether this run records the learning rate in its metrics.

        `True` when at least one callback sets `TrainerCallback.acts_on_learning_rate`
        (ADR 0028). `fitEpoch` then adds `"lr"` to every step's metrics, and the epoch metrics
        carry `"train/lr"`.
        """
        return any(callback.acts_on_learning_rate for callback in self.callbacks)

    def computeLosses(
        self, batch: dict[str, torch.Tensor], step: int, apply_dropout: bool = True
    ) -> StepLosses:
        """Run one forward pass and compute the weighted total loss for `batch`.

        Shared by `fitEpoch` (which calls `.backward()` on the result)
        and `evaluate` (which does not), so the forward/loss logic
        exists exactly once.

        Args:
            batch: Modality name -> raw tensor, already moved to
                `self.device`. Used as the reconstruction target
                unchanged; the subset of it naming a registered encoder
                (`GlobalVae.selectEncoderInputs`, ADR 0019) is used,
                after modality dropout if enabled, as the encoder
                input. A key with no matching encoder is simply never
                encoded, only ever matched as its own decoder's target
                (the translation-style `image_in -> image_out` case;
                see this module's own docstring).
            step: Global step index used to resolve any per-latent-space
                beta schedule for this call. `evaluate` passes the
                current `self.global_step` without advancing it, so a
                validation pass never itself moves a schedule forward.
            apply_dropout: If `True` (default), `self.modality_dropout_p`
                is applied to the encoder input, exactly as during
                training. `evaluate()` passes `False`: modality dropout
                is a *training-time* robustness technique (spec §5), and
                applying it during validation would make the reported
                validation loss depend on which modalities the random
                dropout happened to keep for that call, i.e. a
                different, non-reproducible number on every call even
                for the exact same model and data.

        Returns:
            `StepLosses` still attached to the autograd graph.

        Raises:
            ValueError: If `batch` is empty, if none of its keys names
                one of `self.model`'s encoders (via
                `GlobalVae.selectEncoderInputs`), or (via
                `computeTotalReconstructionLoss`) if a decoder's
                reconstruction and its target in `batch` differ in
                shape, e.g. a `(B, 1, H, W)` target against the 2D
                decoder's `(B, H, W)` output.
            KeyError: Via `computeTotalReconstructionLoss`, if `batch`
                is missing a target for a decoder that produced a
                reconstruction.
        """
        if not batch:
            raise ValueError("Trainer.computeLosses received an empty batch.")

        encoder_inputs = self.model.selectEncoderInputs(batch)
        inputs = self._applyModalityDropout(encoder_inputs) if apply_dropout else encoder_inputs
        with self._autocast():
            outputs = self._forward(inputs)
        if self.mixed_precision is not None:
            outputs = _castOutputsToFloat32(outputs)

        reconstruction_loss = computeTotalReconstructionLoss(
            outputs["reconstructions"],
            batch,
            weights=self.reconstruction_weights,
            loss_fn=self.reconstruction_loss_fn,
            reduction=self.reconstruction_reduction,
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
            This epoch's metrics, averaged over every optimizer step
            (each step's value being the mean over the batches it
            accumulated), keys
            prefixed `"train/"` (e.g. `"train/loss/total"`). When a
            callback acts on the learning rate (`logs_learning_rate`),
            also `"train/lr"`: the mean of the rate each step trained with
            (`"train/lr/group<i>"` with several parameter groups).

        Raises:
            ValueError: If `dataloader` yields no batches.
        """
        self.model.train()
        running_totals: dict[str, float] = {}
        num_steps = 0
        accumulate = self.accumulate_grad_batches
        pending = 0
        group_totals: dict[str, float] = {}

        for callback in self.callbacks:
            callback.onEpochBegin(self, epoch)

        for raw_batch in dataloader:
            batch = self._moveBatchToDevice(raw_batch)
            if pending == 0:
                self.optimizer.zero_grad()
                group_totals = {}
            losses = self.computeLosses(batch, self.global_step)
            self._backward(losses.total / accumulate if accumulate > 1 else losses.total)
            for key, value in losses.asMetrics().items():
                group_totals[key] = group_totals.get(key, 0.0) + value
            pending += 1

            if pending == accumulate:
                self._finishStep(epoch, group_totals, pending, running_totals)
                num_steps += 1
                pending = 0

        if pending > 0:
            # The epoch ended inside a group: the gradient holds `pending / accumulate` of a
            # mean, so scale it back up to the mean of the batches it does hold.
            self._scaleGradients(accumulate / pending)
            self._finishStep(epoch, group_totals, pending, running_totals)
            num_steps += 1

        if num_steps == 0:
            raise ValueError(
                "fitEpoch received an empty dataloader: at least one batch is required."
            )

        return {f"train/{key}": value / num_steps for key, value in running_totals.items()}

    def _autocast(self) -> AbstractContextManager[Any]:
        """Return the context the model's forward pass runs in.

        Returns:
            `torch.autocast` on this trainer's device in `mixed_precision`'s dtype, or a
            do-nothing context when `mixed_precision` is `None`.
        """
        if self.mixed_precision is None:
            return contextlib.nullcontext()
        return torch.autocast(
            device_type=self.device.type, dtype=_AUTOCAST_DTYPES[self.mixed_precision]
        )

    def _backward(self, loss: torch.Tensor) -> None:
        """Backpropagate `loss`, scaled first when a gradient scaler is in use.

        Args:
            loss: The scalar to differentiate (already divided by `accumulate_grad_batches`).
        """
        backward(self.scaler.scale(loss) if self.scaler is not None else loss)

    def _scaleGradients(self, factor: float) -> None:
        """Multiply every gradient of the optimizer's parameters by `factor`, in place.

        Args:
            factor: The multiplier.
        """
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)

    def _finishStep(
        self,
        epoch: int,
        group_totals: dict[str, float],
        num_batches: int,
        running_totals: dict[str, float],
    ) -> None:
        """Clip, take the optimizer step, record its metrics and notify the callbacks.

        Args:
            epoch: Index of the running epoch.
            group_totals: Each loss summed over the `num_batches` batches of this step.
            num_batches: How many batches this step accumulated.
            running_totals: The epoch's running sums of step metrics, updated in place.
        """
        step_metrics = {key: value / num_batches for key, value in group_totals.items()}
        if self.logs_learning_rate:
            # Read just before the step: this is the rate the step trains with, so a rate a
            # callback changes in `onStepEnd` or `onEpochEnd` shows from the next step on.
            step_metrics.update(learningRateMetrics(self.optimizer))

        if self.scaler is not None:
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

        for key, value in step_metrics.items():
            running_totals[key] = running_totals.get(key, 0.0) + value

        if self.global_step % self.log_every_n_steps == 0:
            logger.info(
                "epoch %d step %d: %s",
                epoch,
                self.global_step,
                _formatMetrics(step_metrics),
            )
        for callback in self.callbacks:
            callback.onStepEnd(self, self.global_step, step_metrics)

        self.global_step += 1

    def evaluate(self, dataloader: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Run one pass over `dataloader` without gradient updates.

        Usable both for periodic validation during `fit` and for a
        standalone evaluation pass after training. Never applies
        modality dropout, regardless of `self.modality_dropout_p`:
        dropout is a training-time robustness technique (spec §5), and
        applying it here would make the returned loss a random function
        of which modalities the dropout happened to keep on this call,
        rather than a reproducible measurement of the model on
        `dataloader`. Two calls with the same model and the same
        (non-shuffled) `dataloader` therefore return the exact same
        metrics.

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
                losses = self.computeLosses(batch, self.global_step, apply_dropout=False)
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

        self.should_stop = False
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
                    _formatMetrics(epoch_metrics),
                )
                self.start_epoch = epoch + 1
                for callback in self.callbacks:
                    callback.onEpochEnd(self, epoch, epoch_metrics)

                if self.should_stop:
                    logger.info(
                        "epoch %d: a callback set trainer.should_stop; stopping early.", epoch
                    )
                    break
        finally:
            for callback in self.callbacks:
                callback.onTrainEnd(self)

        return self.history

    def saveCheckpoint(self, path: str | Path, config: Any = None) -> None:
        """Save this trainer's full state (model, optimizer, step/epoch, history) to `path`.

        Thin wrapper around `training.checkpoint.saveCheckpoint`, using
        this instance's own model/optimizer/step/epoch/history; see
        that function for the exact checkpoint contents.

        Args:
            path: Destination file path. Parent directories are
                created if they do not already exist.
            config: Arbitrary, picklable snapshot of whatever
                configuration produced this model/trainer (spec §10:
                "config snapshotted with every run"). `None` (default)
                saves no config.
        """
        saveCheckpointFile(
            path,
            model=self.model,
            optimizer=self.optimizer,
            global_step=self.global_step,
            start_epoch=self.start_epoch,
            history=self.history,
            config=config,
            callbacks_state=[
                {"class": type(callback).__name__, "state": callback.stateDict()}
                for callback in self.callbacks
            ],
            scaler_state_dict=self.scaler.state_dict() if self.scaler is not None else None,
        )

    def loadCheckpoint(
        self,
        path: str | Path,
        restore_optimizer: bool = True,
        strict: bool = True,
        weights_only: bool = True,
    ) -> Any:
        """Restore this trainer's state from a checkpoint saved by `saveCheckpoint`.

        `self.global_step`, `self.start_epoch`, and `self.history` are
        overwritten from the checkpoint, so a subsequent `fit()` call
        continues exactly where the checkpointed run left off (spec
        §10, ADR 0005). RNG state (Python/NumPy/PyTorch) is restored
        too, so resumed training continues drawing from the same
        random sequence it would have without the interruption.

        Args:
            path: Checkpoint file path.
            restore_optimizer: If `True` (default), also restore the
                optimizer's own state (e.g. Adam's moment estimates).
                Set to `False` to load only the model weights, keeping
                this trainer's freshly-constructed optimizer state
                (e.g. to resume with a different learning rate).
            strict: Forwarded to `model.load_state_dict`.
            weights_only: Forwarded to `training.checkpoint.loadCheckpoint`: `True`
                (default) reads the file with the restricted unpickler; pass `False` only
                for a file you trust that it refuses (ADR 0034).

        The state of the callbacks (`TrainerCallback.stateDict`) and of the gradient
        scaler is restored too: each saved callback state goes to the callback of the
        same class, in order, so a resumed run keeps its patience counters and best
        values. A saved state with no matching callback is logged and dropped.

        Returns:
            The `config` snapshot stored in the checkpoint (whatever
            shape was passed to `saveCheckpoint`; `None` if none was
            given), for the caller to inspect or reuse.
        """
        metadata = loadCheckpointFile(
            path,
            model=self.model,
            optimizer=self.optimizer if restore_optimizer else None,
            map_location=self.device,
            strict=strict,
            weights_only=weights_only,
        )
        self.global_step = metadata.global_step
        self.start_epoch = metadata.start_epoch
        self.history = metadata.history
        if self.scaler is not None and metadata.scaler_state_dict is not None:
            self.scaler.load_state_dict(metadata.scaler_state_dict)
        self._restoreCallbacksState(metadata.callbacks_state)
        return metadata.config

    def _restoreCallbacksState(self, saved: list[dict[str, Any]]) -> None:
        """Hand each saved callback state to the callback of the same class.

        Args:
            saved: `CheckpointMetadata.callbacks_state`: `{"class": name, "state": dict}`
                entries in the saving trainer's callback order.
        """
        by_class: dict[str, list[dict[str, Any]]] = {}
        for entry in saved:
            by_class.setdefault(entry["class"], []).append(entry["state"])
        for callback in self.callbacks:
            queue = by_class.get(type(callback).__name__)
            if queue:
                callback.loadStateDict(queue.pop(0))
        dropped = sorted(name for name, queue in by_class.items() if any(queue))
        if dropped:
            logger.warning(
                "The checkpoint holds callback state for %s, but this trainer has no such "
                "callback (or fewer of them); that state is dropped.",
                dropped,
            )

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
            batch: Modality name -> raw tensor, already restricted to
                the model's encoder inputs by the caller
                (`GlobalVae.selectEncoderInputs`); a decoder-only
                target key never reaches this method, so it can never
                be the one modality "kept" by accident. The original,
                unfiltered batch remains available to the caller for
                the reconstruction target regardless of what this
                method returns.

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
