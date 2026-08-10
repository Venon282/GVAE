"""Checkpoint save/restore for model + optimizer + config (spec §10: "config snapshotted
with every run", and the practical need to re-run eval/visualization without retraining).

A checkpoint file (`torch.save`/`torch.load`, PyTorch's own standard
serialization) bundles everything needed to either resume training
exactly, or to load a trained model for evaluation/visualization
without any training state at all:

- `model_state_dict`: always present.
- `optimizer_state_dict`: only present if an optimizer was passed to
  `saveCheckpoint` (omit it entirely for an eval-only checkpoint; there
  is nothing to resume-train, so no optimizer momentum to carry).
- `global_step`, `start_epoch`, `history`: `Trainer`'s own bookkeeping,
  so a resumed `Trainer` continues exactly where it left off (matching
  `docs/adr/0005-training-loop.md`'s note that this state was kept
  simple and instance-level specifically so a checkpoint feature would
  have something clean to serialize).
- `config`: an arbitrary, picklable snapshot of whatever configuration
  produced this model/trainer. This module does not define or enforce
  a config schema (spec §11: the Hydra/Pydantic config binding is
  still an open question); it only provides the slot to store and
  retrieve one, restored unchanged.
- `rng_state`: Python/NumPy/PyTorch RNG state at save time (spec §10's
  reproducibility goal extended to resumed runs: a training run
  resumed from a checkpoint continues drawing from the same random
  sequence it would have without the interruption, rather than
  silently re-seeding from wherever the process's RNGs happen to be).

Security note: like any `torch.save`/`torch.load` file, a checkpoint is
a pickle under the hood. Only load checkpoints from sources you trust,
the same caution PyTorch's own documentation gives for `torch.load`.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.optim import Optimizer

import global_vae
from global_vae.training.callbacks import TrainerCallback

if TYPE_CHECKING:
    from global_vae.training.trainer import Trainer

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Non-tensor bookkeeping restored from a checkpoint by `loadCheckpoint`.

    Attributes:
        global_step: `Trainer.global_step` at save time, or `0` if the
            checkpoint was saved without a step (e.g. saved outside a
            `Trainer`).
        start_epoch: `Trainer.start_epoch` at save time, or `0`.
        history: `Trainer.history` at save time, or `[]`.
        config: Whatever was passed as `config` to `saveCheckpoint`,
            unchanged. `None` if no config was given.
        global_vae_version: `global_vae.__version__` at save time, for
            bookkeeping across framework versions. `None` for
            checkpoints saved before this field existed.
        rng_state_restored: Whether `loadCheckpoint` actually restored
            Python/NumPy/PyTorch RNG state (`True`), or whether the
            checkpoint had none to restore, or the caller asked not to
            (`False`).
    """

    global_step: int = 0
    start_epoch: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    config: Any = None
    global_vae_version: str | None = None
    rng_state_restored: bool = False


def _captureRngState() -> dict[str, Any]:
    """Snapshot every RNG this codebase's randomness can come from (mirrors `utils/seed.py`).

    Returns:
        A dict suitable for storing in a checkpoint and later passing
        to `_restoreRngState`.
    """
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if np is not None:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restoreRngState(state: dict[str, Any]) -> None:
    """Restore an RNG snapshot captured by `_captureRngState`.

    Args:
        state: As returned by `_captureRngState`. Missing keys (e.g. a
            checkpoint saved on a machine without NumPy, or without
            CUDA) are simply skipped rather than raised on, since a
            partial restoration is still strictly better than none.
    """
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def saveCheckpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    global_step: int = 0,
    start_epoch: int = 0,
    history: list[dict[str, float]] | None = None,
    config: Any = None,
    include_rng_state: bool = True,
) -> None:
    """Save model (+ optionally optimizer, step/epoch/history, config) to `path`.

    Args:
        path: Destination file path. Parent directories are created if
            they do not already exist.
        model: Any `nn.Module` (typically a `GlobalVae`); only
            `model.state_dict()` is saved, not the module object
            itself, so loading never depends on unpickling this
            framework's classes.
        optimizer: If given, `optimizer.state_dict()` is saved too
            (needed to resume training with momentum/moment estimates
            intact). Omit for an eval-only checkpoint: there is no
            training to resume, so no optimizer state to carry.
        global_step: `Trainer.global_step` at save time, if any.
        start_epoch: `Trainer.start_epoch` at save time, if any.
        history: `Trainer.history` at save time, if any.
        config: Arbitrary, picklable snapshot of whatever configuration
            produced this model (spec §10: "config snapshotted with
            every run"). This framework does not dictate its shape:
            pass a `dict`, a dataclass, a Pydantic model, whatever your
            own setup uses. `None` (default) saves no config.
        include_rng_state: If `True` (default), also save
            Python/NumPy/PyTorch RNG state, so a resumed run continues
            the same random sequence instead of silently re-seeding.
    """
    checkpoint: dict[str, Any] = {
        "global_vae_version": global_vae.__version__,
        "model_state_dict": model.state_dict(),
        "global_step": global_step,
        "start_epoch": start_epoch,
        "history": history or [],
        "config": config,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if include_rng_state:
        checkpoint["rng_state"] = _captureRngState()

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, resolved_path)
    logger.info(
        "Saved checkpoint to '%s' (step=%d, epoch=%d).", resolved_path, global_step, start_epoch
    )


def loadCheckpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device | None = None,
    restore_rng_state: bool = True,
    strict: bool = True,
) -> CheckpointMetadata:
    """Load a checkpoint saved by `saveCheckpoint` into `model` (and optionally `optimizer`).

    Args:
        path: Checkpoint file path.
        model: Model to load weights into, in place. Must already have
            the exact same architecture the checkpoint was saved from
            (this framework does not reconstruct a `GlobalVae`'s
            architecture from a checkpoint: encoder/decoder/routing-graph
            choices are construction-time decisions the caller owns).
        optimizer: If given, its state is restored from the
            checkpoint's `optimizer_state_dict`. Leave as `None` to
            load model weights only (e.g. for evaluation, or to resume
            training with a freshly-constructed optimizer instead of
            the original one's momentum).
        map_location: Forwarded to `torch.load`; use this to load a
            checkpoint saved on a GPU machine onto a CPU-only one, or
            vice versa.
        restore_rng_state: If `True` (default) and the checkpoint has
            an RNG snapshot, restore it (see `_restoreRngState`).
        strict: Forwarded to `model.load_state_dict`; `False` allows
            loading into a model whose parameter names are a superset
            or subset of the checkpoint's (e.g. after adding a new,
            optional submodule), at the cost of losing the safety net
            that catches an accidental architecture mismatch.

    Returns:
        `CheckpointMetadata` holding everything besides the tensors
        that were just loaded in place.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `optimizer` is given but the checkpoint has no
            `optimizer_state_dict` (it was saved without one).
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"No checkpoint found at '{resolved_path}'.")

    # weights_only=False: this checkpoint intentionally carries non-tensor
    # metadata (config, RNG state) alongside the tensors, so the safer
    # tensors-only loading mode does not apply here (see this module's
    # docstring for the accompanying trust caveat).
    checkpoint = torch.load(resolved_path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint '{resolved_path}' has no optimizer state (it was saved without "
                f"an optimizer). Pass optimizer=None to load model weights only."
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    rng_state_restored = False
    if restore_rng_state and "rng_state" in checkpoint:
        _restoreRngState(checkpoint["rng_state"])
        rng_state_restored = True

    logger.info(
        "Loaded checkpoint from '%s' (step=%d, epoch=%d).",
        resolved_path,
        checkpoint.get("global_step", 0),
        checkpoint.get("start_epoch", 0),
    )
    return CheckpointMetadata(
        global_step=checkpoint.get("global_step", 0),
        start_epoch=checkpoint.get("start_epoch", 0),
        history=checkpoint.get("history", []),
        config=checkpoint.get("config"),
        global_vae_version=checkpoint.get("global_vae_version"),
        rng_state_restored=rng_state_restored,
    )


class CheckpointCallback(TrainerCallback):
    """Periodically saves a training checkpoint via `onEpochEnd`, for resuming an interrupted run.

    This callback's job is narrower than it might first look: it
    exists to let a **long training run be resumed close to where it
    was** after an interruption (a crash, a cluster preemption, a
    manual stop), with `global_step`, optimizer momentum, and RNG state
    intact, so the run does not have to restart from scratch. It is
    **not** a way to recover the best model for evaluation: the most
    recently saved epoch is not necessarily the best one (validation
    performance can get worse in later epochs), and `keep_last_n`
    prunes by save order, not by quality. Resuming from a "best" epoch
    instead of the most recent one would also throw away every epoch
    of progress made after it, defeating the point of resuming at all.

    For "give me the best model to evaluate or visualize", use
    `BestCheckpointCallback` instead (or alongside this one; they solve
    different problems and are not mutually exclusive).

    The running example `training/callbacks.py`'s own docstring already
    used ("a checkpointer only overrides `onEpochEnd`"), made concrete:
    delegates to `Trainer.saveCheckpoint` (which in turn calls
    `saveCheckpoint` above with the trainer's own model, optimizer,
    step, epoch, and history), so the checkpoint format is identical
    whether saved through this callback or called directly.
    """

    def __init__(
        self,
        directory: str | Path,
        every_n_epochs: int = 1,
        config: Any = None,
        keep_last_n: int | None = None,
        filename_pattern: str = "checkpoint_epoch_{epoch:04d}.pt",
    ) -> None:
        """Initialize the callback.

        Args:
            directory: Directory to save checkpoints into (created if
                missing).
            every_n_epochs: Save every `N` epochs (`1`, the default,
                saves after every epoch).
            config: Forwarded unchanged to `Trainer.saveCheckpoint`
                every time this callback saves (spec §10: "config
                snapshotted with every run").
            keep_last_n: If given, delete older checkpoints beyond the
                most recently saved `keep_last_n`, so disk usage does
                not grow unbounded over a long run. `None` (default)
                keeps every checkpoint ever saved by this callback.
                Deletes by save order (oldest first); keeping the best
                `N` by some validation metric instead is a natural
                future extension, not built here, since "best" requires
                choosing a metric and a comparison direction this
                callback has no way to know generically.
            filename_pattern: `str.format` pattern for each
                checkpoint's filename, receiving `epoch` as a keyword
                argument (the epoch index that just finished, matching
                `TrainerCallback.onEpochEnd`'s own `epoch` argument).

        Raises:
            ValueError: If `every_n_epochs` is not positive, or if
                `keep_last_n` is given and not positive.
        """
        if every_n_epochs <= 0:
            raise ValueError(f"every_n_epochs must be positive, got {every_n_epochs}.")
        if keep_last_n is not None and keep_last_n <= 0:
            raise ValueError(f"keep_last_n must be positive when given, got {keep_last_n}.")

        self.directory = Path(directory)
        self.every_n_epochs = every_n_epochs
        self.config = config
        self.keep_last_n = keep_last_n
        self.filename_pattern = filename_pattern
        self._saved_paths: list[Path] = []

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Save a checkpoint if `epoch` lands on an `every_n_epochs` boundary.

        Args:
            trainer: The `Trainer` instance running this training run.
            epoch: Index of the epoch that just finished (0-based).
            metrics: Unused; present only to match `TrainerCallback`'s
                signature.
        """
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        path = self.directory / self.filename_pattern.format(epoch=epoch)
        trainer.saveCheckpoint(path, config=self.config)
        self._saved_paths.append(path)

        if self.keep_last_n is not None:
            while len(self._saved_paths) > self.keep_last_n:
                stale_path = self._saved_paths.pop(0)
                stale_path.unlink(missing_ok=True)


class BestCheckpointCallback(TrainerCallback):
    """Saves a checkpoint only when a monitored metric improves, via `onEpochEnd`.

    This is the "give me the best model" callback: unlike
    `CheckpointCallback` (which saves on a schedule, for resuming an
    interrupted run), this one saves purely based on whether
    `monitor` improved this epoch, always overwriting the **same**
    file, so `path` is always exactly the best model seen so far, no
    pruning logic needed. Load it at any time via `loadCheckpoint(path,
    model=...)` (or `Trainer.loadCheckpoint(path)`) to evaluate or
    visualize the best model without retraining.

    Typically monitors a validation metric (e.g. `"val/loss/total"`,
    which requires `val_dataloader` to be passed to `Trainer.fit`).
    Monitoring a training metric instead is allowed but usually less
    useful for model selection: training loss tends to keep improving
    even as the model overfits, so "best training loss" is often close
    to just "the last epoch".
    """

    def __init__(
        self,
        path: str | Path,
        monitor: str = "val/loss/total",
        mode: str = "min",
        config: Any = None,
    ) -> None:
        """Initialize the callback.

        Args:
            path: File path for the single best-so-far checkpoint.
                Every improvement overwrites this same file.
            monitor: Metric key to track, as it appears in the epoch
                metrics dict `Trainer.fit` builds (e.g.
                `"val/loss/total"`, `"train/loss/reconstruction"`).
            mode: `"min"` (default; lower is better, e.g. a loss) or
                `"max"` (higher is better, e.g. an accuracy or a
                custom metric a `TrainerCallback` might add).
            config: Forwarded to `Trainer.saveCheckpoint` every time
                this callback saves.

        Raises:
            ValueError: If `mode` is not `"min"` or `"max"`.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")

        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode
        self.config = config
        self.best_value: float | None = None

    def onEpochEnd(self, trainer: "Trainer", epoch: int, metrics: dict[str, float]) -> None:
        """Save a checkpoint if `self.monitor` improved this epoch.

        Args:
            trainer: The `Trainer` instance running this training run.
            epoch: Index of the epoch that just finished (0-based).
            metrics: This epoch's metrics; must contain `self.monitor`.

        Raises:
            KeyError: If `self.monitor` is not present in `metrics`
                (e.g. monitoring a `"val/..."` key without passing
                `val_dataloader` to `Trainer.fit`).
        """
        if self.monitor not in metrics:
            raise KeyError(
                f"BestCheckpointCallback is monitoring '{self.monitor}', but it is not present "
                f"in this epoch's metrics ({sorted(metrics)}). If you are monitoring a "
                f"'val/...' key, make sure Trainer.fit was called with a val_dataloader."
            )

        value = metrics[self.monitor]
        improved = self.best_value is None or (
            value < self.best_value if self.mode == "min" else value > self.best_value
        )
        if not improved:
            return

        self.best_value = value
        trainer.saveCheckpoint(self.path, config=self.config)
        logger.info(
            "New best %s=%.6f at epoch %d, saved to '%s'.", self.monitor, value, epoch, self.path
        )
