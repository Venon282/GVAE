"""Checkpoint save/restore for model + optimizer + config.

See spec §10: "config snapshotted with every run", and the practical need to re-run
eval/visualization without retraining.

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

This module owns only the checkpoint *file format*. The two
`TrainerCallback`s that call into it on a schedule (`CheckpointCallback`)
or on metric improvement (`BestCheckpointCallback`) live in
`training/callbacks/checkpoint.py` and `training/callbacks/best_checkpoint.py`
respectively, one class per file (spec §10 "Modularity", ADR 0022); import
them from there, not from here.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

import global_vae

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
