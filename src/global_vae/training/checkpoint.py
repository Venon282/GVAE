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
- `callbacks_state` and `scaler_state_dict`: only present when a `Trainer` saved
  the file and it had callbacks with state (`TrainerCallback.stateDict`) or a
  gradient scaler (mixed precision with `fp16`), so a resumed run continues them too
  (roadmap P2-3, `docs/adr/0032-resume-from-the-training-script.md`).

Every entry is made of tensors, numbers, strings, `None` and plain containers
(`dict`, `list`, `tuple`), never of an object of this framework or of NumPy. That is what
lets `loadCheckpoint` read the file with `torch.load(weights_only=True)`, PyTorch's
restricted unpickler, which is the default
(`docs/adr/0034-safe-checkpoint-loading-by-default.md`). The one thing the framework does
not control is the `config` snapshot a caller passes: a dict of plain values (what
`OmegaConf.to_container` gives, which is what `scripts/train.py` saves) loads in the safe
mode. A dataclass instance is turned into such a dict (`dataclasses.asdict`) when the file
is written, so `config=experiment_config` works; the file then holds the dict, and
`loadCheckpoint` hands back the dict, not the dataclass.

Security note: like any `torch.save`/`torch.load` file, a checkpoint is a pickle under
the hood. `loadCheckpoint` uses the restricted unpickler by default, which refuses to
construct arbitrary objects. Pass `weights_only=False` only for a file you trust, for
example one saved by an earlier version of this framework (whose RNG state holds a NumPy
array), the same caution PyTorch's own documentation gives for `torch.load`.

This module owns only the checkpoint *file format*. The two
`TrainerCallback`s that call into it on a schedule (`CheckpointCallback`)
or on metric improvement (`BestCheckpointCallback`) live in
`training/callbacks/checkpoint.py` and `training/callbacks/best_checkpoint.py`
respectively, one class per file (spec §10 "Modularity", ADR 0022); import
them from there, not from here.
"""

import logging
import pickle
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, cast

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
        callbacks_state: One `{"class": name, "state": dict}` entry per callback of the
            `Trainer` that saved the file, in its order (`Trainer.saveCheckpoint`). `[]`
            for a checkpoint saved without any, or before this field existed.
        scaler_state_dict: The gradient scaler's state when the saving `Trainer` used
            `fp16` mixed precision, otherwise `None`.
    """

    global_step: int = 0
    start_epoch: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    config: Any = None
    global_vae_version: str | None = None
    rng_state_restored: bool = False
    callbacks_state: list[dict[str, Any]] = field(default_factory=list)
    scaler_state_dict: dict[str, Any] | None = None


def _captureRngState() -> dict[str, Any]:
    """Snapshot every RNG this codebase's randomness can come from (mirrors `utils/seed.py`).

    Returns:
        A dict suitable for storing in a checkpoint and later passing
        to `_restoreRngState`.
    """
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if np is not None:
        # NumPy hands back an array of 624 words; a plain list keeps the file readable by the
        # restricted unpickler (`weights_only=True`), which refuses `numpy.ndarray`.
        name, keys, position, has_gauss, cached_gaussian = cast(
            "tuple[str, Any, int, int, float]", np.random.get_state(legacy=True)
        )
        state["numpy"] = (name, keys.tolist(), position, has_gauss, cached_gaussian)
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
        name, keys, position, has_gauss, cached_gaussian = state["numpy"]
        # `keys` is a list in a file written now, an array in one written by an earlier version.
        np.random.set_state(
            (name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian)
        )
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _toPlainConfig(config: Any) -> Any:
    """Turn a dataclass instance into a dict of plain values, leave anything else alone.

    Args:
        config: The `config` argument of `saveCheckpoint`.

    Returns:
        `dataclasses.asdict(config)` for a dataclass instance, `config` otherwise.
    """
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    return config


def saveCheckpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    global_step: int = 0,
    start_epoch: int = 0,
    history: list[dict[str, float]] | None = None,
    config: Any = None,
    include_rng_state: bool = True,
    callbacks_state: list[dict[str, Any]] | None = None,
    scaler_state_dict: dict[str, Any] | None = None,
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
        config: Snapshot of whatever configuration produced this model (spec §10: "config
            snapshotted with every run"). This framework does not dictate its shape. For the
            file to be readable in the safe mode (`weights_only=True`, ADR 0034) it must be
            made of plain values: a `dict`, `list`, `tuple`, `str`, number, `bool` or `None`.
            A dataclass instance is converted with `dataclasses.asdict` before it is written,
            so it is stored, and loaded back, as a dict. Any other object is written as it
            is, and the file then needs `weights_only=False` to be read. `None` (default)
            saves no config.
        include_rng_state: If `True` (default), also save
            Python/NumPy/PyTorch RNG state, so a resumed run continues
            the same random sequence instead of silently re-seeding.
        callbacks_state: The state of the callbacks of the saving `Trainer`
            (`Trainer.saveCheckpoint` builds it). Plain values only, so that the file stays
            readable in the safe mode. Omitted from the file when empty or `None`.
        scaler_state_dict: `GradScaler.state_dict()` of the saving `Trainer`, when it used
            `fp16`. Omitted from the file when `None`.
    """
    checkpoint: dict[str, Any] = {
        "global_vae_version": global_vae.__version__,
        "model_state_dict": model.state_dict(),
        "global_step": global_step,
        "start_epoch": start_epoch,
        "history": history or [],
        "config": _toPlainConfig(config),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if callbacks_state:
        checkpoint["callbacks_state"] = callbacks_state
    if scaler_state_dict is not None:
        checkpoint["scaler_state_dict"] = scaler_state_dict
    if include_rng_state:
        checkpoint["rng_state"] = _captureRngState()

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, resolved_path)
    logger.info(
        "Saved checkpoint to '%s' (step=%d, epoch=%d).", resolved_path, global_step, start_epoch
    )


def upgradeCheckpoint(path: str | Path, output_path: str | Path | None = None) -> Path:
    """Rewrite a checkpoint from an earlier version so that the safe mode can read it.

    Files saved before ADR 0034 hold the NumPy RNG state as an array, which
    `torch.load(weights_only=True)` refuses. This reads `path` with `weights_only=False`,
    which runs pickle code: **only call it on a file you trust**. It turns the array into a
    list, as `saveCheckpoint` now writes it, and saves the result. Nothing else changes.

    Args:
        path: The old checkpoint.
        output_path: Where to write the upgraded file. `None` (default) replaces `path`.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"No checkpoint found at '{resolved_path}'.")
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    numpy_state = checkpoint.get("rng_state", {}).get("numpy")
    if numpy_state is not None and not isinstance(numpy_state[1], list):
        name, keys, position, has_gauss, cached_gaussian = numpy_state
        checkpoint["rng_state"]["numpy"] = (
            name,
            keys.tolist(),
            position,
            has_gauss,
            cached_gaussian,
        )
    destination = Path(output_path) if output_path is not None else resolved_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    try:
        torch.load(destination, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        logger.warning(
            "Upgraded checkpoint '%s' is still not readable in the safe mode: its `config` "
            "snapshot is not made of plain values (dict, list, tuple, str, number, None).",
            destination,
        )
    return destination


def loadCheckpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device | None = None,
    restore_rng_state: bool = True,
    strict: bool = True,
    weights_only: bool = True,
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
        weights_only: If `True` (default), read the file with PyTorch's restricted
            unpickler (`torch.load(weights_only=True)`), which only builds tensors, numbers,
            strings and plain containers and refuses any other object, so that a file from
            an untrusted source cannot run code on load. Every checkpoint this framework
            writes now is readable that way. Pass `False` only for a file you trust and
            that the safe mode refuses: one saved by an earlier version of this framework
            (its RNG state holds a NumPy array), or one whose `config` snapshot is not made
            of plain values (a dataclass, say). See
            `docs/adr/0034-safe-checkpoint-loading-by-default.md`.

    Returns:
        `CheckpointMetadata` holding everything besides the tensors
        that were just loaded in place.

    Raises:
        FileNotFoundError: If `path` does not exist.
        pickle.UnpicklingError: If `weights_only` is `True` and the file holds something the
            restricted unpickler refuses. The message names the two usual causes and the
            `upgradeCheckpoint` and `weights_only=False` remedies for a trusted
            file.
        ValueError: If `optimizer` is given but the checkpoint has no
            `optimizer_state_dict` (it was saved without one).
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"No checkpoint found at '{resolved_path}'.")

    try:
        checkpoint = torch.load(resolved_path, map_location=map_location, weights_only=weights_only)
    except pickle.UnpicklingError as error:
        if not weights_only:
            raise
        raise pickle.UnpicklingError(
            f"Checkpoint '{resolved_path}' could not be read with the restricted unpickler "
            f"(weights_only=True). Either it was saved by an earlier version of this framework "
            f"(its RNG state holds a NumPy array), or its `config` snapshot is not made of plain "
            f"values (dict, list, tuple, str, number, None). If you trust the file, rewrite it "
            f"once with `upgradeCheckpoint` (global_vae.training.checkpoint), or read it as it "
            f"is with weights_only=False (--trust-checkpoint for scripts/evaluate.py and "
            f"scripts/visualize_latent.py). Original error: {error}"
        ) from error

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
        callbacks_state=checkpoint.get("callbacks_state", []),
        scaler_state_dict=checkpoint.get("scaler_state_dict"),
    )
