# 0034: Checkpoint loading is safe by default

**Status:** accepted (roadmap item P2-3; a deliberate change of a default, with a one-line way back for a file you trust)
**Date:** 2026-10-18

## Context

A checkpoint is a `torch.save` file, which is a pickle. Loading one with `weights_only=False` runs
whatever code the file asks for, so a checkpoint from someone else is a program. The module
docstring of `training/checkpoint.py` said so, and `loadCheckpoint` read every file that way,
because the file held two things the restricted unpickler (`weights_only=True`) refuses:

- the NumPy RNG state, which `np.random.get_state` returns as an `ndarray` of 624 words;
- the `config` snapshot when it was a dataclass (`examples/02_config_driven_pipeline.py` passed the
  `ExperimentConfig` itself).

PyTorch changed its own default to `weights_only=True` in 2.6. The roadmap lists "a safer default for
checkpoint loading (it uses `weights_only=False`, with the trust caveat documented)" as an optional
part of P2-3.

## Decision

**1. A checkpoint written now holds only tensors, numbers, strings, `None` and plain containers.**
The restricted unpickler can then read it, and a test loads every kind of file this framework
writes with `torch.load(weights_only=True)`:

- the NumPy RNG state is stored as `(name, keys.tolist(), position, has_gauss, cached_gaussian)`;
  `_restoreRngState` takes a list or an array, so a file from before this ADR still restores it;
- a dataclass instance passed as `config` is converted with `dataclasses.asdict` by `saveCheckpoint`,
  so `config=experiment_config` keeps working, and `loadCheckpoint` returns the dict. Any other
  `config` is written as given: a `dict` of plain values loads (what `scripts/train.py` stores, from
  `OmegaConf.to_container`), an object does not;
- the callback states and the scaler state added by ADRs 0031 and 0032 are plain by construction.

**2. `weights_only=True` is the default, everywhere a checkpoint is read.**
`training.checkpoint.loadCheckpoint(..., weights_only=True)` and
`Trainer.loadCheckpoint(..., weights_only=True)` take the flag, and `scripts/evaluate.py` and
`scripts/visualize_latent.py` pass `not args.trust_checkpoint`. `--trust-checkpoint` (off by default)
reads the file with `weights_only=False`, and its help text says that this runs pickle code.
`training.resume_from` has no such flag: a run that resumes reads a file it wrote, and one from an
earlier version is upgraded first (below).

**3. A refused file gets an error that says what to do.** `loadCheckpoint` catches the
`pickle.UnpicklingError` of the restricted unpickler and raises another, from it, that names the
file, the two usual causes (a file from an earlier version, whose RNG state is an array, and a
`config` that is not made of plain values) and the three ways out: `upgradeCheckpoint`,
`weights_only=False`, and `--trust-checkpoint`. A caller who passes `weights_only=False` sees the
original error, if any.

**4. `upgradeCheckpoint(path, output_path=None)` rewrites an old file once.** It reads `path` with
`weights_only=False` (so call it only on a file you trust, as its docstring says), turns the RNG
array into a list, saves, and checks that the result loads in the safe mode, logging a warning if
it still does not (an object in the `config`). It writes over `path`, or to `output_path` and leaves
the original alone. Nothing else in the file changes: the weights, the optimizer, the history and the
config are compared in a test.

## Alternatives considered

- **Keep `weights_only=False` and rely on the caveat.** It is what the module did, and the only
  protection is that every reader remembers it. The cost of the change is one flag, for a file
  that was written before this ADR.
- **Allow the NumPy and dataclass globals with `torch.serialization.add_safe_globals`.** It makes the
  old files load, but it names NumPy internals (`numpy.core.multiarray._reconstruct`, which moved to
  `numpy._core` in NumPy 2), so it breaks across NumPy versions, and it widens what the
  "safe" mode accepts for every file, instead of changing what the framework writes.
- **A flag in `Trainer` or a global setting.** A function argument is visible at the call that
  reads the file, and a global is not.
- **`safetensors`.** It reads no code, but it stores tensors only: the optimizer state, the history,
  the RNG state and the config would need another file and a format of our own, and a new
  dependency. The checkpoint stays one file.
- **Convert a dataclass config to a dict in `loadCheckpoint` instead of on save.** The file would
  still be unreadable in the safe mode.

## Consequences

- **Files saved by an earlier version are refused by default** if they hold the RNG state (the
  default of `saveCheckpoint`) or a dataclass config. Run `upgradeCheckpoint(path)` on each one
  you trust, or pass `--trust-checkpoint` to the scripts, or `weights_only=False` in Python. A file
  saved with `include_rng_state=False` and a plain config loads as before. This is recorded in the
  changelog as a change of behavior.
- `loadCheckpoint` returns the `config` as a dict where it used to return the dataclass that
  was saved. A caller that rebuilt an object from it rebuilds it from the dict
  (`OmegaConf.structured` and `OmegaConf.merge` do).
- The restricted unpickler is PyTorch's guard against code execution at load time, not a promise
  about a hostile file: a file can still be very large, or hold tensors that are not the model's.
  The how-to says to load only files you trust and gives this as the reason for the default, not as a
  reason to stop checking where a checkpoint comes from.
- A caller's own callback whose `stateDict` holds an object, or a `config` that holds one, makes the
  checkpoints of that run unreadable in the safe mode. The error says which file, and the base
  class and `saveCheckpoint` docstrings say what is allowed.
- Not decided here: signing or hashing checkpoints, and moving the weights alone to `safetensors`
  for sharing a trained model.
