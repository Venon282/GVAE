# 0011 — Hydra-driven config layer (`global_vae/config/`, `scripts/train.py`)

**Status:** accepted
**Date:** 2026-08-16

## Context

Spec §10's "Config management" bullet calls for "Hydra + structured dataclasses (or
Pydantic) for validated, composable configs. No magic strings/dicts scattered through
the code." Nothing implemented this yet: `configs/model/default.yaml` explicitly said
"Not yet consumed by the code... `GlobalVae` is currently built directly in Python", and
every existing integration test built models/trainers by calling Python constructors
directly. Separately, this project's data pipeline (dataset loading, transforms,
train/val/test splitting) was already scoped, in an earlier decision this session, as
entirely the user's own responsibility, never framework code (`data/NOTE.md`,
`configs/data/NOTE.md`). Building the Hydra config layer without contradicting that
scope decision, while still making `configs/data/*.yaml` a real, useful thing (not an
inert placeholder) was the central design question here.

Building this also surfaced a real, pre-existing bug: `encoders/__init__.py`,
`decoders/__init__.py`, and `fusion/__init__.py` were plain docstrings, unlike every
other registry-based subpackage (`assemblers/`, `losses/regularizers/`,
`training/beta_schedules/`, `training/loggers/`), which import every concrete
implementation for the registration side effect, exactly as spec §10 requires ("Each
registry-based subpackage's `__init__.py` must import every concrete implementation for
that side effect"). `getEncoderClass("1d_cnn_encoder_v1")` raised `KeyError` after only
`import global_vae.encoders`, silently relying on some other import elsewhere in the
process (a test file's own `from global_vae.encoders.OneDCnnEncoder import
OneDCnnEncoder`) to have already triggered registration first. A config-driven model
builder that resolves encoder/decoder/fusion names purely from strings has no such
lucky accidental import to rely on, which is what surfaced this.

## Decision

### Schema: dataclasses, not Pydantic

`pyproject.toml` lists both `hydra-core` and `pydantic` as dependencies, and spec §10
explicitly allows either ("structured dataclasses (or Pydantic)"). This uses plain
`@dataclass` throughout (`ModelConfig`, `DataConfig`, `TrainingConfig`,
`ExperimentConfig`, and their nested pieces), for two reasons: it is Hydra's own native
structured-config mechanism (`OmegaConf.structured(...)`, `ConfigStore`), with no
adapter layer needed, and it matches every dataclass already used across this codebase
(`LatentSpace`, `RoutingGraph`, `StepLosses`, `CheckpointMetadata`,
`EvaluationResults`), keeping one consistent idiom rather than introducing a second
validation framework alongside it. `pydantic` remains a listed dependency, unused by
this config layer; nothing here removes it, since a future need for more elaborate
cross-field validation than a dataclass conveniently expresses is exactly the kind of
thing Pydantic is good at, and this decision does not foreclose reaching for it later
for a specific config domain.

### Three config groups, matching spec §8's tree exactly, plus one new group

`configs/model/`, `configs/data/`, `configs/experiment/` already existed (spec §8).
`configs/training/` is a genuinely new fourth group, not in that tree: training
hyperparameters (optimizer, beta schedules, loggers, checkpointing) are numerous enough,
and reused enough across different model/data combinations, to be worth composing
independently rather than folding into each `configs/experiment/*.yaml` by hand. Spec
§9's own examples are explicitly "illustrative, not final", so this is a deliberate,
documented extension of that tree, not a silent deviation from it.

### `ModelConfig`: `latent_mode: "single" | "several"`, only `"single"` wired up

`buildModelFromConfig` is a thin, config-shaped wrapper around
`GlobalVae.createSingleLatent`, covering exactly the `EN-L1-DN` Phase-1 default (ADR
0001) and the single-modality `signal -> z -> signal` case (spec §6.1 milestone 1) this
session's work has been building toward. `latent_mode: "several"` is accepted by the
schema (so a config file does not need to change shape again once a general
`RoutingGraph` config lands) but `buildModelFromConfig` raises `NotImplementedError` for
it today, mirroring the exact "fail loudly, don't guess" stance `GlobalVae.__init__`
itself already takes for encoder fan-out. Building the general multi-latent-space
config schema now would violate spec §6.1's own build order ("don't build the fully
general multimodal machinery before the single-modality signal VAE... actually works
end to end").

One real behavior beyond a pure field-for-field mapping: `EncoderConfig`/
`DecoderConfig.kwargs` may omit `latent_dim`, auto-filled from `SingleLatentConfig.dim`
(an explicit value, if given, is never overwritten). A single latent space
architecturally requires every encoder and decoder to already agree on its
dimensionality, so repeating that number in every modality's YAML kwargs would be pure,
error-prone duplication; this removes it without losing the ability to override.

### `DataConfig`: schema only, never a `Dataset` implementation

Consistent with the earlier scope decision, `global_vae/config/data.py` never reads a
file, resamples a series, or constructs a `torch.utils.data.Dataset`. `DataConfig`
records the *shape* of the information a data pipeline needs (paths, batch size, split,
named transforms), so it can be validated and snapshotted like any other config domain
(spec §10), but the one field that makes it actually operable end to end is
`loader_factory`: a `"module.path:function_name"` reference (the exact convention
`scripts/evaluate.py` already used privately for `--model-factory`/
`--dataloader-factory`, now extracted to `global_vae.utils.imports.importCallable` and
shared) to a function the caller writes themselves: `(DataConfig) -> DataloaderBundle`.
`buildDataloadersFromConfig` does nothing but resolve and call that reference.
`DataloaderBundle` (`train`/`val`/`test` iterables of `dict[str, torch.Tensor]` batches,
the same convention `Trainer`/`GlobalVae.forward` already use) is a plain return-type
contract, not a dataset implementation either.

### `TrainingConfig`: every registry-backed field resolves through this project's own registries

`beta_schedules[...].strategy` and `loggers[...].name` are resolved via
`training.beta_schedules.registry`/`training.loggers.registry`, never hardcoded, so a
new schedule or logger strategy added elsewhere becomes usable from config with zero
changes here (spec §10, §12). `optimizer.name` and `reconstruction_loss` are the two
deliberate exceptions: they select a plain `torch.optim.Optimizer` subclass or a
`torch.nn.functional` loss, neither of which is one of this project's own pluggable
extension points; a small name -> class/function lookup (`resolveOptimizerClass`,
`resolveReconstructionLossFn`) covers the common cases without wrapping every PyTorch
built-in in a registry of its own. Nothing stops constructing a `Trainer` directly,
bypassing config, for an optimizer or loss this lookup does not cover.

### Two entry points into the same schema: `loadExperimentConfig` and `scripts/train.py`

Composition (which YAML files Hydra merges, driven by each
`configs/experiment/*.yaml`'s own `defaults` list) and validation (does the merged
result actually match `ExperimentConfig`'s shape) are kept as two separate steps:
Hydra's `compose()` produces a raw `DictConfig`; `OmegaConf.merge(OmegaConf.structured
(ExperimentConfig), composed)` then validates and casts it, raising a clear
`MissingMandatoryValue` on any missing required field. `loadExperimentConfig`
(`initialize_config_dir` + `compose`, safe to call repeatedly within one process, used
by tests and any other programmatic caller) and `scripts/train.py`'s `@hydra.main`
-decorated `main` (the real CLI entry point, with working-directory management and
`--help`/dotlist-override support) both do this same two-step validation, so neither
one's guarantees differ from the other's. Every `configs/experiment/*.yaml` needs `#
@package _global_` at the top so its `defaults` list (pulling in `model`/`data`/
`training`) lands at the config root instead of nested under an `experiment:` key
(verified empirically: Hydra nests composed content under the source directory's name
by default unless told otherwise).

### Bug fix: `encoders/`, `decoders/`, `fusion/` `__init__.py` now register their concrete implementations

Fixed to match the pattern already used everywhere else (see Context). This was a
latent bug independent of this config work, but this work is what actually surfaced it:
`buildModelFromConfig` needs `getEncoderClass`/`getDecoderClass`/`getFusionClass` to
reliably resolve any registered name after nothing more than `import global_vae.config`
(which imports `global_vae.models.global_vae`, which does not itself import
`global_vae.encoders`/`decoders`/`fusion`'s concrete submodules), not rely on some
unrelated part of the process having already imported the specific concrete
implementation module first.

## Consequences

- The signal-VAE milestone (spec §6.1 milestone 1) is now runnable end to end from
  config: `python scripts/train.py data.train_path=... data.loader_factory=...`, using
  `configs/experiment/signal_vae.yaml` (composing `configs/model/signal_single_latent.yaml`
  + `configs/data/signal.yaml` + `configs/training/default.yaml`) by default. Verified
  by actually running it against a dummy in-memory `loader_factory`, end to end,
  including real `CsvLogger`/`TensorBoardLogger`/`CheckpointCallback`/
  `BestCheckpointCallback` output on disk.
- `configs/model/default.yaml` (the illustrative two-modality signal+image example) is
  now schema-valid against `ModelConfig`, but intentionally not yet buildable
  (`buildModelFromConfig` raises `KeyError` on the unregistered `resnet_encoder_v1`/
  `resnet_decoder_v1`), matching the documented state in README.md's "What's
  deliberately not built yet" (no image encoder/decoder implementation exists yet).
- `scripts/evaluate.py`'s private `_importCallable` is now `global_vae.utils.imports
  .importCallable`, imported back into that module under its old private name for
  backward compatibility with its own existing tests. Behavior is unchanged; this is a
  pure DRY extraction, now shared by `global_vae/config/data.py` and `scripts/train.py`.
- `tests/integration/test_config.py` covers: composition and validation (including
  `${output_dir}` interpolation reaching nested `training.*` fields, CLI-style
  overrides, the `MissingMandatoryValue`/`MissingConfigException` error paths),
  `buildModelFromConfig` (real forward pass, `latent_dim` auto-fill and override,
  `NotImplementedError`/`ValueError`/`KeyError` paths), `buildDataloadersFromConfig`,
  every `training.py` lookup/builder function, and a full config-to-trained-model
  `Trainer.fit` run. `tests/integration/test_train_script.py` runs `scripts/train.py` as
  a real subprocess (its `@hydra.main`-decorated `main` owns global process state not
  safe to exercise twice in one interpreter, unlike `scripts/evaluate.py`'s plain
  argparse `main`), covering the CLI end to end: successful run, logger/checkpoint file
  output, a reloadable checkpoint, hyperparameter overrides actually taking effect, and
  the missing-required-field and invalid-`loader_factory`-spec error paths.
- `configs/data/NOTE.md` and `configs/experiment/NOTE.md` updated: the config *schema*
  and an illustrative signal-VAE experiment now exist; only concrete dataset/transform
  *code* remains deferred, unchanged from the original scope decision.
- Not built: a general `RoutingGraph`-based config for `latent_mode: "several"` (see
  `ModelConfig`'s own docstring); per-modality `reconstruction_loss`/
  `reconstruction_weight` from config (construct a `Trainer` directly for that case
  today); a `configs/data/*.yaml` group example beyond `signal.yaml`.
