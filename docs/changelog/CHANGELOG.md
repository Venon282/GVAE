# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Latent heads (roadmap P1-1, spec §2.2). The new `global_vae.heads` package holds
  `AbstractLatentHead` (`base.py`: built from `in_dim` and `out_dim`, `forward(mu, logvar)` returns
  the adapted `(mu, logvar)`, positive-integer validation, a shape check shared by the concrete
  heads), the registry (`registry.py`: `registerHead`, `getHeadClass`, `listRegisteredHeads`) and
  two built-ins, one class per file. `identity` is the default: it returns the very tensors it
  receives and requires `in_dim == out_dim`. `linear` projects `mu` and `logvar` through two
  separate `nn.Linear` layers (`to_mu`, `to_logvar`); the initialisation is Xavier uniform weights
  and zero biases, so the prior `(0, 0)` maps to itself and two heads on one encoder start
  different. `heads/__init__.py` imports both modules. `GlobalVae` does not use heads yet:
  encoder fan-out is P1-2.
- `tests/unit/test_heads.py`: registry, shape, gradient, initialisation and identity-is-a-no-op
  tests. The latent head registry joins `tests/unit/test_registry_contract.py`, which now covers
  ten registries (`test_nine_registries_are_covered` became `test_ten_registries_are_covered`).
- `docs/how-to/add-a-strategy.md` covers latent heads: the table row, the constructor contract
  (`in_dim` and `out_dim` first) and the `forward` signature. The spec §8 tree no longer marks
  `heads/` as `(planned)`.
- Encoder fan-out through latent heads (roadmap P1-2, spec §2.2, ADR 0024). One encoder can now
  feed several latent spaces, each encoder -> latent edge with its own head, so the shared plus
  private topology builds, trains and evaluates. `latent/base.py` gains `HeadSpec(name, out_dim,
  kwargs)` and the `RoutingGraph.encoder_heads` field (encoder name -> latent space name ->
  `HeadSpec`, declared last so positional construction keeps working). `GlobalVae` builds one head
  module per entry, as `getHeadClass(name)(in_dim=<encoder latent_dim>, out_dim=<latent dim>,
  **kwargs)`, stores them in `GlobalVae.heads` (state keys `heads.<encoder>.<latent>...`) and
  applies each before fusion; `GlobalVae.getHead(encoder, latent)` returns a head, or `None` for
  an identity edge. An edge with no entry is an identity edge and builds no module, so a graph
  without heads builds, runs and checkpoints exactly as before.
- `validateRoutingGraph` has three new rules (spec §10): a head must sit on an existing edge, a
  head's declared `out_dim` must equal its latent space's `dim` (and `kwargs` may not set
  `in_dim` or `out_dim`), and an encoder feeding several latent spaces may have at most one
  identity edge, since two would give identical posteriors. A new `identity_head_names` argument
  says which head names count as the identity.
- `AbstractAssembler.handlesMissingLatents`, the assembler-side counterpart of
  `AbstractFusion.handlesMissingModalities`: `True` for `sum` and `average`, `False` by default
  (so for `concat` and for any assembler that does not say otherwise).
- `docs/adr/0024-encoder-fan-out-through-latent-heads.md`, which supersedes the fan-out guard of
  ADR 0002 (ADR 0002 is left as written).
- `tests/integration/test_encoder_fan_out.py`: a signal plus image shared plus private model with
  `z_shared` through PoE and linear private heads trains with a falling loss, every parameter
  (heads included) receives a gradient, the three latent spaces differ, modality dropout removes
  an absent encoder's edges, and evaluation and the cross-modal reports run. Unit tests cover the
  new validation rules, the preset's head arguments and `handlesMissingLatents`.
- Config schema for several latent spaces (roadmap P1-3, spec §9 and §11, ADR 0025). `buildModelFromConfig`
  builds `latent_mode: several` into a `RoutingGraph` and no longer raises `NotImplementedError`;
  `latent_mode: single` and every shipped YAML file stay valid and build the same model. `ModelConfig`
  gains `encoders` and `decoders` (instance name -> `{name, kwargs}`, added to those of `modalities`,
  so a decoder needs no encoder of the same name, in either mode) and `latent: LatentConfig` with
  `spaces` (`LatentSpaceConfig`: `dim`, `fed_by`, `fusion`, `regularizer`) and `decoders_consume`
  (`DecoderConsumptionConfig`: `spaces`, `assembler`, `kwargs`). `fed_by` is a list of encoder names
  (identity edges) or a mapping `encoder -> {head, kwargs}` (`EdgeHeadConfig`, one latent head per
  edge, mapped to `RoutingGraph.encoder_heads`). Beta stays in `TrainingConfig.beta_schedules`.
  This is the default proposed for roadmap decision D-1, taken without an explicit answer, so it
  needs the owner's approval.
- The config package is split in three. `config/model_schema.py` holds the dataclasses,
  `config/model_resolution.py` the validation and the derivation of `GlobalVae.__init__`'s
  arguments (`validateModelConfig`, `resolveModelConfig`, `parseFedBy`, `ResolvedModel`), and
  `config/model.py` the builder and the re-exports, so `from global_vae.config.model import
  ModelConfig` keeps working. Every error starts with the dotted path of the field to edit
  (`model.latent.spaces.z_shared.fed_by: unknown encoder 'audio'`), a section of the other latent
  mode is refused instead of ignored, and an encoder, decoder or latent space with no partner is
  refused. `latent_dim` is derived and never repeated: an encoder takes the `dim` of the space it
  feeds through an identity edge, a decoder the width its assembler produces (the assembler is
  run once on zero tensors, so a new assembler needs no change here), and an explicit value that
  disagrees is an error in `several` mode.
- `GlobalVae.__init__` takes `assembler_kwargs` (decoder name -> constructor kwargs), the
  counterpart of `fusion_kwargs` and `regularizer_kwargs`. The built-in assemblers take none, so
  nothing changes for them.
- Shipped configs, one model and one experiment per pattern of spec §9:
  `configs/model/signal_image_independent_latents.yaml` and `signal_image_shared_private.yaml`
  with `configs/experiment/signal_image_independent_latents.yaml` and
  `signal_image_shared_private.yaml`, plus `configs/experiment/signal_image_vae.yaml` for the
  single latent space of `configs/model/default.yaml`. `configs/model/signal_image_to_image.yaml`
  is the model of `examples/03_signal_image_to_image.py`, with the same parameters as the Python
  one. `configs/data/signal_image.yaml` is the data contract of a signal plus image model, and
  `configs/training/several_latents.yaml` the training defaults without a beta schedule (the
  experiment files add one per latent space).
- `buildTrainerFromConfig` logs a warning for a `training.beta_schedules` key that is not a latent
  space of the model, instead of dropping the schedule without a sign.
- `docs/adr/0025-config-schema-for-several-latent-spaces.md`, which supersedes the single-only part
  of ADR 0011 (ADR 0011 is left as written), and `docs/how-to/configure-latent-spaces.md`.
- `tests/unit/test_model_config_validation.py` (every validation rule, `parseFedBy`, the derived
  widths and the resolved arguments, 106 tests) and `tests/integration/test_model_config_schema.py`
  (every shipped model and experiment file validates and builds, the two new patterns build, run and
  send a gradient to every parameter, the example 03 model equals the Python one, single-latent
  files equal `GlobalVae.createSingleLatent`, errors name their field). `test_train_script.py` trains
  the three signal plus image experiments through `scripts/train.py` and checks that a wrong
  `fed_by` stops the run with the field path; `_train_script_fixtures.py` gains a paired dummy loader.
- `tests/unit/test_deprecated_module_paths.py` and `tests/unit/test_ci_workflow.py` (roadmap
  P0-2). The first pins the deprecation shims below: each old import path warns and names its
  replacement, returns the very same class, registers nothing twice and defines nothing of its
  own, the `encoders` and `decoders` `__init__` files import no shim, and the only CamelCase
  module files left under `src/global_vae` are the eight shims. The second fails when
  `continue-on-error` returns to the CI workflow or one of its three jobs is dropped.
- `docs/adr/0023-snake-case-module-names.md`, the decision record of the rename below.
- Assemblers `weighted_sum` and `attention` (roadmap P1-4, spec §2.2 and §11, ADR 0026), one class per
  file, registered and imported in `assemblers/__init__.py`. `weighted_sum` has one learned weight per
  latent space: with `normalize=True` (default) the weights are the softmax of the stored values over
  the spaces present in the pass and start equal, so it starts as `average`; with `normalize=False`
  the stored values are the weights and start at 1, so it starts as `sum`. Same dimension rule as
  `sum`. `attention` turns each latent vector into a token with its own linear projection, pools the
  tokens with one learned query (`nn.MultiheadAttention`) and returns an `output_dim`-wide vector; it
  has no dimensionality requirement, `output_dim` defaults to `max(input_dims)` and `num_heads` to 1.
  Both accept a subset of the latent spaces (`handlesMissingLatents` is `True`), so a decoder keeps
  running when a modality is left out. They are the first assemblers with parameters: a checkpoint
  carries them as `assemblers.<decoder>.*`. This is the order proposed for roadmap decision D-3,
  taken without an explicit answer, so it needs the owner's approval.
- Assembler constructor contract (ADR 0026). `AbstractAssembler.__init__(input_dims)` takes the
  width of each latent vector and validates it, `input_dims` and the abstract `output_dim` expose
  what the strategy receives and returns, `requires_matching_dims` is a class-level flag
  (`True` for `sum`, `average` and `weighted_sum`) that the base constructor enforces, and
  `forward(latents, present=None)` gets the position of each latent space when only some are
  present. `_resolvePresent` checks the inputs against `input_dims` and refuses a subset when
  `handlesMissingLatents` is `False`. `RoutingGraph.decoderInputs()` gives, for each decoder, the
  latent spaces it consumes in the order its assembler is built and called with.
- `GlobalVae` builds each assembler as `getAssemblerClass(name)(input_dims, **kwargs)` and checks
  each decoder's `latent_dim` against what it receives (the assembler's `output_dim`, or the `dim` of
  its one latent space) when the model is built. An omitted `latent_dim` is filled in when the
  decoder's constructor takes one. The caller no longer adds up the widths by hand for `concat`, and
  a wrong value no longer waits for the first forward pass to fail with `mat1 and mat2 shapes cannot
  be multiplied`.
- `docs/adr/0026-assembler-constructor-contract-and-learned-assemblers.md`, which supersedes the
  zero-tensor derivation of a decoder width in ADR 0025 (ADR 0025 is left as written).
- `tests/unit/test_learned_assemblers.py` (values, gradients that reach the parameters, subsets of
  the latent spaces, state dict round trip) and `tests/integration/test_learned_assemblers.py`
  (`EN-LN-DN` with real 1D and 2D modules: decoders sized from the assembler, every parameter
  receives a gradient, the loss falls, a modality left out still decodes, the model written in YAML
  with its `kwargs`, a checkpoint round trip). `test_configuration_matrix.py` runs the `LN` rows once
  per registered assembler, five in all, and fails if one is missing.
- Shared encoders and shared decoders, the `E1` and `D1` rows (roadmap P1-5, spec §2.1, ADR 0027).
  `AbstractSharedEncoder` (`encoders/shared_base.py`) declares the modalities it ingests
  (`input_modalities`) and its `latent_dim`, and its `forward` takes a `dict[str, Tensor]` with the
  missing modalities omitted and returns `(mu, logvar)`. `AbstractSharedDecoder`
  (`decoders/shared_base.py`) declares `output_modalities`, and its `forward` returns a
  `dict[str, Tensor]`. Both are siblings of `AbstractEncoder` and `AbstractDecoder`, not subclasses
  (their `forward` types differ), and register in the same registries (`registerEncoder` and
  `registerDecoder`, whose `getEncoderClass` and `getDecoderClass` now return the union of the two
  kinds). The tokenisation lives in the concrete module. `GlobalVae` tells the kinds apart with
  `isinstance`: it groups the batch by owner encoder, calls a shared encoder with a dict in its
  declared order, and flattens a shared decoder's dict into `"reconstructions"` by modality, so the
  losses, `evaluate` and modality dropout are unchanged. The routing graph is unchanged too: a
  shared module is one node named by its instance, so `E1-LN-*` is one encoder with several edges
  and heads. The four rows build from `encoders` and `decoders` in YAML, in either `latent_mode`,
  and one-modality and shared modules can be mixed. This is option A of roadmap decision D-2, taken
  without an explicit answer, so it needs the owner's approval.
- `GlobalVae.input_modalities` and `GlobalVae.output_modalities`: the batch keys the model ingests
  and reconstructs. For a model with no shared module they equal `tuple(encoders)` and
  `tuple(decoders)`. A shared module with an empty, repeated or non-string declaration is refused at
  construction, as is a modality ingested by two encoders or reconstructed by two decoders, and a
  shared decoder that returns other keys than it declares fails the forward pass with both sets in
  the message.
- `docs/adr/0027-shared-encoders-and-decoders.md`, which supersedes the rule of ADR 0019 that a
  batch key names an encoder (ADR 0019 is left as written).
- `tests/integration/_shared_modules.py`: dummy shared modules (`dummy_shared_encoder_v1` averages
  one linear token per modality present, `dummy_shared_decoder_v1` has one linear layer per
  modality) and two that break the contract on purpose. `test_configuration_matrix.py` runs its
  stage 3: `E1-L1-DN`, `E1-L1-D1`, `E1-LN-DN` and `E1-LN-D1` (the `LN` rows once per assembler) check
  the topology, one encoder call, shapes, a finite loss, a gradient on every parameter and a falling
  loss, and the test fails if one of the eight rows is dropped.
  `tests/integration/test_shared_encoder_decoder.py` covers YAML, mixed one-modality and shared
  encoders under PoE, the construction guards, modality dropout with a shared encoder, the
  cross-modal reports and figures, `evaluate` and a checkpoint round trip;
  `tests/unit/test_shared_modules.py` covers the interfaces and registries.
- `docs/how-to/add-a-modality.md` and `configure-latent-spaces.md` explain how to write and
  configure a shared encoder or decoder.
- Configuration matrix test, stage 2 and the unified test (roadmap P1-6, spec §10).
  `tests/integration/test_configuration_matrix.py` adds the two shared plus private variants of
  the `EN-LN-*` rows, built with the real 1D and 2D modules and run once per registered assembler:
  `EN-LN-DN+shared-private` (from `buildSharedPrivateRoutingGraph`) and `EN-LN-D1+shared-private`
  (an explicit graph, one decoder over `z_shared` and both private spaces). In both, `z_shared` is
  fused by PoE through an identity edge from each encoder and each private space sits behind a
  `linear` head. `TestStage2Rows` checks the topology, the edges and heads, that a private
  posterior differs from the shared one, and that each assembler receives its latent spaces in
  graph order and combines them as specified.
- `MATRIX`, the explicit table of the eight row codes (and the two variants) with a builder and the
  expected reconstructions of each, and `TestEveryRow`, the one parametrized test over it. For
  every row, and every assembler of an `LN` row, it checks the reconstruction shapes, the shape of
  each latent space, a finite loss, a finite gradient on every parameter, and that 25 Adam steps on
  a fixed batch lower the loss. `test_the_eight_rows_are_all_in_the_matrix` fails when a row, a
  variant or an assembler is dropped, which was checked by removing each in turn.
- The learning rate in the metrics, the logs and a plot (ADR 0028). `TrainerCallback` gains the
  class attribute `acts_on_learning_rate` (`False` by default), which `ReduceLrOnPlateau` sets to
  `True`, and a callback of your own sets when it changes the optimizer's rate.
  `Trainer.logs_learning_rate` is `True` when any callback of a run sets it, and `fitEpoch` then
  adds the rate each step trained with to the step metrics (`lr`, or `lr/group<i>` with several
  parameter groups). The epoch metrics carry its mean as `train/lr` (`train/lr/group<i>`), so it
  shows in the epoch log line, `Trainer.history`, `CsvLogger`, TensorBoard and the checkpoint
  history. The rate is read before the optimizer step, so a reduction made when epoch `n` ends
  shows from epoch `n + 1`. A run with no such callback records nothing new.
- `training/learning_rate.py`: `LEARNING_RATE_KEY`, `learningRateMetrics(optimizer)` and
  `isLearningRateKey(key)`, which hold the key spelling for the trainer, the log formatting and
  the plots.
- `plotLearningRate(history, keys=None, log_scale=True, ...)` in `visualization/loss_curves.py`,
  one figure with a log y-axis by default. It raises `ValueError` for a history with no rate key,
  naming `acts_on_learning_rate`. `scripts/visualize_latent.py` writes it to `lr_curve.png` next
  to `loss_curves.png` when the checkpoint history holds a rate, and `--skip-history` skips both.
- `docs/adr/0028-learning-rate-in-the-metrics.md` (requested by the owner, not a roadmap item),
  a "Watch the learning rate" section in `docs/how-to/train.md`, the flag in the callbacks note
  of `docs/how-to/add-a-strategy.md`, and a line for `training/learning_rate.py` in the spec §8
  tree (a one-line spec edit, to be confirmed by the owner).
- `tests/unit/test_learning_rate.py` (the key helpers and the flag on the built-in callbacks),
  `tests/integration/test_learning_rate_logging.py` (a `Trainer` run with `ReduceLrOnPlateau`: the
  epoch history is `[1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5]`, the step and epoch log lines, `CsvLogger`
  rows, the checkpoint history, resuming, a custom callback, several parameter groups, a trainer
  built from a config, and a run with no such callback unchanged), and tests of `plotLearningRate`
  and of `lr_curve.png` in `tests/integration/test_loss_curves.py` and
  `tests/integration/test_visualize_latent_script.py`.
- Loss scale convention and a reconstruction reduction per modality (roadmap P2-1, ADR 0029). The
  reconstruction term is a per-element mean while the KL is a per-sample sum over the latent
  dimensions, so `beta = 1` weights the KL `N / 2` times more than the usual ELBO (`N` is the
  number of values a sample reconstructs: 128 times for a 256-point signal, 512 for a 32 by 32
  image). `computeTotalReconstructionLoss` gains `reduction` (a name, or a modality -> name
  mapping; a modality left out keeps `"mean"`): `"mean"` is the default and leaves the loss exactly
  as it was, `"sum"` multiplies the loss function's value by `N` (the per-sample sum, averaged over
  the batch). `SUPPORTED_REDUCTIONS`, `DEFAULT_REDUCTION`, `listSupportedReductions`,
  `validateReduction` and `resolveReduction` are in `losses/reconstruction.py`.
- `Trainer(reconstruction_reduction=...)`, the last argument so positional callers are unaffected,
  and `TrainingConfig.reconstruction_reduction` (`str | dict[str, str]`, typed `Any` in the
  structured config and validated by `buildTrainerFromConfig`, default `"mean"`, written in
  `configs/training/default.yaml` and `several_latents.yaml`). An unsupported name or a decoder the
  model does not have is a `ValueError` that starts with `training.reconstruction_reduction`: a
  mistyped decoder would otherwise stay on `mean` without a word. `resolveReconstructionReduction`
  does the check. The validation loss of `Trainer.evaluate` uses the same reduction.
- Latent diagnostics in the evaluation report. `evaluation/latent_diagnostics.py` has
  `LatentDiagnostics` (the KL of every latent dimension, the variance of its posterior mean, the
  number of active units and the threshold used), `computeLatentDiagnostics` and
  `DEFAULT_ACTIVE_UNIT_THRESHOLD` (0.01, Burda et al. 2016). `EvaluationResults` gains
  `latent_diagnostics` (latent space name -> `LatentDiagnostics`) and `reconstruction_reduction`,
  both with defaults so a result built by hand still works; they are in `toDict`, the JSON file and
  `summary()`. `evaluate()` takes `reconstruction_reduction` and `active_unit_threshold`, and
  `scripts/evaluate.py` has `--reconstruction-reduction` (a name or `modality=name` pairs) and
  `--active-unit-threshold`. The pooled `(mu, logvar)` are the input, so `max_samples` applies.
- `computeKlPerDimension` in `losses/regularizers/kl_standard_normal.py`, the one KL formula behind
  the regularizer, `plotPerDimensionKl` and the diagnostics.
- `docs/adr/0029-loss-scale-convention.md`, `docs/how-to/choose-beta.md` (what `beta` multiplies
  under each reduction, how to read the diagnostics, a sweep recipe) and a "Choose the reconstruction
  reduction" section in `docs/how-to/train.md`, a paragraph in `README.md` and the diagnostics in
  `docs/how-to/evaluate.md`. No value of `beta` is given: the numbers are roadmap decision D-6 and
  wait for P3-1. The spec §2.3, §8, §9 and §11 say the same in a few lines each (spec edits, to be
  confirmed by the owner).
- `tests/unit/test_reconstruction_reduction.py` (values for both reductions, per-modality mixes,
  the default bit for bit equal to the old loss, validation errors, gradients),
  `tests/unit/test_latent_diagnostics.py` (the KL and active-unit maths against hand-computed
  values) and `tests/integration/test_loss_scale_convention.py` (YAML and Hydra overrides to
  `Trainer`, training with `sum`, `evaluate()`, the summary and JSON, and `scripts/evaluate.py`).
- A reconstruction loss and weight per modality, from config (roadmap P2-2, ADR 0030).
  `TrainingConfig.reconstruction_loss` (`str | dict[str, str]`) and `reconstruction_weight`
  (`float | dict[str, float]`) take one value for every modality or a decoder -> value mapping, as
  `Any` in the structured config, validated by `buildTrainerFromConfig`. A decoder left out of a
  mapping keeps `"mse"` and `1.0`; the defaults are unchanged, so a run that sets neither computes
  the same loss bit for bit. An unknown loss name (`KeyError`), a loss that is not a name or a
  mapping, a weight that is not a number, negative, `nan`, infinite or a `bool`, and a key that is
  not a decoder of the model (`ValueError`, listing the decoders and what the decoder would have
  kept) all stop the run with a message that starts with the field's path.
  `validateReconstructionLoss`, `validateReconstructionWeight`, `resolveReconstructionLoss`,
  `resolveReconstructionWeight` and `DEFAULT_RECONSTRUCTION_LOSS` / `DEFAULT_RECONSTRUCTION_WEIGHT`
  are in `config/training.py`, and `resolveReconstructionReduction` takes an optional `field_name`.
- `scripts/evaluate.py` takes `--reconstruction-loss` and `--reconstruction-weight` (a bare value or
  `modality=value` pairs), so that the report's total reconstruction loss can be on the training
  scale for any run and not only for `mse` with weight `1`, and it checks the decoder names of all
  three flags, `--reconstruction-reduction` included. A modality that the model does not have is a
  usage error (exit code 2).
- `docs/adr/0030-per-modality-reconstruction-loss-and-weight.md`, a "Set the loss and weight per
  modality" section in `docs/how-to/train.md`, and the flags in `docs/how-to/choose-beta.md`,
  `docs/how-to/evaluate.md` and `README.md`. The spec §2.3, §9 and §11 say the same in a few lines
  each (spec edits, to be confirmed by the owner): the §11 sentence that kept per-modality loss and
  weight a Python-only option of `Trainer` is replaced.
- `tests/unit/test_reconstruction_config_validation.py` (the validators) and
  `tests/integration/test_per_modality_loss_and_weight.py` (Hydra overrides to `Trainer`, the
  resolvers, the loss values by linearity against one modality weighted at a time, the composition
  with the reduction, gradients, training, and the `scripts/evaluate.py` flags).
- Trainer capabilities (roadmap P2-3, ADRs 0031 to 0034), each opt-in and off by default.
  Mixed precision, gradient accumulation and `torch.compile` (ADR 0031): `Trainer(mixed_precision=
  None | "bf16" | "fp16", accumulate_grad_batches=1, compile_model=False, compile_kwargs=None)` and
  the `TrainingConfig` fields of the same names. The forward pass runs under autocast with the
  losses computed in `float32`; `fp16` adds a `torch.amp.GradScaler` (`Trainer.scaler`) whose state
  is saved and restored with the checkpoint. With `accumulate_grad_batches = N` a step is an optimizer
  update of `N` batches, so `global_step`, the beta schedules, `onStepEnd` and `log_every_n_steps`
  count updates, and the partial last group of an epoch is rescaled to a mean. `compile_model` wraps
  the forward pass in `torch.compile` and leaves `Trainer.model` and the `state_dict` keys alone.
- Resume from the training script (ADR 0032). `training.resume_from` and
  `resumeTrainerFromConfig(trainer, config) -> int`, which `scripts/train.py` calls: **`num_epochs`
  is the total of the run**, so it trains `num_epochs - start_epoch` epochs, or none (the script logs
  it and exits) when the checkpoint already holds them; a missing file is a `FileNotFoundError`
  that starts with `training.resume_from`. `TrainerCallback.stateDict()` and `loadStateDict()`
  (no-ops by default) are implemented by `EarlyStopping`, `ReduceLrOnPlateau`,
  `BestCheckpointCallback`, `CheckpointCallback` and the new `top_k_checkpoint`, stored in the
  checkpoint as `callbacks_state` and given back to the callback of the same class. `CsvLogger`
  gains `append=False`, which `buildCallbacksFromConfig` sets to `True` for a resumed run. **A
  resumed deterministic run reproduces the uninterrupted loss curve to a relative `1e-6` (absolute
  `1e-9`) and the weights to `1e-6`.**
- Callbacks `top_k_checkpoint` (`TopKCheckpointCallback`: the `k` best epochs, one full
  checkpoint each, the worst deleted, ties keep the earlier epoch, a non-finite value is never kept),
  `linear_warmup_lr` and `cosine_lr` (ADR 0033). The two schedules set the rate of every parameter
  group to `initial_lr * factor(step)` at each optimizer step, with `initial_lr` kept in the
  optimizer's parameter groups, so a resume needs no schedule state, and they act on the learning
  rate (ADR 0028), so `train/lr` is recorded. `cosine_lr` warns when another callback that changes
  the rate is present. They are in `training/callbacks/` with the private base `_lr_schedule.py`.
- `upgradeCheckpoint(path, output_path=None)` in `training/checkpoint.py`, which rewrites a
  checkpoint from an earlier version so that the safe mode reads it, and `--trust-checkpoint` on
  `scripts/evaluate.py` and `scripts/visualize_latent.py` (ADR 0034).
- `docs/adr/0031-*.md` to `docs/adr/0034-*.md`, sections of `docs/how-to/train.md` (learning-rate
  schedules, mixed precision, accumulation and compile, top-K checkpoints, resuming, loading
  checkpoints safely) and a note in `docs/how-to/evaluate.md`; the spec §9, §10 and §11 say the same
  in a line each (spec edits, to be confirmed by the owner).
- Tests: `tests/integration/test_trainer_capabilities.py`, `test_resume.py` (six scenarios
  interrupted and resumed, the callback states, the CSV append), `test_top_k_checkpoint.py`,
  `test_lr_schedules.py`, `test_safe_checkpoint_loading.py` and a resume class in
  `test_train_script.py` that runs `scripts/train.py` in subprocesses.

### Changed

- `GlobalVae.forward` and modality dropout (ADR 0024). A decoder that consumes several latent
  spaces and is missing some of them in a pass (the private latent of a modality that was left
  out) runs only if its assembler has `handlesMissingLatents`; otherwise it is absent from
  `"reconstructions"`, like a decoder with no input. For `sum` and `average` nothing changes.
  For `concat` the partial input could only fail with a width mismatch before. A custom assembler
  that accepted a subset must now say so by overriding `handlesMissingLatents`.
- `buildSharedPrivateRoutingGraph` writes a `HeadSpec` on every edge and takes `shared_head`
  (default `"identity"`) and `private_head` (default `"linear"`). Its default graph used to have
  two identity edges per encoder, which `validateRoutingGraph` now refuses, and which the model
  could not build; the new default is the arrangement of spec §9 and builds.
- `tests/integration/test_global_vae_construction_guards.py`: the P0-4(f) fan-out guard test is
  inverted. The same graphs now build, and a graph with two identity edges raises `ValueError`
  from `validateRoutingGraph`. `test_routing_graph_validation.py`,
  `test_routing_graph_presets.py` and `test_assemblers.py` follow.
- The how-to `add-a-strategy.md`, the README, the spec (§2.2 constraints, §8 tree, §10 validation
  rule) and the `cross_modal.py` docstring describe fan-out as built.
- Spec §9 is rewritten to the final schema (ADR 0025): `latent.mode` and `single_latent` are
  `latent_mode` and `single_latent`, `fusion: poe` is `fusion: {strategy: poe}`, `head: linear` moves
  from the latent space to the edge (`fed_by`), the placeholder `joint_decoder_v1` is a registered 2D
  decoder named `image_out`, and a fourth example shows `encoders` and `decoders`. §11's first
  bullet is resolved, and the §8 tree notes the three config modules. The README, the notes in
  `configs/experiment/` and `configs/data/`, the header of `configs/model/default.yaml`, the how-to
  `train.md` and `choose-a-fusion-strategy.md`, and the docstrings of `scripts/train.py` and
  `examples/03_signal_image_to_image.py` no longer say that a config cannot describe several latent
  spaces or the example 03 model.
- `tests/integration/test_config.py`: the test that expected `NotImplementedError` for
  `latent_mode: several` now checks that a `several` model without a `latent` section names
  `model.latent`, and new tests cover the other messages that name a field.
- A `single` model that sets `latent`, or a `several` model that sets `single_latent`, is now an
  error. It was accepted before, the unused section being ignored; no shipped file did it.
- Snake_case module names (roadmap P0-2, decision D-9, ADR 0023). The eight encoder and decoder
  modules named after their class are renamed with `git mv`: `encoders/one_d_cnn_encoder.py`,
  `one_d_cnn_residual_encoder.py`, `two_d_cnn_encoder.py`, `two_d_cnn_residual_encoder.py` and
  `decoders/one_d_cnn_decoder.py`, `one_d_cnn_residual_decoder.py`, `two_d_cnn_decoder.py`,
  `two_d_cnn_residual_decoder.py`. Class names, registry names (`1d_cnn_encoder_v1` and the rest)
  and behaviour are unchanged. Every old path (`global_vae.encoders.OneDCnnEncoder`, ...) still
  imports: it is a shim that returns the same class and emits a `DeprecationWarning` naming the new
  path, and it is removed in 2.0. The shims carry a file-level `# ruff: noqa: N999` with its reason,
  and no project-wide ignore is added. `encoders/__init__.py` and `decoders/__init__.py` import the
  new modules, so `import global_vae.encoders` does not warn. `scripts/gen_ref_pages.py` skips the
  shims, so the API reference documents each class once. Imports in `src/`, `tests/` and the
  docs were updated; spec §8 and §10 and the README now state that modules are snake_case, with
  the owner's approval of D-9. Older ADRs and changelog files keep the CamelCase names they were
  written with.
- The CI `lint` job is blocking (roadmap P0-2). `ruff check .` and `ruff format --check .` exit 0
  now that the eight `N999` findings are gone, so `continue-on-error` is removed. The README and
  `docs/getting-started.md` no longer call the job advisory.
- `validateRoutingGraph` reads the dimension rule from the assembler registry
  (`getAssemblerClass(name).requires_matching_dims`) and names no assembler. An assembler name that
  is not registered now raises the registry's `KeyError` (it lists the available names) from the
  validator, for a decoder that consumes several latent spaces; the config layer looks the names up
  first, so its errors keep the path of the field. `RoutingGraph.decoderInputs()` replaces the
  inverse it computed inline.
- `GlobalVae.forward` sends a decoder's latent vectors through its assembler whenever the decoder
  has one, including when a single latent space is present (it used to pass that vector straight
  to the decoder). The output is unchanged for `concat`, `sum` and `average`; `weighted_sum` and
  `attention` need it, since they transform the vector and the decoder is sized for their output.
  An assembler entry for a decoder that consumes no latent space is skipped, since there is nothing
  to size it from.
- The config layer reads the decoder width from the assembler's `output_dim` instead of running the
  assembler on zero tensors (ADR 0026). It builds that throwaway assembler with the random state
  saved and restored, so resolving a config does not shift the initialisation of the model.
  Assembler errors are reported as before, with the path of the `decoders_consume` entry.
- Spec §2.2 and §10 (assemblers with a matching-dimension rule, not only `sum` and `average`), §8
  tree (`weighted_sum.py` and `attention.py` no longer `(planned)`) and §11 (the assembler question
  is answered), the how-to `add-a-strategy.md` and `configure-latent-spaces.md`, and the roadmap
  (P1-4 done) describe the five assemblers and the constructor contract.
- `tests/unit/test_assemblers.py` builds every assembler from `input_dims` and covers the contract
  (flag, `output_dim`, `present`, subsets); `test_routing_graph_validation.py` checks the flag
  through the registry and that the validator names no assembler; the custom assemblers of the P1-3
  tests follow the new contract; `test_global_vae_construction_guards.py` covers a wrong or omitted
  decoder `latent_dim`.
- `ruff format` applies to three files that failed it: `scripts/visualize_latent.py` (a line over
  100 characters), `src/global_vae/config/model_schema.py` and
  `src/global_vae/training/callbacks/checkpoint.py` (no newline at the end of the file).
- Batch keys are modalities (ADR 0027). `GlobalVae.selectEncoderInputs` keeps the keys that are an
  input modality, which are the encoder names for any model without a shared encoder, so nothing
  changes for those. `GlobalVae.forward` reports an unknown key against `input_modalities`.
- `resolveDefaultInputSubsets` and `collectCrossModalReconstructions` (cross-modal reports),
  `exportCrossModalFigures`, `exportEvaluationFigures` and `scripts/visualize_latent.py` walk
  `input_modalities` and `output_modalities` instead of the encoder and decoder names. A shared
  encoder is one encoder over several modalities, so the reports pair its modalities, and a model
  with fewer than two input modalities still gets no cross-modal figure. The result is identical for
  every model that has no shared module.
- `registerEncoder`, `registerDecoder`, `getEncoderClass` and `getDecoderClass` are typed for the
  union of the one-modality and shared classes, with a bound type variable on the decorators so a
  decorated class keeps its own type. The base-class docstrings of `AbstractEncoder` and
  `AbstractDecoder` say they are the one-modality interfaces.
- Spec §2.1 (who builds the stream), §3 (shared encoder and decoder), §8 tree (`shared_base.py` in
  `encoders/` and `decoders/`), §9 (the shared decoder of the independent-latents example), §10
  (interfaces) and §12, the README and the roadmap (P1-5 done, P1-6 stage 3 delivered) describe the
  `E1` and `D1` rows as built. The spec edits are for the owner to confirm.
- The per-stage classes of `test_configuration_matrix.py` keep the topology checks only
  (no fan-out in stage 1, the edges and heads of stage 2, the single encoder call of stage 3, the
  order of the latent spaces an assembler receives). The shape, loss and gradient tests they each
  held moved to `TestEveryRow`, so stage 1, which never checked that the loss falls, now does.
- The roadmap marks P1-6 and Phase 1 as done, and P1-2 and P1-3, which were implemented without a
  status line, as done.
- The training log lines (`epoch <n> complete` and the per-step line) write a learning rate with
  six significant digits (`train/lr=1e-05`) where a loss keeps four decimals, which would have
  printed `1e-05` as `0.0000`. The loss plot of `scripts/visualize_latent.py` leaves the learning
  rate keys out, since they have their own figure.
- `KlStandardNormalRegularizer.forward` and `plotPerDimensionKl` call `computeKlPerDimension`
  instead of writing the formula each. Their results are unchanged.
- **Checkpoints are loaded with `weights_only=True` by default** (ADR 0034).
  `training.checkpoint.loadCheckpoint` and `Trainer.loadCheckpoint` take `weights_only: bool = True`,
  and a file the restricted unpickler refuses raises a `pickle.UnpicklingError` that names
  `upgradeCheckpoint`, `weights_only=False` and `--trust-checkpoint`. Checkpoints written now hold
  only plain values: the NumPy RNG state is a list (`_restoreRngState` still accepts an array), and a
  dataclass passed as `config` is stored as a dict by `dataclasses.asdict`, so `loadCheckpoint`
  returns a dict where it used to return the dataclass. **A checkpoint saved by an earlier version
  holds the RNG state as an array, so it needs `upgradeCheckpoint`, `weights_only=False` or
  `--trust-checkpoint`.**
- A step of `Trainer` is an optimizer update (ADR 0031). Nothing changes while
  `accumulate_grad_batches` is `1`.
- `configs/training/default.yaml` and `several_latents.yaml` list `best_checkpoint` before
  `checkpoint`, and spell out the new fields (`mixed_precision`, `accumulate_grad_batches`,
  `compile_model`, `compile_kwargs`, `resume_from`) at their defaults, so a Hydra override such as
  `training.resume_from=...` finds the key.

### Removed

- The `NotImplementedError` that `GlobalVae.__init__` raised for an encoder assigned to more than
  one latent space (the fan-out guard of ADR 0002, superseded by ADR 0024).
- The `dimension_locked_assemblers` argument of `validateRoutingGraph` (default `{"sum", "average"}`):
  the rule is the `requires_matching_dims` flag of each assembler class (ADR 0026). Code that passed
  its own set sets the flag on its assembler class instead. The no-argument construction of an
  assembler (`getAssemblerClass(name)()`) is gone with it: assemblers need `input_dims`.

### Fixed
