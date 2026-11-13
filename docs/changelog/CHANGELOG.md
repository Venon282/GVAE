# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Latent distance registry (roadmap P6-3, [ADR 0053](../adr/0053-latent-distance-registry.md)). A package `losses/distances/`, the thirteenth registry: `AbstractLatentDistance` (an `nn.Module` with no state whose `forward` takes the `(mu, logvar)` pair of each latent space for one batch of the same samples and returns one value per sample, shape `(batch,)`) and `registerLatentDistance`, `getLatentDistanceClass` and `listRegisteredLatentDistances`. The base class checks the inputs and names what it found (a pair of floating point tensors of shape `(batch, dim)`, the same width and batch size in every space, at most `max_spaces` spaces, at least `min_batch_size` samples), casts half precision to float32, runs the subclass's `computeDistance` outside autocast and checks the shape of the result; each class declares `is_symmetric`, `is_zero_on_identical`, `is_stochastic`, `max_spaces` and `min_batch_size`. Six built-ins, one class per file: `mean_mse` (the squared gap of the means), `wasserstein2_diag` (the 2-Wasserstein distance of two diagonal Gaussians, squared by default; `squared: false` takes the root with a gradient that is 0 and finite at 0), `symmetric_kl` (clamped log-variances), `cosine` (the direction of the means), `mixture_jsd` (the Jensen-Shannon divergence of K posteriors against their mixture, an unbiased Monte Carlo estimate with reparameterized draws; `sum_over_spaces` gives MMVM's rate term, and it is the only distance that draws random numbers) and `info_nce` (contrastive, with a temperature; each sample against the other samples of the batch, so a batch of at least 2). The scale is ADR 0029's: a sum over the latent dimensions, averaged over the batch by the caller, so that a weight reads next to `beta`; `dimension_reduction: mean` is an option. The registry joins `tests/unit/test_registry_contract.py` and `tests/unit/test_latent_distances.py` checks every distance (non-negative, zero on identical inputs where declared, symmetric where declared, finite gradients at 0, float32 under mixed precision, the closed forms on hand-computed one-dimensional cases and the two Monte Carlo and contrastive ones against a quadrature and a hand-computed cross-entropy). Nothing calls a distance yet: the loss terms that use them are P6-4, and which one stage 3 trains with stays roadmap decision D-13. `docs/how-to/choose-a-latent-distance.md` says which to try first (`wasserstein2_diag`, with `mean_mse` as the ablation) and what each one ignores.
- Warm start of the pair model from two solo checkpoints (roadmap P6-2, [ADR 0052](../adr/0052-warm-start-from-solo-checkpoints.md)). `configs/model/signal_image_pair.yaml` is the pair model of the staged recipe: the encoders and the decoders of the solo models under the names they have there (`signal`, `image`, same registry names and kwargs), two latent spaces `z_signal` and `z_image` of the same width (16), each fed by its own encoder and read by its own decoder. `training.warm_start` (`[]` by default, a list of `WarmStartEntryConfig`: `checkpoint`, `modules`, `latent_space`, `check_manifest`, `min_active_units`) starts a run from the encoder and the decoder of each modality of a solo checkpoint. `training/warm_start.py` (`applyWarmStart`) checks every entry before it copies a weight: the stage manifest (a solo model, of the modality asked for, with a latent space of the pair model's width and the config each copied module was trained with), then the keys and shapes of each module (strict: a missing, an unexpected or a mis-shaped key is an error that names the key), and after the copy it probes the encoder on up to 256 samples of the training loader with the posterior mean and refuses a collapsed solo model (no active latent unit, `min_active_units`, default 1). A call that fails leaves the model as it was. With both files loaded, in eval mode, the pair model's `use_mean=True` reconstruction of each modality from its own input equals the solo model's bit for bit. The reader, `readCheckpointWeights` (`training/checkpoint.py`), returns the model weights and the manifest of a file in the safe mode and nothing else: the optimizer state, the counters, the history and the random-number state are not used, so stage 3 is a new run with its own optimizer, seed and step counter. `StageManifest` gains the optional `module_digests` (the fingerprint of each encoder and decoder config after resolution, `computeModuleDigests`), recorded by a solo run that sets `training.stage`, and `CheckpointMetadata` gains `module_digests` (`{}` for a file without it); a manifest written by ADR 0051 still reads. `scripts/train.py` applies the entries after the model and the data loaders are built and before the trainer is, and a run that resumes does not apply them. Beta stays constant in stage 3: `training: several_latents` has no schedule, so both spaces take `training.beta`. A run with an empty `warm_start` reads and probes nothing and is unchanged; `docs/how-to/train-in-stages.md` covers the stage and what it refuses.
- Stage manifest and solo recipes (roadmap P6-1, [ADR 0051](../adr/0051-stage-manifest-and-solo-recipes.md)). `training/stage_manifest.py` adds `StageManifest` (a free `stage` label, the `modalities` the encoders ingest, the `latent_spaces` with their widths and a `config_digest`), `buildStageManifest` and `computeConfigDigest`, the SHA-256 of the canonical JSON of the model config (sorted keys, so the order of a YAML file does not matter, and a value JSON cannot represent is an error and not a silent `str()`). `saveCheckpoint` takes `stage_manifest` and writes it under one key, `stage_manifest`, only when there is one, as plain values, so the file is still read by `weights_only=True`; `loadCheckpoint` returns it in four new optional fields of `CheckpointMetadata` (`stage`, `modalities`, `latent_spaces`, `config_digest`) that keep the defaults of "nothing recorded" for a file with no manifest, 2.0.0 files included. `training.stage` (`null` by default) is the switch: `buildTrainerFromConfig` (new optional `model_config`, which `scripts/train.py` passes) builds the manifest, `Trainer(stage_manifest=...)` holds it and `Trainer.saveCheckpoint` writes it, so `checkpoint`, `best_checkpoint` and `top_k_checkpoint` write it without a change. `configs/experiment/pair_solo_signal.yaml` and `pair_solo_image.yaml` are the two solo stages of the shipped synthetic signal plus image pair, and `docs/how-to/train-in-stages.md` gives the recipe and says that the pair stage (the warm start, the pair loss terms and `pair_align`) is P6-2 to P6-7. A run with no `training.stage` is unchanged.
- CI hardening, a docs toolchain plan and the causes behind I-14 (roadmap P5-9, findings I-14 and I-16, decision
  D-18, [ADR 0050](../adr/0050-ci-hardening-docs-toolchain-and-the-causes-behind-i-14.md)). `ci.yaml` runs every Monday
  at 05:17 UTC as well as on a push, so a new release of ruff, mypy, numpy, torch or the docs toolchain turns a run red
  between two pushes. A `branch-coverage` job, run by the schedule and by a manual dispatch only, writes the branch
  coverage of `latent/`, `models/` and `training/` to the summary of the run; it is a report with no floor. The `pytest`
  step prints the 15 slowest tests (`--durations=15`). `tests/unit/test_seed.py` covers `utils/seed.py` (11 tests,
  including the `deterministic=True` branch, `warn_only`, the `CUBLAS_WORKSPACE_CONFIG` variable and the reset by a
  later default call, with a fixture that restores every process-wide flag). `test_ci_workflow.py` guards the schedule,
  the floor and its comment, the durations, and the branch job. The ADR records a trial of ProperDocs (a drop-in: the
  same site byte for byte, no change to `mkdocs.yml`) and of Zensical (it cannot build the `reference/` tree yet), and a
  plan for the owner. It also records the two causes of I-14: the 1D transformer pair on one-hot text trails the text
  pair because its embedding starts at a tenth of the scale (62.6% against 99.0% of the tokens, 97.8% once the scale
  is the table's), and the series decoder's gap is a gap of convergence (0.52 against 0.78 of the variance at the 60 epochs of the
  benchmark, 0.86 against 0.84 at 240), not of seed or kernel, and part of it is the scale at which the latent
  vector enters the decoder.
- Loader facts in the data config (roadmap P5-7, finding I-12,
  [ADR 0049](../adr/0049-loader-facts-in-the-data-config.md)). `DataConfig.loader_facts`, a typed,
  optional block keyed by modality name (`LoaderFactsConfig`: `sample_rate`, `vocab_size`,
  `categorical_sizes`), states what the loader fixes and a model repeats. `checkDataContract` (new
  module `config/data_contract.py`) compares it with the arguments of the encoders and decoders of
  that modality, at the end of `loadExperimentConfig` and in `scripts/train.py` before the model is
  built, so a disagreement stops a run before its first step. The message starts with the path of
  the model field, as in ADR 0025: `model.modalities.audio.encoder.kwargs.sample_rate: the model
  declares 16000, but data.loader_facts.audio.sample_rate says the loader gives 22050 ...`. An
  argument left out is compared on the default of its class (a decoder with no `categorical_sizes`
  for a table that has blocks disagrees), an entry for a name the model does not have is refused,
  and a config with no `loader_facts` is not checked at all. `audio_waveform_encoder_v1` and
  `audio_waveform_decoder_v1` gain `sample_rate: int | None = None`, a declaration that changes no
  weight, no state dict key and no output. `StandardizeTransform` takes a list or a tuple of numbers
  for `mean` and `std` (a YAML list is one value per column) and turns it into a `float32` tensor,
  and its errors name the argument; the strict `xfail` test of `test_tabular_model_configs.py`
  passes and loses its marker. The three data files show the block as a comment, and the how-to
  pages for audio, text and a table say where the values go. A run that sets none of them is
  unchanged.
- Reconstruction loss and metric registries (roadmap P5-3,
  [ADR 0048](../adr/0048-reconstruction-loss-and-metric-registries.md)). `losses/reconstruction/` is
  a package with `AbstractReconstructionLoss` (an `nn.Module`), `registerReconstructionLoss`,
  `getReconstructionLossClass` and `listRegisteredReconstructionLosses`, and three class-level
  declarations read through the registry: `requires_matching_shapes`, `reads_nan_as_unobserved`
  and `countTerms`, the number of terms that the `sum` reduction multiplies by. Registered:
  `mse`, `l1`, `bce`, `smooth_l1` (unchanged, with a new `beta` for the last), `bce_with_logits`,
  `masked_mse`, `token_cross_entropy` (token ids or one-hot rows as targets), `chamfer` (clouds of
  different sizes) and `tabular_mixed` (squared error on the numeric columns, cross-entropy per
  categorical block, `categorical_sizes` read from the decoder). `training.reconstruction_loss`
  takes a name, an entry `{name, kwargs}` or a mapping from a reconstruction name to either. The
  parameters of a loss are in the optimizer, clipped, and saved in the checkpoint under
  `reconstruction_loss_state_dict` (a key written only when it is not empty, so a run with the
  classic losses writes the file it wrote). Metrics are a name to function table
  (`registerMetric`, `getMetric`, `listRegisteredMetrics`, `DEFAULT_METRIC_NAMES`) with the five
  default metrics, their `masked_*` forms, `chamfer`, `precision`, `recall`, `f_score`, `edge_auc`
  and `token_accuracy`. `evaluate(reconstruction_metrics=...)` takes `None`, a dict of functions,
  a list of names, or a mapping from a reconstruction name to either, and by default reports what
  fits the loss: the cloud scores for `chamfer`, the token accuracy for `token_cross_entropy`, the
  masked metrics (under the plain names) for a loss that reads `NaN` as unobserved, the five
  otherwise. The streaming evaluation accumulates the masked forms too. Every roadmap item now
  carries `**status**: todo` or `**status**: done`.
- Extra loss terms at the single loss site (roadmap P5-6,
  [ADR 0047](../adr/0047-extra-loss-terms.md)). A registry `losses/terms/` (`AbstractLossTerm`,
  `registerLossTerm`, `getLossTermClass`, `listRegisteredLossTerms`; it ships empty, the terms being
  the objectives of Phases 6 and 7) and `training.loss_terms`, a list of `{name, weight, schedule,
  kwargs}` whose weight is a number or a registered beta schedule. `StepLosses` gains a `terms`
  mapping (empty by default); each term is added to the total after its weight and logged as
  `train/loss/<name>` and `val/loss/<name>`. The trainer holds the terms: their parameters are in
  the optimizer, clipped with the model's, saved in the checkpoint under `loss_terms_state_dict` and
  restored by `resume_from`. A term runs in `float32` and is scaled by the accumulation like the
  total. A run with no `loss_terms` is bit-identical to 2.0.0.

### Changed

- The coverage floor is 97 (it was 95, with a comment that quoted the 96.25% of v1.2.0), one point below the 98.67% of statements measured on Python 3.13; the README and `docs/getting-started.md` say 97, and a test keeps the three
  places and the quoted value together (roadmap P5-9). The `docs` extra takes `mkdocs-material>=9.5,<10`.
- The experiment configs name the loss that fits their data (roadmap P5-3). `text_vae.yaml` uses
  `token_cross_entropy` (the decoder returns raw scores, `output_activation: null`; it trained with
  `bce` as a stand-in), `time_series_vae.yaml` uses `masked_mse`, so a series with gaps no longer
  gives `NaN`, `point_cloud_vae.yaml` uses `chamfer` and loses its placeholder loss and its warning
  header, and `tabular_vae.yaml` uses `tabular_mixed`. The weights are those of 2.0.0 and each
  loss is a mean over its terms, so `reduction: sum` means a sum over the terms of a sample for
  every loss. `trainer.reconstruction_loss_fn` of a trainer built from YAML is an instance of the
  registered class and no longer `torch.nn.functional.mse_loss`; passing a function to `Trainer`
  by hand is unchanged. The four classic names give the numbers of 2.0.0 (a test compares them).
- Loss settings of a shared decoder are keyed by modality (roadmap P5-4, finding I-1,
  [ADR 0045](../adr/0045-settings-keyed-by-reconstruction-name.md)).
  `training.reconstruction_loss`, `training.reconstruction_weight` and
  `training.reconstruction_reduction` take a mapping keyed by **reconstruction name**: the name of a
  decoder that handles one modality (so every existing config keeps its meaning and its numbers), or
  one modality of a decoder shared by several. A `shared_transformer` model can now weight, scale and
  score its signal and its image separately, from YAML, and the values reach the loss. Before, a
  mapping by modality was refused, a weight or reduction keyed by the shared decoder's name was
  accepted and silently not applied, and a loss keyed by it raised `KeyError` at the first step.
  The name of a shared decoder is now refused at build time, and the error lists the reconstruction
  names and the decoder names. The two `strict` `xfail` tests of `test_transformer_model_configs.py`
  are ordinary tests, with new ones for the three settings and a YAML end-to-end check.
- `GlobalVae` gains two public methods and `forward` is now `encode` then `decode` (roadmap P5-5,
  ADR 0046). `encode(inputs, use_mean=False)` runs the encoders, the heads, the fusion and the
  sampling and returns `latent_params` and `latent_samples` (typed `EncodeOutput`).
  `decode(latent_samples)` runs the assemblers and the decoders of every decoder whose latent spaces
  are present, with the rules for a missing latent space that `forward` has, and returns the
  reconstructions. A sample passed under the name of another latent space is decoded by that space's
  decoders (cross decoding), and a width that differs raises a clear error. `forward` returns the
  same three keys, raises the same errors and, under a fixed seed, gives the tensors of 2.0.0 bit
  for bit (a copy of the old loop is kept in the tests). `collectModalityPosteriorMeans` uses
  `encode`, so a paired retrieval no longer calls a decoder or an assembler.
- Specification sync (roadmap P5-2), **proposed to the owner for approval, since the specification is
  the ground truth**. Section 6 gains a subsection 6.3 that lists the six data kinds of 2.0.0
  (tabular, audio, text, time series, graphs, point clouds) with their batch keys, registry names,
  modules, configs and how-to pages (ADR 0039 to 0044). Section 7 records that the transformer
  backbones were delivered (ADR 0037) and what the literature check of the roadmap changed, as notes
  and not as commitments. Section 8 lists the encoders, decoders, `utils/`, `evaluation/`, `configs/`
  and `docs/archives/` as built. Section 11 moves the regularizer question to the resolved items
  (`free_bits_kl` and `mmd` shipped; learned priors are roadmap P4-2), states the planned change of the
  reconstruction loss to a registered name (P5-3), and adds the staged training protocol of Phase 6
  as an open question (D-13 to D-16). Not done on purpose: the Phase 6 subsection and the new registries in sections 8, 10 and 12, which wait for the owner's
  answers and for the code.

### Fixed

- `docs.yaml` failed on `main` since the roadmap commit (`956cc97`): the previous roadmap was
  archived as `docs/archives/roadmap_1.md`, one directory deeper, and its 37 relative links
  (`adr/...` and `global-vae-project-specification.md`) pointed at `archives/`. They now start with
  `../`, so `mkdocs build --strict` passes again and the site deploy, which needs it, can run. The
  deploy never ran on a commit that carried 2.0.0, so the published site had no 2.0.0 changelog.
