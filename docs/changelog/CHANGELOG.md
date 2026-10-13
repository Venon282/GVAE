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

### Removed

- The `NotImplementedError` that `GlobalVae.__init__` raised for an encoder assigned to more than
  one latent space (the fan-out guard of ADR 0002, superseded by ADR 0024).

### Fixed
