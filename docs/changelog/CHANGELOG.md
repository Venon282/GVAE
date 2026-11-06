# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

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
