# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Specification sync (roadmap P5-2), **proposed to the owner for approval, since the specification is
  the ground truth**. Section 6 gains a subsection 6.3 that lists the six data kinds of 2.0.0
  (tabular, audio, text, time series, graphs, point clouds) with their batch keys, registry names,
  modules, configs and how-to pages (ADR 0039 to 0044). Section 7 records that the transformer
  backbones were delivered (ADR 0037) and what the literature check of the roadmap changed, as notes
  and not as commitments. Section 8 lists the encoders, decoders, `utils/`, `evaluation/`, `configs/`
  and `docs/archives/` as built. Section 11 moves the regularizer question to the resolved items
  (`free_bits_kl` and `mmd` shipped; learned priors are roadmap P4-2), states the known limit of the
  per-modality loss settings of a shared decoder and the two planned changes (P5-3, P5-4), and adds the
  staged training protocol of Phase 6 as an open question (D-13 to D-16). Not done on purpose: the
  Phase 6 subsection and the new registries in sections 8, 10 and 12, which wait for the owner's
  answers and for the code.

### Fixed

- `docs.yaml` failed on `main` since the roadmap commit (`956cc97`): the previous roadmap was
  archived as `docs/archives/roadmap_1.md`, one directory deeper, and its 37 relative links
  (`adr/...` and `global-vae-project-specification.md`) pointed at `archives/`. They now start with
  `../`, so `mkdocs build --strict` passes again and the site deploy, which needs it, can run. The
  deploy never ran on a commit that carried 2.0.0, so the published site had no 2.0.0 changelog.
