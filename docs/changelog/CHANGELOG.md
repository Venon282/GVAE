# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `docs.yaml` failed on `main` since the roadmap commit (`956cc97`): the previous roadmap was
  archived as `docs/archives/roadmap_1.md`, one directory deeper, and its 37 relative links
  (`adr/...` and `global-vae-project-specification.md`) pointed at `archives/`. They now start with
  `../`, so `mkdocs build --strict` passes again and the site deploy, which needs it, can run. The
  deploy never ran on a commit that carried 2.0.0, so the published site had no 2.0.0 changelog.
