# Roadmap

**Status:** living document, v0.1. Refresh the baseline (section 2) at each release.
**Baseline:** release `v1.2.0` (commit `6fed310`), measured on 2026-10-04.
**Authority:** the [project specification](global-vae-project-specification.md) is the ground
truth. When this file and the spec disagree, the spec wins, and the disagreement is a bug to fix
in one of the two files. Open questions from spec §11 are asked, never guessed (spec §12): every
item that depends on one says `needs D-n`, and section 7 lists them all.

---

## 1. At a glance

Milestone 1 (a single-modality signal VAE, spec §6.1) is complete. Milestone 2 (signal plus
image, fusion, missing modalities) runs end to end on synthetic data
(`examples/03_signal_image_to_image.py`). Only the real paired dataset is missing, and that
belongs to the caller by design (spec §6.2). Of the 8 configurations of spec §2.1, `EN-L1-DN` is
complete and tested. Three more can already be built (one is exercised only through an example,
two by no test at all). The four shared-encoder rows (`E1-*`) cannot be built, and neither can
anything that needs encoder fan-out or a latent head (spec §2.2).

The code is healthy: 1722 tests pass, mypy strict is clean on `src/`, and a scan of `src/` finds
no snake_case callable and no file with several classes in the registry families. Nothing enforces
it, though. CI only builds the documentation, ruff reports 1071 findings, and the part of the core
that Phase 1 is about to rewrite (`validateRoutingGraph`, the assemblers, the multi-latent path of
`GlobalVae`) is the part the tests reach least: the rejection paths of `validateRoutingGraph`,
every assembler `forward` and the `shared_private` preset are never executed. The plan therefore
starts with a safety net, then completes the configuration space, then moves on to training
quality, real data and larger backbones.

| Phase | Theme | Outcome | Target |
|---|---|---|---|
| 0 | Hygiene and safety net | CI enforces lint, types and tests; the spec's test requirements are met; packaging and docs are consistent | 1.3.0 |
| 1 | Complete the configuration space | latent heads, encoder fan-out, several latent spaces from YAML, all assemblers, all 8 configurations covered by one test | 1.4.0 to 1.6.0 |
| 2 | Training quality and scale | a documented loss scale, per-modality losses from config, mixed precision, accumulation, resume, a tracking backend | unscheduled |
| 3 | Real data and richer backbones | milestones 1 and 2 validated on the real datasets, image plots, transformer and ViT backbones | gated on data |
| 4 | Long-term research | discrete, hierarchical and sequential latents, learned priors, serving | gated, one ADR first |

Sizes used below: **S** up to half a day, **M** one to two days, **L** three to five days, **XL**
more than a week. They are rough estimates meant for ordering, not commitments. For scale, the
project shipped four releases in about five weeks.

**Needed from the owner first.** Phase 1 cannot start without D-1 (routing-graph schema in YAML)
and D-2 (what a shared encoder or decoder means when the modalities have different shapes). The
ruff clean-up wants D-9 (rename the eight CamelCase module files?) and D-10 (docstring rules for
tests?). Each decision has a proposed default in section 7, so Phase 0 can begin today.

---

## 2. Where the project stands

### 2.1 Health check

| Check | Result |
|---|---|
| Tests | 1722 passed, 1 skipped, 0 failed (1723 collected from 44 test modules) in 147 s on 2 vCPUs. The skipped test builds the real docs site and needs the `docs` extra. The suite ran against real PyTorch 2.14.1, which closes the follow-up that ADR 0017 and 0018 recommended (their first runs used a stand-in for torch). |
| Coverage | 96.25% of statements (3305 statements, 124 missed), measured in process, no branch coverage. Weakest: `latent/routing_graph_builders/` 21% (`shared_private.py` 0%), `latent/base.py` 76%, `assemblers/` 79%, `decoders/registry.py` 76%. |
| mypy strict | 0 errors in the 99 files of `src/`. Outside the current scope: `scripts/` and `examples/` (6 errors in 3 files) and `tests/` (90 errors in 17 files). |
| ruff check | 1071 findings: `tests/` 987, `src/` 67, `examples/` 11, `scripts/` 6. 795 of the test findings are missing-docstring rules (D101 to D104). 85 are auto-fixable. |
| ruff format | 14 files would change: 12 `.py` and 2 `.md` (the newer ruff also formats Python blocks inside Markdown). |
| Docs | `mkdocs build --strict` passes. Changelogs 1.1.1 and 1.2.0 are missing from the sidebar. |
| CI | One workflow: docs build and deploy on push to `main`. No lint, type-check or test workflow. |
| Size | 15.3k lines in 99 files under `src/`, 12.6k lines of tests, 5.6k lines of docs, 22 ADRs, 84 commits. |
| Conventions | An AST scan of `src/` finds 0 snake_case callables and 0 files with several classes in the registry families. No bare `print` in `src/`. 5 `type: ignore`, each with an error code. |

### 2.2 The 8 configurations (spec §2.1)

| Row | Today | Evidence |
|---|---|---|
| `EN-L1-DN` | Complete. Buildable from YAML. Tested with dummy and with real modules. | `test_en_l1_dn_default.py`, `test_signal_vae_milestone.py` |
| `EN-L1-D1` | Buildable through an explicit `RoutingGraph` (example 03 has one decoder whose target is not an input). No dedicated test. | `examples/03_signal_image_to_image.py` |
| `EN-LN-DN` and `EN-LN-D1`, one latent per encoder | Buildable and correct: checked by hand with real 1D and 2D modules and the `concat`, `sum` and `average` assemblers (right shapes, every parameter receives a gradient). No test executes it: the assembler call in `GlobalVae.forward` and every assembler `forward` are uncovered. | coverage report, audit script |
| `EN-LN-*` with encoder fan-out (shared plus private) | Not implemented. `GlobalVae.__init__` raises `NotImplementedError`, and the `shared_private` preset is 0% covered. | [ADR 0002](adr/0002-generalize-global-vae-to-routing-graph.md) |
| `E1-L1-D1`, `E1-L1-DN`, `E1-LN-D1`, `E1-LN-DN` | Not implemented. `AbstractEncoder.forward` takes one tensor and `GlobalVae` feeds `inputs[name]` to `encoders[name]`: nothing builds a shared encoder's input from several modalities, or splits a shared decoder's output back into modalities. | D-2 |

### 2.3 What exists

| Area | Implemented | Not implemented |
|---|---|---|
| Encoders and decoders | `1d_cnn`, `1d_cnn_resnet`, `2d_cnn`, `2d_cnn_resnet` (each as `_encoder_v1` and `_decoder_v1`) | transformer, ViT |
| Fusion | `poe`, `moe`, `concat_mlp`, `cross_attention`, plus the residual wrapper | none |
| Latent heads | none (there is no `heads/` package) | `AbstractLatentHead`, `identity`, `linear` |
| Assemblers | `concat`, `sum`, `average` | `weighted_sum`, `attention` |
| Regularizers | `kl_standard_normal`, `free_bits_kl`, `mmd` | learned or autoregressive prior |
| Beta schedules | `constant`, `linear_warmup`, `cyclical_annealing` | none |
| Transforms | `log`, `standardize`, `resample`, `ComposeTransform` | none (by design, nothing dataset specific) |
| Trainer | raw loop, modality dropout, gradient clipping, callbacks, seeds, checkpoint and restore (Python API) | mixed precision, gradient accumulation, multi-GPU, resume from `scripts/train.py` |
| Callbacks | `checkpoint`, `best_checkpoint`, `early_stopping`, `reduce_lr_on_plateau` | top-K best checkpoints, other LR schedules |
| Loggers | `csv`, `tensorboard` | Weights & Biases, MLflow |
| Evaluation | metrics, per-latent KL, cross-modal reports, figure export | retrieval and alignment metrics for paired data |
| Visualization | latent plots, 1D reconstruction overlays, loss curves | image comparison plots |
| Config | single latent (`EN-L1-DN`) through Hydra and dataclasses | `latent_mode: several`; a decoder or encoder without a partner of the same name |
| Docs | spec, 22 ADRs, 5 how-to guides, API reference, per-version changelog | none |

### 2.4 Deviations from the spec

| Spec | Requirement | State | Item |
|---|---|---|---|
| §2.2, §8 | Latent heads (`heads/`) | missing | P1-1 |
| §2.2 | Assemblers `weighted_sum` and `attention` | missing | P1-4 |
| §2.2, §10 | `validateRoutingGraph` checks head output dimensions | cannot exist yet; all four existing rejection paths are untested | P0-4, P1-2 |
| §9, §11 | `latent.mode: several` in config | `buildModelFromConfig` raises `NotImplementedError`; YAML also cannot express a decoder without an encoder of the same name, so example 03 is built in Python | P1-3 |
| §10 CI | lint, type-check and tests on every push | docs only | P0-1 |
| §10 standards | ruff, Google docstrings on every public member | 1071 findings; 15 public methods in `src/` have no docstring | P0-2 |
| §10 typography | no em dashes, no unicode arrows | 95 em dashes in 33 files, 4 arrows in 2 files (README and the spec itself) | P0-3 |
| §10 testing | unit tests for every registry, a `tests/unit/` tree, one test over the 8 configurations | encoders, decoders and assemblers have no registry-path tests; `tests/unit/` is empty; one row tested | P0-4, P0-5, P1-6 |
| §10 versioning | semantic versions, maintained changelog | `pyproject.toml` and `__version__` say 0.1.0, the tags say 1.2.0 | P0-6 |
| §10 tracking | Weights & Biases or MLflow | CSV and TensorBoard shipped ([ADR 0008](adr/0008-experiment-loggers.md)) | D-8, P2-4 |
| §10 logging | no bare `print` | none in `src/`; 9 calls in `scripts/` and `examples/` | D-11 |
| §10 typing | type hints everywhere | `src/` is strict clean; tests, scripts and examples are outside mypy | P0-8 |
| §8 layout | repository structure | drifted: preset paths, encoder and decoder file names, new `callbacks/`, `loggers/` and `beta_schedules/` trees | P0-7 |

Findings outside the spec, handled in P0-6: `numpy` is imported unconditionally by
`data/transforms/resample.py` (so `import global_vae.config` fails without it) but is not a
declared dependency; `pydantic` is declared but imported nowhere; there is no `LICENSE` file
although `pyproject.toml` declares MIT; there is no `py.typed` marker; ruff and mypy are not
pinned.

---

## 3. Sequencing rules

1. **Safety net before features.** Phase 1 starts after P0-1, P0-4 and P0-5. P0-2 and P0-3 should
   land first too, because they touch the same files and would otherwise cause conflicts.
2. **One architectural change, one ADR.** A changed decision is a new ADR that supersedes the old
   one, never an edit in place (spec §10). Phase 1 supersedes part of ADR 0002.
3. **Ask, do not guess.** An item that depends on a spec §11 question or on one of D-1 to D-12
   waits for the answer, or for the owner to accept the proposed default.
4. **Registry pattern, always.** New components follow the registry pattern, one class per file,
   with no edit to `GlobalVae` to add one (spec §10, §12). An item that seems to need a core edit
   is a design flag to raise, not to work around.
5. **Respect the milestone order** (spec §6.1). Milestones 1 and 2 work end to end, so Phase 1 is
   the "richer latent topologies" step that the spec places after them.
6. **Every item ships whole:** code, tests, docs, changelog line, green CI (section 10).

---

## 4. Phase 0: hygiene and safety net (target 1.3.0)

Suggested order: P0-1, P0-5, P0-4, P0-2, P0-3, P0-7, P0-6, P0-8. P0-1 comes first because it only
adds a workflow, and P0-5 comes before P0-4 so that new tests land in the right folder.

### P0-1. CI workflow (S)

**Why.** Spec §10 asks for GitHub Actions running lint, type-check and tests on every push. Today
a regression is only caught when someone runs the checks by hand.
**Do.** Add `.github/workflows/ci.yaml`, triggered on push and pull request, with three jobs:
`lint` (`ruff check`, `ruff format --check`), `types` (`mypy`) and `tests` (`pytest` with
coverage, Python 3.11 and 3.13). Install `.[dev,docs]` so the docs-site test no longer skips.
Fail under a coverage floor set one point below the measured value (95%), and raise it later. Make
the `lint` job advisory until P0-2 lands. Add a status badge to the README, and make the jobs
required checks on `main` (a GitHub setting, not a file).
**Done when.** The workflow is green on `main` for both Python versions, a pull request with a
deliberate type error is blocked, and the previously skipped test runs.

### P0-2. Ruff clean-up (M)

**Why.** 1071 findings, and `ruff>=0.4` is unpinned, so the counts change with the version.
**Do.**
1. Pin `ruff` and `mypy` in the `dev` extra with compatible-release specifiers (for example
   `ruff~=0.16` and `mypy~=2.4`).
2. Apply `ruff check --fix` (85 fixes) and `ruff format` (12 Python files), and review the diff.
3. In `src/` (67 findings), by hand: 24 D205 (summary line over several lines), 15 D102 (missing
   docstring on overriding methods such as `forward` and the `modality_name` property), 4 D417
   (undocumented parameters), 4 D200, 3 E501, 1 B006 (the mutable default `pool_kwargs={}` in
   `OneDCnnEncoder.__init__`), and the garbled class docstring and indentation in that same file.
   The 8 N999 findings wait for D-9.
4. In `scripts/` and `examples/` (17 findings): mostly D205.
5. In `tests/`: with the default of D-10, ignore D101 to D104 through
   `[tool.ruff.lint.per-file-ignores]`. The remaining 192 findings (98 D205, 77 D209 which are
   auto-fixable, 14 E501) are mechanical.
6. Every new `noqa` carries a one-line reason.

**Done when.** `ruff check .` and `ruff format --check .` exit 0 on the pinned versions, and the
CI `lint` job is switched from advisory to required.
**Depends on.** D-9, D-10 (defaults proposed; the rest can start now).

### P0-3. Typography sweep and guard (S)

**Why.** Spec §10 bans em dashes and unicode arrows. There are 95 em dashes in 33 files (README, the spec
itself, `docs/adr/index.md` with 15, most ADRs, how-to guides, 7 in `src/`) and 4 arrows in 2
files (README and the spec).
**Do.** Replace each with a period, a colon, parentheses or `->`, changing punctuation only. ADR
decisions are untouched, so this does not breach the "never edit an ADR in place" rule. Add
`tests/unit/test_typography.py`, which scans the tracked `.py`, `.md`, `.yaml` and `.toml` files
for the banned characters, so the rule cannot regress. The spec's own punctuation changes with the
owner's approval, since the spec is the ground truth.
**Done when.** The scan finds nothing and the test runs in CI.

### P0-4. Close the spec-mandated test gaps (M)

**Why.** Spec §10 requires unit tests per registry, validation of the routing graph at
construction, and one integration test across the 8 configurations. Coverage shows what is
missing.
**Do.**
- **(a) `validateRoutingGraph`.** One test per rejection (a latent space with no encoder, a latent
  space with no decoder, a decoder with several latents and no assembler, `sum` or `average` over
  different dimensions) plus accepting cases. None of the four rejections is executed today.
- **(b) Assemblers.** Value tests for `concat`, `sum` and `average`, and the registry error paths
  (65% covered).
- **(c) Registry contract test.** One parametrized test over all 9 registries (encoders, decoders,
  fusion, assemblers, regularizers, transforms, beta schedules, callbacks, loggers): register,
  look up, duplicate name raises `ValueError`, unknown name raises `KeyError` that lists the
  available names, `listRegistered*` is sorted. Encoders, decoders and assemblers have no such test
  today.
- **(d) Multi-latent end to end, stage 1 of the configuration matrix (P1-6).** The four `EN-*`
  rows without fan-out, with real 1D and 2D modules and each assembler: forward shapes, finite
  loss, a gradient on every parameter. This covers the assembler call in `GlobalVae.forward`
  (line 453), which no test reaches.
- **(e) Presets.** Tests for `buildSingleLatentRoutingGraph` and `buildSharedPrivateRoutingGraph`
  (the latter is 0% covered).
- **(f) Fan-out guard.** A test that `GlobalVae` raises `NotImplementedError` for an encoder that
  feeds two latent spaces. P1-2 inverts it.
- **(g) Shape guard.** Make `computeTotalReconstructionLoss` raise `ValueError` when a
  reconstruction and its target differ in shape. Today `F.mse_loss` only warns and broadcasts: a
  `(B, 1, H, W)` target against the 2D decoder's `(B, H, W)` output silently becomes a
  `(B, B, H, W)` comparison (reproduced during this audit). This is the only Phase 0 item that
  changes behaviour, so it gets a changelog line under "Changed".

**Done when.** Every rejection path of `validateRoutingGraph` is tested, coverage of `latent/` and
`assemblers/` is at least 95%, overall coverage does not drop, and the guard in (g) has a test.

### P0-5. Test layout and markers (S)

**Why.** Spec §8 and §10 expect `tests/unit/` and `tests/integration/`. `tests/unit/` is empty
and all test modules sit in `tests/integration/`, including pure unit tests (`test_conv_math.py`,
`test_beta_schedules.py`, ...). The 30 slowest tests take about 113 of the 147 seconds, mostly
because they run a script or an example in a subprocess, or run UMAP.
**Do.** Move unit-level modules to `tests/unit/` with `git mv` (history is kept). Register a
`slow` marker for the subprocess and UMAP tests, with `--strict-markers`. Document
`pytest tests/unit` and `pytest -m "not slow"` as the fast loops in the README.
**Done when.** `tests/unit/` holds the unit tests, CI runs everything, and the fast loop is
documented.

### P0-6. Packaging, versioning and licence (S)

**Do.**
- Declare `numpy` in `dependencies`, or make its import in `resample.py` lazy.
- Remove `pydantic` (no import anywhere; [ADR 0011](adr/0011-hydra-config-layer.md) chose
  dataclasses), or record why it stays.
- Single-source the version: bump to 1.2.0 now, read it from one place (`global_vae.__version__`
  through `[tool.setuptools.dynamic]`, or from package metadata), and add "tag, `pyproject.toml`
  and changelog agree" to a short release checklist in the contributor docs.
- Add `src/global_vae/py.typed` (declared as package data) so downstream mypy sees the type hints.
- Add a `LICENSE` file matching the `license = { text = "MIT" }` declaration.

**Done when.** `pip install .` in a clean environment imports `global_vae.config`, `pip check` is
clean, `global_vae.__version__` equals the tag, and `LICENSE` exists.
**Depends on.** D-12 (the copyright holder).

### P0-7. Documentation hygiene (S)

**Do.**
- Refresh stale notes: `src/global_vae/losses/NOTE.md` (says the reconstruction and KL modules are
  deferred and cites `computeKlLoss`), `data/NOTE.md` (describes `DataConfig.transforms` as a list;
  it has been a per-modality mapping since [ADR 0015](adr/0015-per-modality-data-transforms.md)),
  `visualization/NOTE.md` (says no image decoder exists) and `configs/experiment/NOTE.md` (cites a
  README section that no longer exists).
- Fix `configs/model/default.yaml`, which still names `resnet_encoder_v1` and says no image encoder
  exists. Replace it with a buildable signal plus image config (see P3-3), or delete it.
- README: `SignalEncoder` and `computeKlLoss` no longer exist; refresh "Extending beyond
  `EN-L1-DN`" and link this roadmap.
- Changelog navigation: add 1.1.1 and 1.2.0 to `docs/changelog/SUMMARY.md`, use one date format in
  file names (`CHANGELOG_1.1.1_2026_09_27.md` has underscores, the others hyphens), and add a test
  like `test_docs_adr_navigation.py` for the changelog.
- Spec §8 (repository layout) and §9 (illustrative names such as `signal_cnn_v1`) no longer match
  the code: presets live in `latent/routing_graph_builders/`, encoder files are `OneDCnn*` and
  `TwoDCnn*`, and `callbacks/`, `loggers/` and `beta_schedules/` exist. Update the spec with the
  owner's approval. Fix the same stale example name in the `encoders/registry.py` and
  `decoders/registry.py` docstrings.
- Spec §10 says "no bare `print`". `src/` has none, but 9 calls in `scripts/evaluate.py` and in
  examples 01 and 02 print their results. With the default of D-11, amend the rule to cover
  library code (ruff does not flag them: the `T20` rules are not selected).

**Done when.** Searching for `computeKlLoss`, `SignalEncoder` and `resnet_encoder_v1` outside the
ADRs and changelogs finds nothing, and `mkdocs build --strict` lists no orphan changelog page.
**Depends on.** D-11 (default proposed), and the owner's approval for the spec edits.

### P0-8. Type-check scope (S)

**Do.**
- Extend mypy to `scripts/` and `examples/` (6 errors in 3 files: optional-dict indexing in
  example 02, numpy returns in example 03, and the untyped `mkdocs_gen_files.Nav` in
  `gen_ref_pages.py`, which needs an override).
- For `tests/` (90 errors in 17 files, 46 of them missing annotations), use a relaxed override
  (`disallow_untyped_defs = false` for `tests.*`) rather than annotating about 900 tests.
- Settle `python_version = "3.11"` against the newest numpy stubs: with Python 3.13 and numpy
  2.5.3, mypy aborts on numpy's own stubs ("Type statement is only supported in Python 3.12 and
  greater"). Run the CI `types` job on a pinned interpreter and set `python_version` to match.

**Done when.** `mypy` with no arguments covers `src`, `scripts` and `examples` and exits 0 in CI.
**Depends on.** P0-1.

---

## 5. Phase 1: complete the configuration space (targets 1.4.0 to 1.6.0)

Goal: all 8 configurations of spec §2.1 build from config and are covered by one test. Suggested
releases: 1.4.0 for P1-1 and P1-2, 1.5.0 for P1-3 and P1-4, 1.6.0 for P1-5 and P1-6.

### P1-1. Latent heads (M)

**Spec.** §2.2, §3, §8, §10, §12.
**Do.** Add `src/global_vae/heads/`: `base.py` (`AbstractLatentHead`, `forward(mu, logvar)`
returning `(mu, logvar)`, built from `in_dim` and `out_dim`), `registry.py` (`registerHead`,
`getHeadClass`, `listRegisteredHeads`), `identity.py` (the default, requires `in_dim == out_dim`)
and `linear.py` (a learned projection per output; the initialisation is documented). Import each
in `heads/__init__.py`, one class per file. Add the new registry to the contract test of P0-4(c).
**Done when.** Registry, shape, gradient and identity-is-a-no-op tests pass, and
`how-to/add-a-strategy.md` covers heads.
**Depends on.** P0-4(c).

### P1-2. Encoder fan-out in `GlobalVae` (L)

**Spec.** §2.2 and §12 ("encoder fan-out goes through a Latent Head, never by giving
`AbstractEncoder` multiple output heads").
**Do.** Write a new ADR that supersedes the fan-out guard of
[ADR 0002](adr/0002-generalize-global-vae-to-routing-graph.md) (ADR 0002 itself stays unedited).
Extend `RoutingGraph` with a head per (encoder, latent) edge. `GlobalVae` builds one head module per
edge and applies it before fusion; graphs without heads run exactly as today. Extend
`validateRoutingGraph` with two rules: a head's output dimension must equal the target latent's
`dim` (spec §10), and an encoder may not feed two latent spaces through two identity edges, which
would give identical posteriors (the reason ADR 0002 added its guard). Make the `shared_private`
preset accept head choices so that it becomes buildable. Remove the `NotImplementedError` and
invert the P0-4(f) test.
**Done when.** A shared plus private model (signal and image: `z_shared` through PoE, two private
latents through linear heads) builds and trains a few steps with a falling loss, every parameter
(heads included) receives a gradient, the three latents differ, modality dropout still removes an
absent encoder's edges, and evaluation and cross-modal reports run unchanged.
**Depends on.** P1-1, P0-4. Exposing it in YAML needs D-1 (P1-3); the Python API can land first.

### P1-3. Config schema for several latent spaces (L)

**Spec.** §9 and §11 (first bullet).
**Do.** After D-1, add dataclasses for latent spaces, per-edge heads and per-decoder consumption
with an assembler, plus separate `encoders` and `decoders` sections keyed by instance name.
Today `ModalityConfig` forces one encoder and one decoder per modality name, which is why example
03 builds its model in Python. `buildModelFromConfig` then builds a `RoutingGraph` for
`latent_mode: several` and stops raising. Keep `latent_mode: single` and every shipped YAML file
valid. Ship one model config and one experiment config per spec §9 pattern. Hydra errors must name
the offending field.
**Done when.** The spec §9 examples 2 and 3 (adapted to the final schema) load, validate, build and
train through `scripts/train.py`, and the model of example 03 can be written in YAML.
**Depends on.** D-1, P1-2.

### P1-4. Assemblers `weighted_sum` and `attention` (M)

**Spec.** §2.2 and §11 (second bullet: none is deferred indefinitely).
**Do.** Both are learned modules, but `GlobalVae` builds assemblers with no arguments
(`getAssemblerClass(name)()`), so first give `AbstractAssembler` a constructor contract
(`input_dims` plus optional kwargs) and an `output_dim`. Use it to validate, or auto-fill, the
decoder's `latent_dim`: today the caller must add up the widths by hand for `concat`, and a wrong
value is accepted at construction and fails at the first forward with
`mat1 and mat2 shapes cannot be multiplied` (reproduced during this audit). Replace the hardcoded
`{"sum", "average"}` name set in `validateRoutingGraph` with a class-level flag read through the
registry, as spec §10 and §12 ask. Then implement `weighted_sum` (learned per-space weights, same
dimension rule as `sum`) and `attention` (cross-attention over the latent vectors, no fixed
dimensionality requirement), in the order of D-3.
**Done when.** Both are registered and tested (values, gradients reach their parameters, subsets
of latents), they work in an `EN-LN-DN` model end to end, and `validateRoutingGraph` no longer
names any assembler.
**Depends on.** P0-4(b). Reaching their kwargs from YAML needs P1-3.

### P1-5. Shared encoder and shared decoder, the `E1` and `D1` rows (XL)

**Spec.** §2.1.
**Why.** `AbstractEncoder.forward` takes one tensor, and `GlobalVae` feeds `inputs[name]` to
`encoders[name]`. The spec says the modalities are "tokenized/concatenated into one stream in and
out" but not who builds the stream.
**Do.** After D-2, write the ADR, then implement the chosen option. The proposed option A lets a
registered encoder declare the modalities it ingests and receive them as `dict[str, Tensor]`
(missing modalities omitted), and lets a registered decoder return `dict[str, Tensor]`. `GlobalVae`
flattens such outputs into `reconstructions` by modality, so the losses, evaluation and
visualization code, which already walk `reconstructions` by name, stay unchanged. The tokenisation
lives inside the concrete shared module, which is where transformer backbones (P3-5) need it anyway.
**Done when.** `E1-L1-DN`, `E1-L1-D1`, `E1-LN-DN` and `E1-LN-D1` build from config and pass the
matrix test with dummy shared modules, and modality dropout and cross-modal reports work with a
shared encoder.
**Depends on.** D-2, P1-2 (the `E1-LN` rows need fan-out), P1-3.

### P1-6. Configuration matrix test (M, three stages)

**Spec.** §10: an integration test that instantiates each of the 8 combinations end to end on dummy
tensors, with shape and gradient checks.
**Do.** `tests/integration/test_configuration_matrix.py`: one parametrized test with an explicit
list of the 8 row codes and a builder per row. Each case checks the reconstruction shapes, a finite
loss, a gradient on every parameter, and that a few optimizer steps lower the loss. Stage 1 (the
four `EN-*` rows without fan-out) is delivered by P0-4(d). Stage 2 adds the shared plus private
variants after P1-2. Stage 3 adds the four `E1-*` rows after P1-5.
**Done when.** 8 of 8 rows pass, and the test fails if a row is dropped from the list.

---

## 6. Phases 2 to 4

### Phase 2: training quality and scale

| ID | Item | Size | Depends on |
|---|---|---|---|
| P2-1 | Loss scale convention and beta guidance | M | P2-2, real data for the numbers |
| P2-2 | Per-modality reconstruction loss and weight from config | S | none |
| P2-3 | Trainer capabilities: mixed precision, accumulation, resume, top-K checkpoints, LR schedules | L | P0-1 |
| P2-4 | Experiment tracking backend | M | D-8 |
| P2-5 | Trigger criteria for a Lightning or Fabric migration | S | Phase 1 |

**P2-1. Loss scale convention.** The reconstruction term is a per-element mean (`F.mse_loss` with
the default reduction) while the regularizer is a per-sample sum over latent dimensions
(`kl_standard_normal`). Relative to a Gaussian likelihood of unit variance (`0.5 * sum of squares`
plus KL), the implemented objective therefore weights the KL term about `N / 2` times more than
`beta` says, with `N` the number of reconstructed values per sample: about 128 for a 256-point
signal and 512 for a 32 by 32 image. So `beta = 1` is not the usual ELBO, the useful `beta`
depends on the data shape, and the early posterior collapse that the default training config
guards against with a warm-up is made likelier. This is the open "loss weighting" question. Write
an ADR that states the convention, add a config-selectable reduction per modality (`mean` stays the
default, so nothing changes silently; a new default needs a minor release and a changelog note),
add a how-to on choosing `beta` with a sweep recipe, and report latent diagnostics (active units,
per-dimension KL) in the evaluation summary. The numbers themselves (D-6) need P3-1.

**P2-2. Per-modality loss and weight.** `TrainingConfig.reconstruction_loss` and
`reconstruction_weight` are single values shared by every modality, while
`computeTotalReconstructionLoss` and `Trainer` already accept per-modality dictionaries. Accept
`str | dict[str, str]` and `float | dict[str, float]`, validate the keys against the modalities,
test, and update `how-to/train.md`.

**P2-3. Trainer capabilities.** Each one opt-in and off by default, each covered by a smoke test in
the style of `test_trainer_smoke.py`:
- mixed precision (autocast with bf16 or fp16, and a gradient scaler when needed);
- gradient accumulation;
- resume from `scripts/train.py` through a `training.resume_from` field (`Trainer.loadCheckpoint`
  exists, but the script never calls it; define whether `num_epochs` is a total or an increment);
- top-K best checkpoints (listed as a natural extension in
  [ADR 0007](adr/0007-best-checkpoint-callback.md));
- LR schedules (cosine, warm-up) as registered callbacks next to `reduce_lr_on_plateau`;
- optionally `torch.compile`, and a safer default for checkpoint loading (it uses
  `weights_only=False`, with the trust caveat documented).

A resumed deterministic run must reproduce the uninterrupted loss curve within a stated tolerance.

**P2-4. Experiment tracking backend.** Spec §10 names Weights & Biases or MLflow. Depending on D-8,
either amend the spec or add one adapter as an optional extra: an `AbstractExperimentLogger`
subclass registered by name, with the import deferred as for TensorBoard, logging losses, latent
figures and reconstructions per run, tested in offline mode.

**P2-5. Migration trigger.** Spec §10 migrates to Lightning (or Fabric first) once the model design
is stable and multi-GPU needs are concrete. Record the triggers as a short ADR: Phase 1 complete,
and a real dataset or model that needs more than one GPU. Until both hold, no migration. When they
do, the `TrainerCallback` seam maps onto Lightning callbacks and the config stays as it is.

### Phase 3: real data and richer backbones

| ID | Item | Size | Depends on |
|---|---|---|---|
| P3-1 | Validate milestone 1 on real SAXS data (owner task) | M | the owner's data, P0-1 |
| P3-2 | Reference `loader_factory` for the paired signal plus image data | M | D-5 |
| P3-3 | Milestone 2 configs and cleanup of `configs/model/default.yaml` | S | P1-3 for translation variants |
| P3-4 | Image reconstruction plots | M | none |
| P3-5 | Transformer and ViT backbones | L each | none (P1-5 for shared modules) |
| P3-6 | Evaluation for paired data | M | P3-2 |
| P3-7 | Variable-length and masked signals | M | a confirmed requirement |
| P3-8 | Further modalities | per modality | a named dataset |

**P3-1. Milestone 1 on real data.** Spec §6.1 step 1 says "trained end to end on SAXS data"; the
repository only proves it on synthetic signals. Train with `scripts/train.py` and your own
`loader_factory`, then keep the config and a short results note: loss curves, per-dimension KL, a
latent plot, and reconstruction overlays through the inverse of the transforms. This gives the
first real values for D-6 and should run in parallel with Phase 1, since it only needs the data.

**P3-2. Paired loader.** Pairing stays the caller's job (spec §6.2, and `datamodule.py` is not
planned). After D-5, write a reference `loader_factory` for the real pairs, kept outside
`src/global_vae` (for example `examples/04_paired_loader.py`), and run `scripts/train.py` on it.

**P3-3. Milestone 2 configs.** [ADR 0017](adr/0017-2d-cnn-encoder-decoder.md) and
[ADR 0018](adr/0018-2d-residual-encoder-decoder.md) leave as next steps two model configs
(`image_single_latent.yaml` and `image_resnet_single_latent.yaml`, mirroring the signal ones) and a
paired signal plus image experiment file. Add those, a signal plus image PoE model config and
`configs/experiment/signal_image_vae.yaml` (this name is proposed here, by analogy with
`signal_vae.yaml`), with schema tests in the style of `test_yaml_shapes.py`. The variant where both
modalities are reconstructed works with today's schema; a translation variant (decoder target that
is not an input) needs P1-3.

**P3-4. Image plots.** A side-by-side original and reconstruction grid (deferred in
`visualization/NOTE.md`) and an image version of the cross-modal matrix, which draws 1D series
only today. Remove the example-local grid helper of example 03 once it exists. Tests use the Agg
backend.

**P3-5. Transformer and ViT backbones.** Spec §6 names a small transformer over the series and a
ViT. Each follows the new-modality checklist with zero core changes, which is the practical test
of the registry design: if a core edit looks necessary, flag it. They also give `cross_attention`
fusion a meaningful scale and are where the tokenisation of P1-5 lives.

**P3-6. Paired-data evaluation.** Cross-modal retrieval and alignment metrics (for example
recall@k between modality-specific posterior means), a per-modality metrics table, and streaming
versions of `mse`, `rmse` and `mae` for very large test sets (noted in `evaluation/NOTE.md`).

**P3-7. Variable-length and masked signals.** Spec §6 describes variable-length series.
`OneDCnnEncoder` is length-agnostic through adaptive pooling, but decoders rebuild a fixed
`output_length`. Confirm that variable length is a real requirement before designing padding, masks
and a mask-aware loss.

**P3-8. Further modalities.** Audio, tabular, text, time series, graphs, point clouds, each through
the checklist and only once a dataset is named. Tabular data is the cheapest way to prove the
genericity claim a second time.

### Phase 4: long-term research (spec §7)

Each item needs an explicit go-ahead and an ADR before any code.

- **P4-0. Interface readiness audit (M, ADR only).** List which core assumptions block the spec §7
  directions: a Gaussian-only posterior (`(mu, logvar)` in `AbstractEncoder`, `AbstractFusion`,
  `LatentSpace.reparameterize` and the regularizer signature), a bipartite routing graph with no
  latent to latent edge, a decoder that takes a single tensor, and a single-step training loop.
  Propose the smallest generalisation first (for example a `LatentDistribution` interface), with no
  code.
- **P4-1. Discrete or VQ latent codes.**
- **P4-2. Learned or autoregressive priors.** The regularizer registry is already open to them.
- **P4-3. Hierarchical or sequential latents.** This changes the spec: the routing graph is
  bipartite today.
- **P4-4. Serving.** Export of encoders and decoders (TorchScript or ONNX), batch inference, then an
  API. Nothing is scheduled until D-7 is answered.

---

## 7. Decisions needed

"Proposed" is what the work assumes if the owner simply says "go with the defaults".

| ID | Question | Options | Proposed | Blocks |
|---|---|---|---|---|
| D-1 | **Routing-graph schema in YAML** (spec §11, first bullet). Spec §9 is illustrative and, read against the code, leaves four points open: (a) heads are per edge, yet §9 puts `head:` on the latent space, which is ambiguous when several encoders feed it; (b) decoders need instance names independent of modality names (as `GlobalVae.__init__` already has, [ADR 0019](adr/0019-decouple-encoder-inputs-from-decoder-targets.md) and example 03); (c) beta per latent space already lives in `TrainingConfig.beta_schedules`; (d) the regularizer is already per latent space. | Extend `modalities`; or add separate `encoders`, `decoders` and `latent.spaces` sections | Keep `modalities` for the common case. Add `encoders`, `decoders`, `latent.spaces.<name>.{dim, fed_by, fusion, regularizer}` where `fed_by` is a list of encoder names or a mapping `encoder -> {head, kwargs}`, and `latent.decoders_consume.<decoder>.{spaces, assembler, kwargs}`. Do not duplicate beta. Update the spec §9 examples. | P1-3 |
| D-2 | **What a shared encoder or decoder means** for modalities of different shapes (spec §2.1, `E1` rows). Not listed in §11. | A: a shared encoder receives `dict[str, Tensor]`, a shared decoder returns one. B: a stream-adapter registry (`pack` and `unpack`) around one-tensor modules. C: weight tying only, or a caller-packed tensor under one key (possible today, but modality dropout and cross-modal reports cannot see inside the stream). | A. B adds a concept and a registry (and spec §12 asks for no new names for known concepts) for little gain. C stays as the zero-cost fallback. | P1-5, P1-6 stage 3 |
| D-3 | **Which assembler first** (spec §11, second bullet). | `weighted_sum` or `attention` first | `weighted_sum` (same dimension rule as `sum`, few parameters), then `attention` (needs an output width and a head count) | P1-4 |
| D-4 | **Further regularizers** (spec §11, third bullet). `free_bits_kl` and `mmd` exist ([ADR 0003](adr/0003-pluggable-latent-regularization.md)). | Keep open, or close | Mark the bullet resolved in the spec and track learned priors as P4-2 | none |
| D-5 | **Pairing mechanism** (spec §11, fourth bullet): which convention pairs a signal with an image (file name, sample ID, a manifest), and what happens to unpaired samples. | per dataset | Pair by sample ID in the caller's `loader_factory`; keep unpaired samples out unless single-modality batches are wanted | P3-2 |
| D-6 | **Beta hyperparameters** (spec §11, fifth bullet): warm-up length, cyclical period, per-space values. Empirical. | none yet | Decide after P2-1 and the real run of P3-1 | none |
| D-7 | **Serving target** (spec §11, sixth bullet): API, batch inference or edge. | unknown | Nothing scheduled until answered | P4-4 |
| D-8 | **Tracking backend.** Spec §10 names Weights & Biases or MLflow; the repository ships CSV and TensorBoard. | amend the spec; add W&B; add MLflow | Add MLflow as an optional extra (open source, easy to self-host), unless the team already uses W&B | P2-4 |
| D-9 | **CamelCase module names.** Eight modules are named after their class (`OneDCnnEncoder.py`, `TwoDCnnResidualDecoder.py`, ...). Ruff flags them (N999), and every other module is snake_case, as in the spec §8 layout. | rename; or keep and ignore N999 | Rename to snake_case (`one_d_cnn_encoder.py`) and leave re-export shims that warn for one minor release. Import paths are public API under semantic versioning, so removal waits for 2.0. | P0-2 |
| D-10 | **Docstring rules in tests.** 795 of the 987 findings in `tests/` ask for docstrings on test classes, methods and functions. | ignore D101 to D104 for `tests/**`; or document every test | Ignore them (module docstrings stay) | P0-2 |
| D-11 | **`print` in scripts and examples** (9 calls print results). | amend spec §10 to cover library code only; or move to `logging` | Amend: a command line tool that prints its result is not a logging problem | P0-7 |
| D-12 | **Copyright holder** (name and year) for the `LICENSE` file. | n/a | needs the owner | P0-6 |

---

## 8. Not planned

- `datamodule.py`, and any code in `src/` that loads, pairs or splits datasets (spec §6.2, a
  permanent boundary, not a gap).
- Modality-specific or dataset-specific transforms, SAXS included (spec §6.2 and §12).
- U-Net style encoder to decoder skip connections ([ADR 0014](adr/0014-residual-1d-encoder-decoder.md)
  rules them out: they bypass the latent).
- A Lightning or Fabric migration before the P2-5 triggers hold.
- Any component wired into `GlobalVae` by name: every new component goes through its registry.

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Phase 1 rewrites `GlobalVae`, today's least tested class | regressions in the default `EN-L1-DN` path | P0-4 and P1-6 stage 1 first; keep the no-head, no-fan-out path unchanged and tested |
| Config schema churn (D-1) | every YAML file and example changes | keep `latent_mode: single` and `modalities` valid; ADR before code; schema tests on every shipped YAML file |
| Everything is validated on synthetic data | design choices and hyperparameters untested on real SAXS or paired data | run P3-1 early, in parallel with Phase 1 |
| Tool drift (`ruff>=0.4`, unpinned `torch` and numpy) | a red CI with no code change; the counts in this file shift | pin ruff and mypy, run one scheduled dependency bump |
| Renaming modules (D-9) | breaks downstream imports | shims with a `DeprecationWarning`, removal at 2.0 |

## 10. Definition of done

Every item is done when:

1. The code follows spec §10: English, type hints, Google docstrings with `Args`, `Returns` and
   `Raises`, one class per file, camelCase callables, no em dashes and no unicode arrows.
2. A new pluggable component follows the registry pattern and is imported in its subpackage
   `__init__.py`; no core file is edited to add it.
3. Unit tests cover the component, and an integration test covers any change to routing or
   training.
4. An ADR records any architectural change. A changed decision is a new ADR, never an edit of an
   old one.
5. `docs/changelog/CHANGELOG.md` has a line under `[Unreleased]`, the docs (how-to or docstrings)
   are updated, and `mkdocs build --strict` passes.
6. CI is green: ruff, mypy, pytest and the docs build.

---

## Appendix A. How the baseline was measured

- **Date and commit.** 2026-10-04, commit `6fed310` (`v1.2.0`).
- **Environment.** Linux container, 2 vCPUs, no GPU, Python 3.13.16, torch 2.14.1 (the default PyPI
  wheel, run on CPU), numpy 2.5.3, ruff 0.16.8, mypy 2.4.0, pytest 9.1.1. Installed with
  `pip install -e ".[dev]"`, then `.[docs]` for the docs build.
- **Commands.**
  - `pytest --cov=global_vae --cov-report=term-missing -q -p no:cacheprovider --durations=30`
  - `ruff check . --output-format json` and `ruff format --check .`
  - `mypy --python-version 3.13`, and the same with `scripts examples` and with `tests`
  - `mkdocs build --strict`
- **Caveats.**
  - ruff: the project asks for `ruff>=0.4`. The counts come from 0.16.8, and 0.16.10 gives the same
    totals (1071 findings, 14 files to reformat), but other versions may not (P0-2 pins the
    version).
  - mypy: `pyproject.toml` sets `python_version = "3.11"`, which makes mypy abort on the numpy 2.5.3
    stubs under Python 3.13, so the runs used `--python-version 3.13` (P0-8).
  - Coverage is in process only: tests that run scripts and examples as subprocesses add nothing,
    and branch coverage is off.
- **Audit scripts** (not part of the repository). One built `EN-LN-DN` models from real 1D and 2D
  modules with each assembler and ran forward and backward; one imported ten public modules with
  `numpy`, `matplotlib`, `scipy`, `sklearn`, `umap`, `tensorboard` or `pydantic` blocked in turn;
  one scanned `src/` with `ast` for naming and one-class-per-file conventions. P0-4(d) turns the
  first into tests.
