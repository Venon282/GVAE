# Roadmap

**Status:** living document, first edition after 2.0. It replaces the roadmap of the 1.x series
(phases 0 to 3, delivered in 2.0.0). Refresh the baseline (section 2) at each release.
**Baseline:** release `v2.0.0` (commit `5685c3a`), measured on 2026-11-02.
**Authority:** the [project specification](global-vae-project-specification.md) is the ground
truth. When this file and the spec disagree, the spec wins, and the disagreement is a bug to fix
in one of the two files. Open questions from spec §11 are asked, never guessed (spec §12): every
item that waits for one says `needs D-n` in its heading, and section 8 lists them all with the
items each one blocks.

---

## 1. At a glance

2.0.0 delivers every item of the previous roadmap except Phase 4 (long-term research, which cannot
start without an explicit go-ahead) and P3-1, the owner task of validating on real data. All 8
configurations of spec §2.1 build from config and pass one matrix test. Ten registries hold 54
components, and eight kinds of data have a backbone: signal, image, tabular, audio, text, time
series, graph and point cloud. On Python 3.11 the full suite passes (4,987 tests, 98.51%
coverage), ruff and mypy are clean, and the docs build in strict mode (section 2.1).

What is missing falls into four groups:

1. **A closed loss and metric layer, and three seams the core lacks** (encode only, decode from a
   latent, more than one term at the loss site). Most of the findings that the Phase 3 modality
   work flagged and did not fix trace back to them (section 2.5). This is Phase 5.
2. **The training protocol the owner asked for:** train one VAE alone, train the other alone, then
   run a third training that minimizes the distance between their latent spaces (PairVAE, Lu and
   Jayaraman, [arXiv 2305.16467](https://arxiv.org/abs/2305.16467)). Nothing in the framework can
   do it today: a model cannot start from two solo checkpoints, no loss compares two latent
   spaces, and no evaluation compares two separate latent spaces. This is Phase 6.
3. **Recent VAE research with a good fit and a low cost.** A literature check (section 6.1,
   appendix B) found five components worth scheduling, each opt-in and behind a go-ahead. This is
   Phase 7.
4. **Phase 4,** developed in section 7. Its first step is an ADR that needs no decision, and it
   is cheaper once Phase 5 has removed two of the blockers.

Two things sit outside the code. No run on real data is recorded in the repository, so the beta
and weight values are still placeholders (D-6). And the `v2.0.0` tag points at a commit that
`main` does not have (P5-1).

| Phase | Theme | Outcome | Target |
|---|---|---|---|
| 5 | Close the open findings | pluggable reconstruction losses and metrics, encode and decode seams, extra loss terms at the single loss site, the findings of Phase 3 closed, the release on `main` | 2.1.0 |
| 6 | Staged training | train VAE A alone, VAE B alone, then align their latent spaces (PairVAE), with an evaluation across two separate latent spaces | 2.2.0 |
| 7 | Research integrations | five opt-in components from the 2023 to 2026 literature | 2.3.0 onwards, one go-ahead each |
| 4 | Long-term research (spec §7) | discrete latents, learned priors, hierarchical and sequential latents, serving | gated, one ADR first; 3.0.0 if the posterior interface changes |

Order of work: 5, 6, 7, then 4. Phase numbers are identifiers, not a schedule: Phase 4 keeps its
number from the previous roadmap, because ADR 0039 to 0044, the 2.0.0 changelog and a how-to page refer to P4-0.

Sizes used below: **S** up to half a day, **M** one to two days, **L** three to five days, **XL**
more than a week. They are rough estimates meant for ordering, not commitments. For scale, the 27
items of the previous roadmap shipped between 2026-10-04 and 2026-10-31.

**Status of an item.** Every item heading is followed by `**status**: todo` or `**status**: done`.
An item is `done` when its pull request is merged with its tests, documentation and changelog entry
(section 11). Change the line in the same pull request that finishes the item, so that the file
says where the work stands (`grep -n "status" docs/roadmap.md` lists them).

**Needed from the owner first.** Phase 5 can start today: only P5-1 (push rights) and the spec
edits of P5-2 (approval) need the owner. Phase 6 waits for D-13 (which distance), D-14 (what is
frozen), D-15 (one run per stage or one driver) and D-16 (equal latent widths). Each has a
proposed default in section 8, so "go with the defaults" is enough. Phases 7 and 4 start item by
item on an explicit go-ahead (D-17). D-6 (the beta numbers) needs the first real run (P5-8), and
D-7 (the serving target) blocks only the last two steps of P4-4. D-18 (the docs toolchain) has a
deadline, the end of life of Material for MkDocs on 2027-05-05, and nothing else waits for it.

---

## 2. Where the project stands

### 2.1 Health check

Measured on `v2.0.0` (appendix A). Nothing was changed to obtain these numbers.

| Check | Result |
|---|---|
| Tests | 4,987 passed and 3 xfailed, none skipped, on Python 3.11, in 10 to 18 minutes: 17 min 44 s on the first run after a container reset, 11 min 51 s and 10 min 5 s on the two later runs (the 2-vCPU container of appendix A). The three `xfail` tests are strict and each pins a finding of section 2.5: two for I-1 and one for I-12. Two warnings, both from dependencies: `umap` (`n_jobs` overridden by the seed) and a SQLAlchemy deprecation raised through MLflow (`noload`). |
| Coverage | 98.51% of statements (6,312 statements, 94 missed), measured in process, no branch coverage. The gate is 95. 117 files have complete coverage. The lowest are `utils/seed.py` (78%: the CUDA branch and the whole `deterministic=True` branch never run) and the abstract base classes (`encoders/base.py` 79%, `decoders/base.py` 82%, `fusion/base.py` 82%), whose missing lines are their `NotImplementedError` bodies. Then `utils/builders.py` (89%) and `encoders/one_d_cnn_encoder.py` (90%). |
| mypy | 0 errors in 266 source files: `src`, `scripts`, `examples` and `tests` (under a relaxed override), on Python 3.11, the interpreter that `python_version` names. |
| ruff | `ruff check .` and `ruff format --check .` both exit 0 on ruff 0.16.10 (348 files). |
| Docs | `mkdocs build --strict` exits 0 in 17 s (73 pages), and the two tests that build the real site pass inside the suite. The build output carries two notices about the toolchain: Material for MkDocs says it reaches end of life on 2027-05-05, and a plugin warns that a future MkDocs 2 may not support today's plugins and themes. The project pins `mkdocs>=1.6,<=1.6.1` and leaves `mkdocs-material>=9.5` unbounded above (I-16). |
| CI | `ci.yaml` runs `lint` and `types` on Python 3.11 and `tests` on 3.11 and 3.13, on every push and pull request (and on demand). `docs.yaml` builds the site on push to `main` and deploys it. The GitHub results for `v2.0.0` could not be read from the measuring session, and only Python 3.11 was run there, so the 3.13 job is unverified here. No scheduled run exists: a change in a tool or a dependency shows up at the next push only. |
| Size | `src/`: 25,336 lines in 146 Python files. `tests/`: 37,710 lines in 111 files (48 unit and 53 integration test modules, plus fixtures and helpers). `docs/`: 73 pages, 13,633 lines, 44 ADRs, 13 how-to pages. 46 YAML configs. 121 commits and 6 tags. |
| Release | The tag `v2.0.0` points at `5685c3a`. `main` is at `262c9fa`, one commit behind: the commit "v2" exists only under the tag (P5-1). |

### 2.2 What exists

Ten registries, one class per file, each imported by its subpackage `__init__.py` and covered by
`tests/unit/test_registry_contract.py`:

| Registry | Registered names |
|---|---|
| Encoders (12) | `1d_cnn`, `1d_cnn_resnet`, `1d_transformer`, `2d_cnn`, `2d_cnn_resnet`, `2d_vit`, `audio_waveform`, `point_cloud`, `shared_transformer`, `tabular_mlp`, `text_transformer`, `time_series`, each with the suffix `_encoder_v1` |
| Decoders (12) | the same, with `graph_cnn` in place of `time_series` (the time series model reuses `1d_cnn_decoder_v1`), each with the suffix `_decoder_v1` |
| Fusion (4) | `poe`, `moe`, `concat_mlp`, `cross_attention` |
| Latent heads (2) | `identity`, `linear` |
| Assemblers (5) | `concat`, `sum`, `average`, `weighted_sum`, `attention` |
| Regularizers (3) | `kl_standard_normal`, `free_bits_kl`, `mmd` |
| Transforms (3) | `log`, `standardize`, `resample` (and `ComposeTransform`) |
| Beta schedules (3) | `constant`, `linear_warmup`, `cyclical_annealing` |
| Callbacks (7) | `checkpoint`, `best_checkpoint`, `top_k_checkpoint`, `early_stopping`, `reduce_lr_on_plateau`, `linear_warmup_lr`, `cosine_lr` |
| Loggers (3) | `csv`, `tensorboard`, `mlflow` |

Three families are **not** registries, and section 2.5 shows where that hurts: the reconstruction
losses (a fixed table of four names in `config/training.py`: `mse`, `l1`, `bce`, `smooth_l1`), the
evaluation metrics (a fixed dictionary of five in `evaluation/metrics.py`) and the reduction
(`mean` or `sum`, ADR 0029).

Beyond the registries:

- **Model.** `GlobalVae` is built from a `RoutingGraph` (latent spaces, encoder -> latent edges
  with optional heads, latent -> decoder edges with optional assemblers, shared encoders and
  decoders). The posterior is a diagonal Gaussian passed as `(mu, logvar)`.
- **Training.** A raw PyTorch loop (`Trainer`) with callbacks, modality dropout, gradient clipping,
  seeds, resume from `scripts/train.py`, and opt-in bf16 or fp16, gradient accumulation and
  `torch.compile`. Checkpoints load in the safe mode (`weights_only=True`). One loss site,
  `Trainer.computeLosses`, sums a reconstruction term and a regularization term.
- **Evaluation.** Reconstruction metrics, per-latent diagnostics (active units, per-dimension
  KL), cross-modal reports, streaming metrics, a metrics table, paired retrieval and alignment
  metrics for **one shared latent space**, and figure export for series and images.
- **Data kinds.** Tabular ([ADR 0039](adr/0039-tabular-modality.md)), audio
  ([0040](adr/0040-audio-modality.md)), text ([0041](adr/0041-text-modality.md)), time series
  ([0042](adr/0042-time-series-modality.md)), graph ([0043](adr/0043-graph-modality.md)) and point
  cloud ([0044](adr/0044-point-cloud-modality.md)), next to the signal and image backbones, each
  added with no core change.

### 2.3 What the previous roadmap delivered

| Phase | Delivered | Recorded in |
|---|---|---|
| 0, hygiene | CI (lint and types on 3.11, tests on 3.11 and 3.13), pinned and clean ruff and mypy, typography guard, the tests the spec requires, `tests/unit` layout and the `slow` marker, packaging (single-sourced version, `py.typed`, `LICENSE`), documentation hygiene | changelog 2.0.0, [ADR 0023](adr/0023-snake-case-module-names.md) |
| 1, configuration space | latent heads, encoder fan-out, several latent spaces from YAML, `weighted_sum` and `attention`, shared encoders and decoders, the 8-row matrix test | ADR 0024 to 0027 |
| 2, training quality | loss scale convention, per-modality loss and weight, mixed precision, accumulation, resume, top-K checkpoints, LR schedules, safe checkpoint loading, MLflow, Lightning triggers | ADR 0029 to 0036 |
| 3, data and backbones | paired loader, model configs, image plots, transformer and ViT, paired-data evaluation, six further modalities | ADR 0037 to 0044 |
| 3, open | P3-1: no run on real data is recorded | P5-8 |
| 4 | not started, gated | section 7 |

The identifiers P0-1 to P3-8 in the changelog, in the code comments and here name items of the
previous roadmap, which stays readable with `git show v2.0.0:docs/roadmap.md`. This roadmap adds
P5-1 to P5-9, P6-1 to P6-8 and P7-1 to P7-5, and develops P4-0 to P4-4.

### 2.4 What is missing

| Gap | Why it matters | Item |
|---|---|---|
| Staged training: one VAE alone, the other alone, then a latent alignment | the owner's request; the solo VAEs can use every unpaired sample, and the scarce pairs are spent only on the alignment | Phase 6 |
| Encode-only and decode-from-latent paths on `GlobalVae` | cross decoding, the cost of retrieval evaluation, sampling from a prior, export | P5-5 |
| More than two terms at the loss site | any objective beyond reconstruction plus regularization needs an edit of `Trainer.computeLosses` | P5-6 |
| Pluggable reconstruction losses and metrics | six data kinds hit the fixed table | P5-3 |
| A run on real data | the beta and weight numbers (D-6), the pairing convention (D-5) | P5-8 |
| A calibrated likelihood | the default loss scale collapses every new modality on the synthetic benchmarks | P7-1 |
| Other multimodal posterior pooling | a third option beside `poe` and `moe` | P7-2 |
| Latent alignment baselines, and alignment of more than two models | the reference that tells whether a learned alignment is needed | P6-6, P7-4 |
| Discrete latent codes | spec §7 | P4-1 |
| Learned priors, and a way to sample (`GlobalVae` has no `sample`) | spec §7 | P4-2 |
| Hierarchical or sequential latents | spec §7 | P4-3 |
| Export and serving | spec §11 | P4-4 |
| A Weights & Biases adapter | spec §10 names it; MLflow shipped under D-8 | not planned, one more adapter of the same shape on request |
| Distributed training | the two triggers of [ADR 0036](adr/0036-lightning-migration-triggers.md) are not met | none until they are |

### 2.5 Issues to solve

These are the findings that Phases 2 and 3 flagged and did not fix, plus what this audit added.
Each row says which item closes it.

| ID | Issue | Effect | Item |
|---|---|---|---|
| I-1 | The loss weight, reduction and loss function of a **shared decoder** are checked against decoder names in `config/training.py` and looked up by modality in `computeTotalReconstructionLoss`. | A mapping by modality is refused; a weight or reduction keyed by decoder name is accepted and silently not applied; a loss function keyed by decoder name raises `KeyError` at the first step. Two strict `xfail` tests state the intended behavior. | P5-4 |
| I-2 | The reconstruction losses are a table of four names, and the shape check demands equal shapes. | No cross-entropy on token ids (text), no logits form of `bce` (graph), no spectral loss (audio), no masked loss (time series), no set loss (point cloud), no mixed loss per column (tabular). Three of them exist as Python functions and work only through `Trainer(reconstruction_loss_fn=...)`. | P5-3 |
| I-3 | The evaluation metrics are entry-wise and blind to masks. | `mse`, `rmse`, `mae`, `r2` and `pearson_r` are `NaN` on series with gaps, meaningless on point clouds, and average over padding for graphs. The modality scores (`computeCloudScores`, `computeEdgeScores`) live outside `evaluate()`. | P5-3 |
| I-4 | The default scale (`mean` reduction, beta 1) **collapses the models that were checked** on the synthetic benchmarks: 0 active latent units for the image, signal plus image, transformer, tabular, audio and text experiments. | Every new experiment file hand-sets `sum` with a weight (the time series, graph and point cloud ones too), and the weight is the lever (a weight `w` is a beta of `1 / w`). It is a placeholder tuned on synthetic data. | P7-1, P5-8 |
| I-5 | One loss per decoder. | A table cannot take a cross-entropy on its categorical columns and a squared error on its numeric ones. | P5-3 |
| I-6 | No encode-only or decode-from-latent path. | Paired retrieval pays one pass through the decoders per modality per batch. Cross decoding, sampling and export have no entry point. | P5-5 |
| I-7 | The loss site takes a reconstruction and a regularization term only. | An alignment, latent reconstruction or reference-alignment objective cannot be added without editing the trainer. | P5-6 |
| I-8 | `evaluation/paired_alignment.py` compares the posterior means of a latent space that **both** modalities reach. Pairs are assumed, not checked (row `i` of every modality must be one sample). | Two separate latent spaces cannot be compared, and a mismatched pair reads as chance. | P6-6, P6-8 |
| I-9 | The framework has no objective that aligns latent spaces. | On synthetic pairs, retrieval was at best weakly above chance (mean rank 28 to 51 of 103 against a chance of 52, `recall@10` 0.097 to 0.25 against a chance of 0.097, [ADR 0038](adr/0038-paired-data-evaluation.md)). | Phase 6 |
| I-10 | One latent per clip, window, text, graph or cloud. | The latent holds what the sample contains and not where. A sequence of latents needs a decoder that takes more than one vector. | P4-0, P4-3 |
| I-11 | `cross_attention` fusion attends over one token per modality. | It keeps a scale of two tokens whatever the backbone. Giving it patch tokens would change `AbstractFusion` and `AbstractEncoder`. | P4-0 |
| I-12 | YAML cannot write per-column `standardize` statistics (a list fails with a `TypeError` that does not name the cause; a strict `xfail` test pins it), and `config/data.py` has no place for a sample rate or a vocabulary size. | A model and a loader can disagree about the sampling rate or the vocabulary with no check. | P5-7 |
| I-13 | Graph node order dominates the result (shuffling the nodes took the AUC from 0.936 to 0.836), and the Chamfer loss is quadratic in the number of points (877 ms for a batch of 32 at 1,024 points). | A matching loss or an order in the loader is needed; large clouds must be sampled. | P5-3 (optional), loader notes |
| I-14 | Two weaker results were never isolated: `1d_transformer_*` on one-hot text (62 to 65% against 87 to 99.6% for the text pair) and `1d_transformer_decoder_v1` on series with gaps (0.53 against 0.80 for the CNN decoder). | The cause is unknown. | P5-9 |
| I-15 | Every result is on synthetic data. | Backbone choices, betas and weights are untested on real SAXS or paired data. | P5-8 |
| I-16 | The docs toolchain is ageing. The notice of Material for MkDocs in the build output says that it reaches end of life on 2027-05-05 (fixes and security updates until then), and a notice from a plugin warns that a future MkDocs 2 may not support today's plugins and themes. `pyproject.toml` pins `mkdocs>=1.6,<=1.6.1` and leaves `mkdocs-material>=9.5` unbounded above. | `mkdocs build --strict` runs in `docs.yaml` and inside the test job (two tests build the real site), so a toolchain break is a red CI and a stalled site with no code change. | P5-9, D-18 |

### 2.6 Deviations from the spec and housekeeping

| Item | State | Item |
|---|---|---|
| Release | `main` is one commit behind the tag `v2.0.0`: it still has the eight CamelCase shim modules, reports 1.x, and the docs workflow deploys a site without the 2.0.0 changelog. | P5-1 |
| Spec §6 and §8 | The text still lists the signal and image backbones. Each modality finding ends with "the spec §6 and §8 text is unchanged". | P5-2 |
| Spec §11, first bullet | "Which regularizer strategies to implement beyond the default" is still open, although `free_bits_kl` and `mmd` shipped (D-4). | P5-2 |
| Spec §10, tracking | MLflow ships instead of Weights & Biases. Accepted: the spec names both as examples ([ADR 0035](adr/0035-mlflow-experiment-logger.md)). | none |
| CI | The coverage floor is 95 and its comment quotes the 96.25% of v1.2.0, while the measured value is now 98.51%. No scheduled run. The 3.13 job is unverified in the measuring session. | P5-9 |
| Checkpoints | Files written by 1.2.1 or earlier need `upgradeCheckpoint` (changelog 2.0.0). | none |
| README | The two roadmap pointers stay valid. Its feature list needs a refresh at each release. | each release |

---

## 3. Sequencing rules

1. **Seams before features.** P5-5 (encode and decode) and P5-6 (extra loss terms) come before
   Phases 6 and 7, because each of their items needs one of the two.
2. **One architectural change, one ADR,** written before the code. A changed decision is a new ADR
   that supersedes the old one, never an edit in place (spec §10).
3. **Ask, do not guess.** An item that depends on a spec §11 question or on one of D-1 to D-18
   waits for the answer, or for the owner to accept the proposed default.
4. **Registry pattern, always.** New components (losses, distances, loss terms, quantizers,
   priors, fusion strategies) follow the registry pattern, one class per file, with no edit to
   `GlobalVae` to add one (spec §10, §12). An item that seems to need a core edit is a design flag
   to raise, not to work around. P5-5 and P5-6 are two such flags, raised here instead of worked
   around later.
5. **Opt-in and bit-identical.** A new capability is off by default, and a model or a config that
   does not use it behaves exactly as in 2.0.0. The configuration matrix test and golden-output
   tests are the guard. A change of default needs a minor release and a changelog note (the rule
   of [ADR 0029](adr/0029-loss-scale-convention.md)).
6. **Gates.** Phase 5 is scheduled work and starts today: no code item of it waits for a decision
   (P5-2, a spec edit, waits for approval). Phase 6 starts once the owner accepts the proposed
   defaults of D-13 to D-16. In both, each architectural item still gets its ADR before code.
   Phases 7 and 4 need an explicit go-ahead per item, then an ADR.
7. **Synthetic first, real data after.** Acceptance tests run on synthetic data, since the
   framework ships no dataset (spec §6.2). A result on real data is recorded as a note next to
   the config (P5-8), not as a test.
8. **Literature claims carry their evidence level** (appendix B). No item rests on a single
   unreviewed claim without a baseline comparison.
9. **Every item ships whole:** code, tests, docs, changelog line, green CI (section 11).

---

## 4. Phase 5: close the open findings (target 2.1.0)

Goal: every finding of section 2.5 that belongs in the core is closed, the core has the three
seams the next phases need, and the release is where users look for it. Suggested order: P5-4,
P5-5, P5-6, P5-3, P5-7, P5-9. P5-1, P5-2 and P5-8 are owner tasks that run in parallel.

### P5-1. Put the release on `main` (S, owner)

**status**: done
**Why.** The tag `v2.0.0` (commit `5685c3a`, "v2") is one commit ahead of `main` (`262c9fa`).
That commit removes the eight CamelCase shim modules, sets `__version__` to 2.0.0, moves the
unreleased entries of `CHANGELOG.md` into `CHANGELOG_2.0.0_2026-10-31.md` and syncs the README and
two how-to pages. Until `main` has it, the default branch reports 1.x, a clone of `main` does not
match the tag, and `docs.yaml`, which deploys from `main`, publishes a site without the 2.0.0
changelog.
**Do.** Fast-forward `main` to the tag, or open a pull request from it. Let `ci.yaml` and
`docs.yaml` run. Create the GitHub release from the 2.0.0 changelog. Read the 3.13 result.
**Done when.** `main` and `v2.0.0` point at the same commit, both workflows are green on it
(Python 3.11 and 3.13), and the published site lists the 2.0.0 changelog.
**Needs.** Push rights on the repository.

### P5-2. Spec and documentation sync (S, needs the owner's approval and D-4)

**status**: done
**Do.**
- Spec §6 and §8: list the six further data kinds, their files and their model configs (the
  facts are in ADR 0039 to 0044).
- Spec §11, first bullet: mark it resolved, since `free_bits_kl` and `mmd` shipped and learned
  priors are tracked as P4-2 (D-4).
- Spec §11, last sentence of the paragraph on the routing schema: it says that
  `reconstruction_loss` and `reconstruction_weight` are a decoder -> value mapping (ADR 0030). After
  P5-4 a shared decoder is keyed by its reconstruction names, and after P5-3 a loss is a registered
  name, so the sentence changes.
- Once D-13 to D-16 are answered, a short subsection for the staged training protocol of Phase 6
  (spec §6 or §2). The registries that Phases 5 and 6 add (reconstruction losses, loss terms,
  latent distances) go into §8, §10 and the list of §12, and the contract test of §10 grows with
  them.
- Spec §7: say that the fourth direction, larger transformer backbones, was delivered
  ([ADR 0037](adr/0037-transformer-and-vit-backbones.md)), and add what the literature check
  changed (section 7 of this file), without turning a direction into a commitment.
**Done when.** The spec and the code agree on modalities, layout and open questions, and the spec
change is flagged to the owner as a proposal, since the spec is the ground truth.

### P5-3. Reconstruction loss and metric registries (L)

**status**: done
**Why.** It is the last family of pluggable components that is a fixed table, although spec §10 and
§12 say pluggable components go through a registry. The modality work of Phase 3 found the table to
be the limit in six data kinds (I-2, I-3, I-5, I-13).
**Do (ADR first).**
1. A package `losses/reconstruction/` with `AbstractReconstructionLoss`, a base class that is an
   `nn.Module`, so a loss can own parameters (a learned noise scale, P7-1) that the trainer adds
   to its optimizer and its checkpoint (the mechanism of P5-6). The base class declares what the
   generic code needs to know, as class-level flags read through the registry (the pattern of
   `requires_matching_dims` on assemblers): whether reconstruction and target must have equal
   shapes (the check that 1.2.1 added, P0-4(g) of the previous roadmap, stays for every loss that
does not opt out), whether the loss
   reads `NaN` as "not observed", and the number of terms one sample's value averages over, which is
   what the `sum` reduction multiplies by. That number is the product of the axes after the batch
   for an element-wise loss, the number of tokens for a cross-entropy over `(batch, length,
   vocabulary)`, and `1` for a set loss that already sums over points.
2. Register the four existing losses unchanged (`mse`, `l1`, `bce`, `smooth_l1`), then the ones the
   findings asked for, one class per file, each wrapping code that exists and is tested:
   `bce_with_logits`, `masked_mse` (`computeMaskedSquaredError`), `token_cross_entropy`
   (`computeTokenCrossEntropy`, ids as targets), `chamfer` (different numbers of points are
   allowed), `tabular_mixed` (squared error on numeric columns, cross-entropy per categorical
   block, from the decoder's `categorical_sizes`). A multi-resolution spectral loss for audio is
   written only if a corpus asks for it.
3. `config/training.py` resolves names through the registry. A bare name keeps its meaning, and a
   `{name, kwargs}` entry carries arguments; the ADR settles how that entry is told from a
   decoder -> loss mapping.
4. Metrics: the metric set becomes a per-modality choice (the default stays the five current
   metrics, so nothing changes), the scores that live outside `evaluate()` become available by name
   (Chamfer, precision, recall and F score for clouds, edge AUC for graphs, token accuracy for
   text), and a metric that is mask-aware ignores a `NaN` target. The docstring of
   `evaluation/metrics.py` argues that metrics are not exclusive strategies, so the registry is a
   name -> function table with a declared default set, not a one-of choice.
**Done when.**
- `scripts/train.py` trains the text, time series, point cloud and tabular experiments with the
  right loss named in YAML. The placeholder loss and the warning header of
  `configs/experiment/point_cloud_vae.yaml` go away.
- The four existing names give the same numbers as 2.0.0 (a test compares them).
- The registry joins `test_registry_contract.py`.
- `evaluate()` is `NaN`-free on series with gaps and gives a meaningful score for a point cloud.
**Depends on.** P5-4, P5-6 (parameters of a loss reach the optimizer through the same mechanism).

### P5-4. Settings of a shared decoder keyed by modality (S)

**status**: done
**Why.** I-1: the three per-decoder settings disagree between two layers, so a shared decoder
cannot take a loss weight per modality, which is what a signal plus image model needs (one weight
cannot balance a signal and an image, see P3-3 in the changelog).
**Do.** Resolve the three mappings (`reconstruction_loss`, `reconstruction_weight`,
`reconstruction_reduction`) against the reconstruction names a decoder produces, which for a shared
decoder are its `output_modalities`. A one-modality decoder keeps its name, so existing configs stay
valid. Check that two modalities of one decoder may differ, and make the error for an unknown key
list both the decoder names and the reconstruction names.
**Done when.** A `shared_transformer` model takes a mapping per modality through YAML and the
values reach the loss. The two strict `xfail` tests in `test_transformer_model_configs.py` become
ordinary tests. Configs without a shared decoder give the same results as before.

### P5-5. Encode and decode seams (M)

**status**: done
**Why.** `GlobalVae.forward` is the only public path, and it always runs encode, fuse, sample,
assemble and decode. Retrieval evaluation pays a pass through the decoders per modality
(I-6). Cross decoding (Phase 6), sampling from a prior (P4-2), a latent reconstruction loss (P7-3)
and export (P4-4) each need to stop at the latent or to start from one. P4-0 would otherwise
find the same blocker.
**Do (ADR first).**
- `GlobalVae.encode(inputs, use_mean=False)` returns `latent_params` and `latent_samples` and stops
  there.
- `GlobalVae.decode(latent_samples)` runs the assemblers and the decoders of every decoder whose
  latent spaces are present, with the rules `forward` has today for missing latents (an assembler
  that cannot handle a subset leaves its decoder out).
- `forward` becomes `encode` then `decode` and returns the same three keys.
- Cross decoding needs no new method: pass the samples of one latent space under the name of
  another, `decode({"z_image": samples["z_signal"]})`. It needs equal latent widths, which the
  caller checks and a clear error reports.
- `collectModalityPosteriorMeans` switches to `encode`.
**Done when.**
- Under a fixed seed, the outputs of `forward` on every row of the configuration matrix and on a
  shared-module model equal those of 2.0.0 bit for bit (golden tensors, or a copy of the old loop
  kept in the test).
- `encode` followed by `decode` equals `forward`.
- Paired retrieval no longer calls a decoder (a test counts the calls).
- The mixed precision, accumulation and `torch.compile` paths of the trainer still pass.

### P5-6. Extra loss terms at the single loss site (M)

**status**: done
**Why.** `Trainer.computeLosses` returns `reconstruction + regularization`, and `StepLosses` has
those two fields (I-7). Every objective that Phases 6 and 7 add (a latent alignment, a cross
reconstruction, a latent reconstruction, a reference alignment) needs the same place, and the
single loss site is a rule worth keeping: training and validation must go through one function.
**Do (ADR first).**
- A registry `losses/terms/`: `AbstractLossTerm`, an `nn.Module` whose `forward` receives the
  model, the batch, the model outputs and the step, and returns a scalar. `registerLossTerm`,
  `getLossTermClass`, `listRegisteredLossTerms`.
- `training.loss_terms`, a list of `{name, weight, schedule, kwargs}`. A weight may follow a
  registered beta schedule, so a term can be ramped (the schedule registry already has `constant`,
  `linear_warmup` and `cyclical_annealing`).
- `StepLosses` gains a `terms` mapping (empty by default). The total adds each weighted term, the
  metrics log each one as `train/<name>` (and `val/<name>`), and the CSV, TensorBoard and MLflow
  loggers need no change.
- Terms with parameters are held by the trainer: in the optimizer, in the checkpoint, and restored
  by `resume_from` under the resume guarantee of ADR 0032.
- Terms are computed in `float32`, as the other losses are (ADR 0031), and scaled by accumulation
  like the total.
**Done when.** A dummy term with one parameter trains, shows up in the logs, is checkpointed and
resumed exactly, a run with no term is bit-identical to 2.0.0, and the registry joins the contract
test.
**Depends on.** none (P5-5 for terms that decode).

### P5-7. Config and data contract gaps (M)

**status**: done
**Why.** I-12.
**Do.**
- Accept a list for the `standardize` statistics (turn it into a tensor, and name the field in the
  error), which flips the strict `xfail` test in `test_tabular_model_configs.py`.
- Give `DataConfig` a typed, optional block for facts the loader owns and the model must agree
  with: the sampling rate of an audio modality, the vocabulary size of a text modality, the
  categorical block sizes of a table. A model that declares a different value fails at build
  time, with the field path at the start of the message (the convention of ADR 0025). The ADR
  settles where the facts live.
- Update the how-to pages that say "no place for" these values
  (`train-on-audio.md`, `train-on-text.md`, `train-on-a-table.md`).
**Done when.** The `xfail` test passes, a 16 kHz model with a 22.05 kHz data block fails before
the first step, and an experiment that sets neither value is unchanged.

### P5-8. First run on real data (M, owner, the paired part needs D-5)

**status**: todo
**Why.** P3-1 of the previous roadmap, carried over: spec §6.1 says the signal VAE is "trained end
to end on SAXS data", and the repository only proves it on synthetic signals (I-15). It also gives
the first real values for D-6, a first real test of the pairing convention (D-5), and the paired
data that Phase 6 needs for its acceptance on real pairs.
**Do.** Train with `scripts/train.py` and your own `loader_factory`. Keep the config and a short
results note: loss curves, per-dimension KL, active units, a latent plot and reconstruction
overlays through the inverse of the transforms. Then the same for the paired signal plus image
data, with a `sum` reduction and a few weights (the recipe of `how-to/choose-beta.md`).
**Done when.** The results note and the config are committed, and D-6 is answered or narrowed.

### P5-9. CI hardening, docs toolchain and two investigations (S)

**status**: todo
**Do.**
- Read the Actions results of `v2.0.0`, the 3.13 job included.
- Add a weekly `schedule:` trigger to `ci.yaml`, so that a new release of ruff, mypy, numpy or
  torch is caught between pushes instead of at the next one.
- Raise `COVERAGE_FLOOR` to one point below the value the CI log shows (98.51% in the measuring
  session, so 97), and refresh the comment that quotes the 96.25% of v1.2.0.
- Compare the duration of the test job with its `timeout-minutes: 30`. The suite alone took 10 to
  18 minutes on the 2 vCPUs of appendix A (install excluded), and its 15 slowest tests take 10 to
  48 s each (mostly tests that train a model to a stated accuracy on synthetic data, plus a UMAP
  projection, the two docs-site builds and a few script runs). A runner differs, so read the real
  log before deciding whether to shard.
- Add a branch coverage report, not a gate, for `latent/`, `models/` and `training/`. Cover the
  `deterministic=True` branch of `utils/seed.py` (78%) with a test that restores the global flags.
- Docs toolchain (I-16, D-18): put an upper bound on `mkdocs-material` in the `docs` extra, and
  try the build with ProperDocs or Zensical on a branch.
- Isolate the two weaker results of I-14: run the 1D transformer pair on the text benchmark with
  the embedding scale and the padding mask varied one at a time, and the transformer decoder on the
  time series benchmark with the seed and the kernel padding varied. Record the cause, or that it
  is not found.
**Done when.** The 3.13 result is known, the schedule runs, the floor and its comment are current,
the docs toolchain has a recorded plan (D-18), and I-14 has a recorded cause.

---

## 5. Phase 6: staged training and latent alignment (target 2.2.0)

Goal: train one VAE on one modality, train another VAE on another modality, separately, then run
a third training that minimizes the distance between their latent spaces. This is the protocol of
PairVAE (Lu and Jayaraman, University of Delaware,
[arXiv 2305.16467](https://arxiv.org/abs/2305.16467)), which links SAXS patterns and SEM images of
the same sample.

**What the paper says.** The text read for this roadmap describes two phases. In the "solo" phase
each VAE trains alone on its own data type. In the "pair" phase the two VAEs train together, with
cross-reconstruction losses added and with the encoders and decoders trained to maximize the
similarity of the two latent spaces. The pair phase needs paired data (the paper uses 72 SAXS and
SEM pairs from Doerk et al., 2023, and repeats the training over five random seeds); a solo phase
needs only its own modality. The text read does not name the distance, the form of the
cross-reconstruction losses or the weights, and the supplementary information was not accessible.
The distance is therefore an open decision (D-13), and every weight below is a placeholder until
measured.

**A precedent with a stated distance.** CADA-VAE ("Generalized Zero- and Few-Shot Learning via
Aligned Variational Autoencoders", [arXiv 1812.01784](https://arxiv.org/abs/1812.01784)) trains two
VAEs on two modalities with separate latent spaces. It adds a cross-alignment loss
(decode one modality from the latent code of the other) and a distribution alignment loss that is
the closed-form 2-Wasserstein distance between the two diagonal Gaussian posteriors of a sample.
That gives D-13 a published default.

**Why staged and not joint.** The framework already trains a joint model with a shared latent
space (`poe` or `moe` fusion). The staged route is not a replacement. It gives three things that
the joint route cannot:

- The solo VAEs use every sample of their modality, paired or not. The scarce pairs are spent on
  the alignment only.
- Each VAE is validated alone before anything is aligned, so a bad solo model is found early.
- A trained VAE can be paired with a new modality without retraining.

P6-6 measures both routes, and a Procrustes baseline, on the same split, so the choice rests on a
number.

| Stage | Trains | Data | Loss | Output |
|---|---|---|---|---|
| 1, solo A | encoder A, decoder A | every sample that has modality A | reconstruction A + beta KL | checkpoint A |
| 2, solo B | encoder B, decoder B | every sample that has modality B | reconstruction B + beta KL | checkpoint B |
| 3, pair | both, from checkpoints A and B | paired samples only | self and cross reconstruction + beta KL + lambda distance | checkpoint pair |

The objective of stage 3, with `z_m` a reparameterized sample of `q(z_m | x_m)`:

```text
L_pair = sum_m  w_m * rec_m( x_m, dec_m(z_m) )                    self reconstruction, as in stages 1 and 2
       + w_x * sum_(m != n)  w_n * rec_n( x_n, dec_n(z_m) )       cross reconstruction
       + beta * ( KL(q(z_a | x_a) || p) + KL(q(z_b | x_b) || p) )
       + lambda * d( q(z_a | x_a), q(z_b | x_b) )                 latent distance, per paired sample
```

**What the framework already has.** A solo stage is today's single-modality model
(`signal_single_latent.yaml`, `image_single_latent.yaml`). The pair model is an `EN-LN-DN` model:
two encoders, two latent spaces of the same width each fed by its own encoder through an identity
edge, and one decoder per modality that consumes its own space with no assembler. It is expressible
in the schema of ADR 0025 (an assembler is optional for a decoder that reads one space), which
P6-7 confirms with a test. What is missing is everything around it: P5-5 (decode a sample under
another latent's name), P5-6 (more loss terms), and P6-1 to P6-6 below.

**Questions this phase answers,** all through P6-6: does a learned alignment beat a closed-form
Procrustes map between the solo latents? Does the staged route beat joint training when pairs are
few? Which distance works? If Procrustes matches the learned alignment, the loss terms stay
available but leave the default recipe, and the recipe says so.

### P6-1. Stage manifest and solo recipes (S)

**status**: todo
**Why.** A solo stage is an ordinary run, so it needs no new training code. But nothing records
which modality it covers, which latent space and width it has, or which config produced it, and a
pair run that loads the wrong file would fail late or train on a mismatch.
**Do.**
- Optional fields on the checkpoint metadata (`CheckpointMetadata` in `training/checkpoint.py`): a
  free `stage` label, the modalities, and a digest of the model config. Plain values only (the
  safe loader of ADR 0034), absent by default, so 2.0.0 checkpoints stay valid.
- One experiment file per solo stage of the shipped synthetic pair, and
  `docs/how-to/train-in-stages.md` with the three-command recipe.
**Done when.** A solo run writes the manifest, a 2.0.0 checkpoint still loads, and the recipe runs
end to end on the synthetic pair (with P6-7).

### P6-2. Warm start: build the pair model from two solo checkpoints (M)

**status**: todo
**Why.** Stage 3 starts where stages 1 and 2 ended. In PairVAE the pair phase builds on the
solo-trained VAEs.
**Do (ADR first).**
- A pair model file, `signal_image_pair.yaml`: encoders and decoders under the same names as the
  solo models, two latent spaces `z_signal` and `z_image` of equal width, each decoder consuming its
  own space.
- `training.warm_start`, a list of `{checkpoint, modules}` entries. The loader reads only the model
  weights of a checkpoint, in the safe mode, copies the encoder and the decoder of each modality
  into the pair model, and checks shapes and the manifest of P6-1 (modality, latent width, config
  digest). The optimizer state is not transferred: stage 3 is a new run with its own optimizer, its
  own seed and a step counter at 0.
- The two solo latent spaces carry no weights and share a name (`z_fused` in the shipped
  single-latent models), so the config renames them. The per-space settings (regularizer, and
  `training.beta_schedules`, which is keyed by space name) follow the new names. In stage 3 the beta
  schedules are constant at the solo end value by default, since a warm-up restarted from step 0
  would pull a trained posterior back toward the prior.
- Strict by default: every parameter of a copied module must be loaded, none missing or unexpected,
  or the build fails and names the key.
**Done when.** With both solo checkpoints loaded, in eval mode, the pair model's `use_mean=True`
reconstruction of each modality from its own input equals the solo model's bit for bit. A wrong
shape, a wrong modality or a collapsed solo checkpoint (no active latent unit) is refused with a
clear message.
**Depends on.** P6-1.

### P6-3. Latent distance registry (M, needs D-13)

**status**: todo
**Why.** "Minimize the distance between the latent spaces" needs a distance, and the paper's is not
available (D-13). Several are known, cheap and comparable, which is what a registry is for.
**Do (ADR first).** A package `losses/distances/` with `AbstractLatentDistance`, an `nn.Module`
whose `forward` takes the `(mu, logvar)` pair of two latent spaces for one batch of samples and
returns one value per sample, and the usual three registry functions. Built-ins, one class per
file:

| Name | What it compares | Closed form | Note |
|---|---|---|---|
| `mean_mse` | the posterior means | yes | the simplest; it ignores the variances |
| `wasserstein2_diag` | the two diagonal Gaussians | yes: squared mean gap plus squared gap of the standard deviations | the CADA-VAE form; the paper takes the square root, which has no gradient at zero, so a `squared` option exists |
| `symmetric_kl` | the two diagonal Gaussians | yes | explodes when one variance is tiny, so it clamps |
| `cosine` | the direction of the means | yes | blind to scale |
| `mixture_jsd` | each posterior against the mixture of the two | Monte Carlo | the rate term of MMVM (Sutter et al., NeurIPS 2024) is M times the Jensen-Shannon divergence of the unimodal posteriors; extends to K spaces |
| `info_nce` | each sample against the other samples of the batch | no | contrastive, with a temperature; the objective that I-9 found missing; needs more than one sample per batch |

The scale follows ADR 0029: a sum over latent dimensions, averaged over the batch, which is the
footing of the KL term, so that `lambda` reads next to `beta`. A mean over dimensions is an option.
**Tests.** Non-negative; zero on identical inputs (not `info_nce`); symmetric where it claims to
be; equal to the closed form on hand-computed one-dimensional cases (the 2-Wasserstein distance
between N(0, 1^2) and N(1, 2^2) is the square root of 2); finite gradients at zero; float32 under
mixed precision.
**Done when.** The registry joins the contract test, all six pass those tests, and the how-to says
which to try first (proposed: `wasserstein2_diag`, with `mean_mse` as the ablation).

### P6-4. Pair-stage loss terms (M, needs D-13 and D-16)

**status**: todo
Two loss terms, registered through P5-6 and configured in YAML:

- `latent_alignment`: a list of `pairs` of latent space names, a `distance` from P6-3 with its
  kwargs, a weight `lambda` and an optional schedule (a ramp from zero protects the solo solutions
  at the start).
- `cross_reconstruction`: for each ordered pair, decode the target's decoder from a sample of the
  source's latent space (P5-5, the sample passed under the target's latent name) and compare with
  the target's batch tensor, using that modality's own loss, reduction and weight (P5-4). A
  `stop_gradient` option applies to either side.

**Rules.**
- Every batch must carry every modality of a pair. A batch that lacks one raises a `ValueError`
  saying that the stage needs paired batches, instead of skipping the term, because a skipped term
  would hide a loader bug.
- The two spaces of a pair must have equal widths, checked at build time (D-16).
- Modality dropout is rejected in stage 3: there is no fusion to be robust for.
- The distance acts on the posterior parameters of one sample from each modality, not on two
  independent samples.

**Done when.** On the synthetic pair, stage 3 lowers the latent distance over the epochs without
raising any self-reconstruction loss by more than a stated tolerance, each term shows in the logs,
and stage 3 resumes under the guarantee of ADR 0032.
**Depends on.** P5-4, P5-5, P5-6, P6-2, P6-3.

### P6-5. Freeze and learning-rate policy (M, needs D-14)

**status**: todo
**Why.** The pair stage can undo what the solo stages learned, especially with few pairs. PairVAE
trains both VAEs together. A gentler variant is useful in itself, and the same mechanism serves a
one-sided alignment, a frozen reference (P7-5) and a post-hoc prior (P4-2).
**Do.**
- A registered callback `freeze_modules`: module name patterns (`encoders.signal`, `decoders.*`)
  and an optional `until_epoch` after which they are released. A frozen module is also put in eval
  mode, since the CNN blocks use `BatchNorm` by default (`nn.BatchNorm1d` and `nn.BatchNorm2d` in
  `utils/conv_blocks.py`), whose running statistics would otherwise keep moving under the small
  batches that few pairs give.
- `training.param_groups`: a learning-rate factor per module pattern, so that pretrained modules
  run at a fraction of the rate. The trainer builds the optimizer from groups, and the default stays
  one group.
**Proposed default (D-14).** Nothing frozen, as in the paper. The options are the decoders frozen
(the encoders adapt to them), one side frozen (an anchor), and both VAEs frozen with only an
alignment head trained (the adapter of P6-8), the post-hoc route that sits between the Procrustes
baseline and the full stage 3.
**Done when.** The parameters and the buffers of a frozen module are bit-identical after training
(a test), an unfreeze epoch works across a resume, and the groups reach the optimizer and the
checkpoint.

### P6-6. Evaluation across two separate latent spaces, with baselines (M)

**status**: todo
**Why.** I-8. A distance that goes to zero proves nothing, since two collapsed latent spaces also
minimize it.
**Do.** `evaluation/latent_alignment.py`, and flags of `scripts/evaluate.py` that name a pair of
latent spaces. For each pair:
- every registered distance of P6-3 on held-out pairs;
- retrieval `recall@k` and mean rank in both directions, each with its chance level (reusing
  `computeRetrievalMetrics`);
- cross-reconstruction metrics next to the self-reconstruction ones, and the gap between them (the
  metrics table of P3-6 gets the cross rows);
- the latent diagnostics of both spaces (active units, per-dimension KL), as the collapse guard: a
  space with fewer active units than its solo stage is flagged;
- a pair-integrity check: the retrieval of the Procrustes baseline below must be above chance on
  the training pairs, or the pairs are suspect (I-8).

Baselines, on the same split:

| # | Baseline | What it tells |
|---|---|---|
| 0 | The solo latent spaces as they are | Expected at chance: two independent spaces |
| 1 | Post-hoc orthogonal Procrustes: one orthogonal map between the solo posterior means, fitted on the training pairs in closed form, applied to held-out pairs | The reference from the alignment literature (Maystre et al., [arXiv 2510.13406](https://arxiv.org/abs/2510.13406); Achara et al. for more than two models, [arXiv 2602.06205](https://arxiv.org/abs/2602.06205)). If it matches stage 3, the learned alignment is not needed |
| 2 | Stage 3, once per distance | The point of the phase |
| 3 | Joint training from scratch with a shared latent (`poe`) | The framework's existing route |

The reference paper used five random seeds on 72 pairs. Report the mean and the spread over
repeated random splits and seeds, never one run.
**Done when.** The report is written next to `results.json` for a pair model, the four baselines run
in one command on the synthetic pair, and the collapse guard fires on a deliberately collapsed run
(a test).
**Depends on.** P5-5 (encode only), P6-2.

### P6-7. Configs, script wiring, how-to and end-to-end test (M, needs D-15)

**status**: todo
**Do.**
- Experiment files for the three stages on the synthetic signal plus image pair (`pair_solo_signal`,
  `pair_solo_image`, `pair_align`), and the wiring in `scripts/train.py` of `training.warm_start`
  and `training.loss_terms`.
- `docs/how-to/train-in-stages.md`: the three commands, what each stage needs (stages 1 and 2:
  every sample of the modality; stage 3: pairs only), what to look at, which distance to try first.
- `tests/integration/test_staged_training.py`, through `scripts/train.py` in subprocesses (marked
  `slow` like the other script tests): stage 1 on the signal, stage 2 on the image, stage 3.
  It asserts: (a) the warm start is exact (P6-2); (b) the latent distance after stage 3 is below its
  value before; (c) `recall@10` is at chance before stage 3 and above it after, by a margin stated
  in the test; (d) self-reconstruction does not degrade beyond a stated tolerance; (e)
  cross-reconstruction beats the mean-image baseline; (f) stage 3 resumes; (g) every existing suite
  is unchanged.
- The pair model joins the configuration matrix as one more variant (the matrix lists its variants
  explicitly and fails when one is dropped).
**Done when.** The integration test passes in CI on 3.11 and 3.13, and the how-to commands run as
written.

### P6-8. Extensions (L, gated, one go-ahead each)

**status**: todo
- **More than two models.** `pairs` is already a list. Add an anchor mode (every model aligned to
  one), and for a symmetric treatment the multi-way scheme of Achara et al. as a baseline first,
  then as a loss (P7-4).
- **Unequal latent widths.** An adapter head used by the alignment term only (a `linear` head
  exists in `heads/`), so the spaces keep their own widths (D-16).
- **Few pairs, many unpaired samples.** Distribution-level alignment toward a frozen flow prior
  fitted on the other space (Li et al., [arXiv 2506.05240](https://arxiv.org/abs/2506.05240)), or
  the pair loss interleaved with unpaired reconstruction batches.
- **Pair integrity.** An optional `sample_id` entry in a batch, which the pair terms compare across
  modalities, so that a mismatched pair fails loudly instead of reading as chance (I-8).

---

## 6. Phase 7: research integrations (targets 2.3.0 onwards)

Goal: bring in the parts of the recent VAE literature that fit this framework, the way Phase 6
brings in PairVAE: as opt-in components behind the registry pattern, each with a baseline to beat
and an acceptance test. Every item needs an explicit go-ahead (D-17) and an ADR before code.

### 6.1 State of the art, as checked on 2026-11-02

The check covered the VAE literature of 2023 to 2026, plus the older papers that remain the
reference for a topic. Appendix B gives the method and its limits: it is a targeted survey, not a
systematic one. "Evidence" says what the page read for this roadmap showed.

| Line of work | What the literature shows | Evidence | Item |
|---|---|---|---|
| **The VAE as a tokenizer for latent diffusion** | Raising the latent dimension improves reconstruction but makes generation harder. Aligning the latent space with a pretrained vision foundation model resolves part of it (VA-VAE, [2501.01423](https://arxiv.org/abs/2501.01423)). Representation Autoencoders drop the learned encoder for a pretrained representation encoder with a trained decoder ([2510.11690](https://arxiv.org/abs/2510.11690)). | CVPR 2025; arXiv technical report (Oct 2025) | P7-5; a watch item |
| **Discrete tokenizers, simplified** | Finite scalar quantization rounds a few bounded scalars: no codebook, no auxiliary loss, no collapse ([2309.15505](https://arxiv.org/abs/2309.15505)). Binary spherical quantization binarizes a projection on a hypersphere, with a parameter-free implicit codebook and a bounded error ([2406.07548](https://arxiv.org/abs/2406.07548)). VQ itself is repaired by a rotation of the gradient through the quantizer ([2410.06424](https://arxiv.org/abs/2410.06424)) and by one linear layer over the codebook, SimVQ ([2411.02038](https://arxiv.org/abs/2411.02038)). | arXiv (no venue on the pages read) | P4-1 |
| **Multimodal VAEs beyond PoE and MoE** | MMVM keeps one encoder, decoder and latent per modality and ties them with a prior that is a mixture of the unimodal posteriors of the sample; its rate term is M times their Jensen-Shannon divergence ([2403.05300](https://arxiv.org/abs/2403.05300)). Hellinger pooling (Holder pooling with alpha 0.5, then moment matching to one diagonal Gaussian) trains without modality sub-sampling ([2601.06572](https://arxiv.org/abs/2601.06572)). SBM-VAE trains one VAE per modality, then a score-based model over the concatenated latents ([2305.15708](https://arxiv.org/abs/2305.15708)). Diff-MVAE adds diffusion decoders ([2408.16883](https://arxiv.org/abs/2408.16883)). OmniVAE aligns an audio-video VAE with a segment-level contrastive objective and distillation from pretrained encoders ([2607.23855](https://huggingface.co/papers/2607.23855)). | NeurIPS 2024; AISTATS 2026; TMLR 2024 listing; arXiv; preprint (Jul 2026) | P6-3, P7-2, P4-2 |
| **Aligning separately trained latent spaces** | PairVAE ([2305.16467](https://arxiv.org/abs/2305.16467)) and CADA-VAE ([1812.01784](https://arxiv.org/abs/1812.01784)) are the training protocols. When two embedding sets approximately preserve pairwise dot products, an orthogonal map aligns them closely, with a tight bound on the error ([2510.13406](https://arxiv.org/abs/2510.13406)). Multi-way alignment builds one shared orthogonal reference for K models with Generalized Procrustes Analysis, so the number of maps grows with the number of models and not with the number of pairs, and a post-hoc correction of the directional mismatch recovers the retrieval that strict isometric alignment loses against CCA ([2602.06205](https://arxiv.org/abs/2602.06205)). Relative representations describe a point by its similarity to anchors, which makes latents invariant to isometries ([2209.15430](https://arxiv.org/abs/2209.15430)). | arXiv; arXiv; ICML 2026; ICLR 2023 | Phase 6, P7-4 |
| **Priors** | The VampPrior mixes posteriors at learned pseudo-inputs ([1705.07120](https://arxiv.org/abs/1705.07120)). A diffusion model over amortized pseudo-inputs makes it practical for deep hierarchical VAEs ([2412.01373](https://arxiv.org/abs/2412.01373)). A frozen flow model can pull latents toward a target distribution with no likelihood evaluation ([2506.05240](https://arxiv.org/abs/2506.05240)). | arXiv; reviewed on OpenReview; arXiv | P4-2 |
| **Posterior collapse and the loss scale** | The sigma-VAE learns one decoder variance, or sets it analytically to the mean squared error, and so learns the reconstruction-versus-KL balance in one run ([2006.13202](https://arxiv.org/abs/2006.13202)). A latent reconstruction loss, `z` against the encoding of its decoding, pushes the decoder toward local bi-Lipschitz behavior ([2508.12530](https://arxiv.org/abs/2508.12530)). Collapse is analyzed as a phase transition ([2510.01621](https://arxiv.org/abs/2510.01621)) and attacked with contrastive critics ([2207.09535](https://arxiv.org/abs/2207.09535)). | ICML 2021; arXiv preprints | P7-1, P7-3 |
| **Hierarchical and sequential latents** | NVAE ([2007.03898](https://arxiv.org/abs/2007.03898)) and VDVAE ([2011.10650](https://arxiv.org/abs/2011.10650)) are the deep hierarchical references; a review organizes the dynamical (sequential) VAEs ([2008.12595](https://arxiv.org/abs/2008.12595)). | NeurIPS 2020; arXiv | P4-3 |
| **Export** | In the torch of this project (2.15), `torch.onnx.export` defaults to `dynamo=True`, which builds on `torch.export` and takes `dynamic_shapes`, and `torch.jit.script` emits a `FutureWarning`: deprecated, "switch to `torch.compile` or `torch.export`". | checked locally on torch 2.15.0 | P4-4 |

What the survey changes for the project:

- The recurring finding of Phase 3, a default scale that collapses the models (I-4), has a
  direct, old and well-supported answer in the sigma-VAE. It becomes P7-1.
- Multimodal VAEs are moving from fixed pooling rules toward alignment by regularization and toward
  generative models fitted after training. That agrees with Phase 6, and the distances of P6-3
  come from this line of work.
- Procrustes-style alignment gives a closed-form baseline that the learned alignment must beat,
  which is why P6-6 has it.
- Discrete latents and learned priors have simpler, better-founded options than in 2017 (FSQ, a
  diffusion prior), which lowers the cost of P4-1 and P4-2.

### 6.2 Selected and not selected

Selected, when an item (1) fits the registry pattern with no core change beyond P5-5 and P5-6,
(2) answers an issue of section 2.5 or a request of the owner, (3) has a baseline in the framework
to compare with, and (4) rests on evidence that is strong enough or is tested against that
baseline. P7-1 to P7-5 below.

Not scheduled, with the trigger that would reopen each:

| Idea | Why not now | Reopen when |
|---|---|---|
| Replace the learned encoder by a frozen pretrained one (RAE) | No pretrained encoder exists for SAXS signals, and the framework downloads nothing | A pretrained scientific-signal encoder is chosen (P7-5 then applies) |
| Diffusion decoders (Diff-MVAE) | A diffusion stack inside the decoder is outside what this framework is for | A named need for sharp image generation |
| Relative representations | They give similarity vectors with no variance, which does not fit a Gaussian posterior | Procrustes (P6-6, baseline 1) disappoints |
| Spherical VAE with feasible regions ([2603.10935](https://arxiv.org/abs/2603.10935)) | Independent authors; a revised version says the first was wrong; the guarantee is narrower than the abstract | Independent confirmation |
| Joint audio-video generation at the scale of OmniVAE | It shows contrastive alignment plus distillation at scale; nothing beyond the `info_nce` distance of P6-3 is needed | A generation use case |
| Collapse diagnostics from the phase-transition analysis | Read before P7-3 to see whether they sharpen the active-unit count | With P7-3 |

### P7-1. Calibrated reconstruction likelihood, the sigma-VAE (M)

**status**: todo
**Why.** I-4: every model that was checked collapses under the default scale, and each experiment
hand-sets a weight. The sigma-VAE (Rybkin, Daniilidis and Levine, ICML 2021) learns one Gaussian
decoder variance shared by the output, or sets it analytically to the mean squared error, so that
the reconstruction-versus-KL balance is learned in one run and beta does not need hand-tuning.
**Do (ADR first).** A reconstruction loss `gaussian_nll` (registered by P5-3) with one noise scale
per modality: `sigma: learned | analytic | fixed`. A learned scale is a parameter of the loss module,
held by the trainer (P5-3, P5-6). In the analytic mode sigma squared is the mean squared error of
the batch (a running average), detached. A floor `sigma_min` keeps it from collapsing to zero on near-noiseless data, and sigma is
logged per modality. The reduction is `sum` (a Gaussian log-likelihood sums over elements, ADR
0029) and the reconstruction weight stays 1.0, with a warning if it is not, so that beta reads as
the weight of an ELBO. It applies to continuous modalities only: categorical blocks, tokens, edges
and sets keep their own losses.
**Done when.** On the synthetic benchmarks that collapsed under the default scale (I-4), the
calibrated loss leaves at least as many active latent units as the hand-set `sum` baseline of each
experiment file, with a reconstruction R^2 within a stated tolerance of it, and with no hand-set
weight. The reference numbers are the ones recorded in ADR 0037 to 0044 and the how-to pages.
**Risks.** Sigma driven toward zero on near-noiseless synthetic data (the floor), instability at the
start (begin analytic), and the Gaussian assumption on bounded data.
**Depends on.** P5-3, P5-6. Interacts with D-6: it narrows the question to the choice of beta.

### P7-2. Hellinger pooling fusion (S)

**status**: todo
**Why.** A third way to pool unimodal posteriors beside `poe` and `moe`. The authors
(Vo and Valera, AISTATS 2026) report that it outperforms the state-of-the-art multimodal VAEs on
three benchmarks (PolyMNIST, CUB image-captions and bimodal CelebA), judged by generative coherence
and FID, and that it trains with no modality sub-sampling. Those are image and text benchmarks, so
what carries over to signal plus image has to be measured.
**Do.** `fusion/hellinger.py`, registered as `hellinger`: the Hellinger case of Holder pooling
(alpha 0.5, uniform weights over the modalities present), with the closed-form mean and variance of
the pooled density and a moment matching onto one diagonal Gaussian that drops the off-diagonal
covariance terms, following the paper's definition. Other values of alpha are not offered in the
first version; the ADR checks whether the paper's closed form covers them. It handles any subset
of modalities. No core change.
**Tests.** Pooling one posterior returns it, the result is symmetric in the experts, a
hand-computed two-expert case, gradients reach every expert, shapes.
**Done when.** It is registered and tested, and a benchmark on the synthetic signal plus image pair
compares `poe`, `moe` and `hellinger` (reconstruction of each modality from the other alone and from
both, with and without modality dropout). The fusion how-to says when to prefer it, or that the
benchmark found no case. The spec §4 row is a proposal for the owner.

### P7-3. Latent reconstruction loss against posterior collapse (S)

**status**: todo
**Why.** A second, cheap lever on collapse, with a different mechanism from P7-1.
Song, Kim and Lee ([arXiv 2508.12530](https://arxiv.org/abs/2508.12530)) add a penalty on the gap
between a latent code and the encoding of its decoding. It is a preprint, so it is accepted only
if it beats the two baselines the framework already has: `free_bits_kl` and the hand-set `sum`
weight of each experiment file.
**Do.** A loss term `latent_reconstruction` (P5-6) that decodes a latent sample and encodes the
result (P5-5), at a weight with an optional schedule. It costs one extra decode and encode per step,
which the benchmark records.
**Done when.** The term is registered and tested (a zero weight changes nothing, gradients flow),
and the benchmark on the collapsing models compares it with `free_bits_kl` and with P7-1. If it
adds nothing, the item closes as "no gain", and the note stays in the how-to.
**Depends on.** P5-5, P5-6.

### P7-4. Alignment of more than two models (M, gated on need)

**status**: todo
**Why.** Phase 6 aligns pairs. The multi-way scheme of Achara et al. (ICML 2026,
[arXiv 2602.06205](https://arxiv.org/abs/2602.06205)) adapts Generalized Procrustes Analysis to
build one shared orthogonal reference for K models, so the number of maps grows with the number of
models and not with the number of pairs. It finds that strict isometric alignment retrieves worse
than CCA, and adds a post-hoc correction of the directional mismatch.
**Do.** First the closed-form baseline of P6-6 extended to K models. Then, if it is not enough, a
loss term that ties K latent spaces to one learned reference.
**Gate.** A project with three or more modalities that need aligning. No such case is named, so the
item does not start.
**Depends on.** P6-6.

### P7-5. Alignment to a frozen reference encoder (M, gated on need)

**status**: todo
**Why.** VA-VAE aligns the latent space with a pretrained vision foundation model, and RAE builds on
a pretrained representation encoder. Where a pretrained encoder exists for a modality, the same
alignment term works here.
**Do.** A user-supplied frozen module (the framework never downloads one) wrapped as a reference,
and a loss term `reference_alignment` that applies a distance of P6-3 between a projected latent and
the reference's features of the same sample. The projection is a small head from `heads/`. When the
reference is another VAE, the frozen anchor of P6-5 already covers it.
**Gate.** A modality with a pretrained encoder that the owner wants to use.
**Depends on.** P5-6, P6-3, P6-5.

---

## 7. Phase 4: long-term research (spec §7)

Each item needs an explicit go-ahead and an ADR before any code (D-17). Spec §7 lists four
directions that are "not built now, but not blocked either": hierarchical or sequential latents,
discrete latents, learned priors, and larger transformer backbones. The last one was delivered in
Phase 3 ([ADR 0037](adr/0037-transformer-and-vit-backbones.md)). This section tests the claim "not
blocked" against the code for the other three, and develops each with what the literature now
offers. Serving comes from spec §1 and §11, which sequence it after the model is functionally
complete. It stays in this phase because the previous roadmap put it here.

Two things make Phase 4 cheaper than it was. Phase 5 removes two of its blockers: the missing
decode and sample path (P5-5) and the single loss term (P5-6). And the literature of 2023 to 2026
has simpler options for discrete latents and priors than the ones the spec had in mind (section
6.1).

A suggested order, once the owner opens the phase: P4-0, then P4-4 steps 0 and 1 (they need no
answer to D-7), then P4-2 in its post-hoc form, then P4-1, then P4-3.

### P4-0. Interface readiness audit (M, ADR only)

**status**: todo
**Goal.** Find which core assumptions block each direction of spec §7, and propose the smallest
generalization first, with no code.

**A preliminary reading of the code,** to be confirmed or corrected by the audit, with file and
line references in the ADR:

| # | Assumption in the code | Where | P4-1 discrete | P4-2 priors | P4-3 hierarchical, sequential | P4-4 serving |
|---|---|---|---|---|---|---|
| B1 | The posterior is a diagonal Gaussian passed as `(mu, logvar)` | `AbstractEncoder.forward`, `AbstractFusion.forward`, `AbstractLatentHead.forward`, `LatentSpace.reparameterize`, `GlobalVae.forward()["latent_params"]`, the diagnostics, the plots | blocks | partly | blocks | no |
| B2 | A regularizer receives `(mu, logvar)` and not the sample `z` | `AbstractLatentRegularizer`, `GlobalVae.computeRegularizationLoss` | partly (a codebook term needs the code) | blocks (a Monte Carlo KL needs `log p(z)` at the sample) | blocks (a conditional prior needs the parent's sample) | no |
| B3 | The routing graph is bipartite: encoders, then latent spaces, then decoders | `RoutingGraph`, `validateRoutingGraph` | no | no | blocks | no |
| B4 | A decoder takes one tensor and returns a mean, and a latent space has one integer `dim` | `AbstractDecoder`, `LatentSpace` | no | no | blocks (a latent per step, per token or per node) | no |
| B5 | One forward pass per step | `Trainer.computeLosses` | partly | partly | blocks (unrolled steps, carried state) | no |
| B6 | No decode-from-latent or sampling path | `GlobalVae` | no | blocks (a prior needs `sample`) | partly | blocks |
| B7 | No test says which of the 24 encoders and decoders can be exported | the backbones | no | no | no | partly |

B6 and the "one loss term" half of the loss-site question are removed by P5-5 and P5-6. B4 is also
the "single-tensor decoder" item that the audio, text, time series, graph and point cloud findings
point to, and that `docs/how-to/train-on-audio.md` names as P4-0. B1 and B2 decide whether Phase 4
is a minor or a major release.

**Proposal for the ADR to test, not a decision.** A small `LatentDistribution` interface (`mean`,
`rsample()`, `logProb(z)`, `kl(prior)`) with `DiagonalGaussian`, a class that still unpacks as
`(mu, logvar)`, so every `mu, logvar = params` in the code, the tests and users' code keeps working.
Encoders keep returning `(mu, logvar)`. A latent space declares a `kind` (`gaussian` by default),
and a regularizer receives the distribution and the sample. With tuple compatibility the change is
a minor release; changing what `AbstractEncoder.forward` returns would be 3.0.0.

**Deliverable.** One ADR with: the blocker table verified against the code; the proposed interface
and a migration path; for each of P4-1 to P4-4, what it needs and in what order; the semantic
versioning impact; and the list of golden-output tests to write before any change, so that "today's
behavior" is pinned.
**Done when.** The owner has accepted or amended the ADR. No code changes.

### P4-1. Discrete or VQ latent codes (L, one quantizer at a time)

**status**: todo
**Why.** Spec §7. A discrete latent gives symbols that can be counted, compared and modeled by a
second-stage model, and it cannot suffer the KL collapse of a Gaussian latent. Whether the project
needs one depends on a use case; the gate is the owner's go-ahead.

**What the literature offers.**
- **FSQ** ([2309.15505](https://arxiv.org/abs/2309.15505)) projects the latent to a few dimensions
  and rounds each to a small fixed set of values, so the implicit codebook is the product of the
  sets. No codebook, no commitment or codebook loss, and no collapse.
- **BSQ** ([2406.07548](https://arxiv.org/abs/2406.07548)) projects onto a hypersphere and
  binarizes, so the implicit codebook has no parameters and the quantization error is bounded.
- **Repairs to VQ:** the rotation trick for the gradient through the quantizer
  ([2410.06424](https://arxiv.org/abs/2410.06424)), and SimVQ, a linear layer over the codebook so
  that every code is updated ([2411.02038](https://arxiv.org/abs/2411.02038)).

**Plan.** FSQ first, since it is the smallest: no parameters, no extra loss, a straight-through
rounding. Then an EMA codebook with dead-code restarts and the rotation trick, if a learned codebook
is wanted. BSQ and SimVQ only if FSQ or VQ shows a need.

**Interface (after P4-0).**
- A latent space with `kind: fsq` and the number of levels per dimension. The rate is fixed by the
  levels (the sum of their base-2 logarithms), so FSQ has no regularizer.
- Fusion happens **before** quantization. `poe` and `moe` fuse Gaussians, so a discrete latent
  takes a single encoder or `concat_mlp` or `cross_attention` fusion.
- A VQ codebook adds a commitment term and a usage term, as loss terms (P5-6).
- Diagnostics in `evaluation/`: code usage, perplexity, dead codes.
- Decoders are unchanged: they read the quantized vector.

**Measure.** Reconstruction at equal rate against the Gaussian latent with `free_bits_kl`, code
usage, and the absence of collapse.
**Done when.** An `EN-L1-DN` model with an FSQ latent trains through `scripts/train.py` with a
falling loss and non-zero code usage, the Gaussian path is bit-identical to before, and a how-to
exists.
**Depends on.** P4-0, P5-6.

### P4-2. Learned or autoregressive priors (L)

**status**: todo
**Why.** Spec §7. The standard normal prior leaves "holes" that a decoder has never seen, and a
learned prior also makes sampling meaningful. The regularizer registry is already open to priors,
but a regularizer receives `(mu, logvar)` only (B2), and `GlobalVae` has no way to sample (B6).

**What the literature offers.**
- A **learnable mixture** and the **VampPrior** (a mixture of the posteriors at learned
  pseudo-inputs, [1705.07120](https://arxiv.org/abs/1705.07120)).
- A **diffusion-based VampPrior** for deep hierarchical VAEs, which amortizes the pseudo-inputs and
  reaches competitive likelihoods with fewer parameters ([2412.01373](https://arxiv.org/abs/2412.01373)).
- **Flow or autoregressive priors** over the latent, and a **frozen flow prior used as a
  regularizer** that pulls latents toward a target distribution with no likelihood evaluation
  ([2506.05240](https://arxiv.org/abs/2506.05240)).
- **A prior fitted after training.** SBM-VAE trains one VAE per modality, then a score-based model
  over their concatenated latents ([2305.15708](https://arxiv.org/abs/2305.15708)).

**Two modes.**
- **Post-hoc (first).** Freeze the VAE and fit the prior on its latents in a second stage, reusing
  the warm start (P6-2) and the freeze policy (P6-5). It changes nothing in the VAE, so the matrix
  test stays valid, and it extends to several latent spaces: a joint prior over `[z_a, z_b]`, which
  after Phase 6 also gives coherent cross-modal generation.
- **Joint.** The prior's parameters train with the VAE, with a one-sample Monte Carlo KL,
  `log q(z | x) - log p(z)`, which needs the regularizer to receive the sample (P4-0).

**Interface (after P4-0).** `AbstractPrior`, an `nn.Module` with `logProb(z)` and `sample(n)`, in
its own registry. A regularizer `prior_kl` that takes a prior. `GlobalVae.sample(n, prior)`, built
on `decode` (P5-5). The standard normal and its closed-form KL stay as they are.
**Measure.** The held-out ELBO with and without the learned prior, and the gap between the
aggregate posterior and the prior (an MMD between their samples; the `mmd` regularizer already
holds the kernel). Whether the samples are plausible for SAXS is the owner's domain check.
**Done when.** A post-hoc prior fits on a trained model, `GlobalVae.sample` draws from it, the
held-out ELBO and the gap are reported, and the model's own tests are unchanged.
**Depends on.** P4-0, P5-5, P5-6, and P6-2 and P6-5 for the post-hoc mode.

### P4-3. Hierarchical or sequential latents (XL, two sub-items)

**status**: todo
**Why.** Spec §7, and the "one latent per sample" findings (I-10). This changes the spec: the
routing graph is bipartite today (B3), and spec §2.2 and §12 say that latent spaces are independent
by construction. A latent that depends on another is a change to those sentences, so it needs the
owner's approval before code.

**P4-3a. Hierarchical.** A top-down chain of latent spaces with conditional priors
`p(z_l | z_(l+1))`.
- **Literature.** NVAE ([2007.03898](https://arxiv.org/abs/2007.03898)) and VDVAE
  ([2011.10650](https://arxiv.org/abs/2011.10650)) are the deep references, and the diffusion
  VampPrior makes the prior of such a model practical.
- **Interface.** Latent -> latent edges in `RoutingGraph` (a parent list per latent space),
  checked acyclic by `validateRoutingGraph`. A regularizer that sees the parent's sample (B2).
  `free_bits_kl` already works per latent space, which helps against collapse at the upper levels.
- **First step.** Two levels with Gaussian conditionals.

**P4-3b. Sequential.** A latent per step: a `(batch, steps, dim)` latent space with a transition
prior `p(z_t | z_(t-1))`. The review of dynamical VAEs
([2008.12595](https://arxiv.org/abs/2008.12595)) organizes the model families.
- **Interface.** A latent space with a length, assemblers and regularizers that reduce over steps,
  and a decoder that takes a sequence (B4).
- **First step.** A per-step latent with a first-order Gaussian transition prior and no recurrence
  in the encoder.

**Order.** Take first the one that a named dataset needs: sequential for time series and audio,
hierarchical for images. The ADR for P4-0 may split this item.
**Risks.** Collapse at the upper levels of a hierarchy, training cost, and evaluation: per-level
diagnostics are needed.
**Done when,** for each sub-item: the first step above trains through `scripts/train.py` on a
synthetic benchmark with a falling loss and active units at every level, every earlier model gives
the same results as before, and the spec text is amended with the owner's approval.
**Depends on.** P4-0, P5-5, P5-6, and P4-2 for the conditional priors.

### P4-4. Serving (L, steps 2 and 3 need D-7)

**status**: todo
**Why.** Spec §1 and §11: serving is in scope, sequenced after the model is functionally complete.
D-7 (API, batch inference or edge) is unanswered, so the work is laid out as a chain in which the
answer unlocks the later steps.

| Step | Content | Needs |
|---|---|---|
| 0, inference bundle | A bundle (the weights in the safe format, the config, the versions, and the transforms with their inverses) and a loader that rebuilds a model for inference. Built on `encode` and `decode`. | P5-5 and a go-ahead |
| 1, batch inference | `scripts/encode.py` and `scripts/decode.py`: latents from a dataset through the caller's `loader_factory` (spec §6.2), and reconstructions or samples from latents. | step 0 |
| 2, export | ONNX through `torch.onnx.export`, which defaults to `dynamo=True` (built on `torch.export`) and takes `dynamic_shapes` for the batch axis. One graph per encoder, with `mu` and `logvar` as outputs, and one per decoder, on the mean path. A numeric parity test against PyTorch for every registered encoder and decoder (24), with a stated tolerance, and a list of the modules that cannot be exported and why (data-dependent control flow in the `NaN`-aware encoders and the set operations are the suspects). An optional `export` extra: `onnx`, `onnxscript` and `onnxruntime` are not installed in the development environment today, and their import is deferred as for TensorBoard and MLflow. | D-7 answered "edge" or "interchange" |
| 3, API | A thin HTTP service kept outside `src/global_vae`, like the examples, that loads a bundle, batches requests and exposes a health check, with a CPU latency and throughput measurement. | D-7 answered "API" |

**TorchScript is not a target.** `torch.jit.script` emits a deprecation `FutureWarning` in torch
2.15 and points to `torch.compile` and `torch.export`.
**Not in scope.** Quantization, pruning, TensorRT and mobile runtimes.
**Done when,** per step: step 0, a model rebuilt from a bundle reproduces the outputs of the model
it came from; step 1, the two scripts round-trip a dataset; step 2, the parity test passes for the
modules that can be exported and the rest are listed; step 3, the service answers a batch and is
measured.
**Depends on.** P5-5, and D-7 from step 2.

---

## 8. Decisions needed

"Proposed" is what the work assumes if the owner simply says "go with the defaults".

| ID | Question | Options | Proposed | Blocks |
|---|---|---|---|---|
| D-4 | **Spec §11, first bullet** (regularizer strategies beyond the default). `free_bits_kl` and `mmd` shipped ([ADR 0003](adr/0003-pluggable-latent-regularization.md)). | keep it open; mark it resolved | Mark it resolved in the spec, and track learned priors as P4-2 | P5-2 (text only) |
| D-5 | **Pairing mechanism** (spec §11, second bullet). P3-2 took the default and the owner has not confirmed it. | per dataset | Confirm: pair by sample ID in the caller's `loader_factory`, and keep unpaired samples out of the paired stages | P5-8 (paired part), real-data acceptance of Phase 6 |
| D-6 | **Beta hyperparameters** (spec §11, third bullet): warm-up length, cyclical period, per-space values, the loss weights. Empirical. | none yet | Decide after the first real run (P5-8). P7-1 narrows the question to the choice of beta | none |
| D-7 | **Serving target** (spec §11, fourth bullet): API, batch inference or edge. | unknown | Nothing beyond steps 0 and 1 of P4-4 until answered | P4-4 steps 2 and 3 |
| D-13 | **The latent distance of the pair stage.** The text of PairVAE read for this roadmap does not name its distance, and its supplementary information was not accessible. | one of the six of P6-3, or several | Ship all six. Train with `wasserstein2_diag` (the squared form) by default, and `mean_mse` as the ablation. Settle it after reading the paper's supplementary information and after the P6-6 measurements | P6-3, P6-4 |
| D-14 | **What is frozen in stage 3.** The paper trains both VAEs together. | nothing; the decoders; one side; both VAEs, with an alignment head only | Nothing frozen. Freezing is opt-in (P6-5) | P6-5 |
| D-15 | **Stage orchestration.** | A: three ordinary runs chained by checkpoints. B: one driver that runs the three stages from one config | A, since each stage is then a normal, resumable, reproducible run. Revisit B after P6-7 | P6-7 |
| D-16 | **Latent widths in a pair.** | require equal widths; or an adapter head used by the alignment only | Require equal widths. The adapter is a P6-8 extension | P6-4 |
| D-17 | **Go-ahead and order for Phases 7 and 4.** | per item | Phase 7: P7-1, P7-2 and P7-3 first, since they answer I-4 and need only Phase 5. P7-4 and P7-5 on a named need. Phase 4: P4-0, then the order of section 7 | Phases 7 and 4 |
| D-18 | **Docs toolchain after the end of life of Material for MkDocs (2027-05-05).** | stay on the pinned MkDocs 1.6.x and Material 9.x; move to ProperDocs, which the plugin notice names as a drop-in continuation of MkDocs 1.x (the Material team says that notice is not its own); move to Zensical, which the Material notice names as its successor | Stay pinned for now, with an upper bound on `mkdocs-material`. Try the build with ProperDocs or Zensical on a branch before the end of life, and decide then | P5-9 (the trial only) |

Closed since the previous roadmap:

| ID | Question | Answer |
|---|---|---|
| D-1 | Routing-graph schema in YAML | [ADR 0025](adr/0025-config-schema-for-several-latent-spaces.md) |
| D-2 | What a shared encoder or decoder means | option A, [ADR 0027](adr/0027-shared-encoders-and-decoders.md), now in spec §10 and §12 |
| D-3 | Which assembler first | both exist, [ADR 0026](adr/0026-assembler-constructor-contract-and-learned-assemblers.md), recorded in spec §11 |
| D-8 | Tracking backend | MLflow as an optional extra, [ADR 0035](adr/0035-mlflow-experiment-logger.md) |
| D-9 | CamelCase module names | renamed to snake_case, shims removed in 2.0.0, [ADR 0023](adr/0023-snake-case-module-names.md) |
| D-10 | Docstring rules in tests | D101 to D104 ignored for `tests/**` (`pyproject.toml`) |
| D-11 | `print` in scripts and examples | spec §10 now covers library code only |
| D-12 | Copyright holder | `LICENSE`: Esteban THEVENON, 2026 |

---

## 9. Not planned

- `datamodule.py`, and any code in `src/` that loads, pairs or splits datasets (spec §6.2, a
  permanent boundary, not a gap).
- Modality-specific or dataset-specific transforms, SAXS included (spec §6.2 and §12).
- U-Net style encoder to decoder skip connections ([ADR 0014](adr/0014-residual-1d-encoder-decoder.md)
  rules them out: they bypass the latent).
- A Lightning or Fabric migration before the two triggers of
  [ADR 0036](adr/0036-lightning-migration-triggers.md) hold.
- Any component wired into `GlobalVae` by name: every new component goes through its registry.
- Downloading or bundling pretrained models. A reference encoder (P7-5) is supplied by the caller,
  and the framework makes no network access.
- A diffusion or autoregressive image-generation stack inside the decoder. The framework hosts
  priors (P4-2), not a generative image model.
- TorchScript export: it is deprecated upstream (P4-4).
- A pair-stage alignment without pairs. With no pairs, the terms of Phase 6 have nothing to compare;
  the post-hoc priors (P4-2) and the extensions of P6-8 are the routes.

## 10. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The alignment has a trivial solution: both latent spaces collapse and the distance goes to zero | A "successful" stage 3 that carries no information | Keep the reconstruction and KL terms in the objective, ramp `lambda`, flag a space with fewer active units than its solo stage, and report retrieval against its chance level (P6-4, P6-6) |
| Few pairs (72 in the reference paper) overfit stage 3 | An alignment that does not generalize | Held-out pairs, repeated splits and seeds, optional freezing of the decoders, early stopping on the held-out distance and retrieval (P6-5, P6-6) |
| BatchNorm statistics drift under the small batches that pairs give | Silent damage to the solo solutions | Eval mode for frozen modules and a bit-identity test (P6-5); self-reconstruction measured before and after (P6-4) |
| The paper's equations were not read | The distance and weights may differ from the paper's | D-13 stays open, the six distances are compared (P6-6), and nothing here claims to reproduce the paper's numbers |
| The seams touch `GlobalVae.forward`, the most reused path | Regressions in every model | Golden-output tests on every matrix row before the refactor, and `forward` keeps its outputs (P5-5) |
| Three new registries (reconstruction losses, loss terms, latent distances) | More surface to maintain | One registry per decision, each in the contract test (13 registries) |
| Literature claims age, or do not carry over to signals and images | A component with no benefit | The evidence level in 6.1, a baseline to beat in every item, and a "close as no gain" outcome (P7-3) |
| Phase 4 changes the posterior interface | Users' code breaks and a major release is forced | A tuple-compatible `DiagonalGaussian`, golden tests first, 3.0.0 only if unavoidable (P4-0) |
| Everything is validated on synthetic data | Choices tuned on toy data | P5-8, and the Phase 6 acceptance on real pairs |
| Tool and dependency drift (torch, numpy, ruff, mypy, the docs toolchain; export APIs move between torch releases) | A red CI with no code change; export code that breaks | A weekly scheduled run (P5-9), the pins of the `dev` and `docs` extras, a decision on the docs toolchain (D-18), and a tested torch range for the `export` extra |
| The test suite keeps growing (4,987 tests, 10 to 18 minutes in the measuring session) | The job approaches its `timeout-minutes` | Measure it (P5-9), then shard or split it |

## 11. Definition of done

Every item is done when:

1. The code follows spec §10: English, type hints, Google docstrings with `Args`, `Returns` and
   `Raises`, one class per file, camelCase callables, no em dashes and no unicode arrows.
2. A new pluggable component follows the registry pattern, is imported in its subpackage
   `__init__.py` and joins `test_registry_contract.py`. No core file is edited to add it. The two
   edits that P5-5 and P5-6 make to the core are the exceptions, and each has its ADR.
3. Unit tests cover the component, and an integration test covers any change to routing or
   training.
4. A new capability is opt-in, and a run that does not use it is bit-identical to the previous
   release.
5. An ADR records any architectural change. A changed decision is a new ADR, never an edit of an old
   one.
6. A component taken from the literature cites its source in its docstring and its ADR, states the
   evidence level, and has a benchmark against the baseline named in its item, or is closed as "no
   gain" with the note kept.
7. `docs/changelog/CHANGELOG.md` has a line under `[Unreleased]`, the docs (how-to or docstrings)
   are updated, and `mkdocs build --strict` passes.
8. CI is green: ruff, mypy, pytest on Python 3.11 and 3.13, and the docs build.

---

## Appendix A. How the baseline was measured

- **Date and commit.** 2026-11-02, commit `5685c3a` (tag `v2.0.0`), in a detached worktree with no
  change.
- **Environment.** Linux container, 2 vCPUs, 8 GB of memory, no GPU, Python 3.11.18, torch 2.15.0 (the
  default PyPI wheel, run on CPU), numpy 2.4.6, pytest 9.2.0, pytest-cov 7.1.0, coverage 7.17.0,
  ruff 0.16.10, mypy 2.4.1, mkdocs 1.6.1. Installed with `pip install -e ".[dev,docs]"`. The CPU
  wheel index that `ci.yaml` uses is not reachable from the container, so the default wheel was
  used.
- **Commands.**
  - `pytest -rs -q --durations=15 --cov=global_vae --cov-report=term-missing:skip-covered
    --cov-fail-under=95 -p no:cacheprovider`
  - `ruff check .` and `ruff format --check .`
  - `mypy`
  - `mkdocs build --strict`
- **Counted from the tree** (not measured by a run): lines, files, ADRs, how-to pages, YAML configs,
  commits and tags in section 2.1, and the registry contents in section 2.2 (read from the
  `listRegistered*` functions).
- **Caveats.**
  - Only Python 3.11 was run. The 3.13 job of `ci.yaml` was not reproduced, and the GitHub Actions
    results could not be read from the container.
  - Coverage is in process only: tests that run scripts and examples as subprocesses add nothing,
    and branch coverage is off.
  - Durations are those of a 2-vCPU container. The suite took 17 min 44 s on the first run after
    the container started, then 11 min 51 s and 10 min 5 s on the two runs of the final tree.
    GitHub's runners differ.
- **Final tree.** On 2026-11-03 the same commands were run again on the tree that contains this
  file, the only file that differs from `v2.0.0`: 4,987 passed, 3 xfailed, 98.51% coverage, ruff and
  mypy clean, and `mkdocs build --strict` exits 0.
- **A finding while measuring** (not a result of the suite): the shell of the container cannot reach
  `download.pytorch.org` or arXiv, which is why section 6.1 was checked through a search and fetch
  tool and not through scripts.

## Appendix B. How the literature was checked

- **Date and tools.** 2026-11-02. Web search and fetch of arXiv abstract and PDF pages, conference
  and publisher pages (CVPR, NeurIPS, ICML, AAAI and AISTATS listings) and Hugging Face paper
  pages. The torch facts of P4-4 were checked locally on torch 2.15.0: the signature and docstring
  of `torch.onnx.export`, the `FutureWarning` of `torch.jit.script`, and whether `onnx`,
  `onnxscript` and `onnxruntime` are installed (they are not).
- **What was read.** For each paper cited in sections 5 to 7, the abstract page or the PDF text as
  the fetch tool returned it, for the title, the authors, the venue when the page shows one, and the
  core contribution. Claims in this file stay within that text. Where a page named no venue, the
  table says "arXiv", even if the paper was published later at a venue that this check did not see.
- **What was not read.** The supplementary information of PairVAE, so its distance, the form of its
  cross-reconstruction losses and its weights are unknown (D-13). The experiment sections of most
  papers: no paper's result is quoted here as an expected result for this framework.
- **Limits.** A targeted survey (the 27 papers cited here), not a systematic review. It can miss
  relevant work, and most 2026 papers are preprints. The venues of FSQ, BSQ, the rotation trick,
  SimVQ, the VampPrior, VDVAE, the flow-prior alignment and Diff-MVAE were not verified, and the
  TMLR 2024 listing of SBM-VAE comes from a search result. One 2026 paper (the spherical VAE with
  feasible regions) is by independent authors and says in its latest version that the first
  version was wrong, so it is listed and not scheduled.
