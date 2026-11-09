# 0050: CI hardening, the docs toolchain plan and the causes behind I-14

**Status:** accepted (roadmap item P5-9; findings I-14 and I-16, decision D-18, and the "CI" row of the housekeeping
table; it adds a weekly run, a branch coverage report and a test of `utils/seed.py`, raises the coverage floor, bounds
`mkdocs-material`, records a trial of two replacements for MkDocs, and records the causes that ADR 0041 and ADR 0042
left open, which are left as written)
**Date:** 2026-11-09

## Context

Roadmap item P5-9 asks for five things about the checks around the code and two investigations. The checks:

- Read the Actions results of `v2.0.0`, the Python 3.13 job included.
- Catch a new release of ruff, mypy, numpy or torch between two pushes, which a workflow that runs on a push alone
  cannot do.
- Bring the coverage floor (95, with a comment that quotes the 96.25% of v1.2.0) up to the measured value (98.67%).
- Compare the duration of the test job with its 30 minutes.
- Report branch coverage, and cover the `deterministic=True` branch of `utils/seed.py` (78%).
- Bound the docs toolchain, whose maker says that Material for MkDocs reaches end of life on 2027-05-05 (finding I-16,
  decision D-18).

The investigations are the two results that ADR 0041 and ADR 0042 recorded and did not explain (finding I-14): the 1D
transformer pair on one-hot text (62 to 65% of the tokens, against 87 to 99.6% for the text pair), and the 1D
transformer decoder on series with gaps (0.53 of the variance, against 0.80 for the CNN decoder).

**What this session could not do.** It has anonymous read access to the git history and no access to the Actions
results: the Actions API answers 403 and the pages of github.com are not reachable from it. So the `v2.0.0` run was not
read, the duration of a real runner was not measured, and a scheduled run cannot be seen before the workflow is on the
default branch. The numbers below that stand in for them come from the full suite on Python 3.13.18 with 2 vCPUs (the
machine of roadmap appendix A), and the three checks left are listed under Consequences.

## Decision

**1. A weekly run.** `ci.yaml` gains `schedule: - cron: "17 5 * * 1"` (Mondays, 05:17 UTC, off the hour on purpose,
since runs queued at the top of an hour are the ones GitHub delays). It runs the same three blocking jobs as a push,
and a test (`tests/unit/test_ci_workflow.py`) fails if the trigger is removed, runs more often than weekly, or if one
of the three jobs becomes conditional. Nothing in `pyproject.toml` bounds ruff, mypy, numpy or torch below the next
major version, on purpose: a bound would hide the release the run is there to catch. The two tests that build the real
docs site run in the `tests` job, so the same weekly run also catches a release of the docs toolchain that breaks the
build. GitHub runs a schedule only on the default branch and switches the schedule of a public repository off after 60
days without repository activity, which the comment above the trigger says.

**2. The floor is 97, and its comment is guarded.** The value measured here on Python 3.13.18 is 98.67% of statements (6,987 statements, 93 missed, 5,832 passed and 2 skipped; 98.61% before the tests of decision 5), so the floor sits one point below it, rounded down: 97 (the
roadmap said 97 for the 98.51% of its own session). The comment above `COVERAGE_FLOOR` quotes the value and the two
older ones, and `test_ci_workflow.py` fails when the quoted value is not one to two points above the floor, so the
floor and its comment move together. The README and `docs/getting-started.md` repeat the number and a test compares
them with the workflow. The 95 of the old floor had stayed in both since v1.2.0.

**3. The timeout stays at 30 minutes, and every log gains the durations.** The suite took 7:01 to 10:49 in five full runs here (10:49 for the last, with the new tests, 8:01 for the best run with no failure, 9:55 with branch tracing), on 2 vCPUs, install excluded, against a limit
of 30: a margin of three, with a runner that may be slower. No shard is added, and `pytest-xdist` is not a dependency.
The `pytest` step gains `--durations=15`, so the 15 slowest tests are in every log. On the machine above they run 9.5 to
32 s (a UMAP projection at 32 s, then the graph node-order test at 17 s, the series test that trains to a stated
recovery at 17 s, and tests of the same kind at 10 to 13 s, plus the two builds of the docs site at 11 s each and a few
runs of `scripts/train.py`). The threshold to shard is written next to the step: 15 minutes for the `pytest` step of a
real log.

**4. Branch coverage is a report, in a job of its own.** A `branch-coverage` job runs on the weekly schedule and on a
manual dispatch only, on Python 3.11, with `--cov-branch`, and writes `coverage report --format=markdown` for `latent/`,
`models/` and `training/` to the summary of the run. It has no floor and it is not a required check. The gated run does
not trace branches: with branches on, coverage's headline number combines statements and branches (98.17% here, against 98.61% of statements, on the tree before the new tests), so enabling them in the `tests` job would change the number the floor is compared with, and it slows the suite. A test checks that the job is conditional on the schedule and the dispatch, has no `--cov-fail-under`,
and that the `tests` job has no `--cov-branch`.

The measurement of the job, run here on the full suite before the new tests (Python 3.13.18): 98.61% of statements and 96.75% of branches for
the package (2,113 of 2,184), and 99% for the three packages (1,348 statements with 8 missed, 460 branches with 11
partially covered). The files with a gap are `models/global_vae.py` (lines 557->577 and 608), `training/checkpoint.py`
(132->139, 140, 153->155, 155->157, 157->163, 164 and 449), `training/callbacks/_plateau.py` (111),
`training/callbacks/_lr_schedule.py` (51), `training/beta_schedules/base.py` (43), `training/loggers/base.py` (87) and
`training/loggers/csv_logger.py` (74->exit). They are listed here as a starting point and are not a work item.

**5. `utils/seed.py` is tested.** `tests/unit/test_seed.py` (11 tests) covers what the 78% left out: the flags that
`deterministic=True` sets (PyTorch's deterministic algorithms, the two cuDNN flags), `warn_only`, the
`CUBLAS_WORKSPACE_CONFIG` variable being set when it is absent and left alone when the caller set it, a later default
call resetting every flag, and the same seed giving the same draws from `random`, NumPy and PyTorch in both modes and
the same weights for a model built twice. The function changes process-wide state, so an autouse fixture saves and
restores all of it (the flags, the variable and the three generators), and no other test sees what these do. `utils/seed.py` is now at 96%: the line left is `torch.cuda.manual_seed_all`, which runs
only where CUDA is available.

**6. The docs toolchain: bounded now, replacement trialled, decision later (D-18).** `docs` in `pyproject.toml` takes
`mkdocs-material>=9.5,<10`. The newest release on PyPI is 9.8.1 and there is no 10, so the bound costs nothing today and
stops an unreviewed major version of the theme. `mkdocs` stays at `>=1.6,<=1.6.1`. Two replacements were built against
the repository, from a clean checkout, without a change to `mkdocs.yml`:

| | `mkdocs 1.6.1` (today) | `properdocs 1.6.7` | `zensical 0.1.0` |
|---|---|---|---|
| Command | `mkdocs build --strict` | `properdocs build --strict` | `zensical build --strict` |
| Reads `mkdocs.yml` | yes | yes | yes |
| Exit code | 0 | 0 | 1 |
| Time | 9.7 s | 9.2 s | 1.7 s |
| Pages | 246 | 246 | 80 |
| Needs | the `docs` extra | the `docs` extra (it is already installed with it) | `mkdocstrings` installed by hand |

- **ProperDocs is a drop-in.** The 294 files of its site are the 294 files of the MkDocs site, and the 246 HTML pages,
  the 36 JavaScript files, the 3 style sheets, the search index and the sitemap are byte for byte the same, except for the
  `generator` meta tag (`mkdocs-ProperDocs 1.6.7` in place of `mkdocs-1.6.1`). It redirects `mkdocs.*` imports to its
  own modules, which is why the plugins load unchanged. Three plugins of the `docs` extra (`mkdocs-gen-files` 0.6.1,
  `mkdocs-literate-nav` 0.6.3, `mkdocs-section-index` 0.3.12) already depend on it, so it is installed today, next to
  MkDocs. The same plugins print a notice on `mkdocs build` that MkDocs may break the plugins soon and that ProperDocs
  is the continuation (silenced by `DISABLE_MKDOCS_2_WARNING=true`), and Material 9.7 prints a banner of its own. Neither
  is printed by `properdocs build`.
- **Zensical is not a replacement for this site yet.** It builds the 80 pages that need no generated content in 1.7 s
  and exits 1 under `--strict`: `scripts/gen_ref_pages.py` runs under `mkdocs-gen-files`, which it does not run, so
  the whole `reference/` tree (166 pages) is missing and six links to it are broken. The ADR and changelog entries
  are in the sidebar as before (checked on one page). It needs `mkdocstrings` installed apart.

**The plan, for the owner.** Stay on MkDocs 1.6.1 with Material below 10 until the end of life. Switching to ProperDocs
is a change of a few lines and no change of content: the command of `docs.yaml`, `properdocs` in the `docs` extra, and
the `-m mkdocs` of the two tests that build the site (`test_docs_adr_navigation.py`, `test_docs_changelog_navigation.py`)
and their `_DOCS_TOOLCHAIN` lists. It is not made here, because D-18 reserves the decision, and the trial says it can
wait: the weekly run of decision 1 turns red on the first release that breaks the pinned set. Zensical is to be tried
again when it runs `gen-files` scripts or ships its own reference generator, and the result recorded as a new ADR.

**7. The causes behind I-14.** Both results were isolated, one factor at a time, on the benchmarks of the two ADRs. The
text side is explained (the scale of the embedding, then the padding mask), and the series side is the budget of
training (the decoder converges more slowly, and part of that is the scale of the latent signal), with a part left
unexplained. The protocol, the tables and what they say follow.

## Measurements: the 1D transformer pair on one-hot text

The benchmark of ADR 0041: the grammar of `tests/integration/_text_corpus.py`, 4,000 training and 1,000 validation texts
(8 positions, a vocabulary of 32 with the padding token), the shipped size (`d_model` 64, 2 layers, 4 heads, a latent of
16), Adam at 1e-3, batches of 64, 60 epochs (3,780 steps), a 1,000-step warm-up of beta, the exact cross-entropy on
raw scores with weight 2.0. Token accuracy counts every position from the posterior mean. Three seeds (0, 1, 2), means.
The script and the raw lines are not kept in the repository: this table is the record.

The 1D pair is `1d_transformer_encoder_v1` (`in_channels` 32, `patch_size` 1) and `1d_transformer_decoder_v1`
(`out_channels` 32, `patch_size` 1), with a one-hot batch laid out channels first. Its differences with the text pair
are three: the **embedding** (a convolution of kernel 1 with its default initialization, against a table initialized at
`N(0, 1)`), the **padding mask** (none, against a key mask and a masked mean), and the **pooling** of the default
(`cls` by default, against `mean`; the rows below use `mean` unless they say `cls`). The decoders are the same function, as the first rows show.

| Pair | Embedding (std) | Mask | Pooling | Token accuracy | Exact texts |
|---|---|---|---|---|---|
| text (shipped) | 1.0 | on | mean | **99.0%** (97.8, 99.3, 100.0) | 92.5% |
| text | 1.0 | off | mean | 97.5% | 80.4% |
| text | 0.1 | on | mean | 75.5% | 5.8% |
| text | 0.1 | off | mean | 64.4% | 0.3% |
| 1D | 0.10 | off | mean | **62.6%** (59.1, 64.0, 64.6) | 0.5% |
| 1D, its default pooling | 0.10 | off | cls | 61.1% | 0.4% |
| 1D, table-scale embedding, bias zeroed | 1.0 | off | mean | **97.8%** (99.0, 99.5, 94.9) | 82.5% |
| 1D, table-scale embedding, bias kept | 1.0 | off | mean | 97.0% | 76.6% |
| text encoder with the 1D decoder (seed 0) | 1.0 | on | mean | 97.8% | 82.4% |
| 1D encoder with the text decoder (seed 0) | 0.10 | off | mean | 59.1% | 0.4% |

The dose, the standard deviation of the embedding alone (everything else as shipped):

| Std | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|---|
| text pair | 56.5% | 75.5% | 98.2% | 99.0% | 96.1% |
| 1D pair | | 62.6% (shipped) | 94.1% | 97.8% | 95.5% |

Four times the steps (240 epochs) take the 1D pair from 59.1% and 64.0% to 62.4% and 72.5% on seeds 0 and 1: it
learns, slowly, and it does not catch up.

What it says:

- **The cause is the scale of the embedding.** A convolution of kernel 1 over a one-hot row picks one column of its
  weight, and the default initialization (bound `1 / sqrt(32)`) makes that column `0.10` in root mean square, where
  the table is `1.0`. The part of a token vector that tells tokens apart is `0.10` against sinusoidal positions of
  `0.71` in the 1D encoder (a ratio of 0.14), and `0.99` against `0.71` in the text encoder (a ratio of 1.4). Scaling
  the 1D embedding to the table's gives 97.8%, from 62.6%, and scaling the text embedding down to the convolution's
  gives 75.5%, from 99.0%. The accuracy rises with the scale up to 1.0 on both pairs and is nearly flat from 0.3 to 1.0 on the text pair.
  A scale of 3.0 is worse than 1.0 on both (96.1% and 95.5%), so the table's scale is a good value and not a limit.
- **The padding mask is a second, small factor, and it needs the first to show.** At the table's scale it is worth 1.5
  points of accuracy and 12 of exact texts (99.0% against 97.5%; ADR 0041 had three points and 26 of exact texts on two seeds); at the
  convolution's scale it is worth 11 points (75.5% against 64.4%). The 1D pair given a table-scale embedding and no mask
  (97.8%) is where the text pair without its mask is (97.5%): the two factors account for the whole gap.
- **The decoder and the pooling are not factors.** A text encoder with the 1D decoder reproduces the text pair to the
  digit on seed 0 (97.8% and 82.4% of texts), and a 1D encoder with the text decoder reproduces the 1D pair (59.1% and
  0.4%): the two decoders compute the same function, so every difference is in the encoder. `cls` pooling is 61.1%
  against 62.6% for `mean` in the 1D pair, inside the spread of seeds.
- **Why a small embedding hurts, in one line.** The layers are pre-norm, so the first thing a token meets is a layer
  norm of its vector plus its position, and when the token part is a seventh of the position part the layer reads the
  position and the identity of the token has to be learned out of a small residue. Adam moves a weight by about its
  learning rate per step (1e-3), so growing a column from 0.1 to 1.0 takes on the order of a thousand steps, a
  quarter of the run, which is what four times the steps recovers in part. This is the reading the numbers favor, and the dose
  response is its test; no run measured the gradient directly.
- **What it means for the classes.** `PatchEmbed1d` is right for real-valued series, whose values have a scale of
  their own. It is the wrong stem for a one-hot channel, where it replaces a table by a convolution of ten times the
  smaller scale. Use the text pair for tokens, which the shipped experiment does.

## Measurements: the 1D transformer decoder on series with gaps

The benchmark of ADR 0042: the series of `tests/integration/_time_series_corpus.py` (3 channels of 96 steps, 17% of the
values unobserved), 3,000 training and 600 held-out series, the shipped `time_series_encoder_v1` (155,296 parameters, a
latent of 16), AdamW at 2e-3, batches of 64, 60 epochs (2,820 steps), a constant beta of 1, the masked loss with weight
50. Every number is the share of the variance of the complete series recovered from the posterior mean, on the
observed values, the gaps and the tail, as there. Seed `s` changes the initialization and the data (training series
seed `1 + 2 s`, held-out `2 + 2 s`). Three seeds (0, 1, 2) unless a row says two. The decoder is
`1d_transformer_decoder_v1` (patch 8, `d_model` 64, three layers, 154,288 parameters) against `1d_cnn_decoder_v1` of the
shipped model (a seed of 12, kernel 5, padding 2, 78,083 parameters). The roadmap's "kernel padding" has no meaning for a
decoder that has no kernel, so it was read as the kernel and the same-size padding of the CNN decoder it is compared
with: 1 with 0, 3 with 1, 5 with 2 (shipped), 9 with 4. A kernel larger than 1 with no padding changes the length (kernel 5 gives 68 samples, not 96), and the class refuses it.

**The reference, the seed and the kernel** (60 epochs):

| Decoder | Observed | Gaps | Tail |
|---|---|---|---|
| transformer, default | **0.52** (0.53, 0.52, 0.51) | 0.39 | 0.21 |
| CNN, kernel 5, padding 2 (shipped) | **0.78** (0.78, 0.78, 0.78) | 0.69 | 0.46 |
| CNN, kernel 1, padding 0 | 0.65 | 0.55 | 0.34 |
| CNN, kernel 3, padding 1 | 0.77 | 0.67 | 0.45 |
| CNN, kernel 9, padding 4 | 0.77 | 0.68 | 0.40 |

**One thing changed in the transformer decoder** (60 epochs):

| Change | Observed | Gaps | Tail |
|---|---|---|---|
| weight 500 (instead of 50) | 0.57 | 0.44 | 0.17 |
| learned positions | 0.46 | 0.37 | 0.28 |
| positions multiplied by 0.1 | 0.34 | 0.30 | 0.12 |
| latent projection multiplied by 3 | 0.66 (0.60, 0.73, 0.64) | 0.55 | 0.23 |
| latent projection multiplied by 10 | 0.66 (0.61, 0.69, 0.69) | 0.56 | 0.26 |
| patch 2 | 0.66 (0.71, 0.55, 0.72) | 0.55 | 0.37 |
| a learned convolution of kernel 5 after the decoder (two seeds) | 0.50 | 0.39 | 0.23 |
| latent multiplied by 3 and that convolution (two seeds) | 0.70 | 0.57 | 0.23 |

**The number of epochs**, two seeds:

| Epochs (steps) | Transformer, observed (gaps, tail) | CNN, observed (gaps, tail) |
|---|---|---|
| 60 (2,820) | 0.52 (0.39, 0.21) | 0.78 (0.69, 0.46) |
| 120 (5,640) | 0.80 (0.72, 0.28) | not run |
| 240 (11,280) | **0.86** (0.79, 0.33) | **0.84** (0.78, 0.47) |

What it says:

- **The gap of ADR 0042 is mostly a gap in convergence at its budget, not a limit of the decoder.** At 60 epochs the
  transformer decoder is at 0.52 and the CNN decoder at 0.78. At 120 epochs the transformer decoder is at 0.80, and at 240
  it is at 0.86 on observed values and 0.79 on gaps, where the CNN decoder at the same budget is at 0.84 and 0.78. The
  tail stays behind (0.33 against 0.47), and the tail is not a training target. The decoder needs about twice to four times the steps. A run takes the same time with either decoder here (90 s and 88 s for 60 epochs on one thread), so the extra steps are not paid for by a slower step.
- **Seed is not a factor.** Three seeds of the default decoder give 0.51 to 0.53, and three of the CNN decoder give 0.78
  each. The up to 0.07 spread of ADR 0042 belongs to the encoder comparison it was written for.
- **The kernel explains part of the CNN's lead, but not the transformer's deficit.** A kernel of 1 (no mixing between
  neighbouring samples; the output is then constant over blocks of 8) costs the CNN 0.13 (0.78 to 0.65), and from 3 up
  it is flat (0.77). A kernel of 1 still scores above the transformer decoder (0.65 against 0.52), which writes 8
  independent values per token and so is not short of resolution. Adding a learned convolution of kernel 5 after the
  transformer decoder changes nothing alone (0.50) and little with the next change (0.70 against 0.66, inside the
  spread): the missing local smoothing is not what holds it back.
- **Part of the slowness is the scale at which the latent vector enters.** Multiplying the weights of the decoder's
  latent projection by 3 or by 10 gives the same gain (0.52 to 0.66 on observed values, 0.39 to 0.55 on gaps), a gain that does not grow with the factor. It looks like the effect of the embedding of the text encoder (the default initialization makes a small signal that Adam must grow), which is a reading and not a measurement of the scales, and it accounts for about half of the gap at 60 epochs. Making the positions ten times smaller instead is much worse (0.34), so
  the two scales are not interchangeable and the point is the strength of the latent signal, not its ratio to the
  positions. The number of tokens helps as much (patch 2: 0.66, with a wide spread between seeds), and a larger weight
  helps a little (0.57), while learned positions hurt (0.46).
- **What is left unexplained.** The latent scale and the number of tokens each gain about 0.14 at 60 epochs, but no variant reaches the CNN's 0.78 in 60 epochs, and two or three seeds cannot rank changes of a few hundredths. What the budget table settles is that the decoder is not worse once it has converged. Why it needs more steps was narrowed (the latent scale is one lever) and not isolated to one cause.
- **What it means for the shipped model.** Nothing changes: `1d_cnn_decoder_v1` stays the decoder of the experiment, since
  it reaches a good recovery in the 60 epochs of the benchmark. The ADR 0042 statement that the CNN decoder "recovers
  0.80 where the transformer decoder recovers 0.53" is true at that budget and is not a statement about the two
  architectures. A user who wants the transformer decoder should train it for two to four times as long.


## Alternatives considered

- **Switch on branch tracing in the `tests` job.** It would change the number the floor is compared with (98.17% against
  98.61%) and the run time, for a report that a weekly job gives as well.
- **Run the branch report on every push.** It costs a second full suite per push (about ten minutes on 2 vCPUs) for a
  table that changes slowly.
- **Gate on branch coverage.** The item says a report, and a gate on a number of 11 partial branches in 460 would turn
  one new `if` into a red run.
- **Shard the suite with `pytest-xdist`.** The margin is three times the limit on the measured machine, and a new
  dependency changes the order of tests that share process-wide state. The threshold to revisit is written next to the
  step.
- **Tighten `timeout-minutes` to twice the measured run.** A runner that is slower than 2 vCPUs would then fail on a
  good commit. The limit exists to stop a hung run, and 30 minutes does that.
- **Bound ruff, mypy, numpy and torch to their current minors.** It hides exactly what the weekly run is for, and the one
  place where a bound is needed (numpy 2.5 stubs with `python_version = "3.11"`) is already held by the interpreter of
  the `types` job.
- **Bound `mkdocs-material` to `<9.9`.** The 9.x line only gets fixes until its end of life, so a minor bound would only
  add a monthly chore.
- **Move the docs build to ProperDocs or Zensical now.** ProperDocs is ready and costs a few lines, and the decision is
  D-18's, which asked for a trial and a plan. Zensical cannot build the API reference of this site.
- **Change `PatchEmbed1d` (or add an `embedding_scale` argument) to fix the 1D pair on one-hot text.** The classes are
  right for real-valued series, and nothing in the roadmap asks for categorical channels in the 1D family. It is a
  finding, below.

## Consequences

- **The checks run every week.** A release of ruff, mypy, numpy, torch or the docs toolchain that breaks the build turns
  the Monday run red, with the tests of the weekly run being the ones of every push.
- **The floor is 97, with a test that keeps its comment honest.** Raise the number and the quoted value together.
- **Three checks need the Actions page, and are left to the owner.** First, the `v2.0.0` run, 3.13 job included: this
  session's own full run on 3.13.18 passed (5,832 tests, 2 skipped, 98.67%), which is evidence and not that log.
  Second, the first Monday run after the workflow reaches `main`, which cannot be seen from here. Third, the duration of
  the `pytest` step and the list of slowest tests of a real runner, against the 15-minute threshold of decision 3. The
  status of P5-9 stays `todo` until those three are read, since its "done when" asks for the 3.13 result and the
  schedule to be known to run.
- **Finding, not fixed: the 1D transformer classes on categorical channels.** A model that feeds one-hot channels to
  `1d_transformer_encoder_v1` trains at a tenth of the embedding scale of a table. An `embedding_std` argument, or a
  table-based stem, would remove it; it is not added since no shipped experiment feeds categories to that class.
- **Finding, not fixed: the coverage gaps of decision 4.** Seven files of the three packages have a missed statement or
  a partial branch; the list is above and each is small.
- **Specification proposal, for the owner.** Section 10 names CI. A sentence would match the workflow: tests, lint and
  types run on every push and once a week, the coverage floor is a percentage of statements that sits one point below the
  measured value, and a branch report runs on the weekly schedule.
