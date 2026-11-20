# 0059: More than two models: anchor mode and a multi-way Procrustes baseline

**Status:** accepted (roadmap item P6-8, extension "more than two models"; it adds `anchor` and `others` to the two pair
loss terms, one module and one script flag for the evaluation, and changes nothing for a config that lists `pairs`)
**Date:** 2026-11-16

## Context

`pairs` is already a list, so a model with three latent spaces can be aligned today by listing the pairs. Listing every
pair costs `K (K - 1) / 2` distances for the alignment and `K (K - 1)` decodings for the cross reconstruction, and the
config grows with it. The roadmap asks for two things: an anchor mode (every model aligned to one), and for a symmetric
treatment the multi-way scheme of Achara et al. ([arXiv 2602.06205](https://arxiv.org/abs/2602.06205)) as a baseline
first, then as a loss (P7-4).

That paper adapts Generalized Procrustes Analysis to build one shared orthogonal space for `K` models, so the number of
maps grows with the number of models and not with the number of pairs, and adds a correction of the directional mismatch
on top of it (GCPA) because strict isometric alignment retrieves worse than CCA. Its page, which is what was read, does
not give the correction's formulas.

## Decision

**1. `anchor` and `others` in `latent_alignment` and `cross_reconstruction`.** `pairs` stays and is unchanged. A term
takes either `pairs` or `anchor` with `others` (`resolvePairs` in `losses/terms/pairs.py`), and a mix is refused by
name: `pairs` with `anchor`, `anchor` without `others` or the reverse, an anchor that is also in `others`, a name twice,
an empty `others`.

- `latent_alignment` builds `[other, anchor]` for each of `others`. The anchor is the second space of every pair, so
  `stop_gradient: target` keeps it where it is and the others move to it, which is what "an anchor" means. With an
  asymmetric distance the direction is the one from each space to the anchor.
- `cross_reconstruction` takes `directions`: `both` (default: the anchor is decoded from every other's sample and every
  other from the anchor's), `to_anchor` (the anchor's decoders read the others' samples) or `from_anchor`.
- The two terms of a stage name the same anchor and the same others in their own entries; nothing couples them.

Anchor mode costs `K - 1` distances and `2 (K - 1)` decodings. It aligns each space to the anchor and **not** the
others to each other: two non-anchor spaces agree through the anchor, up to what each one's alignment left over. That is
a property to read, not to assume, which is the second decision.

**2. A multi-way baseline in the evaluation, before any loss.** `evaluation/multiway_alignment.py`
(`evaluateMultiwayAlignment`) takes `K >= 2` latent views (as `--latent-pair` does: `space` or `space@modality+modality`),
held-out batches and, optionally, training batches, and reads for **every ordered pair** `a -> b` the retrieval of the
true partner (`recall@k`, mean rank, `mrr`, with the chance level) for four rows:

| row | maps | what it is |
|---|---|---|
| identity | 0 | the spaces as training left them: what a learned alignment is judged by |
| pairwise | `K (K - 1) / 2` | one orthogonal Procrustes map per pair, fitted on the training means of that pair alone |
| common | `K` | generalized Procrustes: one orthogonal map per view into a common frame; `a -> b` is `a`'s map followed by the inverse of `b`'s |
| anchor | `K - 1` | one orthogonal map from each view to the anchor, and two other views go through it |

The common frame starts from each view's Procrustes map to the first view and refits every map against the mean of the
others' current images until the consensus stops moving. Each step cannot lower the norm of the sum of the images, and
the sum of the squared distances between every two images is a constant minus that norm, so the iteration converges;
for views that are rotations of one another it reaches zero residual. It is deterministic, closed form at each step, and
fitted on the training means only. `fitCommonFrame` and `fitAnchoredFrame` return a `CommonFrame` with the maps, the
`relative_residual` (what the fit leaves of the views' size, `0` to `1`), and, for the common frame, the rounds and
whether it converged; the report gives the residual on the training and on the held-out means.

The table is a bill. `pairwise` above `identity` says the trained spaces are not aligned and an orthogonal map would make
them so; `common` close to `pairwise` says `K` maps do what `K (K - 1) / 2` do; `anchor` close to `common` says a hub is
enough. A learned alignment that does not beat the identity row of a model trained without it by more than these maps
give is not needed (the rule of baseline 1, ADR 0056).

`scripts/evaluate.py --latent-set VIEW VIEW VIEW ...` (repeatable) with `--latent-anchor VIEW` and the flags that
`--latent-pair` shares (`--train-dataloader-factory`, `--retrieval-k`, `--retrieval-similarity`, `--procrustes-center`,
`--max-samples`, `--max-train-samples`) writes `multiway_alignment.json` and `.md` next to `results.json`. A set has at
least three views: for two, `--latent-pair` reads more. The sample ids of ADR 0058 are compared across every modality of
the views, and the report counts the rows.

**3. What is left out.** The loss of P7-4 (a term that ties `K` spaces to one learned reference) does not start: its
gate, a project with three modalities or more that need aligning, is not met, and the baseline above is what that item
asked for first. The correction on top of the common frame (GCPA) is not implemented: the formulas were not available,
and a re-derivation from the abstract would be a guess presented as the paper's method. The common frame is the first
stage of the scheme, which is also the baseline GCPA is judged against. Views of different widths are refused (D-16,
ADR 0060 reads them through an adapter).

## Alternatives considered

- **An `all` mode that expands to every pair.** Rejected: it is one list comprehension in a config, it hides a
  quadratic cost behind one word, and the roadmap names anchor.
- **The anchor as the first space of each pair under `stop_gradient: source`.** Rejected: "target" is the second space
  of a pair everywhere else, and an anchor that is the second element reads as "toward" in the config.
- **A learned multi-way reference as the loss now.** Rejected: gated (P7-4), and without the baseline there is nothing
  to say that it is needed.
- **Reading the pair evaluation `K (K - 1) / 2` times with `--latent-pair`.** Rejected: it fits `K (K - 1) / 2` maps, which
  is the baseline's own cost, and it cannot show the frame or the anchor.

## Consequences

- No change for a config that lists `pairs`: the terms and their error messages are the same.
- A model with an anchor reads in the report as `anchor` against `identity`: the learned alignment to the hub shows in
  the identity rows of the pairs that include it, and how far the non-anchor spaces agree through it shows in the rows
  that do not.
- The common frame is orthogonal, so it keeps the geometry of each space and cannot correct a directional mismatch; where
  `identity` or a learned alignment beats it, that is a difference worth reading, and the correction of the paper is
  where it would be closed.
- Limits: the maps are fitted on posterior means, in sample, with the fit and the reading on different samples; the
  retrieval needs the views to share a width; no real data (I-15).

Tests: `tests/unit/test_pair_loss_terms.py` (`TestResolvePairs`, `TestAnchorMode`, `TestAnchorModeFromTheConfig`: the pairs
that anchor mode builds, the trainer running it, the config composed from the experiment file),
`tests/unit/test_multiway_alignment.py` (the frame recovers views that are rotations of one another, equals the
two-view Procrustes, converges and never lowers its agreement, the bill of maps, the report) and
`tests/integration/test_multiway_script.py` (the flag, its files and each refusal).
