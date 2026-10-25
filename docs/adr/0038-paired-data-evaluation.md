# 0038: Paired-data evaluation

**Status:** accepted (roadmap item P3-6; it adds opt-in modules to `evaluation/` and flags to
`scripts/evaluate.py`, and changes no core module)
**Date:** 2026-10-25

## Context

`evaluate` reports how well each decoder reconstructs its modality, and
`computeCrossModalReconstructionMetrics` ([ADR 0016](0016-cross-modal-reconstruction-reporting.md))
reports the same after the model was given only a subset of the inputs. Neither says whether the
modalities of one sample end up at the same place in a latent space, which is what spec §5 asks of
a model "trained and queried with any subset of available modalities". Three things were missing:

- **Cross-modal retrieval and alignment.** Given the posterior mean of a sample seen through one
  modality, is the posterior mean of the same sample seen through another the nearest one?
- **A per-modality table.** The results of an evaluation are a nested dictionary. A report, a pull
  request or a spreadsheet wants one row per modality and one column per metric.
- **Metrics for a very large test set.** `evaluate` keeps every original and every reconstruction
  of every modality until the end, so its memory grows with the test set. `evaluation/NOTE.md` had
  deferred "streaming versions of `mse`, `rmse` and `mae`" to this item.

The framework has no contrastive or alignment loss, and spec §5 asks for none. What retrieval
measures is what joint training through a fused or shared latent space induces.

## Decision

**1. A modality-specific posterior mean is the `mu` of a latent space when the model is queried
with that modality alone.** `collectModalityPosteriorMeans` runs `GlobalVae.forward` once per input
modality per batch, with `use_mean=True`, and keeps `latent_params[name][0]` for every latent space
that appears. By the rule of ADR 0016, a latent space fed by one active encoder is that encoder's
own output, so this is the encoder's belief about the sample, before any fusion. A latent space
that a modality alone does not reach (the private space of the other modality in a shared plus
private model, [ADR 0024](0024-encoder-fan-out-through-latent-heads.md)) is absent from that
pair: there is nothing to compare, and "absent, not an error" is the convention of
`exportCrossModalFigures`. A model with fewer than two input modalities
(`model.input_modalities`, which counts the modalities of a shared encoder one by one,
[ADR 0027](0027-shared-encoders-and-decoders.md)) returns empty results without reading the data.
The collector is the only part that runs the model, and a batch is skipped if it carries fewer than
two input modalities. Row `i` of every modality in a batch is taken to be the same sample, which is
what the paired loader of `examples/04_paired_loader.py` produces.

**2. Retrieval ranks the true partner among all the others.** `computeRetrievalMetrics(queries,
gallery)` takes two `(N, D)` matrices whose row `i` is the same sample, scores each query against
every gallery row by cosine similarity (default) or negative squared Euclidean distance (`l2`),
and reports, for each direction (`a->b` and `b->a`): `recall@k` for each requested `k` (default
1, 5, 10), `chance_recall@k` (`min(k, N) / N`, what a random ranking gets), `mean_rank`,
`median_rank` (1 is best, `(N + 1) / 2` is chance) and `mrr`. Scoring is done `chunk_size` queries
at a time (default 1024), so the `N x N` score matrix never exists: the memory is `chunk_size x N`.

**Ties count as a random order would count them.** Several gallery rows can score the same as the
partner, and a collapsed latent space (every sample at one point) makes all of them tie. Sorting
with an arbitrary tie-break would let such a model score perfect retrieval or the worst, by the
order of the rows. With `greater` rows strictly better than the partner and `tied` rows equal to it
(within `1e-9 * (1 + |score|)`), the rank is `greater + 1 + tied / 2`, the recall at `k` is
`clamp((k - greater) / (tied + 1), 0, 1)` (the probability that a random order of the tied rows puts
the partner within `k`), and the reciprocal rank is its expectation
`(digamma(greater + tied + 2) - digamma(greater + 1)) / (tied + 1)`. A collapsed latent space
therefore scores exactly chance (`recall@k = k / N`, mean rank `(N + 1) / 2`), which is what it
deserves, and a test pins it.

**3. Alignment is closed form.** `computeAlignmentMetrics(first, second)` reports the mean cosine of
paired rows and of mismatched rows (`i != j`), their difference `cosine_gap`, the mean squared
distance of paired and of mismatched rows and their ratio (below 1 when partners are closer than
strangers), and `dimension_correlation`: the mean over the dimensions of the Pearson correlation
between the two modalities' values of that dimension (`dimensions_compared` says how many; a
dimension that is constant in either modality is left out). Every mismatched mean comes from sums
over the rows (`sum_i sum_j x_i . y_j` is `(sum x) . (sum y)`), so the cost is `O(N * D)`, and the
tests check each value against a brute-force double loop.

**4. Results are a dataclass, and the module is opt-in.** `evaluatePairedAlignment(model,
dataloader, ks, similarity, chunk_size, device, max_samples)` returns `PairedAlignmentResults`
(`retrieval[latent]["a->b"]`, `alignment[latent]["a|b"]`, `num_pairs`, `toDict`, `save`,
`summary`). The settings are validated before any data is read. It is its own module, like
`cross_modal.py` and for the same reason: `evaluate()` and every existing call site keep their
behavior, and a single-modality caller never reasons about a branch that does not apply to it.

**5. The metrics table is a view over the two existing results.** `buildMetricsTable(evaluation,
cross_modal, omit_subsets)` returns a `MetricsTable`: one row per `(inputs, modality)`, one column
per metric in order of appearance, `all` for the ordinary pass and the sorted names joined by `+`
for a subset of the inputs. The rows of an input subset come smaller subsets first. A metric a cell
does not have is an empty cell, and an undefined one (`nan`) is written as `nan`. `toMarkdown`
(6 significant digits, `|` escaped), `toCsv` (10), `toDict` and `save` (the suffix `.md`, `.csv`
or `.json` picks the format, anything else is a `ValueError` and writes nothing).

**6. All five metrics stream, not three.** `evaluation/NOTE.md` expected `r2` and `pearson_r` to
need the pooled tensors. They do not: both are functions of running moments. `streaming_metrics.py`
holds `StreamingMse`, `StreamingRmse`, `StreamingMae`, `StreamingR2` and `StreamingPearsonR`, each
with `update(reconstruction, target)`, `compute()` and `count`, the same names and order as
`DEFAULT_RECONSTRUCTION_METRICS` in `STREAMING_RECONSTRUCTION_METRICS`, and `buildStreamingMetrics()`.
The state is a count, a mean and a sum of squared deviations (and a co-moment for the pair), merged
batch by batch with the parallel update of Chan, Golub and LeVeque, in `float64`. The result does
not depend on the batch size, and it does not lose digits at a large offset: the textbook
`sum(x^2) - n * mean^2` form is off by about 0.2% for data with a mean of `1e8`, where this one is
exact to rounding (a test pins it at that offset). Edge cases follow the pooled functions: `r2` is `nan` for a
constant target, `pearson_r` is `nan` if either side is constant and is clamped to `[-1, 1]`, and
`compute()` before any value raises.

**7. `evaluate(streaming=True)` and what stays pooled.** With `streaming=True` the per-modality
metrics are fed batch by batch and nothing about reconstructions or originals is kept, and
`EvaluationResults.metrics_streaming` records it (`summary()` says "streamed"). Combining it with
custom `reconstruction_metrics` raises, since those are functions of the pooled tensors. Three
things are still held in memory: the latent `(mu, logvar)` of every sample (`N x latent_dim`, which
is small), the reconstructions that `--paired` pools for the cross-modal table (bounded by
`--max-samples`), and, for the retrieval, the posterior means (`N x latent_dim` per modality). A
test replaces the pooled metric functions with ones that raise, to show that a streamed evaluation
never calls them.

**8. The script exposes it as flags.** `scripts/evaluate.py` gains `--streaming`, `--paired`,
`--retrieval-k` (default `1 5 10`, validated with the same function as the library) and
`--retrieval-similarity`. With `--output-dir` it always writes `metrics_table.md` and
`metrics_table.csv`, and with `--paired` also `paired_results.json`. `--paired` on a model with a
single input modality logs that it does nothing. A dataloader factory that returns a generator is
read once per pass by calling the factory again, which is why `--paired` can follow the main
evaluation.

**9. No core change.** `models/`, `latent/`, `losses/` and `config/` are untouched: everything is
built on `GlobalVae.forward`, `selectEncoderInputs` and the results the framework already returns.

**10. Measured on the synthetic pairs (a check that the metrics run and respond, not a target).**
Three experiments trained for 40 epochs on the 1,026 synthetic pairs of
`examples/04_paired_loader.py` (one seed, `sum` at 0.5 for all three, as in ADR 0037), then
evaluated with `evaluatePairedAlignment` on the 103 test pairs. At `N = 103` a random ranking has a
mean rank of 52 and a `recall@10` of 0.097. `z_fused` is the only latent space these models share,
so it is the only one with numbers.

| Experiment | Direction | Mean rank | `recall@10` | `cosine_gap` | `dimension_correlation` |
|---|---|---:|---:|---:|---:|
| `signal_image_transformer_vae` (`cross_attention`) | signal -> image | 28.0 | 0.252 | 0.058 | 0.300 |
| | image -> signal | 51.2 | 0.117 | | |
| `signal_image_shared_transformer_vae` | signal -> image | 28.8 | 0.214 | 0.089 | -0.225 |
| | image -> signal | 49.7 | 0.097 | | |
| `signal_image_vae` (PoE, 2D CNN) | signal -> image | 44.8 | 0.097 | 0.061 | 0.357 |
| | image -> signal | 37.1 | 0.155 | | |

What they say: all three are only weakly aligned, and nothing in them is close to retrieving the
partner. That is what the data allow as much as what the models do. The generator gives the signal
and the image of a sample one thing in common, the size of the object, and the image also carries
a position that the signal has no trace of, so even a perfect model could not tell two samples of
nearly the same size apart. The metrics do respond: every `cosine_gap` is positive and every
`squared_distance_ratio` is below 1 (0.92, 0.65 and 0.88), the retrieval beats chance in at least
one direction for each model, and on the 820 training pairs the same numbers hold at about the same
distance from chance (mean rank 226 and 235 for signal -> image with a chance of 410), so the
weak alignment is not a gap between training and test. Two things are visible and not investigated
here: the two directions differ (which one is better depends on the model), and the shared
transformer has a negative mean per-dimension correlation next to a distance ratio below 1, which
is what two modalities coding one factor along different axes would give. The cosine, distance
and per-dimension metrics are not invariant to a rotation of the latent space, by design, because
the encoders of one latent space are meant to share its axes. A real dataset is what these numbers
are waiting for.

## Consequences

- **A recall must be read against its chance level.** `recall@5` of 0.2 is strong at `N = 5000`
  (chance `0.001`) and nothing at `N = 20`. Every report carries `chance_recall@k` for that reason,
  and the how-to says so. Compare two runs only at the same `N`, or by mean rank divided by
  `(N + 1) / 2`.
- **Retrieval costs one forward per modality per batch, decoders included.** The collector runs
  the whole model, because `GlobalVae.forward` has no encode-only path. On a large set,
  `--max-samples` bounds it, and an encode-only method would make it cheaper; it is a core change,
  so it is not made here.
- **The numbers describe what training induced.** No loss asks two modalities to agree in a
  latent space except through the fusion or the shared decoder. A contrastive term is a different
  decision (a new loss in `losses/`, with a weight and a schedule), and if it is taken, these
  metrics are how it is checked. A model with a poor alignment is not a bug here.
- **A pair is assumed, not checked.** If row `i` of two modalities is not the same sample, the
  recall is chance and nothing says why. The pairing is the loader's job (spec §6.2).
- **A latent space that only one modality reaches has no retrieval numbers.** The private spaces
  of a shared plus private model never appear. The shared space does, and it is the one to read.
- **The streaming metrics are exact for any batch size.** They are not an approximation, so there
  is no switch that trades accuracy for memory.
- **Retrieval is exact and quadratic in time.** It scores every query against every gallery row,
  `O(N^2 * D)`. A gallery too large for that would need an approximate nearest-neighbour index,
  which is not built because nothing at the sizes in view needs it.
- **Not done.** The spec §8 file tree was not edited, and the retrieval compares two modalities
  at a time: with three, each pair is its own entry.
