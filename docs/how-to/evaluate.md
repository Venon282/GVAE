# Evaluate and visualize a checkpoint

Two standalone scripts cover this, neither needing a `Trainer` (only a
`GlobalVae` and a dataloader). Both restrict a raw batch to the keys
naming one of the model's encoders before the forward pass
(`GlobalVae.selectEncoderInputs`,
`docs/adr/0019-decouple-encoder-inputs-from-decoder-targets.md`), so a
decoder whose reconstruction target is not itself an encoder input (a
translation-style model) works with no special-casing on your side.

## Full evaluation report

```bash
python scripts/evaluate.py \
    --checkpoint runs/model.pt \
    --model-factory mypackage.models:build_model \
    --dataloader-factory mypackage.data:build_test_dataloader \
    --output-dir results/
```

`--model-factory` and `--dataloader-factory` are `"module.path:function_name"`
references to your own code, the same convention as `data.loader_factory` in
the training config: the framework never guesses your architecture or data
format. This prints reconstruction metrics (mse / rmse / mae / r2 /
pearson_r), per-latent-space regularization values and latent diagnostics to
the console, and (with `--output-dir`) saves a JSON report plus
reconstruction / latent-space figures.

The reconstruction figure of a modality depends on what its decoder produces. A decoder of 1D
series gets a grid of overlaid curves (`reconstructions_<modality>.png`). A decoder of images
(`(N, H, W)` or `(N, C, H, W)`) gets a grid with the original and the reconstruction side by
side, one row per example, under the same file name. For a model with several input modalities,
the cross-modal figures show what each decoder produces from each subset of the inputs: the
decoders of 1D series share `cross_modal_reconstructions.png`, and each image decoder has its
own `cross_modal_reconstructions_<decoder>.png`. In both image figures a row shares one colour
scale between the original and the outputs, so the same colour means the same value across it.

From Python, the same figures are `plotImageReconstructionGrid` (with `show_error=True` for a
column of absolute error) and `plotImageCrossModalReconstructionMatrix` in
`global_vae.visualization.image_plot`. The second takes the `inputs` you want drawn before the
target (a signal as a line, an image as an image) and several `example_indices`, which the
command line, showing one example, does not.

The checkpoint is read in the safe mode (`weights_only=True`,
[ADR 0034](../adr/0034-safe-checkpoint-loading-by-default.md)). A file saved by an
earlier version of the framework is refused with a message that says so; if you trust it,
either rewrite it once with `upgradeCheckpoint` (from `global_vae.training.checkpoint`) or pass
`--trust-checkpoint`, which reads it as it is and runs pickle code. The same flag exists on
`scripts/visualize_latent.py`.

The latent diagnostics are, for each latent space, the KL of every latent
dimension and the number of **active units**: dimensions whose posterior mean
varies over the evaluation set by more than a threshold (the variance of `mu`
above 0.01; `--active-unit-threshold` changes it). A latent space with few
active units and a KL near zero on the rest has collapsed on those
dimensions. [Choose beta](choose-beta.md) says how to read them.

Pass the reduction, the loss and the weight the model was trained with, so that
the total reconstruction loss of the report is on the scale of the training
curves: `--reconstruction-reduction` (`training.reconstruction_reduction`:
`mean`, the default, `sum`, or `modality=name` pairs such as
`signal=sum,image=mean`, [ADR 0029](../adr/0029-loss-scale-convention.md)),
`--reconstruction-loss` (a registered name: `mse`, the default, `l1`, `bce`, `smooth_l1`,
`bce_with_logits`, `masked_mse`, `token_cross_entropy`, `chamfer` or `tabular_mixed`, or pairs
such as `signal=mse,mask=bce`) and `--reconstruction-weight` (a number, default
`1`, or pairs such as `signal=0.5,mask=0.2`,
[ADR 0030](../adr/0030-per-modality-reconstruction-loss-and-weight.md)). A modality name that the model does not have is a usage
error. The reduction and the weight do not change the per-modality metrics. The loss does, as the
next section says.

## Which metrics a modality gets

The metrics of a modality follow the loss it was trained with
([ADR 0048](../adr/0048-reconstruction-loss-and-metric-registries.md)), so that the report says
something about the data and not only about a squared error. By default:

| The loss of the modality | The report has |
|---|---|
| `mse`, `l1`, `bce`, `smooth_l1`, `bce_with_logits`, `tabular_mixed`, or a function | `mse`, `rmse`, `mae`, `r2`, `pearson_r`, as in 2.0.0 |
| `masked_mse`, which reads a `NaN` target as "not observed" | the same five names, computed on the observed values only, so a series with gaps gives numbers and no `NaN` |
| `chamfer` | `chamfer`, `precision`, `recall`, `f_score` of the clouds (threshold `0.1`, in the units of the data), where a row of `NaN` in the target is a point that does not exist |
| `token_cross_entropy` | `token_accuracy`, the share of positions where the highest score is the target token |

Ask for others by name with `reconstruction_metrics` in Python. It takes `None` (the default
above), a list of names for every modality, a dict of functions for every modality (the form of
2.0.0), or a mapping from a reconstruction name to a list of names or a dict:

```python
results = evaluate(
    model,
    dataloader,
    reconstruction_loss_fn=reconstruction_loss_fn,
    reconstruction_metrics={"point_cloud": ["chamfer", "f_score"], "image": ["mse", "r2"]},
)
```

`listRegisteredMetrics()` lists the names: the five above, their `masked_*` forms (a metric
computed on the values whose target is not `NaN`), `chamfer`, `precision`, `recall`, `f_score`,
`edge_auc` (graphs, every node present) and `token_accuracy`. A function of your own joins them with
`@registerMetric("name")`, a function of the reconstruction and the target that returns a float. A
name that is not registered is refused before any batch is read, with the list of the registered
ones. In the table of `--output-dir` a modality has a blank where its set has no such metric. The
cross-modal metrics of `--paired` keep the five default metrics.

If your own data pipeline applied a `data.transforms` step (`log` /
`standardize` / `resample`, spec §6.2) before training, reconstruction
figures are plotted in that transformed space by default. Pass
`--inverse-transform-factory mypackage.data:build_inverse_transforms`
(a factory returning `dict[str, Callable[[Tensor], Tensor]]`, e.g. built
straight from a `DataConfig` via `buildTransformPipeline`) to have them
plotted in the original, physical-units space instead.

## A per-modality table

With `--output-dir`, the reconstruction metrics are also written as a table, one row per modality
and one column per metric, as `metrics_table.md` (for a report or a pull request) and
`metrics_table.csv` (for a spreadsheet). The values below are made up to show the layout:

```text
| inputs | modality | mse | rmse | mae | r2 | pearson_r |
|---|---|---:|---:|---:|---:|---:|
| all | signal | 0.0123 | 0.111 | 0.0832 | 0.971 | 0.986 |
| all | image | 0.0456 | 0.214 | 0.153 | 0.62 | 0.79 |
```

The first column says which inputs the model was given: `all` for the ordinary pass. A model
with several modalities can be queried with a subset of them (spec §5), so with `--paired` the
table also has one row per input subset and modality: the row with `signal` as its inputs and
`image` as its modality is the image reconstructed from the signal alone. From Python,
`buildMetricsTable(results, cross_modal)` in `global_vae.evaluation.metrics_table` builds the same
`MetricsTable`, and `MetricsTable.save(path)` writes `.md`, `.csv` or `.json` by the suffix.

## A test set too large to pool

`evaluate` computes each metric over the whole pooled test set, which keeps every original and
every reconstruction in memory. `--streaming` (`evaluate(..., streaming=True)` in Python) computes
the same five metrics, and their masked forms, one batch at a time from running statistics, so the
memory they need does not grow with the test set. The values equal the pooled ones up to the
rounding of the `float32` pool, whatever the batch size. It cannot be combined with custom
`reconstruction_metrics`, which are functions of the pooled tensors, and it refuses a metric that
needs the pool (`chamfer`, `token_accuracy`) with a message that names it. What stays in memory is the latent posteriors
(`num_samples x latent_dim` per latent space); `--max-samples` still bounds everything.

## Paired data: retrieval and alignment

A model fed a signal and an image of the same sample should put the two at the same point of its
latent space. Reconstruction metrics do not say whether it does. `--paired`, for a model with
several input modalities, queries the model with each modality alone, keeps the posterior mean
of each latent space, and compares the means of the two modalities of every sample
(`paired_results.json`, and a summary on the console):

- **Retrieval.** Each sample's mean from one modality is a query; all the means from the other
  modality are ranked by similarity (`--retrieval-similarity cosine`, the default, or `l2`). The
  true partner is the match. Reported in both directions, `signal->image` and `image->signal`:
  `recall@k` for each `--retrieval-k` (default 1 5 10), the mean and median rank of the partner
  (`1` is best, `(N + 1) / 2` is chance), and the mean reciprocal rank. Each `recall@k` comes with
  `chance_recall@k = k / N`, the value of a random ranking, which is the number to read it against.
- **Alignment.** `cosine_paired` against `cosine_mismatched` (the mean cosine of a sample's two
  views, and of views of different samples; `cosine_gap` is the difference), the squared distance
  of paired and of mismatched samples and their ratio (below `1` when partners are closer than
  strangers), and `dimension_correlation`, the mean over the latent dimensions of the correlation
  between the two modalities' values of that dimension.

A tie counts as a random ordering would count it. A model whose latent space has collapsed (every
sample at the same point) therefore reports `recall@k = k / N`, not a perfect retrieval.

Read it with care. Nothing in the training loss asks the two modalities to agree except through
the fusion and the shared decoder, so these numbers measure what that induced, and they depend on
the size of the latent space and of the evaluation set: compare a run with another only at the same
`N`, or against its own `chance_recall@k`. The batches must carry the modalities of one sample
together, and row `i` of each modality must be the same sample, which the paired loader of
`examples/04_paired_loader.py` guarantees. A latent space that only one modality feeds (the private
spaces of a shared plus private model) has nothing to compare and is left out.

From Python:

```python
from global_vae.evaluation.paired_alignment import evaluatePairedAlignment

results = evaluatePairedAlignment(model, test_dataloader, ks=(1, 5, 10), similarity="cosine")
print(results.summary())
results.retrieval["z_fused"]["signal->image"]["recall@5"]
results.alignment["z_fused"]["signal|image"]["cosine_gap"]
```

`computeRetrievalMetrics(queries, gallery)` and `computeAlignmentMetrics(first, second)` take two
`(samples, latent_dim)` matrices directly, for means you collected yourself. Retrieval is computed
`chunk_size` queries at a time, so a large `N` does not need an `N x N` matrix.

## Quick latent-space inspection

```bash
python scripts/visualize_latent.py \
    --checkpoint runs/model.pt \
    --model-factory mypackage.models:build_model \
    --dataloader-factory mypackage.data:build_dataloader \
    --output-dir results/latent/
```

Faster than a full evaluation pass when you just want to look at the latent
space and training curves: saves a latent-space scatter plot (PCA, the
default projection, is an exact, deterministic decomposition, so this and
`scripts/evaluate.py`'s own latent scatter plot agree on the same
checkpoint and dataloader) and a per-dimension KL bar chart per latent
space (for spotting posterior collapse), plus the training-curve plot if
the checkpoint carries a history: split by default onto two
independently-scaled axes (reconstruction/total loss vs. regularization
loss, spec §2.3; `--history-no-twin-axis` to disable, `--history-log-scale`
/`--history-twin-log-scale` for either axis). Supports `--label-key` (color
the scatter by a batch field), `--latent-names` (restrict which latent
spaces to plot), and `--use-samples` (plot realized samples instead of the
posterior mean).

Run either script with `--help` for the full option list.
