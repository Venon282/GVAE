# 0044: The point cloud modality

**Status:** accepted (roadmap item P3-8, sixth and last modality of the list after tabular, audio, text, time
series and graphs; it adds one encoder class, one decoder class, one set loss and one scoring function through
the registries and `utils/`, and changes no core module)
**Date:** 2026-10-30

## Context

Roadmap P3-8 lists point clouds among the further modalities, each through the new-modality checklist of spec
§10 and only once a dataset is named. No point cloud collection has been named, so nothing here is tuned to
one: the batch contract is stated, and the measurements below are on synthetic clouds. The tabular
([ADR 0039](0039-tabular-modality.md)), audio ([ADR 0040](0040-audio-modality.md)), text
([ADR 0041](0041-text-modality.md)), time series ([ADR 0042](0042-time-series-modality.md)) and graph
([ADR 0043](0043-graph-modality.md)) modalities followed the checklist before it.

The graph ADR ended on a finding: a loss that compares two tensors position by position makes the order of
the entries part of the data, and no encoder removes that. A point cloud is that finding in its pure form. It
is a **set**: `N` points in 3 dimensions with no order at all, so the same cloud written in two row orders is
the same object and two different tensors. The framework's unit is one tensor per modality, an encoder that
maps it to `(mu, logvar)` and a decoder that maps a latent vector back to a tensor of the same shape, compared
by `computeTotalReconstructionLoss`. A cloud fits that as `(batch, N, 3)`. What does not fit is the
comparison.

Three questions were measured before anything was written:

- **The loss.** Every loss a YAML file can name (`mse`, `l1`, `bce`, `smooth_l1`) pairs row `i` of the
  reconstruction with row `i` of the target. For a set that pairs two unrelated points. The order-free
  alternative is the Chamfer distance, and a YAML file cannot name it.
- **The encoder.** Whether the registered classes, which read their input as a sequence, already do well on a
  set, and what an encoder that is exactly blind to the order and to padding would add.
- **The decoder.** Whether one latent vector can write a set of points with the registered decoders, and what
  a decoder with no sequence structure does instead.

## Decision

**1. A point cloud is a `(batch, N, 3)` float tensor, a row of `NaN` is a point that does not exist.** The
batch key is `point_cloud`, and `N` is the same for the whole dataset (`sequence_length.point_cloud` in the data
file, `num_points` in the decoder). A cloud with fewer than `N` points is padded with rows of `NaN`, which are
a point that is not there, and not the origin, which is a point. Points that carry more than coordinates
(normals, colors) are `(batch, N, 3 + k)`, with `k` set in the encoder (`in_dim`) and the decoder (`out_dim`).
Reading files, sampling `N` points, centering and scaling each cloud and padding are the `loader_factory`'s
(spec §6.2), and `docs/how-to/train-on-point-clouds.md` has one.

**2. Two classes, one for each side of the checklist.**

| Registry name | Class | Reads or writes |
|---|---|---|
| `point_cloud_encoder_v1` | `PointCloudEncoder` | a cloud `(batch, N, in_dim)`, any `N`, with `NaN` rows allowed |
| `point_cloud_decoder_v1` | `PointCloudDecoder` | a cloud `(batch, num_points, out_dim)` from one latent vector |

The encoder is the PointNet design: a shared MLP applied to every point alone, a pooling over the points that
ignores the `NaN` rows (`max_mean` by default, `max` or `mean`), and a head to `(mu, logvar)`. It is exactly
invariant to the order of the points and to the padding (tests pin both, with the padding in the middle and
scattered, and a cloud of no point gives a finite latent vector), and its cost is linear in `N`. Batch
normalization is refused (it would mix the points of different clouds), layer normalization is allowed. The
decoder is a folding of a learned template: `num_points` learned codes, each concatenated with the latent
vector and mapped to a point by one shared MLP. Its `num_points` is any positive integer, nothing has to land
on a size, and a draw of the prior decodes to a cloud with no further step. Both are built from blocks that
already existed (`utils/mlp_blocks.buildMlp`, `utils/transformer_blocks.buildPoolingHead`). They are registered
by the import in `encoders/__init__.py` and `decoders/__init__.py`. `models/`, `latent/`, `losses/`,
`training/`, `evaluation/`, `config/` and `data/` are untouched, and a test reads their source to check that
none of them names a class or the `point_cloud` batch key.

**3. The loss is the Chamfer distance, a Python function, and the YAML file only points to it.**
`utils/point_cloud_metrics.computeChamferDistance(reconstruction, target)` sums, for each cloud, the squared
distance from every reconstructed point to its nearest target point and from every target point to its nearest
reconstructed one, and averages over the clouds. Rows of `NaN` in the target are left out, so padded clouds
train as they are. It takes any two numbers of points. The distances use the `|a|^2 + |b|^2 - 2 a.b` form so
that the gradient is finite, and a test compares it with the definition written out point by point.

`Trainer(reconstruction_loss_fn=...)` accepts any Python function of `(reconstruction, target)`, so the route is
to build the trainer from the experiment file, which gives the optimizer, the beta schedule and the weight, and
to set two attributes:

```python
trainer = buildTrainerFromConfig(model, config.training)
trainer.reconstruction_loss_fn = computeChamferDistance
trainer.reconstruction_reduction = "mean"  # the Chamfer distance is already a sum over points
```

The reduction stays `mean` because the Chamfer distance is already a sum over points: `sum` would multiply it
by the `N * 3` values of a cloud once more (a test pins the factor). The weight is `100.0`, and a weight `w`
on a summed reconstruction is a `beta` of `1 / w` on the KL (roadmap decision D-6, [ADR 0029](0029-loss-scale-convention.md)).

**4. The experiment file is a composed config with a loud warning, and its own loss is a placeholder.**
`configs/experiment/point_cloud_vae.yaml` composes the model, data and training files and validates. Its
`training` block says `mse` with `sum` and weight `100.0` because a YAML file cannot say anything else, and so
that `scripts/train.py` wires up (a slow test runs it on dummy complete clouds). That is **not a
recommendation**: on synthetic clouds the model the script trains this way ends farther from the clouds than a
cloud that ignores its input (Table 4), and on a padded cloud the loss is `NaN`. The header of the file says so
and gives the Python route.

**5. A scoring function for reconstructions.** `computeCloudScores(reconstruction, target, threshold=0.1)`
returns the Chamfer distance per point (both directions added, averaged over the clouds), the precision (the
fraction of reconstructed points within `threshold` of a target point), the recall (the other way) and their
F score, and the number of target points, over the real points only. The report of `evaluate` is not this (see
Consequences).

**6. One model file, one data file and one experiment.** `configs/model/point_cloud_single_latent.yaml` (64
points, a latent of 16, no fusion), `configs/data/point_cloud.yaml` (an empty transform list: a cloud is
centered and scaled one at a time, where the framework's transforms use one mean and scale for a dataset) and
`configs/experiment/point_cloud_vae.yaml`. Points with more channels, another `N`, the pooling and layer
normalization are YAML values and are tested through the shipped file.

## Measurements

Synthetic clouds (`tests/integration/_point_cloud_corpus.py`), not a collection: four shapes (an ellipsoid, a
torus, a cylinder and a box) with random proportions, a random turn, noise of 0.01, each cloud centered and
scaled to a farthest point at distance 1, 40 to 64 points padded to 64 with `NaN` rows, in a random order.
2,000 training clouds and 600 held out, 40 epochs of batches of 64, AdamW at 2e-3, a latent of 16, KL weight 1
from the first step, one CPU thread. **Chamfer** below is the per-point score of `computeCloudScores` (both
directions added, lower is better) on the held-out clouds decoded from the mean latent vector, and **F** is
the F score at a threshold of 0.1. The best cloud that ignores its input (one fixed cloud fitted to the
training set by the same Chamfer loss) scores 0.105 and an F of 0.09. These are single runs unless three seeds
are listed.

**Table 1. The encoder and the decoder, at weight 100.** The registered classes were used as they are, with
`NaN` rows filled with zeros since they cannot read them, and the clouds in a random order. "PointNet" and
"folding" are the layers of the two new classes (written out in the measurement script, with their default
widths), and "self-attention" is two transformer layers of width 64 with the `NaN` rows masked and no
positions. "Reordering shift" is the mean absolute change of the latent mean when the points of the same
clouds are shuffled (0 is invariant).

| Encoder | Decoder | Chamfer (seeds 0, 1, 2) | Mean | F | Reordering shift |
|---|---|---|---|---|---|
| PointNet | folding | 0.0590, 0.0592, 0.0598 | 0.0593 | 0.228 | 0 |
| PointNet | `OneDCnnDecoder` | 0.0602, 0.0591, 0.0606 | 0.0600 | 0.222 | 0 |
| PointNet | a fixed 8 by 8 grid folded | 0.0631 (seed 0) | | 0.205 | 0 |
| PointNet | `TabularMlpDecoder` | 0.0675 (seed 0) | | 0.179 | 0 |
| Self-attention | folding | 0.0583 (seed 0) | | 0.240 | 0 |
| `OneDTransformerEncoder`, `patch_size=1` | folding | 0.0606, 0.0584, 0.0610 | 0.0600 | 0.219 | 0.19 to 0.42 |
| `OneDTransformerEncoder`, `patch_size=1` | `OneDCnnDecoder` | 0.0618, 0.0606, 0.0624 | 0.0616 | 0.201 | 0.16 to 0.33 |
| `OneDTransformerEncoder`, `patch_size=1` | `TabularMlpDecoder` | 0.0630 (seed 0) | | 0.200 | 0.27 |
| `TabularMlpEncoder`, flattened | folding | 0.0956 (seed 0) | | 0.108 | 0.38 |
| `TabularMlpEncoder`, flattened | `TabularMlpDecoder` | 0.0890 (seed 0) | | 0.113 | 0.16 |

- **The registered sequence classes are not far behind.** The 1D transformer encoder (one point per token, no
  convolution to mix neighbours) with the registered 1D CNN decoder lands within 4 percent of the new pair, and
  the registered 1D CNN decoder within 1 percent on the new encoder. The flattened tabular MLPs are clearly
  worse (0.089 to 0.096, near the 0.105 of no input at all). At 64 points the gap to the registered pair is
  small, close to the spread between seeds of a single pair (0.0590 to 0.0598).
- **What the new encoder adds is not accuracy at 64 points.** It adds an exact invariance (shift 0, against
  0.16 to 0.42 for every order-aware encoder: the latent vector of a cloud changes when its rows are
  reshuffled), a `NaN` row that means "no point" and is skipped instead of being read as a point at the origin,
  and a cost that is linear in `N` (Table 6).
- **Folding a learned template is the best decoder tried, and equal to the registered CNN decoder.** It beat the
  fixed grid (0.0631) and the MLP (0.0675) clearly, and the 1D CNN decoder by 1 percent.
- **A set transformer without positions did not beat the PointNet pooling** (0.0583 against 0.0590) and took
  120 s against 69 s, so it was not added.

**Table 2. The weight, through the Chamfer loss (D-6).** PointNet and folding, seed 0.

| Weight | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| Chamfer | 0.0858 | 0.0712 | 0.0590 | 0.0582 |
| F | 0.116 | 0.163 | 0.228 | 0.240 |

**Table 3. The pooling.** PointNet and folding, weight 100, seed 0: `max_mean` 0.0590, `max` 0.0584, `mean`
0.0677. The mean alone loses the extremes of a shape, and the maximum alone is as good as both, within the
noise. `max_mean` stays the default for the cost of one more feature.

**Table 4. The loss.** 2,000 complete clouds of 64 points (no padding, since the loss of a YAML file is `NaN`
on a `NaN` row), PointNet and folding, seed 0. The best cloud that ignores its input scores 0.0976.

| Loss | Order of the training points | Chamfer | F |
|---|---|---|---|
| Chamfer, weight 100 | random | 0.0531 | 0.239 |
| `mse` summed, weight 100 | random | 0.8149 | 0.000 |
| `mse` summed, weight 10 | sorted by the first coordinate | 0.4479 | 0.024 |
| `mse` summed, weight 100 | sorted by the first coordinate | 0.3928 | 0.035 |
| `mse` summed, weight 1000 | sorted by the first coordinate | 0.3764 | 0.038 |

The loss of a YAML file trains a model **several times worse than ignoring the input** (0.81 and 0.38 to 0.45
against 0.10), because it matches point `i` of a reconstruction with point `i` of an unrelated cloud and the
best answer to that is every point at the center. Sorting the points helps and does not rescue it, and a
different decoder (the CNN one, 0.3859) does not either.

**Table 5. Another number of points.** 80 to 128 points, padded to 128, 2,000 clouds, weight 100, seed 0. The
best cloud that ignores its input scores 0.0737.

| Encoder | Decoder | Chamfer | F | Seconds |
|---|---|---|---|---|
| PointNet | folding | 0.0331 | 0.419 | 180 |
| PointNet | `OneDCnnDecoder` | 0.0329 | 0.424 | 142 |

At 128 points the two decoders are equal. The sixty-four-point result (folding 1 percent ahead) is not a
trend.

**Table 6. The cost.** Forward and backward, a batch of 32 clouds, one CPU thread, milliseconds.

| Points | PointNet encoder | `OneDTransformerEncoder`, `patch_size=1` | Folding decoder | Chamfer loss |
|---|---|---|---|---|
| 64 | 13 | 43 | 9 | 2 |
| 256 | 36 | 251 | 43 | 22 |
| 1024 | 184 | 2262 | 192 | 877 |

The encoder and the decoder are linear in `N` and the attention encoder is quadratic, so the PointNet encoder
is 3 times faster at 64 points and 12 times faster at 1,024. **The Chamfer loss is quadratic too**, and at
1,024 points it costs more than the encoder and the decoder together.

The pooling, the number of points and the weight are tuned to nothing: they are one synthetic shape set, three
seeds at most for the architectures and one for the other tables. They show that the model trains, how the
registered classes compare and what the loss of a YAML file does, not what a real collection would give.

## Consequences

- **The registry design held a sixth time, with two classes.** One encoder, one decoder, a loss and a scoring
  function, three config files and the tests are all that was added, and no core edit was needed or made.
- **No new class would have been a legitimate alternative.** The registered 1D transformer encoder and 1D CNN
  decoder, on rows with `NaN` replaced by zero and the Chamfer loss, score within 4 percent of the new classes
  at 64 points and equal at 128 (Tables 1 and 5). The new classes are justified by the exact invariance, the
  `NaN` convention that matches the loss, the `(batch, points, dim)` layout without a transpose, the free
  `num_points` and the cost (Table 6), and not by a large gain in the reconstruction. A project that wants
  fewer classes can use the registered pair and loses a few percent.
- **Finding, not fixed: the loss table has no set loss, so the experiment file cannot name the loss that
  works.** `config/training.py` has a fixed table of names, and a position-by-position loss on a set is worse
  than no model (Table 4). The Chamfer distance is a Python function the trainer accepts, so the route is
  Python (above) and `scripts/train.py` on this experiment trains the wrong objective. A `chamfer` entry in the
  loss table, or a loss given as a dotted path, is a core change, flagged and not made.
- **Finding, not fixed: the trainer refuses a reconstruction whose shape differs from its target's, for any
  loss.** `computeChamferDistance` takes two different numbers of points, so a decoder could write 256 points
  for clouds of 64, and the check in `computeTotalReconstructionLoss` refuses before the loss is called (a test
  pins it). `num_points` must equal the `N` of the data file, and a different number is possible only by
  calling the function directly, as `computeCloudScores` does. Relaxing the check for a loss that declares it
  does not need equal shapes is a core change, flagged and not made.
- **Finding, not fixed: the metrics of `evaluate` are position by position.** `evaluate` and
  `scripts/evaluate.py` run on a point cloud model (a test covers them and the figures), and with the Chamfer
  loss the total reconstruction loss is right, but the `mse`, `rmse`, `mae`, `r2` and `pearson_r` of the report
  compare point `i` with point `i`, so they mean nothing for a set and are `NaN` on a padded cloud (a test pins
  it). Read the total loss and `computeCloudScores`. Set-level metrics in `evaluation/` are a core change,
  flagged and not made.
- **Finding, not fixed: no `NaN` handling in the loss table either.** A row of `NaN` is "no point" to the
  encoder and the Chamfer loss and `NaN` to every other loss, so a dataset padded for this modality cannot be
  trained with a YAML loss at all. The slow test of `scripts/train.py` uses complete clouds for that reason.
- **Not modeled: the number of points, and what a point carries beyond coordinates.** The decoder writes
  exactly `num_points` points and no count, so a generated cloud has the size the model was built for. Normals
  and colors are extra channels (`in_dim`, `out_dim`) that the Chamfer distance compares together with the
  coordinates, as one vector, with no weight between them. A per-channel weight is a different loss.
- **Not modeled: large clouds, local structure and one latent vector per cloud.** The Chamfer loss is quadratic
  in `N` (1,024 points cost 877 ms in a batch of 32), so a large cloud is sampled to `N` points by the loader,
  and the encoder pools to one vector with no neighbourhoods, so it reads global shape and not fine detail. A
  latent per region would be the single-tensor decoder item of the roadmap audit P4-0.
- **No dataset, no loader.** Reading files, sampling, centering, scaling and padding are the loader's, no named
  collection exists, and the proof is the synthetic clouds and the dummy loader of the tests. With this
  modality the list of P3-8 is complete.
- **Cost.** The shipped file is 181,152 parameters in the encoder and 81,155 in the decoder (262,307 in all),
  and about 70 s for 40 epochs of 2,000 clouds on one CPU thread.
