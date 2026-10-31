# Train on point clouds

A point cloud is the point cloud modality ([ADR 0044](../adr/0044-point-cloud-modality.md)): a set of points in 3
dimensions, read by `point_cloud_encoder_v1` and rebuilt by `point_cloud_decoder_v1`. Reading files, sampling,
centering, scaling and padding all happen in your `loader_factory`, as for any other data (spec §6.2). The one
thing that is different from the other modalities is the loss: a set has no order, so the loss is the Chamfer
distance, and it is set in Python.

## What a batch is

A batch is a dict with one key, `point_cloud`, holding a **float tensor `(batch, N, 3)`**: one row per point.

- **The rows are in no particular order.** The encoder does not depend on it, and the Chamfer loss does not
  either. Nothing has to be sorted.
- **`N` is the number of rows**, the same for the whole dataset. It is written in the data file
  (`sequence_length.point_cloud`) and in the decoder (`num_points`), and the two have to be equal: the trainer
  refuses a reconstruction whose shape differs from its target's. Nothing fills either one in for you, and any
  positive integer works.
- **A row of `NaN` is a point that does not exist.** A cloud with fewer than `N` points is padded with rows of
  `NaN`, at the end or anywhere else. The encoder leaves them out and the Chamfer loss does too, so a cloud of 40
  points among 64 rows is a cloud of 40 points. Do not pad with zeros: the origin is a point.
- **Every cloud is centered and scaled by the loader.** The model reads the coordinates as they are, and a cloud
  that is not normalized is a cloud the model has to learn the position and the size of. The framework's
  transforms standardize with one mean and one scale for the whole dataset, so `transforms` stays empty in the data
  file (the identity), and the normalization of each cloud is part of the loader.
- **Points that carry more than coordinates** (normals, colors) are `(batch, N, 3 + k)`, with `in_dim: 3 + k` in
  the encoder and `out_dim: 3 + k` in the decoder. The Chamfer distance compares all the channels as one vector,
  with no weight between them, so scale the extra channels like the coordinates.
- **The number of points of each cloud** can ride in the batch under another key, such as `num_points`. The
  trainer, `evaluate` and the figures restrict a batch to the keys the model encodes before calling it
  (ADR 0019), so they ignore it. Calling the model yourself on such a batch needs the same restriction,
  `model.selectEncoderInputs(batch)`, or it raises a `KeyError`.

## A `loader_factory`

`configs/data/point_cloud.yaml` describes the contract and leaves `loader_factory` and `train_path` for you to fill
in. This is one that follows it. `loadMyClouds` is yours: it reads your files and returns a list of float tensors,
each `(n, 3)`, of any `n` of at least one.

```python
import torch
from torch.utils.data import DataLoader

from global_vae.config.data import DataConfig, DataloaderBundle


def normalizeCloud(points: torch.Tensor) -> torch.Tensor:
    """Center a cloud on its mean and scale it so that its farthest point is at distance 1."""
    centered = points - points.mean(dim=0)
    return centered / centered.norm(dim=1).max().clamp_min(1e-8)


def sampleOrPad(points: torch.Tensor, size: int, generator: torch.Generator) -> torch.Tensor:
    """Keep a random `size` of the points of a large cloud, or pad a small one with rows of NaN."""
    if len(points) >= size:
        return points[torch.randperm(len(points), generator=generator)[:size]]
    padding = torch.full((size - len(points), points.shape[1]), float("nan"))
    return torch.cat([points, padding])


def collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def buildPointCloudDataloaders(config: DataConfig) -> DataloaderBundle:
    clouds = loadMyClouds(config.train_path)  # a list of (n, 3) float tensors
    size = (config.sequence_length or {})["point_cloud"]
    generator = torch.Generator().manual_seed(config.seed)
    rows = [
        {"point_cloud": sampleOrPad(normalizeCloud(cloud), size, generator)} for cloud in clouds
    ]
    order = torch.randperm(len(rows), generator=generator).tolist()
    num_val = int(len(rows) * config.val_split)
    num_test = int(len(rows) * config.test_split)
    split = {
        "train": order[num_val + num_test :],
        "val": order[:num_val],
        "test": order[num_val : num_val + num_test],
    }

    def loader(name: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            [rows[i] for i in split[name]],
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    return DataloaderBundle(
        train=loader("train", config.shuffle_train),
        val=loader("val", False),
        test=loader("test", False),
    )
```

Sampling happens once here, so each cloud is one fixed subset of its points. A dataset that draws a new subset
each time a cloud is read is a cheap augmentation, and it is yours to write. So are rotations: the framework
has none, and a model trained on clouds in one orientation reads another orientation as another shape.

**Choose `N` with the loss in mind.** The Chamfer loss compares every reconstructed point with every target
point, so its cost grows with `N^2`: in a batch of 32 it was 2 ms at 64 points, 22 ms at 256 and 877 ms at 1,024,
which is more than the encoder and the decoder together (ADR 0044). A large cloud is better sampled to a few
hundred points than fed whole.

## The model

`configs/model/point_cloud_single_latent.yaml` is the complete file. What to set:

- **`num_points`** in the decoder, to the `N` of `data.sequence_length.point_cloud`. For another size, set both:

  ```bash
  data.sequence_length.point_cloud=256 model.modalities.point_cloud.decoder.kwargs.num_points=256
  ```

- **`in_dim`** in the encoder and **`out_dim`** in the decoder, for points with more than coordinates (above).
- **`pooling`** in the encoder: `max_mean` (the default), `max` or `mean`. The maximum alone was as good as both
  in the measurements, and the mean alone was clearly worse (ADR 0044).
- **`point_dims`** and **`head_hidden_dims`** of the encoder, **`code_dim`** and **`hidden_dims`** of the decoder
  and the latent `dim` are the size of the model. The shipped file has 181,152 parameters in the encoder and 81,155
  in the decoder. `normalization` can be `layer` in either; `batch` is refused in the encoder, since it would mix
  the points of different clouds.

## Train

`configs/experiment/point_cloud_vae.yaml` composes the model, data and training files, and it is where the
optimizer, the beta schedule and the weight come from. **Do not train it through `scripts/train.py`.** A YAML
file can name only `mse`, `l1`, `bce` and `smooth_l1`, and each of them compares a point with whichever point
has the same row index, which for a set is an unrelated one. On synthetic clouds the model trained that way ended
up farther from the clouds than one that ignores its input (Chamfer distance 0.81 on clouds in a random order and
0.39 with the points sorted, against 0.10), and on a padded cloud the loss is `NaN`. Build the trainer from the file
and switch its loss to the Chamfer distance, which is a plain Python function:

```python
from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import buildTrainerFromConfig
from global_vae.utils.point_cloud_metrics import computeChamferDistance

config = loadExperimentConfig(
    "experiment/point_cloud_vae",
    overrides=[
        "data.train_path=data/raw/shapes",
        "data.loader_factory=my_project.data:buildPointCloudDataloaders",
    ],
)
model = buildModelFromConfig(config.model)
bundle = buildPointCloudDataloaders(config.data)
trainer = buildTrainerFromConfig(model, config.training)
trainer.reconstruction_loss_fn = computeChamferDistance
trainer.reconstruction_reduction = "mean"  # the Chamfer distance is already a sum over points
trainer.fit(bundle.train, num_epochs=60, val_dataloader=bundle.val)
```

The same trainer is used by `evaluate`, so the validation loss, the best checkpoint and the report read the
Chamfer loss too.

## The loss

`computeChamferDistance(reconstruction, target)` sums, for each cloud, the squared distance from every
reconstructed point to the nearest target point and from every target point to the nearest reconstructed one,
and averages over the clouds. Rows of `NaN` in the target are left out. It is a sum over points, which is why the
reduction stays `mean`: `sum` would multiply it by `N * 3` again.

**The weight is the lever.** A weight of `w` on the reconstruction is a `beta` of `1 / w` on the KL term. The
experiment sets `100.0`. On synthetic clouds of 40 to 64 points ([ADR 0044](../adr/0044-point-cloud-modality.md)) the
Chamfer distance per point was 0.086 at weight 1, 0.071 at 10, 0.059 at 100 and 0.058 at 1,000, against 0.105 for
the best cloud that ignores its input, so 100 is the start of a plateau and not a tuned value (the choice of
`beta` is [roadmap](../roadmap.md) decision D-6). Try a few weights on your own data, and watch the number of
active latent units that `scripts/evaluate.py` reports.

## Reading a reconstruction

`computeCloudScores` scores a reconstruction on the real points only. The `mse`, `r2` and `pearson_r` of the report
of `evaluate` compare point `i` with point `i`, so they mean nothing for a set and are `NaN` on a padded cloud. Read
the total reconstruction loss of the report (it is the Chamfer loss) and these:

```python
import torch

from global_vae.utils.point_cloud_metrics import computeCloudScores

model.eval()
with torch.no_grad():
    batch = next(iter(bundle.val))
    reconstruction = model(batch, use_mean=True)["reconstructions"]["point_cloud"]

scores = computeCloudScores(reconstruction, batch["point_cloud"], threshold=0.1)
# {"chamfer": ..., "precision": ..., "recall": ..., "f_score": ..., "num_points": ...}
```

`chamfer` is per point, with both directions added. `precision` is the share of the reconstructed points within
`threshold` of a target point, `recall` is the share of the target points within `threshold` of a reconstructed
one, and `f_score` is their harmonic mean. The threshold is in the units of your normalized clouds (the farthest
point is at distance 1), so 0.1 is a tenth of the size of the cloud. `computeCloudScores` and
`computeChamferDistance` take any two numbers of points, so a reconstruction of 256 points can be scored against a
cloud of 64 by calling them. The trainer cannot train that, since it compares shapes first.

To draw clouds from the model, decode latent vectors from the prior:

```python
with torch.no_grad():
    z = torch.randn(8, model.latent_spaces["z_fused"].dim)
    clouds = model.decoders["point_cloud"](z)  # (8, N, 3)
```

The decoder writes exactly `num_points` points and no count, so a generated cloud has the size the model was built
for.

## What it does not do

- **A loss a YAML file can name.** There is no set loss in the loss table of `config/training.py`, so
  `scripts/train.py` on this experiment trains the wrong objective, and the Chamfer distance is set in Python. A
  `chamfer` entry in the table is a change to the core (ADR 0044).
- **Different numbers of points in the reconstruction and the target.** The loss function allows it and the
  trainer does not, so `num_points` equals `N` when training.
- **Set metrics in the report.** `evaluate` reports position-by-position metrics, so use `computeCloudScores`.
- **Rotations, sampling and normalization.** They are the loader's.
- **Large clouds and fine detail.** The Chamfer loss is quadratic in `N`, and the encoder pools the points to one
  vector with no neighbourhoods, so it reads global shape and not local structure.
- **A named dataset.** No point cloud collection has confirmed this modality is needed, so the proof is on
  synthetic clouds (a sphere-like ellipsoid, a torus, a cylinder and a box).
