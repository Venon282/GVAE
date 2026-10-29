# Train on time series

A time series is the time series modality ([ADR 0042](../adr/0042-time-series-modality.md)): one encoder
(`time_series_encoder_v1`) that reads a window of one or several channels, with gaps and with padding, and the
existing `1d_cnn_decoder_v1` that rebuilds it. If your series are complete and all of one length, the plain 1D
classes ([Train on a signal](train.md)) do the same job, and this modality is for the case where some values
were not observed. Reading files, cutting windows and standardizing all happen in your `loader_factory`, as for
any other data (spec §6.2).

## What a batch is

A batch is a dict with one key, `time_series`, holding a **float tensor `(batch, channels, length)`**, or
`(batch, length)` for a single channel.

**A `NaN` is a value that was not observed.** It can be a gap in the middle of a recording, a value that was
dropped, or the end of a series that is shorter than the window, padded with `NaN` up to `length`. The
encoder reads it as a missing value and not as a zero, so a series padded with `NaN` gives the same latent
vector as the same series unpadded, and a loss written for the purpose (below) counts only what was observed.
No second tensor is needed, because the batch tensor is the one thing that both the encoder and the loss read.

- **`channels`** is written in the encoder (`in_channels`) and in the decoder (`out_channels`), and it is the
  second axis of the batch. Nothing fills it in for you. Several channels are one multivariate series: the
  encoder reads them together, so give it the channels that were measured together.
- **One length**, equal to `sequence_length.time_series` in the data file and to `output_length` in the
  decoder. Longer recordings are cut into windows and shorter windows padded with `NaN` by the loader, because
  the decoder reconstructs one fixed length and never resizes it. The encoder takes any length.
- **A regular grid.** The position of a value is its step index, so the observations of a window must be
  evenly spaced. A series sampled at irregular times is resampled by the loader, with the holes left `NaN`.

## A `loader_factory`

`configs/data/time_series.yaml` describes the contract and leaves `loader_factory` and `train_path` for you to
fill in. This is one that follows it. `loadMyRecordings` is yours: it reads your files and returns a list of
recordings, each a float tensor `(channels, steps)` of any number of steps, `NaN` where the sensor gave
nothing.

```python
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from global_vae.config.data import DataConfig, DataloaderBundle
from global_vae.data.transforms.standardize import StandardizeTransform


def cutWindows(recording: torch.Tensor, length: int) -> list[torch.Tensor]:
    """Windows of exactly `length` steps; the last, shorter one is padded with NaN."""
    windows = []
    for start in range(0, recording.shape[-1], length):
        window = recording[:, start : start + length]
        windows.append(functional.pad(window, (0, length - window.shape[-1]), value=float("nan")))
    return windows


def channelStandardization(windows: list[torch.Tensor]) -> StandardizeTransform:
    """One mean and one scale per channel, from the observed values of the training windows."""
    channels = torch.stack(windows).transpose(0, 1).flatten(start_dim=1)  # (channels, values)
    mean = torch.nanmean(channels, dim=1, keepdim=True)
    scale = torch.nanmean((channels - mean) ** 2, dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    return StandardizeTransform(mean=mean, std=scale)  # (channels, 1) broadcasts over a window


def buildTimeSeriesDataloaders(config: DataConfig) -> DataloaderBundle:
    recordings = loadMyRecordings(config.train_path)  # a list of (channels, steps) tensors
    length = (config.sequence_length or {})["time_series"]
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(recordings), generator=generator).tolist()
    num_val = int(len(recordings) * config.val_split)
    num_test = int(len(recordings) * config.test_split)
    split = {
        "train": order[num_val + num_test :],
        "val": order[:num_val],
        "test": order[num_val : num_val + num_test],
    }
    # Split by recording, not by window: two windows of one recording must not be in two splits.
    windows = {
        name: [window for i in indices for window in cutWindows(recordings[i], length)]
        for name, indices in split.items()
    }
    standardize = channelStandardization(windows["train"])  # from the training split only

    def loader(name: str, shuffle: bool) -> DataLoader:
        series = standardize(torch.stack(windows[name]))  # a NaN stays a NaN
        return DataLoader(
            list(series),
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=lambda rows: {"time_series": torch.stack(rows)},
        )

    return DataloaderBundle(
        train=loader("train", config.shuffle_train),
        val=loader("val", False),
        test=loader("test", False),
    )
```

A YAML file can only write one number for `standardize`, which is why the per-channel statistics are built
here, from tensors of shape `(channels, 1)`. The data file's `transforms` entry for `time_series` stays empty.
The statistics come from the observed values of the training split only, so that nothing about the other
splits leaks into the model, and the transform leaves a `NaN` a `NaN` and inverts on the values that were
observed.

## The model

`configs/model/time_series_single_latent.yaml` is the complete file. What to set:

- **`in_channels`** in the encoder and **`out_channels`** in the decoder, to the number of channels. From the
  command line: `model.modalities.time_series.encoder.kwargs.in_channels=5
  model.modalities.time_series.decoder.kwargs.out_channels=5`. With one channel, the decoder returns
  `(batch, length)` with no channel axis, so the batch is `(batch, length)` as well.
- **`output_length`** in the decoder, to the number of steps of a window, and the same number as
  `data.sequence_length.time_series`. This decoder checks the exact length and never resizes, so the rest of
  its configuration has to land on it. In the shipped file, `seed_length: 12` and a kernel of 5 with padding 2
  double three times, 12, 24, 48, 96. For another length, ask the class before building the model:

  ```python
  from global_vae.decoders.one_d_cnn_decoder import OneDCnnDecoder

  OneDCnnDecoder.computeOutputLength(
      seed_length=12, hidden_channels=(128, 64, 32), kernel_sizes=5, paddings=2
  )  # 96
  ```

  and set it with `data.sequence_length.time_series=48
  model.modalities.time_series.decoder.kwargs.output_length=48
  model.modalities.time_series.decoder.kwargs.seed_length=6`.
- **`patch_size`** in the encoder, the number of steps per token. Attention is quadratic in the number of
  tokens, so a long window wants a larger patch and not more layers. A patch that holds no observed value in
  any channel is skipped.
- **The stack**: `d_model`, `num_heads` (dividing `d_model`), `num_layers`. `pooling` is `mean` (the mean of
  the patches that hold data, the default) or `cls` (a learned token). `positional_encoding` is `sinusoidal`
  (any length) or `learned` (a vector per patch, which needs `max_length`).

The decoder is the 1D CNN one because it measured much better than the 1D transformer decoder on these
series (0.80 of the variance against 0.53, [ADR 0042](../adr/0042-time-series-modality.md)), and no decoder
class was added.

## Train

If your windows are complete, the experiment trains through `scripts/train.py` like the others:

```bash
PYTHONPATH=. python scripts/train.py --config-name experiment/time_series_vae \
    data.train_path=data/raw/sensors \
    data.loader_factory=my_project.data:buildTimeSeriesDataloaders
```

If some values are `NaN`, that route cannot be used. The loss names of a YAML file (`mse`, `l1`, `bce`,
`smooth_l1`) know no mask, and the `mse` of the experiment is `NaN` as soon as one value of a batch is. Train
through Python with the masked loss, which is a plain function and needs no change to the framework:

```python
from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.training.trainer import Trainer
from global_vae.utils.series_blocks import computeMaskedSquaredError

config = loadExperimentConfig(
    "experiment/time_series_vae",
    overrides=[
        "data.train_path=data/raw/sensors",
        "data.loader_factory=my_project.data:buildTimeSeriesDataloaders",
    ],
)
model = buildModelFromConfig(config.model)
bundle = buildTimeSeriesDataloaders(config.data)

trainer = Trainer(
    model,
    optimizer_kwargs={"lr": 2e-3},
    reconstruction_loss_fn=computeMaskedSquaredError,
    reconstruction_weights=50.0,  # the reduction stays "mean": the function sums over values itself
)
trainer.fit(bundle.train, num_epochs=60, val_dataloader=bundle.val)
```

## The loss

`computeMaskedSquaredError` sums the squared error over the observed values of a series and averages over the
batch. A `NaN` in the target adds nothing, and the reconstruction there receives no gradient. On a series with
no gap it is the sum of squared errors, which is what `reconstruction_loss: mse` with
`reconstruction_reduction: sum` computes, so the shipped experiment and the Python route agree where both
apply. The decoder still reconstructs every step of every channel: a gap is filled in, and only the loss
looks away.

**The weight is the lever.** A weight of `w` on the reconstruction is a `beta` of `1 / w` on the KL term, and
a weight of `0.5` is a Gaussian of variance one, which is right only when the noise of your standardized
series is of that size. The right weight scales as `0.5 / sigma^2` for a noise of `sigma`. On synthetic series
of unit scale and a noise of 0.05 ([ADR 0042](../adr/0042-time-series-modality.md)), the share of the variance
recovered was 0.50 at weight 0.5, 0.79 at 5, 0.80 at 50 and 0.80 at 500, so 50 is a middle setting and not a
tuned value (the choice of `beta` is [roadmap](../roadmap.md) decision D-6). Try a few weights on your own
data, and watch the number of active latent units that `scripts/evaluate.py` reports.

## Reading a reconstruction

To fill the gaps of a window, keep what was observed and take the reconstruction elsewhere:

```python
model.eval()
with torch.no_grad():
    batch = next(iter(bundle.val))
    reconstruction = model(batch, use_mean=True)["reconstructions"]["time_series"]
    series = batch["time_series"]
    filled = torch.where(torch.isnan(series), reconstruction, series)
```

**The metrics of `evaluate` know no mask either.** On series with gaps, `scripts/evaluate.py` and `evaluate`
return `NaN` for the `mse`, `rmse`, `mae`, `r2` and `pearson_r` of the modality, while the total loss
(with the masked loss passed as `reconstruction_loss_fn`) and the figures work. Score the gaps yourself, by
hiding values of a complete window that the model never sees and comparing the reconstruction there:

```python
def scoreHiddenValues(model, series: torch.Tensor, hidden: torch.Tensor) -> float:
    """Share of the variance recovered on the values that `hidden` (a boolean mask) hides.

    `series` is a complete batch `(batch, channels, length)`; the hidden values are replaced by NaN
    before the model sees it, so the score is on values it never saw.
    """
    model.eval()
    with torch.no_grad():
        shown = series.masked_fill(hidden, float("nan"))
        reconstruction = model({"time_series": shown}, use_mean=True)["reconstructions"][
            "time_series"
        ]
    truth, guess = series[hidden], reconstruction[hidden]
    return float(1 - (guess - truth).pow(2).sum() / (truth - truth.mean()).pow(2).sum())
```

Hide a stretch the way your gaps look (a run of steps in the middle, not scattered values, if that is what
your sensor does), because a model fills a short hole far more easily than a long one.

## What it does not do

- **A mask in the core.** The `NaN` convention and the Python loss need no change to the framework, and they
  leave two things only a core change could give: a masked loss that `scripts/train.py` can name, and
  metrics over the observed values in `evaluate`. Both are flagged in ADR 0042 and not made.
- **One latent vector per window.** The encoder pools over the patches, so the latent holds what the window
  contains and not when, and the decoder writes the whole window from it.
- **Irregular sampling, per-window normalization, forecasting.** The grid is regular, a window's own scale
  and offset are encoded in its latent vector (standardize the channels in the loader), and the model
  reconstructs a window and does not predict the next one.
- **A named dataset.** No recording has confirmed this modality is needed, so the proof is on synthetic series.
