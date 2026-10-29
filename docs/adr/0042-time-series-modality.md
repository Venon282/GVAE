# 0042: The time series modality

**Status:** accepted (roadmap item P3-8, fourth modality after tabular, audio and text; it adds one class
through the registries, reuses an existing decoder, and changes no core module)
**Date:** 2026-10-29

## Context

Roadmap P3-8 lists time series among the further modalities, each through the new-modality checklist of spec
§10 and only once a dataset is named. No recording has been named, so nothing here is tuned to one: the model
is generic, the batch contract is stated, and the measurements below are on synthetic series. The tabular
([ADR 0039](0039-tabular-modality.md)), audio ([ADR 0040](0040-audio-modality.md)) and text
([ADR 0041](0041-text-modality.md)) modalities followed the checklist before it.

A time series is the case where the registered 1D classes look like the whole answer, so the first thing
checked was what they already do. `1d_transformer_encoder_v1` and `1d_cnn_encoder_v1` take `in_channels`,
and `1d_cnn_decoder_v1` and `1d_transformer_decoder_v1` take `out_channels`, so a multivariate series is
covered. The transformer pair takes any length (the encoder pads to whole patches, the decoder crops). What
neither covers is what real recordings do and a clean signal does not: **values that were not observed**. A
sensor drops out (a gap in the middle) and a window cut at the end of a recording is shorter than the others
(padding at the end). To the 1D encoders a missing value is a number, usually a zero, which is also a real
value, so a series padded with zeros has another latent vector than the same series unpadded, and a loss over
the padded batch counts values that were never measured. This is the substance of roadmap item P3-7
(variable-length and masked signals).

Two questions follow. **Where does a mask travel?** The loss of the framework compares a decoder's
reconstruction with the batch tensor of the same name and nothing else
(`computeTotalReconstructionLoss`), so a second mask tensor would never reach it. **Is a new decoder needed?**
It was decided by measurement, below.

## Decision

**1. One registered class, one utility module, and an existing decoder.** The checklist of spec §10 asks for
an encoder and a decoder. Here the decoder already existed and measured better than the alternative, so it is
reused and no decoder class is added:

| Registry name | Class | Reads or writes |
|---|---|---|
| `time_series_encoder_v1` | `TimeSeriesEncoder` | a series `(batch, channels, length)`, or `(batch, length)` for one channel, `NaN` where unobserved |
| `1d_cnn_decoder_v1` (existing) | `OneDCnnDecoder` | a series of a fixed `output_length`, `out_channels` channels |

`utils/series_blocks.py` holds `splitObservations`, `computeEmptyPatchMask`, `ObservedPatchEmbed` and
`computeMaskedSquaredError`. Positions, the transformer stack and the pooling head are the unchanged pieces of
`utils/transformer_blocks.py` (ADR 0037), and the masked mean is the one of `utils/text_blocks.py` (ADR
0041). The class is registered by the import of `encoders/__init__.py`. `models/`, `latent/`, `losses/`,
`training/`, `evaluation/`, `config/` and `data/` are untouched, and a test reads their source to check that
none of them names the new class or the `"time_series"` batch key.

**2. A `NaN` is a value that was not observed.** The batch key `time_series` holds a float tensor, `(batch,
channels, length)` or `(batch, length)` for one channel, with **one length** for the dataset
(`sequence_length.time_series` in the data file, `output_length` in the decoder) and **one number of channels**
(`in_channels` of the encoder, `out_channels` of the decoder; no field of `DataConfig` carries it). A `NaN`
marks a value that was not observed, wherever it is: a gap in the middle, the padding of a short series up to
the length of the batch, or a dropped value. The convention needs no second tensor, because the batch tensor
is the one thing both the encoder and the loss read:

| | A value | A `NaN` |
|---|---|---|
| Encoder | read as it is, with its mask bit set | read as zero, with its mask bit unset, so it is not mistaken for a real zero |
| Loss (`computeMaskedSquaredError`) | counts the squared error | adds nothing, and the reconstruction there gets no gradient |
| Decoder | reconstructs every step of every channel | still reconstructs it: the gap is filled in |

Cutting windows, padding a batch to one length, putting the observations on a regular time grid and the
per-channel standardization are the `loader_factory`'s (spec §6.2), and
`docs/how-to/train-on-time-series.md` has one.

**3. The encoder reads values and mask together, and skips the empty patches.** The series is cut into
non-overlapping patches of `patch_size` steps (the end is padded with unobserved steps up to a whole patch).
`ObservedPatchEmbed` stacks the values and the observed mask as `2 * in_channels` channels and projects each
patch to a `d_model` token, so the projection can tell a zero from a gap. A patch with no observed value in
any channel is **empty**: it is skipped as an attention key and left out of the mean. Positions are
sinusoidal (no length limit, the default) or learned (up to `max_length`). Pooling is the mean of the tokens
that hold data (default) or a learned `[CLS]` token. So **a series padded with `NaN` to a longer length has
the same `(mu, logvar)`** (a unit test to `1e-5` for both poolings, and a test through the whole model that
also pins the reconstruction), a series that is unobserved everywhere keeps its first patch so attention
never has every key masked (which would give `NaN`), and a series with no gap is a patch transformer like
`1d_transformer_encoder_v1` whose mask channels are all ones. `minimal_input_length` is `1`.

**4. The masked loss is a Python function, and the YAML route is for complete series.** The loss names of a
training config are the fixed table of `config/training.py` (`mse`, `l1`, `bce`, `smooth_l1`), and none knows
a mask: **the `mse` of a YAML file is `NaN` as soon as one value of the batch is a `NaN`** (a test pins it).
`computeMaskedSquaredError` is a plain function of `(reconstruction, target)`, so
`Trainer(model, reconstruction_loss_fn=computeMaskedSquaredError, reconstruction_weights=...)` takes it
without a core change. It sums the squared error over the observed values of a series and averages over the
batch, so on a series with no gap it equals the `sum` reduction of `mse` (a test pins that), and with weight
`0.5` it is the reconstruction term of a Gaussian of unit variance. It zeroes the target before subtracting,
because `NaN * 0` is `NaN`. It is used with the default `mean` reduction, which leaves the sum alone; the
`sum` reduction would multiply it by the number of values again. The shipped experiment trains **complete
series** (`mse`, `sum`, weight 50) through `scripts/train.py`; series with gaps or padding train through
Python, and the how-to shows the call.

**5. The decoder is `1d_cnn_decoder_v1`, whose length must land exactly.** This class checks the exact
length and never resizes, so the configuration has to produce `output_length`: `seed_length: 12`, a kernel
of 5 with padding 2 and the three default doubling stages give 12, 24, 48, 96 (`OneDCnnDecoder.computeOutputLength`
explores other configurations before building). With `out_channels: 1` it returns `(batch, length)`, with
no channel axis, which is the shape of a one-channel batch of that layout.

**6. One model file, one data file and one experiment.** `configs/model/time_series_single_latent.yaml`
(3 channels of 96 steps, `patch_size` 8, `d_model` 64, three layers, a latent of 16, no fusion),
`configs/data/time_series.yaml` and `configs/experiment/time_series_vae.yaml`, which trains through
`scripts/train.py` like the others.

## Measurements

Synthetic series (`tests/integration/_time_series_corpus.py`), not a recording: three channels of 96 steps
that share one seasonal period (10 to 30 steps) and differ by amplitude, phase, slope and offset, with a
noise of 0.05. 31% of the series are cut to 48 to 95 steps (the end is `NaN`), half have one hole of 4 to 15
steps, and 5% of the other values are dropped at random: 17% of all values are unobserved. 3,000 training
series and 600 held-out ones, the shipped model (a latent of 16; 155,296 parameters in the encoder and 78,083
in the decoder), AdamW at 2e-3, batches of 64, 60 epochs (2,820 steps), constant `beta` of 1 with no warm-up,
and the masked loss. Every number is the share of the variance of the **complete** series recovered by the
reconstruction from the mean of the posterior, on the **observed** values, on the **gaps** inside the
recording (values the model never saw), and on the **tail** after a short recording ends (values that no
training target holds either, so it scores how far the latent vector carries the dynamics, not a goal).
Seed 0 unless stated.

**The decoder** (new encoder, weight 50):

| Decoder | Observed | Gaps | Tail |
|---|---|---|---|
| `1d_transformer_decoder_v1`, default (patch 8, `d_model` 64, three layers) | 0.53 | 0.42 | 0.33 |
| `1d_transformer_decoder_v1`, patch 4, `d_model` 128, four layers (three times slower) | 0.59 | 0.48 | 0.33 |
| **`1d_cnn_decoder_v1`** | **0.80** | **0.71** | **0.49** |

**The encoder against the one that exists**, with the CNN decoder, weight 50, and three seeds (data and
initialization both change). The baseline is `1d_cnn_encoder_v1` fed the same batch with `NaN` replaced by
zero, trained with the same masked loss:

| Seed | New encoder (observed / gaps / tail) | `1d_cnn_encoder_v1`, zero-filled |
|---|---|---|
| 0 | 0.80 / 0.71 / 0.49 | 0.81 / 0.73 / 0.35 |
| 1 | 0.84 / 0.77 / 0.59 | 0.77 / 0.71 / 0.30 |
| 2 | 0.83 / 0.75 / 0.54 | 0.83 / 0.78 / 0.37 |
| **Mean** | **0.82 / 0.74 / 0.54** | **0.81 / 0.74 / 0.34** |

**The weight** (new encoder, CNN decoder):

| Weight | Observed | Gaps | Tail |
|---|---|---|---|
| 0.5 | 0.50 | 0.41 | 0.35 |
| 5 | 0.79 | 0.70 | 0.50 |
| **50 (the experiment)** | **0.80** | **0.71** | **0.49** |
| 500 | 0.80 | 0.72 | 0.57 |

What it says:

- **The decoder mattered more than the encoder.** The CNN decoder recovers 0.80 of the variance where the
  transformer decoder recovers 0.53, and a transformer decoder twice as wide and a layer deeper is still at
  0.59 and slower. That is why no decoder class is added. Why a decoder that writes every token from the same
  vector does worse here was not isolated. The series have a smooth trend and a period, which an upsampling
  stack with local kernels suits.
- **The new encoder is not better than a zero-filled one on what was observed or on the gaps.** Across three
  seeds the difference is +0.016 on observed values and -0.002 on gaps, inside the spread between seeds (up to
  0.07). With the transformer decoder it was ahead on every slice at weights 5 and 50 (0.53 against 0.44 on
  observed values at 50), so the lead came from a weak decoder and went away with a good one. The masked
  loss does the heavy lifting here: the decoder learns to fill a gap from the rest of the series whichever
  encoder it has.
- **It is better on the tail, by about 0.20 in every seed** (0.54 against 0.34). A zero-filled encoder reads
  the padding as a flat segment at zero, and the new one cannot see it. This is the structural property that
  is pinned by the tests (the latent vector of a padded series is the same), and the one to choose it for. It
  costs about 1.3 times the training time of the CNN encoder in these runs. That the tail score follows from
  the latent vector being unchanged by the padding was not isolated.
- **The weight is the lever, and D-6 again.** A weight `w` on the summed reconstruction is a `beta` of `1 / w`
  on the KL. At 0.5 (a Gaussian of unit variance, on data whose noise is 0.05) the model recovers 0.50, and
  from 5 up it is flat at about 0.80. With the transformer decoder the weight mattered more (0.35 at 0.5,
  0.45 at 5, 0.53 at 50). The right weight scales as `0.5 / sigma^2` for a noise of `sigma` after
  standardization, so 50 is a middle setting for series of unit scale and noise 0.05, not a tuned value. The
  choice of `beta` stays roadmap decision D-6.
- **The tests use a smaller run.** 1,024 series and 30 epochs at a learning rate of 3e-3 (480 steps) recover
  0.40 to 0.42 on observed values and 0.34 to 0.35 on gaps over three seeds, against 0.00 for the best answer
  that ignores the series (the mean series of training), and the tests assert a margin of 0.2 over it.

These are one synthetic family, three seeds at most and no tuning: they show that the model trains, how it
compares with a plain encoder and how the weight behaves, not what a recording would give.

## Consequences

- **The registry design held a fifth time, with one class instead of two.** One encoder, one utility module,
  three config files and the tests are all that was added, and no core edit was needed or made. The checklist
  step "subclass Decoder" was met by an existing decoder because it measured better, which is a deviation
  from the letter of §10 and not from its point (the core needed no change).
- **Finding, not fixed: no mask in the loss table.** A YAML file cannot name `computeMaskedSquaredError`, and
  its `mse` is `NaN` on a gap, so `scripts/train.py` trains complete series only, and series with gaps need a
  Python script that builds a `Trainer`. A masked loss in the table of `config/training.py`, or a way to name
  a loss by import path, is a core change, flagged and not made.
- **Finding, not fixed: the evaluation metrics know no mask.** On series with gaps, `evaluate` returns `NaN`
  for `mse`, `rmse`, `mae`, `r2` and `pearson_r` of the modality (a test pins it), while its total loss (with
  the masked loss passed in) and the figure export work. Metrics over the observed values are a change to
  `evaluation/`, flagged and not made; the how-to computes them in a few lines.
- **Finding, not fixed: one latent vector per window.** The encoder pools over the patches, so the latent
  holds what the window contains and not when, and the decoder writes the whole window from one vector. A
  sequence of latents is the single-tensor decoder item of the roadmap audit P4-0.
- **Finding, not fixed: no per-window normalization, and per-channel statistics need code.** The scale and
  offset of a window are encoded in the latent vector and spend its capacity, where a reversible
  instance normalization (a per-window mean and scale removed and restored) would not. A transform like that
  does not exist here. And `standardize` takes a float or a tensor but a YAML list fails, so the per-channel
  statistics of a multichannel series are built in the loader, the same finding as for tabular data in
  ADR 0039. `configs/data/time_series.yaml` says so, and a test shows a standardization built in code leaves
  a `NaN` a `NaN` and inverts on the observed values.
- **Not modeled: irregular sampling, and forecasting.** The position of a value is its step index, so the
  observations must be on a regular grid, and a series sampled at irregular times is resampled, or has its
  holes marked `NaN`, in the loader. The model reconstructs a window; it does not forecast one, though the
  tail scores above hint at how far a latent vector carries a dynamic.
- **Roadmap P3-7 is partly met on the model side, and the requirement is still unconfirmed.** The mechanism
  now exists (a `NaN`-aware encoder and a masked loss, both registered or passed in with no core change), but
  P3-7 is "a confirmed requirement" item and no named dataset has confirmed one. The decision to build a
  mask into the core (a mask key in the batch, a masked loss in the table, masked metrics) is still open.
- **The one-channel layout has no channel axis on the output.** `1d_cnn_decoder_v1` with `out_channels: 1`
  returns `(batch, length)`, so a one-channel model takes and returns `(batch, length)`; a batch of
  `(batch, 1, length)` would be refused by the loss with both shapes (a test covers the matching layout).
- **No dataset, no loader.** Windowing, padding to one length, the time grid and the standardization are the
  loader's, no named recording exists, and the proof is the synthetic series and the dummy loader of the
  tests. The remaining modalities of P3-8 (graphs, point clouds) stay open.
- **Cost.** The shipped file is 155,296 parameters in the encoder and 78,083 in the decoder. Attention is
  quadratic in the number of patches, so a long window wants a larger `patch_size` and not more layers.
