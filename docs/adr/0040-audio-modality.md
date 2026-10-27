# 0040: The audio modality

**Status:** accepted (roadmap item P3-8, second modality; it adds two classes through the registries
and changes no core module)
**Date:** 2026-10-26

## Context

Roadmap P3-8 lists audio among the further modalities, each through the new-modality checklist of spec
§10 and only once a dataset is named. No corpus has been named, so nothing here is tuned to one: the model
is generic, the batch contract is stated, and the measurements below are on synthetic tones. The tabular
modality ([ADR 0039](0039-tabular-modality.md)) was the first to follow the checklist after the core was
written. Audio is the second, and a harder case for the claim of spec §1 and §6, for a different reason: it
is long (16,000 values per second at 16 kHz), so a stack that reads it has to downsample hard, and it is
not an arbitrary length (a clip of 16,000 or 22,050 samples).

Spec §6 says that other signal sources "should slot into the same signal encoder/decoder family later,
with only preprocessing differing, not the architecture", and a waveform is a 1D series. So the first
question is whether the 1D classes already are the audio classes. They take a waveform (a series with
`in_channels` for stereo, any length through adaptive pooling), so a model file could name
`1d_cnn_resnet_encoder_v1` and `1d_cnn_resnet_decoder_v1` and train. Three things stop them from being a
good answer, all read in the code and none a defect of those classes for the data they were written for:

- **The decoder must reach `output_length` exactly.** `1d_cnn_resnet_decoder_v1` verifies at construction
  that its stack of strides produces `output_length` and raises `ValueError` otherwise, which is right for
  a 256-point spectrum and a chore for 16,000 or 22,050 samples, where the output paddings have to be
  engineered by hand.
- **No bounded output.** The last layer has no output activation and there is no parameter for one, and a
  waveform lives in `[-1, 1]`.
- **One dilation per stage.** `dilations` is a single value per stage, so the standard waveform design,
  residual units with dilations such as 1, 3 and 9 inside each stage, cannot be written.

A fourth point decides the shape of the answer: audio is often a spectrogram, and a spectrogram is an
image. The 2D classes take one already.

## Decision

**1. Two registered classes, one module of building blocks and one of activations.** Nothing else changes:

| Registry name | Class | Reads or writes |
|---|---|---|
| `audio_waveform_encoder_v1` | `AudioWaveformEncoder` | a waveform, `(batch, samples)` or `(batch, channels, samples)` |
| `audio_waveform_decoder_v1` | `AudioWaveformDecoder` | a waveform of a fixed `output_length` |

`utils/waveform_blocks.py` holds what they share (the stage builders and the length helpers) and
`utils/activations.py` the activation names. The latter used to live in `utils/mlp_blocks.py`, added for
ADR 0039, and moved because two backbones now need it (`mlp_blocks` imports it, the behavior of the
tabular classes is unchanged and `elu` is a new name). The classes are registered by the imports of
`encoders/__init__.py` and `decoders/__init__.py`. `models/`, `latent/`, `losses/`, `training/`,
`evaluation/`, `config/` and `data/` are untouched, and a test reads their source to check that none of
them names the new classes or the `"audio"` batch key. The existing 1D classes are not changed either.

**2. A clip is a float waveform made by the caller.** The batch key `audio` holds `(batch, samples)` for
mono or `(batch, channels, samples)`, within `[-1, 1]`, at one sample rate and with one number of samples
for the whole dataset. Decoding files, resampling (with a proper resampler: the `resample` transform
interpolates with no anti-aliasing filter, which is not a sample rate conversion), normalizing the level
and cutting or padding to a length are the `loader_factory`'s (spec §6.2), and `docs/how-to/train-on-audio.md`
has one. The sample rate is not in the contract, since the model knows a number of samples, and the number
of samples reuses the `sequence_length` of the data config, which `configs/data/audio.yaml` sets to 2048.
The data file ships an empty transform list on purpose: the decoder ends in a `tanh`, which a
`standardize` would take the data out of.

**3. The waveform, not the spectrogram, is the modality, and the spectrogram needs nothing.** A
log-magnitude spectrogram, `(frequency bins, frames)`, is an image to `2d_cnn_encoder_v1` and
`2d_cnn_decoder_v1`, and a test builds that model from `configs/model/image_single_latent.yaml` with its
modality renamed and runs a forward and backward pass on a real STFT. The two answer different questions: a waveform model
reconstructs the sound, phase included, and a spectrogram model reconstructs what a clip contains in time
and frequency and drops the phase, so it cannot be played back without a vocoder. Both are available, and
this ADR only adds the one that was not.

**4. The length rules are exact, and the decoder crops.** A downsampling convolution of stride `s` has
kernel `2s + 1` and padding `s`, which maps a length `L` to `ceil(L / s)`. So no input collapses a feature
map, `minimal_input_length` is `1`, and the encoder accepts a clip of any length from one sample up (with a
reduction over time, the mean by default or the max with `global_pool: max`, which is what makes the output
shape independent of the length). An upsampling stage is nearest-neighbour upsampling followed by a
convolution and multiplies the length by exactly `s`, with no checkerboard. The decoder projects the latent
vector to a seed of `ceil(output_length / prod(strides))` frames, upsamples, and crops the surplus, which
is fewer than `prod(strides)` samples. So any `output_length` is reached whatever the strides, and the crop
is the only adjustment, as in the transformer decoder of ADR 0037: it never resizes or interpolates, and a
test checks lengths such as 1, 7, 777, 1,000 and 16,000.

**5. Each stage is dilated residual units and one strided convolution.** A stage runs one
`utils.conv_blocks.Residual1DBlock` per value of `dilations` (default `[1, 3, 9]` in the class, `[1, 3]` in
the shipped file, each of two convolutions with "same" padding, so the length is kept), then the strided
convolution that moves to the next width. A first convolution of `stem_kernel_size` lifts the channels of
the waveform, and a last one of `output_kernel_size` brings the decoder back to `out_channels`. Defaults:
four stages of stride 4 (256 samples per frame), widths `16, 32, 64, 128` in the encoder and `128, 64, 32,
16` in the decoder, `elu` activations.

**6. Activation, normalization and the output bound are names.** `activation` is a name
(`elu`, `relu`, `gelu`, `silu`, `leaky_relu`, `tanh`), as in the tabular classes, so that YAML can write it.
`normalization` is `group` (default, one-group `GroupNorm`, which normalizes each sample over its channels
and time and so depends neither on the batch nor on the length), `batch` or `null`. The decoder's
`output_activation` is `tanh` by default and `null` for data that is not within `[-1, 1]`. Every argument
has a default, and an argument out of range is a `ValueError` that names it.

**7. One model file, one data file and one experiment.** `configs/model/audio_waveform_single_latent.yaml`
(2048 samples, four stages of stride 4, widths of 8 to 64, a latent of 16, no fusion),
`configs/data/audio.yaml` and `configs/experiment/audio_vae.yaml`, which trains through `scripts/train.py`
like the signal, image and tabular experiments and sets `reconstruction_reduction: sum` and
`reconstruction_weight: 0.5`, with the numbers below. The clip length is written twice, in the data file
and in the decoder's `output_length`, as for a signal.

## Measurements

The loss scale of the experiment, on synthetic tones: 480 training clips of 2,048 samples at 16 kHz, each a
note of three harmonics (a fundamental drawn log-uniformly between 150 and 500 Hz) with an exponential decay
(3 to 12 per second), a little noise, peak-normalized to 0.9. The model is the shipped one (a latent of 16),
with the warm-up of `training/default.yaml` (1,000 steps, and 40 epochs are 600, so `beta` had reached 0.6 when the
40-epoch rows stopped) and one seed. R² is against the mean clip of the 160 validation
clips, and the cosine is the mean over clips of the cosine similarity between a clip and its
reconstruction.

| Reduction and weight | Epochs | R² | Mean cosine | Active latent units |
|---|---|---|---|---|
| `mean`, 1.0 | 40 | -0.01 | 0.17 | 0 of 16 |
| `sum`, 0.5 | 40 | 0.31 | 0.56 | 16 of 16 |
| `sum`, 0.5 | 150 | 0.65 | 0.82 | 16 of 16 |

The pattern is that of the signal, image and tabular models ([ADR 0029](0029-loss-scale-convention.md),
[ADR 0037](0037-transformer-and-vit-backbones.md), [ADR 0039](0039-tabular-modality.md)): under `mean` and
`beta = 1` the KL weighs `N / 2` = 1,024 times more than the ELBO does for a clip of 2,048 samples, and the
model collapses, while `sum` at 0.5 trains. Unlike the tabular model, this one keeps improving with training
(R² of 0.31 at 40 epochs and 0.65 at 150, once the warm-up is done), so the 40-epoch rows understate it. A
waveform is a hard target for a squared error, because a
phase error of a fraction of a period costs as much as a wrong note, so the R² is modest and says that the
model trains, not that it is tuned. These are numbers from synthetic data, one seed and no tuning, and the
choice of beta stays roadmap decision D-6.

## Consequences

- **The registry design held a third time, and the existing classes were left alone.** Two classes, two
  utility modules, three config files and the tests are all that was added, and no core edit was needed or
  made. This is the first modality where the question was whether an existing family already did the job,
  and the answer was no for three specific reasons (above) that are the new pair's reasons to exist, not
  preferences.
- **Finding, not fixed: one latent vector per clip.** The encoder averages over time, so the latent holds
  what a clip contains and not when, and a decoder asked for the exact waveform from 16 numbers does best
  on what a few numbers describe (a note and its decay). Audio is where a latent per frame, a sequence of
  latents, would pay, and that is the single-tensor decoder item of the roadmap audit P4-0, a change to the
  core.
- **Finding, not fixed: no spectral loss.** The reconstruction loss is a squared error on samples, which
  cares about phase and not about what the ear hears, and a distance between STFT magnitudes at several
  resolutions is the usual remedy. From Python, `Trainer(reconstruction_loss_fn=...)` takes any function
  `(reconstruction, target) -> scalar`, so one works in code. From YAML the loss names are the fixed table
  of `config/training.py` (`mse`, `l1`, `bce`, `smooth_l1`), so naming a spectral loss there is a change
  to a core module, which is flagged and not made.
- **Finding, not fixed: the sample rate has no place in the data contract.** `DataConfig` has no field for
  it, since for a signal it never mattered, so it lives in the loader and in the comment of
  `configs/data/audio.yaml`. A model trained at 16 kHz and a clip at 44.1 kHz are a mismatch nothing
  checks. A field would be a change to `config/data.py`.
- **No companding or pre-emphasis.** A mu-law transform would be an invertible, registered `data.transforms`
  entry (the sanctioned extension of `docs/how-to/add-a-strategy.md`, not a core change) and is left until a
  dataset shows that it is wanted. The loader can do it today.
- **Cost.** The class defaults (widths 16 to 128, dilations 1, 3 and 9) are about 640,000 parameters in the
  encoder and 660,000 in the decoder, and the shipped file, which a CPU run can afford, is much smaller. A
  real corpus wants a longer clip, so more samples, so larger strides or more stages: the number of frames
  is `ceil(samples / prod(strides))` and the decoder's seed layer has `hidden_channels[0]` times that many
  outputs.
- **No dataset was used.** The roadmap asks for a named corpus before a modality, and none was named, so the
  proof is synthetic tones and the dummy loader of the tests. There is no reference loader for a real
  corpus, and no audio decoding library is a dependency: `loadMyClips` in the how-to is the caller's. The
  other modalities of P3-8 (text, time series, graphs, point clouds) stay open.
