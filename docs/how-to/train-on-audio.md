# Train on audio

Audio is the audio modality ([ADR 0040](../adr/0040-audio-modality.md)): one encoder and one decoder
(`audio_waveform_encoder_v1` and `audio_waveform_decoder_v1`) that read and rebuild a raw waveform. The
model knows a number of samples, not a sample rate, and everything about files and rates happens in
your `loader_factory`, as for any other data (spec §6.2).

## What a batch is

A batch is a dict with one key, `audio`, holding a float tensor within `[-1, 1]`:

- `(batch, samples)` for mono, or `(batch, channels, samples)` for stereo and more;
- **one sample rate** for the whole dataset, chosen by you (16 kHz for speech, 22.05 or 44.1 kHz for
  music, 8 kHz to keep a first run small). The model never sees it. Remember it when you turn a
  reconstruction back into sound;
- **one number of samples**, equal to `sequence_length.audio` in the data file and to `output_length` in
  the decoder: 2048 samples are 0.128 s at 16 kHz. Longer clips are cut and shorter ones padded by the
  loader, because the decoder reconstructs one fixed length and never resizes it.

## A `loader_factory`

`configs/data/audio.yaml` describes the contract and leaves `loader_factory` and `train_path` for you to
fill in. This is one that follows it. `loadMyClips` is yours: it decodes your files and returns a list of
mono float tensors, one per clip, already resampled to the rate you chose (use a real resampler for that,
the `resample` transform of this framework interpolates without an anti-aliasing filter).

```python
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from global_vae.config.data import DataConfig, DataloaderBundle

PEAK = 0.9  # leaves a margin below the bound of the decoder's tanh


class ClipDataset(Dataset[torch.Tensor]):
    """Clips of exactly `num_samples` samples: cut at random to train, from the start otherwise."""

    def __init__(self, clips: list[torch.Tensor], num_samples: int, random_crop: bool) -> None:
        self.clips = clips
        self.num_samples = num_samples
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> torch.Tensor:
        clip = self.clips[index]
        excess = len(clip) - self.num_samples
        if excess >= 0:
            start = int(torch.randint(0, excess + 1, ())) if self.random_crop else 0
            clip = clip[start : start + self.num_samples]
        else:
            clip = functional.pad(clip, (0, -excess))  # silence at the end
        return PEAK * clip / clip.abs().max().clamp_min(1e-6)


def buildAudioDataloaders(config: DataConfig) -> DataloaderBundle:
    clips = loadMyClips(config.train_path)  # a list of 1D float tensors at one sample rate
    num_samples = (config.sequence_length or {})["audio"]
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(clips), generator=generator).tolist()
    num_val = int(len(clips) * config.val_split)
    num_test = int(len(clips) * config.test_split)
    parts = (order[num_val + num_test :], order[:num_val], order[num_val : num_val + num_test])

    def loader(indices: list[int], train: bool) -> DataLoader:
        dataset = ClipDataset([clips[i] for i in indices], num_samples, random_crop=train)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=train and config.shuffle_train,
            collate_fn=lambda items: {"audio": torch.stack(items)},
        )

    train, val, test = parts
    return DataloaderBundle(
        train=loader(train, True), val=loader(val, False), test=loader(test, False)
    )
```

Split by file, as here, not by clip, if you cut several clips out of one recording: two clips of one
recording in the training and the validation set would make the validation loss look better than it is.

## The model

`configs/model/audio_waveform_single_latent.yaml` is the complete file. What to set:

- **`output_length`** in the decoder, to the number of samples of a clip. It is never filled in for you.
  Any value is reached: the decoder upsamples a seed of `ceil(output_length / prod(strides))` frames and
  crops the surplus, so 16,000 or 22,050 samples need no arithmetic. From the command line:
  `data.sequence_length.audio=16000 model.modalities.audio.decoder.kwargs.output_length=16000`.
- **Channels**: `in_channels` in the encoder and `out_channels` in the decoder, `2` for stereo.
- **The stack**: `hidden_channels` (one entry per stage), `strides` (one value, or one per stage), and
  `dilations` (a residual unit per value in every stage, so `[1, 3, 9]` widens the view of each stage by
  about 13 times). The product of the strides is the number of samples one frame covers (256 for four
  strides of 4). Larger strides are cheaper and blurrier.
- **`output_activation`**: `tanh` by default, which keeps the waveform within `[-1, 1]`. Set it to
  `null` if your data is not (a waveform you standardized, say).

The encoder takes a clip of any length, from one sample up, since it averages over time. The decoder
always gives `output_length` samples.

## Train

```bash
PYTHONPATH=. python scripts/train.py --config-name experiment/audio_vae \
    data.train_path=data/raw/clips \
    data.loader_factory=my_project.data:buildAudioDataloaders \
    data.sequence_length.audio=16000 \
    model.modalities.audio.decoder.kwargs.output_length=16000
```

`experiment/audio_vae` trains on the `sum` reconstruction loss with weight 0.5, since the default `mean`
collapsed the model on synthetic tones (no latent unit active; see
[choose beta](choose-beta.md#choose-a-reduction)). The numbers behind that are in the ADR and are not
tuned to your data. `training.num_epochs` and the sizes above are the first things to change.

## Spectrograms

A spectrogram is an image, and the 2D classes take it with no audio code. Compute a log-magnitude STFT in
your `loader_factory` (`torch.stft`, then `log1p` of the magnitude), give the batch a key of its own such
as `spectrogram` of shape `(batch, frequency bins, frames)`, and use `configs/model/image_single_latent.yaml`
with its modality renamed (the decoder's `output_shape` is `[bins, frames]`).
`tests/integration/test_audio_model_configs.py` runs exactly that. The two representations answer
different questions: a waveform model reconstructs the sound itself, phase included, and a spectrogram
model reconstructs what a clip contains in time and frequency and drops the phase, so it cannot be played
back without a vocoder.

## What it does not do

- **One latent vector per clip.** The encoder averages over time, so the latent holds what the clip
  contains and not when. A decoder asked for the exact waveform of a long or complex clip from 16 numbers
  has little to work with, and reconstructs best what a few numbers describe (a note, a decay). A latent
  per frame is a change to the core (a sequence of latents, the decoder item of roadmap P4-0).
- **A waveform loss.** The reconstruction loss is a squared error on samples, which cares about phase
  and not about what the ear hears. A spectral loss (a distance between STFT magnitudes at several
  resolutions) is a new loss, chosen per decoder like the others, and is not part of this modality.
- **Files and rates**, as above: your loader's job.
