# 0041: The text modality

**Status:** accepted (roadmap item P3-8, third modality after tabular and audio; it adds two classes
through the registries and changes no core module)
**Date:** 2026-10-28

## Context

Roadmap P3-8 lists text among the further modalities, each through the new-modality checklist of spec §10
and only once a dataset is named. No corpus has been named, so nothing here is tuned to one: the model is
generic, the batch contract is stated, and the measurements below are on a small synthetic language. The
tabular ([ADR 0039](0039-tabular-modality.md)) and audio ([ADR 0040](0040-audio-modality.md)) modalities
followed the checklist before it. Text differs from both in what it is: a sequence of **categorical**
tokens, not numbers. That raises two questions before any network is drawn.

**What a batch looks like.** The decoder gives a distribution over the vocabulary at every position, and
the loss compares it with the batch tensor of the same name: `computeTotalReconstructionLoss` refuses a
reconstruction and a target of different shapes (a deliberate guard, written to stop a silent broadcast).
So token ids `(batch, length)` cannot be the target of scores `(batch, length, vocab_size)`, and the target
has to be a tensor of that shape: one-hot. The encoder reads the same batch tensor, so it takes one-hot
rows too. This is read in the core, not a preference, and it has a cost (below).

**Whether a registered family is already the answer.** The 1D transformer pair is the nearest thing:
`1d_transformer_encoder_v1` with `patch_size: 1` and `in_channels: vocab_size` reads a one-hot tensor laid
out `(batch, vocab, length)` (its patch projection then is a linear embedding), and
`1d_transformer_decoder_v1` with `out_channels: vocab_size` writes one. Both run, and both were measured
(below). What they lack is what text needs: the decoder has no output activation, so no probabilities and no
`bce` (`binary_cross_entropy` refuses values outside `[0, 1]`); the encoder has no padding mask, and
padding is the normal case for text; and the layout is channels-first. They also train much worse, as the
table shows.

## Decision

**1. Two registered classes and one module of building blocks.** Nothing else changes:

| Registry name | Class | Reads or writes |
|---|---|---|
| `text_transformer_encoder_v1` | `TextTransformerEncoder` | a one-hot text `(batch, length, vocab_size)`, or integer ids `(batch, length)` |
| `text_transformer_decoder_v1` | `TextTransformerDecoder` | a text of a fixed `output_length`, a distribution over the vocabulary at each position |

`utils/text_blocks.py` holds `TokenEmbedding`, `computePaddingMask`, `computeMaskedMean` and
`computeTokenCrossEntropy`. Positions, the transformer stack and the pooling head are the unchanged pieces of
`utils/transformer_blocks.py` (ADR 0037). The classes are registered by the imports of
`encoders/__init__.py` and `decoders/__init__.py`. `models/`, `latent/`, `losses/`, `training/`,
`evaluation/`, `config/` and `data/` are untouched, and a test reads their source to check that none of
them names the new classes or the `"text"` batch key. The existing transformer classes are not changed.

**2. A text is a one-hot tensor made by the caller.** The batch key `text` holds a float tensor `(batch,
length, vocab_size)`, one row per token, with **one length** for the dataset (`sequence_length.text` in the
data file, `output_length` in the decoder) and **one vocabulary** (`vocab_size`, in the encoder and the
decoder: no field of `DataConfig` carries it, and the config layer never fills it in). Padding is a token
like another, with an id the loader chooses (`pad_token_id`, `0` in the shipped files), and the decoder
predicts it, which is how it learns the length of a text. Reading the files, tokenizing, building the
vocabulary, cutting or padding and the one-hot encoding are the `loader_factory`'s (spec §6.2), and
`docs/how-to/train-on-text.md` has one. The data file's `transforms` entry for `text` is empty and has to
stay so: every transform acts on numbers, and a one-hot row is not a quantity.

**3. The encoder reads ids or one-hot rows, and skips the padding.** `TokenEmbedding` is one table of
vectors. Ids are looked up, a one-hot (or soft) row is multiplied by the table, and the two give identical
vectors (a test), so a trained model encodes with ids and needs no one-hot tensor in memory. With
`pad_token_id` set, padded positions are skipped as attention keys and left out of the mean (a position is
padding when its mass outside the pad class is below one half, which covers pad rows, rows of zeros and a
decoder's probabilities fed back), so **a text padded to a longer length has the same `(mu, logvar)`** (a
test, to `1e-5`, with sinusoidal positions). A text that is padding everywhere keeps its first position, so
attention never has every key masked, which would give `NaN`. Pooling is the masked mean (default) or a
learned `[CLS]` token, the two giving the same accuracy below. Positions are sinusoidal (no length limit,
the default) or learned (up to `max_length`). `minimal_input_length` is `1`.

**4. The decoder gives the whole text in one pass.** Each output token is its position vector plus a linear
projection of the latent vector, as in `1d_transformer_decoder_v1`; the layers of self-attention turn one
vector into a text, and a linear layer gives `vocab_size` scores per position. `output_activation` is
`softmax` (default: probabilities that sum to one, which `mse` and `bce` compare with a one-hot target) or
`null` (raw scores, for a loss that applies its own softmax). There is no autoregression, and that is a
choice with two sides: a decoder that reads the previous tokens needs the target in `forward`, which the
`AbstractDecoder` contract (`forward(z)`) does not carry, and it is the usual way a text VAE comes to ignore
its latent vector (posterior collapse). This decoder cannot, because the latent vector is its only input.
Its price is that tokens are not conditioned on each other, so a long or varied text is hard to rebuild.

**5. The loss: `bce` from YAML, the exact cross-entropy from Python.** The loss names of a training config
are the fixed table of `config/training.py` (`mse`, `l1`, `bce`, `smooth_l1`), with no cross-entropy, so
the experiment sets `reconstruction_loss: bce` with `reconstruction_reduction: sum` and weight `2.0`
(numbers below). `bce` on probabilities is a Bernoulli likelihood on every class of every position, the
negative log-probability of the right token plus a term on the other classes, never below the categorical
one (a test pins it): a proper score and not the exact likelihood. The exact one needs no change to the
framework: `computeTokenCrossEntropy` (summed over the tokens of a text, averaged over the batch, softmax
applied inside) is a plain function of `(reconstruction, target)`, so `Trainer(model,
reconstruction_loss_fn=computeTokenCrossEntropy)` takes it, with the decoder built with
`output_activation: null` (a test trains it end to end).

**6. One model file, one data file and one experiment.** `configs/model/text_transformer_single_latent.yaml`
(a vocabulary of 32, 8 tokens, `d_model` 64, two layers each side, a latent of 16, no fusion),
`configs/data/text.yaml` and `configs/experiment/text_vae.yaml`, which trains through `scripts/train.py` like
the others.

## Measurements

A synthetic language, not a corpus: four sentence templates over a lexicon of 31 words and the padding token
(determiner, adjective, noun, verb, adverb: for example `det adj noun verb det noun`, from 3 to 6 tokens, 8
positions with padding), each slot drawn uniformly, so between 9 and 17 bits per text (14 on average). 4,000 training texts
and 1,000 validation texts, the shipped model (a latent of 16, 104,000 parameters in the encoder and 103,000
in the decoder), Adam at 1e-3, batches of 64, 60 epochs (3,780 steps, with the warm-up of
`training/default.yaml`, 1,000 steps) and two seeds. Token accuracy counts every position, padding included,
from the mean of the posterior; an exact text is one with every position right; the best constant guess (the
most frequent token at each position, ignoring the text) is 45.5%. Means of the two seeds.

| Loss, reduction, weight | Token accuracy | Exact texts | Active units |
|---|---|---|---|
| any `mean` reduction (`mse` or `bce`), 1.0 | 45.4% | 0% | 0 of 16 |
| `mse`, `sum`, 0.5 | 45.4% | 0% | 0 of 16 |
| `mse`, `sum`, 1.0 | 47.9% | 0% | 1 of 16 |
| `mse`, `sum`, 8.0 | 86.8% | 31% | 10 to 11 |
| `bce`, `sum`, 0.5 | 54.5% | 0.4% | 1 |
| `bce`, `sum`, 1.0 | 87.0% | 31% | 12 to 13 |
| `bce`, `sum`, 1.0, `cls` pooling | 86.9% | 30% | 11 to 12 |
| **`bce`, `sum`, 2.0 (the experiment)** | **99.3%** | **94%** | 15 to 16 |
| `bce`, `sum`, 4.0 | 100.0% | 99.95% | 16 |
| exact cross-entropy, 1.0 | 54.6% | 0.7% | 2 |
| exact cross-entropy, 2.0 | 99.6% | 97% | 15 to 16 |
| exact cross-entropy, 4.0 | 100.0% | 99.9% | 16 |
| exact cross-entropy, 2.0, no padding mask (`pad_token_id: null`) | 96.4% | 71% | 12 to 14 |
| 1D transformer pair, one-hot channels-first, `mse`, `sum`, 8.0 | 62.0% | 0.4% | 2 |
| 1D transformer pair, one-hot channels-first, exact cross-entropy, 2.0 | 64.6% | 0.5% | 3 |

What it says:

- **The pattern of the other modalities holds, and the weight is the lever.** Under `mean` the KL outweighs
  the reconstruction and the latent vector is unused, so the model sits at the constant guess. A weight `w`
  on a summed reconstruction is a `beta` of `1 / w` on the KL term, and every setting at `w <= 1` is poor
  while `w` of 2 and 4 is nearly perfect. The exact likelihood at `beta = 1` (cross-entropy at 1.0) leaves 14
  of 16 units unused: encoding a uniformly random word costs about what it saves, so the true ELBO has
  little reason to use the latent vector on this language. The choice of beta stays roadmap decision D-6.
- **`bce` is a fair stand-in for the cross-entropy.** At equal weight they are close (99.3% against 99.6%
  at 2.0, with 94% and 97% exact texts), so the YAML route costs little. The squared error needed a weight of
  8 to reach what `bce` has at 1.0.
- **The pair earns its place against the 1D transformer classes.** Under the same loss, 87% against 62%
  (`mse`, 8.0) and 99.6% against 64.6% (cross-entropy, 2.0). Why was not isolated. One difference was: the
  padding mask is worth about three points of accuracy and 26 of exact texts (and a steadier result
  between seeds), which is not the 35 points of the gap. The next suspect, not tested, is the scale of the
  embedding: a convolution's default initialization gives token vectors much smaller than the unit-size
  position vectors, against a table initialized at unit scale.

These are numbers from one synthetic language, two seeds and no tuning: they show that the model trains and
how the weight behaves, not what a corpus would give. A language of 14 bits per text on average fits in 16
latent numbers; real text carries far more, and nothing here says how the model does on it.

## Consequences

- **The registry design held a fourth time.** Two classes, one utility module, three config files and the
  tests are all that was added, and no core edit was needed or made. As for audio, the question was whether
  a registered family already did the job, and it did not, for reasons that were measured.
- **Finding, not fixed: no cross-entropy in the loss table, and ids cannot be targets.** The one-hot batch
  tensor exists because the loss demands a target of the shape of the reconstruction, and it costs
  `batch * length * vocab_size` floats: 64 texts of 128 tokens over a vocabulary of 30,000 are about a
  gigabyte. Characters and bytes (a vocabulary of 100 to 256) and word vocabularies of a few thousand fit;
  subword vocabularies at the usual size do not. Taking ids as targets, with a `cross_entropy` in the table
  of `config/training.py` and a shape rule in `computeTotalReconstructionLoss` that allows it, is a change to
  two core modules, flagged and not made. Until then the encoder takes ids at inference, and the exact loss
  is a Python function.
- **Finding, not fixed: the exact loss cannot be named in YAML.** `scripts/train.py` can only build the
  losses of the table, so the better-founded training is a Python script that builds a `Trainer`
  (`docs/how-to/train-on-text.md` shows the call). A way to name a loss by import path in the config is a
  core change, and was not part of this item.
- **Finding, not fixed: one latent vector per text.** The encoder pools over the tokens, so the latent
  holds what the text contains and not where, and the decoder writes every token from that one vector. A
  sequence of latents is the single-tensor decoder item of the roadmap audit P4-0.
- **Finding, not fixed: no vocabulary in `DataConfig`.** The vocabulary size is the last axis of the batch
  and the `vocab_size` of two classes, three places with no check between the loader and the model but the
  shapes (a test pins the two errors). A field would be a change to `config/data.py`.
- **Evaluation needs nothing.** `scripts/evaluate.py` and the figure export run on a text model with no
  text code (a test). The reconstruction grid treats a 3D batch as images, so it draws the one-hot target
  and the probabilities as position-by-vocabulary heatmaps, and `r2` and `pearson_r` of probabilities
  against one-hot rows are not the token accuracy, which a user reads with `argmax`.
- **No tokenizer, no pretrained embeddings, no dataset.** Tokenizing and the vocabulary are the loader's,
  the embedding table is learned from the texts, and no named corpus exists, so the proof is the synthetic
  language and the dummy loader of the tests. The remaining modalities of P3-8 (time series, graphs, point
  clouds) stay open.
- **Cost.** The shipped file is about 104,000 parameters on each side, the class defaults (three layers)
  about 154,000, and a vocabulary of 5,000 words with `d_model` 256 and four layers about 4.5 million on
  each side, most of it the embedding and the output layer. Attention is quadratic in the number of tokens.
