# Train on text

Text is the text modality ([ADR 0041](../adr/0041-text-modality.md)): one encoder and one decoder
(`text_transformer_encoder_v1` and `text_transformer_decoder_v1`) that read and rebuild a sequence of
tokens. The model knows token ids, not words: reading files, tokenizing and building the vocabulary all
happen in your `loader_factory`, as for any other data (spec §6.2).

## What a batch is

A batch is a dict with one key, `text`, holding a **one-hot float tensor `(batch, length, vocab_size)`**:
one row per token, padding included.

The shape is not a matter of taste. The loss compares a decoder's reconstruction with the batch tensor of
the same name, and the two must have one shape. The decoder gives a distribution over the vocabulary at
every position, so the target has to be one-hot, and so does the encoder's input, which is the same tensor.
The encoder also takes integer ids `(batch, length)` and gives the same output, which is how to encode with
no one-hot tensor in memory once the model is trained.

- **`vocab_size`** counts every token the model can see, the padding token included. It is written in the
  encoder and in the decoder, and it is the last axis of the batch. Nothing fills it in for you.
- **One length**, equal to `sequence_length.text` in the data file and to `output_length` in the decoder.
  Longer texts are cut and shorter ones padded by the loader, because the decoder reconstructs one fixed
  length and never resizes it.
- **A padding token**, id `0` in the shipped files (`pad_token_id: 0` in the encoder). The decoder predicts
  it, which is how it learns the length of a text. The encoder skips it, so a text padded to a longer length
  gives the same latent vector.

## A `loader_factory`

`configs/data/text.yaml` describes the contract and leaves `loader_factory` and `train_path` for you to fill
in. This is one that follows it, with a vocabulary of whole words. `loadMyTexts` is yours: it reads your
files and returns a list of texts, each a list of string tokens (split a sentence, or take its characters,
however suits your language).

```python
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from global_vae.config.data import DataConfig, DataloaderBundle

PAD, UNKNOWN = 0, 1  # the first two ids of the vocabulary
VOCAB_SIZE = 32  # must equal vocab_size in the model file: it counts PAD and UNKNOWN


def buildVocabulary(texts: list[list[str]]) -> dict[str, int]:
    """The most frequent tokens of the training texts, from id 2 up."""
    counts: dict[str, int] = {}
    for tokens in texts:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    frequent = sorted(counts, key=lambda token: (-counts[token], token))[: VOCAB_SIZE - 2]
    return {token: index for index, token in enumerate(frequent, start=2)}


def encodeText(tokens: list[str], vocabulary: dict[str, int], length: int) -> torch.Tensor:
    """Ids of exactly `length` tokens: cut if longer, padded with PAD if shorter."""
    ids = [vocabulary.get(token, UNKNOWN) for token in tokens][:length]
    return torch.tensor(ids + [PAD] * (length - len(ids)))


def buildTextDataloaders(config: DataConfig) -> DataloaderBundle:
    texts = loadMyTexts(config.train_path)  # a list of lists of string tokens
    length = (config.sequence_length or {})["text"]
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(texts), generator=generator).tolist()
    num_val = int(len(texts) * config.val_split)
    num_test = int(len(texts) * config.test_split)
    train, val, test = (
        order[num_val + num_test :],
        order[:num_val],
        order[num_val : num_val + num_test],
    )
    vocabulary = buildVocabulary([texts[i] for i in train])  # from the training texts only

    def loader(indices: list[int], shuffle: bool) -> DataLoader:
        ids = [encodeText(texts[i], vocabulary, length) for i in indices]
        return DataLoader(
            ids,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=lambda rows: {
                "text": functional.one_hot(torch.stack(rows), VOCAB_SIZE).float()
            },
        )

    return DataloaderBundle(
        train=loader(train, config.shuffle_train), val=loader(val, False), test=loader(test, False)
    )
```

The vocabulary is built from the training texts only, so that nothing about the validation texts leaks into
the model. A token that is not in it becomes `UNKNOWN`, which the decoder can predict like any other token.
Keep `buildVocabulary` and the file of the model in step: the number of tokens it returns, plus `PAD` and
`UNKNOWN`, is `vocab_size`.

## The model

`configs/model/text_transformer_single_latent.yaml` is the complete file. What to set:

- **`vocab_size`** in the encoder and in the decoder, to the width of your one-hot rows. From the command
  line: `model.modalities.text.encoder.kwargs.vocab_size=5000
  model.modalities.text.decoder.kwargs.vocab_size=5000`.
- **`output_length`** in the decoder, to the number of tokens of a text, and the same number as
  `data.sequence_length.text`.
- **`pad_token_id`** in the encoder, `0` here, or `null` if your texts all have the same length and need no
  padding.
- **The stack**: `d_model`, `num_heads` (dividing `d_model`), `num_layers`, in each of the two. `pooling` in
  the encoder is `mean` (the mean of the real tokens, the default) or `cls` (a learned token).
  `positional_encoding` is `sinusoidal` (any length) or `learned` (a vector per position, which needs
  `max_length` in the encoder).
- **`output_activation`** in the decoder: `softmax` by default, `null` for raw scores (see the loss below).

The encoder takes a text of any length from one token up. The decoder always gives `output_length` tokens,
produced all at once from the latent vector, with no token feeding the next. A decoder that reads the tokens
before it is the usual way a text VAE learns to ignore its latent vector, and this one has nothing else to
read.

## Train

```bash
PYTHONPATH=. python scripts/train.py --config-name experiment/text_vae \
    data.train_path=data/raw/sentences.txt \
    data.loader_factory=my_project.data:buildTextDataloaders \
    data.sequence_length.text=24 \
    model.modalities.text.decoder.kwargs.output_length=24
```

## The loss

`experiment/text_vae` sets `reconstruction_loss: bce` with the `sum` reduction and weight `2.0`. The decoder
returns probabilities, and `bce` against the one-hot target is a Bernoulli likelihood on every class of every
position: the negative log-probability of the right token plus a term on the other classes. It is a proper
score and not exactly the categorical likelihood of a text, and it is the closest thing a YAML file can name,
since its loss names are the fixed table of `config/training.py` (`mse`, `l1`, `bce`, `smooth_l1`).

**The weight is the lever.** A weight of `w` on the reconstruction is a `beta` of `1 / w` on the KL term, and a
text VAE at `beta = 1` uses little of its latent vector: encoding a random word costs about what it saves in
reconstruction. On a synthetic language ([ADR 0041](../adr/0041-text-modality.md)), `bce` at weight 1.0 gave
87% of the tokens right, 2.0 gave 99% and 4.0 gave 100%, while any `mean` reduction and `sum` at 0.5 left the
latent vector unused. These are numbers from a small synthetic grammar and not a tuned value (the choice of
`beta` is [roadmap](../roadmap.md) decision D-6), so try a few weights on your own data, and watch the number
of active latent units that `scripts/evaluate.py` reports.

**The exact likelihood needs no change to the framework, only Python.** Set the decoder's `output_activation`
to `null`, so that it returns raw scores, and give the trainer `computeTokenCrossEntropy`, which applies its
own softmax and sums over the tokens of a text:

```python
from global_vae.training.trainer import Trainer
from global_vae.utils.text_blocks import computeTokenCrossEntropy

trainer = Trainer(
    model,  # built with model.modalities.text.decoder.kwargs.output_activation=null
    reconstruction_loss_fn=computeTokenCrossEntropy,
    reconstruction_weights=2.0,  # the reduction stays "mean": the function sums over tokens itself
)
```

It trained the synthetic language about as well as `bce` at the same weight (99.6% of the tokens at 2.0,
against 99.3%), so `bce` loses little. The YAML files cannot name it, because a `cross_entropy` in the loss
table is a change to a core module, which ADR 0041 flags and does not make. A position whose target row is all
zeros adds nothing to it, so pad with the pad class and not with zeros if the decoder is to learn lengths.

## Reading a reconstruction

The most likely tokens are the `argmax` of the reconstruction. Invert your vocabulary to get words back, and
read each text up to its first `PAD`:

```python
reverse = {index: token for token, index in vocabulary.items()} | {PAD: "<pad>", UNKNOWN: "<unk>"}

model.eval()
with torch.no_grad():
    # A reconstruction of a batch, from the mean of its posterior.
    reconstruction = model(batch, use_mean=True)["reconstructions"]["text"]
    # Eight texts from the prior: the decoder takes a latent vector, whatever its origin.
    sampled = model.decoders["text"](torch.randn(8, model.latent_spaces["z_fused"].dim))

for scores in (reconstruction, sampled):
    ids = scores.argmax(dim=-1)  # (batch, length)
    sentences = [[reverse[int(i)] for i in row] for row in ids]
```

## What it does not do

- **A big vocabulary.** A target has to be a tensor of the shape of the reconstruction, so a batch holds
  `batch * length * vocab_size` floats: 64 texts of 128 tokens over a vocabulary of 30,000 are about a
  gigabyte. Characters or bytes (a vocabulary of 100 to 256) and word vocabularies of a few thousand tokens
  fit. Subword vocabularies at the usual 30,000 do not, until the loss can take ids as targets, which is a
  change to the core (ADR 0041).
- **One latent vector per text.** The encoder pools over the tokens, so the latent holds what the text
  contains and not where. A decoder that has to give every token of a long text from 16 numbers does best on
  short texts with few degrees of freedom.
- **Pretrained embeddings, a tokenizer, a language model.** None of them is part of this modality. The
  embedding table is learned from your texts, and tokenizing is your loader's job.
