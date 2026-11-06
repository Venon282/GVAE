# 0046: Encode and decode seams

**Status:** accepted (roadmap item P5-5; it adds two public methods to `GlobalVae` and one typed dict, and
changes no output of `forward`)
**Date:** 2026-11-05

## Context

`GlobalVae.forward` is the only public path through the model, and it always runs the five stages: encode,
fuse, sample, assemble and decode. Several pieces of work need to stop after the third stage or to start after
it:

- **Retrieval evaluation** (I-6, `evaluation/paired_alignment.py`) reads the posterior mean of each modality
  alone. `collectModalityPosteriorMeans` called `model(alone, use_mean=True)` and kept `latent_params`, so every
  query paid for the assemblers and the decoders of every modality, and threw their output away.
- **Cross decoding** (Phase 6) encodes with one modality and decodes with the decoder of another.
- **Sampling from a prior** (P4-2) has no inputs: it draws a latent sample and decodes it.
- **A latent reconstruction loss** (P7-3) encodes, decodes, encodes the reconstruction and compares the two
  latent vectors.
- **Export** (P4-4) writes the posterior of a dataset and needs no decoder.
- **P4-0** would have found the same blocker.

The loop of `forward` had the cut already in it, as two phases that share a dictionary of latent samples. This
ADR makes the cut public.

Two properties had to survive. Under a fixed seed the outputs of `forward` must stay what they were in 2.0.0
to the last bit, because checkpoints, golden numbers in the docs and every published comparison depend on
them. And `forward` must keep the error messages it has, which name `GlobalVae.forward()` and which tests and
users match on.

## Decision

**1. `GlobalVae.encode(inputs, use_mean=False)` runs the encoders, the heads, the fusion and the sampling, and
stops.** It returns a dict with two keys, typed as `EncodeOutput`:

| Key | Content |
|---|---|
| `latent_params` | latent space name -> `(mu, logvar)` |
| `latent_samples` | latent space name -> a sample, or `mu` when `use_mean=True` |

The rules for missing modalities are the ones of `forward`: any non-empty subset of the input modalities is
accepted, a latent space with no available input is absent from both dicts. An empty `inputs`, or a key that no
encoder ingests, raises as it does in `forward`, with the message naming `GlobalVae.encode()`.

**2. `GlobalVae.decode(latent_samples)` runs the assemblers and the decoders, and returns the
reconstructions.** It returns the dict that `forward` puts under `"reconstructions"`: the decoder name for a
one-modality decoder, each declared `output_modalities` for a shared one. For every decoder, in decoder order:

- no latent space it consumes is present: the decoder is left out;
- it has no assembler: its one latent sample is its input;
- it has an assembler and a strict subset of its latent spaces is present: the decoder is left out when the
  assembler does not handle a subset (`handlesMissingLatents` is `False`, `concat`), and otherwise the
  assembler receives the present spaces and their indices (spec §5);
- a shared decoder that returns other modalities than it declares raises, as before.

These are the rules of `forward` today, moved and not changed.

**3. `forward` is `encode` followed by `decode`.** It returns the same three keys (`reconstructions`,
`latent_params`, `latent_samples`) as before. The two halves are private helpers, `_encode` and `_decode`,
that do not check their arguments, and `forward`, `encode` and `decode` each check their own and name
themselves in the message. This keeps the messages of `forward` as they were without a second copy of the
loop. The order of the operations does not change: every latent space is sampled, in graph order, before the
first decoder runs, so the draws from the global random state fall in the same order and a fixed seed gives
the same tensors.

**4. Cross decoding needs no new method.** The sample of one latent space is passed under the name of
another, for example `decode({"z_image": samples["z_signal"]})`, and the decoders that read `z_image` decode
it. It needs equal latent widths, which the caller checks. `decode` checks the last axis of every sample
against the width of the latent space it is passed under, and a mismatch raises a `ValueError` that names
both widths and says that cross decoding needs equal widths. A name that is not a latent space of the model
raises a `KeyError` that lists the real ones, so a typo cannot silently decode nothing, and an empty mapping
raises a `ValueError`. A value that is not a tensor raises a `TypeError`.

**5. `collectModalityPosteriorMeans` uses `encode`.** `model.encode(alone, use_mean=True)["latent_params"]`
returns what `model(alone, use_mean=True)["latent_params"]` returned (a test compares them for equality),
without a pass through the assemblers and the decoders.

**6. `EncodeOutput` is a `TypedDict`.** It is a plain dict at run time. It is there so that
`model.decode(model.encode(x)["latent_samples"])` type checks without a cast, which the union type of
`forward`'s return does not allow. `forward` keeps its return annotation, so no caller of `forward` changes.

## Alternatives considered

- **A `crossDecode(samples, source, target)` method.** It is the renamed call of Decision 4 with a name for it
  and one more method to keep. It also has to decide what happens with several latent spaces and several
  decoders, which the plain call does not (the caller names the space and the decoders follow the graph).
- **A flag on `forward`, such as `forward(inputs, stop_at="latent")`.** The type of the result would depend on
  the value of a string, so a type checker cannot follow it, and it still leaves no way to start from a latent.
- **Keep `forward` as it is and add the two methods with their own copy of the loop.** Two loops drift, and
  the bit for bit guarantee would have to be tested between them forever. One loop, two doors, is what the
  tests below pin.
- **`decode` takes `latent_params` and samples.** Sampling is the last stage of `encode`, and a decode from
  a prior or from another model's latent space has samples and no parameters. A caller that holds
  `(mu, logvar)` calls `LatentSpace.reparameterize` or passes `mu`.
- **`decode` returns the three keys of `forward`.** It has no parameters to return and its samples are its
  input, so two of the three would be copies of the argument.
- **No check of names and widths in `decode`.** A typo would be ignored when another name is right, and
  would decode nothing and return `{}` when none is. A width mismatch would fail inside a decoder, far from
  its cause. The check costs one comparison per sample.

## Measurements

Retrieval pays one query per modality. Median wall time of one pass over a batch of 32 samples, signal and
image alone, on one CPU, 6 timed runs after a warm-up (the figures are indicative, a loaded machine moves
them by a few milliseconds):

| Model config | `forward` (ms) | `encode` (ms) | Ratio |
|---|---|---|---|
| `signal_image_shared_private` | 27.7 | 20.9 | 1.33 |
| `signal_image_shared_transformer` | 43.9 | 16.7 | 2.63 |
| `signal_image_transformer_cross_attention` | 34.1 | 11.9 | 2.87 |

What is saved is the decoders and the assemblers, so the gain is largest where the decoders are the heavier
half (the transformer configs) and smallest for the shared-plus-private model, whose encoders dominate. The
measurement is of the call and not of an evaluation run, which adds the retrieval ranking on top.

## Consequences

- **`forward` is unchanged for every caller.** A test keeps a copy of the 2.0.0 loop and compares it with
  `forward` bit for bit, under the same seed, on every row of the configuration matrix with each assembler
  and on a model with a shared transformer encoder and decoder, with `use_mean` off and on, and again on every
  subset of the input modalities. `encode` is compared with the first half of the copy, `decode` with the
  second half on every subset of the latent spaces, and `encode` then `decode` with `forward`.
- **A paired retrieval no longer calls a decoder.** A test counts the forward calls of every decoder and every
  assembler during `collectModalityPosteriorMeans` and during `evaluatePairedAlignment` on the shipped model
  configs (shared plus private, shared transformer): none. The control in the same test is that `forward`, on
  the same queries, does call them.
- **The trainer is untouched.** It calls the model, so `forward`, which keeps its signature, its keys and its
  messages. The mixed precision, gradient accumulation and `torch.compile` tests pass unchanged. A compiled
  model compiles `forward` and traces the two private helpers through it.
- **The seams are the contract that the next items build on.** Prior sampling (P4-2), export (P4-4), the
  latent reconstruction loss (P7-3) and cross decoding (Phase 6) call `encode` and `decode`, and none of them
  has to add a method to the core.
- **Finding, not fixed: `decode` cannot tell a width that matches by chance from a space that matches by
  meaning.** Passing `z_signal` under the name `z_image` when both have 32 dimensions runs, and whether the
  decoder of the image can read the latent space of the signal is a property of the training, not of the
  call. Phase 6 trains for it; this ADR only makes the call possible and its misuse loud when the widths
  differ.
- **Specification unchanged.** The specification describes the model through its constructors and its stages
  (spec §2, §5 and §6) and does not define the public methods of `GlobalVae`, so no spec sentence is affected
  and no owner approval is needed for this item.
