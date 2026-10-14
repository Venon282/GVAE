# 0027: Shared encoders and shared decoders

**Status:** accepted (option A of roadmap decision D-2, the default the roadmap proposes; it needs
the owner's approval)
**Date:** 2026-10-14
**Supersedes:** the rule of [ADR 0019](0019-decouple-encoder-inputs-from-decoder-targets.md) that a
batch key must name one of the model's encoders. A key now names a modality that some encoder
ingests (the two are the same set for every model built before this ADR). The rest of ADR 0019
stands, and ADR 0019 itself is left as written.

## Context

Spec §2.1 has four rows with a single shared encoder (`E1-*`) and four with a single shared decoder
(`*-D1`), and says the modalities are "tokenized/concatenated into one stream in and out". It does
not say who builds that stream, and the code gave no way to build it:

- `AbstractEncoder.forward` takes one tensor, and `GlobalVae.forward` feeds `inputs[name]` to
  `encoders[name]`. The batch key and the encoder name are the same string, so one encoder can only
  ever see one modality.
- `AbstractDecoder.forward` returns one tensor, and `reconstructions[decoder_name]` holds it. The
  loss, the evaluation and the figures pair each reconstruction with the batch key of the same
  name, so a decoder can only ever reconstruct one modality.
- The `*-D1` rows that did build (`EN-L1-D1`, `EN-LN-D1`) have one decoder that produces one tensor.
  With a signal and an image of different shapes, the decoder reconstructs one of them (the
  translation model of example 03). That is a valid model, but it is not the "one stream out" of the
  spec.

Roadmap decision D-2 listed three ways to close the gap:

- **A.** A shared encoder receives `dict[str, Tensor]`, a shared decoder returns one.
- **B.** A registry of stream adapters (`pack` and `unpack`) around one-tensor modules.
- **C.** Weight tying only, or a caller-packed tensor under one key. It works today, but modality
  dropout and the cross-modal reports cannot see inside the stream.

The roadmap proposes A. This ADR records that choice and the design that follows from it.

## Decision

**A shared encoder declares the modalities it ingests and receives them as a dict.**
`encoders/shared_base.py` holds `AbstractSharedEncoder`:

- `forward(inputs: Mapping[str, Tensor]) -> (mu, logvar)`, with `inputs` holding a non-empty subset
  of `input_modalities`. A modality that is not in the batch is **omitted** from the dict. It is
  never `None` and never replaced by a placeholder tensor.
- `input_modalities: tuple[str, ...]`, an abstract property: the batch keys the encoder ingests.
- `latent_dim: int`, an abstract property, as for `AbstractEncoder`, because latent heads and the
  config layer size themselves from it.

**A shared decoder returns one tensor per modality it reconstructs.** `decoders/shared_base.py`
holds `AbstractSharedDecoder`: `forward(z) -> dict[str, Tensor]` and
`output_modalities: tuple[str, ...]`. `forward` returns exactly one entry per declared output
modality.

**The two live in the existing registries.** `registerEncoder` and `registerDecoder` accept either
kind of class, so a config names a shared module the way it names any other, under `encoders` and
`decoders`. There is no new registry and no new config key. The two kinds are told apart by
`isinstance`, never by a name.

**A batch key names an input modality.** For an `AbstractEncoder` the modality is the encoder's
instance name, exactly as before (ADR 0019). For an `AbstractSharedEncoder` it is each entry of
`input_modalities`. `GlobalVae` computes `input_modalities` (the batch keys it can encode) and
`output_modalities` (the keys of `"reconstructions"`) once at construction, in encoder and decoder
order. It refuses to build a model in which two encoders ingest the same modality, or two decoders
reconstruct the same one, since a batch key must name exactly one owner.

**`GlobalVae.forward` routes by modality.** It groups `inputs` by the encoder that owns each key and
calls each group's encoder once: with the tensor for an `AbstractEncoder`, with a dict of the
present modalities, in `input_modalities` order, for an `AbstractSharedEncoder`. An encoder with no
modality in `inputs` does not run, as before. A shared decoder's dict is checked against its
`output_modalities` and flattened into `"reconstructions"` by modality name, so a shared decoder
looks like several one-modality decoders to everything downstream.

**The routing graph does not change.** A shared encoder or decoder is one node, named by its
instance name, like any other. `E1-LN-*` is a single encoder node with several outgoing edges, each
with a latent head (ADR 0024), and `*-D1` is a single decoder node whose assembler merges the
latent spaces it consumes (ADR 0026). `validateRoutingGraph` needs no new rule.

**What reads modality names switches to the new sets.** `selectEncoderInputs` filters a batch by
`input_modalities`. The cross-modal reports (`resolveDefaultInputSubsets`,
`collectCrossModalReconstructions`, `exportCrossModalFigures`), `exportEvaluationFigures` and
`scripts/visualize_latent.py` read `input_modalities` and `output_modalities` instead of
`model.encoders` and `model.decoders`. For a model with one-tensor modules the two are equal, so
none of them changes behaviour. The losses, `evaluate` and the plots, which already walk
`"reconstructions"` by name, need no change, and neither does `Trainer`: modality dropout removes
keys from the encoder inputs, and a key is a modality whether or not a shared encoder reads it.

**A shared encoder owns its missing-modality behaviour.** With one encoder there is nothing to fuse,
so the robustness that spec §5 gets from the fusion strategy has to come from the module: it must
accept every non-empty subset of `input_modalities`. A module that cannot fails loudly inside its
own `forward` the first time modality dropout hides a modality. `GlobalVae` does not skip such an
encoder, since skipping the only encoder would leave a pass with no latent space at all.

**The tokenisation lives in the concrete module.** How a signal and an image become one stream
(tokens, a concatenated vector, a sum of projections) is the module's own business. This is where a
transformer backbone (roadmap P3-5) needs it anyway, and it keeps `GlobalVae` free of any layout
assumption. No concrete shared module ships in `src/` with this change: the tests use small dummy
modules, and the how-to shows the contract.

## Alternatives considered

- **B, a registry of stream adapters.** It reuses every one-tensor module unchanged, but it adds a
  concept (an adapter), a registry and a pair of methods to every shared topology, and spec §12 asks
  for no new names for known concepts. It also moves the tokenisation out of the module that needs
  to own it, and an adapter in front of a one-tensor encoder cannot let that encoder skip a missing
  modality.
- **C, a caller-packed tensor under one key.** It stays available as the zero-cost fallback. But
  modality dropout can only drop the whole pack, and the cross-modal matrix has one row for it, so
  the per-modality reports that the shared rows are meant to be compared with do not exist.
- **Widen `AbstractEncoder.forward` to `Tensor | Mapping[str, Tensor]`.** One ABC instead of two, but
  every existing encoder would then narrow its parameter type, which a type checker rejects, and
  every caller would have to narrow it again. Two sibling classes keep each signature exact.
- **Make `AbstractSharedEncoder` a subclass of `AbstractEncoder`.** The same signature problem, and
  `modality_name` and `minimal_input_length` would be meaningless for it.
- **Skip a shared encoder that receives a strict subset, like a `concat` decoder in ADR 0024.** With
  several decoders, skipping one leaves the others running. With one encoder, skipping it leaves
  nothing to decode, and a training step with no reconstruction fails far from its cause.
- **A shared decoder that may return a subset of its modalities.** It would let a decoder skip what
  it cannot produce, but the loss would then depend on which keys came back. Exactly the declared
  set is simpler to check and to reason about.

## Consequences

- The `E1-L1-DN`, `E1-L1-D1`, `E1-LN-DN` and `E1-LN-D1` rows build from config (`encoders` and
  `decoders` sections, shared modules named by registry key) and from a `RoutingGraph`, and run
  through `Trainer`, `evaluate` and the cross-modal reports. The configuration matrix test covers
  them with dummy shared modules (roadmap P1-6, stage 3).
- A shared encoder is not written under `modalities`, which pairs one encoder and one decoder under
  one name. A shared module goes under `encoders` or `decoders`, and its modality names go in its own
  `kwargs`, since only the module knows them. The config layer still fills in `latent_dim`, so a
  shared module must accept that argument.
- `model.encoders` and `model.decoders` are keyed by instance name and no longer always by modality
  name. Code outside this repository that reads them as modality names keeps working for one-tensor
  models and must switch to `model.input_modalities` and `model.output_modalities` for the shared
  ones.
- A modality name is now global to the model: it is a batch key, so it cannot be ingested by two
  encoders, and it cannot be reconstructed by two decoders. The check raises a `ValueError` at
  construction, without a config path, because only the built modules know their modalities.
- Checkpoints of models without shared modules are unchanged: no state key moves.
- Per-modality `reconstruction_loss` and `reconstruction_weight` keep working for a shared decoder,
  because the reconstructions they key are modality names.
- Shipping a concrete shared module (a transformer over tokens, a ViT) is roadmap P3-5 and needs no
  further change to `GlobalVae`. If it does, that is a design flag to raise.
