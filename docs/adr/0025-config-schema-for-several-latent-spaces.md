# 0025: Config schema for several latent spaces

**Status:** accepted (the schema is the default proposed for roadmap decision D-1, taken without
an explicit answer: it needs the owner's approval)
**Date:** 2026-10-12
**Supersedes:** the "only `single` is wired up" part of
[ADR 0011](0011-hydra-config-layer.md) (`ModelConfig`, `buildModelFromConfig` and the paragraph
that has `latent_mode: "several"` raise `NotImplementedError`). The rest of ADR 0011 stands, and
ADR 0011 itself is left as written.

## Context

ADR 0011 gave `ModelConfig` a `latent_mode` of `"single"` or `"several"` and wired only the first
to a builder. Since then the Python side has caught up: `GlobalVae.__init__` takes an arbitrary
`RoutingGraph` with independent encoder and decoder names (ADR 0002, ADR 0019), and ADR 0024 added
latent heads so one encoder can feed several latent spaces. The YAML side stayed behind, for two
reasons:

- `ModalityConfig` holds one encoder and one decoder under one name. A decoder with no encoder of
  the same name, which is the translation model of `examples/03_signal_image_to_image.py`, cannot
  be written, so that example builds its model in Python.
- `buildModelFromConfig` raised `NotImplementedError` for `latent_mode: "several"`, so the second
  and third patterns of spec §9 (several independent latent spaces, shared plus private) had no
  config at all.

Spec §9 is illustrative, and read against the code it leaves four points open, which roadmap
decision D-1 lists: heads are per edge but §9 puts `head:` on the latent space (ambiguous when
several encoders feed it); decoders need instance names independent of modality names; beta per
latent space already lives in `TrainingConfig.beta_schedules`; the regularizer is already per
latent space. D-1's proposed default answers all four. No answer was available, so this ADR takes
that default and says so.

## Decision

**Keep `modalities`, add `encoders` and `decoders`.** `modalities.<name>.{encoder,decoder}` stays
as the shorthand for the common case: one encoder and one decoder, both under `<name>`. Two new
sections, `encoders` and `decoders`, each map an instance name to `{name, kwargs}`
(`EncoderConfig`, `DecoderConfig`) with no tie between the two. The model's encoders are those of
`modalities` followed by those of `encoders`, and likewise for decoders. A name used in both
places is an error naming both fields. The two sections work in both latent modes, so a
single-latent model whose decoder target is not an encoder input is now writable too
(`configs/model/signal_image_to_image.yaml`, the model of example 03).

**`latent_mode: several` takes a `latent` section.** `latent_mode` and `single_latent` keep their
names, since every shipped file uses them (the spec's `latent.mode` was illustrative). The new
`latent: LatentConfig` has two parts:

- `spaces`: latent space name -> `LatentSpaceConfig{dim, fed_by, fusion, regularizer}`.
  `fusion` and `regularizer` reuse `FusionConfig` and `RegularizerConfig`, so a latent space is
  configured exactly like the single one. `fusion` is required when several encoders feed the
  space and refused when one does (it would never run).
- `decoders_consume`: decoder instance name -> `DecoderConsumptionConfig{spaces, assembler,
  kwargs}`. `assembler` is required when a decoder consumes several spaces and refused when it
  consumes one. `kwargs` go to the assembler's constructor, which needed one change in the model:
  `GlobalVae.__init__` gains `assembler_kwargs` (decoder name -> kwargs), the counterpart of
  `fusion_kwargs` and `regularizer_kwargs`. The built-in assemblers take none, so nothing changes
  for them; the parameterised assemblers of roadmap P1-4 will.

**Heads sit on the edge, written in `fed_by`.** `fed_by` is either a list of encoder names, each an
identity edge, or a mapping `encoder -> {head, kwargs}` (`EdgeHeadConfig`, or `null` for an
identity edge), one latent head per edge. This is D-1 point (a): `head:` moves from the latent
space to the edge, so a space fed by several encoders can give each its own head. A list entry
builds no module; a mapping entry always builds a `HeadSpec` and so a head module, and an
explicit `identity` head checks that the two widths agree (ADR 0024). Both forms map straight onto
`RoutingGraph.encoder_heads`. `fed_by` is typed `Any` in the dataclass because OmegaConf cannot
express a list-or-mapping union; `parseFedBy` checks the shape instead and names the field when it
is wrong.

**Beta is not duplicated.** D-1 points (c) and (d): beta stays in `TrainingConfig.beta_schedules`
keyed by the latent space names declared here, and the regularizer is a field of each latent
space. `buildTrainerFromConfig` logs a warning for a `beta_schedules` key that is not a latent
space of the model, since a schedule written for another model's names would otherwise be dropped
without a sign. It stays a warning, because `configs/training/default.yaml` names `z_fused` and
works with every single-latent model.

**Widths are filled in, never repeated.** As in single mode, `latent_dim` is left out of the
`kwargs` and derived, and an explicit value always wins over the derived one:

- an encoder takes the `dim` of the latent space it feeds through an identity edge. An encoder
  whose every edge goes through a head has no such space, so it must say `latent_dim` itself
  (the error names the field);
- a decoder takes the width of what its assembler produces from the spaces it consumes: the
  space's `dim` for one space, otherwise the assembler is built and run once on zero tensors of
  the right widths, so any registered assembler works without a width formula in the config layer.

An explicit value that disagrees with the derived one is an error in `several` mode (an identity
edge between two widths has no module to catch it, and a decoder with the wrong width fails at the
first forward). `single` mode keeps its behaviour exactly: every explicit value is respected.

**Errors name the field.** Every check that goes beyond a type starts its message with the dotted
path Hydra and OmegaConf use, such as `model.latent.spaces.z_shared.fed_by`. Type errors (an
unknown key, a string for an int, a missing required value) already carry the path from OmegaConf
(`full_key`). The checks are in `validateModelConfig`: the section each mode needs and the other
one refused rather than ignored, unique names, every name in `fed_by` and `decoders_consume`
exists, a fusion exactly where several encoders feed a space, an assembler exactly where a decoder
consumes several, and no encoder, decoder or latent space left without a partner (a decoder that
consumes nothing never runs, an encoder that feeds nothing is never used). `validateModelConfig`
looks at the config only; registry names are resolved by `resolveModelConfig`, which raises
`KeyError` with the path of the field that holds the name. The routing graph's own rules
(`validateRoutingGraph`, ADR 0024) run too, and their errors are prefixed with `model.latent`.

**Three modules.** `config/model_schema.py` holds the dataclasses, `config/model_resolution.py` the
validation and the derivation of `GlobalVae.__init__`'s arguments (`ResolvedModel`), and
`config/model.py` the builder and the re-exports, so `from global_vae.config.model import
ModelConfig` keeps working. `buildModelFromConfig` builds a `RoutingGraph` for `several` and no
longer raises `NotImplementedError`. In `single` mode it now goes through `GlobalVae.__init__`
with the graph of `buildSingleLatentRoutingGraph`, which is what `createSingleLatent` does, so the
model and its state dict are the same.

**Shipped configs.** One model and one experiment file per pattern of spec §9:

| Pattern | Model | Experiment |
|---|---|---|
| Single latent space | `default.yaml` (signal + image, PoE), `signal_single_latent.yaml` | `signal_image_vae.yaml`, `signal_vae.yaml` |
| Several independent latent spaces | `signal_image_independent_latents.yaml` | `signal_image_independent_latents.yaml` |
| Shared plus private | `signal_image_shared_private.yaml` | `signal_image_shared_private.yaml` |

plus `signal_image_to_image.yaml` (the model of example 03), `configs/data/signal_image.yaml` (the
data contract of a signal plus image model) and `configs/training/several_latents.yaml` (training
defaults without a beta schedule, since the latent space names belong to the model; the experiment
files add one per space).

**The examples of spec §9 are rewritten to this schema.** The differences from the illustrative
text: `latent.mode` and `single_latent` are `latent_mode` and `single_latent`; `fusion: poe` is
`fusion: {strategy: poe}`; `head: linear` moves from the latent space to the edge, in `fed_by`; the
placeholder decoder `joint_decoder_v1` is a registered 2D decoder named `image_out`, so that the
example builds and trains as written; the `training` block is no longer shown inside `model`,
which it never was in the code.

## Alternatives considered

- **Extend `modalities` instead of adding sections.** Giving `ModalityConfig` an optional encoder
  or decoder would cover the translation model, but a shared decoder (spec §9, second pattern)
  would still need a modality name that is not a modality. Separate sections say what they mean
  and keep `modalities` untouched, so every shipped file stays valid.
- **A `kind` or `latent_mode` that infers itself from the sections present.** Less to write, but a
  stray `latent:` left in a `single` model would silently change what is built. The mode is
  explicit and the other section is refused.
- **A fixed list of `head` names on the latent space, as the spec showed.** It cannot say which
  encoder a head belongs to when several feed the space, which is D-1 point (a).
- **Typed `fed_by` (`list[str]`, with the heads in a second field).** It would keep OmegaConf
  validating the field, but splits one fact (which encoders feed this space, and through what)
  over two keys that have to agree. The list form is still the short way to write the common
  case.
- **A width formula per assembler in the config layer** (`concat` sums, `sum` and `average` take
  the common dim). It would hard-code assembler names, against spec §12, and every new assembler
  of P1-4 would need an edit here. Running the assembler on zeros costs one tiny forward pass.
- **Raising on a stale `beta_schedules` key.** It would catch typos, but `training/default.yaml`
  names `z_fused` and every single-latent experiment composes it, so a warning is the safe level.

## Consequences

- The `NotImplementedError` for `latent_mode: "several"` is gone. The second and third patterns of
  spec §9 load, validate, build and train through `scripts/train.py`, and the model of example 03
  is a YAML file with the same state dict as the Python one. This closes the roadmap deviation row
  for §9 and §11.
- Every model YAML shipped before this change loads and builds unchanged, with the same state
  dict. `tests/integration/test_model_config_schema.py` validates every `configs/model/*.yaml` and
  builds every shipped experiment config.
- A `single` model that sets both `single_latent` and `latent`, or a `several` model with both
  sections, used to be accepted (the unused section was ignored) and is now an error. No shipped
  file did it.
- Spec §11's first bullet is answered for the model section. The rest of that bullet, the
  `training` side (per-modality `reconstruction_loss` and `reconstruction_weight`), stays open as
  `TrainingConfig` documents.
- The `E1-*` rows (one shared encoder over several modalities) still cannot be written: they need
  roadmap D-2 and P1-5. A config can say what an encoder feeds, not what an encoder reads.
- An assembler that cannot be built from `kwargs`, or cannot run on zero tensors, is reported at
  resolution time with the path of its `decoders_consume` entry, before any module is built.
- The paired signal plus image data file and the `signal_image_vae.yaml` experiment exist now
  because the patterns above need them. Roadmap P3-3 still owns the image-only model configs and
  the review of `default.yaml`, and P3-2 the loader: pairing stays the caller's.
