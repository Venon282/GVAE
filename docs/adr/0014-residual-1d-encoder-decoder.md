# 0014 — Residual (ResNet-style) 1D encoder and decoder

**Status:** accepted
**Date:** 2026-09-06

## Context

Spec §7 explicitly keeps the door open for "scaling toward larger backbones", and the
existing `OneDCnnEncoder`/`OneDCnnDecoder` (spec §6.1 milestone 1's own modules) are
plain conv stacks: as they get deeper, they inherit the same optimization difficulty
any deep plain conv stack does, without a way to add depth cheaply. Residual
("ResNet-style", He et al., 2015) connections are the standard mitigation, and were
requested as a variant a user could opt into per modality, without touching the
existing, already-tested milestone-1 modules.

Two materially different kinds of "skip connection" were considered:

1. **Internal to one encoder or decoder** (a residual block's own shortcut, from a
   stage's input to that same stage's output). This is a purely additive
   architectural variant: it changes nothing about `AbstractEncoder`/
   `AbstractDecoder`'s interfaces, `GlobalVae`, Fusion, or the Assembler, and composes
   with the existing registry pattern exactly like any other encoder/decoder choice
   (spec §10's "new-modality checklist": subclass, register, add a config entry, add a
   test, zero changes to the core).
2. **U-Net-style, encoder-to-decoder** (a decoder consuming an earlier encoder stage's
   features directly, bypassing the latent bottleneck). This was considered and
   rejected for this framework: it would require `AbstractEncoder`/`AbstractDecoder`
   to change shape (an encoder would need to expose intermediate features, not just
   `(mu, logvar)`), would break the encoder/decoder/fusion/assembler interchangeability
   the routing graph (§2.2) depends on (a decoder would no longer be usable with any
   encoder, only one built to match it feature-for-feature), and — the more fundamental
   problem — gives the decoder a deterministic, unregularized path around `z`, which
   is exactly what posterior collapse via skip paths looks like: the model can satisfy
   reconstruction through the skip path alone and stop using `z`, defeating both
   latent-space visualization (spec §6.1 milestone 1) and, more seriously, generation
   from the prior (`z ~ N(0, I)` has no encoder features to skip from at all, so a
   decoder trained to depend on them cannot generate). This ADR implements only (1);
   a hierarchical, per-depth *stochastic* latent (matching each stage's features with
   its own regularized `(mu, logvar)`, combined via an `Assembler`, spec §2.2/§2.3)
   would be the framework-consistent way to reach for something like (2), but that
   depends on `AbstractLatentHead`/encoder fan-out (mentioned in spec §3/§9, not yet
   implemented, see `docs/adr/0002-generalize-global-vae-to-routing-graph.md`'s own
   note on this) and is out of scope here.

## Decision

### `utils/conv_blocks.py`: two shared, plain-`nn.Module` building blocks

`Residual1DBlock` (encoder side) and `Residual1DUpBlock` (decoder side), neither
registered on their own (they are building blocks, not modalities). Each stacks
`depth` conv layers (`depth >= 1`, independently configurable, e.g. `3` layers before
one stage's shortcut and `4` for a deeper stage) plus one shortcut per block, `y =
activation(F(x) + shortcut(x))`. Only a block's first layer may change channel width
and/or length (a stride for the encoder block, an upsampling transition — reusing
`utils.builders.build1DUpSampleStage`, both `"conv_transpose"`/`"interpolate_conv"`
modes — for the decoder block); every later layer keeps both fixed. This is what
makes the block's contribution to the surrounding encoder's/decoder's overall length
bookkeeping identical to a plain, single-layer stage of the same
stride/kernel_size/padding/dilation, letting `OneDCnnResidualEncoder`/
`OneDCnnResidualDecoder` reuse the exact same length-solving machinery
(`utils/conv_math.py`) the non-residual classes already use, unmodified.

The shortcut's hyperparameters are resolved, and *verified at construction time*, to
reach the exact same output length as the main path for every input length, not
merely for one example length (two stride-1 `Conv1d` layers, or two `ConvTranspose1d`
layers at the same stride, produce identical output lengths for every input length if
and only if their length "offsets" match; two new pure functions,
`computeConv1dLengthOffset`/`computeConvTranspose1dLengthOffset`, added to
`utils/conv_math.py`, make this an exact, checkable condition rather than a spot
check). A mismatch raises `ValueError` immediately, naming both offsets and how to fix
them (adjust padding, or `shortcut_kernel_size`), following this codebase's existing
"verify a configuration instead of silently producing a shape mismatch" convention
(`OneDCnnDecoder`'s own exact-output-length verification).

A block's internal (non-first) layers must exactly preserve length, which is only
achievable with an odd `kernel_size` whenever `depth > 1` (no integer padding makes a
stride-1, even-kernel layer exactly length-preserving); construction raises a clear,
named `ValueError` if violated. `depth=1` has no internal layer and is unaffected.

`Residual1DUpBlock` additionally exposes `apply_output_normalization`/
`apply_output_activation` (default `True`), letting a decoder's very last transition
suppress only its own final layer's normalization/activation (matching
`OneDCnnDecoder`'s convention that the last transition must produce unconstrained
reconstruction values) without disabling normalization/activation on that same
transition's earlier internal layers, so a deep last-stage block still benefits from
them.

### `OneDCnnResidualEncoder`/`OneDCnnResidualDecoder`: registered, as permissive as their non-residual counterparts

`encoders/OneDCnnResidualEncoder.py` (`1d_cnn_resnet_encoder_v1`) and
`decoders/OneDCnnResidualDecoder.py` (`1d_cnn_resnet_decoder_v1`) mirror
`OneDCnnEncoder`/`OneDCnnDecoder` along every axis those classes already vary per
stage/transition (`kernel_sizes`, `strides`, `paddings`, `dilations`, `poolings` and
its own sub-parameters, `activations`, `normalizations`, `output_paddings`,
`upsample_modes`, all via the existing `utils.stage_config.broadcastPerStage`), plus
two additions: `block_depths` (the feature actually requested: each stage's own
residual block depth, per stage or shared) and `shortcut_kernel_sizes`.

One deliberate, documented scope narrowing: unlike `OneDCnnEncoder`'s own
`hidden_channels: tuple[int | nn.Module, ...]` escape hatch (dropping in an arbitrary
custom stage), `OneDCnnResidualEncoder`'s `hidden_channels` is always `tuple[int, ...]`.
A residual block is a coupled main-path/shortcut pair whose shortcut's shape must be
provably tied to the main path's; an opaque, arbitrary `nn.Module` standing in for one
stage does not compose with that requirement without either silently skipping the
residual connection for that stage or requiring the caller's module to expose enough
structure for a shortcut to be derived from it. A caller wanting a fully custom stage
inside an otherwise-residual architecture can still compose one directly outside this
class.

One deliberate default difference from `OneDCnnDecoder`: `kernel_sizes` defaults to
`3` here, not `4`. `OneDCnnDecoder`'s own docstring already notes its `kernel_sizes=4`
default is tuned for `upsample_modes="conv_transpose"` and does not reach an exact
length doubling under that class's own default mode, `"interpolate_conv"`
(`kernel_sizes=3, paddings=1` is the pairing that does, per that same docstring).
Since `OneDCnnResidualDecoder` additionally *requires* an odd `kernel_size` whenever a
transition's `block_depths` is greater than `1` (the default, `2`), `kernel_sizes=3`
is both self-consistent with this class's own default `upsample_mode` and the only
choice its own default `block_depths` would accept anyway; this class's bare defaults
(unlike `OneDCnnDecoder`'s own bare defaults) therefore reach an exact length
doubling per transition out of the box.

### Shared refactor: `computeUpsampleStackOutputLength` relocated to `utils/conv_math.py`

`OneDCnnDecoder.py`'s private `_computeLengthFromResolved` (chaining the
per-transition upsample length formula across a resolved stack) is exactly what
`OneDCnnResidualDecoder` also needs, since only a transition's first layer ever changes
length either way. Relocated, unchanged, to `utils/conv_math.py` as the public
`computeUpsampleStackOutputLength`, with `OneDCnnDecoder.py` updated to import it
(`as _computeLengthFromResolved`, preserving its own internal name and every existing
call site unchanged). Pure relocation: `tests/integration/test_signal_decoder.py`'s
full existing suite passes unchanged, verifying no behavior moved with it.

## Testing

- `tests/integration/test_conv_blocks.py`: `Residual1DBlock`/`Residual1DUpBlock`
  directly — shapes across depth/stride/projection combinations, the flexible-depth
  case itself (spec's own "3 layers... then 4" example), gradient flow, both decoder
  upsample modes, the `apply_output_normalization`/`apply_output_activation` flags
  (confirming only the final layer is suppressed, internal layers of a deep block are
  not, and unconstrained/negative values survive), and every documented error path
  (`depth < 1`, even `kernel_size` with `depth > 1`, an offset-mismatched shortcut in
  both `Residual1DBlock` and each of `Residual1DUpBlock`'s two upsample modes).
- `tests/integration/test_residual_signal_encoder.py` /
  `test_residual_signal_decoder.py`: mirror `test_signal_encoder.py`/
  `test_signal_decoder.py`'s own coverage (shapes, variable length, gradients,
  registration, per-stage configurability, minimum/exact length solving, error paths)
  for the new classes, plus their own additions (`block_depths`,
  `shortcut_kernel_sizes`, and — decoder only — the last-transition
  unconstrained-output guarantee and its interaction with a deep last-stage block).
- Full existing suite (`pytest tests/`) passes unchanged (496 tests total after this
  change), confirming the `OneDCnnDecoder.py` relocation is behavior-preserving.

## Consequences

- A user can opt a modality into a ResNet-style 1D encoder/decoder, with a
  per-stage-flexible residual block depth, purely by registry name
  (`1d_cnn_resnet_encoder_v1`/`1d_cnn_resnet_decoder_v1`), with zero changes to `GlobalVae`,
  Fusion, the Assembler, or the routing graph — exactly spec §10's extension
  contract.
- U-Net-style encoder-to-decoder skip connections remain explicitly out of scope,
  for the architectural reason given above (posterior collapse via a deterministic
  skip path, and loss of generation from the prior), not merely deferred; a
  hierarchical *stochastic* latent is the framework-consistent path there instead,
  and depends on the not-yet-built `AbstractLatentHead`/encoder-fan-out mechanism.
- `utils/conv_math.py` gains four small, pure, well-tested length-arithmetic helpers
  (`computeConv1dLengthOffset`, `isLengthPreservingConv1d`,
  `computeConvTranspose1dLengthOffset`, `computeUpsampleStackOutputLength`), reusable
  by any future conv-based building block that needs the same shortcut/shape
  guarantees this one does.
- `ruff check`, `ruff format --check`, and `mypy --strict` (checked at
  `--python-version 3.12` locally due to an unrelated numpy-stub/mypy version
  mismatch in this environment; see the accompanying summary) pass clean on every
  new/changed file, with the sole exception of two pre-existing, unrelated
  `N999 invalid-module-name` warnings on `OneDCnnEncoder.py`/`OneDCnnDecoder.py`-style
  `CamelCase.py` filenames (this project's own established naming convention, present
  before this change and also present on the two new files for the same reason).
