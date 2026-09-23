# 0018: 2D residual encoder/decoder

**Status:** accepted
**Date:** 2026-09-20

## Context

`docs/adr/0017-2d-cnn-encoder-decoder.md` built the plain (non-residual) 2D CNN
pair (`TwoDCnnEncoder`/`TwoDCnnDecoder`, spec §6's image modality) and explicitly
named its residual counterpart as the deferred next step: "Residual 2D
encoder/decoder (`2d_cnn_resnet_encoder_v1`/`2d_cnn_resnet_decoder_v1`, spec §7,
mirroring `OneDCnnResidualEncoder`/`OneDCnnResidualDecoder` and ADR 0014's own
`Residual1DBlock`/`Residual1DUpBlock`) is deferred, not built here." This ADR is
that deferred item.

The 1D signal modality already has both a plain pair
(`OneDCnnEncoder`/`OneDCnnDecoder`) and a residual pair
(`OneDCnnResidualEncoder`/`OneDCnnResidualDecoder`, ADR 0014, built from
`utils/conv_blocks.py`'s `Residual1DBlock`/`Residual1DUpBlock`). The image
modality, after ADR 0017, only had the plain pair. This ADR closes that gap the
same way ADR 0017 closed the 1D-vs-2D gap for the plain pair: not a new
mechanism, a direct 2D generalization of an existing one, applied per axis via
this codebase's now-established `broadcastPerStageShape`/`list`-vs-`tuple`
convention (ADR 0017) and `computeConv1dLengthOffset`-style per-axis shortcut
verification (ADR 0014).

## Decision

- `utils/conv_blocks.py` gained `Residual2DBlock` (encoder side) and
  `Residual2DUpBlock` (decoder side), direct 2D generalizations of
  `Residual1DBlock`/`Residual1DUpBlock`: every shape-like constructor argument
  (`kernel_size`, `stride`, `padding`, `dilation`, `shortcut_kernel_size`) is an
  already-resolved `(height, width)` tuple (resolved upstream by
  `TwoDCnnResidualEncoder`/`TwoDCnnResidualDecoder` via
  `broadcastPerStageShape`, exactly like every other 2D building block in this
  codebase), and every length-changing computation is the 1D block's own
  arithmetic applied once per axis, since the two spatial axes of a
  `Conv2d`/`ConvTranspose2d` never interact under PyTorch's own formulas (the
  same reasoning `utils/conv_math.py`'s and `utils/builders.py`'s own 2D
  sections already document). A residual shortcut's shape is verified,
  independently on both axes, to match its main path's for every input shape,
  not merely for one example shape, at construction time; a mismatch raises
  `ValueError` naming both offsets, exactly mirroring `Residual1DBlock`'s own
  "verify, don't silently resize" convention.
- `utils/conv_math.py` gained `computeConv2dLengthOffset` and
  `computeConvTranspose2dLengthOffset`, the 2D counterparts of
  `computeConv1dLengthOffset`/`computeConvTranspose1dLengthOffset`, each simply
  the 1D offset formula applied once per axis. No new arithmetic is derived;
  these compose functions this codebase already has and already trusts (the
  same "every 2D function is its 1D counterpart applied per axis, nothing
  re-derived" principle ADR 0017 already established for
  `computeConv2dOutputShape` and friends).
- `encoders/TwoDCnnResidualEncoder.py` (`2d_cnn_resnet_encoder_v1`) and
  `decoders/TwoDCnnResidualDecoder.py` (`2d_cnn_resnet_decoder_v1`): the 2D
  generalization of `OneDCnnResidualEncoder`/`OneDCnnResidualDecoder`,
  generalized the exact same way ADR 0017 generalized the plain pair. As
  permissive as `TwoDCnnEncoder`/`TwoDCnnDecoder` along every axis those
  classes already vary per stage/transition (kernel/stride/padding/dilation,
  each `int` (square, shared) / `tuple[int, int]` (non-square, shared) / `list`
  (per-stage, ADR 0017's own convention), pooling/activation/normalization,
  both decoder upsample modes, exact per-axis output-shape verification instead
  of resizing), plus the same two additions `OneDCnnResidualEncoder`/
  `OneDCnnResidualDecoder` already add over the plain 1D pair: `block_depths`
  (per stage/transition, e.g. `(3, 4)`) and `shortcut_kernel_sizes` (itself
  possibly non-square, broadcastable the same way).
  - `TwoDCnnResidualEncoder` narrows `hidden_channels` to a plain
    `tuple[int, ...]`, not `TwoDCnnEncoder`'s `tuple[int | nn.Module, ...]`
    custom-stage escape hatch, mirroring the identical, already-documented
    narrowing `OneDCnnResidualEncoder` makes relative to `OneDCnnEncoder` (a
    residual block is a coupled main-path/shortcut pair that does not compose
    with an opaque custom stage the way a single plain conv stage does).
  - `TwoDCnnResidualDecoder`'s default `kernel_sizes` is `3`, not
    `TwoDCnnDecoder`'s `4`, mirroring the identical, already-documented
    difference `OneDCnnResidualDecoder` makes relative to `OneDCnnDecoder`:
    this class's residual blocks require an odd `kernel_size` on both axes
    whenever `block_depths > 1` (the default, `2`), and `kernel_sizes=3` is
    both the only value the default `block_depths` would accept and the
    self-consistent pairing with the default `upsample_modes="interpolate_conv"`
    for an exact shape doubling on both axes out of the box (`seed_shape=(8, 8)`,
    `hidden_channels=(128, 64, 32)` reaches `(64, 64)`, the same value
    `TwoDCnnDecoder` reaches with its own, differently-tuned defaults).
  - Both classes' `computeMinimumInputShape`/`computeOutputShape` static
    methods mirror `TwoDCnnEncoder`/`TwoDCnnDecoder`'s own private
    `_computeMinimumInputShapeFromResolved`-style split (a private,
    already-resolved-tuples core, delegated to both by the public static
    method after broadcasting and by `__init__` directly): re-broadcasting an
    already-resolved `tuple[tuple[int, int], ...]` through
    `broadcastPerStageShape` would misread it as one wrongly-sized shared
    shape, exactly the `mypy --strict`-driven reason ADR 0017 already
    documents. `computeMinimumInputShape` additionally accounts for
    `block_depths`, generalizing `OneDCnnResidualEncoder.
    computeMinimumInputLength`'s identical inner loop to 2D;
    `computeOutputShape` does not take `block_depths`/`shortcut_kernel_sizes`
    at all, for the same reason `OneDCnnResidualDecoder.computeOutputLength`
    doesn't: only a transition's first layer ever changes shape.
- Registered like every other pluggable strategy in this codebase: added to
  `encoders/__init__.py`/`decoders/__init__.py`'s registration imports.

## Consequences

- No behavior change to anything existing: `TwoDCnnEncoder`, `TwoDCnnDecoder`,
  `OneDCnnResidualEncoder`, `OneDCnnResidualDecoder`, `Residual1DBlock`,
  `Residual1DUpBlock`, and every existing `conv_math.py`/`builders.py`/
  `stage_config.py` function are untouched; every addition here is either a new
  function/class or a new file.
- Same `list`-means-per-stage / `tuple`-means-shared behavior difference from
  the 1D residual pair that ADR 0017 already documents and tests for the plain
  2D pair: migrating a 1D-style per-stage tuple (e.g. `strides=(1, 1, 2)`) to
  either new 2D class requires switching to a `list` (`strides=[1, 1, 2]`); the
  old tuple form raises a clear, actionable `ValueError` rather than silently
  misinterpreting the value.
- `tests/integration/test_residual_image_encoder.py`/
  `test_residual_image_decoder.py` cover: shapes (square, non-square,
  multi-channel, implicit-channel, resolution-agnostic), the `tuple`-vs-`list`
  convention and its error path, `block_depths` flexibility (including that a
  deeper block with odd kernels never raises the architecture's own minimum
  input shape), `shortcut_kernel_sizes` per stage/transition,
  pooling/activation/normalization configurability, minimum-input-shape
  solving (including a non-square minimum from non-square hyperparameters),
  both decoder upsample modes, the decoder's per-axis output-padding auto-solve
  (including a gap on only one axis, which must not perturb the other),
  non-square seed/output shapes end to end, the last-transition
  unconstrained-output guarantee, internal-layer normalization in a deep
  last-stage block, gradient flow, registration, and every documented error
  path (non-positive/even-kernel `block_depths`, a shortcut that cannot reach
  the main path's shape, an unreachable `output_shape`, an unrecognized
  `upsample_mode`). An encoder-decoder round trip test (square RGB and
  rectangular grayscale) mirrors how `GlobalVae` wires a modality's encoder and
  decoder together, matching `test_image_decoder.py`'s own equivalent.
- No PyTorch installation was available in the environment this was built in,
  for the same reasons ADR 0017 already records (network policy blocks the
  wheel host; the default index's CUDA-bundled build exceeds the available
  disk quota). The same mitigation ADR 0017 used was used again here: the
  shape arithmetic was verified directly (the new offset functions were
  cross-checked against their already-correct 1D counterparts, and the
  residual decoder's default-doubling claim was hand-derived and confirmed),
  and the full new test suite (`test_residual_image_encoder.py`/
  `test_residual_image_decoder.py`, minus the handful of assertions that
  inherently need real autograd or real numeric values: gradient-flow checks
  and the "unconstrained values can be negative" check) was run and passed
  against a minimal, shape-tracking `torch`/`torch.nn` stand-in driven by that
  same verified arithmetic, not against real PyTorch. Re-running the real
  suite in an environment with PyTorch installed is the recommended follow-up
  before merging, exactly as ADR 0017 already recommends for the plain pair.
- Not done here, as natural next steps (mirroring ADR 0017's own list of what
  it left out): a `configs/model/*.yaml` for this pair (e.g.
  `image_resnet_single_latent.yaml`, mirroring
  `signal_resnet_single_latent.yaml`), and wiring it into a paired
  signal+image experiment file (spec §6.1 milestone 2, still gated on the open
  pairing-mechanism question, spec §11).
