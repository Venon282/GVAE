# 0017: 2D CNN encoder/decoder

**Status:** accepted
**Date:** 2026-09-16

## Context

Spec §6 names images as Phase 1's second modality ("Candidate encoders: CNN
(ResNet-style) or ViT, depending on resolution/dataset size"), but no image
encoder/decoder existed yet: `configs/model/default.yaml` references
`resnet_encoder_v1`/`resnet_decoder_v1` as a placeholder, and README's own "What's
deliberately not built yet" section listed "concrete image encoders/decoders" as a
known gap. The 1D signal modality already has this exact split, decided by ADR
0014: `OneDCnnEncoder`/`OneDCnnDecoder` (plain conv stack) and
`OneDCnnResidualEncoder`/`OneDCnnResidualDecoder` (residual/ResNet-style, spec §7).
The ask here was the direct 2D counterpart of the *plain* pair only, matching its
flexibility exactly; a residual 2D pair is the natural next step, mirroring ADR
0014 a second time, but is not built by this ADR.

Generalizing a per-stage hyperparameter from 1D to 2D is not a mechanical
find-and-replace: a 1D "kernel size" is always a scalar, so `tuple`/`list` are
interchangeable as the per-stage wrapper (`OneDCnnEncoder`'s own
`kernel_sizes: int | Sequence[int]`). A 2D kernel can itself be non-square
(`(height, width)`), which makes a bare `Sequence`-based broadcast genuinely
ambiguous: with 2 stages, `(3, 5)` could mean "one (3, 5) kernel every stage" or
"kernel 3 at stage 0, kernel 5 at stage 1", and nothing in the value itself
resolves which was meant. `utils/stage_config.py`'s own module docstring had
already anticipated this (it names a `broadcastPerStageShape` helper for exactly
this purpose) but never actually implemented it: `stage_config.py` shipped only
`broadcastPerStage` (scalar) and `resolveSpatialShape` (a single, not-yet-per-stage
shape). This ADR is also what actually builds the missing helper.

## Decision

- `encoders/TwoDCnnEncoder.py` (`2d_cnn_encoder_v1`) and `decoders/TwoDCnnDecoder.py`
  (`2d_cnn_decoder_v1`): direct 2D generalizations of `OneDCnnEncoder`/
  `OneDCnnDecoder`, matching every flexibility feature of the 1D pair rather than a
  cut-down version of it: per-stage kernel/stride/padding/dilation, per-stage
  pooling/activation/normalization (or disabled entirely), adaptive global pooling
  for size-agnostic encoding (any `(height, width)` at or above the architecture's
  own minimum), the custom-`nn.Module`-as-a-stage escape hatch on the encoder side,
  and the decoder's central guarantee: the *exact* output shape a configuration
  produces is verified at construction time, on both axes independently, raising
  `ValueError` on any mismatch instead of ever resizing its way there (both
  `conv_transpose` and `interpolate_conv` upsampling modes, output-padding
  auto-solved independently per axis when `conv_transpose` is the last transition).
- `utils/stage_config.py` gained the two helpers `broadcastPerStageShape` and
  `broadcastPerStageOptionalShape`, resolving the ambiguity above by giving `list`
  and `tuple` different, non-overlapping jobs: a `list` is always the per-stage
  wrapper (one entry per stage), a `tuple` is always a single, explicit,
  multi-dimensional shape shared by every stage. Split into two functions rather
  than one `Optional`-typed function: `list[...]` is invariant under `mypy
  --strict`, so a call site statically known to never pass `None` (`kernel_sizes`,
  `paddings`, ...) cannot satisfy a parameter typed to *possibly* contain `None`
  without a `cast`; `mypy --strict` caught this during review, it was not a
  stylistic choice from the start.
- `utils/conv_math.py` gained the 2D counterpart of every 1D shape-arithmetic
  function the decoder needs (`computeConv2dOutputShape`,
  `computeConvTranspose2dOutputShape`, `computeUpsampleThenConv2dOutputShape`,
  `solveConvTranspose2dOutputPadding`, `solveMinimumInputShapeForConv2d`,
  `computeUpsampleStack2dOutputShape`). None of these derive new math: `Conv2d`/
  `ConvTranspose2d`'s own PyTorch formulas are separable per axis (each axis's
  output length depends only on that same axis's own hyperparameters), so every
  2D function is its 1D counterpart applied once per axis, composed.
- `utils/builders.py` gained `build2DPoolLayer`/`build2DUpSampleStage`, mirroring
  `build1DPoolLayer`/`build1DUpSampleStage`.
- `AbstractEncoder.minimal_input_length` is a single `int`, a contract written
  before any 2D encoder existed. A 2D encoder's true minimum is a `(height,
  width)` pair that does not collapse into one number without losing information
  (both axes must independently clear their own bound). Rather than widen the
  base-class contract for this one implementation, `TwoDCnnEncoder` still
  implements `minimal_input_length` (returning `max(minimal_input_shape)`, a
  documented, conservative summary: the smallest side length a *square* input can
  have and still clear both axes) and adds a new, precise
  `minimal_input_shape -> tuple[int, int]` property alongside it for exact,
  per-axis use.
- Registered like every other pluggable strategy in this codebase: added to
  `encoders/__init__.py`/`decoders/__init__.py`'s registration imports.

## Consequences

- No behavior change to anything existing: `OneDCnnEncoder`, `OneDCnnDecoder`,
  `OneDCnnResidualEncoder`, `OneDCnnResidualDecoder`, and every existing
  `broadcastPerStage`/`resolveSpatialShape` call site are untouched; the two new
  `stage_config.py` functions and every 2D `conv_math.py`/`builders.py` function
  are pure additions.
- **Behavior difference callers must know about:** for any shape-like per-stage
  argument, a bare `tuple` no longer means "per-stage" the way it did (interchangeably
  with `list`) in the 1D encoders/decoders; it now always means "one shape shared
  by every stage". Code migrating 1D-style per-stage tuples (e.g.
  `strides=(1, 1, 2)`) to a 2D class must switch to a `list` (`strides=[1, 1, 2]`);
  the old tuple form still runs, but raises a clear, actionable `ValueError` rather
  than silently misinterpreting the value, since a mis-sized shared shape is
  rejected by `resolveSpatialShape`. Documented in `broadcastPerStageShape`'s own
  docstring and in both new classes' docstrings.
- `configs/model/default.yaml` still targets the future residual/ResNet-style pair
  (`resnet_encoder_v1`/`resnet_decoder_v1`) and remains unbuildable as-is; this ADR
  does not add a `configs/model/*.yaml` for the plain 2D CNN pair built here
  (mirroring `signal_single_latent.yaml`, e.g. `image_single_latent.yaml`) or wire
  it into a paired signal+image experiment file (spec §6.1 milestone 2, still
  gated on the open pairing-mechanism question, spec §11). A natural next step,
  not done here.
- `tests/integration/test_image_encoder.py`/`test_image_decoder.py` cover: shapes
  (square, non-square, multi-channel, implicit-channel 3D input), per-stage shape
  flexibility including the tuple-vs-list behavior difference above and its error
  path, pooling/activation/normalization configurability, minimum-input-shape
  solving (including a non-square minimum from non-square hyperparameters), the
  decoder's exact-shape verification and its per-axis output-padding auto-solve
  for a non-square gap, the custom-`nn.Module`-stage escape hatch, gradient flow,
  registration, and an encoder-decoder round trip (square RGB and rectangular
  grayscale) mirroring how `GlobalVae` wires a modality's encoder and decoder
  together.
- No PyTorch installation was available in the environment this was built in
  (network policy blocks the CPU-only wheel index, and the default PyPI wheel's
  bundled CUDA dependencies do not fit the available disk quota). The shape
  arithmetic above was independently verified (randomized and boundary tests in
  plain Python, plus a direct cross-check that the 2D minimum-input-shape solve
  reduces to `OneDCnnEncoder`'s own reference value on a symmetric configuration),
  and the full test suite was run against a minimal, shape-tracking `torch`/
  `torch.nn` stand-in driven by that same verified arithmetic, not against real
  PyTorch. Re-running the real suite in an environment with PyTorch installed is
  the recommended follow-up before merging.
- Residual 2D encoder/decoder (`2d_cnn_resnet_encoder_v1`/`2d_cnn_resnet_decoder_v1`,
  spec §7, mirroring `OneDCnnResidualEncoder`/`OneDCnnResidualDecoder` and ADR
  0014's own `Residual1DBlock`/`Residual1DUpBlock`) is deferred, not built here:
  this ADR is scoped to the plain CNN pair, matching what was actually asked for.
