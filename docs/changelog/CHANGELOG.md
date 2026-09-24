# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `encoders/TwoDCnnResidualEncoder.py` (`2d_cnn_resnet_encoder_v1`) and
  `decoders/TwoDCnnResidualDecoder.py` (`2d_cnn_resnet_decoder_v1`): the 2D residual
  ("ResNet-style") encoder/decoder pair for spec §6's image modality (spec §7), the 2D
  generalization of `OneDCnnResidualEncoder`/`OneDCnnResidualDecoder`, generalized the same
  way ADR 0017 generalized the plain pair (`int` / `tuple[int, int]` / `list` convention for
  every shape-like hyperparameter, per-stage `block_depths` and `shortcut_kernel_sizes`, exact
  per-axis output-shape verification instead of resizing). Built on `Residual2DBlock` /
  `Residual2DUpBlock` (`utils/conv_blocks.py`), whose shortcut is verified at construction time
  to reach the main path's shape for every input shape, independently on both axes, through the
  new `computeConv2dLengthOffset` / `computeConvTranspose2dLengthOffset`
  (`utils/conv_math.py`). See `docs/adr/0018-2d-residual-encoder-decoder.md`.
- `tests/integration/test_residual_image_encoder.py` and
  `tests/integration/test_residual_image_decoder.py` for the pair above.
- `tests/integration/test_conv_math.py` (new): unit tests for `utils/conv_math.py`, which had
  none. Every 1D and 2D formula is cross-checked against the shape a real `nn.Conv1d`/`Conv2d`/
  `ConvTranspose1d`/`ConvTranspose2d`/`MaxPool`/`Upsample` stack actually produces, over sweeps
  that include even kernels, strides and dilations above 1, and non-square configurations where
  the two axes disagree (so an axis mix-up cannot pass by symmetry). The length-offset helpers
  are checked against their defining property (equal offsets imply equal output shapes for every
  input length, the guarantee a residual shortcut relies on), and the solvers
  (`solveConvTranspose*OutputPadding`, `solveMinimumInput*`) against exact-minimum properties.
- `tests/integration/test_conv_blocks.py` gained `TestResidual2DBlock` and
  `TestResidual2DUpBlock` (it only covered the 1D blocks): shapes with non-square kernels and
  strides, flexible depths, identity vs projection shortcut, both upsample modes, gradient flow,
  the decoder-only output-suppression flags, every error path (including a shortcut mismatch on a
  *single* axis, which a symmetric test could not detect), and a sweep over odd and even input
  shapes on each axis independently for the "every input shape, not one example shape" guarantee.
- `examples/03_signal_image_to_image.py` and `tests/integration/test_signal_image_example.py`:
  the multimodal example, `(signal, image) -> image` where the output image is not the input
  image. Two encoders (1D and 2D residual) feed one latent space through PoE fusion and a single
  decoder produces the clean target, on synthetic data built so that neither input suffices
  alone (the 1D signal is blind to the vertical position, the degraded input image is noisy and
  partly erased). Reports, per input subset, what the decoder produces from each input alone and
  from both (ADR 0016), next to the "return the degraded input" baseline (typical test-set R^2:
  about 0.33 from the signal, 0.57 from the image, 0.76 from both), and draws a translation grid
  figure (the framework's cross-modal matrix plot handles 1D series only). The model is built
  from an explicit `RoutingGraph`, since `createSingleLatent` ties decoder names to encoder
  names. `Trainer` uses the whole batch as encoder input *and* as reconstruction target, and
  `GlobalVae.forward` raises `KeyError` on a key with no encoder, so the example defines a small
  `TranslationTrainer` (one overridden method, plus validation without modality dropout); a test
  documents why it is needed and will fail if `Trainer` ever no longer requires it.
- `tests/integration/test_docs_adr_navigation.py`: fails when an ADR file is missing from
  `docs/adr/index.md` or `docs/adr/SUMMARY.md`, when ADR numbers have a gap or a duplicate, when
  a listed link is dead, when an index row is a placeholder, when `mkdocs.yml` stops using the
  directory form for the ADR and changelog nav entries, and (if the `docs` extra is installed)
  when the *built* sidebar does not list every ADR.
- `docs/adr/SUMMARY.md`: one sidebar entry per ADR (see "Fixed").

### Changed

- `docs/adr/index.md` now lists ADRs 0017 and 0018, which existed but were missing. Its rows for
  0015 and 0016 had been committed as "*Guess:*" placeholders written without reading the ADRs;
  they now describe the ADRs as written.
- `utils/conv_math.py` gained a module docstring (it had none) and two single-line summaries
  where a wrapped one violated `D205`. No formula changed.
- `examples/README.md` and `docs/getting-started.md` describe example 03; the README no longer
  calls the single-modality pipeline "the only configuration the framework fully supports".
- `TwoDCnnResidualEncoder.py`, `test_residual_image_encoder.py` and
  `test_residual_image_decoder.py` formatted with `ruff format`.

### Fixed

- The ADRs (and the per-version changelog pages) were not visible in the site sidebar.
  `mkdocs.yml` pointed the nav at a file (`adr/index.md`, `changelog/SUMMARY.md`), and
  `literate-nav` only expands a `SUMMARY.md` when the nav entry points at its *directory*: the
  ADR section showed a single page, and the changelog entry rendered `SUMMARY.md` as one ordinary
  page titled "SUMMARY". Both entries now use the directory form (`adr/`, `changelog/`); all 18
  ADRs and the 3 changelog pages appear in the built sidebar, and `mkdocs build --strict` passes.

**Verification note.** The whole test suite was run against real PyTorch 2.14 (CPU) for the first time for the 2D
pair, which ADRs 0017 and 0018 record as only verified against a shape-tracking stand-in:
every test passes, including the gradient-flow and unconstrained-output tests the stand-in
could not run, apart from `test_latent_plot.py::test_umap_projects_to_the_requested_dimensionality`
(needs the optional `umap-learn`, absent from the environment). `mypy --strict` reports no
issue on the package. The new tests were mutation-checked: deliberately breaking a formula,
a per-axis check, or a documented behavior makes them fail.
