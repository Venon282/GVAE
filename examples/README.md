# Examples

Runnable, self-contained walkthroughs of what this framework currently supports.
Distinct from `scripts/` (CLI entry points that expect *you* to supply a model/data
factory or a Hydra config) and `notebooks/` (interactive exploration): everything here
runs top-to-bottom with `python examples/<file>.py` and no external data, using
synthetic data generated in-memory, so a new contributor can run it immediately after
cloning the repository. `_synthetic_signal_data.py` is shared scaffolding the first
two scripts below both import their data from (the leading underscore marks it as not
meant to be run directly, mirroring `tests/integration/_script_fixtures.py`'s own
convention); it is not an example of its own. `03_signal_image_to_image.py` generates
its own data and needs no shared helper.

## `01_signal_vae_pipeline.py`

The full spec §6.1 milestone 1 pipeline, end to end, on simple synthetic 1D signals
(the single-modality case: no fusion; see `03_signal_image_to_image.py` below for two
modalities fused into one latent space), built directly through the Python API (see
`02_config_driven_pipeline.py` below for the same pipeline built from `configs/` YAML
instead). Deliberately generates each curve on its own irregular grid, then uses
`ResampleTransform(interpolation="scipy")` (spec §6.2) to resample every curve onto one
shared, common grid *by position*, not just by point count, before the rest of the
pipeline (log/standardize preprocessing, model assembly, training, checkpointing,
evaluation, and visualization) runs exactly as it would on any fixed-length signal
dataset. Regularizes the single latent space with `free_bits_kl` rather than the
framework's own default, `kl_standard_normal`: an earlier version of this example used
the default and collapsed the posterior (test-set R^2 stuck around 0.3 regardless of
model capacity); the script's own module docstring ("On regularization") documents
that lesson, and `BestCheckpointCallback` is configured to monitor
`val/loss/reconstruction` accordingly (see `CHANGELOG.md` for the full reasoning).

```bash
pip install -e ".[dev]"   # or at least ".[interpolation]" for the scipy-backed step
python examples/01_signal_vae_pipeline.py
```

Every output (checkpoint, CSV metrics, evaluation report, and PNG figures) is written
to `examples/outputs/01_signal_vae_pipeline/` (git-ignored, matching `.gitignore`'s
`outputs/` pattern; re-running the script overwrites it). Runs in well under a minute
on CPU.

## `02_config_driven_pipeline.py`

The same pipeline as above, but assembled entirely from the `configs/` YAML files
(spec §9, §10 "Config management") instead of hand-written Python kwargs, with a
synthetic, in-memory `loader_factory` (`_synthetic_signal_data`'s own
`buildSyntheticSignalDataloaders`) standing in for a real dataset. Also demonstrates
something the first example cannot show on its own: **versioned, comparable
experiment runs**. Two named variants of the composed config are trained back to
back, each expressed as nothing but a short list of Hydra dotlist overrides on top of
the same base config: `"baseline"` (no overrides) and `"tuned"` (switches the
regularizer to `free_bits_kl`, slows the beta warm-up, and monitors
`val/loss/reconstruction` for best-checkpoint selection, mirroring
`01_signal_vae_pipeline.py`'s own choices above). Each variant gets its own
`output_dir` (so its checkpoint, config snapshot, logs, and figures never collide with
the other's) and its own evaluation report; every selected variant is compared side by
side at the end.

By default this composes `configs/experiment/signal_resnet_vae.yaml`, the residual
("ResNet-style") 1D encoder/decoder (spec §7, `docs/adr/0014-residual-1d-encoder-
decoder.md`) instead of the plain conv one. **Choosing which configs to use, and a
handful of other important parameters, is itself part of what this script
demonstrates**, via a real CLI:

```bash
pip install -e ".[dev]"
python examples/02_config_driven_pipeline.py                      # default: residual encoder/decoder
python examples/02_config_driven_pipeline.py --model-config signal_single_latent  # compare against the plain conv one
python examples/02_config_driven_pipeline.py --experiment-config experiment/signal_vae  # a whole different experiment file
python examples/02_config_driven_pipeline.py --variants baseline --num-epochs 5   # quick one-variant smoke check
python examples/02_config_driven_pipeline.py --override training.optimizer.kwargs.lr=0.01  # repeatable, arbitrary Hydra overrides
python examples/02_config_driven_pipeline.py --help                # every option
```

Every output is written to `<output-root>/<variant>/` (default
`examples/outputs/02_config_driven_pipeline/<variant>/`, git-ignored, same pattern as
above). Two variants of 100 epochs each (this script's own defaults) take noticeably
longer on CPU than `01_signal_vae_pipeline.py` alone; use `--variants`/`--num-epochs`
for a faster check.

## `03_signal_image_to_image.py`

The multimodal case: **`(signal, image) -> image`, where the output image is not the
input image** (spec §2.1 `EN-L1-DN`, §4 fusion, §5 missing-modality robustness, §6
image modality). Two encoders (`1d_cnn_resnet_encoder_v1` for the 1D signal,
`2d_cnn_resnet_encoder_v1` for the image, the 2D pair of
`docs/adr/0018-2d-residual-encoder-decoder.md`) feed one latent space through
Product-of-Experts fusion, and a single decoder (`2d_cnn_resnet_decoder_v1`) produces
the target image.

The synthetic task is built so that neither input is enough alone: each sample is a
Gaussian blob, the `signal` is its horizontal profile (blind to its vertical
position), and `image_in` is a heavily noised, partly erased picture of it (carrying
the vertical position, unreliably). The target `image_out` is the clean picture.
Modality dropout (spec §5) hides either input at random during training, so the one
fused latent learns to work from either alone. The script then reports, per input
subset, what the decoder produces from `signal` alone, `image_in` alone, and both
(`collectCrossModalReconstructions`, `docs/adr/0016-cross-modal-reconstruction-reporting.md`),
next to the trivial baseline of returning the degraded input. Typical test-set R^2 with
the defaults: about 0.33 from the signal alone, 0.57 from the image alone, **0.76 from
both**, and strongly negative for the baseline.

Two things it shows that the other examples cannot:

- **A decoder target that is not an encoder input.** `Trainer` uses the batch both as
  encoder input and as reconstruction target, and `GlobalVae.forward` rejects a key
  without an encoder, so training on a separate `image_out` needs a small
  `TranslationTrainer` subclass (defined in the script; see its docstring). The model
  itself is built from an explicit `RoutingGraph` rather than `createSingleLatent`,
  which ties decoder names to encoder names.
- **The figure.** `translation_grid.png` lines up, for several test samples, the signal,
  the degraded input, the target, and the output from each input subset. The
  signal-only output is a blob smeared along y (right x, unknown y), the image-only
  output can put the blob in the wrong place when it is occluded, and the fused output
  gets both right. It is drawn by the script itself: the framework's cross-modal
  matrix plot handles 1D series only.

```bash
pip install -e ".[dev]"
python examples/03_signal_image_to_image.py
python examples/03_signal_image_to_image.py --num-epochs 5 --num-train 256   # quick check
python examples/03_signal_image_to_image.py --help                            # every option
```

Every output (best checkpoint, CSV metrics, `metrics_by_input_subset.json`,
`translation_grid.png`, `loss_curves.png`) is written to
`examples/outputs/03_signal_image_to_image/` (git-ignored, same pattern as above). The
defaults (2048 training samples, 40 epochs) take a few minutes on CPU.
