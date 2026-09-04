# Examples

Runnable, self-contained walkthroughs of what this framework currently supports.
Distinct from `scripts/` (CLI entry points that expect *you* to supply a model/data
factory or a Hydra config) and `notebooks/` (interactive exploration): everything here
runs top-to-bottom with `python examples/<file>.py` and no external data, using
synthetic data generated in-memory, so a new contributor can run it immediately after
cloning the repository. `_synthetic_signal_data.py` is shared scaffolding the two
scripts below both import their data from (the leading underscore marks it as not
meant to be run directly, mirroring `tests/integration/_script_fixtures.py`'s own
convention); it is not a third example of its own.

## `01_signal_vae_pipeline.py`

The full spec §6.1 milestone 1 pipeline, end to end, on simple synthetic 1D signals
(this is the only configuration the framework fully supports today: single modality,
no fusion, no image encoder/decoder yet), built directly through the Python API (see
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
(spec §9, §10 "Config management") instead of hand-written Python kwargs: the exact
files `scripts/train.py` composes by default (`configs/experiment/signal_vae.yaml`),
with a synthetic, in-memory `loader_factory` (`_synthetic_signal_data`'s own
`buildSyntheticSignalDataloaders`) standing in for a real dataset. Also demonstrates
something the first example cannot show on its own: **versioned, comparable
experiment runs**. Two named variants of the shipped config are composed and trained
back to back, each expressed as nothing but a short list of Hydra dotlist overrides on
top of the same baseline: `"baseline"` (the config exactly as shipped) and `"tuned"`
(switches the regularizer to `free_bits_kl`, slows the beta warm-up, and monitors
`val/loss/reconstruction` for best-checkpoint selection, mirroring
`01_signal_vae_pipeline.py`'s own choices above). Each variant gets its own
`output_dir` (so its checkpoint, config snapshot, logs, and figures never collide with
the other's) and its own evaluation report; the two are compared side by side at the
end.

```bash
pip install -e ".[dev]"
python examples/02_config_driven_pipeline.py
```

Every output is written to `examples/outputs/02_config_driven_pipeline/<variant>/`
(git-ignored, same pattern as above). Trains two variants of 100 epochs each back to
back, so budget noticeably more time on CPU than `01_signal_vae_pipeline.py` alone.
