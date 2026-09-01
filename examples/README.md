# Examples

Runnable, self-contained walkthroughs of what this framework currently supports.
Distinct from `scripts/` (CLI entry points that expect *you* to supply a model/data
factory or a Hydra config) and `notebooks/` (interactive exploration): everything here
runs top-to-bottom with `python examples/<file>.py` and no external data, using
synthetic data generated in-memory, so a new contributor can run it immediately after
cloning the repository.

## `01_signal_vae_pipeline.py`

The full spec §6.1 milestone 1 pipeline, end to end, on simple synthetic 1D signals
(this is the only configuration the framework fully supports today: single modality,
no fusion, no image encoder/decoder yet). Deliberately generates each curve on its own
irregular grid, then uses `ResampleTransform(interpolation="scipy")` (spec §6.2) to
resample every curve onto one shared, common grid *by position*, not just by point
count, before the rest of the pipeline (log/standardize preprocessing, model assembly,
training, checkpointing, evaluation, and visualization) runs exactly as it would on
any fixed-length signal dataset.

```bash
pip install -e ".[dev]"   # or at least ".[interpolation]" for the scipy-backed step
python examples/01_signal_vae_pipeline.py
```

Every output (checkpoint, CSV metrics, evaluation report, and PNG figures) is written
to `examples/outputs/01_signal_vae_pipeline/` (git-ignored, matching `.gitignore`'s
`outputs/` pattern; re-running the script overwrites it). Runs in well under a minute
on CPU.
