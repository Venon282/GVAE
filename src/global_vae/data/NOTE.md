# Status

`transforms/` is a small library of generic, invertible data transforms
(`AbstractTransform` + registry, mirroring every other pluggable strategy in
this codebase): `LogTransform` (`log`), `StandardizeTransform`
(`standardize`), `ResampleTransform` (`resample`), plus `ComposeTransform`
for chaining several into one invertible pipeline. Every transform here
operates on a tensor of *any* shape/dimensionality (elementwise, or via an
explicit `num_spatial_dims`/mean-std broadcast parameter): none of them is
written for, or aware of, one specific modality or dataset (e.g. SAXS). See
`docs/adr/0012-generic-data-transforms.md` and spec §6.2.

`ResampleTransform` additionally supports coordinate-aware resampling
(`interpolation="scipy"`, spec §6.2, `docs/adr/0013-coordinate-aware-resampling.md`):
explicit `source_coords`/`target_coords` (shared, or per-call for samples whose
own positions differ from one another), and a choice of interpolation method
beyond `torch.nn.functional.interpolate`'s evenly-spaced-grid modes
(`scipy.interpolate`'s `interp1d` kinds plus `cubic_spline`/`pchip`/`akima`
splines). `scipy` is a soft dependency of this mode only; the default
`interpolation="torch"` mode needs nothing beyond torch.

`DataConfig.transforms` (`config/data.py`) is a list of these transforms by
registry name; `buildTransformPipeline(config)` resolves it into a composed
`ComposeTransform`. Nothing in this framework calls it automatically: a
`loader_factory` may call it inside its own pipeline if it wants to, and its
`.inverse` also works directly as `visualization.reconstruction_plot`'s own
`inverse_transform` hook (`pipeline.inverse`) or
`evaluation.visual_export.exportEvaluationFigures`'s `inverse_transforms`
dict. `examples/01_signal_vae_pipeline.py` demonstrates the full pipeline,
including per-sample coordinate-aware resampling, end to end on synthetic
data.

# Permanent scope boundary (not deferred)

`datamodule.py` is explicitly out of scope, permanently, not merely pending
an open question: dataset loading, matching/pairing samples across
modalities, and train/val/test splitting are all dataset-specific in a way
that offers no reusable structure to extract into this framework (unlike the
transforms above, which are generic tensor operations independent of file
format or dataset identity). `DataConfig.loader_factory` (`config/data.py`)
is this framework's one integration point for a caller's own data pipeline,
and stays that way. See spec §6.2 for the full reasoning; do not treat the
absence of `datamodule.py` as a gap to fill.
