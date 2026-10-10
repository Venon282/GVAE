# Status

Implemented: `latent_plot.py` (projection + scatter plot + collection
helpers + per-dimension KL diagnostic), `reconstruction_plot.py` (1D
line overlay + grid + collection helper, plus cross-modal reconstruction
reporting: `resolveDefaultInputSubsets`/`collectCrossModalReconstructions`/
`plotCrossModalReconstructionMatrix`, spec §5,
`docs/adr/0016-cross-modal-reconstruction-reporting.md`), `loss_curves.py`
(epoch-level, step-level, and beta-schedule curves), `history_callback.py`
(in-memory step/epoch metric collection). See
`docs/adr/0009-visualization.md`.

Requires the `visualization` extra (`pip install -e ".[visualization]"`,
matplotlib) to import at all; `latent_plot.projectLatentSamples`'s
`"tsne"`/`"umap"` methods are further, separately optional (`.[tsne]`/
`.[umap]`). No other part of this framework imports
`global_vae.visualization`, so this is the one subpackage where an
extra dependency is genuinely required to use it, not merely to use
one specific strategy within it.

# Deferred

- **Image-comparison reconstruction plot** (side-by-side original/
  reconstruction images, as opposed to `reconstruction_plot.py`'s 1D
  line overlay), and an image version of the cross-modal matrix, which
  draws 1D series only. The 2D encoders and decoders it was waiting for
  exist now (`docs/adr/0017-2d-cnn-encoder-decoder.md`,
  `docs/adr/0018-2d-residual-encoder-decoder.md`), so nothing blocks it
  any more; it is simply not written yet (roadmap P3-4). Until then
  `examples/03_signal_image_to_image.py` draws its own image grid
  locally. That helper should be removed once the library one exists.

The real, non-dummy cross-modal demonstration that used to be listed here
exists: `examples/03_signal_image_to_image.py` (signal and image to image
through PoE fusion) uses `collectCrossModalReconstructions` for its
evaluation by input subset.
