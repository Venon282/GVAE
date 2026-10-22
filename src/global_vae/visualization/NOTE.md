# Status

Implemented: `latent_plot.py` (projection + scatter plot + collection
helpers + per-dimension KL diagnostic), `reconstruction_plot.py` (1D
line overlay + grid + collection helper, plus cross-modal reconstruction
reporting: `resolveDefaultInputSubsets`/`collectCrossModalReconstructions`/
`plotCrossModalReconstructionMatrix`, spec §5,
`docs/adr/0016-cross-modal-reconstruction-reporting.md`), `image_plot.py`
(the image counterparts: `plotImageReconstructionGrid`, original and
reconstruction side by side, and `plotImageCrossModalReconstructionMatrix`,
one image decoder per figure; `isImageBatch` tells an image batch from a
batch of 1D series; the collection helpers of `reconstruction_plot.py` feed
both), `loss_curves.py`
(epoch-level, step-level, beta-schedule and learning-rate curves; `plotLearningRate` needs a
history that carries `train/lr`, see `docs/adr/0028-learning-rate-in-the-metrics.md`), `history_callback.py`
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

Nothing about images is deferred any more: `image_plot.py` is the image
reconstruction grid and the image cross-modal matrix (roadmap P3-4), and
`examples/03_signal_image_to_image.py` uses the second one for its figure
instead of a local helper. What `image_plot.py` does not do, on purpose: it
does not draw a latent traversal or any other generative view, and it does not
pick a colour scale for you beyond the shared per-row one (`vmin`/`vmax`
override it).

The real, non-dummy cross-modal demonstration that used to be listed here
exists: `examples/03_signal_image_to_image.py` (signal and image to image
through PoE fusion) uses `collectCrossModalReconstructions` for its
evaluation by input subset.
