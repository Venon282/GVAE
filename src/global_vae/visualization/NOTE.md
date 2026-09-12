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
  line overlay): natural once an image decoder exists (spec §6.1
  milestone 2). Not built yet, since no image decoder exists yet
  either (`encoders/`/`decoders/` NOTE.md).
- A real, non-dummy cross-modal demonstration (`examples/`): awaits the
  same milestone as the item above, for the same reason; the dummy
  fixtures in `tests/integration/test_cross_modal_reconstruction.py`
  already exercise every code path, so this is only deferred, not
  blocking anything.
