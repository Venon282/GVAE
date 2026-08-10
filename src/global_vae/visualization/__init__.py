"""visualization subpackage of global_vae (spec §10: "logging losses, latent-space
visualizations, and reconstructions per run"; spec §6.1 milestone 1: "the ability to
inspect training curves and visualize the latent space").

Every function here returns a plain `matplotlib.figure.Figure` and never displays,
saves, or logs it: what to do with the figure (`fig.savefig(...)`, display inline in a
notebook, or `logger.logFigure(name, fig, step)` via `training/loggers/`) is the
caller's decision, keeping this subpackage decoupled from both file I/O and any
particular experiment-tracking backend.

Requires the `visualization` extra (`pip install -e ".[visualization]"`, matplotlib
only): unlike every other subpackage, importing `global_vae.visualization` does
require an extra dependency, since there is no meaningful "plotting without a
plotting library" fallback. No other part of this framework imports
`global_vae.visualization`, so this requirement never leaks into code that does not
use it. `latent_plot.projectLatentSamples`'s `"tsne"`/`"umap"` methods are further,
separately optional (`pip install -e ".[tsne]"` / `".[umap]"`); the default `"pca"`
method needs neither, using `torch.pca_lowrank` instead.
"""
