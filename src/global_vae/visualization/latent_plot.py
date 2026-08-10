"""Latent-space visualization (spec §6.1 milestone 1: "visualize the latent space").

Two families of function:

- `projectLatentSamples`/`plotLatentSpace`: reduce a batch of latent vectors to 1, 2, or
  3 dimensions and scatter-plot them, optionally colored by a label. Direct (identity)
  when the latent space is already that size (spec's own framing: "directe si
  latent_dim=2"), otherwise PCA (`torch.pca_lowrank`, no extra dependency), t-SNE
  (scikit-learn, soft dependency), or UMAP (umap-learn, soft dependency).
- `collectLatentParams`/`collectLatentSamples`: convenience helpers that run a
  `GlobalVae` over a dataloader and gather one latent space's `(mu, logvar)` or
  realized samples, so a caller does not have to hand-write the batch loop before
  calling the plotting functions above.
- `plotPerDimensionKl`: a bar chart of average KL divergence per latent dimension,
  making posterior collapse (spec §11; the exact failure `free_bits_kl`, `mmd`, and
  `cyclical_annealing` all exist to mitigate) directly visible rather than something
  only inferable from a scalar training curve.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from global_vae.models.global_vae import GlobalVae

_VALID_METHODS = ("auto", "pca", "tsne", "umap", "none")


def _pcaProject(z: torch.Tensor, n_components: int) -> torch.Tensor:
    """Project `z` via PCA (`torch.pca_lowrank`, no extra dependency beyond torch)."""
    q = min(n_components, z.shape[0], z.shape[1])
    centered = z - z.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    return centered @ v[:, :n_components]


def _tsneProject(
    z: torch.Tensor, n_components: int, seed: int | None, **kwargs: Any
) -> torch.Tensor:
    """Project `z` via t-SNE (scikit-learn, soft dependency)."""
    try:
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise ImportError(
            "method='tsne' requires scikit-learn, which is not installed. Install it with "
            '`pip install scikit-learn` or `pip install -e ".[tsne]"`.'
        ) from error
    projected = TSNE(n_components=n_components, random_state=seed, **kwargs).fit_transform(
        z.detach().cpu().numpy()
    )
    return torch.from_numpy(projected)


def _umapProject(
    z: torch.Tensor, n_components: int, seed: int | None, **kwargs: Any
) -> torch.Tensor:
    """Project `z` via UMAP (umap-learn, soft dependency)."""
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "method='umap' requires umap-learn, which is not installed. Install it with "
            '`pip install umap-learn` or `pip install -e ".[umap]"`.'
        ) from error
    projected = umap.UMAP(n_components=n_components, random_state=seed, **kwargs).fit_transform(
        z.detach().cpu().numpy()
    )
    return torch.from_numpy(np.asarray(projected))


def projectLatentSamples(
    z: torch.Tensor,
    method: str = "auto",
    n_components: int = 2,
    seed: int | None = None,
    **method_kwargs: Any,
) -> torch.Tensor:
    """Reduce `z` to `n_components` dimensions for visualization.

    Args:
        z: Latent vectors, shape `(N, latent_dim)`.
        method: `"auto"` (identity if `latent_dim == n_components`,
            otherwise PCA), `"pca"`, `"tsne"`, `"umap"`, or `"none"`
            (identity, raising if the dimensionality does not already
            match rather than silently falling back to a projection).
        n_components: Target dimensionality.
        seed: Forwarded to the stochastic methods (`"tsne"`, `"umap"`)
            for reproducible layouts; `"pca"` is deterministic and
            ignores it.
        **method_kwargs: Forwarded to the underlying scikit-learn/
            umap-learn constructor (e.g. `perplexity` for t-SNE,
            `n_neighbors`/`min_dist` for UMAP), letting a caller tune
            the projection without this function needing to know every
            possible keyword in advance.

    Returns:
        Projected tensor, shape `(N, n_components)`, on CPU.

    Raises:
        ValueError: If `method` is not one of `_VALID_METHODS`, if
            `n_components` is not positive, if `z` is empty, or if
            `method="none"` but `z.shape[-1] != n_components`.
        ImportError: If `method` is `"tsne"`/`"umap"` and the
            corresponding soft dependency is not installed.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Expected one of {_VALID_METHODS}.")
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")
    if z.numel() == 0:
        raise ValueError("projectLatentSamples received an empty `z`.")

    already_right_size = z.shape[-1] == n_components
    if method == "none" or (method == "auto" and already_right_size):
        if not already_right_size:
            raise ValueError(
                f"method='none' requires z's last dimension ({z.shape[-1]}) to already equal "
                f"n_components ({n_components})."
            )
        return z.detach().cpu()

    resolved_method = "pca" if method == "auto" else method
    if resolved_method == "pca":
        return _pcaProject(z.detach().cpu(), n_components)
    if resolved_method == "tsne":
        return _tsneProject(z.detach().cpu(), n_components, seed, **method_kwargs)
    return _umapProject(z.detach().cpu(), n_components, seed, **method_kwargs)


def plotLatentSpace(
    z: torch.Tensor,
    labels: torch.Tensor | Sequence[Any] | None = None,
    method: str = "auto",
    n_components: int = 2,
    seed: int | None = None,
    title: str | None = None,
    label_name: str = "label",
    figsize: tuple[float, float] = (6.0, 6.0),
    point_size: float = 10.0,
    alpha: float = 0.7,
    **method_kwargs: Any,
) -> Figure:
    """Scatter-plot a 1D/2D/3D projection of `z`, optionally colored by `labels`.

    Args:
        z: Latent vectors, shape `(N, latent_dim)`.
        labels: Optional per-sample value used to color points.
            Floating-point values use a continuous colormap with a
            colorbar; anything else (ints, strings, a plain Python
            list/tuple) is treated as categorical, with one color and
            a legend entry per distinct value.
        method: Projection method, forwarded to `projectLatentSamples`.
        n_components: `1`, `2` (default), or `3`.
        seed: Forwarded to stochastic projection methods.
        title: Plot title. Defaults to a description of the method and
            dimensionality if not given.
        label_name: Legend/colorbar title when `labels` is given.
        figsize: Matplotlib figure size.
        point_size: Scatter point size.
        alpha: Scatter point transparency.
        **method_kwargs: Forwarded to `projectLatentSamples`.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `n_components` is not `1`, `2`, or `3`, if `z`
            is empty, or if `labels` is given with a length different
            from `z`'s.
    """
    if n_components not in (1, 2, 3):
        raise ValueError(f"n_components must be 1, 2, or 3 for plotting, got {n_components}.")

    projected = projectLatentSamples(
        z, method=method, n_components=n_components, seed=seed, **method_kwargs
    ).numpy()

    labels_array: np.ndarray | None = None
    is_continuous = False
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels_array = labels.detach().cpu().numpy()
        else:
            labels_array = np.asarray(labels)
        if labels_array.shape[0] != projected.shape[0]:
            raise ValueError(
                f"labels has {labels_array.shape[0]} entries but z has {projected.shape[0]}."
            )
        is_continuous = np.issubdtype(labels_array.dtype, np.floating)

    resolved_title = title or f"Latent space ({method}, {n_components}D)"
    fig = plt.figure(figsize=figsize)
    # matplotlib's stubs do not model `add_subplot(projection="3d")` returning an
    # `Axes3D` (whose `scatter`/`set_zlabel` differ from the base 2D `Axes`), so `ax`
    # is explicitly `Any` here rather than fighting an inaccurate stub.
    ax: Any = fig.add_subplot(projection="3d") if n_components == 3 else fig.add_subplot()

    x = projected[:, 0]
    y = projected[:, 1] if n_components >= 2 else np.zeros_like(x)
    z_coords = projected[:, 2] if n_components == 3 else None

    def scatterMasked(mask: np.ndarray | None, **kwargs: Any) -> Any:
        xs, ys = (x, y) if mask is None else (x[mask], y[mask])
        if z_coords is None:
            return ax.scatter(xs, ys, **kwargs)
        zs = z_coords if mask is None else z_coords[mask]
        return ax.scatter(xs, ys, zs, **kwargs)

    if labels_array is None:
        scatterMasked(None, s=point_size, alpha=alpha)
    elif is_continuous:
        scatter = scatterMasked(None, c=labels_array, cmap="viridis", s=point_size, alpha=alpha)
        fig.colorbar(scatter, ax=ax, label=label_name)
    else:
        for value in sorted(set(labels_array.tolist()), key=str):
            mask = labels_array == value
            scatterMasked(mask, s=point_size, alpha=alpha, label=str(value))
        ax.legend(title=label_name)

    ax.set_xlabel("component 1")
    if n_components >= 2:
        ax.set_ylabel("component 2")
    if n_components == 3:
        ax.set_zlabel("component 3")
    ax.set_title(resolved_title)
    fig.tight_layout()
    return fig


def collectLatentParams(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    latent_name: str,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` over `dataloader` and collect one latent space's `(mu, logvar)` pairs.

    Args:
        model: A `GlobalVae` instance. This function does not call
            `model.eval()` or `model.train()` itself, leaving that
            decision to the caller (visualizing an in-progress training
            run might deliberately want train-mode statistics).
        dataloader: Yields `dict[str, torch.Tensor]` batches, the same
            convention `Trainer` uses.
        latent_name: Which latent space to collect (a key of
            `model.latent_spaces`, e.g. `"z_fused"` for
            `GlobalVae.createSingleLatent`'s default).
        device: Batches are moved here before the forward pass.
            Defaults to `model`'s own device (inferred from its first
            parameter).
        max_samples: Stop after collecting at least this many samples
            (the last batch may slightly overshoot before being
            trimmed). `None` (default) collects the entire dataloader.

    Returns:
        `(mu, logvar)`, each shape `(N, latent_dim)`, on CPU.

    Raises:
        ValueError: If `dataloader` yields no batches, or no batch
            ever produced `latent_name` (e.g. every batch happened to
            be missing every modality feeding it).
    """
    resolved_device = device if device is not None else next(model.parameters()).device
    collected_mu: list[torch.Tensor] = []
    collected_logvar: list[torch.Tensor] = []
    total = 0

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = {name: tensor.to(resolved_device) for name, tensor in raw_batch.items()}
            outputs = model(batch)
            if latent_name not in outputs["latent_params"]:
                continue
            mu, logvar = outputs["latent_params"][latent_name]
            collected_mu.append(mu.cpu())
            collected_logvar.append(logvar.cpu())
            total += mu.shape[0]
            if max_samples is not None and total >= max_samples:
                break

    if not collected_mu:
        raise ValueError(
            f"collectLatentParams never observed latent space '{latent_name}' across the "
            f"given dataloader."
        )
    mu_all = torch.cat(collected_mu, dim=0)
    logvar_all = torch.cat(collected_logvar, dim=0)
    if max_samples is not None:
        mu_all, logvar_all = mu_all[:max_samples], logvar_all[:max_samples]
    return mu_all, logvar_all


def collectLatentSamples(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    latent_name: str,
    device: str | torch.device | None = None,
    use_mean: bool = True,
    max_samples: int | None = None,
) -> torch.Tensor:
    """Run `model` over `dataloader` and collect one latent space's realized values.

    A thin convenience wrapper around `collectLatentParams` for the
    common "just give me points to plot" case.

    Args:
        model: As in `collectLatentParams`.
        dataloader: As in `collectLatentParams`.
        latent_name: As in `collectLatentParams`.
        device: As in `collectLatentParams`.
        use_mean: If `True` (default), returns the posterior mean
            (`mu`: deterministic, lower-variance, usually preferred for
            visualization). If `False`, returns a reparameterized
            sample instead (showing the actual stochasticity the model
            samples from at training/generation time).
        max_samples: As in `collectLatentParams`.

    Returns:
        Latent vectors, shape `(N, latent_dim)`, on CPU.
    """
    mu, logvar = collectLatentParams(
        model, dataloader, latent_name, device=device, max_samples=max_samples
    )
    if use_mean:
        return mu
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def plotPerDimensionKl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    collapse_threshold: float = 0.01,
    title: str = "Per-dimension KL divergence",
    figsize: tuple[float, float] = (8.0, 4.0),
) -> Figure:
    """Bar chart of average KL-to-standard-normal divergence per latent dimension.

    Makes posterior collapse directly visible: a dimension whose bar
    sits near `collapse_threshold` (highlighted in a different color)
    is carrying little to no information, regardless of which
    regularization strategy (`losses/regularizers/`) was actually used
    during training, since this only depends on the encoder's own
    `(mu, logvar)` output, not on the training loss.

    Args:
        mu: Posterior mean, shape `(N, latent_dim)` (e.g. from
            `collectLatentParams`).
        logvar: Posterior log-variance, shape `(N, latent_dim)`.
        collapse_threshold: Dimensions with average KL below this
            value (nats) are drawn in a different color. Purely
            cosmetic: does not affect the computed values.
        title: Plot title.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `mu`/`logvar` have different shapes, or are
            empty.
    """
    if mu.shape != logvar.shape:
        raise ValueError(
            f"mu and logvar must have the same shape, got {mu.shape} and {logvar.shape}."
        )
    if mu.numel() == 0:
        raise ValueError("plotPerDimensionKl received empty mu/logvar.")

    kl_per_dim = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)
    kl_values = kl_per_dim.detach().cpu().numpy()
    colors = ["tab:red" if value < collapse_threshold else "tab:blue" for value in kl_values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(kl_values)), kl_values, color=colors)
    ax.axhline(
        collapse_threshold,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"collapse threshold ({collapse_threshold})",
    )
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("average KL (nats)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
