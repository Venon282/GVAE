"""A frozen flow prior over a latent space, and the loss that pulls latents toward it (ADR 0061).

When few samples carry every modality and many carry one, the pair loss has little to say about
the space of the modality with no partner. A **flow prior** says something that needs no pairs: the
distribution that the *other* space gives to its own unpaired samples. Li et al. ("Aligning Latent
Spaces with Flow Priors", [arXiv 2506.05240](https://arxiv.org/abs/2506.05240)) fit a flow model to
the target features once, freeze it, and regularize a learnable latent `y` with the flow matching
loss in which `y` plays the part of the data:

- the flow `v(x, t)` is trained on target points `x1` with `x0 ~ N(0, I)`, `t ~ U[0, 1]` and
  `x_t = (1 - t) x0 + t x1`, to predict the velocity `x1 - x0` (flow matching);
- the alignment loss of a latent `y` is `E || v((1 - t) x0 + t y, t) - (y - x0) ||^2` with the same
  draws and the flow frozen: a latent that looks like a target sample has the velocity that a target
  sample would have. It is the paper's tractable surrogate for the log-likelihood of `y` under the
  target distribution, and it needs no likelihood evaluation and no ODE solve.

This module has the three parts that are not a loss term (`losses/terms/flow_prior_alignment.py` is
the term): `FlowPrior`, the module (the velocity network, a fixed standardization of the target
space and, when the widths of the two spaces differ, the fixed random projection of the paper);
`fitFlowPrior`, which fits one to an `(N, width)` tensor of target points; and `saveFlowPrior` and
`loadFlowPrior`, which keep one in a file (tensors and numbers only, readable with
`torch.load(weights_only=True)`).

What the loss is not: a likelihood. The flow matching loss is a surrogate for one (the paper keeps
only part of its bound), and a cloud concentrated toward the middle of the target scores *below*
the target's own points (ADR 0061 measures it), so nothing but the other terms of the objective
keeps a latent from drifting inward. Its minimum is not zero either (the velocity of a target
sample given the point `x_t` is not determined by `x_t`), so its value means something only against
the loss that the target's own points get, which a fit stores as `reference_loss` and
`FlowPrior.alignmentGap` subtracts.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from global_vae.utils.autograd import backward

FLOW_PRIOR_FORMAT = "global_vae.flow_prior"
"""The `format` entry of a prior file, so that another torch file is refused by name."""

FLOW_PRIOR_VERSION = 1
"""The `version` entry of a prior file."""

_CONFIG_KEYS = ("dim", "hidden", "layers", "time_features", "source_dim")
_MIN_POINTS = 8
_MIN_HELD_OUT_POINTS = 100
_HELD_OUT_FRACTION = 0.1
_LOSS_TAIL = 50
_REFERENCE_DRAWS = 8
_REFERENCE_POINTS = 4096


class FlowPrior(nn.Module):
    """A velocity network `v(x, t)` over a space of width `dim`, with its fixed preprocessing.

    The network reads a point and the time through Fourier features of the time. A prior is built
    empty by its constructor and filled by `fitFlowPrior` or `loadFlowPrior`; once fitted its
    parameters do not require a gradient (it is frozen), and its buffers hold the mean and scale of
    the target points, the projection of the target space onto `dim` if there is one, and the
    reference loss.

    Attributes:
        dim: The width of the space the prior is over: the width of the latent it pulls.
        source_dim: The width of the target points before the fixed projection, or `None` when
            they already have `dim` coordinates.
        mean: The mean of the (projected) target points, `(dim,)`.
        scale: The one scale that the centered target points were divided by, a scalar.
        projection: The fixed `(source_dim, dim)` random projection, or `None`.
        reference_loss: The alignment loss of the target's own points, a scalar (`nan` before a
            fit).
    """

    mean: torch.Tensor
    scale: torch.Tensor
    reference_loss: torch.Tensor
    frequencies: torch.Tensor
    projection: torch.Tensor | None

    def __init__(
        self,
        dim: int,
        hidden: int = 128,
        layers: int = 3,
        time_features: int = 8,
        source_dim: int | None = None,
    ) -> None:
        """Build an unfitted prior.

        Args:
            dim: The width of the space, at least 1.
            hidden: The width of the hidden layers, at least 1.
            layers: The number of hidden layers, at least 1.
            time_features: The number of frequencies of the Fourier features of the time, at least
                1 (the network reads twice as many inputs for the time).
            source_dim: The width of the target points when it differs from `dim`, in which case
                the prior holds a fixed projection from that width. `None`: no projection.

        Raises:
            ValueError: If an integer argument is below its minimum.
        """
        super().__init__()
        for name, value, minimum in (
            ("dim", dim, 1),
            ("hidden", hidden, 1),
            ("layers", layers, 1),
            ("time_features", time_features, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"FlowPrior: {name} must be an integer of {minimum} or more, got {value!r}."
                )
        if source_dim is not None and (
            isinstance(source_dim, bool) or not isinstance(source_dim, int) or source_dim < 1
        ):
            raise ValueError(
                f"FlowPrior: source_dim must be a positive integer, got {source_dim!r}."
            )
        self.dim = dim
        self.hidden = hidden
        self.layers = layers
        self.time_features = time_features
        self.source_dim = source_dim
        blocks: list[nn.Module] = []
        width = dim + 2 * time_features
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        blocks.append(nn.Linear(width, dim))
        self.net = nn.Sequential(*blocks)
        self.register_buffer("frequencies", math.pi * torch.arange(1, time_features + 1).float())
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("reference_loss", torch.full((), float("nan")))
        self.register_buffer(
            "projection", None if source_dim is None else torch.zeros(source_dim, dim)
        )

    def config(self) -> dict[str, int | None]:
        """The constructor arguments that rebuild an empty prior of this shape."""
        return {
            "dim": self.dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "time_features": self.time_features,
            "source_dim": self.source_dim,
        }

    def prepare(self, points: torch.Tensor) -> torch.Tensor:
        """Project target points to `dim` coordinates, if the prior has a projection.

        Args:
            points: `(N, source_dim)` target points, or `(N, dim)` when there is no projection.

        Returns:
            `(N, dim)` points, not yet standardized.

        Raises:
            ValueError: If the width of `points` is not the one the prior expects.
        """
        expected = self.dim if self.projection is None else int(self.projection.shape[0])
        if points.dim() != 2 or points.shape[1] != expected:
            raise ValueError(
                f"FlowPrior: expected target points of shape (N, {expected}), got "
                f"{tuple(points.shape)}."
            )
        return points if self.projection is None else points @ self.projection

    def standardize(self, points: torch.Tensor) -> torch.Tensor:
        """Centre on the target's mean and divide by its one scale (the flow's coordinates)."""
        return (points - self.mean) / self.scale

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The predicted velocity at point `x` and time `t`.

        Args:
            x: `(B, dim)` points in the standardized coordinates.
            t: `(B, 1)` times in `[0, 1]`.

        Returns:
            `(B, dim)`.
        """
        angles = t * self.frequencies
        features = torch.cat([x, torch.sin(angles), torch.cos(angles)], dim=-1)
        out: torch.Tensor = self.net(features)
        return out

    def alignmentLoss(
        self,
        y: torch.Tensor,
        num_draws: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """The flow matching loss with `y` as the data (the alignment loss of Li et al.).

        `y` is in the coordinates of the space the prior was fitted for (already projected, if
        there is a projection: this is the loss of a latent of `dim` coordinates, not of a target
        point of `source_dim`). Gradients flow into `y` and not into the flow, which should be
        frozen.

        Args:
            y: `(B, dim)` latents.
            num_draws: How many `(x0, t)` draws to average, at least 1.
            generator: The random generator of the draws, or `None` for the global one.

        Returns:
            A scalar: the mean over the batch (and the draws) of the squared norm of the
            difference between the predicted velocity and `y - x0`, in the standardized
            coordinates.

        Raises:
            ValueError: If `y` is not `(B, dim)` or `num_draws` is below 1.
        """
        if y.dim() != 2 or y.shape[1] != self.dim:
            raise ValueError(
                f"FlowPrior: the latents must have shape (B, {self.dim}), got {tuple(y.shape)}."
            )
        if num_draws < 1:
            raise ValueError(f"FlowPrior: num_draws must be at least 1, got {num_draws}.")
        target = self.standardize(y)
        total = target.new_zeros(())
        for _ in range(num_draws):
            x0 = torch.randn(
                target.shape, generator=generator, device=target.device, dtype=target.dtype
            )
            t = torch.rand((target.shape[0], 1), generator=generator, device=target.device)
            x_t = (1.0 - t) * x0 + t * target
            total = total + ((self.velocity(x_t, t) - (target - x0)) ** 2).sum(dim=-1).mean()
        return total / num_draws

    def alignmentGap(
        self,
        y: torch.Tensor,
        num_draws: int = _REFERENCE_DRAWS,
        generator: torch.Generator | None = None,
    ) -> float:
        """The alignment loss of `y` minus the loss that the target's own points got.

        A latent that is distributed like the target scores near `0`, and one that is not scores
        above it. It is a noisy estimate (the draws are random) and a gap of a few hundredths is
        the floor of the estimate, not a difference.

        Args:
            y: `(B, dim)` latents.
            num_draws: How many draws to average.
            generator: The random generator of the draws, or `None` for the global one.

        Returns:
            The gap, as a float. `nan` for a prior that was never fitted.
        """
        with torch.no_grad():
            return float(self.alignmentLoss(y, num_draws, generator) - self.reference_loss)


@dataclass(frozen=True)
class FlowPriorFit:
    """What `fitFlowPrior` returns.

    Attributes:
        prior: The fitted, frozen prior.
        num_points: How many target points the fit used (held-out ones included).
        num_held_out: How many of them were held out of the fit (`0` for a small set).
        steps: How many optimizer steps ran.
        initial_loss: The mean flow matching loss over the first steps.
        final_loss: The mean flow matching loss over the last steps.
        held_out_loss: The alignment loss of the held-out points, or `None` without any.
        reference_loss: The alignment loss that the prior stores as its reference: the held-out
            points' when there are some, otherwise (an optimistic number) the fit points'.
    """

    prior: FlowPrior
    num_points: int
    num_held_out: int
    steps: int
    initial_loss: float
    final_loss: float
    held_out_loss: float | None
    reference_loss: float


def _referenceLoss(prior: FlowPrior, prepared: torch.Tensor, seed: int) -> float:
    """The mean alignment loss of target points, with a fixed generator so that it is repeatable."""
    generator = torch.Generator(device=prepared.device).manual_seed(seed)
    points = prepared[:_REFERENCE_POINTS]
    with torch.no_grad():
        return float(prior.alignmentLoss(points, _REFERENCE_DRAWS, generator))


def fitFlowPrior(
    points: torch.Tensor,
    *,
    dim: int | None = None,
    hidden: int = 128,
    layers: int = 3,
    time_features: int = 8,
    steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> FlowPriorFit:
    """Fit a flow prior to target points by flow matching, then freeze it.

    Args:
        points: `(N, width)` target points, at least 8 (typically the posterior means that the
            other space gives to its unpaired samples).
        dim: The width of the space the prior will pull. `None` (default) or the width of
            `points`: the prior is over the width of `points`. Another width: the points are
            mapped to it by a fixed Gaussian random projection (the paper's default), drawn from
            `seed` and kept in the prior.
        hidden: The width of the hidden layers of the velocity network.
        layers: The number of hidden layers.
        time_features: The number of frequencies of the Fourier features of the time.
        steps: The number of optimizer steps (Adam), at least 1.
        batch_size: The size of the minibatch of target points, at least 1.
        lr: The learning rate, above 0.
        seed: The seed of the initial weights, the projection, the minibatches and the draws. The
            global random stream is left as it was.

    Returns:
        The `FlowPriorFit`. With at least 100 points, 10 % of them (at least 10) are held out of
        the fit, and the reference loss is read on them.

    Raises:
        ValueError: If `points` is not a 2D tensor of at least 8 finite points, or an argument is
            out of range.
    """
    if points.dim() != 2 or points.shape[0] < _MIN_POINTS:
        raise ValueError(
            f"fitFlowPrior needs an (N, width) tensor of at least {_MIN_POINTS} points, got "
            f"shape {tuple(points.shape)}."
        )
    if not torch.isfinite(points).all():
        raise ValueError("fitFlowPrior: the target points hold a nan or an infinity.")
    for name, value, minimum in (("steps", steps, 1), ("batch_size", batch_size, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"fitFlowPrior: {name} must be an integer of {minimum} or more, got {value!r}."
            )
    if not (lr > 0.0) or not math.isfinite(lr):
        raise ValueError(f"fitFlowPrior: lr must be a finite number above 0, got {lr!r}.")
    source_width = int(points.shape[1])
    target_dim = source_width if dim is None else dim
    # The initial weights come from `seed` and the caller's random stream is left as it was.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        prior = FlowPrior(
            dim=target_dim,
            hidden=hidden,
            layers=layers,
            time_features=time_features,
            source_dim=None if target_dim == source_width else source_width,
        )
    generator = torch.Generator().manual_seed(seed)
    points = points.detach().float().cpu()
    if prior.projection is not None:
        with torch.no_grad():
            prior.projection.copy_(
                torch.randn(source_width, target_dim, generator=generator) / math.sqrt(target_dim)
            )
    prepared = prior.prepare(points).detach()
    order = torch.randperm(prepared.shape[0], generator=generator)
    num_held_out = (
        max(int(_HELD_OUT_FRACTION * prepared.shape[0]), 10)
        if prepared.shape[0] >= _MIN_HELD_OUT_POINTS
        else 0
    )
    held_out = prepared[order[:num_held_out]]
    train = prepared[order[num_held_out:]]
    with torch.no_grad():
        prior.mean.copy_(train.mean(dim=0))
        prior.scale.copy_(((train - train.mean(dim=0)).pow(2).mean().sqrt()).clamp_min(1e-6))
    optimizer = torch.optim.Adam(prior.parameters(), lr=lr)
    losses: list[float] = []
    prior.train()
    for _ in range(steps):
        rows = torch.randint(0, train.shape[0], (batch_size,), generator=generator)
        loss = prior.alignmentLoss(train[rows], 1, generator)
        optimizer.zero_grad()
        backward(loss)
        optimizer.step()
        losses.append(float(loss.detach()))
    prior.eval()
    prior.zero_grad(set_to_none=True)  # a frozen module holds no stale gradient
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    held_out_loss = _referenceLoss(prior, held_out, seed + 1) if num_held_out else None
    reference = (
        held_out_loss if held_out_loss is not None else _referenceLoss(prior, train, seed + 1)
    )
    prior.reference_loss.fill_(reference)
    tail = min(_LOSS_TAIL, steps)
    return FlowPriorFit(
        prior=prior,
        num_points=int(prepared.shape[0]),
        num_held_out=num_held_out,
        steps=steps,
        initial_loss=sum(losses[:tail]) / tail,
        final_loss=sum(losses[-tail:]) / tail,
        held_out_loss=held_out_loss,
        reference_loss=reference,
    )


def saveFlowPrior(prior: FlowPrior, path: str | Path, fit: FlowPriorFit | None = None) -> Path:
    """Write a prior to a file that `loadFlowPrior` reads (tensors and numbers only).

    Args:
        prior: The prior.
        path: The file. Its parent directory is created.
        fit: The result of the fit, whose numbers are kept in the file for reading (`fit` entry).

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if fit is not None:
        summary = {
            "num_points": fit.num_points,
            "num_held_out": fit.num_held_out,
            "steps": fit.steps,
            "initial_loss": fit.initial_loss,
            "final_loss": fit.final_loss,
            "held_out_loss": -1.0 if fit.held_out_loss is None else fit.held_out_loss,
            "reference_loss": fit.reference_loss,
        }
    torch.save(
        {
            "format": FLOW_PRIOR_FORMAT,
            "version": FLOW_PRIOR_VERSION,
            "config": prior.config(),
            "state_dict": prior.state_dict(),
            "fit": summary,
        },
        target,
    )
    return target


def loadFlowPrior(path: str | Path, map_location: str | torch.device | None = "cpu") -> FlowPrior:
    """Read a prior written by `saveFlowPrior`: frozen, in evaluation mode.

    The file is read with `torch.load(weights_only=True)`, so a file from another source cannot
    run code.

    Args:
        path: The file.
        map_location: Where the tensors go.

    Returns:
        The `FlowPrior`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file is not a flow prior file of this version, or its entries do not
            fit one (the message says which).
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"No flow prior file at '{source}'.")
    try:
        content = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as error:  # the restricted unpickler raises several types
        raise ValueError(
            f"'{source}' could not be read as a flow prior file ({error}). A prior is written by "
            f"`saveFlowPrior` or `scripts/fit_flow_prior.py`."
        ) from error
    if not isinstance(content, Mapping) or content.get("format") != FLOW_PRIOR_FORMAT:
        raise ValueError(
            f"'{source}' is not a flow prior file (its `format` is not '{FLOW_PRIOR_FORMAT}')."
        )
    if content.get("version") != FLOW_PRIOR_VERSION:
        raise ValueError(
            f"'{source}' is flow prior file version {content.get('version')!r}; this code reads "
            f"version {FLOW_PRIOR_VERSION}."
        )
    config = content.get("config")
    if not isinstance(config, Mapping) or sorted(config) != sorted(_CONFIG_KEYS):
        raise ValueError(
            f"'{source}': the `config` entry must have exactly the keys {list(_CONFIG_KEYS)}."
        )
    prior = FlowPrior(**dict(config))
    try:
        prior.load_state_dict(content["state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError(
            f"'{source}': the weights do not fit the stored config ({error})."
        ) from error
    prior.eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    return prior
