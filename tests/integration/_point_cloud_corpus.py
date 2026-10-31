"""Synthetic point clouds of several shapes and sizes, for the point cloud tests.

Each cloud is a random sample of points on the surface of one of four shapes, in a random order
(a set has no order): an ellipsoid, a torus, a cylinder and a box. A shape has random proportions
and a random turn about the vertical axis, and the cloud is centered and scaled so that its
farthest point is at distance 1. The shapes share the same range of sizes, so a model has to read
the geometry and not the scale. It is a stand-in for a dataset: no real point cloud collection is
named anywhere in this framework (`data/NOTE.md`).
"""

import math

import torch

N_MAX = 64
N_MIN = 40
DIM = 3
FAMILIES = ("ellipsoid", "torus", "cylinder", "box")
NOISE = 0.01


def _ellipsoid(n: int, generator: torch.Generator) -> torch.Tensor:
    direction = torch.randn(n, DIM, generator=generator)
    sphere = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
    axes = 0.4 + 0.6 * torch.rand(DIM, generator=generator)
    scaled: torch.Tensor = sphere * axes
    return scaled


def _torus(n: int, generator: torch.Generator) -> torch.Tensor:
    minor = 0.15 + 0.3 * torch.rand(1, generator=generator)
    major = 0.6 + 0.4 * torch.rand(1, generator=generator)
    u = 2 * math.pi * torch.rand(n, generator=generator)
    v = 2 * math.pi * torch.rand(n, generator=generator)
    ring = major + minor * torch.cos(v)
    return torch.stack([ring * torch.cos(u), ring * torch.sin(u), minor * torch.sin(v)], dim=1)


def _cylinder(n: int, generator: torch.Generator) -> torch.Tensor:
    radius = 0.3 + 0.5 * torch.rand(1, generator=generator)
    height = 0.5 + 1.0 * torch.rand(1, generator=generator)
    angle = 2 * math.pi * torch.rand(n, generator=generator)
    z = height * (torch.rand(n, generator=generator) - 0.5)
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle), z], dim=1)


def _box(n: int, generator: torch.Generator) -> torch.Tensor:
    size = 0.4 + 0.6 * torch.rand(DIM, generator=generator)
    area = torch.stack([size[1] * size[2], size[0] * size[2], size[0] * size[1]])
    axis = torch.multinomial(area / area.sum(), n, replacement=True, generator=generator)
    side = torch.where(torch.rand(n, generator=generator) < 0.5, -1.0, 1.0)
    points = (torch.rand(n, DIM, generator=generator) - 0.5) * size
    points[torch.arange(n), axis] = side * size[axis] / 2
    return points


_BUILDERS = (_ellipsoid, _torus, _cylinder, _box)


def _turn(points: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Turn a cloud by a random angle about the vertical axis."""
    angle = 2 * math.pi * torch.rand(1, generator=generator).item()
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return points @ rotation.T


def sortPoints(clouds: torch.Tensor) -> torch.Tensor:
    """Sort the points of every cloud by their first coordinate, leaving `NaN` rows at the end.

    A canonical order a loader can give a cloud, and the only way to make a position-by-position
    loss mean something for a set.
    """
    key = torch.nan_to_num(clouds[..., 0], nan=float("inf"))
    order = key.argsort(dim=1)
    return clouds.gather(1, order.unsqueeze(-1).expand_as(clouds))


def makeClouds(
    count: int, seed: int, variable: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build `count` point clouds of `N_MAX` rows, in a random order, padded with `NaN`.

    Args:
        count: Number of clouds.
        seed: Seed of the generator, so a call is reproducible.
        variable: Whether the number of points varies, from `N_MIN` to `N_MAX`. `False` gives
            clouds of `N_MAX` points, so there is no padding.

    Returns:
        `(clouds, num_points, family)`: the tensor `(count, N_MAX, 3)` with rows of `NaN` after
        the points a cloud has, the number of points of each cloud, and the index of its shape in
        `FAMILIES`.
    """
    generator = torch.Generator().manual_seed(seed)
    clouds = torch.full((count, N_MAX, DIM), float("nan"))
    num_points = torch.zeros(count, dtype=torch.long)
    family = torch.zeros(count, dtype=torch.long)
    for index in range(count):
        n = int(torch.randint(N_MIN, N_MAX + 1, (1,), generator=generator)) if variable else N_MAX
        kind = int(torch.randint(len(_BUILDERS), (1,), generator=generator))
        points = _turn(_BUILDERS[kind](n, generator), generator)
        points = points + NOISE * torch.randn(n, DIM, generator=generator)
        points = points - points.mean(dim=0)
        points = points / points.norm(dim=1).max()
        clouds[index, :n] = points[torch.randperm(n, generator=generator)]
        num_points[index] = n
        family[index] = kind
    return clouds, num_points, family
