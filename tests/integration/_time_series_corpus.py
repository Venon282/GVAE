"""Synthetic multivariate series with gaps and short recordings, for the time series tests.

Each series has three channels that share one seasonal period and differ by amplitude, phase,
slope and offset, with a little noise. It is the data of the measurements of ADR 0042, and a
stand-in for a recording: no real dataset is named anywhere in this framework (`data/NOTE.md`).
"""

import math

import torch

LENGTH = 96
CHANNELS = 3
NOISE = 0.05


def makeSeries(
    count: int, seed: int, gaps: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build `count` series, with their gaps and their padding if asked.

    With `gaps`, about 30% of the series are shorter than `LENGTH` (the end is `NaN`, the padding
    of a short recording), about half have one hole of 4 to 16 steps, and 5% of the remaining
    values are dropped at random.

    Args:
        count: Number of series.
        seed: Seed of the generator, so a call is reproducible.
        gaps: Whether to remove values. `False` gives complete series.

    Returns:
        `(observed, complete, valid_length)`: the series `(count, CHANNELS, LENGTH)` with `NaN`
        where unobserved, the same series with every value, and the number of leading time steps
        of each series that belong to the recording (`LENGTH` unless it was cut short).
    """
    generator = torch.Generator().manual_seed(seed)

    def uniform(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator)

    steps = torch.arange(LENGTH)
    period = 10 + 20 * uniform(count, 1, 1)
    amplitude = 0.5 + uniform(count, CHANNELS, 1)
    phase = 2 * math.pi * uniform(count, CHANNELS, 1)
    slope = 2 * uniform(count, CHANNELS, 1) - 1
    offset = 2 * uniform(count, CHANNELS, 1) - 1
    complete = offset + slope * (2 * steps / LENGTH - 1)
    complete = complete + amplitude * torch.sin(2 * math.pi * steps / period + phase)
    complete = complete + NOISE * torch.randn(count, CHANNELS, LENGTH, generator=generator)
    observed = complete.clone()
    valid = torch.full((count,), LENGTH)
    if gaps:
        nan = torch.full_like(observed, float("nan"))
        short = uniform(count) < 0.3
        valid = torch.where(short, (48 + (LENGTH - 48) * uniform(count)).long(), valid)
        observed = torch.where(steps[None, None, :] >= valid[:, None, None], nan, observed)
        has_hole = uniform(count) < 0.5
        start = (uniform(count) * (LENGTH - 20)).long()
        size = (4 + 12 * uniform(count)).long()
        inside = (steps[None, :] >= start[:, None]) & (steps[None, :] < (start + size)[:, None])
        observed = torch.where((inside & has_hole[:, None])[:, None, :], nan, observed)
        observed = torch.where(uniform(count, CHANNELS, LENGTH) < 0.05, nan, observed)
    return observed, complete, valid
