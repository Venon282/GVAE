"""Training curve visualization (spec §10 "logging losses"; spec §6.1 milestone 1: "the
ability to inspect training curves").

`plotLossCurves` plots directly from `Trainer.history` (or any list shaped the same
way); `plotStepCurves` plots from a step-level history such as
`visualization.history_callback.HistoryCallback.step_history`, which explicitly carries
a `"step"` key since step numbers are not simply the list index (resuming a `Trainer`
advances `global_step` without resetting it, spec §10, `docs/adr/0005-training-loop.md`).
`plotBetaSchedule` plots a schedule's own `beta(step)` curve independent of any actual
training run, for sanity-checking a chosen schedule before committing to a long run.
"""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from global_vae.training.beta_schedules.base import AbstractBetaSchedule


def _plotSeries(
    series: dict[str, list[tuple[float, float]]],
    log_scale: bool,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float],
) -> Figure:
    """Shared plotting body for `plotLossCurves`/`plotStepCurves`.

    Args:
        series: Metric name -> list of `(x, value)` points, already
            filtered to the epochs/steps where that metric was present.
        log_scale: Whether to use a logarithmic y-axis.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib `Figure`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, points in series.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, label=name, linewidth=1.5)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plotLossCurves(
    history: Sequence[Mapping[str, float]],
    metrics: Sequence[str] | None = None,
    log_scale: bool = False,
    title: str = "Training curves",
    figsize: tuple[float, float] = (8.0, 5.0),
) -> Figure:
    """Plot one or more metric curves over epochs from `Trainer.history`.

    Args:
        history: `Trainer.history` (or any list shaped the same way):
            one dict per epoch, mapping metric name to value, in
            epoch order (the list index is the epoch number).
        metrics: Which metric keys to plot, one line each. `None`
            (default) plots every key starting with `"train/loss/"` or
            `"val/loss/"`, i.e. every reconstruction/regularization/
            total loss curve `Trainer` itself produces (train and, if
            present, validation), so the common case needs no argument.
        log_scale: If `True`, use a logarithmic y-axis (loss curves
            often span orders of magnitude early in training).
        title: Plot title.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `history` is empty, or if no requested (or
            default-selected) metric key appears in any epoch.
    """
    if not history:
        raise ValueError("plotLossCurves received an empty history.")

    if metrics is None:
        all_keys = {key for epoch_metrics in history for key in epoch_metrics}
        resolved_metrics = sorted(
            key for key in all_keys if key.startswith(("train/loss/", "val/loss/"))
        )
    else:
        resolved_metrics = list(metrics)

    series: dict[str, list[tuple[float, float]]] = {name: [] for name in resolved_metrics}
    for epoch, epoch_metrics in enumerate(history):
        for name in resolved_metrics:
            if name in epoch_metrics:
                series[name].append((epoch, epoch_metrics[name]))

    non_empty_series = {name: points for name, points in series.items() if points}
    if not non_empty_series:
        raise ValueError(
            f"None of the requested metrics {resolved_metrics} appear in any epoch of `history`."
        )

    return _plotSeries(non_empty_series, log_scale, title, "epoch", "loss", figsize)


def plotStepCurves(
    step_history: Sequence[Mapping[str, float]],
    metrics: Sequence[str] | None = None,
    log_scale: bool = False,
    title: str = "Training curves (per step)",
    figsize: tuple[float, float] = (8.0, 5.0),
) -> Figure:
    """Plot one or more metric curves over training steps.

    Args:
        step_history: One dict per step, each containing a `"step"`
            key (the x-axis value) alongside metric keys, e.g.
            `visualization.history_callback.HistoryCallback.step_history`.
            Unlike `plotLossCurves`, the list index is *not* used as
            the x-axis, since step numbers are not necessarily
            contiguous with the list position (a `Trainer` resumed
            from a checkpoint continues `global_step` from where it
            left off).
        metrics: Which metric keys to plot. `None` (default) plots
            every key present except `"step"` itself.
        log_scale: As in `plotLossCurves`.
        title: Plot title.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `step_history` is empty, if any entry is
            missing the `"step"` key, or if no requested (or
            default-selected) metric key appears in any entry.
    """
    if not step_history:
        raise ValueError("plotStepCurves received an empty step_history.")
    if any("step" not in entry for entry in step_history):
        raise ValueError("Every entry of step_history must contain a 'step' key.")

    if metrics is None:
        all_keys = {key for entry in step_history for key in entry}
        resolved_metrics = sorted(all_keys - {"step"})
    else:
        resolved_metrics = list(metrics)

    series: dict[str, list[tuple[float, float]]] = {name: [] for name in resolved_metrics}
    for entry in step_history:
        for name in resolved_metrics:
            if name in entry:
                series[name].append((entry["step"], entry[name]))

    non_empty_series = {name: points for name, points in series.items() if points}
    if not non_empty_series:
        raise ValueError(
            f"None of the requested metrics {resolved_metrics} appear in any entry of "
            f"`step_history`."
        )

    return _plotSeries(non_empty_series, log_scale, title, "step", "value", figsize)


def plotBetaSchedule(
    schedules: AbstractBetaSchedule | Mapping[str, AbstractBetaSchedule],
    num_steps: int,
    title: str = "Beta schedule",
    figsize: tuple[float, float] = (8.0, 4.0),
) -> Figure:
    """Plot `beta(step)` for one or several beta schedules, for sanity-checking before training.

    Args:
        schedules: A single `AbstractBetaSchedule`, or a
            latent-space-name -> schedule mapping (`Trainer`'s own
            `beta_schedules` shape), plotted as one line per schedule.
        num_steps: Number of steps to evaluate and plot, `0` to
            `num_steps - 1`.
        title: Plot title.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `num_steps` is not positive.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")

    resolved_schedules: dict[str, AbstractBetaSchedule] = (
        {"beta": schedules} if isinstance(schedules, AbstractBetaSchedule) else dict(schedules)
    )
    steps = list(range(num_steps))
    series: dict[str, list[tuple[float, float]]] = {
        name: [(float(step), schedule(step)) for step in steps]
        for name, schedule in resolved_schedules.items()
    }
    return _plotSeries(
        series, log_scale=False, title=title, xlabel="step", ylabel="beta", figsize=figsize
    )
