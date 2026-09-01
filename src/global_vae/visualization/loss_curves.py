"""Training curve visualization (spec §10 "logging losses"; spec §6.1 milestone 1: "the
ability to inspect training curves").

`plotLossCurves` plots directly from `Trainer.history` (or any list shaped the same
way); `plotStepCurves` plots from a step-level history such as
`visualization.history_callback.HistoryCallback.step_history`, which explicitly carries
a `"step"` key since step numbers are not simply the list index (resuming a `Trainer`
advances `global_step` without resetting it, spec §10, `docs/adr/0005-training-loop.md`).
`plotBetaSchedule` plots a schedule's own `beta(step)` curve independent of any actual
training run, for sanity-checking a chosen schedule before committing to a long run.

Both `plotLossCurves` and `plotStepCurves` accept an optional `twin_metrics`: a second
group of metric keys plotted on a secondary (right-hand) y-axis instead of the primary
one. This exists for the common case where two groups of metrics live on genuinely
different scales (the textbook example being this framework's own reconstruction loss
vs. its regularization loss, which is frequently one to several orders of magnitude
smaller, especially early in training or with a light beta): sharing one axis, even a
log-scaled one, still visually crushes whichever group is smaller, since matplotlib
autoscales the axis to the larger group's range and the smaller group's own variation
is squeezed into a thin band near the axis's edge. Plotting the two groups on their
own independently-scaled axes keeps both readable.
"""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from global_vae.training.beta_schedules.base import AbstractBetaSchedule


def _collectSeries(
    history: Sequence[Mapping[str, float]], keys: Sequence[str], x_key: str | None
) -> dict[str, list[tuple[float, float]]]:
    """Collect `(x, value)` points for each requested metric key.

    Args:
        history: One dict per x-position (an epoch, if `x_key` is `None`: the list
            index itself is the x-value; or a step, if `x_key` is given: that dict's
            own `x_key` entry is the x-value).
        keys: Metric keys to collect.
        x_key: `None` to use the list index as x (the `plotLossCurves` convention);
            otherwise the dict key holding the x-value (the `plotStepCurves`
            convention, e.g. `"step"`).

    Returns:
        Metric name -> list of `(x, value)` points, entries with no matching points
        omitted entirely.
    """
    series: dict[str, list[tuple[float, float]]] = {name: [] for name in keys}
    for index, entry in enumerate(history):
        x = float(entry[x_key]) if x_key is not None else float(index)
        for name in keys:
            if name in entry:
                series[name].append((x, entry[name]))
    return {name: points for name, points in series.items() if points}


def _plotSeries(
    series: dict[str, list[tuple[float, float]]],
    log_scale: bool,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float],
    twin_series: dict[str, list[tuple[float, float]]] | None = None,
    twin_log_scale: bool = False,
    twin_ylabel: str = "value (secondary axis)",
) -> Figure:
    """Shared plotting body for `plotLossCurves`/`plotStepCurves`.

    Args:
        series: Metric name -> list of `(x, value)` points for the primary
            (left-hand) axis, already filtered to the epochs/steps where present.
        log_scale: Whether the primary axis uses a logarithmic y-scale.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Primary y-axis label.
        figsize: Matplotlib figure size.
        twin_series: Metric name -> `(x, value)` points for a secondary
            (right-hand) axis (`ax.twinx()`), drawn with a dashed linestyle to stay
            visually distinct from the primary group even where colors repeat.
            `None` or empty draws no secondary axis, matching this function's
            previous (single-axis) behavior exactly.
        twin_log_scale: Whether the secondary axis uses a logarithmic y-scale.
        twin_ylabel: Secondary y-axis label.

    Returns:
        The matplotlib `Figure`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    lines: list[Line2D] = []
    for name, points in series.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        (line,) = ax.plot(xs, ys, label=name, linewidth=1.5)
        lines.append(line)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if twin_series:
        twin_ax = ax.twinx()
        for name, points in twin_series.items():
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            (line,) = twin_ax.plot(xs, ys, label=name, linewidth=1.5, linestyle="--")
            lines.append(line)
        if twin_log_scale:
            twin_ax.set_yscale("log")
        twin_ax.set_ylabel(twin_ylabel)

    ax.set_title(title)
    ax.legend(lines, [str(line.get_label()) for line in lines], loc="best")
    fig.tight_layout()
    return fig


def plotLossCurves(
    history: Sequence[Mapping[str, float]],
    metrics: Sequence[str] | None = None,
    twin_metrics: Sequence[str] | None = None,
    log_scale: bool = False,
    twin_log_scale: bool = False,
    title: str = "Training curves",
    ylabel: str = "loss",
    twin_ylabel: str = "loss (secondary axis)",
    figsize: tuple[float, float] = (8.0, 5.0),
) -> Figure:
    """Plot one or more metric curves over epochs from `Trainer.history`.

    Args:
        history: `Trainer.history` (or any list shaped the same way):
            one dict per epoch, mapping metric name to value, in
            epoch order (the list index is the epoch number).
        metrics: Which metric keys to plot on the primary axis, one line each.
            `None` (default) plots every key starting with `"train/loss/"` or
            `"val/loss/"` that is not already claimed by `twin_metrics`, i.e.
            every reconstruction/regularization/total loss curve `Trainer`
            itself produces (train and, if present, validation), so the
            common case needs no argument beyond `twin_metrics` itself.
        twin_metrics: Metric keys plotted on a secondary y-axis instead
            (`ax.twinx()`), for a group of metrics on a different scale
            from `metrics` (see the module docstring). A common choice:
            `metrics=["train/loss/reconstruction", "train/loss/total", ...]`,
            `twin_metrics=["train/loss/regularization", ...]`, since the
            regularization term is frequently much smaller than the
            reconstruction/total loss it is added to. `None` (default)
            draws everything on one axis, unchanged from before this
            parameter existed.
        log_scale: If `True`, use a logarithmic y-axis for the primary axis
            (loss curves often span orders of magnitude early in training).
        twin_log_scale: As `log_scale`, for the secondary axis.
        title: Plot title.
        ylabel: Primary y-axis label.
        twin_ylabel: Secondary y-axis label (only shown if `twin_metrics` is
            given).
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `history` is empty, or if no requested (or
            default-selected) primary-axis metric key appears in any epoch.
    """
    if not history:
        raise ValueError("plotLossCurves received an empty history.")

    twin_keys = list(twin_metrics) if twin_metrics else []
    if metrics is None:
        all_keys = {key for epoch_metrics in history for key in epoch_metrics}
        excluded = set(twin_keys)
        resolved_metrics = sorted(
            key
            for key in all_keys
            if key.startswith(("train/loss/", "val/loss/")) and key not in excluded
        )
    else:
        resolved_metrics = list(metrics)

    non_empty_series = _collectSeries(history, resolved_metrics, x_key=None)
    twin_series = _collectSeries(history, twin_keys, x_key=None) if twin_keys else None

    if not non_empty_series:
        raise ValueError(
            f"None of the requested metrics {resolved_metrics} appear in any epoch of `history`."
        )

    return _plotSeries(
        non_empty_series,
        log_scale,
        title,
        "epoch",
        ylabel,
        figsize,
        twin_series=twin_series,
        twin_log_scale=twin_log_scale,
        twin_ylabel=twin_ylabel,
    )


def plotStepCurves(
    step_history: Sequence[Mapping[str, float]],
    metrics: Sequence[str] | None = None,
    twin_metrics: Sequence[str] | None = None,
    log_scale: bool = False,
    twin_log_scale: bool = False,
    title: str = "Training curves (per step)",
    ylabel: str = "value",
    twin_ylabel: str = "value (secondary axis)",
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
        metrics: Which metric keys to plot on the primary axis. `None`
            (default) plots every key present except `"step"` itself and
            anything already claimed by `twin_metrics`.
        twin_metrics: As in `plotLossCurves`: metric keys plotted on a
            secondary y-axis instead, for a group of metrics on a
            different scale from `metrics`.
        log_scale: As in `plotLossCurves`.
        twin_log_scale: As in `plotLossCurves`.
        title: Plot title.
        ylabel: Primary y-axis label.
        twin_ylabel: Secondary y-axis label (only shown if `twin_metrics` is
            given).
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib `Figure`.

    Raises:
        ValueError: If `step_history` is empty, if any entry is
            missing the `"step"` key, or if no requested (or
            default-selected) primary-axis metric key appears in any entry.
    """
    if not step_history:
        raise ValueError("plotStepCurves received an empty step_history.")
    if any("step" not in entry for entry in step_history):
        raise ValueError("Every entry of step_history must contain a 'step' key.")

    twin_keys = list(twin_metrics) if twin_metrics else []
    if metrics is None:
        all_keys = {key for entry in step_history for key in entry}
        resolved_metrics = sorted(all_keys - {"step"} - set(twin_keys))
    else:
        resolved_metrics = list(metrics)

    non_empty_series = _collectSeries(step_history, resolved_metrics, x_key="step")
    twin_series = _collectSeries(step_history, twin_keys, x_key="step") if twin_keys else None

    if not non_empty_series:
        raise ValueError(
            f"None of the requested metrics {resolved_metrics} appear in any entry of "
            f"`step_history`."
        )

    return _plotSeries(
        non_empty_series,
        log_scale,
        title,
        "step",
        ylabel,
        figsize,
        twin_series=twin_series,
        twin_log_scale=twin_log_scale,
        twin_ylabel=twin_ylabel,
    )


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
