"""Evaluation across more than two latent spaces: a multi-way Procrustes baseline (P6-8, ADR 0059).

`evaluation.latent_alignment` compares two latent views. A model with three modalities or more has
a latent space for each, and `anchor` mode (ADR 0059) trains them toward one of them. Before a
multi-way **loss** is built (roadmap P7-4), this module is the baseline that says how much a
post-hoc map already gives: the same closed-form, nothing-to-train fit that baseline 1 is for two
spaces, extended to `K` spaces by generalized Procrustes analysis, the shared orthogonal space that
Achara et al. (arXiv 2602.06205, "Multi-Way Representation Alignment") build for any-to-any
retrieval between three models or more. The correction that they add on top of that space to close
the gap with CCA (GCPA) is **not** implemented here: this is its first stage, which is also the
baseline it is judged against.

For `K` views with one width, on held-out samples, it reads the retrieval of the true partner
(`recall@k`, mean rank, `mrr`, with the chance level) for **every ordered pair** `a -> b`, after
each of these maps, all fitted on the training means only:

- **identity**: no map. The spaces as training left them, which is what a learned alignment is
  judged by. Zero maps.
- **pairwise**: one orthogonal Procrustes map for every pair of views, fitted on that pair alone.
  `K (K - 1) / 2` maps, and the best a map can do for each pair, since none is shared.
- **common**: generalized Procrustes. One orthogonal map for **each** view into a common frame
  (`K` maps), found by iterating the closed-form Procrustes step against the mean of the others
  until the consensus stops moving. A map `a -> b` is `a`'s map followed by the inverse of `b`'s,
  so every pair is consistent through the frame: going from `a` to `b` to `c` is going from `a` to
  `c`.
- **anchor**: one orthogonal map from every view to the chosen anchor view (`K - 1` maps), and a
  pair of other views goes through it (`b -> anchor -> c`). It is what training with `anchor` and
  `others` asks of the spaces, as a map.

Read the table as a bill. If `pairwise` beats `identity` by a wide margin, the trained spaces are
not aligned and an orthogonal map would make them so; if `common` is close to `pairwise`, `K` maps
do what `K (K - 1) / 2` do, which is the reason to align through a frame; if `anchor` is close to
`common`, a hub is enough. A learned alignment that does not beat `identity` of the same model by
more than these maps give is not needed.

The estimate is closed form: it depends on the training means and nothing else (the iteration of
the common frame starts from the maps to the first view, and is deterministic). Only posterior
means are kept, so memory grows with `K x samples x latent_dim`.

```python
report = evaluateMultiwayAlignment(
    model, ["z_signal", "z_image", "z_text"], heldout, train=train, anchor="z_signal"
)
print(report.summary())
```
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from global_vae.data.sample_ids import checkSampleIds
from global_vae.evaluation.latent_alignment import (
    LatentView,
    OrthogonalMap,
    _checkBatchHasInputs,
    _encodeView,
    _rowsToKeep,
    fitOrthogonalProcrustes,
    resolveLatentView,
)
from global_vae.evaluation.paired_alignment import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL_KS,
    computeRetrievalMetrics,
    validateRetrievalSettings,
)
from global_vae.models.global_vae import GlobalVae

DEFAULT_MAX_ITERATIONS = 100
"""The most rounds of the common-frame iteration."""

DEFAULT_TOLERANCE = 1e-10
"""The iteration stops when the consensus gains less than this, relative to its size."""

BASELINES = ("identity", "pairwise", "common", "anchor")
"""The maps a report reads the retrieval after, in the order of the tables."""

_MIN_VIEWS = 2
_MIN_SAMPLES = 2


def viewLabels(views: Sequence[LatentView]) -> tuple[str, ...]:
    """Name each view: its space, or `space@inputs` when two views read the same space.

    Args:
        views: The views, in order.

    Returns:
        One label per view, all different.

    Raises:
        ValueError: If two views are the same (the same space read from the same inputs).
    """
    labels = [view.toText() for view in views]
    if len(set(labels)) != len(labels):
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(f"The views {repeated} are listed more than once.")
    spaces = [view.space for view in views]
    return tuple(view.space if spaces.count(view.space) == 1 else view.toText() for view in views)


@dataclass
class ViewMeans:
    """The posterior means of `K` views over the same samples, row `i` being sample `i` in all.

    Attributes:
        labels: The name of each view, in order.
        means: Label -> posterior means, `(samples, latent_dim)`, `float32`, on the CPU.
        sample_ids_checked: How many rows had their `sample_id.<modality>` entries compared
            across the modalities of the views (0 when the batches carried none, ADR 0058).
    """

    labels: tuple[str, ...]
    means: dict[str, torch.Tensor]
    sample_ids_checked: int = 0

    @property
    def num_samples(self) -> int:
        """The number of samples."""
        return int(self.means[self.labels[0]].shape[0])

    @property
    def width(self) -> int:
        """The width the views share."""
        return int(self.means[self.labels[0]].shape[1])


def collectViewMeans(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    views: Sequence[LatentView],
    device: str | torch.device | None = None,
    max_samples: int | None = None,
) -> ViewMeans:
    """Encode paired batches and keep the posterior mean of each view (no decoder runs).

    Calls `model.eval()`. Each view is encoded from its own inputs alone, with `use_mean=True`.

    Args:
        model: The model to query.
        dataloader: Yields batches that carry every modality of every view, row `i` of each being
            the same sample (and the `sample_id.<modality>` entries, if the loader gives them,
            are compared: ADR 0058).
        views: At least two different views of one width.
        device: Batches are moved here. Defaults to `model`'s own device.
        max_samples: Stop after this many samples, trimming the last batch. `None` reads it all.

    Returns:
        The `ViewMeans`.

    Raises:
        ValueError: If there are fewer than two views, `max_samples` is below `1`, a batch lacks
            a modality of a view or its sample ids disagree, the views do not share a width, or
            `dataloader` gives no sample.
    """
    if len(views) < _MIN_VIEWS:
        raise ValueError(f"A multi-way evaluation needs at least two views, got {len(views)}.")
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be at least 1, got {max_samples}.")
    labels = viewLabels(views)
    resolved_device = device if device is not None else next(model.parameters()).device
    needed = list(dict.fromkeys(name for view in views for name in view.inputs))
    model.eval()
    chunks: dict[str, list[torch.Tensor]] = {label: [] for label in labels}
    seen = 0
    ids_checked = 0
    with torch.no_grad():
        for batch in dataloader:
            _checkBatchHasInputs(batch, needed)
            keep = _rowsToKeep(batch, needed, None if max_samples is None else max_samples - seen)
            if checkSampleIds(batch, needed, "Evaluating several latent spaces") is not None:
                ids_checked += keep
            for label, view in zip(labels, views, strict=True):
                mu, _ = _encodeView(model, batch, view, resolved_device, keep)
                chunks[label].append(mu.float().cpu())
            seen += keep
            if max_samples is not None and seen >= max_samples:
                break
    if seen == 0:
        raise ValueError(
            "The dataloader gave no batch: a multi-way evaluation needs samples that carry "
            "every modality."
        )
    means = {label: torch.cat(parts) for label, parts in chunks.items()}
    widths = {label: int(mu.shape[1]) for label, mu in means.items()}
    if len(set(widths.values())) != 1:
        raise ValueError(
            f"The views have different widths {widths}. Comparing latent spaces needs equal "
            "widths (roadmap D-16); a view of another width can be read through an adapter "
            "(ADR 0060)."
        )
    return ViewMeans(labels=labels, means=means, sample_ids_checked=ids_checked)


@dataclass(frozen=True)
class CommonFrame:
    """Orthogonal maps that carry every view into one frame (generalized Procrustes).

    The frame is the space the views are mapped to: a mean `x` of view `a` is `(x - mean_a) R_a`
    there. Going from `a` to `b` is `a`'s map followed by the transpose of `b`'s, so every path
    between views is consistent.

    Attributes:
        rotations: Label -> the orthogonal matrix `R` of the view, `(latent_dim, latent_dim)`,
            `float64`. A reflection is allowed.
        means: Label -> the mean subtracted from the view before its map (zeros when the fit was
            not centered), added back when a map ends in that view.
        center: Whether each view was centered on its training mean.
        num_samples: How many samples the fit used.
        num_iterations: How many rounds the iteration ran (`0` for a fit that has none).
        converged: Whether the iteration stopped because the consensus stopped moving.
        relative_residual: What the fit leaves, as a fraction of what the views hold: the mean
            squared distance of a mapped mean to the consensus, over the mean squared norm of a
            centered mean. `0` is a perfect agreement, `1` is no agreement at all.
        anchor: The label of the view that is the frame itself (its map is the identity and was
            not fitted), or `None` for a frame in which every view has a fitted map.
    """

    rotations: dict[str, torch.Tensor]
    means: dict[str, torch.Tensor]
    center: bool
    num_samples: int
    num_iterations: int
    converged: bool
    relative_residual: float
    anchor: str | None = None

    @property
    def num_maps(self) -> int:
        """How many orthogonal maps were fitted: one per view, less the anchor's identity."""
        return len(self.rotations) - (0 if self.anchor is None else 1)

    def toFrame(self, label: str, means: torch.Tensor) -> torch.Tensor:
        """Carry means of view `label` into the common frame."""
        mapped: torch.Tensor = (means.double() - self.means[label]) @ self.rotations[label]
        return mapped

    def between(self, source: str, target: str, means: torch.Tensor) -> torch.Tensor:
        """Carry means of view `source` into view `target`, through the frame."""
        mapped: torch.Tensor = self.toFrame(source, means) @ self.rotations[target].T
        return mapped + self.means[target]


def _checkFitMeans(means: Mapping[str, torch.Tensor]) -> list[str]:
    """Check the matrices a fit reads and return their labels in order."""
    labels = list(means)
    if len(labels) < _MIN_VIEWS:
        raise ValueError(f"A multi-way fit needs at least two views, got {len(labels)}.")
    reference = means[labels[0]]
    for label in labels:
        matrix = means[label]
        if matrix.ndim != 2 or matrix.shape != reference.shape:
            raise ValueError(
                f"Every view must give a matrix (samples, latent_dim) of one shape: '{label}' "
                f"gives {tuple(matrix.shape)} and '{labels[0]}' gives {tuple(reference.shape)}."
            )
        if not torch.isfinite(matrix).all():
            raise ValueError(f"The posterior means of '{label}' must be finite.")
    if reference.shape[0] < 1:
        raise ValueError("A multi-way fit needs at least one sample.")
    return labels


def _centered(
    means: Mapping[str, torch.Tensor], labels: Sequence[str], center: bool
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The double-precision means, and the mean taken from each (zeros when not centering)."""
    matrices = {label: means[label].detach().double() for label in labels}
    offsets = {
        label: matrices[label].mean(dim=0)
        if center
        else torch.zeros(matrices[label].shape[1], dtype=torch.float64)
        for label in labels
    }
    return {label: matrices[label] - offsets[label] for label in labels}, offsets


def _procrustesRotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The orthogonal `R` that carries `source` closest to `target`: `U V^T` of `S^T T`."""
    left, _, right_transposed = torch.linalg.svd(source.T @ target)
    rotation: torch.Tensor = left @ right_transposed
    return rotation


def _relativeResidual(aligned: Sequence[torch.Tensor], total: float) -> float:
    """The disagreement of the mapped views with their consensus, relative to what they hold."""
    stacked = torch.stack(list(aligned))
    consensus = stacked.mean(dim=0)
    residual = float(((stacked - consensus) ** 2).sum()) / (stacked.shape[0] * stacked.shape[1])
    return residual / total if total > 0 else 0.0


def fitCommonFrame(
    means: Mapping[str, torch.Tensor],
    center: bool = False,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CommonFrame:
    """Fit one orthogonal map per view into a common frame (generalized Procrustes).

    Each view starts at its Procrustes map to the first view. A round then refits every map
    against the **mean of the other views' current images**, which cannot lower the summed
    agreement (the sum of the squared distances between every two images is a constant minus the
    squared norm of their sum, and each step raises that norm), so the iteration converges, to a
    local optimum that is the global one for views that are rotations of one another.

    Args:
        means: Label -> posterior means `(samples, latent_dim)`, one shape for all, row `i` the
            same sample.
        center: Subtract each view's mean over the samples before the fit and add it back when a
            map ends in that view. `False` (default) is the single orthogonal map of the roadmap.
        max_iterations: The most rounds.
        tolerance: Stop when a round raises the norm of the sum of the images by less than this
            fraction of it.

    Returns:
        The `CommonFrame`.

    Raises:
        ValueError: If there are fewer than two views, the matrices differ in shape or hold a
            value that is not finite, or `max_iterations` is below `0` or `tolerance` is negative.
    """
    if max_iterations < 0 or tolerance < 0:
        raise ValueError(
            f"max_iterations and tolerance must be at least 0, got {max_iterations} and "
            f"{tolerance}."
        )
    labels = _checkFitMeans(means)
    data, offsets = _centered(means, labels, center)
    rotations = {labels[0]: torch.eye(data[labels[0]].shape[1], dtype=torch.float64)}
    for label in labels[1:]:
        rotations[label] = _procrustesRotation(data[label], data[labels[0]])
    images = {label: data[label] @ rotations[label] for label in labels}
    total = sum(float((matrix**2).sum()) for matrix in data.values()) / (
        len(labels) * data[labels[0]].shape[0]
    )
    iterations = 0
    converged = max_iterations == 0 or len(labels) == _MIN_VIEWS
    summed = torch.stack(list(images.values())).sum(dim=0)
    while not converged and iterations < max_iterations:
        before = float((summed**2).sum())
        for label in labels:
            others = (summed - images[label]) / (len(labels) - 1)
            rotations[label] = _procrustesRotation(data[label], others)
            summed = summed - images[label]
            images[label] = data[label] @ rotations[label]
            summed = summed + images[label]
        iterations += 1
        after = float((summed**2).sum())
        converged = after - before <= tolerance * max(after, 1e-300)
    return CommonFrame(
        rotations=rotations,
        means=offsets,
        center=center,
        num_samples=int(data[labels[0]].shape[0]),
        num_iterations=iterations,
        converged=converged,
        relative_residual=_relativeResidual(list(images.values()), total),
    )


def fitAnchoredFrame(
    means: Mapping[str, torch.Tensor], anchor: str, center: bool = False
) -> CommonFrame:
    """Fit one orthogonal map from every view to the anchor view (`K - 1` maps).

    The anchor's own map is the identity, so the frame is the anchor's space, and a map between
    two other views goes through it.

    Args:
        means: Label -> posterior means, as in `fitCommonFrame`.
        anchor: The label of the anchor view.
        center: As in `fitCommonFrame`.

    Returns:
        A `CommonFrame` with no iteration (`num_iterations` is `0`).

    Raises:
        ValueError: If `anchor` is not one of the views, or as in `fitCommonFrame`.
    """
    labels = _checkFitMeans(means)
    if anchor not in labels:
        raise ValueError(f"The anchor '{anchor}' is not one of the views {labels}.")
    data, offsets = _centered(means, labels, center)
    rotations = {
        label: torch.eye(data[label].shape[1], dtype=torch.float64)
        if label == anchor
        else _procrustesRotation(data[label], data[anchor])
        for label in labels
    }
    images = [data[label] @ rotations[label] for label in labels]
    total = sum(float((matrix**2).sum()) for matrix in data.values()) / (
        len(labels) * data[anchor].shape[0]
    )
    return CommonFrame(
        rotations=rotations,
        means=offsets,
        center=center,
        num_samples=int(data[anchor].shape[0]),
        num_iterations=0,
        converged=True,
        relative_residual=_relativeResidual(images, total),
        anchor=anchor,
    )


@dataclass
class MultiwayReport:
    """What the evaluation of `K` latent views reports (roadmap P6-8, ADR 0059).

    Attributes:
        labels: The name of each view.
        views: Label -> `{"space": ..., "inputs": [...]}`.
        anchor: The label of the anchor view, or `None`.
        num_heldout: How many held-out samples the retrieval was read on.
        num_train: How many training samples the maps were fitted on (0: no training data).
        ks: The `k` of each `recall@k`.
        similarity: What retrieval ranked by.
        center: Whether the maps were fitted after centering each view.
        retrieval: Baseline (`identity`, `pairwise`, `common`, `anchor`) -> `"a->b"` -> retrieval
            metrics of the held-out means after that baseline's map. A baseline that was not run
            (no training data, no anchor) is absent.
        num_maps: Baseline -> how many orthogonal maps it fitted.
        frame_fit: Baseline (`common`, `anchor`) -> `{num_iterations, converged,
            relative_residual_train, relative_residual_heldout}`.
        sample_ids_checked: How many rows were compared by sample id (held-out and training).
    """

    labels: list[str]
    views: dict[str, dict[str, Any]]
    anchor: str | None
    num_heldout: int
    num_train: int
    ks: list[int]
    similarity: str
    center: bool
    retrieval: dict[str, dict[str, dict[str, float]]]
    num_maps: dict[str, int]
    frame_fit: dict[str, dict[str, float | int | bool]] = field(default_factory=dict)
    sample_ids_checked: int = 0

    def meanRetrieval(self, baseline: str) -> dict[str, float]:
        """The retrieval metrics of a baseline averaged over the ordered pairs of views.

        Args:
            baseline: One of `BASELINES` that the report holds.

        Returns:
            Metric name -> its mean over every `a->b`.

        Raises:
            KeyError: If the report has no such baseline (the message lists the ones it has).
        """
        if baseline not in self.retrieval:
            raise KeyError(
                f"No baseline '{baseline}' in this report: it has {list(self.retrieval)}."
            )
        pairs = self.retrieval[baseline]
        names = next(iter(pairs.values())).keys()
        return {name: sum(row[name] for row in pairs.values()) / len(pairs) for name in names}

    def toDict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable dict (`dataclasses.asdict(self)`)."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save as indented JSON, creating the parent directories.

        Args:
            path: Destination file path.
        """
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("w", encoding="utf-8") as handle:
            json.dump(self.toDict(), handle, indent=2)

    def _firstK(self) -> int:
        return self.ks[0]

    def summary(self) -> str:
        """A short, human-readable multi-line summary.

        Returns:
            The views, then one line per baseline: its number of maps and its retrieval averaged
            over the ordered pairs, against the chance level.
        """
        k = self._firstK()
        key = f"recall@{k}"
        anchor = f", anchor {self.anchor}" if self.anchor else ""
        lines = [
            f"Multi-way alignment: {len(self.labels)} views ({', '.join(self.labels)}{anchor}) "
            f"over {self.num_heldout} held-out samples"
        ]
        for baseline in BASELINES:
            if baseline not in self.retrieval:
                continue
            mean = self.meanRetrieval(baseline)
            lines.append(
                f"  {baseline:<9} maps={self.num_maps[baseline]:<3} "
                f"{key}={mean[key]:.4g} (chance {mean[f'chance_{key}']:.4g}) "
                f"mean_rank={mean['mean_rank']:.4g} mrr={mean['mrr']:.4g}"
            )
        if self.num_train == 0:
            lines.append(
                "  No training data: only the identity was read. Give `train` to fit the maps."
            )
        return "\n".join(lines)

    def toMarkdown(self) -> str:
        """The report as Markdown: the baselines side by side, then every ordered pair.

        Returns:
            The text, ending with a newline.
        """
        k = self._firstK()
        key = f"recall@{k}"
        lines = [
            f"# Multi-way alignment: {len(self.labels)} latent views",
            "",
            *[
                f"- `{label}`: space `{self.views[label]['space']}` read from "
                f"`{'+'.join(self.views[label]['inputs'])}`"
                for label in self.labels
            ],
            f"- anchor: {f'`{self.anchor}`' if self.anchor else 'none'}",
            f"- held-out samples: {self.num_heldout}; training samples: {self.num_train}",
            f"- similarity: `{self.similarity}`; "
            f"maps {'centered' if self.center else 'not centered'}",
            "",
            "The identity is the spaces as training left them. The other rows are post-hoc "
            "orthogonal maps, fitted on the training means and read on the held-out ones: "
            "`pairwise` one per pair, `common` one per view into a shared frame, `anchor` one "
            "per view to the anchor.",
            "",
            "## Retrieval, averaged over the ordered pairs",
            "",
            "| baseline | maps | "
            + " | ".join(f"recall@{n}" for n in self.ks)
            + " | mean rank | mrr |",
            "|---|---|" + "---|" * (len(self.ks) + 2),
        ]
        for baseline in BASELINES:
            if baseline not in self.retrieval:
                continue
            mean = self.meanRetrieval(baseline)
            cells = [f"{mean[f'recall@{n}']:.4g}" for n in self.ks]
            lines.append(
                f"| {baseline} | {self.num_maps[baseline]} | {' | '.join(cells)} | "
                f"{mean['mean_rank']:.4g} | {mean['mrr']:.4g} |"
            )
        chance = " | ".join(
            f"{self.meanRetrieval('identity')[f'chance_recall@{n}']:.4g}" for n in self.ks
        )
        lines += [f"| chance | 0 | {chance} | {(self.num_heldout + 1) / 2:.4g} | |", ""]
        present = [baseline for baseline in BASELINES if baseline in self.retrieval]
        lines += [
            f"## `{key}` of every ordered pair",
            "",
            "| pair | " + " | ".join(present) + " |",
            "|---|" + "---|" * len(present),
        ]
        for pair in self.retrieval[present[0]]:
            cells = [f"{self.retrieval[baseline][pair][key]:.4g}" for baseline in present]
            lines.append(f"| {pair} | {' | '.join(cells)} |")
        lines.append("")
        for baseline, fit in self.frame_fit.items():
            lines.append(
                f"- `{baseline}` fit: {fit['num_iterations']} rounds, "
                f"{'converged' if fit['converged'] else 'not converged'}; what it leaves of "
                f"the views' size, on the training means {fit['relative_residual_train']:.4g} "
                f"and on the held-out ones {fit['relative_residual_heldout']:.4g}"
            )
        if self.num_train == 0:
            lines.append("- no training data: the maps were not fitted, only the identity is read")
        lines.append(_sampleIdsSentence(self.sample_ids_checked))
        return "\n".join(lines) + "\n"

    def saveMarkdown(self, path: str | Path) -> None:
        """Write `toMarkdown()` to `path`, creating the parent directories.

        Args:
            path: Destination file path.
        """
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(self.toMarkdown(), encoding="utf-8")


def _sampleIdsSentence(checked: int) -> str:
    """One line for the report: were the samples checked by the loader's own sample ids."""
    if checked > 0:
        return (
            f"- sample ids: the `sample_id.<modality>` entries of {checked} rows agreed across "
            "the modalities of the views (ADR 0058)"
        )
    return (
        "- sample ids: the batches carried none, so that the rows are the same samples was "
        "taken on trust (ADR 0058)"
    )


def _retrievalAfter(
    mapped: Mapping[str, Mapping[str, torch.Tensor]],
    heldout: Mapping[str, torch.Tensor],
    ks: Sequence[int],
    similarity: str,
    chunk_size: int,
) -> dict[str, dict[str, float]]:
    """Retrieval of every ordered pair `a -> b`: `mapped[a][b]` against `heldout[b]`."""
    return {
        f"{a}->{b}": computeRetrievalMetrics(mapped[a][b], heldout[b], ks, similarity, chunk_size)
        for a in mapped
        for b in heldout
        if a != b
    }


def evaluateMultiwayAlignment(
    model: GlobalVae,
    views: Sequence[LatentView | str],
    heldout: Iterable[dict[str, torch.Tensor]],
    train: Iterable[dict[str, torch.Tensor]] | None = None,
    anchor: str | None = None,
    ks: Sequence[int] = DEFAULT_RETRIEVAL_KS,
    similarity: str = "cosine",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
    max_train_samples: int | None = None,
    center: bool = False,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> MultiwayReport:
    """Evaluate `K` latent views on held-out samples, against the post-hoc maps fitted on training.

    `heldout` is read once. `train`, when given, is read once to fit the three maps; without it
    only the identity is read (the report says so).

    Args:
        model: The model to evaluate. `model.eval()` is called.
        views: At least two views: a `LatentView`, or `space`, or `space@modality+modality`.
        heldout: Held-out batches that carry every modality of every view, row `i` of each being
            the same sample.
        train: Training batches of the same kind, for the `pairwise`, `common` and `anchor` maps.
        anchor: The label of the anchor view (its space name, or `space@inputs` when two views
            read one space), for the `anchor` baseline. `None` leaves it out.
        ks: The `k` of each `recall@k`.
        similarity: `"cosine"` (default) or `"l2"`.
        chunk_size: How many queries are scored at once.
        device: Batches are moved here. Defaults to `model`'s own device.
        max_samples: Read at most this many held-out samples.
        max_train_samples: Read at most this many training samples.
        center: Center each view on its training mean before the fit.
        max_iterations: The most rounds of the common-frame iteration.
        tolerance: The convergence tolerance of the common-frame iteration.

    Returns:
        The `MultiwayReport`.

    Raises:
        KeyError: If a space is not a latent space of the model.
        ValueError: If a setting is invalid, fewer than two views are given, `anchor` is not one
            of their labels, the views do not share a width, or a batch is not paired.
    """
    resolved_ks = validateRetrievalSettings(ks, similarity, chunk_size)
    resolved_views = [resolveLatentView(model, view) for view in views]
    labels = viewLabels(resolved_views)
    if anchor is not None and anchor not in labels:
        raise ValueError(f"The anchor '{anchor}' is not one of the views {list(labels)}.")
    held = collectViewMeans(model, heldout, resolved_views, device, max_samples)
    if held.num_samples < _MIN_SAMPLES:
        raise ValueError(
            f"Retrieval needs at least {_MIN_SAMPLES} held-out samples, got {held.num_samples}."
        )
    ids_checked = held.sample_ids_checked
    identity = {label: {other: held.means[label] for other in labels} for label in labels}
    retrieval = {
        "identity": _retrievalAfter(identity, held.means, resolved_ks, similarity, chunk_size)
    }
    num_maps = {"identity": 0}
    frame_fit: dict[str, dict[str, float | int | bool]] = {}
    num_train = 0
    if train is not None:
        fitted = collectViewMeans(model, train, resolved_views, device, max_train_samples)
        num_train = fitted.num_samples
        ids_checked += fitted.sample_ids_checked
        pairwise: dict[str, dict[str, torch.Tensor]] = {label: {} for label in labels}
        maps: dict[tuple[str, str], OrthogonalMap] = {}
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                maps[(a, b)] = fitOrthogonalProcrustes(fitted.means[a], fitted.means[b], center)
                pairwise[a][b] = maps[(a, b)].toSecond(held.means[a])
                pairwise[b][a] = maps[(a, b)].toFirst(held.means[b])
        retrieval["pairwise"] = _retrievalAfter(
            pairwise, held.means, resolved_ks, similarity, chunk_size
        )
        num_maps["pairwise"] = len(maps)
        frames = {"common": fitCommonFrame(fitted.means, center, max_iterations, tolerance)}
        if anchor is not None:
            frames["anchor"] = fitAnchoredFrame(fitted.means, anchor, center)
        for name, frame in frames.items():
            through = {
                a: {b: frame.between(a, b, held.means[a]) for b in labels if b != a} for a in labels
            }
            retrieval[name] = _retrievalAfter(
                through, held.means, resolved_ks, similarity, chunk_size
            )
            num_maps[name] = frame.num_maps
            frame_fit[name] = {
                "num_iterations": frame.num_iterations,
                "converged": frame.converged,
                "relative_residual_train": frame.relative_residual,
                "relative_residual_heldout": _residualOf(frame, held.means),
            }
    return MultiwayReport(
        labels=list(labels),
        views={
            label: {"space": view.space, "inputs": list(view.inputs)}
            for label, view in zip(labels, resolved_views, strict=True)
        },
        anchor=anchor,
        num_heldout=held.num_samples,
        num_train=num_train,
        ks=list(resolved_ks),
        similarity=similarity,
        center=center,
        retrieval=retrieval,
        num_maps=num_maps,
        frame_fit=frame_fit,
        sample_ids_checked=ids_checked,
    )


def _residualOf(frame: CommonFrame, means: Mapping[str, torch.Tensor]) -> float:
    """What a fitted frame leaves on other means: the same measure as `relative_residual`."""
    labels = list(means)
    images = [frame.toFrame(label, means[label]) for label in labels]
    centered = [means[label].double() - frame.means[label] for label in labels]
    total = sum(float((matrix**2).sum()) for matrix in centered) / (
        len(labels) * centered[0].shape[0]
    )
    return _relativeResidual(images, total)
