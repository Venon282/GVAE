"""Paired-data evaluation: do the modalities of one sample land at the same point of a latent space?

A model trained on paired data (a signal and an image of the same sample) is asked, by its fusion
and its shared decoder, to put the two modalities in one latent space. Reconstruction metrics say
whether each modality is rebuilt; they do not say whether the two *agree*. This module measures
that, from the **modality-specific posterior means**: the `mu` that a latent space gets when the
model is queried with one modality alone (spec §5, `GlobalVae.forward` with a single key). For a
pair of modalities `a` and `b` and a latent space `L`, sample `i` has a mean `mu_a[i]` and a mean
`mu_b[i]`, and the two should be closer to each other than to the mean of any other sample.

- **Cross-modal retrieval** (`computeRetrievalMetrics`). Use `mu_a[i]` as a query and rank all
  the `mu_b[j]` by similarity: the true partner is `j = i`. Reported: `recall@k` (the fraction of
  queries whose partner is among the `k` nearest), the mean and median rank of the partner, and the
  mean reciprocal rank, in each direction, next to the value a random ranking would give.
- **Alignment** (`computeAlignmentMetrics`). Closed-form comparisons of the paired means with the
  means of mismatched samples: the cosine similarity, the squared distance (and their ratio, below
  `1` when paired samples are closer than strangers), and the correlation of each latent
  dimension across the two modalities.

**Ties are not hits.** A collapsed latent space puts every sample at the same point, so every
candidate scores the same as the true partner. Counting the partner as first among its ties would
turn a model that encodes nothing into a perfect retriever, so a tie is resolved as a random
ordering would resolve it: the partner's expected rank is `1 + greater + ties / 2`, and the
probability of a hit at `k` is the share of the tied positions that fall within `k`. Two scores
are tied when they differ by less than `1e-9` of their scale. A collapsed model therefore reports
chance, which is what it is.

Retrieval needs the `N x N` similarity matrix, and is computed `chunk_size` queries at a time, so
the memory is `chunk_size x N` and not `N x N`. The alignment metrics are `O(N x latent_dim)`.
Neither reads the reconstructions, only the posterior means, which `collectModalityPosteriorMeans`
gathers with one forward pass per modality per batch (`docs/adr/0038-paired-data-evaluation.md`).
"""

import itertools
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from global_vae.models.global_vae import GlobalVae

DEFAULT_RETRIEVAL_KS = (1, 5, 10)
"""The `k` of `recall@k` that `evaluatePairedAlignment` reports by default."""

SIMILARITIES = ("cosine", "l2")
"""The similarities a retrieval can rank by: `cosine`, or `l2`, the negative squared distance."""

DEFAULT_CHUNK_SIZE = 1024
"""How many queries are scored against the whole gallery at once."""

_TIE_TOLERANCE = 1e-9
"""Scores closer than this, relative to `1 + abs(score)`, are tied (see the module docstring)."""

_MIN_PAIRS = 2
"""Retrieval and the comparison with mismatched samples need at least this many pairs."""

_MIN_MODALITIES = 2
"""A pair needs at least this many input modalities."""

_CONSTANT_VARIANCE = 1e-14
"""A latent dimension whose variance is below this (relative to `1 + mean ** 2`) is constant."""

PosteriorMeans = list[dict[str, dict[str, torch.Tensor]]]
"""Per batch: modality -> latent space name -> posterior mean `(rows, latent_dim)` of that
modality alone. A batch that did not carry a modality has no entry for it."""


def validateRetrievalSettings(
    ks: Sequence[int], similarity: str, chunk_size: int
) -> tuple[int, ...]:
    """Check the settings of a retrieval and normalize `ks`.

    Args:
        ks: The `k` of each `recall@k`, positive integers.
        similarity: One of `SIMILARITIES`.
        chunk_size: How many queries are scored at once, at least `1`.

    Returns:
        `ks` sorted and without duplicates.

    Raises:
        ValueError: If `ks` is empty or holds something that is not a positive integer, if
            `similarity` is not supported, or if `chunk_size` is below `1`.
    """
    if not ks:
        raise ValueError("ks must hold at least one value of k.")
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"Every k must be a positive integer, got {k!r}.")
    if similarity not in SIMILARITIES:
        raise ValueError(f"Unknown similarity '{similarity}'. Supported: {list(SIMILARITIES)}.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
    return tuple(sorted(set(ks)))


def _checkEmbeddings(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate two aligned sets of vectors and return them as `float64` matrices.

    Raises:
        ValueError: If they are not matrices of one shape, hold fewer than `_MIN_PAIRS` rows or a
            non-finite value.
    """
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError(
            "Both sets of means must be matrices (samples, latent_dim) of one shape, got "
            f"{tuple(first.shape)} and {tuple(second.shape)}."
        )
    if first.shape[0] < _MIN_PAIRS:
        raise ValueError(
            f"At least {_MIN_PAIRS} pairs are needed to compare paired and mismatched samples, "
            f"got {first.shape[0]}."
        )
    first, second = first.detach().double(), second.detach().double()
    if not (torch.isfinite(first).all() and torch.isfinite(second).all()):
        raise ValueError("The posterior means must be finite.")
    return first, second


def _normalizeRows(vectors: torch.Tensor) -> torch.Tensor:
    """Scale each row to unit length; a zero row stays zero, so its cosine with anything is 0."""
    normalized: torch.Tensor = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return normalized


def _scoreChunk(queries: torch.Tensor, gallery: torch.Tensor, similarity: str) -> torch.Tensor:
    """Score each query against every gallery row: higher is more similar."""
    if similarity == "cosine":
        similarities: torch.Tensor = queries @ gallery.T
        return similarities
    distances: torch.Tensor = torch.cdist(
        queries, gallery, compute_mode="donot_use_mm_for_euclid_dist"
    )
    return -distances.pow(2)


def computeRetrievalMetrics(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    ks: Sequence[int] = DEFAULT_RETRIEVAL_KS,
    similarity: str = "cosine",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, float]:
    """Rank a gallery for each query and report where the true partner lands.

    `queries[i]` and `gallery[i]` are the two views of sample `i`. Each query is scored against
    every row of `gallery`, and the rank of row `i` is `1 +` the number of rows that score higher.
    Ties are resolved as a random ordering would (see the module docstring).

    Args:
        queries: Posterior means of one modality, `(samples, latent_dim)`.
        gallery: Posterior means of the other modality, the same shape and sample order.
        ks: The `k` of each `recall@k`.
        similarity: `"cosine"` (default), or `"l2"` for the negative squared Euclidean distance.
        chunk_size: How many queries are scored at once. The memory is `chunk_size x samples`.

    Returns:
        For each `k`, in increasing order, `recall@k` (the expected fraction of queries whose
        partner is among the `k` best, so a tie counts for its share) and `chance_recall@k` (what
        a random ranking gives, `min(k, samples) / samples`), then `mean_rank` and `median_rank`
        (the partner's expected rank, `1` is best, `(samples + 1) / 2` is chance) and `mrr`, the
        mean reciprocal rank.

    Raises:
        ValueError: If a setting is invalid (`validateRetrievalSettings`), the inputs are not two
            matrices of one shape with at least 2 rows, or a value is not finite.
    """
    resolved_ks = validateRetrievalSettings(ks, similarity, chunk_size)
    query_matrix, gallery_matrix = _checkEmbeddings(queries, gallery)
    num_samples = query_matrix.shape[0]
    if similarity == "cosine":
        query_matrix, gallery_matrix = _normalizeRows(query_matrix), _normalizeRows(gallery_matrix)

    greater = torch.empty(num_samples, dtype=torch.float64)
    tied = torch.empty(num_samples, dtype=torch.float64)
    for start in range(0, num_samples, chunk_size):
        stop = min(start + chunk_size, num_samples)
        scores = _scoreChunk(query_matrix[start:stop], gallery_matrix, similarity)
        partner = scores[torch.arange(stop - start), torch.arange(start, stop)].unsqueeze(1)
        tolerance = _TIE_TOLERANCE * (1.0 + partner.abs())
        greater[start:stop] = (scores > partner + tolerance).sum(dim=1).double()
        tied[start:stop] = ((scores - partner).abs() <= tolerance).sum(dim=1).double() - 1.0

    expected_rank = greater + 1.0 + tied / 2.0
    metrics: dict[str, float] = {}
    for k in resolved_ks:
        hit_probability = ((k - greater) / (tied + 1.0)).clamp(0.0, 1.0)
        metrics[f"recall@{k}"] = hit_probability.mean().item()
        metrics[f"chance_recall@{k}"] = min(k, num_samples) / num_samples
    ranks = expected_rank.sort().values
    middle = num_samples // 2
    metrics["mean_rank"] = expected_rank.mean().item()
    metrics["median_rank"] = (
        ranks[middle].item()
        if num_samples % 2
        else (ranks[middle - 1].item() + ranks[middle].item()) / 2.0
    )
    # The mean of 1 / position over the tied positions `greater + 1 .. greater + tied + 1`.
    reciprocal = (
        torch.special.digamma(greater + tied + 2.0) - torch.special.digamma(greater + 1.0)
    ) / (tied + 1.0)
    metrics["mrr"] = reciprocal.mean().item()
    return metrics


def computeAlignmentMetrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    """Compare paired posterior means with the means of mismatched samples.

    "Paired" is sample `i` of `first` with sample `i` of `second`; "mismatched" is every `(i, j)`
    with `i != j`. Both means are exact (not sampled) and cost `O(samples x latent_dim)`, using
    that the sum over all pairs of `x_i . y_j` is `(sum x) . (sum y)`.

    Args:
        first: Posterior means of one modality, `(samples, latent_dim)`.
        second: Posterior means of the other modality, the same shape and sample order.

    Returns:
        `cosine_paired`, `cosine_mismatched` and `cosine_gap` (their difference, above `0` when
        paired samples point the same way more than strangers do); `squared_distance_paired`,
        `squared_distance_mismatched` and `squared_distance_ratio` (below `1` when paired samples
        are closer than strangers, `nan` if every sample is the same point); and
        `dimension_correlation`, the mean over the non-constant latent dimensions of the Pearson
        correlation between the two modalities' values of that dimension, with
        `dimensions_compared`, how many dimensions that is (`nan` if there is none).

    Raises:
        ValueError: If the inputs are not two matrices of one shape with at least 2 rows, or a
            value is not finite.
    """
    first, second = _checkEmbeddings(first, second)
    num_samples = first.shape[0]
    num_mismatched = num_samples * (num_samples - 1)

    first_unit, second_unit = _normalizeRows(first), _normalizeRows(second)
    paired_cosines = (first_unit * second_unit).sum(dim=1)
    all_cosines = first_unit.sum(dim=0) @ second_unit.sum(dim=0)
    cosine_paired = paired_cosines.mean().item()
    cosine_mismatched = ((all_cosines - paired_cosines.sum()) / num_mismatched).item()

    paired_squared = (first - second).pow(2).sum(dim=1)
    all_squared = (
        num_samples * first.pow(2).sum()
        + num_samples * second.pow(2).sum()
        - 2.0 * (first.sum(dim=0) @ second.sum(dim=0))
    )
    squared_paired = paired_squared.mean().item()
    squared_mismatched = ((all_squared - paired_squared.sum()) / num_mismatched).item()
    squared_ratio = squared_paired / squared_mismatched if squared_mismatched > 0 else math.nan

    centered_first = first - first.mean(dim=0)
    centered_second = second - second.mean(dim=0)
    variance_first = centered_first.pow(2).mean(dim=0)
    variance_second = centered_second.pow(2).mean(dim=0)
    varying = (variance_first > _CONSTANT_VARIANCE * (1.0 + first.mean(dim=0).pow(2))) & (
        variance_second > _CONSTANT_VARIANCE * (1.0 + second.mean(dim=0).pow(2))
    )
    dimensions_compared = int(varying.sum().item())
    if dimensions_compared:
        covariance = (centered_first * centered_second).mean(dim=0)
        correlation = (
            covariance[varying] / (variance_first[varying] * variance_second[varying]).sqrt()
        )
        dimension_correlation = correlation.clamp(-1.0, 1.0).mean().item()
    else:
        dimension_correlation = math.nan

    return {
        "cosine_paired": cosine_paired,
        "cosine_mismatched": cosine_mismatched,
        "cosine_gap": cosine_paired - cosine_mismatched,
        "squared_distance_paired": squared_paired,
        "squared_distance_mismatched": squared_mismatched,
        "squared_distance_ratio": squared_ratio,
        "dimension_correlation": dimension_correlation,
        "dimensions_compared": float(dimensions_compared),
    }


def collectModalityPosteriorMeans(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    device: str | torch.device | None = None,
    max_samples: int | None = None,
) -> PosteriorMeans:
    """Gather, for each batch, the posterior means that each modality gives on its own.

    Calls `model.eval()`. For every batch that carries at least two input modalities
    (`model.input_modalities`), each of them is fed to the model alone, with `use_mean=True`, and
    the `mu` of every latent space it reaches is kept. A latent space that a modality does not
    feed (the private space of another modality, in a shared plus private model) has no entry for
    it. A batch with fewer than two input modalities holds no pair and is skipped.

    Args:
        model: The model to query.
        dataloader: Yields `dict[str, torch.Tensor]` batches, the convention of `Trainer`. Keys
            that are not input modalities (a decoder-only target, a label) are ignored.
        device: Batches are moved here. Defaults to `model`'s own device.
        max_samples: Stop after this many samples of paired batches, trimming the last batch.
            `None` (default) reads the whole `dataloader`.

    Returns:
        One dict per batch used, in order. See `PosteriorMeans`.

    Raises:
        ValueError: If `max_samples` is below `1`, or a batch has no key naming an input
            modality of `model` (`GlobalVae.selectEncoderInputs`).
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be at least 1, got {max_samples}.")
    resolved_device = device if device is not None else next(model.parameters()).device
    model.eval()
    records: PosteriorMeans = []
    num_samples = 0
    with torch.no_grad():
        for raw_batch in dataloader:
            inputs = model.selectEncoderInputs(raw_batch)
            present = [name for name in model.input_modalities if name in inputs]
            if len(present) < _MIN_MODALITIES:
                continue
            rows = inputs[present[0]].shape[0]
            keep = rows if max_samples is None else min(rows, max_samples - num_samples)
            record: dict[str, dict[str, torch.Tensor]] = {}
            for modality in present:
                alone = {modality: inputs[modality][:keep].to(resolved_device)}
                latent_params = model(alone, use_mean=True)["latent_params"]
                record[modality] = {
                    latent_name: mu.cpu() for latent_name, (mu, _) in latent_params.items()
                }
            records.append(record)
            num_samples += keep
            if max_samples is not None and num_samples >= max_samples:
                break
    return records


def _pairedMeans(
    records: PosteriorMeans, first: str, second: str, latent_name: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Stack the means of two modalities over the batches that carry both, or `None`."""
    pairs = [
        (record[first][latent_name], record[second][latent_name])
        for record in records
        if latent_name in record.get(first, {}) and latent_name in record.get(second, {})
    ]
    if not pairs:
        return None
    return torch.cat([a for a, _ in pairs]), torch.cat([b for _, b in pairs])


def _formatMetrics(metrics: dict[str, float], names: Sequence[str]) -> str:
    """Write `name=value` for the metrics that exist, `nan` included."""
    return ", ".join(f"{name}={metrics[name]:.4f}" for name in names if name in metrics)


@dataclass
class PairedAlignmentResults:
    """Everything one `evaluatePairedAlignment()` call produces.

    A pair of modalities is written `"a|b"` where the direction does not matter (alignment) and
    `"a->b"` where it does (retrieval: `a` is the query, `b` the gallery).

    Attributes:
        similarity: The similarity retrieval ranked by (`SIMILARITIES`).
        ks: The `k` of each `recall@k`, in increasing order.
        num_pairs: `"a|b"` -> the number of samples that carry both modalities.
        retrieval: Latent space name -> `"a->b"` -> metric name -> value
            (`computeRetrievalMetrics`). Both directions of every pair are present.
        alignment: Latent space name -> `"a|b"` -> metric name -> value
            (`computeAlignmentMetrics`).

    Empty `retrieval` and `alignment` mean that the model has fewer than two input modalities.
    """

    similarity: str
    ks: list[int]
    num_pairs: dict[str, int] = field(default_factory=dict)
    retrieval: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    alignment: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)

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

    def summary(self) -> str:
        """A short, human-readable multi-line summary.

        Returns:
            One block per latent space and pair: the alignment numbers, then each retrieval
            direction with the recall of a random ranking in brackets.
        """
        if not self.retrieval and not self.alignment:
            return "Paired evaluation: the model has fewer than two input modalities."
        lines = [f"Paired evaluation (retrieval by {self.similarity} similarity)"]
        for latent_name, per_pair in self.alignment.items():
            for pair, alignment in per_pair.items():
                lines.append(f"  [{latent_name}] {pair} over {self.num_pairs[pair]} pairs")
                lines.append(
                    "    alignment: "
                    + _formatMetrics(
                        alignment,
                        (
                            "cosine_paired",
                            "cosine_mismatched",
                            "squared_distance_ratio",
                            "dimension_correlation",
                            "dimensions_compared",
                        ),
                    )
                )
                first, second = pair.split("|")
                for direction in (f"{first}->{second}", f"{second}->{first}"):
                    metrics = self.retrieval[latent_name][direction]
                    recalls = ", ".join(
                        f"recall@{k}={metrics[f'recall@{k}']:.4f} "
                        f"(chance {metrics[f'chance_recall@{k}']:.4f})"
                        for k in self.ks
                    )
                    lines.append(
                        f"    {direction}: {recalls}, "
                        + _formatMetrics(metrics, ("mean_rank", "median_rank", "mrr"))
                    )
        return "\n".join(lines)


def evaluatePairedAlignment(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    ks: Sequence[int] = DEFAULT_RETRIEVAL_KS,
    similarity: str = "cosine",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
) -> PairedAlignmentResults:
    """Measure retrieval and alignment between each pair of modalities in each shared latent space.

    For each pair of `model.input_modalities` and each latent space that both reach when queried
    alone, the posterior means of the samples that carry both modalities are compared with
    `computeRetrievalMetrics` (in both directions) and `computeAlignmentMetrics`.

    A model with fewer than two input modalities has nothing to compare: the results are empty and
    `dataloader` is not read, in the "absent, not an error" convention of
    `evaluation.cross_modal.exportCrossModalFigures`.

    Args:
        model: The model to evaluate. `model.eval()` is called.
        dataloader: Yields `dict[str, torch.Tensor]` batches that carry the modalities of one
            sample together. A batch with a single modality holds no pair and is skipped.
        ks: The `k` of each `recall@k`.
        similarity: `"cosine"` (default) or `"l2"`, see `computeRetrievalMetrics`.
        chunk_size: How many queries are scored at once, see `computeRetrievalMetrics`.
        device: Batches are moved here. Defaults to `model`'s own device.
        max_samples: Use at most this many paired samples (the first ones). `None` uses all.

    Returns:
        `PairedAlignmentResults`.

    Raises:
        ValueError: If a setting is invalid, or if the model has several input modalities and no
            pair of them has at least 2 samples that carry both in `dataloader`.
    """
    resolved_ks = validateRetrievalSettings(ks, similarity, chunk_size)
    results = PairedAlignmentResults(similarity=similarity, ks=list(resolved_ks))
    if len(model.input_modalities) < _MIN_MODALITIES:
        return results

    records = collectModalityPosteriorMeans(model, dataloader, device, max_samples)
    latent_names = list(
        dict.fromkeys(name for record in records for m in record.values() for name in m)
    )
    for first, second in itertools.combinations(model.input_modalities, 2):
        for latent_name in latent_names:
            means = _pairedMeans(records, first, second, latent_name)
            if means is None or means[0].shape[0] < _MIN_PAIRS:
                continue
            first_means, second_means = means
            results.num_pairs[f"{first}|{second}"] = first_means.shape[0]
            results.alignment.setdefault(latent_name, {})[f"{first}|{second}"] = (
                computeAlignmentMetrics(first_means, second_means)
            )
            per_direction = results.retrieval.setdefault(latent_name, {})
            for query, gallery, name in (
                (first_means, second_means, f"{first}->{second}"),
                (second_means, first_means, f"{second}->{first}"),
            ):
                per_direction[name] = computeRetrievalMetrics(
                    query, gallery, resolved_ks, similarity, chunk_size
                )
    if not results.alignment:
        raise ValueError(
            "No pair of input modalities "
            f"{list(model.input_modalities)} has at least {_MIN_PAIRS} samples that carry both in "
            "the dataloader: paired evaluation needs batches that hold the modalities of one "
            "sample together."
        )
    return results
