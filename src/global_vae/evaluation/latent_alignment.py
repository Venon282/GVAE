"""Evaluation across two separate latent spaces (roadmap P6-6, finding I-8, ADR 0056).

`evaluation.paired_alignment` compares the posterior means of one latent space that **both**
modalities reach. The staged recipe of Phase 6 has two latent spaces, `z_signal` and `z_image`,
each reached by one modality, and a distance between them that goes to zero proves nothing: two
collapsed spaces also minimize it. This module is the evaluation that does not take the distance
at its word. For a **pair of latent views** it reports, on held-out pairs:

- **every registered distance** of `losses.distances` (ADR 0053), next to the same distance on
  mismatched pairs (sample `i` of one space against sample `i + 1` of the other), so that a value
  is read against what strangers score and a distance that is 0 for strangers too is flagged;
- **retrieval** (`recall@k`, mean rank, in both directions, each with its chance level), by
  `paired_alignment.computeRetrievalMetrics`;
- **cross reconstruction** next to **self reconstruction**, and the gap between them: what the
  decoder of one modality produces from the latent code of the other (`GlobalVae.decode` with the
  code passed under the other space's name, ADR 0046), scored by the five default metrics;
- the **latent diagnostics** of both spaces (`evaluation.latent_diagnostics`) as a **collapse
  guard**: a space with fewer active units than its solo stage is flagged;
- a **pair-integrity check**: the retrieval of the post-hoc orthogonal Procrustes map, fitted on the
  training pairs and read on those same pairs, must be above chance and above the same fit on
  shuffled pairs, or the pairs are suspect.

It also fits the **Procrustes baseline** (baseline 1 of the roadmap): one orthogonal map between
the two spaces' posterior means, in closed form, fitted on the training pairs and applied to the
held-out ones (Maystre et al., arXiv 2510.13406, for the alignment of independently trained
representations; Achara et al., arXiv 2602.06205, for more than two). If it matches a learned
alignment, the learned alignment is not needed.

**A view** is a latent space and the modalities the model is queried with
(`LatentView`). In a pair model the two views are `z_signal` queried with `signal` and `z_image`
queried with `image`. In a joint model with a shared latent space they are the **same** space
queried with each modality alone (`z_fused` with `signal`, then with `image`), which is how the
joint route, baseline 3, is read with the same code. Only the posterior means and variances of the
held-out pairs are kept, and the reconstruction metrics are accumulated batch by batch
(`evaluation.streaming_metrics`), so memory grows with `num_samples x latent_dim` and not with the
size of the data.

Nothing here changes `GlobalVae`, the loss or the routing: it calls `encode` and `decode`
(ADR 0046) and the registries, and it is opt-in. `scripts/evaluate.py --latent-pair` runs it for a
checkpoint, and `evaluation.alignment_baselines` runs the four baselines of the roadmap together.
"""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

import global_vae.losses.distances  # noqa: F401 (registers the built-in distances)
from global_vae.data.sample_ids import checkSampleIds
from global_vae.evaluation.latent_diagnostics import (
    DEFAULT_ACTIVE_UNIT_THRESHOLD,
    LatentDiagnostics,
    computeLatentDiagnostics,
    validateActiveUnitThreshold,
)
from global_vae.evaluation.paired_alignment import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL_KS,
    computeAlignmentMetrics,
    computeRetrievalMetrics,
    validateRetrievalSettings,
)
from global_vae.evaluation.streaming_metrics import StreamingMetric, buildStreamingMetrics
from global_vae.losses.distances.base import AbstractLatentDistance
from global_vae.losses.distances.registry import (
    getLatentDistanceClass,
    listRegisteredLatentDistances,
)
from global_vae.models.global_vae import GlobalVae

DEFAULT_MIN_ACTIVE_UNITS = 1
"""A space with fewer active units than this is flagged when no solo reference is given."""

DEFAULT_INTEGRITY_SHUFFLES = 20
"""How many shuffles of the training pairs the integrity check fits the map on."""

DEFAULT_INTEGRITY_MARGIN = 0.05
"""The recall of the Procrustes map on the training pairs must beat chance by this much."""

DEFAULT_MIN_ADAPTER_SPREAD = 0.1
"""An adapter whose output spreads less than this fraction of its input, per dimension, is flagged.

A trained adapter can lower the distance by shrinking its output, which the active-unit guard does
not see (the variance stays above its threshold). On the synthetic measurement of ADR 0060 (three
seeds, a toy problem) one adapter kept 0.4 to 1.1 of the per-dimension spread, and two trainable
adapters without a stop gradient kept 0.01 to 0.07, so 0.1 separates them there; with a stop
gradient on one space and two adapters it kept 0.03 to 0.23, which 0.1 does not. It is a place to
start reading, not a proof of either: the retrieval is what says whether the alignment works.
"""

INTEGRITY_OK = "ok"
INTEGRITY_SUSPECT = "suspect"
INTEGRITY_NOT_CHECKED = "not_checked"

_MIN_PAIRS = 2
_MAX_INTEGRITY_CHANCE = 0.25
"""The integrity check reads the largest `k` whose chance level is at most this."""

_SEPARATION_TOLERANCE = 1e-9
"""A distance on mismatched pairs below this, relative to `1 + abs(paired)`, is zero."""


@dataclass(frozen=True)
class LatentView:
    """A latent space as one set of inputs sees it.

    Attributes:
        space: The name of a latent space of the model (`z_signal`).
        inputs: The input modalities the model is queried with, whose encoders feed `space`
            (`("signal",)`). The posterior of `space` is read from `GlobalVae.encode` of these
            modalities alone.
    """

    space: str
    inputs: tuple[str, ...]

    def toText(self) -> str:
        """Write the view as `space@input+input`, the form `--latent-pair` takes."""
        return f"{self.space}@{'+'.join(self.inputs)}"


def parseLatentView(text: str) -> tuple[str, tuple[str, ...] | None]:
    """Split the text of a view into a space name and, if given, its input modalities.

    Args:
        text: A latent space name (`z_signal`), or `space@modality` or `space@modality+modality`
            (`z_fused@signal`).

    Returns:
        `(space, inputs)`, with `inputs` `None` when the text names the space only.

    Raises:
        ValueError: If a part is empty.
    """
    space, separator, rest = text.partition("@")
    space = space.strip()
    if not space:
        raise ValueError(f"A latent view needs a space name, got '{text}'.")
    if not separator:
        return space, None
    inputs = tuple(part.strip() for part in rest.split("+"))
    if not all(inputs):
        raise ValueError(f"A latent view '{text}' has an empty modality after '@'.")
    return space, inputs


def resolveLatentView(
    model: GlobalVae, view: LatentView | str, inputs: Sequence[str] | None = None
) -> LatentView:
    """Check a view against the model and fill in its inputs.

    Args:
        model: The model the view is read from.
        view: A `LatentView`, or the text of one (`parseLatentView`).
        inputs: The modalities to query with, for a space name given as text without `@`. `None`
            (default): every modality that the encoders feeding the space ingest
            (`GlobalVae.modalitiesEncodedInto`).

    Returns:
        The `LatentView`.

    Raises:
        KeyError: If the space is not a latent space of `model` (the message lists the real ones).
        ValueError: If an input is repeated, none is given, or an input is not ingested by an
            encoder that feeds the space.
    """
    if isinstance(view, str):
        space, parsed_inputs = parseLatentView(view)
        chosen: Sequence[str] | None = parsed_inputs if parsed_inputs is not None else inputs
    else:
        space, chosen = view.space, view.inputs
    fed = model.modalitiesEncodedInto(space)
    resolved = tuple(fed) if chosen is None else tuple(chosen)
    if not resolved:
        raise ValueError(f"The view of latent space '{space}' needs at least one input modality.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"The view of '{space}' repeats a modality: {list(resolved)}.")
    foreign = [name for name in resolved if name not in fed]
    if foreign:
        raise ValueError(
            f"Latent space '{space}' is not fed by the modality {foreign}: the encoders that feed "
            f"it ingest {list(fed)}."
        )
    return LatentView(space, resolved)


def pairLabels(first: LatentView, second: LatentView) -> tuple[str, str]:
    """Name the two views of a pair for the report.

    The space name when the two views read different spaces (`z_signal`, `z_image`), the inputs
    joined by `+` when they read the same one (`signal`, `image`: a shared latent space).

    Args:
        first: The first view.
        second: The second view.

    Returns:
        Two different labels.

    Raises:
        ValueError: If the two views are the same space with the same inputs.
    """
    if first.space != second.space:
        return first.space, second.space
    labels = ("+".join(first.inputs), "+".join(second.inputs))
    if labels[0] == labels[1]:
        raise ValueError(
            f"The two views are the same: space '{first.space}' read from {list(first.inputs)}. "
            "A pair needs two different spaces, or one space read from different inputs."
        )
    return labels


@dataclass
class PairPosteriors:
    """The posteriors of two views over the same samples, row `i` being sample `i` in both.

    Attributes:
        first_mu: Posterior mean of the first view, `(samples, latent_dim)`, `float32`, on the CPU.
        first_logvar: Its log-variance, the same shape.
        second_mu: Posterior mean of the second view.
        second_logvar: Its log-variance.
        sample_ids_checked: How many rows had their `sample_id.<modality>` entries compared across
            the modalities of the two views (0 when the batches carried no ids, ADR 0058).
    """

    first_mu: torch.Tensor
    first_logvar: torch.Tensor
    second_mu: torch.Tensor
    second_logvar: torch.Tensor
    sample_ids_checked: int = 0

    @property
    def num_samples(self) -> int:
        """The number of paired samples."""
        return int(self.first_mu.shape[0])


def adaptPosteriors(
    posteriors: PairPosteriors,
    first: LatentView,
    second: LatentView,
    adapters: Mapping[str, nn.Module],
) -> PairPosteriors:
    """Read the posteriors of two views through the adapters of their spaces (ADR 0060).

    A view whose space has no adapter is returned as it is. The adapters run on their own device
    with no gradient, and the result is back on the CPU in `float32`, like the posteriors.

    Args:
        posteriors: The two views' posteriors at their own widths.
        first: The first view (its `space` names the adapter).
        second: The second view.
        adapters: Latent space name -> a latent head (`heads/`), as the alignment term trained it.

    Returns:
        New `PairPosteriors`, with the same `sample_ids_checked`.
    """

    def read(
        space: str, mu: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adapter = adapters.get(space)
        if adapter is None:
            return mu, logvar
        parameter = next(adapter.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        with torch.no_grad():
            adapted_mu, adapted_logvar = adapter(mu.to(device), logvar.to(device))
        return adapted_mu.float().cpu(), adapted_logvar.float().cpu()

    first_mu, first_logvar = read(first.space, posteriors.first_mu, posteriors.first_logvar)
    second_mu, second_logvar = read(second.space, posteriors.second_mu, posteriors.second_logvar)
    return PairPosteriors(
        first_mu=first_mu,
        first_logvar=first_logvar,
        second_mu=second_mu,
        second_logvar=second_logvar,
        sample_ids_checked=posteriors.sample_ids_checked,
    )


def _checkBatchHasInputs(batch: Mapping[str, torch.Tensor], needed: Sequence[str]) -> None:
    """Raise if a batch lacks a modality that one of the views is read from."""
    missing = [name for name in needed if name not in batch]
    if missing:
        raise ValueError(
            f"Evaluating a pair of latent spaces needs batches that carry the modalities of one "
            f"sample together: the batch lacks {missing} (it has {sorted(batch)})."
        )


def _encodeView(
    model: GlobalVae,
    batch: Mapping[str, torch.Tensor],
    view: LatentView,
    device: str | torch.device,
    keep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the first `keep` rows of a batch with the view's inputs only: its `(mu, logvar)`."""
    inputs = {name: batch[name][:keep].to(device) for name in view.inputs}
    latent_params = model.encode(inputs, use_mean=True)["latent_params"]
    if view.space not in latent_params:
        raise ValueError(
            f"Encoding {list(view.inputs)} did not reach latent space '{view.space}' "
            f"(it reached {sorted(latent_params)})."
        )
    return latent_params[view.space]


def _checkBatchSampleIds(
    batch: Mapping[str, torch.Tensor], needed: Sequence[str], keep: int
) -> int:
    """Compare the sample ids of a batch across the modalities read, and count the rows kept.

    Returns:
        `keep` when the batch carries ids and they agree, 0 when it carries none.

    Raises:
        ValueError: As `checkSampleIds`, with the evaluation as the one that found it.
    """
    checked = checkSampleIds(batch, needed, "Evaluating a pair of latent spaces")
    return 0 if checked is None else keep


def _rowsToKeep(batch: Mapping[str, torch.Tensor], needed: Sequence[str], room: int | None) -> int:
    """How many rows of a batch to use, given how many more samples may be read (`None`: all)."""
    rows = int(batch[needed[0]].shape[0])
    return rows if room is None else min(rows, room)


def collectPairPosteriors(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    first: LatentView,
    second: LatentView,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
    require_equal_widths: bool = True,
) -> PairPosteriors:
    """Encode paired batches and keep the posterior of each view (no decoder runs).

    Calls `model.eval()`. Each view is encoded from its own inputs alone, with `use_mean=True`
    (`GlobalVae.encode`), so this is the encode-only path of ADR 0046.

    Args:
        model: The model to query.
        dataloader: Yields `dict[str, torch.Tensor]` batches that carry every modality of both
            views, row `i` of each being the same sample.
        first: The first view.
        second: The second view.
        device: Batches are moved here. Defaults to `model`'s own device.
        max_samples: Stop after this many samples, trimming the last batch. `None` reads it all.
        require_equal_widths: Refuse two views of different widths (default). `False` keeps the
            posteriors at their own widths, for a pair that is read through adapters (ADR 0060).

    Returns:
        The `PairPosteriors`.

    Raises:
        ValueError: If `max_samples` is below `1`, a batch lacks a modality of a view, the two
            views have different widths and `require_equal_widths` is set, or `dataloader` gives
            no sample.
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be at least 1, got {max_samples}.")
    resolved_device = device if device is not None else next(model.parameters()).device
    needed = list(dict.fromkeys([*first.inputs, *second.inputs]))
    model.eval()
    chunks: list[tuple[torch.Tensor, ...]] = []
    seen = 0
    ids_checked = 0
    with torch.no_grad():
        for batch in dataloader:
            _checkBatchHasInputs(batch, needed)
            keep = _rowsToKeep(batch, needed, None if max_samples is None else max_samples - seen)
            ids_checked += _checkBatchSampleIds(batch, needed, keep)
            first_mu, first_logvar = _encodeView(model, batch, first, resolved_device, keep)
            second_mu, second_logvar = _encodeView(model, batch, second, resolved_device, keep)
            if require_equal_widths:
                _checkWidths(first_mu.shape[-1], second_mu.shape[-1], first, second)
            chunks.append(
                tuple(t.float().cpu() for t in (first_mu, first_logvar, second_mu, second_logvar))
            )
            seen += keep
            if max_samples is not None and seen >= max_samples:
                break
    if not chunks:
        raise ValueError("The dataloader gave no batch: a pair evaluation needs paired samples.")
    posteriors = PairPosteriors(
        first_mu=torch.cat([chunk[0] for chunk in chunks]),
        first_logvar=torch.cat([chunk[1] for chunk in chunks]),
        second_mu=torch.cat([chunk[2] for chunk in chunks]),
        second_logvar=torch.cat([chunk[3] for chunk in chunks]),
        sample_ids_checked=ids_checked,
    )
    return posteriors


def _checkWidths(
    first_width: int, second_width: int, first: LatentView, second: LatentView, after: str = ""
) -> None:
    """Raise if the two views are not read at one width (roadmap D-16)."""
    if first_width != second_width:
        raise ValueError(
            f"'{first.toText()}' has width {first_width} and '{second.toText()}' has width "
            f"{second_width}{after}. Comparing two latent spaces needs equal widths (roadmap "
            "D-16): a distance and a retrieval are undefined between spaces of different widths. "
            "To compare spaces that keep their own widths, read them through adapters of one width "
            "(`adapters`, ADR 0060)."
        )


@dataclass(frozen=True)
class OrthogonalMap:
    """A closed-form orthogonal map between two latent spaces, fitted on pairs of posterior means.

    `toSecond` carries a mean of the first space into the second, and `toFirst` is its inverse
    (the transpose of the same orthogonal matrix, so that one fit serves both directions).

    Attributes:
        rotation: The orthogonal matrix `R` with `R R^T = I`, `(latent_dim, latent_dim)`,
            `float64`. A reflection is allowed: the map is any element of the orthogonal group.
            For spaces of different widths `(d1, d2)` it has orthonormal rows or columns, whichever
            are fewer (`fitOrthogonalProcrustes`).
        first_mean: The mean subtracted from the first space before the map, and added back by
            `toFirst`. Zeros when the fit was not centered.
        second_mean: The mean added to the image of the first space. Zeros when not centered.
        center: Whether the two spaces were centered on their training means before the fit.
        num_pairs: How many pairs the fit used.
    """

    rotation: torch.Tensor
    first_mean: torch.Tensor
    second_mean: torch.Tensor
    center: bool
    num_pairs: int

    def toSecond(self, first_means: torch.Tensor) -> torch.Tensor:
        """Map means of the first space into the second: `(x - mean_1) R + mean_2`."""
        mapped: torch.Tensor = (first_means.double() - self.first_mean) @ self.rotation
        return mapped + self.second_mean

    def toFirst(self, second_means: torch.Tensor) -> torch.Tensor:
        """Map means of the second space into the first: `(y - mean_2) R^T + mean_1`."""
        mapped: torch.Tensor = (second_means.double() - self.second_mean) @ self.rotation.T
        return mapped + self.first_mean


def fitOrthogonalProcrustes(
    first: torch.Tensor, second: torch.Tensor, center: bool = False
) -> OrthogonalMap:
    """Fit the orthogonal matrix that carries `first` closest to `second`, in closed form.

    It minimizes `||X R - Y||_F` over orthogonal `R` (the orthogonal Procrustes problem): with the
    singular value decomposition `X^T Y = U S V^T`, the solution is `R = U V^T`. There is nothing
    to train, no hyperparameter and no initialization, so the result depends only on the pairs.

    The two spaces may have different widths `d1` and `d2` (spaces that keep their own, ADR 0060):
    the thin decomposition gives `R` of shape `(d1, d2)` with orthonormal rows (if `d1 <= d2`) or
    columns (if `d1 > d2`), a **partial isometry**: it embeds the narrower space in the wider one,
    and `toFirst` projects back, so the map loses what the narrower direction cannot hold.

    Args:
        first: Posterior means of the first space, `(samples, d1)`.
        second: Posterior means of the second space, `(samples, d2)`, row `i` the same sample.
        center: Subtract each space's mean over the pairs before the fit, and add them back in
            `toSecond` and `toFirst`. `False` (default) is the single orthogonal map of the
            roadmap; `True` also removes a constant offset between the spaces.

    Returns:
        The `OrthogonalMap`.

    Raises:
        ValueError: If the inputs are not two matrices with the same number of rows and at least
            one, or a value is not finite.
    """
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError(
            "Both sets of means must be matrices (samples, latent_dim) with one row per sample, "
            f"got {tuple(first.shape)} and {tuple(second.shape)}."
        )
    if first.shape[0] < 1:
        raise ValueError("The orthogonal map needs at least one pair.")
    x, y = first.detach().double(), second.detach().double()
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("The posterior means must be finite.")
    first_mean = x.mean(dim=0) if center else torch.zeros(x.shape[1], dtype=torch.float64)
    second_mean = y.mean(dim=0) if center else torch.zeros(y.shape[1], dtype=torch.float64)
    left, _, right_transposed = torch.linalg.svd(
        (x - first_mean).T @ (y - second_mean), full_matrices=False
    )
    return OrthogonalMap(
        rotation=left @ right_transposed,
        first_mean=first_mean,
        second_mean=second_mean,
        center=center,
        num_pairs=int(x.shape[0]),
    )


def _procrustesRetrieval(
    mapping: OrthogonalMap,
    first: torch.Tensor,
    second: torch.Tensor,
    ks: Sequence[int],
    similarity: str,
    chunk_size: int,
) -> dict[str, dict[str, float]]:
    """Retrieval in both directions after the map: `first -> second` and `second -> first`."""
    return {
        "forward": computeRetrievalMetrics(
            mapping.toSecond(first), second, ks, similarity, chunk_size
        ),
        "backward": computeRetrievalMetrics(
            mapping.toFirst(second), first, ks, similarity, chunk_size
        ),
    }


@dataclass
class PairIntegrity:
    """The result of the pair-integrity check (roadmap I-8).

    The Procrustes map is fitted on the training pairs and read on those same pairs: if row `i` of
    one space and row `i` of the other are the same sample, the map recovers the correspondence,
    and the recall of the true partner is above chance. The check also fits the map on shuffled
    copies of the pairs, which an orthogonal map with many free parameters can partly fit, so
    that a mismatched set of pairs that the map happens to over-fit is not mistaken for a good
    one.

    Attributes:
        status: `"ok"`, `"suspect"`, or `"not_checked"` (no training pairs were given, or fewer
            than two).
        message: One sentence that says what was found, and what to look at when it is suspect.
        k: The `k` of the recall that was read (`None` when not checked).
        num_pairs: How many training pairs the check used.
        recall: The recall at `k` of the map on the true pairs, averaged over the two directions.
        chance_recall: What a random ranking gives, `min(k, pairs) / pairs`.
        shuffled_recall_mean: The same recall for the map fitted on shuffled pairs, averaged over
            the shuffles.
        shuffled_recall_max: The largest recall over the shuffles.
        shuffles: How many shuffles were fitted.
        margin: The amount by which `recall` has to exceed chance.
        sample_ids_checked: How many rows, held-out and training together, had their
            `sample_id.<modality>` entries compared across the modalities of the two views: 0 when
            the batches carried none, and then the pairs were not checked by id (ADR 0058).
    """

    status: str
    message: str
    k: int | None = None
    num_pairs: int = 0
    recall: float = math.nan
    chance_recall: float = math.nan
    shuffled_recall_mean: float = math.nan
    shuffled_recall_max: float = math.nan
    shuffles: int = 0
    margin: float = DEFAULT_INTEGRITY_MARGIN
    sample_ids_checked: int = 0


def chooseSelectiveK(ks: Sequence[int], num_pairs: int) -> int:
    """The `k` whose recall is still selective for `num_pairs` samples.

    The largest of `ks` whose chance level (`min(k, pairs) / pairs`) is at most 25%, else the
    smallest: a recall at a `k` near the number of samples is high for any ranking, and says little.

    Args:
        ks: The candidate `k`, at least one.
        num_pairs: How many samples are ranked, at least 1.

    Returns:
        One of `ks`.
    """
    selective = [k for k in ks if min(k, num_pairs) / num_pairs <= _MAX_INTEGRITY_CHANCE]
    return max(selective) if selective else min(ks)


def _integrityK(ks: Sequence[int], num_pairs: int, requested: int | None) -> int:
    """The `k` the integrity check reads: the one asked for, else the largest still selective."""
    if requested is not None:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError(f"integrity_k must be a positive integer, got {requested!r}.")
        return requested
    return chooseSelectiveK(ks, num_pairs)


def _inSampleRecall(
    first: torch.Tensor,
    second: torch.Tensor,
    k: int,
    center: bool,
    similarity: str,
    chunk_size: int,
) -> float:
    """Fit the map on the pairs and return the mean of the two directions' `recall@k` on them."""
    mapping = fitOrthogonalProcrustes(first, second, center)
    both = _procrustesRetrieval(mapping, first, second, [k], similarity, chunk_size)
    return (both["forward"][f"recall@{k}"] + both["backward"][f"recall@{k}"]) / 2.0


def checkPairIntegrity(
    first: torch.Tensor,
    second: torch.Tensor,
    ks: Sequence[int] = DEFAULT_RETRIEVAL_KS,
    similarity: str = "cosine",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    center: bool = False,
    k: int | None = None,
    shuffles: int = DEFAULT_INTEGRITY_SHUFFLES,
    margin: float = DEFAULT_INTEGRITY_MARGIN,
    seed: int = 0,
) -> PairIntegrity:
    """Check that the training pairs are pairs: the Procrustes retrieval must beat chance.

    Args:
        first: Posterior means of the first space on the **training** pairs, `(pairs, latent_dim)`.
        second: The same pairs in the second space, row `i` the same sample.
        ks: The `k` that the report carries, used to pick `k` when it is not given.
        similarity: `"cosine"` or `"l2"`, the retrieval similarity.
        chunk_size: How many queries are scored at once.
        center: Whether the map is fitted after centering (see `fitOrthogonalProcrustes`).
        k: The `k` of the recall to read. `None` (default): the largest of `ks` whose chance level
            is at most 25%, else the smallest, so that the check stays selective.
        shuffles: How many random permutations of the second space's rows the map is also fitted
            on. `0` skips this control.
        margin: The recall must exceed `chance + margin`, and exceed every shuffled recall.
        seed: Seeds the permutations.

    Returns:
        A `PairIntegrity`. Fewer than two pairs give `"not_checked"`, not an error.

    Raises:
        ValueError: If a retrieval setting is invalid, `shuffles` is negative, `margin` is negative
            or the two sets differ in shape.
    """
    resolved_ks = validateRetrievalSettings(ks, similarity, chunk_size)
    if shuffles < 0:
        raise ValueError(f"shuffles must be at least 0, got {shuffles}.")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"margin must be finite and at least 0, got {margin}.")
    if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
        raise ValueError(
            "Both sets of means must be matrices (pairs, latent_dim) with one row per pair, got "
            f"{tuple(first.shape)} and {tuple(second.shape)}."
        )
    num_pairs = int(first.shape[0])
    if num_pairs < _MIN_PAIRS:
        return PairIntegrity(
            status=INTEGRITY_NOT_CHECKED,
            message=f"Fewer than {_MIN_PAIRS} training pairs ({num_pairs}): nothing to check.",
            num_pairs=num_pairs,
            margin=margin,
        )
    resolved_k = _integrityK(resolved_ks, num_pairs, k)
    chance = min(resolved_k, num_pairs) / num_pairs
    recall = _inSampleRecall(first, second, resolved_k, center, similarity, chunk_size)
    generator = torch.Generator().manual_seed(seed)
    shuffled = [
        _inSampleRecall(
            first,
            second[torch.randperm(num_pairs, generator=generator)],
            resolved_k,
            center,
            similarity,
            chunk_size,
        )
        for _ in range(shuffles)
    ]
    shuffled_mean = sum(shuffled) / len(shuffled) if shuffled else math.nan
    shuffled_max = max(shuffled) if shuffled else math.nan
    bar = max(chance + margin, shuffled_max if shuffled else -math.inf)
    ok = recall > bar
    if ok:
        message = (
            f"The Procrustes map recovers the partner on the {num_pairs} training pairs: recall@"
            f"{resolved_k} is {recall:.3f}, against {chance:.3f} for chance"
            + (f" and at most {shuffled_max:.3f} for shuffled pairs." if shuffled else ".")
        )
    else:
        message = (
            f"The Procrustes map does not recover the partner on the {num_pairs} training pairs: "
            f"recall@{resolved_k} is {recall:.3f}, and it has to exceed {bar:.3f} (chance "
            f"{chance:.3f} plus a margin of {margin:g}"
            + (f", and the {shuffled_max:.3f} that shuffled pairs reach" if shuffled else "")
            + "). Row i of the two modalities may not be the same sample (roadmap I-8), one of the "
            "two spaces may be collapsed, or the two spaces are not related by an orthogonal map."
        )
    return PairIntegrity(
        status=INTEGRITY_OK if ok else INTEGRITY_SUSPECT,
        message=message,
        k=resolved_k,
        num_pairs=num_pairs,
        recall=recall,
        chance_recall=chance,
        shuffled_recall_mean=shuffled_mean,
        shuffled_recall_max=shuffled_max,
        shuffles=shuffles,
        margin=margin,
    )


def _chunkBounds(count: int, size: int, minimum: int) -> list[tuple[int, int]]:
    """Split `count` rows into consecutive `(start, stop)` chunks of `size`.

    The last chunk is merged into the one before it when it would hold fewer than `minimum` rows.
    """
    bounds = [(start, min(start + size, count)) for start in range(0, count, size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < minimum:
        (previous_start, _), (_, stop) = bounds[-2], bounds[-1]
        bounds[-2:] = [(previous_start, stop)]
    return bounds


def _meanDistance(
    distance: AbstractLatentDistance,
    first: tuple[torch.Tensor, torch.Tensor],
    second: tuple[torch.Tensor, torch.Tensor],
    chunk_size: int,
) -> float:
    """The mean over samples of a distance, taken chunk by chunk (each chunk is one call)."""
    count = first[0].shape[0]
    total = 0.0
    with torch.no_grad():
        for start, stop in _chunkBounds(count, chunk_size, distance.min_batch_size):
            value = distance(
                (first[0][start:stop], first[1][start:stop]),
                (second[0][start:stop], second[1][start:stop]),
            )
            total += value.double().sum().item()
    return total / count


@dataclass
class DistanceResult:
    """One latent distance, on paired and on mismatched held-out samples.

    Attributes:
        paired: The mean of the distance between the two posteriors of the same sample.
        mismatched: The mean of the same distance between sample `i` of the first space and sample
            `i + 1` of the second (wrapping round), which no alignment can make small.
        ratio: `paired / mismatched`: below 1 when the pairs are closer than strangers, near 1 when
            the distance does not tell them apart, `nan` when the mismatched value is 0.
        separates: `False` when the distance is 0 for strangers too, so that a small paired value
            proves nothing (two collapsed spaces give exactly that).
    """

    paired: float
    mismatched: float
    ratio: float
    separates: bool


def computeDistanceTable(
    posteriors: PairPosteriors,
    names: Sequence[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    seed: int = 0,
) -> tuple[dict[str, DistanceResult], dict[str, str]]:
    """Evaluate registered latent distances on paired and on mismatched samples.

    Each distance is built with its default arguments and called on consecutive chunks of
    `chunk_size` samples (a distance whose negatives are the other samples of the call,
    `info_nce`, is therefore read at the size of a chunk), and the value is the mean over all
    samples. A distance that draws random numbers (`mixture_jsd`) runs under a private generator
    seeded with `seed`, and the global generator is left as it was.

    Args:
        posteriors: The two views' posteriors over the same samples, at least two.
        names: The distances to evaluate. `None` (default) takes every registered one.
        chunk_size: How many samples one call of a distance sees, at least 1.
        seed: Seeds a stochastic distance.

    Returns:
        `(values, skipped)`: name -> `DistanceResult`, and name -> why a distance was not
        evaluated (it needs more samples than there are, or it cannot be built without
        arguments).

    Raises:
        KeyError: If a name is not a registered distance (the message lists the real ones).
        ValueError: If `chunk_size` is below 1 or there are fewer than two samples.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
    if posteriors.num_samples < _MIN_PAIRS:
        raise ValueError(
            f"At least {_MIN_PAIRS} pairs are needed to compare paired and mismatched samples, "
            f"got {posteriors.num_samples}."
        )
    resolved = list(names) if names is not None else listRegisteredLatentDistances()
    classes = {name: getLatentDistanceClass(name) for name in resolved}
    first = (posteriors.first_mu, posteriors.first_logvar)
    second = (posteriors.second_mu, posteriors.second_logvar)
    shifted = (second[0].roll(-1, dims=0), second[1].roll(-1, dims=0))
    values: dict[str, DistanceResult] = {}
    skipped: dict[str, str] = {}
    for name, distance_class in classes.items():
        try:
            distance = distance_class()
        except TypeError as error:
            skipped[name] = f"it cannot be built without arguments ({error})"
            continue
        if posteriors.num_samples < distance.min_batch_size:
            skipped[name] = (
                f"it needs at least {distance.min_batch_size} samples, there are "
                f"{posteriors.num_samples}"
            )
            continue
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            paired = _meanDistance(distance, first, second, chunk_size)
            mismatched = _meanDistance(distance, first, shifted, chunk_size)
        separates = abs(mismatched) > _SEPARATION_TOLERANCE * (1.0 + abs(paired))
        values[name] = DistanceResult(
            paired=paired,
            mismatched=mismatched,
            ratio=paired / mismatched if separates else math.nan,
            separates=separates,
        )
    return values, skipped


@dataclass
class PairReconstruction:
    """Self and cross reconstruction metrics of a pair of views, over the held-out samples.

    Attributes:
        self_reconstruction: View label -> modality -> metric -> value: the modalities the view is
            read from, rebuilt from the view's own latent code.
        cross_reconstruction: `"a->b"` -> modality -> metric -> value: the modalities of view `b`,
            rebuilt from the latent code of view `a` passed under the name of `b`'s space.
        mapped_cross_reconstruction: The same, with the code of `a` carried by the orthogonal map
            first (empty when no map was given).
    """

    self_reconstruction: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    cross_reconstruction: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    mapped_cross_reconstruction: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )


@dataclass
class PairStatistics:
    """What one pass over the held-out pairs keeps.

    Attributes:
        posteriors: The two views' posteriors.
        reconstruction: The self and cross reconstruction metrics.
    """

    posteriors: PairPosteriors
    reconstruction: PairReconstruction


def _decodableTargets(model: GlobalVae, view: LatentView) -> list[str]:
    """The modalities of a view that the decoders of its space rebuild from that space alone."""
    decodable = set(model.modalitiesDecodedFrom([view.space]))
    return [name for name in view.inputs if name in decodable]


def collectPairStatistics(
    model: GlobalVae,
    dataloader: Iterable[dict[str, torch.Tensor]],
    first: LatentView,
    second: LatentView,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
    mapping: OrthogonalMap | None = None,
    require_equal_widths: bool = True,
) -> PairStatistics:
    """One pass over held-out pairs: the posteriors, and the self and cross reconstruction.

    Calls `model.eval()`. For each batch, each view is encoded from its own inputs with
    `use_mean=True`. The **self** reconstruction of a view decodes its own code through its own
    space, and the **cross** reconstruction of view `b` from view `a` decodes the code of `a`
    passed under the name of `b`'s space (`GlobalVae.decode`, ADR 0046), then scores the
    modalities `b` is read from against the batch. With a `mapping`, the same cross reconstruction
    is also scored with the code of `a` carried by the orthogonal map first. The five default
    metrics are accumulated batch by batch, so no reconstruction is pooled.

    Args:
        model: The model to evaluate.
        dataloader: Yields batches that carry every modality of both views.
        first: The first view.
        second: The second view.
        device: Batches are moved here. Defaults to `model`'s own device.
        max_samples: Stop after this many samples, trimming the last batch. `None` reads it all.
        mapping: An `OrthogonalMap` from `first`'s space to `second`'s, fitted on training pairs.
        require_equal_widths: Refuse two views of different widths (default). `False` is for a
            pair read through adapters (ADR 0060): the posteriors keep their own widths, the cross
            reconstruction, which would hand a code of one width to the decoders of another, is not
            computed, and the reconstruction through a `mapping` is (its image has the width of
            the target).

    Returns:
        The `PairStatistics`. A direction whose target space has no decoder that can run from it
        alone (a space read only through a `concat` assembler) has no entry.

    Raises:
        ValueError: If `max_samples` is below `1`, a batch lacks a modality of a view, the two
            views have different widths and `require_equal_widths` is set, the two views are the
            same, or `dataloader` is empty.
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be at least 1, got {max_samples}.")
    labels = pairLabels(first, second)
    views = (first, second)
    resolved_device = device if device is not None else next(model.parameters()).device
    needed = list(dict.fromkeys([*first.inputs, *second.inputs]))
    targets = [_decodableTargets(model, view) for view in views]
    directions = {(0, 1): f"{labels[0]}->{labels[1]}", (1, 0): f"{labels[1]}->{labels[0]}"}
    meters: dict[tuple[str, str, str], dict[str, StreamingMetric]] = {}

    def update(kind: str, key: str, modality: str, estimate: torch.Tensor, target: Any) -> None:
        group = meters.setdefault((kind, key, modality), buildStreamingMetrics())
        for meter in group.values():
            meter.update(estimate, target)

    model.eval()
    chunks: list[tuple[torch.Tensor, ...]] = []
    seen = 0
    ids_checked = 0
    with torch.no_grad():
        for batch in dataloader:
            _checkBatchHasInputs(batch, needed)
            keep = _rowsToKeep(batch, needed, None if max_samples is None else max_samples - seen)
            ids_checked += _checkBatchSampleIds(batch, needed, keep)
            encoded = [_encodeView(model, batch, view, resolved_device, keep) for view in views]
            chunks.append(
                tuple(t.float().cpu() for pair in encoded for t in pair),
            )
            codes = [mu for mu, _ in encoded]
            same_width = codes[0].shape[-1] == codes[1].shape[-1]
            if require_equal_widths:
                _checkWidths(codes[0].shape[-1], codes[1].shape[-1], first, second)
            for index, view in enumerate(views):
                if targets[index]:
                    decoded = model.decode({view.space: codes[index]})
                    for modality in targets[index]:
                        update(
                            "self",
                            labels[index],
                            modality,
                            decoded[modality],
                            batch[modality][:keep].to(resolved_device),
                        )
            for (source, target), direction in directions.items():
                if not targets[target]:
                    continue
                reference = {m: batch[m][:keep].to(resolved_device) for m in targets[target]}
                if same_width:
                    decoded = model.decode({views[target].space: codes[source]})
                    for modality in targets[target]:
                        update("cross", direction, modality, decoded[modality], reference[modality])
                if mapping is not None:
                    carried = (mapping.toSecond if source == 0 else mapping.toFirst)(
                        codes[source].cpu()
                    )
                    decoded = model.decode(
                        {views[target].space: carried.to(codes[source].dtype).to(resolved_device)}
                    )
                    for modality in targets[target]:
                        update(
                            "mapped", direction, modality, decoded[modality], reference[modality]
                        )
            seen += keep
            if max_samples is not None and seen >= max_samples:
                break
    if not chunks:
        raise ValueError("The dataloader gave no batch: a pair evaluation needs paired samples.")
    posteriors = PairPosteriors(
        first_mu=torch.cat([chunk[0] for chunk in chunks]),
        first_logvar=torch.cat([chunk[1] for chunk in chunks]),
        second_mu=torch.cat([chunk[2] for chunk in chunks]),
        second_logvar=torch.cat([chunk[3] for chunk in chunks]),
        sample_ids_checked=ids_checked,
    )

    reconstruction = PairReconstruction()
    sections = {
        "self": reconstruction.self_reconstruction,
        "cross": reconstruction.cross_reconstruction,
        "mapped": reconstruction.mapped_cross_reconstruction,
    }
    for (kind, key, modality), group in meters.items():
        sections[kind].setdefault(key, {})[modality] = {
            name: meter.compute() for name, meter in group.items()
        }
    return PairStatistics(posteriors=posteriors, reconstruction=reconstruction)


def computeReconstructionGap(
    self_reconstruction: Mapping[str, Mapping[str, Mapping[str, float]]],
    cross_reconstruction: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """The gap between cross and self reconstruction: `cross - self`, per metric.

    For `mse`, `rmse` and `mae` a positive gap means that the cross reconstruction is worse; for
    `r2` and `pearson_r` a negative gap does. A gap near 0 means that the decoder of one modality
    reads the code of the other as well as it reads its own.

    Args:
        self_reconstruction: View label -> modality -> metric -> value.
        cross_reconstruction: `"a->b"` -> modality -> metric -> value. The self value that a
            direction is compared with is the one of its target view, `b`.

    Returns:
        `"a->b"` -> modality -> metric -> gap, for the cells that have both values.
    """
    gap: dict[str, dict[str, dict[str, float]]] = {}
    for direction, per_modality in cross_reconstruction.items():
        target_label = direction.split("->", 1)[1]
        for modality, metrics in per_modality.items():
            own = self_reconstruction.get(target_label, {}).get(modality)
            if own is None:
                continue
            gap.setdefault(direction, {})[modality] = {
                name: value - own[name] for name, value in metrics.items() if name in own
            }
    return gap


@dataclass
class CollapseCheck:
    """The collapse guard for one view.

    Attributes:
        active_units: The active units of the view over the held-out samples.
        num_dimensions: The width of the space.
        required: The fewest active units the view may have: the solo stage's count when it was
            given, else `min_active_units`.
        reference_active_units: The solo stage's count, `None` when none was given.
        flagged: `True` when `active_units < required`.
    """

    active_units: int
    num_dimensions: int
    required: int
    reference_active_units: int | None
    flagged: bool


def checkCollapse(
    diagnostics: Mapping[str, LatentDiagnostics],
    reference_active_units: Mapping[str, int] | None = None,
    min_active_units: int = DEFAULT_MIN_ACTIVE_UNITS,
) -> dict[str, CollapseCheck]:
    """Flag the views that have fewer active units than their solo stage.

    The reason is that a distance between two spaces is also minimized by two collapsed ones: a
    space that has lost units since its solo stage has traded information for agreement. When no
    solo count is known for a view, the check falls back to an absolute floor, `min_active_units`.

    Args:
        diagnostics: View label -> `LatentDiagnostics` on the held-out samples.
        reference_active_units: View label -> the active units of the solo stage, measured the
            same way (the same threshold, preferably the same samples).
        min_active_units: The floor for a view without a reference, at least 0.

    Returns:
        View label -> `CollapseCheck`.

    Raises:
        ValueError: If a reference names a view that is not in `diagnostics`, a count is negative,
            or `min_active_units` is negative.
    """
    if isinstance(min_active_units, bool) or min_active_units < 0:
        raise ValueError(f"min_active_units must be at least 0, got {min_active_units!r}.")
    references = dict(reference_active_units or {})
    unknown = sorted(set(references) - set(diagnostics))
    if unknown:
        raise ValueError(
            f"reference_active_units names {unknown}, which are not views of this pair "
            f"(views: {sorted(diagnostics)})."
        )
    for label, count in references.items():
        if isinstance(count, bool) or count < 0:
            raise ValueError(
                f"The solo active units of '{label}' must be at least 0, got {count!r}."
            )
    checks: dict[str, CollapseCheck] = {}
    for label, diagnostic in diagnostics.items():
        reference = references.get(label)
        required = reference if reference is not None else min_active_units
        checks[label] = CollapseCheck(
            active_units=diagnostic.active_units,
            num_dimensions=diagnostic.num_dimensions,
            required=required,
            reference_active_units=reference,
            flagged=diagnostic.active_units < required,
        )
    return checks


@dataclass
class ProcrustesResult:
    """Baseline 1: the orthogonal map fitted on the training pairs, read on the held-out ones.

    Attributes:
        center: Whether the fit was centered.
        num_fit_pairs: How many training pairs the map was fitted on.
        retrieval: `"a->b"` -> retrieval metrics of the held-out means after the map.
        alignment: Alignment metrics (`paired_alignment.computeAlignmentMetrics`) between the
            mapped first means and the second means.
        cross_reconstruction: `"a->b"` -> modality -> metric -> value, with the code of `a`
            carried by the map before the decoder of `b` reads it.
        reconstruction_gap: The gap to the self reconstruction of `b`.
    """

    center: bool
    num_fit_pairs: int
    retrieval: dict[str, dict[str, float]]
    alignment: dict[str, float]
    cross_reconstruction: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    reconstruction_gap: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


def _formatMetricValues(metrics: Mapping[str, float], names: Sequence[str]) -> str:
    """Write `name=value` for the metrics that exist."""
    return ", ".join(f"{name}={metrics[name]:.4g}" for name in names if name in metrics)


@dataclass
class LatentAlignmentReport:
    """Everything `evaluateLatentAlignment` produces for one pair of latent views.

    Attributes:
        labels: The names of the two views, in order (`["z_signal", "z_image"]`).
        views: View label -> `{"space": ..., "inputs": [...]}`.
        num_heldout: How many held-out pairs the numbers below were computed on.
        num_train: How many training pairs the Procrustes map and the integrity check used
            (`0` when none were given).
        similarity: The similarity retrieval ranked by.
        ks: The `k` of each `recall@k`.
        retrieval: `"a->b"` -> retrieval metrics (`computeRetrievalMetrics`), with the chance level.
        alignment: The alignment metrics of the two views' means (`computeAlignmentMetrics`).
        distances: Distance name -> `DistanceResult`.
        skipped_distances: Distance name -> why it was not evaluated.
        self_reconstruction: View label -> modality -> metric -> value.
        cross_reconstruction: `"a->b"` -> modality -> metric -> value.
        reconstruction_gap: `"a->b"` -> modality -> metric -> `cross - self`.
        diagnostics: View label -> `LatentDiagnostics` over the held-out samples.
        collapse: View label -> `CollapseCheck`.
        procrustes: Baseline 1, `None` when no training pairs were given.
        pair_integrity: The integrity check (`"not_checked"` without training pairs).
        flags: The warnings, one sentence each: an empty list means nothing was flagged.
        adapters: Latent space name -> `{"head", "in_dim", "out_dim", "parameters",
            "spread_ratio"}` for each space the pair was read through an adapter (ADR 0060):
            `spread_ratio` is the per-dimension spread of the posterior mean after the adapter over
            before it (`None` if there was none before). Empty without adapters.
        widths: View label -> the latent width of the view as the model has it.
        notes: What a reader must know to read the numbers, one sentence each (that the retrieval
            reads adapted posteriors, that the cross reconstruction was not computed). Not
            warnings: they are in `flags` when something is wrong.
    """

    labels: list[str]
    views: dict[str, dict[str, Any]]
    num_heldout: int
    num_train: int
    similarity: str
    ks: list[int]
    retrieval: dict[str, dict[str, float]]
    alignment: dict[str, float]
    distances: dict[str, DistanceResult]
    skipped_distances: dict[str, str]
    self_reconstruction: dict[str, dict[str, dict[str, float]]]
    cross_reconstruction: dict[str, dict[str, dict[str, float]]]
    reconstruction_gap: dict[str, dict[str, dict[str, float]]]
    diagnostics: dict[str, LatentDiagnostics]
    collapse: dict[str, CollapseCheck]
    procrustes: ProcrustesResult | None
    pair_integrity: PairIntegrity
    flags: list[str] = field(default_factory=list)
    adapters: dict[str, dict[str, Any]] = field(default_factory=dict)
    widths: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

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

    def crossReconstructionBySubset(self) -> dict[frozenset[str], dict[str, dict[str, float]]]:
        """The cross reconstruction in the shape `buildMetricsTable` reads.

        Input subset (the inputs of the view that was the source) -> modality -> metrics, so that
        "image, read from the signal alone" is the row `inputs=signal`, `modality=image`, as in
        `evaluation.cross_modal`.
        """
        return self._bySubset(self.cross_reconstruction)

    def gapBySubset(self) -> dict[frozenset[str], dict[str, dict[str, float]]]:
        """The gap between cross and self reconstruction, shaped like the cross reconstruction."""
        return self._bySubset(self.reconstruction_gap)

    def _bySubset(
        self, per_direction: Mapping[str, Mapping[str, Mapping[str, float]]]
    ) -> dict[frozenset[str], dict[str, dict[str, float]]]:
        """Re-key a `"a->b"` mapping by the input subset of its source view."""
        subsets: dict[frozenset[str], dict[str, dict[str, float]]] = {}
        for direction, per_modality in per_direction.items():
            source = direction.split("->", 1)[0]
            subset = frozenset(self.views[source]["inputs"])
            for modality, metrics in per_modality.items():
                subsets.setdefault(subset, {})[modality] = dict(metrics)
        return subsets

    def summary(self) -> str:
        """A short, human-readable multi-line summary.

        Returns:
            The views, then the flags (or a line saying that none was raised), the retrieval of
            each direction with its chance level, the distances paired against mismatched, the
            gap between cross and self reconstruction, the active units, and the baseline and the
            integrity check when there are training pairs.
        """
        first, second = self.labels
        lines = [
            f"Latent alignment: {first} ({self.views[first]['space']} from "
            f"{self.views[first]['inputs']}) and {second} ({self.views[second]['space']} from "
            f"{self.views[second]['inputs']}) over {self.num_heldout} held-out pairs"
        ]
        lines.extend(f"  FLAG: {flag}" for flag in self.flags)
        if not self.flags:
            lines.append("  no flag raised")
        lines.extend(f"  note: {note}" for note in self.notes)
        for direction, metrics in self.retrieval.items():
            recalls = ", ".join(
                f"recall@{k}={metrics[f'recall@{k}']:.3f} "
                f"(chance {metrics[f'chance_recall@{k}']:.3f})"
                for k in self.ks
            )
            lines.append(
                f"  retrieval {direction}: {recalls}, "
                + _formatMetricValues(metrics, ("mean_rank", "median_rank", "mrr"))
            )
        for name, result in self.distances.items():
            lines.append(
                f"  distance {name}: paired={result.paired:.4g}, "
                f"mismatched={result.mismatched:.4g}, ratio={result.ratio:.4g}"
            )
        for name, reason in self.skipped_distances.items():
            lines.append(f"  distance {name}: not evaluated, {reason}")
        for direction, per_modality in self.cross_reconstruction.items():
            for modality, metrics in per_modality.items():
                gap = self.reconstruction_gap.get(direction, {}).get(modality, {})
                lines.append(
                    f"  cross reconstruction {direction} ({modality}): "
                    + _formatMetricValues(metrics, ("mse", "r2"))
                    + "; gap to self: "
                    + _formatMetricValues(gap, ("mse", "r2"))
                )
        for label, check in self.collapse.items():
            reference = "solo stage" if check.reference_active_units is not None else "the floor"
            lines.append(
                f"  active units [{label}]: {check.active_units}/{check.num_dimensions} "
                f"(needs {check.required}, from {reference})"
            )
        if self.procrustes is not None:
            both = self.procrustes.retrieval
            k = self.ks[-1]
            lines.append(
                f"  procrustes baseline (fitted on {self.procrustes.num_fit_pairs} pairs): "
                + ", ".join(
                    f"{direction} recall@{k}={metrics[f'recall@{k}']:.3f}"
                    for direction, metrics in both.items()
                )
            )
        lines.append(
            f"  pair integrity: {self.pair_integrity.status}. {self.pair_integrity.message}"
        )
        lines.append(f"  {_sampleIdsSentence(self.pair_integrity.sample_ids_checked)}")
        return "\n".join(lines)

    def toMarkdown(self) -> str:
        """The report as Markdown: flags, retrieval, distances, reconstruction, units, baseline.

        Returns:
            The text, ending with a newline.
        """
        first, second = self.labels
        lines = [
            f"# Latent alignment: {first} and {second}",
            "",
            f"- `{first}`: space `{self.views[first]['space']}` read from "
            f"`{'+'.join(self.views[first]['inputs'])}`",
            f"- `{second}`: space `{self.views[second]['space']}` read from "
            f"`{'+'.join(self.views[second]['inputs'])}`",
            f"- held-out pairs: {self.num_heldout}; training pairs: {self.num_train}",
            *[f"- {note}" for note in self.notes],
            "",
            "## Flags",
            "",
        ]
        lines.extend([f"- {flag}" for flag in self.flags] or ["- none"])
        lines += ["", "## Retrieval (held-out pairs)", ""]
        header = [
            "direction",
            *[f"recall@{k} (chance)" for k in self.ks],
            "mean rank",
            "chance rank",
        ]
        lines += ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
        chance_rank = (self.num_heldout + 1) / 2
        for direction, metrics in self.retrieval.items():
            cells = [
                f"{metrics[f'recall@{k}']:.3f} ({metrics[f'chance_recall@{k}']:.3f})"
                for k in self.ks
            ]
            lines.append(
                f"| {direction} | "
                + " | ".join(cells)
                + f" | {metrics['mean_rank']:.2f} | {chance_rank:.2f} |"
            )
        lines += ["", "## Distances (held-out pairs)", ""]
        lines += [
            "| distance | paired | mismatched | ratio | separates |",
            "|---|---:|---:|---:|---|",
        ]
        for name, result in self.distances.items():
            lines.append(
                f"| {name} | {result.paired:.4g} | {result.mismatched:.4g} | {result.ratio:.4g} | "
                f"{'yes' if result.separates else 'no'} |"
            )
        lines.extend(
            f"\nNot evaluated: `{name}`, {reason}."
            for name, reason in self.skipped_distances.items()
        )
        lines += ["", "## Reconstruction (held-out pairs)", ""]
        metric_names = list(
            dict.fromkeys(
                name
                for table in (self.self_reconstruction, self.cross_reconstruction)
                for per_modality in table.values()
                for metrics in per_modality.values()
                for name in metrics
            )
        )
        lines += [
            "| kind | from | modality | " + " | ".join(metric_names) + " |",
            "|---|---|---|" + "---:|" * len(metric_names),
        ]
        rows: list[tuple[str, str, str, Mapping[str, float]]] = []
        for label, per_modality in self.self_reconstruction.items():
            rows.extend(("self", label, m, metrics) for m, metrics in per_modality.items())
        for direction, per_modality in self.cross_reconstruction.items():
            rows.extend(("cross", direction, m, metrics) for m, metrics in per_modality.items())
            rows.extend(
                ("gap", direction, m, metrics)
                for m, metrics in self.reconstruction_gap.get(direction, {}).items()
            )
        for kind, source, modality, row_metrics in rows:
            cells = [
                f"{row_metrics[name]:.4g}" if name in row_metrics else "" for name in metric_names
            ]
            lines.append(f"| {kind} | {source} | {modality} | " + " | ".join(cells) + " |")
        lines += ["", "## Collapse guard", ""]
        lines += [
            "| view | active units | width | needs | from | flagged |",
            "|---|---:|---:|---:|---|---|",
        ]
        for label, check in self.collapse.items():
            origin = "solo stage" if check.reference_active_units is not None else "floor"
            lines.append(
                f"| {label} | {check.active_units} | {check.num_dimensions} | {check.required} | "
                f"{origin} | {'yes' if check.flagged else 'no'} |"
            )
        lines += [
            "",
            "## Pair integrity",
            "",
            f"**{self.pair_integrity.status}**: {self.pair_integrity.message}",
            "",
            _sampleIdsSentence(self.pair_integrity.sample_ids_checked),
        ]
        if self.procrustes is not None:
            lines += ["", "## Procrustes baseline (held-out pairs)", ""]
            lines += [
                f"Orthogonal map fitted on {self.procrustes.num_fit_pairs} training pairs"
                f"{', centered' if self.procrustes.center else ''}.",
                "",
                "| direction | " + " | ".join(f"recall@{k}" for k in self.ks) + " | mean rank |",
                "|---|" + "---:|" * (len(self.ks) + 1),
            ]
            for direction, metrics in self.procrustes.retrieval.items():
                cells = [f"{metrics[f'recall@{k}']:.3f}" for k in self.ks]
                lines.append(
                    f"| {direction} | " + " | ".join(cells) + f" | {metrics['mean_rank']:.2f} |"
                )
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
    """One sentence for the report: were the pairs checked by the loader's own sample ids."""
    if checked > 0:
        return (
            f"Sample ids: the `sample_id.<modality>` entries of {checked} rows agreed across the "
            f"modalities, so those rows are the same samples by the loader's own account."
        )
    return (
        "Sample ids: the batches carry no `sample_id.<modality>` entries, so the pairs were not "
        "checked by id and are trusted as the loader gave them (ADR 0058)."
    )


def _collectFlags(
    collapse: Mapping[str, CollapseCheck],
    distances: Mapping[str, DistanceResult],
    integrity: PairIntegrity,
) -> list[str]:
    """The sentences of the report's flags."""
    flags: list[str] = []
    for label, check in collapse.items():
        if check.flagged:
            origin = (
                f"its solo stage had {check.reference_active_units}"
                if check.reference_active_units is not None
                else f"the floor is {check.required}"
            )
            flags.append(
                f"Collapse: '{label}' has {check.active_units} of {check.num_dimensions} active "
                f"units, fewer than required ({origin}). A distance between collapsed spaces is "
                "small for the wrong reason."
            )
    flat = sorted(name for name, result in distances.items() if not result.separates)
    if flat:
        flags.append(
            f"The distance {flat} is 0 for mismatched pairs too: it does not tell a pair from a "
            "stranger, so a small value of it proves nothing (the spaces are likely collapsed)."
        )
    if integrity.status == INTEGRITY_SUSPECT:
        flags.append(f"Pair integrity: {integrity.message}")
    return flags


def evaluateLatentAlignment(
    model: GlobalVae,
    first: LatentView | str,
    second: LatentView | str,
    heldout: Iterable[dict[str, torch.Tensor]],
    train: Iterable[dict[str, torch.Tensor]] | None = None,
    distances: Sequence[str] | None = None,
    ks: Sequence[int] = DEFAULT_RETRIEVAL_KS,
    similarity: str = "cosine",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
    max_train_samples: int | None = None,
    active_unit_threshold: float = DEFAULT_ACTIVE_UNIT_THRESHOLD,
    reference_active_units: Mapping[str, int] | None = None,
    min_active_units: int = DEFAULT_MIN_ACTIVE_UNITS,
    procrustes_center: bool = False,
    integrity_k: int | None = None,
    integrity_shuffles: int = DEFAULT_INTEGRITY_SHUFFLES,
    integrity_margin: float = DEFAULT_INTEGRITY_MARGIN,
    seed: int = 0,
    adapters: Mapping[str, nn.Module] | None = None,
    min_adapter_spread: float = DEFAULT_MIN_ADAPTER_SPREAD,
) -> LatentAlignmentReport:
    """Evaluate a pair of latent views on held-out pairs, with the collapse and integrity guards.

    `heldout` is read once. `train`, when given, is read once to fit the Procrustes baseline and to
    run the pair-integrity check; without it both are left out (the report says so) and nothing
    else changes.

    Args:
        model: The model to evaluate. `model.eval()` is called.
        first: The first view: a `LatentView`, or `space`, or `space@modality+modality`.
        second: The second view.
        heldout: Held-out batches that carry every modality of both views, row `i` of each being
            the same sample.
        train: Training pairs, in the same form, for the Procrustes baseline and the integrity
            check. `None` skips both.
        distances: The registered latent distances to evaluate. `None` (default): all of them.
        ks: The `k` of each `recall@k`.
        similarity: `"cosine"` (default) or `"l2"`, the retrieval similarity.
        chunk_size: How many queries or samples are scored at once.
        device: Batches are moved here. Defaults to `model`'s own device.
        max_samples: Use at most this many held-out samples (the first ones).
        max_train_samples: Use at most this many training samples (the first ones).
        active_unit_threshold: A dimension is active when the variance of its posterior mean over
            the held-out samples is above this value.
        reference_active_units: View label -> the active units of the solo stage, for the collapse
            guard (`checkCollapse`).
        min_active_units: The floor of the collapse guard for a view without a reference.
        procrustes_center: Fit the Procrustes map after centering each space on its training mean.
        integrity_k: The `k` of the integrity check (`checkPairIntegrity`).
        integrity_shuffles: How many shuffles of the training pairs the integrity check fits.
        integrity_margin: How far above chance the integrity recall has to be.
        seed: Seeds the stochastic distance and the integrity shuffles.
        adapters: Latent space name -> the latent head that `latent_alignment` was trained to read
            that space through (ADR 0060). With adapters the two views may have different widths:
            the retrieval, the alignment metrics and the distances read the adapted posteriors,
            which must have one width; the collapse guard reads the posteriors as the model has
            them and the adapted ones; the Procrustes baseline and the integrity check read the
            posteriors as the model has them (a partial isometry when the widths differ); and the
            cross reconstruction, which hands a code to the decoders of the other space, is not
            computed when the widths differ. `None` (default): equal widths are required.
        min_adapter_spread: The floor of an adapter's `spread_ratio`, the per-dimension spread of
            the posterior mean after the adapter over the spread before it. An adapter below it is
            flagged as possibly shrinking its output to make the distance small.

    Returns:
        The `LatentAlignmentReport`.

    Raises:
        KeyError: If a space or a distance name is not registered or not in the model.
        ValueError: If a setting is invalid, a view is not valid for the model, the two views are
            the same or are read at different widths, `adapters` names a space that is not one
            of the two, a batch lacks a modality, or `heldout` holds fewer than two samples.
    """
    resolved_ks = validateRetrievalSettings(ks, similarity, chunk_size)
    validateActiveUnitThreshold(active_unit_threshold)
    first_view = resolveLatentView(model, first)
    second_view = resolveLatentView(model, second)
    labels = pairLabels(first_view, second_view)
    if not math.isfinite(min_adapter_spread) or min_adapter_spread < 0:
        raise ValueError(
            f"`min_adapter_spread` must be a finite, non-negative number, got {min_adapter_spread}."
        )
    adapter_map = dict(adapters or {})
    stray = sorted(set(adapter_map) - {first_view.space, second_view.space})
    if stray:
        raise ValueError(
            f"`adapters` names {stray}, which is not a space of the two views "
            f"('{first_view.space}' and '{second_view.space}')."
        )
    equal_widths = not adapter_map

    mapping: OrthogonalMap | None = None
    train_posteriors: PairPosteriors | None = None
    if train is not None:
        train_posteriors = collectPairPosteriors(
            model, train, first_view, second_view, device, max_train_samples, equal_widths
        )
        mapping = fitOrthogonalProcrustes(
            train_posteriors.first_mu, train_posteriors.second_mu, procrustes_center
        )
    statistics = collectPairStatistics(
        model, heldout, first_view, second_view, device, max_samples, mapping, equal_widths
    )
    raw = statistics.posteriors
    posteriors = adaptPosteriors(raw, first_view, second_view, adapter_map) if adapter_map else raw
    _checkWidths(
        posteriors.first_mu.shape[1],
        posteriors.second_mu.shape[1],
        first_view,
        second_view,
        " once the adapters have read them" if adapter_map else "",
    )
    if posteriors.num_samples < _MIN_PAIRS:
        raise ValueError(
            f"At least {_MIN_PAIRS} held-out pairs are needed, got {posteriors.num_samples}."
        )

    forward_name, backward_name = f"{labels[0]}->{labels[1]}", f"{labels[1]}->{labels[0]}"
    retrieval = {
        forward_name: computeRetrievalMetrics(
            posteriors.first_mu, posteriors.second_mu, resolved_ks, similarity, chunk_size
        ),
        backward_name: computeRetrievalMetrics(
            posteriors.second_mu, posteriors.first_mu, resolved_ks, similarity, chunk_size
        ),
    }
    distance_values, skipped = computeDistanceTable(posteriors, distances, chunk_size, seed)
    reconstruction = statistics.reconstruction
    gap = computeReconstructionGap(
        reconstruction.self_reconstruction, reconstruction.cross_reconstruction
    )
    diagnostics = {
        labels[0]: computeLatentDiagnostics(raw.first_mu, raw.first_logvar, active_unit_threshold),
        labels[1]: computeLatentDiagnostics(
            raw.second_mu, raw.second_logvar, active_unit_threshold
        ),
    }
    for label, view, mu, logvar in (
        (labels[0], first_view, posteriors.first_mu, posteriors.first_logvar),
        (labels[1], second_view, posteriors.second_mu, posteriors.second_logvar),
    ):
        if view.space in adapter_map:
            diagnostics[f"{label} (adapted)"] = computeLatentDiagnostics(
                mu, logvar, active_unit_threshold
            )
    collapse = checkCollapse(diagnostics, reference_active_units, min_active_units)

    procrustes: ProcrustesResult | None = None
    if mapping is not None and train_posteriors is not None:
        mapped_gap = computeReconstructionGap(
            reconstruction.self_reconstruction, reconstruction.mapped_cross_reconstruction
        )
        both = _procrustesRetrieval(
            mapping, raw.first_mu, raw.second_mu, resolved_ks, similarity, chunk_size
        )
        procrustes = ProcrustesResult(
            center=mapping.center,
            num_fit_pairs=mapping.num_pairs,
            retrieval={forward_name: both["forward"], backward_name: both["backward"]},
            alignment=computeAlignmentMetrics(mapping.toSecond(raw.first_mu), raw.second_mu),
            cross_reconstruction=reconstruction.mapped_cross_reconstruction,
            reconstruction_gap=mapped_gap,
        )
        integrity = checkPairIntegrity(
            train_posteriors.first_mu,
            train_posteriors.second_mu,
            resolved_ks,
            similarity,
            chunk_size,
            procrustes_center,
            integrity_k,
            integrity_shuffles,
            integrity_margin,
            seed,
        )
    else:
        integrity = PairIntegrity(
            status=INTEGRITY_NOT_CHECKED,
            message="No training pairs were given, so the Procrustes baseline and the check that "
            "the pairs match were not run.",
            margin=integrity_margin,
        )
    integrity.sample_ids_checked = posteriors.sample_ids_checked + (
        0 if train_posteriors is None else train_posteriors.sample_ids_checked
    )
    raw_widths = {labels[0]: int(raw.first_mu.shape[1]), labels[1]: int(raw.second_mu.shape[1])}
    adapter_report, notes = _describeAdapters(
        adapter_map,
        (first_view, second_view),
        labels,
        raw,
        posteriors,
        mapping is not None,
        diagnostics,
    )
    flags = _collectFlags(collapse, distance_values, integrity)
    flags.extend(_adapterSpreadFlags(adapter_report, min_adapter_spread))

    return LatentAlignmentReport(
        labels=list(labels),
        views={
            label: {"space": view.space, "inputs": list(view.inputs)}
            for label, view in zip(labels, (first_view, second_view), strict=True)
        },
        num_heldout=posteriors.num_samples,
        num_train=0 if train_posteriors is None else train_posteriors.num_samples,
        similarity=similarity,
        ks=list(resolved_ks),
        retrieval=retrieval,
        alignment=computeAlignmentMetrics(posteriors.first_mu, posteriors.second_mu),
        distances=distance_values,
        skipped_distances=skipped,
        self_reconstruction=reconstruction.self_reconstruction,
        cross_reconstruction=reconstruction.cross_reconstruction,
        reconstruction_gap=gap,
        diagnostics=diagnostics,
        collapse=collapse,
        procrustes=procrustes,
        pair_integrity=integrity,
        flags=flags,
        adapters=adapter_report,
        widths=raw_widths,
        notes=notes,
    )


def _describeAdapters(
    adapters: Mapping[str, nn.Module],
    views: tuple[LatentView, LatentView],
    labels: tuple[str, str],
    raw: PairPosteriors,
    adapted: PairPosteriors,
    has_map: bool,
    diagnostics: Mapping[str, LatentDiagnostics],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """What the report says about the adapters, and what a reader must know to read the numbers."""
    widths = (
        (raw.first_mu.shape[1], adapted.first_mu.shape[1]),
        (raw.second_mu.shape[1], adapted.second_mu.shape[1]),
    )
    described: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    for view, label, (in_dim, out_dim) in zip(views, labels, widths, strict=True):
        adapter = adapters.get(view.space)
        if adapter is None:
            continue
        described[view.space] = {
            "head": type(adapter).__name__,
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "parameters": sum(parameter.numel() for parameter in adapter.parameters()),
            "spread_ratio": _spreadRatio(diagnostics[label], diagnostics[f"{label} (adapted)"]),
        }
        notes.append(
            f"'{label}' is read through the adapter of '{view.space}' "
            f"({type(adapter).__name__}, {in_dim} -> {out_dim})."
        )
    if not described:
        return described, notes
    notes.append(
        "The retrieval, the alignment metrics and the distances read the adapted posteriors, "
        "which the alignment term trained on the training pairs; the collapse guard reads the "
        "posteriors as the model has them, and the adapted ones as `(adapted)`."
    )
    if raw.first_mu.shape[1] != raw.second_mu.shape[1]:
        notes.append(
            f"The spaces have widths {raw.first_mu.shape[1]} and {raw.second_mu.shape[1]}: the "
            "cross reconstruction, which hands a code of one width to the decoders of the other, "
            "was not computed"
            + (
                ", and the Procrustes baseline is a partial isometry that embeds the narrower "
                "space in the wider one."
                if has_map
                else "."
            )
        )
    return described, notes


def _spreadRatio(before: LatentDiagnostics, after: LatentDiagnostics) -> float | None:
    """Per-dimension spread of the posterior mean after an adapter over before it.

    The mean over dimensions of the variance over samples of each dimension's posterior mean, so
    that a narrower and a wider space compare. `None` when the space had no spread to begin with.
    """
    start = sum(before.mu_variance_per_dimension) / max(before.num_dimensions, 1)
    end = sum(after.mu_variance_per_dimension) / max(after.num_dimensions, 1)
    return float(end / start) if start > 0 else None


def _adapterSpreadFlags(described: Mapping[str, Mapping[str, Any]], floor: float) -> list[str]:
    """One sentence per adapter whose output is much less spread than its input."""
    flags: list[str] = []
    for space, entry in described.items():
        ratio = entry["spread_ratio"]
        if ratio is not None and ratio < floor:
            flags.append(
                f"Adapter: the adapter of '{space}' leaves {ratio:.3f} of the per-dimension "
                f"spread of the posterior mean it reads (the floor is {floor}). An adapter can "
                "lower a distance by shrinking its output: read the retrieval, which does not "
                "change with the scale, before reading the distance."
            )
    return flags
