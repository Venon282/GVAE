"""Metrics that describe one kind of data: point clouds, graphs, text (roadmap P5-3, ADR 0048).

The five default metrics (`mse`, `rmse`, `mae`, `r2`, `pearson_r`) pool every value into one number,
which describes an image or a spectrum and says little about a set of points, an adjacency matrix or
a sequence of tokens. These are the scores that do, registered by name in `evaluation.metrics`, so
they can be asked for through `evaluate(reconstruction_metrics=...)` and from a script, with the
same signature as every metric: `(reconstruction, target) -> float`.

- Point clouds: `chamfer`, `precision`, `recall`, `f_score`, each read from `computeCloudScores`
  with its default threshold (`0.1`, in the units of the data). Another threshold goes through a
  `functools.partial` of `computeCloudScores`, in a dict of metrics, as before.
- Graphs: `edge_auc`, the area under the ROC curve of the edge probabilities over the pairs above
  the diagonal. It has no node counts, so padding counts as pairs; a caller who has them calls
  `computeEdgeScores`.
- Text: `token_accuracy`, the share of positions where the arg max of the scores is the target
  token, target ids or one-hot, every position counted.

A target row of `NaN` in a cloud is padding, and `computeCloudScores` leaves it out already.
"""

import torch

from global_vae.evaluation.metrics import registerMetric
from global_vae.utils.graph_metrics import computeEdgeScores
from global_vae.utils.point_cloud_metrics import computeCloudScores
from global_vae.utils.text_blocks import computeTokenAccuracy


def _cloudScore(key: str, reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Return one score of `computeCloudScores` with its default threshold."""
    return computeCloudScores(reconstruction, target)[key]


@registerMetric("chamfer")
def computeChamferScore(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Chamfer distance of point clouds: mean squared distance to the nearest point, both ways.

    Args:
        reconstruction: `(batch, points, dim)`.
        target: `(batch, target_points, dim)`, a row of `NaN` is padding.

    Returns:
        The distance, in the squared units of the data, averaged over clouds.
    """
    return _cloudScore("chamfer", reconstruction, target)


@registerMetric("precision")
def computePointPrecision(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Share of reconstructed points within the default threshold of a real point.

    Args:
        reconstruction: `(batch, points, dim)`.
        target: `(batch, target_points, dim)`, a row of `NaN` is padding.

    Returns:
        The precision in `[0, 1]`.
    """
    return _cloudScore("precision", reconstruction, target)


@registerMetric("recall")
def computePointRecall(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Share of real points within the default threshold of a reconstructed point.

    Args:
        reconstruction: `(batch, points, dim)`.
        target: `(batch, target_points, dim)`, a row of `NaN` is padding.

    Returns:
        The recall in `[0, 1]`.
    """
    return _cloudScore("recall", reconstruction, target)


@registerMetric("f_score")
def computePointFScore(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Harmonic mean of the point precision and recall.

    Args:
        reconstruction: `(batch, points, dim)`.
        target: `(batch, target_points, dim)`, a row of `NaN` is padding.

    Returns:
        The F score in `[0, 1]`.
    """
    return _cloudScore("f_score", reconstruction, target)


@registerMetric("edge_auc")
def computeEdgeAuc(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """Area under the ROC curve of the edge probabilities of undirected graphs.

    Args:
        reconstruction: Edge probabilities `(batch, N, N)`.
        target: The adjacency `(batch, N, N)`, `1` for an edge.

    Returns:
        The AUC in `[0, 1]`, or `nan` if the pairs hold only edges or only non-edges.
    """
    return computeEdgeScores(reconstruction, target)["auc"]


registerMetric("token_accuracy")(computeTokenAccuracy)
