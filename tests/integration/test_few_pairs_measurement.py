"""The synthetic measurement behind ADR 0061: a few pairs, many unpaired samples.

Two views of one 3-dimensional factor `u` (`signal = u W`, `image = tanh(u V)`, a little noise),
two spaces of width 3, a small KL weight. 16 pairs, two pools of 2000 unpaired samples (one per
modality, none of them a pair) and 512 held-out pairs. Both VAEs are first trained on their own
unpaired pool, then a pair stage runs for 150 epochs:

- **A**: the pair term on the 16 pairs alone;
- **B**: the same, with unpaired batches interleaved (`interleaveBatches`, `on_unpaired: skip`);
- **C**: B plus a frozen flow prior on `z_image`, fitted on the posterior means of `z_signal` over
  the unpaired signals, at a weight (0.1 or 1.0);
- **D**: no pair term: the unpaired batches and the flow prior only.

Two worlds: `u` Gaussian (retrieval of the partner among 512), and `u` a mixture of 6 separated
clusters (the share of held-out images whose nearest signal posterior mean is of their own cluster,
chance 1/6). Three seeds each.

What is pinned is the finding of the ADR and nothing finer, in the form of inequalities:

- the pairs alone retrieve far above chance, and the unpaired batches cut the held-out
  reconstruction error of the signal VAE;
- a distribution term alone does not find the pairing (retrieval at chance), and shrinks the space
  it moves, which the active-unit guard does not see;
- the flow prior has a cost that grows with its weight (spread, retrieval, cluster accuracy), and at
  a small weight no measurable effect, on these toy worlds. It is a surrogate loss: the latent
  scores below the target's own points.

The numbers are those of a toy problem (roadmap I-15), not of any data.
"""

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from global_vae.data.interleave import interleaveBatches
from global_vae.evaluation.latent_alignment import evaluateLatentAlignment
from global_vae.evaluation.space_means import collectSpaceMeans
from global_vae.losses.priors import fitFlowPrior, saveFlowPrior
from global_vae.losses.terms.base import AbstractLossTerm
from global_vae.losses.terms.flow_prior_alignment import FlowPriorAlignmentTerm
from global_vae.losses.terms.latent_alignment import LatentAlignmentTerm
from global_vae.training.beta_schedules.base import AbstractBetaSchedule
from global_vae.training.trainer import Trainer
from tests.unit._tiny_pair_model import IN_DIM, buildTinyModel

pytestmark = pytest.mark.slow

_BATCH = 64
_BETA = 1e-3
_PAIRS = 16
_POOL = 2000
_HELD_OUT = 512
_EPOCHS = 150
_CLUSTERS = 6
_SEEDS = (0, 1, 2)
_Batch = dict[str, torch.Tensor]


@dataclass
class Outcome:
    """What one run measured on the held-out pairs."""

    recall: float
    cluster_accuracy: float
    signal_mse: float
    image_spread: float
    image_gap: float | None
    active_units: dict[str, int]
    flagged: list[str]


def _world(seed: int, clusters: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 + seed)
    weights = torch.randn(3, IN_DIM, generator=generator)
    mixing = torch.randn(3, IN_DIM, generator=generator)
    centres = 2.0 * torch.randn(max(clusters, 1), 3, generator=generator)
    return weights, mixing, centres


def _sample(
    count: int, seed: int, clusters: int, world: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights, mixing, centres = world
    generator = torch.Generator().manual_seed(seed)
    if clusters:
        label = torch.randint(clusters, (count,), generator=generator)
        factor = centres[label] + 0.3 * torch.randn(count, 3, generator=generator)
    else:
        label = torch.zeros(count, dtype=torch.long)
        factor = torch.randn(count, 3, generator=generator)
    signal = factor @ weights + 0.05 * torch.randn(count, IN_DIM, generator=generator)
    image = torch.tanh(factor @ mixing) + 0.05 * torch.randn(count, IN_DIM, generator=generator)
    return signal, image, label


def _chunks(name: str, values: torch.Tensor) -> list[_Batch]:
    return [{name: values[i : i + _BATCH]} for i in range(0, values.shape[0], _BATCH)]


def _meanOf(model: Any, modality: str, values: torch.Tensor, space: str) -> torch.Tensor:
    with torch.no_grad():
        model.eval()
        encoded = model.encode({modality: values}, use_mean=True)
        mean: torch.Tensor = encoded["latent_params"][space][0]
        return mean


def _measure(config: str, seed: int, clusters: int, flow_weight: float, tmp: Path) -> Outcome:
    torch.manual_seed(seed)
    world = _world(seed, clusters)
    signal_pool, _, _ = _sample(_POOL, 1000 + seed, clusters, world)
    _, image_pool, _ = _sample(_POOL, 2000 + seed, clusters, world)
    pair_signal, pair_image, _ = _sample(_PAIRS, 3000 + seed, clusters, world)
    held_signal, held_image, held_label = _sample(_HELD_OUT, 4000 + seed, clusters, world)
    signal_loader = _chunks("signal", signal_pool)
    image_loader = _chunks("image", image_pool)
    held = [{"signal": held_signal, "image": held_image}]
    pairs = [{"signal": pair_signal, "image": pair_image}]

    model = buildTinyModel(seed)
    solo = Trainer(model, device="cpu", beta=_BETA, optimizer_kwargs={"lr": 0.005})
    solo.fit(interleaveBatches(signal_loader, {"image": image_loader}, 1.0), 6)

    terms: dict[str, AbstractLossTerm] = {}
    weights: dict[str, float | AbstractBetaSchedule] = {}
    if config != "D":
        terms["latent_alignment"] = LatentAlignmentTerm(
            [["z_signal", "z_image"]], distance="wasserstein2_diag", on_unpaired="skip"
        )
        weights["latent_alignment"] = 1.0
    if config in ("C", "D"):
        means = collectSpaceMeans(model, "z_signal", signal_loader)
        fit = fitFlowPrior(means, dim=3, steps=1500, hidden=64, layers=2, seed=seed)
        path = saveFlowPrior(fit.prior, tmp / f"prior_{config}_{seed}.pt", fit)
        terms["flow_prior_alignment"] = FlowPriorAlignmentTerm("z_image", path, on_missing="skip")
        weights["flow_prior_alignment"] = flow_weight

    trainer = Trainer(
        model,
        device="cpu",
        beta=_BETA,
        loss_terms=terms,
        loss_term_weights=weights,
        optimizer_kwargs={"lr": 0.003},
    )
    stream: Any
    if config == "A":
        stream = pairs
    elif config in ("B", "C"):
        stream = interleaveBatches(
            pairs, {"signal": signal_loader, "image": image_loader}, unpaired_per_paired=2.0
        )
    else:
        # D has no pairs: one image batch stands in for the paired loader, each epoch is then
        # that batch and two unpaired ones.
        stream = interleaveBatches(
            image_loader[:1], {"signal": signal_loader, "image": image_loader}, 2.0
        )
    trainer.fit(stream, _EPOCHS)

    report = evaluateLatentAlignment(
        model, "z_signal", "z_image", held, train=pairs, ks=(1, 5), seed=seed
    )
    mean_signal = _meanOf(model, "signal", held_signal, "z_signal")
    mean_image = _meanOf(model, "image", held_image, "z_image")
    with torch.no_grad():
        reconstruction = model.decode({"z_signal": mean_signal})["signal"]
    nearest = torch.cdist(mean_image, mean_signal).argmin(dim=1)
    gap: float | None = None
    flow_term = terms.get("flow_prior_alignment")
    if isinstance(flow_term, FlowPriorAlignmentTerm):
        gap = float(flow_term.prior.alignmentGap(mean_image, 16, torch.Generator().manual_seed(0)))
    return Outcome(
        recall=statistics.mean(m["recall@1"] for m in report.retrieval.values()),
        cluster_accuracy=float((held_label[nearest] == held_label).float().mean()),
        signal_mse=float(((reconstruction - held_signal) ** 2).mean()),
        image_spread=float(mean_image.var(0).mean()),
        image_gap=gap,
        active_units={label: report.diagnostics[label].active_units for label in report.labels},
        flagged=sorted(label for label in report.labels if report.collapse[label].flagged),
    )


def _runs(config: str, clusters: int, flow_weight: float, tmp: Path) -> list[Outcome]:
    return [_measure(config, seed, clusters, flow_weight, tmp) for seed in _SEEDS]


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


@pytest.fixture(scope="module")
def tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("few_pairs")


@pytest.fixture(scope="module")
def pairsOnly(tmp: Path) -> list[Outcome]:
    return _runs("A", 0, 0.0, tmp)


@pytest.fixture(scope="module")
def withUnpaired(tmp: Path) -> list[Outcome]:
    return _runs("B", 0, 0.0, tmp)


@pytest.fixture(scope="module")
def strongPrior(tmp: Path) -> list[Outcome]:
    return _runs("C", 0, 1.0, tmp)


@pytest.fixture(scope="module")
def priorOnly(tmp: Path) -> list[Outcome]:
    return _runs("D", 0, 0.1, tmp)


@pytest.fixture(scope="module")
def clusteredUnpaired(tmp: Path) -> list[Outcome]:
    return _runs("B", _CLUSTERS, 0.0, tmp)


@pytest.fixture(scope="module")
def clusteredSmallPrior(tmp: Path) -> list[Outcome]:
    return _runs("C", _CLUSTERS, 0.01, tmp)


@pytest.fixture(scope="module")
def clusteredStrongPrior(tmp: Path) -> list[Outcome]:
    return _runs("C", _CLUSTERS, 1.0, tmp)


def test_sixteen_pairs_retrieve_the_partner_far_above_chance(pairsOnly: list[Outcome]) -> None:
    chance = 1.0 / _HELD_OUT
    assert _mean([run.recall for run in pairsOnly]) > 10 * chance


def test_unpaired_batches_cut_the_held_out_reconstruction_error(
    pairsOnly: list[Outcome], withUnpaired: list[Outcome]
) -> None:
    assert _mean([run.signal_mse for run in withUnpaired]) < 0.75 * _mean(
        [run.signal_mse for run in pairsOnly]
    )


def test_the_unpaired_batches_do_not_cost_retrieval(
    pairsOnly: list[Outcome], withUnpaired: list[Outcome]
) -> None:
    assert _mean([run.recall for run in withUnpaired]) > 0.8 * _mean(
        [run.recall for run in pairsOnly]
    )


def test_a_distribution_term_alone_does_not_find_the_pairing(priorOnly: list[Outcome]) -> None:
    assert all(run.recall < 0.02 for run in priorOnly)


def test_the_flow_prior_shrinks_the_space_it_moves_and_the_guard_does_not_see_it(
    withUnpaired: list[Outcome], strongPrior: list[Outcome], priorOnly: list[Outcome]
) -> None:
    for runs in (strongPrior, priorOnly):
        assert _mean([run.image_spread for run in runs]) < 0.5 * _mean(
            [run.image_spread for run in withUnpaired]
        )
        for run in runs:
            assert run.active_units["z_image"] == 3
            assert not run.flagged


def test_the_flow_prior_is_a_surrogate_the_latent_scores_below_the_target(
    strongPrior: list[Outcome], priorOnly: list[Outcome]
) -> None:
    for run in strongPrior + priorOnly:
        assert run.image_gap is not None
        assert run.image_gap < 0.0


def test_a_strong_flow_prior_costs_retrieval(
    withUnpaired: list[Outcome], strongPrior: list[Outcome]
) -> None:
    assert _mean([run.recall for run in strongPrior]) < _mean([run.recall for run in withUnpaired])


def test_clustered_data_pairs_find_the_cluster_of_the_partner(
    clusteredUnpaired: list[Outcome],
) -> None:
    assert _mean([run.cluster_accuracy for run in clusteredUnpaired]) > 0.8


def test_a_small_flow_prior_changes_nothing_measurable_on_clustered_data(
    clusteredUnpaired: list[Outcome], clusteredSmallPrior: list[Outcome]
) -> None:
    gain = _mean([run.cluster_accuracy for run in clusteredSmallPrior]) - _mean(
        [run.cluster_accuracy for run in clusteredUnpaired]
    )
    assert abs(gain) < 0.03


def test_a_strong_flow_prior_lowers_the_cluster_accuracy(
    clusteredUnpaired: list[Outcome], clusteredStrongPrior: list[Outcome]
) -> None:
    assert _mean([run.cluster_accuracy for run in clusteredStrongPrior]) < _mean(
        [run.cluster_accuracy for run in clusteredUnpaired]
    )
