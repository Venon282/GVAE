"""The synthetic measurement behind ADR 0060: what trainable adapters do to an alignment.

Two views of one 3-dimensional factor `u` (`signal = u W`, `text = tanh(u V)`, a little noise), a
`signal` space of width 3 and a `text` space of width 5, trained with a small KL weight so that the
VAEs do not collapse by themselves (the control, with no alignment term, shows it). Three seeds of
each of three runs:

- **control**: widths 3 and 3, no alignment term;
- **one adapter**: `z_text` read through a linear head 5 -> 3, `stop_gradient: target`;
- **two adapters**: both spaces read through linear heads, 3 -> 4 and 5 -> 4, no stop gradient.

What is pinned is the finding of the ADR and nothing finer: one adapter retrieves the partner far
above the control and keeps the spread of the posterior it reads, and two trainable adapters
retrieve less and shrink their output, which the active-unit guard does not see and the spread
ratio of the report does. The numbers are those of a toy problem (roadmap I-15), not of any data.
"""

import statistics
from typing import Any

import pytest
import torch

from global_vae.evaluation.latent_alignment import LatentAlignmentReport, evaluateLatentAlignment
from global_vae.losses.terms.base import AbstractLossTerm
from global_vae.losses.terms.latent_alignment import LatentAlignmentTerm
from global_vae.training.trainer import Trainer
from tests.integration._latent_alignment_fixtures import buildTripleModel
from tests.unit._tiny_pair_model import IN_DIM

pytestmark = pytest.mark.slow

_BATCH = 64
_STEPS = 25
_EPOCHS = 16
_BETA = 1e-3
_SEEDS = (0, 1, 2)


def _batches(count: int, seed: int, world: tuple[torch.Tensor, torch.Tensor]) -> list[Any]:
    weights, mixing = world
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        factor = torch.randn(_BATCH, 3, generator=generator)
        noise = 0.05 * torch.randn(_BATCH, IN_DIM, generator=generator)
        out.append(
            {
                "signal": factor @ weights + noise,
                "text": torch.tanh(factor @ mixing)
                + 0.05 * torch.randn(_BATCH, IN_DIM, generator=generator),
            }
        )
    return out


def _measure(
    widths: dict[str, int], term_kwargs: dict[str, Any] | None, seed: int
) -> LatentAlignmentReport:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(100 + seed)
    world = (
        torch.randn(3, IN_DIM, generator=generator),
        torch.randn(3, IN_DIM, generator=generator),
    )
    model = buildTripleModel(widths, seed=seed)
    terms: dict[str, AbstractLossTerm] = {}
    if term_kwargs is not None:
        terms["latent_alignment"] = LatentAlignmentTerm(
            [["z_signal", "z_text"]], distance="wasserstein2_diag", **term_kwargs
        )
    trainer = Trainer(
        model, device="cpu", loss_terms=terms, beta=_BETA, optimizer_kwargs={"lr": 0.005}
    )
    train = _batches(_STEPS, seed + 1, world)
    trainer.fit(train, _EPOCHS)
    term = terms.get("latent_alignment")
    adapters = dict(term.adapters) if isinstance(term, LatentAlignmentTerm) else {}
    return evaluateLatentAlignment(
        model,
        "z_signal",
        "z_text",
        _batches(6, seed + 50, world),
        adapters=adapters or None,
        ks=(1, 5),
        seed=seed,
    )


def _recall(reports: list[LatentAlignmentReport]) -> float:
    return statistics.mean(
        metrics["recall@1"] for report in reports for metrics in report.retrieval.values()
    )


@pytest.fixture(scope="module")
def control() -> list[LatentAlignmentReport]:
    return [_measure({"signal": 3, "text": 3}, None, seed) for seed in _SEEDS]


@pytest.fixture(scope="module")
def one_adapter() -> list[LatentAlignmentReport]:
    kwargs = {"stop_gradient": "target", "adapters": {"z_text": 3}}
    return [_measure({"signal": 3, "text": 5}, kwargs, seed) for seed in _SEEDS]


@pytest.fixture(scope="module")
def two_adapters() -> list[LatentAlignmentReport]:
    kwargs = {"adapters": {"z_signal": 4, "z_text": 4}}
    return [_measure({"signal": 3, "text": 5}, kwargs, seed) for seed in _SEEDS]


def test_without_an_alignment_term_the_spaces_do_not_retrieve_each_other(
    control: list[LatentAlignmentReport],
) -> None:
    assert _recall(control) < 0.05
    assert not any(report.collapse[label].flagged for report in control for label in report.labels)


def test_one_adapter_retrieves_the_partner_far_above_the_control(
    control: list[LatentAlignmentReport], one_adapter: list[LatentAlignmentReport]
) -> None:
    assert _recall(one_adapter) > 0.2
    assert _recall(one_adapter) > 5 * _recall(control)


def test_one_adapter_keeps_the_spread_of_what_it_reads(
    one_adapter: list[LatentAlignmentReport],
) -> None:
    for report in one_adapter:
        assert report.adapters["z_text"]["spread_ratio"] > 0.3
        assert not [flag for flag in report.flags if flag.startswith("Adapter:")]


def test_two_trainable_adapters_retrieve_less_than_one(
    one_adapter: list[LatentAlignmentReport], two_adapters: list[LatentAlignmentReport]
) -> None:
    assert _recall(two_adapters) < _recall(one_adapter) / 2


def test_two_trainable_adapters_shrink_their_output_and_the_report_says_so(
    two_adapters: list[LatentAlignmentReport],
) -> None:
    for report in two_adapters:
        for space in ("z_signal", "z_text"):
            assert report.adapters[space]["spread_ratio"] < 0.2, space
        assert [flag for flag in report.flags if flag.startswith("Adapter:")]
        # What the active-unit guard cannot see: no adapted space is flagged as collapsed.
        assert not report.collapse["z_signal (adapted)"].flagged
        assert not report.collapse["z_text (adapted)"].flagged
