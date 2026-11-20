"""A loader that joins two sources and a pair stage that checks the join (roadmap P6-8, ADR 0058).

The scenario of roadmap finding I-8: two modalities that come from two sources, each with its own
list of sample names, joined by a `torch.utils.data.Dataset`. A loader that joins them by position
works while both lists are sorted the same way and breaks quietly when one is not. The loader here
gives each modality's ids (`sample_id.<modality>`, from `hashSampleIds` of that modality's own
names) and goes through the default collate of a `DataLoader`, so the entries are what a real
loader would hand over. What is pinned:

- a loader whose two sources agree trains a pair stage, and the evaluation counts its rows;
- a loader whose second source is in another order stops the first step of the pair stage, and the
  evaluation, with a message that names the rows and the ids each source gave them;
- without the ids the same wrong loader is not noticed by the term, which is what the ids add.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from global_vae.data.sample_ids import hashSampleIds
from global_vae.evaluation.latent_alignment import evaluateLatentAlignment
from global_vae.losses.terms.cross_reconstruction import CrossReconstructionTerm
from global_vae.losses.terms.latent_alignment import LatentAlignmentTerm
from global_vae.training.trainer import Trainer
from tests.unit._tiny_pair_model import IN_DIM, buildTinyModel

SAMPLES = 24
BATCH_SIZE = 8


class _JoinedSources(Dataset[dict[str, torch.Tensor]]):
    """Two sources joined by position, each with the names of its own rows."""

    def __init__(self, second_order: list[int] | None = None, with_ids: bool = True) -> None:
        generator = torch.Generator().manual_seed(0)
        self.signal = torch.randn(SAMPLES, IN_DIM, generator=generator)
        # The image of sample `i` is the signal of sample `i` (so a correct join is learnable).
        self.image = self.signal.clone()
        self.signal_names = [f"sample_{i:03d}" for i in range(SAMPLES)]
        self.image_names = list(self.signal_names)
        if second_order is not None:
            self.image = self.image[second_order]
            self.image_names = [self.image_names[i] for i in second_order]
        self.with_ids = with_ids
        self.signal_ids = hashSampleIds(self.signal_names)
        self.image_ids = hashSampleIds(self.image_names)

    def __len__(self) -> int:
        return SAMPLES

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = {"signal": self.signal[index], "image": self.image[index]}
        if self.with_ids:
            row["sample_id.signal"] = self.signal_ids[index]
            row["sample_id.image"] = self.image_ids[index]
        return row


def _loader(second_order: list[int] | None = None, with_ids: bool = True) -> DataLoader[object]:
    return DataLoader(_JoinedSources(second_order, with_ids), batch_size=BATCH_SIZE, shuffle=False)


def _trainer() -> Trainer:
    terms = {
        "latent_alignment": LatentAlignmentTerm(
            pairs=[["z_signal", "z_image"]], distance="mean_mse"
        ),
        "cross_reconstruction": CrossReconstructionTerm(
            pairs=[["z_signal", "z_image"], ["z_image", "z_signal"]]
        ),
    }
    return Trainer(buildTinyModel(0), device="cpu", loss_terms=terms)


_REVERSED = list(reversed(range(SAMPLES)))


class TestALoaderThatJoinsTwoSources:
    def test_the_collate_gives_one_integer_id_per_row_and_modality(self) -> None:
        batch = next(iter(_loader()))
        assert batch["sample_id.signal"].shape == (BATCH_SIZE,)
        assert batch["sample_id.signal"].dtype == torch.int64
        assert torch.equal(batch["sample_id.signal"], batch["sample_id.image"])

    def test_a_correct_join_trains_the_pair_stage(self) -> None:
        trainer = _trainer()
        trainer.fit(_loader(), num_epochs=2, val_dataloader=_loader())
        assert "train/loss/latent_alignment" in trainer.history[-1]

    def test_a_join_in_another_order_stops_the_first_step(self) -> None:
        trainer = _trainer()
        with pytest.raises(ValueError, match=r"not the same sample.*pairs its sources"):
            trainer.fit(_loader(_REVERSED), num_epochs=1)
        assert trainer.global_step == 0

    def test_the_message_gives_the_row_and_the_id_each_source_gave_it(self) -> None:
        loader = _loader(_REVERSED)
        batch = next(iter(loader))
        signal_id, image_id = int(batch["sample_id.signal"][0]), int(batch["sample_id.image"][0])
        with pytest.raises(ValueError, match=rf"row 0 \(signal {signal_id}, image {image_id}\)"):
            _trainer().fit(loader, num_epochs=1)

    def test_without_the_ids_the_same_wrong_join_goes_through_the_term(self) -> None:
        # What the ids add: the tensors alone cannot tell, so the stage trains on strangers.
        trainer = _trainer()
        trainer.fit(_loader(_REVERSED, with_ids=False), num_epochs=1)
        assert trainer.global_step > 0

    def test_the_evaluation_counts_the_rows_of_a_correct_join(self) -> None:
        report = evaluateLatentAlignment(
            buildTinyModel(0), "z_signal", "z_image", _loader(), train=_loader()
        )
        assert report.pair_integrity.sample_ids_checked == 2 * SAMPLES

    def test_the_evaluation_stops_on_a_join_in_another_order(self) -> None:
        with pytest.raises(ValueError, match="Evaluating a pair of latent spaces"):
            evaluateLatentAlignment(
                buildTinyModel(0), "z_signal", "z_image", _loader(_REVERSED), train=_loader()
            )
