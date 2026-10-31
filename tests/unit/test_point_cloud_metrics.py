"""Unit tests for the Chamfer distance and the cloud scores (roadmap P3-8, ADR 0044)."""

import pytest
import torch

from global_vae.utils.autograd import backward
from global_vae.utils.point_cloud_metrics import computeChamferDistance, computeCloudScores

NAN = float("nan")


def _cloud(rows: int = 3, points: int = 12, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, points, 3, generator=generator)


def _bruteForceChamfer(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    """The definition, one cloud at a time: nearest squared distances in both directions, summed."""
    totals = []
    for guess, truth in zip(reconstruction, target, strict=True):
        truth = truth[~torch.isnan(truth).any(dim=-1)]
        total = 0.0
        for point in guess:
            total += float(((truth - point) ** 2).sum(dim=-1).min())
        for point in truth:
            total += float(((guess - point) ** 2).sum(dim=-1).min())
        totals.append(total)
    return sum(totals) / len(totals)


def _padded(cloud: torch.Tensor, extra: int) -> torch.Tensor:
    """`cloud` followed by `extra` rows of `NaN`, the padding of a cloud with fewer points."""
    padding = torch.full((cloud.shape[0], extra, cloud.shape[2]), NAN)
    return torch.cat([cloud, padding], dim=1)


class TestChamferDistance:
    def test_a_cloud_that_is_reproduced_scores_zero(self) -> None:
        cloud = _cloud()
        assert float(computeChamferDistance(cloud, cloud)) == pytest.approx(0.0, abs=1e-5)

    def test_the_order_of_the_points_does_not_matter(self) -> None:
        cloud = _cloud()
        shuffled = cloud[
            :, torch.randperm(cloud.shape[1], generator=torch.Generator().manual_seed(1))
        ]
        assert float(computeChamferDistance(shuffled, cloud)) == pytest.approx(0.0, abs=1e-5)

    def test_it_matches_the_definition_on_random_clouds(self) -> None:
        reconstruction, target = _cloud(seed=1), _cloud(points=9, seed=2)
        assert float(computeChamferDistance(reconstruction, target)) == pytest.approx(
            _bruteForceChamfer(reconstruction, target), rel=1e-4
        )

    def test_it_is_symmetric_when_both_clouds_have_the_same_number_of_points(self) -> None:
        a, b = _cloud(seed=1), _cloud(seed=2)
        assert float(computeChamferDistance(a, b)) == pytest.approx(
            float(computeChamferDistance(b, a)), rel=1e-5
        )

    def test_rows_of_nan_in_the_target_are_not_points(self) -> None:
        reconstruction, target = _cloud(seed=1), _cloud(points=9, seed=2)
        padded = _padded(target, 5)
        assert float(computeChamferDistance(reconstruction, padded)) == pytest.approx(
            float(computeChamferDistance(reconstruction, target)), rel=1e-5
        )

    def test_a_row_with_a_nan_in_one_coordinate_is_not_a_point_either(self) -> None:
        reconstruction, target = _cloud(seed=1), _cloud(points=9, seed=2)
        damaged = target.clone()
        damaged[:, 4, 1] = NAN
        kept = torch.cat([target[:, :4], target[:, 5:]], dim=1)
        assert float(computeChamferDistance(reconstruction, damaged)) == pytest.approx(
            float(computeChamferDistance(reconstruction, kept)), rel=1e-5
        )

    def test_each_cloud_of_a_batch_has_its_own_number_of_points(self) -> None:
        reconstruction = _cloud(rows=2, points=10, seed=1)
        target = _cloud(rows=2, points=8, seed=2)
        target[1, 5:] = NAN
        expected = (
            _bruteForceChamfer(reconstruction[:1], target[:1])
            + _bruteForceChamfer(reconstruction[1:], target[1:])
        ) / 2
        assert float(computeChamferDistance(reconstruction, target)) == pytest.approx(
            expected, rel=1e-4
        )

    def test_the_loss_does_not_depend_on_padding_the_reconstruction_never_sees(self) -> None:
        """The reconstruction has a fixed number of points, so only the target is padded."""
        reconstruction, target = _cloud(points=16, seed=1), _padded(_cloud(points=6, seed=2), 10)
        value = computeChamferDistance(reconstruction, target)
        assert torch.isfinite(value)
        assert value.dim() == 0

    def test_the_gradient_is_finite_and_a_perfect_cloud_has_none(self) -> None:
        guess = _cloud(seed=1).requires_grad_()
        backward(computeChamferDistance(guess, _padded(_cloud(points=8, seed=2), 4)))
        assert guess.grad is not None
        assert torch.isfinite(guess.grad).all()
        perfect = _cloud(seed=3).requires_grad_()
        backward(computeChamferDistance(perfect, perfect.detach()))
        assert perfect.grad is not None
        assert float(perfect.grad.abs().max()) < 1e-3

    def test_moving_a_point_away_from_the_cloud_raises_the_loss(self) -> None:
        cloud = _cloud()
        moved = cloud.clone()
        moved[:, 0] += 5.0
        assert computeChamferDistance(moved, cloud) > computeChamferDistance(cloud, cloud)

    def test_it_is_the_sum_over_points_and_not_the_mean(self) -> None:
        """A cloud twice as dense in the same places gives about twice the sum."""
        target = _cloud(rows=1, points=20, seed=1)
        guess = target + 0.3
        doubled_guess = torch.cat([guess, guess], dim=1)
        doubled_target = torch.cat([target, target], dim=1)
        assert float(computeChamferDistance(doubled_guess, doubled_target)) == pytest.approx(
            2 * float(computeChamferDistance(guess, target)), rel=1e-4
        )

    @pytest.mark.parametrize(
        "reconstruction, target, match",
        [
            (torch.zeros(4, 3), torch.zeros(4, 3), r"\(batch, points, dim\)"),
            (torch.zeros(2, 4, 3), torch.zeros(3, 4, 3), "batch size"),
            (torch.zeros(2, 4, 3), torch.zeros(2, 4, 2), "point dimension"),
        ],
    )
    def test_clouds_that_do_not_fit_are_refused_with_both_shapes(
        self, reconstruction: torch.Tensor, target: torch.Tensor, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            computeChamferDistance(reconstruction, target)

    def test_a_target_cloud_with_no_point_is_refused(self) -> None:
        target = _cloud(rows=2)
        target[1] = NAN
        with pytest.raises(ValueError, match="at least one point"):
            computeChamferDistance(_cloud(rows=2), target)


class TestCloudScores:
    def test_a_perfect_reconstruction_scores_perfectly(self) -> None:
        cloud = _cloud()
        scores = computeCloudScores(cloud, cloud)
        assert scores["chamfer"] == pytest.approx(0.0, abs=1e-5)
        assert scores["precision"] == scores["recall"] == scores["f_score"] == 1.0
        assert scores["num_points"] == 36

    def test_the_order_of_the_points_does_not_matter(self) -> None:
        cloud = _cloud()
        flipped = cloud.flip(dims=(1,))
        assert computeCloudScores(flipped, cloud)["f_score"] == 1.0

    def test_the_chamfer_is_per_point_and_in_both_directions(self) -> None:
        reconstruction, target = _cloud(seed=1), _cloud(points=9, seed=2)
        scores = computeCloudScores(reconstruction, target)
        points = reconstruction.shape[1], target.shape[1]
        expected = 0.0
        for guess, truth in zip(reconstruction, target, strict=True):
            for point in guess:
                expected += float(((truth - point) ** 2).sum(dim=-1).min()) / points[0]
            for point in truth:
                expected += float(((guess - point) ** 2).sum(dim=-1).min()) / points[1]
        assert scores["chamfer"] == pytest.approx(expected / reconstruction.shape[0], rel=1e-4)

    def test_padded_target_rows_are_not_counted(self) -> None:
        reconstruction, target = _cloud(seed=1), _cloud(points=9, seed=2)
        plain = computeCloudScores(reconstruction, target)
        padded = computeCloudScores(reconstruction, _padded(target, 6))
        assert padded["chamfer"] == pytest.approx(plain["chamfer"], rel=1e-5)
        assert padded["num_points"] == plain["num_points"]

    def test_the_threshold_decides_precision_and_recall(self) -> None:
        cloud = _cloud()
        shifted = cloud + 0.05
        near = computeCloudScores(shifted, cloud, threshold=0.2)
        far = computeCloudScores(shifted, cloud, threshold=0.01)
        assert near["f_score"] == 1.0
        assert far["f_score"] == 0.0

    def test_a_cloud_of_the_wrong_place_has_a_zero_f_score(self) -> None:
        scores = computeCloudScores(_cloud() + 100.0, _cloud())
        assert scores["precision"] == scores["recall"] == scores["f_score"] == 0.0

    def test_a_threshold_that_is_not_positive_is_refused(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            computeCloudScores(_cloud(), _cloud(), threshold=0.0)

    def test_clouds_that_do_not_fit_are_refused(self) -> None:
        with pytest.raises(ValueError, match="point dimension"):
            computeCloudScores(torch.zeros(2, 4, 3), torch.zeros(2, 4, 2))
