"""The frozen flow prior and its alignment loss (roadmap P6-8, ADR 0061)."""

from pathlib import Path

import pytest
import torch

from global_vae.losses.priors import FlowPrior, fitFlowPrior, loadFlowPrior, saveFlowPrior
from global_vae.losses.priors.flow_prior import FLOW_PRIOR_FORMAT, FLOW_PRIOR_VERSION
from global_vae.utils.autograd import backward


def _ring(count: int = 1500, seed: int = 0, radius: float = 2.0) -> torch.Tensor:
    """A noisy ring in two dimensions: a distribution that is not a Gaussian."""
    generator = torch.Generator().manual_seed(seed)
    angle = torch.rand(count, generator=generator) * 2 * torch.pi
    points = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1) * radius
    return points + 0.1 * torch.randn(count, 2, generator=generator)


@pytest.fixture(scope="module")
def fitted() -> FlowPrior:
    return fitFlowPrior(_ring(), steps=800, hidden=64, layers=2, seed=0).prior


class TestConstruction:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dim": 0},
            {"dim": 2, "hidden": 0},
            {"dim": 2, "layers": 0},
            {"dim": 2, "time_features": 0},
        ],
    )
    def test_an_integer_below_its_minimum_is_refused_by_name(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="FlowPrior: .* must be an integer"):
            FlowPrior(**kwargs)

    @pytest.mark.parametrize("value", [True, 2.5, "3"])
    def test_a_value_that_is_not_an_integer_is_refused(self, value: object) -> None:
        with pytest.raises(ValueError, match="dim must be an integer"):
            FlowPrior(dim=value)  # type: ignore[arg-type]

    def test_a_bad_source_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="source_dim"):
            FlowPrior(dim=2, source_dim=0)

    def test_the_velocity_has_the_shape_of_its_input(self) -> None:
        prior = FlowPrior(dim=3, hidden=8, layers=2)
        out = prior.velocity(torch.randn(5, 3), torch.rand(5, 1))
        assert out.shape == (5, 3)

    def test_an_unfitted_prior_has_no_reference_loss(self) -> None:
        assert torch.isnan(FlowPrior(dim=2).reference_loss)


class TestAlignmentLoss:
    def test_it_is_a_scalar_that_the_latent_can_be_trained_through(self, fitted: FlowPrior) -> None:
        y = torch.randn(32, 2, requires_grad=True)
        loss = fitted.alignmentLoss(y)
        assert loss.dim() == 0
        backward(loss)
        assert y.grad is not None and torch.isfinite(y.grad).all() and y.grad.abs().sum() > 0

    def test_the_fitted_flow_is_frozen_and_gets_no_gradient(self, fitted: FlowPrior) -> None:
        assert not any(p.requires_grad for p in fitted.parameters())
        y = torch.randn(8, 2, requires_grad=True)
        backward(fitted.alignmentLoss(y))
        assert all(p.grad is None for p in fitted.parameters())

    def test_the_same_generator_state_gives_the_same_loss(self, fitted: FlowPrior) -> None:
        y = torch.randn(16, 2)
        first = fitted.alignmentLoss(y, 2, torch.Generator().manual_seed(3))
        second = fitted.alignmentLoss(y, 2, torch.Generator().manual_seed(3))
        assert torch.equal(first, second)

    def test_more_draws_average_the_noise_down(self, fitted: FlowPrior) -> None:
        y = _ring(64, seed=5)
        few = torch.stack(
            [fitted.alignmentLoss(y, 1, torch.Generator().manual_seed(s)) for s in range(20)]
        )
        many = torch.stack(
            [fitted.alignmentLoss(y, 16, torch.Generator().manual_seed(s)) for s in range(20)]
        )
        assert many.std() < few.std()

    @pytest.mark.parametrize("shape", [(4,), (4, 3), (4, 2, 1)])
    def test_latents_of_another_shape_are_refused(
        self, fitted: FlowPrior, shape: tuple[int, ...]
    ) -> None:
        with pytest.raises(ValueError, match=r"must have shape \(B, 2\)"):
            fitted.alignmentLoss(torch.zeros(shape))

    def test_no_draws_are_refused(self, fitted: FlowPrior) -> None:
        with pytest.raises(ValueError, match="num_draws"):
            fitted.alignmentLoss(torch.zeros(4, 2), 0)


class TestWhatTheLossSeparates:
    """The gap is near zero for the target and large for a distribution that is not it."""

    def test_fresh_samples_of_the_target_score_near_the_reference(self, fitted: FlowPrior) -> None:
        gap = fitted.alignmentGap(_ring(1024, seed=9), generator=torch.Generator().manual_seed(0))
        assert abs(gap) < 0.3

    @pytest.mark.parametrize("shift", [3.0, 6.0])
    def test_a_cloud_far_from_the_target_scores_well_above_it(
        self, fitted: FlowPrior, shift: float
    ) -> None:
        moved = _ring(1024, seed=9) + shift
        near = fitted.alignmentGap(_ring(1024, seed=9), generator=torch.Generator().manual_seed(0))
        far = fitted.alignmentGap(moved, generator=torch.Generator().manual_seed(0))
        assert far > near + 2.0

    def test_a_cloud_shrunk_toward_the_middle_scores_below_the_target(
        self, fitted: FlowPrior
    ) -> None:
        # The loss is a surrogate of a log-likelihood, not one: this is the property that the
        # other terms of the objective have to counterweight (ADR 0061).
        target = _ring(1024, seed=9)
        on_target = fitted.alignmentGap(target, 16, torch.Generator().manual_seed(0))
        for factor in (0.5, 0.0):
            shrunk = fitted.alignmentGap(target * factor, 16, torch.Generator().manual_seed(0))
            assert shrunk < on_target - 0.3

    def test_descending_the_loss_moves_a_cloud_toward_the_target(self, fitted: FlowPrior) -> None:
        y = (_ring(256, seed=2) + 4.0).requires_grad_(True)
        optimizer = torch.optim.Adam([y], lr=0.05)
        before = fitted.alignmentGap(y.detach(), generator=torch.Generator().manual_seed(0))
        for _ in range(150):
            optimizer.zero_grad()
            backward(fitted.alignmentLoss(y))
            optimizer.step()
        after = fitted.alignmentGap(y.detach(), generator=torch.Generator().manual_seed(0))
        assert after < 0.5 * before
        assert (y.detach().mean(dim=0) - _ring().mean(dim=0)).norm() < 2.5


class TestFit:
    def test_the_fit_reports_what_it_did(self) -> None:
        fit = fitFlowPrior(_ring(500), steps=300, hidden=32, layers=2)
        assert (fit.num_points, fit.num_held_out, fit.steps) == (500, 50, 300)
        assert fit.final_loss < fit.initial_loss
        assert fit.held_out_loss is not None and fit.reference_loss == fit.held_out_loss
        assert float(fit.prior.reference_loss) == pytest.approx(fit.reference_loss)

    def test_a_small_set_holds_nothing_out_and_says_so(self) -> None:
        fit = fitFlowPrior(_ring(60), steps=50, hidden=16, layers=1)
        assert fit.num_held_out == 0 and fit.held_out_loss is None
        assert fit.reference_loss == pytest.approx(float(fit.prior.reference_loss))

    def test_the_same_seed_gives_the_same_prior(self) -> None:
        first = fitFlowPrior(_ring(200), steps=40, hidden=16, layers=1, seed=4).prior
        second = fitFlowPrior(_ring(200), steps=40, hidden=16, layers=1, seed=4).prior
        for (name, a), (_, b) in zip(
            first.state_dict().items(), second.state_dict().items(), strict=True
        ):
            assert torch.equal(a, b), name

    def test_a_fit_leaves_the_global_random_stream_where_it_was(self) -> None:
        torch.manual_seed(11)
        expected = torch.rand(3)
        torch.manual_seed(11)
        fitFlowPrior(_ring(100), steps=5, hidden=8, layers=1)
        assert torch.equal(torch.rand(3), expected)

    def test_another_seed_gives_another_prior(self) -> None:
        first = fitFlowPrior(_ring(200), steps=40, hidden=16, layers=1, seed=1).prior
        second = fitFlowPrior(_ring(200), steps=40, hidden=16, layers=1, seed=2).prior
        assert not torch.equal(
            first.state_dict()["net.0.weight"], second.state_dict()["net.0.weight"]
        )

    def test_the_target_is_standardized_by_its_own_mean_and_scale(self) -> None:
        points = _ring(400) * 5.0 + 10.0
        prior = fitFlowPrior(points, steps=20, hidden=8, layers=1).prior
        assert prior.mean.shape == (2,) and prior.scale.dim() == 0
        assert torch.allclose(prior.mean, points.mean(dim=0), atol=1.5)
        assert 5.0 < float(prior.scale) < 15.0

    def test_another_width_gets_a_fixed_random_projection(self) -> None:
        points = torch.randn(300, 6)
        fit = fitFlowPrior(points, dim=3, steps=20, hidden=8, layers=1, seed=7)
        prior = fit.prior
        assert prior.dim == 3 and prior.source_dim == 6
        assert prior.projection is not None and prior.projection.shape == (6, 3)
        assert prior.prepare(points).shape == (300, 3)
        again = fitFlowPrior(points, dim=3, steps=20, hidden=8, layers=1, seed=7).prior
        assert again.projection is not None and torch.equal(prior.projection, again.projection)

    def test_the_projection_keeps_the_scale_of_the_points_on_average(self) -> None:
        points = torch.randn(2000, 8)
        prior = fitFlowPrior(points, dim=4, steps=5, hidden=8, layers=1).prior
        projected = prior.prepare(points).pow(2).sum(dim=1).mean()
        assert 0.8 < float(projected / points.pow(2).sum(dim=1).mean()) < 1.2

    def test_a_matching_width_needs_no_projection(self) -> None:
        prior = fitFlowPrior(torch.randn(50, 3), dim=3, steps=5, hidden=8, layers=1).prior
        assert prior.projection is None and prior.source_dim is None

    def test_points_of_the_wrong_width_are_refused_by_name(self) -> None:
        prior = fitFlowPrior(torch.randn(50, 6), dim=3, steps=5, hidden=8, layers=1).prior
        with pytest.raises(ValueError, match=r"expected target points of shape \(N, 6\)"):
            prior.prepare(torch.randn(4, 3))

    @pytest.mark.parametrize(
        ("points", "match"),
        [
            (torch.randn(5, 2), "at least 8 points"),
            (torch.randn(20), "an \\(N, width\\) tensor"),
            (torch.full((20, 2), float("nan")), "nan or an infinity"),
        ],
    )
    def test_points_that_cannot_be_fitted_are_refused(
        self, points: torch.Tensor, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            fitFlowPrior(points)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"steps": 0}, "steps must be an integer"),
            ({"batch_size": 0}, "batch_size must be an integer"),
            ({"lr": 0.0}, "lr must be a finite number above 0"),
            ({"lr": float("nan")}, "lr must be a finite number above 0"),
        ],
    )
    def test_a_setting_out_of_range_is_refused(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            fitFlowPrior(torch.randn(20, 2), **kwargs)  # type: ignore[arg-type]


class TestFile:
    def test_a_prior_survives_a_round_trip(self, fitted: FlowPrior, tmp_path: Path) -> None:
        path = saveFlowPrior(fitted, tmp_path / "deep" / "prior.pt")
        loaded = loadFlowPrior(path)
        y = _ring(64, seed=3)
        first = fitted.alignmentLoss(y, 2, torch.Generator().manual_seed(0))
        second = loaded.alignmentLoss(y, 2, torch.Generator().manual_seed(0))
        assert torch.equal(first, second)
        assert float(loaded.reference_loss) == pytest.approx(float(fitted.reference_loss))
        assert not any(p.requires_grad for p in loaded.parameters())

    def test_a_projection_survives_a_round_trip(self, tmp_path: Path) -> None:
        fit = fitFlowPrior(torch.randn(100, 5), dim=2, steps=10, hidden=8, layers=1)
        loaded = loadFlowPrior(saveFlowPrior(fit.prior, tmp_path / "p.pt", fit))
        assert loaded.source_dim == 5 and loaded.projection is not None
        assert fit.prior.projection is not None
        assert torch.equal(loaded.projection, fit.prior.projection)

    def test_the_file_is_read_with_the_restricted_unpickler(
        self, fitted: FlowPrior, tmp_path: Path
    ) -> None:
        path = saveFlowPrior(fitted, tmp_path / "prior.pt")
        content = torch.load(path, weights_only=True)
        assert content["format"] == FLOW_PRIOR_FORMAT and content["version"] == FLOW_PRIOR_VERSION

    def test_a_missing_file_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No flow prior file at"):
            loadFlowPrior(tmp_path / "nothing.pt")

    def test_a_file_that_is_not_torch_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.pt"
        path.write_text("not a torch file")
        with pytest.raises(ValueError, match="could not be read as a flow prior file"):
            loadFlowPrior(path)

    def test_another_torch_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "other.pt"
        torch.save({"weights": torch.zeros(2)}, path)
        with pytest.raises(ValueError, match="is not a flow prior file"):
            loadFlowPrior(path)

    def test_another_version_is_refused(self, fitted: FlowPrior, tmp_path: Path) -> None:
        path = saveFlowPrior(fitted, tmp_path / "prior.pt")
        content = torch.load(path, weights_only=True)
        content["version"] = 99
        torch.save(content, path)
        with pytest.raises(ValueError, match="version 99"):
            loadFlowPrior(path)

    def test_a_config_with_the_wrong_keys_is_refused(
        self, fitted: FlowPrior, tmp_path: Path
    ) -> None:
        path = saveFlowPrior(fitted, tmp_path / "prior.pt")
        content = torch.load(path, weights_only=True)
        content["config"] = {"dim": 2}
        torch.save(content, path)
        with pytest.raises(ValueError, match="exactly the keys"):
            loadFlowPrior(path)

    def test_weights_that_do_not_fit_the_config_are_refused(
        self, fitted: FlowPrior, tmp_path: Path
    ) -> None:
        path = saveFlowPrior(fitted, tmp_path / "prior.pt")
        content = torch.load(path, weights_only=True)
        content["config"] = {**content["config"], "hidden": 7}
        torch.save(content, path)
        with pytest.raises(ValueError, match="do not fit the stored config"):
            loadFlowPrior(path)
