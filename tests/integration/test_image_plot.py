"""Integration tests for `visualization.image_plot` (roadmap P3-4) and its exporter wiring.

The plot functions are exercised on hand-built tensors, so every assertion can name the pixels
it expects to find on an axis; the exporters (`exportEvaluationFigures`,
`exportCrossModalFigures`) are exercised on a real tiny model with 1D and 2D CNN pairs, the
shapes an actual image model produces. Everything runs on the non-interactive Agg backend.
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from global_vae.evaluation.cross_modal import exportCrossModalFigures
from global_vae.evaluation.visual_export import exportEvaluationFigures
from global_vae.models.global_vae import GlobalVae
from global_vae.visualization.image_plot import (
    isImageBatch,
    plotImageCrossModalReconstructionMatrix,
    plotImageReconstructionGrid,
)
from global_vae.visualization.reconstruction_plot import (
    collectCrossModalReconstructions,
    collectReconstructions,
    plotCrossModalReconstructionMatrix,
    plotReconstructionGrid,
)
from tests.integration._script_fixtures import (
    IMAGE_SIDE,
    SIGNAL_LENGTH,
    buildSignalImageDataloaderForScript,
    buildSignalImageModelForScript,
)

NUM_SAMPLES = 5
HEIGHT = 6
WIDTH = 4


@pytest.fixture(autouse=True)
def _closeFigures() -> Iterator[None]:
    yield
    plt.close("all")


def _images(seed: int, shape: tuple[int, ...] = (NUM_SAMPLES, HEIGHT, WIDTH)) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=generator)


def _shown(ax: Axes) -> np.ndarray:
    """The array an axis displays through `imshow`."""
    assert len(ax.images) == 1
    return np.asarray(ax.images[0].get_array())


def _clim(ax: Axes) -> tuple[float, float]:
    low, high = ax.images[0].get_clim()
    return float(low), float(high)


def _collected(
    decoder: str = "image_out", subsets: tuple[frozenset[str], ...] | None = None
) -> dict[frozenset[str], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Hand-built `collectCrossModalReconstructions` output with one decoder."""
    chosen = subsets or (
        frozenset({"signal"}),
        frozenset({"image"}),
        frozenset({"image", "signal"}),
    )
    target = _images(0)
    return {subset: {decoder: (target, _images(10 + index))} for index, subset in enumerate(chosen)}


class TestIsImageBatch:
    def test_series_batch_is_not_an_image_batch(self) -> None:
        assert not isImageBatch(torch.zeros(3, 32))

    def test_a_vector_is_not_an_image_batch(self) -> None:
        assert not isImageBatch(torch.zeros(3))

    @pytest.mark.parametrize("shape", [(3, 8, 8), (3, 1, 8, 8), (3, 3, 8, 8)])
    def test_image_shapes_are_image_batches(self, shape: tuple[int, ...]) -> None:
        assert isImageBatch(torch.zeros(shape))


class TestPlotImageReconstructionGrid:
    def test_returns_a_figure_with_one_row_per_example_and_two_columns(self) -> None:
        fig = plotImageReconstructionGrid(_images(0), _images(1))
        assert isinstance(fig, Figure)
        assert len(fig.axes) == NUM_SAMPLES * 2

    def test_does_not_save_or_close_the_figure(self) -> None:
        fig = plotImageReconstructionGrid(_images(0), _images(1))
        assert plt.fignum_exists(fig.number)

    def test_each_panel_shows_the_matching_tensor(self) -> None:
        originals, reconstructions = _images(0), _images(1)
        fig = plotImageReconstructionGrid(originals, reconstructions)
        axes = np.asarray(fig.axes).reshape(NUM_SAMPLES, 2)
        for row in range(NUM_SAMPLES):
            np.testing.assert_allclose(_shown(axes[row][0]), originals[row].numpy())
            np.testing.assert_allclose(_shown(axes[row][1]), reconstructions[row].numpy())

    def test_max_examples_caps_the_rows(self) -> None:
        fig = plotImageReconstructionGrid(_images(0), _images(1), max_examples=2)
        assert len(fig.axes) == 4

    def test_max_examples_above_the_batch_size_plots_the_whole_batch(self) -> None:
        fig = plotImageReconstructionGrid(_images(0), _images(1), max_examples=100)
        assert len(fig.axes) == NUM_SAMPLES * 2

    def test_original_and_reconstruction_of_a_row_share_a_colour_scale(self) -> None:
        originals = torch.zeros(1, HEIGHT, WIDTH)
        reconstructions = torch.full((1, HEIGHT, WIDTH), 3.0)
        fig = plotImageReconstructionGrid(originals, reconstructions)
        assert _clim(fig.axes[0]) == (0.0, 3.0)
        assert _clim(fig.axes[1]) == (0.0, 3.0)

    def test_scales_are_per_row(self) -> None:
        originals = torch.stack([torch.zeros(HEIGHT, WIDTH), torch.full((HEIGHT, WIDTH), 10.0)])
        fig = plotImageReconstructionGrid(originals, originals + 1.0)
        assert _clim(fig.axes[0]) == (0.0, 1.0)
        assert _clim(fig.axes[2]) == (10.0, 11.0)

    def test_explicit_vmin_and_vmax_apply_to_every_panel(self) -> None:
        fig = plotImageReconstructionGrid(_images(0), _images(1), vmin=-1.0, vmax=2.0)
        assert all(_clim(ax) == (-1.0, 2.0) for ax in fig.axes)

    def test_inverse_transform_is_applied_to_both_images(self) -> None:
        originals, reconstructions = _images(0), _images(1)
        fig = plotImageReconstructionGrid(
            originals, reconstructions, inverse_transform=lambda x: x * 10.0 + 1.0
        )
        np.testing.assert_allclose(
            _shown(fig.axes[0]), originals[0].numpy() * 10.0 + 1.0, rtol=1e-5
        )
        np.testing.assert_allclose(
            _shown(fig.axes[1]), reconstructions[0].numpy() * 10.0 + 1.0, rtol=1e-5
        )

    def test_inverse_transform_is_called_on_one_image_at_a_time(self) -> None:
        seen: list[tuple[int, ...]] = []

        def record(image: torch.Tensor) -> torch.Tensor:
            seen.append(tuple(image.shape))
            return image

        plotImageReconstructionGrid(
            _images(0), _images(1), max_examples=2, inverse_transform=record
        )
        assert seen == [(HEIGHT, WIDTH)] * 4

    def test_inputs_are_left_untouched(self) -> None:
        originals, reconstructions = _images(0), _images(1)
        originals_before, reconstructions_before = originals.clone(), reconstructions.clone()
        plotImageReconstructionGrid(
            originals, reconstructions, inverse_transform=lambda x: x + 1.0, show_error=True
        )
        assert torch.equal(originals, originals_before)
        assert torch.equal(reconstructions, reconstructions_before)

    def test_show_error_adds_a_third_column_with_the_absolute_difference(self) -> None:
        originals, reconstructions = _images(0), _images(1)
        fig = plotImageReconstructionGrid(originals, reconstructions, show_error=True)
        axes = np.asarray(fig.axes).reshape(NUM_SAMPLES, 3)
        expected = (originals[0] - reconstructions[0]).abs().numpy()
        np.testing.assert_allclose(_shown(axes[0][2]), expected, rtol=1e-5)
        assert _clim(axes[0][2])[0] == 0.0
        assert axes[0][2].get_xlabel() == f"max {expected.max():.3g}"

    def test_error_is_computed_after_the_inverse_transform(self) -> None:
        originals, reconstructions = _images(0), _images(1)
        fig = plotImageReconstructionGrid(
            originals, reconstructions, inverse_transform=lambda x: x * 4.0, show_error=True
        )
        expected = (originals[0] * 4.0 - reconstructions[0] * 4.0).abs().numpy()
        np.testing.assert_allclose(_shown(fig.axes[2]), expected, rtol=1e-5)

    def test_column_titles_are_on_the_first_row_only(self) -> None:
        fig = plotImageReconstructionGrid(_images(0), _images(1), show_error=True)
        axes = np.asarray(fig.axes).reshape(NUM_SAMPLES, 3)
        assert [axes[0][column].get_title() for column in range(3)] == [
            "original",
            "reconstruction",
            "absolute error",
        ]
        assert all(ax.get_title() == "" for ax in axes[1:].flat)

    def test_custom_labels_and_figure_title(self) -> None:
        fig = plotImageReconstructionGrid(
            _images(0),
            _images(1),
            title="Reconstructions: image",
            original_label="truth",
            reconstruction_label="model",
        )
        assert fig.get_suptitle() == "Reconstructions: image"
        assert fig.axes[0].get_title() == "truth"
        assert fig.axes[1].get_title() == "model"

    def test_default_row_labels_and_custom_titles(self) -> None:
        fig = plotImageReconstructionGrid(_images(0), _images(1), max_examples=2)
        assert [fig.axes[0].get_ylabel(), fig.axes[2].get_ylabel()] == ["example 0", "example 1"]
        fig = plotImageReconstructionGrid(_images(0), _images(1), max_examples=2, titles=["a", "b"])
        assert [fig.axes[0].get_ylabel(), fig.axes[2].get_ylabel()] == ["a", "b"]

    def test_panels_have_no_ticks(self) -> None:
        fig = plotImageReconstructionGrid(_images(0), _images(1), max_examples=1)
        for ax in fig.axes:
            assert len(ax.get_xticks()) == 0
            assert len(ax.get_yticks()) == 0

    def test_a_single_channel_axis_is_accepted(self) -> None:
        originals = _images(0, (NUM_SAMPLES, 1, HEIGHT, WIDTH))
        fig = plotImageReconstructionGrid(originals, _images(1, originals.shape))
        np.testing.assert_allclose(_shown(fig.axes[0]), originals[0, 0].numpy())

    def test_rgb_images_are_drawn_in_colour(self) -> None:
        originals = _images(0, (NUM_SAMPLES, 3, HEIGHT, WIDTH))
        reconstructions = _images(1, originals.shape)
        fig = plotImageReconstructionGrid(originals, reconstructions, show_error=True)
        axes = np.asarray(fig.axes).reshape(NUM_SAMPLES, 3)
        shown = _shown(axes[0][0])
        assert shown.shape == (HEIGHT, WIDTH, 3)
        np.testing.assert_allclose(shown, originals[0].permute(1, 2, 0).numpy())
        # The error of an RGB pair is a single channel, the mean over colours.
        assert _shown(axes[0][2]).shape == (HEIGHT, WIDTH)

    def test_rgb_values_outside_zero_one_are_clipped_not_rejected(self) -> None:
        originals = torch.full((1, 3, HEIGHT, WIDTH), 2.0)
        fig = plotImageReconstructionGrid(originals, -originals)
        assert float(_shown(fig.axes[0]).max()) <= 1.0
        assert float(_shown(fig.axes[1]).min()) >= 0.0

    def test_integer_images_are_accepted(self) -> None:
        originals = torch.randint(0, 255, (2, HEIGHT, WIDTH), dtype=torch.uint8)
        fig = plotImageReconstructionGrid(originals, originals.clone(), show_error=True)
        assert len(fig.axes) == 6

    def test_tensors_that_require_grad_are_accepted(self) -> None:
        originals = _images(0).requires_grad_(True)
        fig = plotImageReconstructionGrid(originals, _images(1))
        assert len(fig.axes) == NUM_SAMPLES * 2

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            plotImageReconstructionGrid(_images(0), _images(1, (NUM_SAMPLES, HEIGHT, WIDTH + 1)))

    def test_empty_tensors_raise(self) -> None:
        empty = torch.zeros(0, HEIGHT, WIDTH)
        with pytest.raises(ValueError, match="empty"):
            plotImageReconstructionGrid(empty, empty)

    def test_series_batches_raise_and_name_the_series_plot(self) -> None:
        with pytest.raises(ValueError, match="plotReconstructionGrid"):
            plotImageReconstructionGrid(torch.zeros(3, 32), torch.zeros(3, 32))

    def test_a_channel_count_other_than_one_or_three_raises(self) -> None:
        batch = torch.zeros(2, 2, HEIGHT, WIDTH)
        with pytest.raises(ValueError, match="got shape"):
            plotImageReconstructionGrid(batch, batch)

    @pytest.mark.parametrize("max_examples", [0, -1])
    def test_non_positive_max_examples_raises(self, max_examples: int) -> None:
        with pytest.raises(ValueError, match="max_examples"):
            plotImageReconstructionGrid(_images(0), _images(1), max_examples=max_examples)

    def test_titles_of_the_wrong_length_raise(self) -> None:
        with pytest.raises(ValueError, match="titles"):
            plotImageReconstructionGrid(_images(0), _images(1), max_examples=2, titles=["only one"])


class TestPlotImageCrossModalReconstructionMatrix:
    def test_returns_a_figure_with_a_target_column_and_one_column_per_subset(self) -> None:
        fig = plotImageCrossModalReconstructionMatrix(_collected(), "image_out")
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1 + 3

    def test_does_not_save_or_close_the_figure(self) -> None:
        fig = plotImageCrossModalReconstructionMatrix(_collected(), "image_out")
        assert plt.fignum_exists(fig.number)

    def test_every_row_is_one_example_under_every_input_condition(self) -> None:
        collected = _collected()
        indices = [3, 1]
        fig = plotImageCrossModalReconstructionMatrix(
            collected, "image_out", example_indices=indices
        )
        axes = np.asarray(fig.axes).reshape(len(indices), 4)
        subsets = sorted(collected, key=lambda subset: " + ".join(sorted(subset)))
        for row, sample in enumerate(indices):
            target = collected[subsets[0]]["image_out"][0][sample].numpy()
            np.testing.assert_allclose(_shown(axes[row][0]), target)
            for column, subset in enumerate(subsets, start=1):
                expected = collected[subset]["image_out"][1][sample].numpy()
                np.testing.assert_allclose(_shown(axes[row][column]), expected)

    def test_default_column_order_sorts_subsets_by_label(self) -> None:
        fig = plotImageCrossModalReconstructionMatrix(_collected(), "image_out")
        titles = [ax.get_title() for ax in fig.axes]
        assert titles == [
            "original (target)",
            "output from\nimage",
            "output from\nimage + signal",
            "output from\nsignal",
        ]

    def test_subset_order_is_honoured(self) -> None:
        order = [frozenset({"signal"}), frozenset({"image"}), frozenset({"image", "signal"})]
        collected = _collected()
        fig = plotImageCrossModalReconstructionMatrix(collected, "image_out", subset_order=order)
        assert [ax.get_title() for ax in fig.axes][1:] == [
            "output from\nsignal",
            "output from\nimage",
            "output from\nimage + signal",
        ]
        np.testing.assert_allclose(
            _shown(fig.axes[1]), collected[order[0]]["image_out"][1][0].numpy()
        )

    def test_subset_order_can_leave_subsets_out(self) -> None:
        fig = plotImageCrossModalReconstructionMatrix(
            _collected(), "image_out", subset_order=[frozenset({"signal"})]
        )
        assert len(fig.axes) == 2

    def test_inputs_are_drawn_before_the_target(self) -> None:
        signal = torch.arange(NUM_SAMPLES * 8, dtype=torch.float32).reshape(NUM_SAMPLES, 8)
        image_in = _images(7)
        fig = plotImageCrossModalReconstructionMatrix(
            _collected(),
            "image_out",
            example_indices=[2],
            inputs={"signal": signal, "image": image_in},
        )
        assert len(fig.axes) == 2 + 1 + 3
        assert [ax.get_title() for ax in fig.axes][:3] == [
            "signal (input)",
            "image (input)",
            "original (target)",
        ]
        # A series input is a line, an image input an image.
        assert len(fig.axes[0].lines) == 1 and not fig.axes[0].images
        np.testing.assert_allclose(np.asarray(fig.axes[0].lines[0].get_ydata()), signal[2].numpy())
        np.testing.assert_allclose(_shown(fig.axes[1]), image_in[2].numpy())

    def test_inputs_may_be_images_with_a_channel_axis(self) -> None:
        image_in = _images(7, (NUM_SAMPLES, 1, HEIGHT, WIDTH))
        fig = plotImageCrossModalReconstructionMatrix(
            _collected(), "image_out", inputs={"image": image_in}
        )
        np.testing.assert_allclose(_shown(fig.axes[0]), image_in[0, 0].numpy())

    def test_target_and_outputs_of_a_row_share_a_colour_scale(self) -> None:
        collected = _collected()
        fig = plotImageCrossModalReconstructionMatrix(collected, "image_out")
        scales = {_clim(ax) for ax in fig.axes}
        assert len(scales) == 1

    def test_explicit_vmin_and_vmax_apply_to_target_outputs_and_image_inputs(self) -> None:
        fig = plotImageCrossModalReconstructionMatrix(
            _collected(),
            "image_out",
            inputs={"image": _images(7)},
            vmin=0.0,
            vmax=1.0,
        )
        assert all(_clim(ax) == (0.0, 1.0) for ax in fig.axes)

    def test_inverse_transform_applies_to_target_and_outputs_but_not_to_inputs(self) -> None:
        collected = _collected()
        image_in = _images(7)
        fig = plotImageCrossModalReconstructionMatrix(
            collected,
            "image_out",
            inputs={"image": image_in},
            inverse_transform=lambda x: x * 100.0,
        )
        np.testing.assert_allclose(_shown(fig.axes[0]), image_in[0].numpy())
        target = collected[frozenset({"signal"})]["image_out"][0][0].numpy()
        np.testing.assert_allclose(_shown(fig.axes[1]), target * 100.0, rtol=1e-5)

    def test_titles_and_labels(self) -> None:
        fig = plotImageCrossModalReconstructionMatrix(
            _collected(),
            "image_out",
            example_indices=[4, 0],
            title="cross-modal",
            original_label="truth",
        )
        assert fig.get_suptitle() == "cross-modal"
        assert fig.axes[0].get_title() == "truth"
        assert fig.axes[0].get_ylabel() == "example 4"
        assert fig.axes[4].get_ylabel() == "example 0"

    def test_other_decoders_in_the_collection_are_ignored(self) -> None:
        collected = _collected()
        for per_decoder in collected.values():
            per_decoder["signal_out"] = (torch.zeros(NUM_SAMPLES, 8), torch.zeros(NUM_SAMPLES, 8))
        fig = plotImageCrossModalReconstructionMatrix(collected, "image_out")
        assert len(fig.axes) == 1 + 3

    def test_a_subset_without_the_decoder_leaves_a_blank_cell(self) -> None:
        collected = _collected()
        del collected[frozenset({"signal"})]["image_out"]
        collected[frozenset({"signal"})]["signal_out"] = (torch.zeros(5, 8), torch.zeros(5, 8))
        fig = plotImageCrossModalReconstructionMatrix(collected, "image_out")
        blank = [ax for ax in fig.axes if not ax.axison]
        assert len(blank) == 1
        assert blank[0].get_title() == "output from\nsignal"

    def test_empty_collected_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            plotImageCrossModalReconstructionMatrix({}, "image_out")

    def test_unknown_decoder_raises_and_lists_the_available_ones(self) -> None:
        with pytest.raises(ValueError, match=r"'nope'.*\['image_out'\]"):
            plotImageCrossModalReconstructionMatrix(_collected(), "nope")

    def test_a_decoder_of_series_raises_and_names_the_series_plot(self) -> None:
        collected = {
            frozenset({"a"}): {"y": (torch.zeros(3, 8), torch.zeros(3, 8))},
        }
        with pytest.raises(ValueError, match="plotCrossModalReconstructionMatrix"):
            plotImageCrossModalReconstructionMatrix(collected, "y")

    def test_subset_order_naming_an_unknown_subset_raises(self) -> None:
        with pytest.raises(ValueError, match="subset_order"):
            plotImageCrossModalReconstructionMatrix(
                _collected(), "image_out", subset_order=[frozenset({"ghost"})]
            )

    def test_empty_example_indices_raise(self) -> None:
        with pytest.raises(ValueError, match="example_indices"):
            plotImageCrossModalReconstructionMatrix(_collected(), "image_out", example_indices=[])

    @pytest.mark.parametrize("index", [NUM_SAMPLES, -1])
    def test_out_of_range_example_index_raises(self, index: int) -> None:
        with pytest.raises(ValueError, match="out of range"):
            plotImageCrossModalReconstructionMatrix(
                _collected(), "image_out", example_indices=[index]
            )

    def test_an_input_shorter_than_the_collection_raises_naming_it(self) -> None:
        with pytest.raises(ValueError, match="input 'signal'"):
            plotImageCrossModalReconstructionMatrix(
                _collected(),
                "image_out",
                example_indices=[NUM_SAMPLES - 1],
                inputs={"signal": torch.zeros(2, 8)},
            )

    def test_an_input_that_is_neither_series_nor_image_raises(self) -> None:
        with pytest.raises(ValueError, match="input 'signal'"):
            plotImageCrossModalReconstructionMatrix(
                _collected(), "image_out", inputs={"signal": torch.zeros(NUM_SAMPLES)}
            )


@pytest.fixture(scope="module")
def signalImageModel() -> GlobalVae:
    torch.manual_seed(0)
    model = buildSignalImageModelForScript()
    model.eval()
    return model


class TestWithARealImageModel:
    """The collection helpers feed the image plots unchanged, on real model outputs."""

    def test_collected_reconstructions_are_image_batches(self, signalImageModel: GlobalVae) -> None:
        data = buildSignalImageDataloaderForScript()
        originals, reconstructions = collectReconstructions(
            signalImageModel, data, "image", device="cpu"
        )
        assert originals.shape == reconstructions.shape == (16, IMAGE_SIDE, IMAGE_SIDE)
        assert isImageBatch(originals)
        series, _ = collectReconstructions(signalImageModel, data, "signal", device="cpu")
        assert series.shape[1:] == (SIGNAL_LENGTH,)
        assert not isImageBatch(series)

    def test_the_grid_plots_collected_reconstructions(self, signalImageModel: GlobalVae) -> None:
        originals, reconstructions = collectReconstructions(
            signalImageModel, buildSignalImageDataloaderForScript(), "image", device="cpu"
        )
        fig = plotImageReconstructionGrid(originals, reconstructions, show_error=True)
        assert len(fig.axes) == 8 * 3

    def test_the_matrix_plots_a_collected_cross_modal_result(
        self, signalImageModel: GlobalVae
    ) -> None:
        collected = collectCrossModalReconstructions(
            signalImageModel, buildSignalImageDataloaderForScript(), device="cpu"
        )
        data = buildSignalImageDataloaderForScript()
        inputs = {
            "signal": torch.cat([batch["signal"] for batch in data]),
            "image": torch.cat([batch["image"] for batch in data]),
        }
        fig = plotImageCrossModalReconstructionMatrix(
            collected, "image", example_indices=[0, 1], inputs=inputs
        )
        assert len(fig.axes) == 2 * (2 + 1 + 3)

    def test_the_series_matrix_still_refuses_an_image_decoder(
        self, signalImageModel: GlobalVae
    ) -> None:
        """The two matrices are complementary: neither silently draws the other's decoders."""
        collected = collectCrossModalReconstructions(
            signalImageModel, buildSignalImageDataloaderForScript(), device="cpu"
        )
        with pytest.raises(ValueError, match="image_plot"):
            plotCrossModalReconstructionMatrix(collected)


class TestExportEvaluationFiguresWithImages:
    def test_an_image_modality_gets_an_image_grid(
        self, signalImageModel: GlobalVae, tmp_path: Path
    ) -> None:
        paths = exportEvaluationFigures(
            signalImageModel, buildSignalImageDataloaderForScript(), tmp_path, device="cpu"
        )
        names = {path.name for path in paths}
        assert names == {
            "reconstructions_signal.png",
            "reconstructions_image.png",
            "latent_z_fused.png",
            "kl_z_fused.png",
        }
        assert all(path.exists() and path.stat().st_size > 0 for path in paths)

    def test_each_modality_goes_to_the_grid_that_fits_its_shape(
        self, signalImageModel: GlobalVae, tmp_path: Path
    ) -> None:
        with (
            patch(
                "global_vae.evaluation.visual_export.plotImageReconstructionGrid",
                wraps=plotImageReconstructionGrid,
            ) as image_grid,
            patch(
                "global_vae.evaluation.visual_export.plotReconstructionGrid",
                wraps=plotReconstructionGrid,
            ) as series_grid,
        ):
            exportEvaluationFigures(
                signalImageModel, buildSignalImageDataloaderForScript(), tmp_path, device="cpu"
            )
        assert image_grid.call_count == 1
        assert image_grid.call_args.args[0].shape == (16, IMAGE_SIDE, IMAGE_SIDE)
        assert series_grid.call_count == 1
        assert series_grid.call_args.args[0].shape == (16, SIGNAL_LENGTH)

    def test_inverse_transforms_reach_the_image_grid(
        self, signalImageModel: GlobalVae, tmp_path: Path
    ) -> None:
        seen: list[tuple[int, ...]] = []

        def record(image: torch.Tensor) -> torch.Tensor:
            seen.append(tuple(image.shape))
            return image

        exportEvaluationFigures(
            signalImageModel,
            buildSignalImageDataloaderForScript(),
            tmp_path,
            device="cpu",
            max_examples=2,
            inverse_transforms={"image": record},
        )
        assert seen == [(IMAGE_SIDE, IMAGE_SIDE)] * 4


class TestExportCrossModalFiguresWithImages:
    def test_series_matrix_and_one_figure_per_image_decoder(
        self, signalImageModel: GlobalVae, tmp_path: Path
    ) -> None:
        paths = exportCrossModalFigures(
            signalImageModel, buildSignalImageDataloaderForScript(), tmp_path, device="cpu"
        )
        assert [path.name for path in paths] == [
            "cross_modal_reconstructions.png",
            "cross_modal_reconstructions_image.png",
        ]
        assert all(path.exists() and path.stat().st_size > 0 for path in paths)

    def test_the_series_matrix_holds_only_the_series_decoders(
        self, signalImageModel: GlobalVae, tmp_path: Path
    ) -> None:
        with patch(
            "global_vae.evaluation.cross_modal.plotCrossModalReconstructionMatrix",
            wraps=plotCrossModalReconstructionMatrix,
        ) as series_matrix:
            exportCrossModalFigures(
                signalImageModel, buildSignalImageDataloaderForScript(), tmp_path, device="cpu"
            )
        assert series_matrix.call_count == 1
        passed = series_matrix.call_args.args[0]
        assert {name for per_decoder in passed.values() for name in per_decoder} == {"signal"}

    def test_inverse_transform_of_an_image_decoder_reaches_its_figure(
        self, signalImageModel: GlobalVae, tmp_path: Path
    ) -> None:
        seen: list[tuple[int, ...]] = []

        def record(image: torch.Tensor) -> torch.Tensor:
            seen.append(tuple(image.shape))
            return image

        exportCrossModalFigures(
            signalImageModel,
            buildSignalImageDataloaderForScript(),
            tmp_path,
            device="cpu",
            inverse_transforms={"image": record},
        )
        # One target plus one output per input subset (signal, image, both), one example.
        assert seen == [(IMAGE_SIDE, IMAGE_SIDE)] * 4

    def test_example_index_selects_the_row(
        self, signalImageModel: GlobalVae, tmp_path: Path
    ) -> None:
        paths = exportCrossModalFigures(
            signalImageModel,
            buildSignalImageDataloaderForScript(),
            tmp_path,
            device="cpu",
            example_index=3,
        )
        assert len(paths) == 2

    def test_a_model_with_only_image_decoders_writes_no_series_figure(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = GlobalVae.createSingleLatent(
            modality_configs={
                "image_a": {"encoder": "2d_cnn_encoder_v1", "decoder": "2d_cnn_decoder_v1"},
                "image_b": {"encoder": "2d_cnn_encoder_v1", "decoder": "2d_cnn_decoder_v1"},
            },
            latent_dim=4,
            fusion_strategy="poe",
            encoder_kwargs={
                name: {"latent_dim": 4, "hidden_channels": (4, 8), "modality_name": name}
                for name in ("image_a", "image_b")
            },
            decoder_kwargs={
                name: {
                    "latent_dim": 4,
                    "output_shape": (IMAGE_SIDE, IMAGE_SIDE),
                    "hidden_channels": (8, 4),
                    "seed_shape": (4, 4),
                    "upsample_modes": "conv_transpose",
                    "modality_name": name,
                }
                for name in ("image_a", "image_b")
            },
        )
        model.eval()
        data = [
            {
                "image_a": torch.rand(4, IMAGE_SIDE, IMAGE_SIDE),
                "image_b": torch.rand(4, IMAGE_SIDE, IMAGE_SIDE),
            }
        ]
        paths = exportCrossModalFigures(model, data, tmp_path, device="cpu")
        assert [path.name for path in paths] == [
            "cross_modal_reconstructions_image_a.png",
            "cross_modal_reconstructions_image_b.png",
        ]
        assert not (tmp_path / "cross_modal_reconstructions.png").exists()
