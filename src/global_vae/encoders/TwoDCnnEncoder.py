"""2D CNN encoder (spec §6, §12)."""

from collections.abc import Callable, Sequence, Sized
from typing import Any, cast

import torch
from torch import nn

from global_vae.encoders.base import AbstractEncoder
from global_vae.encoders.registry import registerEncoder
from global_vae.utils.builders import build2DPoolLayer
from global_vae.utils.conv_math import solveMinimumInputShapeForConv2d
from global_vae.utils.stage_config import (
    ShapeLike,
    broadcastPerStage,
    broadcastPerStageOptionalShape,
    broadcastPerStageShape,
)


def _computeMinimumInputShapeFromResolved(
    num_stages: int,
    kernel_sizes: tuple[tuple[int, int], ...],
    strides: tuple[tuple[int, int], ...],
    paddings: tuple[tuple[int, int], ...],
    dilations: tuple[tuple[int, int], ...],
    poolings: tuple[str | None, ...],
    pool_kernel_sizes: tuple[tuple[int, int] | None, ...],
    pool_strides: tuple[tuple[int, int] | None, ...],
    pool_paddings: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    """Core of `computeMinimumInputShape`, operating on already-resolved per-stage tuples.

    Factored out so `__init__` can reuse the exact same solving logic
    against the per-stage tuples it has *already* resolved, without
    re-broadcasting them: unlike `broadcastPerStage` (1D), which is
    idempotent on an already-length-`num_stages` tuple regardless of
    whether it holds plain scalars, `broadcastPerStageShape` is
    deliberately *not* idempotent on a bare `tuple` (a `tuple` always
    means "one shared shape", `list` always means "one entry per
    stage", see that function's own docstring) -- so passing an
    already-resolved `tuple[tuple[int, int], ...]` of per-stage shapes
    back through `computeMinimumInputShape`'s own broadcasting would be
    misread as a single, wrongly-sized shared shape. This function is
    the shared "no broadcasting, just solve" core both
    `computeMinimumInputShape` (broadcasts raw, possibly-shared user
    arguments once, then delegates here) and `__init__` (already has
    fully per-stage-resolved values, and delegates here directly) use,
    exactly mirroring how `TwoDCnnDecoder` calls
    `utils.conv_math.computeUpsampleStack2dOutputShape` directly with
    already-resolved tuples rather than re-broadcasting them.

    Args: as `computeMinimumInputShape`, except every shape argument is
        already an exact `tuple` of length `num_stages`.

    Returns:
        The minimum `(height, width)` this configuration can accept.

    Raises:
        ValueError: If some stage requests pooling but has no
            `pool_kernel_sizes` entry.
    """
    required_min_shape = (1, 1)
    for stage in reversed(range(num_stages)):
        if poolings[stage] is not None and pool_kernel_sizes[stage] is None:
            raise ValueError(
                f"pooling='{poolings[stage]}' requires a kernel_size, but kernel_size is None."
            )
        if poolings[stage] is not None:
            kernel_size_for_stage = pool_kernel_sizes[stage]
            # Guaranteed non-None by the ValueError check above: reaching this
            # branch means poolings[stage] is not None, so the compound
            # condition there would already have raised if this were None.
            assert kernel_size_for_stage is not None
            stride_for_stage = pool_strides[stage]
            pool_stride = (
                stride_for_stage if stride_for_stage is not None else kernel_size_for_stage
            )
            required_min_shape = solveMinimumInputShapeForConv2d(
                required_min_shape, kernel_size_for_stage, pool_stride, pool_paddings[stage], (1, 1)
            )
        required_min_shape = solveMinimumInputShapeForConv2d(
            required_min_shape,
            kernel_sizes[stage],
            strides[stage],
            paddings[stage],
            dilations[stage],
        )
    return required_min_shape


@registerEncoder("2d_cnn_encoder_v1")
class TwoDCnnEncoder(AbstractEncoder):
    """2D convolutional encoder, robust to variable input spatial size.

    Direct 2D generalization of `OneDCnnEncoder` (spec §6's image
    modality: "Candidate encoders: CNN (ResNet-style) or ViT"; this is
    the plain, non-residual CNN candidate, mirroring how
    `OneDCnnEncoder`/`OneDCnnResidualEncoder` already split the 1D
    signal case). A stack of conv (+ optional normalization, optional
    activation, optional pooling) stages, followed by global adaptive
    pooling. The adaptive pool is what makes the encoder
    resolution-agnostic: two images of different height/width produce
    feature maps of different spatial size after the conv stack, but
    the same fixed-size vector after pooling, so a single set of
    weights covers images of any size (subject to the architecture's
    own minimum, see `computeMinimumInputShape`).

    Every per-stage hyperparameter that is shape-like (`kernel_sizes`,
    `strides`, `paddings`, `dilations`, `pool_kernel_sizes`,
    `pool_strides`, `pool_paddings`) accepts either one value shared by
    every stage or a sequence of exactly `len(hidden_channels)`
    per-stage values, exactly like `OneDCnnEncoder` -- generalized to
    the fact that a 2D shape can itself be non-square:
      - a single `int`, applied as a square shape to every stage;
      - a single `tuple[int, int]`, an explicit non-square
        `(height, width)` shape applied to every stage;
      - a `list` of exactly `len(hidden_channels)` entries, one per
        stage, each itself an `int` or a `tuple[int, int]` (so
        different stages may each have their own, possibly non-square,
        shape).
    See `utils.stage_config.broadcastPerStageShape` for why `list` and
    `tuple` are given these different, non-overlapping jobs (a plain
    `Sequence`-based broadcast would be ambiguous for a 2-stage 2D
    kernel: `(3, 5)` could mean either "one (3, 5) kernel every stage"
    or "kernel 3 at stage 0, kernel 5 at stage 1"). `poolings`,
    `activations`, `normalizations`, and `pool_kwargs` are not
    shape-like and keep `OneDCnnEncoder`'s own `broadcastPerStage`
    convention unchanged (a plain scalar, or a plain sequence of
    exactly `len(hidden_channels)` per-stage values; there is no
    list-vs-tuple ambiguity for these, since none of them is itself a
    multi-component numeric shape).

    Note: if a stage's `poolings` entry is not `None`, that stage's
    pooling step reduces spatial size along each axis independently;
    if that stage's `strides` are also greater than 1 on either axis,
    both reductions compound. The input height and width must each
    stay large enough, after however much downsampling the chosen
    configuration performs along that axis, that no intermediate
    feature map collapses to a length `<= 0` along either axis. This is
    the exact same real constraint `OneDCnnEncoder` already documents
    for a strided/pooled conv stack, now checked independently per
    axis rather than something silently handled.
    """

    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 1,
        hidden_channels: tuple[int | nn.Module, ...] = (32, 64, 128),
        kernel_sizes: ShapeLike | list[ShapeLike] = 5,
        strides: ShapeLike | list[ShapeLike] = 1,
        paddings: ShapeLike | list[ShapeLike] | None = None,
        dilations: ShapeLike | list[ShapeLike] = 1,
        poolings: str | None | Sequence[str | None] = "max",
        pool_kernel_sizes: ShapeLike | None | list[ShapeLike | None] = 2,
        pool_strides: ShapeLike | None | list[ShapeLike | None] = None,
        pool_paddings: ShapeLike | list[ShapeLike] = 0,
        pool_kwargs: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        activations: Callable[[], nn.Module]
        | Sequence[Callable[[], nn.Module] | None]
        | None = nn.ReLU,
        normalizations: Callable[[int], nn.Module]
        | Sequence[Callable[[int], nn.Module] | None]
        | None = nn.BatchNorm2d,
        global_pool: str = "avg",
        head_hidden_dims: tuple[int, ...] = (),
        head_activation: Callable[[], nn.Module] | None = nn.ReLU,
        modality_name: str = "image",
    ) -> None:
        """Build the encoder.

        Args:
            latent_dim: Dimensionality of the `(mu, logvar)` output.
            in_channels: Number of input channels (`1` for a plain
                grayscale image; `3` for RGB; more if further
                co-registered channels are stacked).
            hidden_channels: Output channel width of each conv stage,
                applied in order. Its length fixes the number of
                stages. Or directly a layer (an `nn.Module` stage,
                exactly like `OneDCnnEncoder`'s own escape hatch: it
                must expose an `out_channels` or `out_features`
                attribute so later stages know this stage's output
                width).
            kernel_sizes: Convolution kernel shape, per stage or
                shared (see the class docstring for the `int` /
                `tuple[int, int]` / `list` convention).
            strides: Convolution stride shape, per stage or shared.
                Use this (with `poolings=None`) to downsample via
                strided convolutions instead of a separate pooling
                layer.
            paddings: Convolution padding shape, per stage or shared.
                Defaults (`None`) to `dilation * (kernel_size // 2)`
                per axis for each stage, which keeps a stride-1 stage's
                output shape equal to its input shape on that axis.
            dilations: Convolution dilation shape, per stage or shared.
            poolings: `"max"`, `"avg"`, or `None` to disable pooling,
                per stage or shared. Different stages may use
                different pooling strategies.
            pool_kernel_sizes: Pooling window shape, per stage or
                shared. Required (not `None`) for any stage whose
                `poolings` entry is not `None`; ignored otherwise.
            pool_strides: Pooling stride shape, per stage or shared.
                Defaults (`None`) to that stage's pooling kernel shape
                (non-overlapping windows). Ignored for stages whose
                `poolings` entry is `None`.
            pool_paddings: Pooling padding shape, per stage or shared.
                Ignored for stages whose `poolings` entry is `None`.
            pool_kwargs: Additional keyword arguments forwarded to the
                pooling layer's constructor (e.g. `ceil_mode`,
                `count_include_pad`), per stage or shared. Defaults
                (`None`) to no extra arguments for any stage.
            activations: Zero-argument factory returning a fresh
                activation module, per stage or shared. Pass `None`
                (either overall, or as one stage's entry in a sequence)
                to disable activation for that stage.
            normalizations: One-argument factory taking a channel count
                and returning a fresh normalization module, per stage
                or shared. Pass `None` the same way to disable
                normalization for a stage.
            global_pool: `"avg"` or `"max"`: which adaptive pooling
                reduces the final feature map to a single fixed-size
                vector, regardless of input spatial size.
            head_hidden_dims: Hidden layer sizes for an optional small
                MLP inserted between the pooled features and the
                `to_mu`/`to_logvar` heads. Empty tuple (default) keeps
                a single linear layer straight from pooled features to
                each head.
            head_activation: Optional head activation layer.
            modality_name: Name of the modality this encoder handles.

        Raises:
            ValueError: If any per-stage shape argument does not have
                exactly `len(hidden_channels)` entries (when given as a
                `list`) or exactly `ndim=2` components (when given as
                an explicit shared `tuple`), if a stage requests
                pooling without a `pool_kernel_sizes` entry, or if
                `global_pool` is not a recognized choice.
        """
        super().__init__()
        self._latent_dim = latent_dim
        self._modality_name = modality_name
        num_stages = len(hidden_channels)

        kernel_sizes_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(kernel_sizes, num_stages, 2, "kernel_sizes"),
        )
        strides_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(strides, num_stages, 2, "strides"),
        )
        dilations_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(dilations, num_stages, 2, "dilations"),
        )
        if paddings is None:
            paddings_: tuple[tuple[int, int], ...] = tuple(
                (
                    dilations_[stage][0] * (kernel_sizes_[stage][0] // 2),
                    dilations_[stage][1] * (kernel_sizes_[stage][1] // 2),
                )
                for stage in range(num_stages)
            )
        else:
            paddings_ = cast(
                "tuple[tuple[int, int], ...]",
                broadcastPerStageShape(paddings, num_stages, 2, "paddings"),
            )
        poolings_: tuple[str | None, ...] = broadcastPerStage(poolings, num_stages, "poolings")
        pool_kernel_sizes_: tuple[tuple[int, int] | None, ...] = cast(
            "tuple[tuple[int, int] | None, ...]",
            broadcastPerStageOptionalShape(pool_kernel_sizes, num_stages, 2, "pool_kernel_sizes"),
        )
        pool_strides_: tuple[tuple[int, int] | None, ...] = (
            cast(
                "tuple[tuple[int, int] | None, ...]",
                broadcastPerStageOptionalShape(pool_strides, num_stages, 2, "pool_strides"),
            )
            if pool_strides is not None
            else (None,) * num_stages
        )
        pool_paddings_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(pool_paddings, num_stages, 2, "pool_paddings"),
        )
        resolved_pool_kwargs: dict[str, Any] | Sequence[dict[str, Any]] = (
            pool_kwargs if pool_kwargs is not None else {}
        )
        pool_kwargs_ = broadcastPerStage(resolved_pool_kwargs, num_stages, "pool_kwargs")
        activations_ = broadcastPerStage(activations, num_stages, "activations")
        normalizations_: tuple[Callable[[int], nn.Module] | None, ...] = broadcastPerStage(
            normalizations, num_stages, "normalizations"
        )

        # Get the minimal input shape needed for this configuration.
        resolved_pool_strides = tuple(
            pool_strides_[stage] if pool_strides_[stage] is not None else pool_kernel_sizes_[stage]
            for stage in range(num_stages)
        )
        self._min_input_shape = _computeMinimumInputShapeFromResolved(
            num_stages,
            kernel_sizes_,
            strides_,
            paddings_,
            dilations_,
            poolings_,
            pool_kernel_sizes_,
            resolved_pool_strides,
            pool_paddings_,
        )

        layers: list[nn.Module] = []
        channels = in_channels
        for stage in range(num_stages):
            stage_channels = hidden_channels[stage]
            if isinstance(stage_channels, nn.Module):
                # nn.Module does not statically declare `out_channels`; every
                # standard PyTorch layer that has one sets it as a plain int
                # (e.g. Conv2d.out_channels), a convention mypy's generic
                # Module.__getattr__ stub (-> Tensor | Module) cannot express.
                if hasattr(stage_channels, "out_channels"):
                    out_channels = cast(int, stage_channels.out_channels)
                elif hasattr(stage_channels, "out_features"):
                    out_channels = cast(int, stage_channels.out_features)
                else:
                    raise AttributeError(
                        f"The {stage_channels.__class__.__name__} passed as stage {stage} of "
                        f"hidden_channels has no out_channels or out_features attribute, which "
                        f"is required so later stages know this stage's output width."
                    )
                layer: nn.Module = stage_channels
            else:
                out_channels = stage_channels
                layer = nn.Conv2d(
                    channels,
                    out_channels,
                    kernel_size=kernel_sizes_[stage],
                    stride=strides_[stage],
                    padding=paddings_[stage],
                    dilation=dilations_[stage],
                )

            layers.append(layer)

            normalization = normalizations_[stage]
            if normalization is not None:
                layers.append(normalization(out_channels))

            activation = activations_[stage]
            if activation is not None:
                layers.append(activation())

            pool_layer = build2DPoolLayer(
                poolings_[stage],
                pool_kernel_sizes_[stage],
                pool_strides_[stage],
                pool_paddings_[stage],
                **pool_kwargs_[stage],
            )
            if pool_layer is not None:
                layers.append(pool_layer)

            channels = out_channels
        self.conv = nn.Sequential(*layers)

        if global_pool == "avg":
            self.pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unknown global_pool '{global_pool}'. Expected 'avg' or 'max'.")

        # add an optional mlp before the mu and log var
        head_layers: list[nn.Module] = []
        head_in = channels
        for hidden_dim in head_hidden_dims:
            head_layers.append(nn.Linear(head_in, hidden_dim))
            if head_activation is not None:
                head_layers.append(head_activation())
            head_in = hidden_dim
        self.head: nn.Module = nn.Sequential(*head_layers) if head_layers else nn.Identity()

        self.to_mu = nn.Linear(head_in, self._latent_dim)
        self.to_logvar = nn.Linear(head_in, self._latent_dim)

    @staticmethod
    def computeMinimumInputShape(
        hidden_channels: Sized,
        kernel_sizes: ShapeLike | list[ShapeLike] = 5,
        strides: ShapeLike | list[ShapeLike] = 1,
        paddings: ShapeLike | list[ShapeLike] | None = None,
        dilations: ShapeLike | list[ShapeLike] = 1,
        poolings: str | None | Sequence[str | None] = "max",
        pool_kernel_sizes: ShapeLike | None | list[ShapeLike | None] = 2,
        pool_strides: ShapeLike | None | list[ShapeLike | None] = None,
        pool_paddings: ShapeLike | list[ShapeLike] = 0,
    ) -> tuple[int, int]:
        """Compute the minimum input `(height, width)` a given configuration can accept.

        Lets a caller check, before constructing a full `TwoDCnnEncoder` (or after, to
        understand why `forward()` raised), the smallest `(height, width)` that keeps
        every intermediate feature map at a length `>= 1` on *both* axes for this
        architecture (spec §12: verify a configuration before committing to it, rather
        than discovering a mismatch only once a strided/pooled conv stack has already
        collapsed). Mirrors `OneDCnnEncoder.computeMinimumInputLength` exactly, solved
        independently per axis via `utils.conv_math.solveMinimumInputShapeForConv2d`
        (each axis of every layer in this stack only ever depends on that same axis's
        own hyperparameters, so the two axes never interact in this computation).

        Solves the requirement backward, from the last stage to the first, inverting
        each stage's convolution (and pooling layer, if present).

        Args:
            hidden_channels: As in `__init__`. Only its length (the number of stages)
                affects the result.
            kernel_sizes: As in `__init__`.
            strides: As in `__init__`.
            paddings: As in `__init__`. `None` resolves the same way `__init__` does:
                `dilation * (kernel_size // 2)` per axis, per stage.
            dilations: As in `__init__`.
            poolings: As in `__init__`.
            pool_kernel_sizes: As in `__init__`.
            pool_strides: As in `__init__`. `None` resolves to that stage's
                `pool_kernel_sizes` entry, matching `build2DPoolLayer`'s own default.
            pool_paddings: As in `__init__`.

        Returns:
            The minimum `(height, width)` this configuration can accept without any
            intermediate feature map collapsing to a length `<= 0` on either axis.

        Raises:
            ValueError: If any per-stage shape argument does not resolve cleanly (see
                `utils.stage_config.broadcastPerStageShape`).
        """
        num_stages = len(hidden_channels)
        kernel_sizes_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(kernel_sizes, num_stages, 2, "kernel_sizes"),
        )
        strides_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(strides, num_stages, 2, "strides"),
        )
        dilations_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(dilations, num_stages, 2, "dilations"),
        )
        if paddings is None:
            paddings_: tuple[tuple[int, int], ...] = tuple(
                (
                    dilations_[stage][0] * (kernel_sizes_[stage][0] // 2),
                    dilations_[stage][1] * (kernel_sizes_[stage][1] // 2),
                )
                for stage in range(num_stages)
            )
        else:
            paddings_ = cast(
                "tuple[tuple[int, int], ...]",
                broadcastPerStageShape(paddings, num_stages, 2, "paddings"),
            )
        poolings_: tuple[str | None, ...] = broadcastPerStage(poolings, num_stages, "poolings")
        pool_kernel_sizes_: tuple[tuple[int, int] | None, ...] = cast(
            "tuple[tuple[int, int] | None, ...]",
            broadcastPerStageOptionalShape(pool_kernel_sizes, num_stages, 2, "pool_kernel_sizes"),
        )
        pool_strides_: tuple[tuple[int, int] | None, ...] = (
            cast(
                "tuple[tuple[int, int] | None, ...]",
                broadcastPerStageOptionalShape(pool_strides, num_stages, 2, "pool_strides"),
            )
            if pool_strides is not None
            else (None,) * num_stages
        )
        pool_paddings_: tuple[tuple[int, int], ...] = cast(
            "tuple[tuple[int, int], ...]",
            broadcastPerStageShape(pool_paddings, num_stages, 2, "pool_paddings"),
        )

        return _computeMinimumInputShapeFromResolved(
            num_stages,
            kernel_sizes_,
            strides_,
            paddings_,
            dilations_,
            poolings_,
            pool_kernel_sizes_,
            pool_strides_,
            pool_paddings_,
        )

    def _validateInputShape(self, height: int, width: int) -> None:
        """Verify that `(height, width)` meets this architecture's precomputed minimum.

        The minimum itself is computed once, at construction time, by
        `computeMinimumInputShape` (architecture-only, independent of any actual
        input); this check is therefore a pair of integer comparisons per
        `forward()` call, not a replay of the stage-by-stage computation, so it adds
        no meaningful cost next to the convolutions that follow.

        Args:
            height: Height of the raw input image for this call.
            width: Width of the raw input image for this call.

        Raises:
            ValueError: If `height` or `width` is below the architecture's minimum
                on that axis, i.e. some intermediate feature map would collapse to a
                length `<= 0` along that axis.
        """
        min_height, min_width = self._min_input_shape
        if height < min_height or width < min_width:
            raise ValueError(
                f"TwoDCnnEncoder: input shape (height={height}, width={width}) is below "
                f"the minimum shape (height={min_height}, width={min_width}) this "
                f"architecture can accept without an intermediate feature map "
                f"collapsing to length <= 0 along some axis. Use "
                f"TwoDCnnEncoder.computeMinimumInputShape(...) with the same "
                f"architecture arguments to see which stage/axis is responsible."
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images.

        Args:
            x: Raw images, shape `(batch, height, width)` or
                `(batch, in_channels, height, width)`. A 3D input is
                treated as `(batch, height, width)` and given an
                explicit channel dimension of `1`.

        Returns:
            A `(mu, logvar)` tuple, each of shape `(batch, latent_dim)`.
        """
        images = x.unsqueeze(1) if x.dim() == 3 else x
        self._validateInputShape(images.shape[-2], images.shape[-1])
        features = self.conv(images)
        pooled: torch.Tensor = self.pool(features).flatten(1)
        pooled = self.head(pooled)
        mu: torch.Tensor = self.to_mu(pooled)
        logvar: torch.Tensor = self.to_logvar(pooled)
        return mu, logvar

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def modality_name(self) -> str:
        return self._modality_name

    @property
    def minimal_input_length(self) -> int:
        """Conservative scalar summary of `minimal_input_shape` (spec of `AbstractEncoder`).

        `AbstractEncoder.minimal_input_length` is a single `int`, a
        contract written with 1D encoders in mind; a 2D encoder's true
        minimum is a `(height, width)` pair, which does not collapse
        into one number without losing information (see
        `minimal_input_shape` for the exact, independent per-axis
        requirement). This property still implements the abstract
        contract, returning the larger of the two per-axis minimums:
        the smallest side length a *square* input can have and still
        be guaranteed acceptable on both axes.

        Returns:
            `max(minimal_input_shape)`.
        """
        return max(self._min_input_shape)

    @property
    def minimal_input_shape(self) -> tuple[int, int]:
        """Minimal `(height, width)` this encoder can accept without collapsing.

        The precise, per-axis counterpart of `minimal_input_length`
        (see that property's own docstring for why a single `int`
        cannot fully express a 2D minimum): a real, non-square input
        must satisfy both bounds independently, not merely have its
        larger side above `minimal_input_length`.

        Returns:
            The minimal `(height, width)`.
        """
        return self._min_input_shape
