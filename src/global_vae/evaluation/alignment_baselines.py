"""The four baselines of roadmap P6-6, run together on one pool of pairs (ADR 0056).

A distance between two latent spaces that falls during stage 3 is not a result: two collapsed
spaces also minimize it (roadmap I-8). The result is that a way of aligning two separately trained
spaces gives **better retrieval and better cross reconstruction than the alternatives, on pairs it
never saw, without losing active units**. `evaluation.latent_alignment` measures one model that
way. This module runs the alternatives side by side on **the same split**:

- **0, the solo spaces as they are**: the two solo models, warm-started into the pair model
  (ADR 0052), with no stage 3. Expected at chance: nothing made the two spaces say the same thing.
- **1, post-hoc orthogonal Procrustes**: one orthogonal map between the solo posterior means,
  fitted on the training pairs in closed form and applied to the held-out ones. If it matches a
  learned alignment, the learned one is not needed.
- **2, stage 3, once per distance**: the pair model, warm-started, trained with `latent_alignment`
  (one run per registered distance) and `cross_reconstruction` (ADR 0054).
- **3, joint training from scratch**: one model with a shared latent space and a product-of-experts
  fusion (`poe`), trained on the pairs from the start.

Every baseline is scored by `evaluateLatentAlignment` on the same held-out pairs. The result is
reported as a **mean and a spread over `repeats` random splits and seeds**, never one run: the
reference paper uses five seeds on 72 pairs, and a number from one split of so few pairs says
little (roadmap P6-6). Each repeat draws its own split of the pool and trains every model from its
own seed.

`runAlignmentBaselines` takes a `PairPool` of paired tensors and a `BaselineRecipe`, which names
the experiment configs the stages are composed from (the defaults are the shipped signal plus image
recipe of Phase 6), and trains in this process: it is a **benchmark harness**, not the recipe. The
recipe stays three ordinary runs chained by checkpoints (roadmap D-15, `docs/how-to/train-in-
stages.md`); this module only makes the comparison between routes a single command. The dataset is
the caller's, as everywhere (spec §6.2): `examples/05_alignment_baselines.py` runs it on a synthetic
linked pair.

All the numbers it produces are those of whatever data it is given, with placeholder settings
(roadmap I-15): on the synthetic pair they show that the harness works and what the guards catch,
not which route is best on real data.
"""

import json
import logging
import math
import statistics
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from global_vae.config.experiment import ExperimentConfig, loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import (
    BetaScheduleConfig,
    LossTermConfig,
    WarmStartEntryConfig,
    applyWarmStartFromConfig,
    buildTrainerFromConfig,
)
from global_vae.evaluation.latent_alignment import (
    LatentAlignmentReport,
    chooseSelectiveK,
    evaluateLatentAlignment,
)
from global_vae.evaluation.paired_alignment import DEFAULT_RETRIEVAL_KS
from global_vae.losses.distances.registry import listRegisteredLatentDistances
from global_vae.models.global_vae import GlobalVae
from global_vae.utils.seed import setGlobalSeed

logger = logging.getLogger(__name__)

BASELINE_SOLO = "0 solo spaces"
BASELINE_PROCRUSTES = "1 procrustes"
BASELINE_STAGE3 = "2 stage 3"
BASELINE_JOINT = "3 joint poe"

_PLACEHOLDER_LOADER = "alignment_baselines:unused"
"""`data.loader_factory` of the composed configs: required by the schema, never called here."""


@dataclass(frozen=True)
class PairPool:
    """The samples the baselines are drawn from.

    Attributes:
        paired: Modality -> tensor `(samples, ...)`. Row `i` of every modality is the same
            sample. These are split into training pairs and held-out pairs, once per repeat.
        unpaired: Modality -> extra rows of that modality that have no partner. They are added
            to the training rows of that modality in its **solo stage** only (the solo stages are
            where the unpaired samples are used, roadmap Phase 6), never to the pairs and never
            to the held-out set. Empty by default.
    """

    paired: Mapping[str, torch.Tensor]
    unpaired: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that the pool has paired rows, one per pair, and that unpaired ones fit it."""
        if not self.paired:
            raise ValueError("A pool needs at least one paired modality.")
        counts = {name: int(rows.shape[0]) for name, rows in self.paired.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"Every paired modality needs the same number of rows (one per pair), got {counts}."
            )
        foreign = sorted(set(self.unpaired) - set(self.paired))
        if foreign:
            raise ValueError(
                f"unpaired names {foreign}, which are not paired modalities {sorted(self.paired)}."
            )

    @property
    def num_pairs(self) -> int:
        """How many pairs the pool holds."""
        return int(next(iter(self.paired.values())).shape[0])


@dataclass
class PairSplit:
    """One random split of a `PairPool`, as batches.

    Attributes:
        train: Batches of training pairs, for stage 3, the joint model, the Procrustes fit and the
            integrity check.
        heldout: Batches of held-out pairs, on which every baseline is scored.
        solo: Modality -> batches of that modality alone, for its solo stage: the training rows,
            then the unpaired rows of the pool.
    """

    train: list[dict[str, torch.Tensor]]
    heldout: list[dict[str, torch.Tensor]]
    solo: dict[str, list[dict[str, torch.Tensor]]]


def _toBatches(
    rows: Mapping[str, torch.Tensor], indices: torch.Tensor, batch_size: int
) -> list[dict[str, torch.Tensor]]:
    """Cut the selected rows into batches of `batch_size`, dropping a last batch of one row.

    A batch of one row cannot go through a normalization layer in training mode, and it would be
    one sample's gradient: dropping it costs at most one sample.
    """
    batches = []
    for start in range(0, int(indices.numel()), batch_size):
        chosen = indices[start : start + batch_size]
        if chosen.numel() >= 2:
            batches.append({name: tensor[chosen] for name, tensor in rows.items()})
    return batches


def splitPool(pool: PairPool, heldout_fraction: float, seed: int, batch_size: int) -> PairSplit:
    """Split a pool into training pairs and held-out pairs at random, and cut both into batches.

    Args:
        pool: The pool.
        heldout_fraction: The fraction of the pairs that is held out, in `(0, 1)`. At least two
            pairs are held out whatever the fraction.
        seed: Seeds the permutation: one seed, one split.
        batch_size: Rows per batch, at least 2.

    Returns:
        The `PairSplit`.

    Raises:
        ValueError: If a setting is invalid, or fewer than two pairs are left for training.
    """
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError(f"heldout_fraction must be in (0, 1), got {heldout_fraction}.")
    if batch_size < 2:
        raise ValueError(f"batch_size must be at least 2, got {batch_size}.")
    count = pool.num_pairs
    num_heldout = max(2, round(count * heldout_fraction))
    if count - num_heldout < 2:
        raise ValueError(
            f"A pool of {count} pairs leaves {count - num_heldout} for training after holding out "
            f"{num_heldout}: at least 2 are needed."
        )
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    heldout_rows, train_rows = order[:num_heldout], order[num_heldout:]
    solo: dict[str, list[dict[str, torch.Tensor]]] = {}
    for name, tensor in pool.paired.items():
        extra = pool.unpaired.get(name)
        rows = tensor[train_rows] if extra is None else torch.cat([tensor[train_rows], extra])
        solo[name] = _toBatches({name: rows}, torch.arange(rows.shape[0]), batch_size)
    return PairSplit(
        train=_toBatches(pool.paired, train_rows, batch_size),
        heldout=_toBatches(pool.paired, heldout_rows, batch_size),
        solo=solo,
    )


@dataclass(frozen=True)
class BaselineRecipe:
    """The experiment configs the stages of the baselines are composed from.

    The defaults are the shipped signal plus image recipe of Phase 6
    (`configs/experiment/pair_solo_signal.yaml`, `pair_solo_image.yaml`,
    `configs/model/signal_image_pair.yaml`, and the single-latent `configs/model/default.yaml` as
    the joint model). Another pair of modalities writes its own recipe.

    Attributes:
        modalities: The two modalities of a pair, in order.
        solo_experiments: Modality -> the experiment config name of its solo stage (stage 1 and 2).
        pair_overrides: Hydra overrides that turn a solo experiment into the pair stage: the pair
            model, the several-latent training group and the stage label.
        pair_spaces: The two latent spaces of the pair model, in the order of `modalities`.
        joint_overrides: Hydra overrides that turn a solo experiment into the joint model with a
            shared latent space and a product-of-experts fusion, of the same width as the pair
            spaces (roadmap D-16), with modality dropout so that each modality alone has a
            posterior of its own.
        joint_space: The shared latent space of the joint model.
    """

    modalities: tuple[str, str] = ("signal", "image")
    solo_experiments: Mapping[str, str] = field(
        default_factory=lambda: {
            "signal": "experiment/pair_solo_signal",
            "image": "experiment/pair_solo_image",
        }
    )
    pair_overrides: tuple[str, ...] = (
        "model=signal_image_pair",
        "training=several_latents",
        "training.stage=pair",
    )
    pair_spaces: tuple[str, str] = ("z_signal", "z_image")
    joint_overrides: tuple[str, ...] = (
        "model=default",
        "model.single_latent.dim=16",
        "training.stage=joint",
        "training.modality_dropout_p=0.3",
    )
    joint_space: str = "z_fused"


@dataclass(frozen=True)
class BaselineSettings:
    """How the baselines are run. Every value is a placeholder to be tuned on real data (I-15).

    Attributes:
        repeats: How many random splits and seeds to run. The reference paper uses five.
        heldout_fraction: The fraction of the pairs held out in each split.
        batch_size: Rows per batch in every stage.
        seed: The first seed: repeat `r` uses `seed + r` for its split and its models.
        solo_epochs: Epochs of each solo stage.
        pair_epochs: Epochs of each stage 3 (one per distance).
        joint_epochs: Epochs of the joint model. `None` (default): `solo_epochs + pair_epochs`, the
            total the staged route spends, so that the comparison is at equal training.
        beta: The KL weight of the pair stage and of the joint model (`training.beta`).
        alignment_weight: The end value of the linear ramp of the `latent_alignment` term.
        cross_weight: The end value of the linear ramp of the `cross_reconstruction` terms.
        ramp_steps: Steps of both ramps (a ramp protects the solo solutions at the start).
        reconstruction_reduction: `training.reconstruction_reduction` of every stage. `"sum"`
            (default) because a model that has just started collapses under `"mean"` over a long
            signal (`docs/how-to/choose-beta.md`).
        distances: The registered distances to run stage 3 with, one run each. `None` (default):
            all of them.
        ks: The `k` of each `recall@k`.
        similarity: The retrieval similarity.
        procrustes_center: Fit the Procrustes baseline after centering.
        include_joint: Run baseline 3. `False` skips it (it is the slowest to explain, not to run).
        device: `"cpu"` or a CUDA device; every model is trained and scored there.
    """

    repeats: int = 5
    heldout_fraction: float = 0.25
    batch_size: int = 16
    seed: int = 0
    solo_epochs: int = 40
    pair_epochs: int = 100
    joint_epochs: int | None = None
    beta: float = 0.1
    alignment_weight: float = 0.3
    cross_weight: float = 0.3
    ramp_steps: int = 100
    reconstruction_reduction: str = "sum"
    distances: tuple[str, ...] | None = None
    ks: tuple[int, ...] = DEFAULT_RETRIEVAL_KS
    similarity: str = "cosine"
    procrustes_center: bool = False
    include_joint: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Refuse a count below one and a weight that is negative or not finite."""
        for name in ("repeats", "solo_epochs", "pair_epochs", "ramp_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}.")
        if self.joint_epochs is not None and self.joint_epochs < 1:
            raise ValueError(f"joint_epochs must be at least 1, got {self.joint_epochs}.")
        for name in ("beta", "alignment_weight", "cross_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be finite and at least 0, got {getattr(self, name)}."
                )

    @property
    def total_joint_epochs(self) -> int:
        """The epochs of the joint model: `joint_epochs`, or what the staged route spends."""
        return (
            self.joint_epochs
            if self.joint_epochs is not None
            else (self.solo_epochs + self.pair_epochs)
        )


@dataclass
class Spread:
    """The mean and the spread of one number over the repeats.

    Attributes:
        mean: The mean over the repeats that have the number.
        std: The sample standard deviation (`n - 1`), 0 with a single repeat.
        minimum: The smallest value.
        maximum: The largest value.
        count: How many repeats have the number (`nan` values are left out).
    """

    mean: float
    std: float
    minimum: float
    maximum: float
    count: int


def summarizeValues(values: Sequence[float]) -> Spread:
    """The mean and the spread of a list of numbers over the repeats.

    Args:
        values: One number per repeat; `nan` values are left out.

    Returns:
        A `Spread`; all `nan` with `count` 0 when nothing is left.
    """
    kept = [value for value in values if not math.isnan(value)]
    if not kept:
        return Spread(math.nan, math.nan, math.nan, math.nan, 0)
    return Spread(
        mean=statistics.fmean(kept),
        std=statistics.stdev(kept) if len(kept) > 1 else 0.0,
        minimum=min(kept),
        maximum=max(kept),
        count=len(kept),
    )


def flattenAlignmentReport(
    report: LatentAlignmentReport, procrustes: bool = False
) -> dict[str, float]:
    """Flatten the numbers of a report that a comparison averages over repeats.

    Keys (each a float): `recall@k` and `chance_recall@k` (mean of the two directions),
    `mean_rank` and `chance_mean_rank`, `recall@k/<direction>` and `mean_rank/<direction>`,
    `cross_mse/<direction>/<modality>`, `self_mse/<view>/<modality>`, `gap_mse/<direction>/
    <modality>`, `cross_r2`, `self_r2` and `gap_r2` (means over every cell), `active_units` (mean
    over the views), `collapsed_views`, `distance/<name>/ratio`, `num_flags`.

    Args:
        report: The report.
        procrustes: Read the Procrustes baseline of the report (its retrieval and its mapped cross
            reconstruction) instead of the model's own. The self reconstruction, the active units
            and the distances are those of the model either way, and the distances and the
            collapse numbers are left out for the baseline, which has no model of its own.

    Returns:
        Name -> value.

    Raises:
        ValueError: If `procrustes` is set and the report has none.
    """
    if procrustes and report.procrustes is None:
        raise ValueError(
            "This report has no Procrustes baseline: it was run without training pairs."
        )
    retrieval = (
        report.procrustes.retrieval if procrustes and report.procrustes else report.retrieval
    )
    cross = (
        report.procrustes.cross_reconstruction
        if procrustes and report.procrustes
        else report.cross_reconstruction
    )
    gap = (
        report.procrustes.reconstruction_gap
        if procrustes and report.procrustes
        else report.reconstruction_gap
    )
    flat: dict[str, float] = {}
    directions = list(retrieval)
    for k in report.ks:
        for direction in directions:
            flat[f"recall@{k}/{direction}"] = retrieval[direction][f"recall@{k}"]
        flat[f"recall@{k}"] = statistics.fmean(retrieval[d][f"recall@{k}"] for d in directions)
        flat[f"chance_recall@{k}"] = statistics.fmean(
            retrieval[d][f"chance_recall@{k}"] for d in directions
        )
    for direction in directions:
        flat[f"mean_rank/{direction}"] = retrieval[direction]["mean_rank"]
    flat["mean_rank"] = statistics.fmean(retrieval[d]["mean_rank"] for d in directions)
    flat["chance_mean_rank"] = (report.num_heldout + 1) / 2.0
    cross_r2: list[float] = []
    self_r2: list[float] = []
    gap_r2: list[float] = []
    for direction, per_modality in cross.items():
        for modality, metrics in per_modality.items():
            flat[f"cross_mse/{direction}/{modality}"] = metrics["mse"]
            cross_r2.append(metrics["r2"])
            if modality in gap.get(direction, {}):
                flat[f"gap_mse/{direction}/{modality}"] = gap[direction][modality]["mse"]
                gap_r2.append(gap[direction][modality]["r2"])
    for view, per_modality in report.self_reconstruction.items():
        for modality, metrics in per_modality.items():
            flat[f"self_mse/{view}/{modality}"] = metrics["mse"]
            self_r2.append(metrics["r2"])
    if cross_r2:
        flat["cross_r2"] = statistics.fmean(cross_r2)
    if self_r2:
        flat["self_r2"] = statistics.fmean(self_r2)
    if gap_r2:
        flat["gap_r2"] = statistics.fmean(gap_r2)
    if not procrustes:
        flat["active_units"] = statistics.fmean(c.active_units for c in report.collapse.values())
        flat["collapsed_views"] = float(sum(c.flagged for c in report.collapse.values()))
        for name, result in report.distances.items():
            flat[f"distance/{name}/ratio"] = result.ratio
        flat["num_flags"] = float(len(report.flags))
    return flat


@dataclass
class BaselineRow:
    """One baseline over the repeats.

    Attributes:
        name: The baseline (`"0 solo spaces"`, `"2 stage 3 wasserstein2_diag"`, ...).
        description: One sentence on what it is.
        runs: One flat dict (`flattenAlignmentReport`) per repeat.
        spread: Number name -> `Spread` over the repeats.
    """

    name: str
    description: str
    runs: list[dict[str, float]] = field(default_factory=list)
    spread: dict[str, Spread] = field(default_factory=dict)

    def finish(self) -> None:
        """Compute `spread` from `runs`."""
        keys = list(dict.fromkeys(key for run in self.runs for key in run))
        self.spread = {
            key: summarizeValues([run[key] for run in self.runs if key in run]) for key in keys
        }


@dataclass
class BaselineComparison:
    """The four baselines, over every repeat.

    Attributes:
        labels: The two views of the pair model (`["z_signal", "z_image"]`).
        settings: The `BaselineSettings`, as a dict.
        headline_k: The `k` whose recall the table shows.
        rows: One `BaselineRow` per baseline, in the order of the roadmap.
        integrity: Pair-integrity status -> in how many repeats (`ok`, `suspect`, `not_checked`).
        num_pairs: Pairs in the pool.
        num_train: Training pairs of a repeat.
        num_heldout: Held-out pairs of a repeat.
    """

    labels: list[str]
    settings: dict[str, Any]
    headline_k: int
    rows: list[BaselineRow]
    integrity: dict[str, int]
    num_pairs: int
    num_train: int
    num_heldout: int

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

    def row(self, name: str) -> BaselineRow:
        """Return the row called `name`.

        Raises:
            KeyError: If there is none (the message lists the rows).
        """
        for candidate in self.rows:
            if candidate.name == name:
                return candidate
        raise KeyError(f"No baseline '{name}'. Available: {[r.name for r in self.rows]}.")

    def toMarkdown(self) -> str:
        """The comparison as a Markdown table: mean +/- standard deviation over the repeats.

        Returns:
            The text, ending with a newline.
        """
        k = self.headline_k
        repeats = self.settings["repeats"]
        lines = [
            f"# Alignment baselines: {self.labels[0]} and {self.labels[1]}",
            "",
            f"{self.num_pairs} pairs, {self.num_train} for training and {self.num_heldout} "
            f"held out in each of {repeats} random splits and seeds. Mean +/- standard deviation "
            "over the repeats, on the held-out pairs. Every setting is a placeholder (roadmap "
            "I-15).",
            "",
            f"Pair integrity of the training pairs (the Procrustes retrieval above chance), over "
            f"the repeats: {self.integrity}.",
            "",
            f"| baseline | recall@{k} (chance) | mean rank (chance) | cross R2 | self R2 | "
            "active units | flagged |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in self.rows:
            recall = _cell(row, f"recall@{k}")
            chance = _cell(row, f"chance_recall@{k}", 3)
            lines.append(
                f"| {row.name} | {recall} ({chance}) "
                f"| {_cell(row, 'mean_rank', 2)} ({_cell(row, 'chance_mean_rank', 2)}) "
                f"| {_cell(row, 'cross_r2')} | {_cell(row, 'self_r2')} "
                f"| {_cell(row, 'active_units', 1)} | {_flaggedCell(row)} |"
            )
        lines += ["", "Rows:", ""]
        lines.extend(f"- **{row.name}**: {row.description}" for row in self.rows)
        return "\n".join(lines) + "\n"

    def saveMarkdown(self, path: str | Path) -> None:
        """Write `toMarkdown()` to `path`, creating the parent directories.

        Args:
            path: Destination file path.
        """
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(self.toMarkdown(), encoding="utf-8")

    def summary(self) -> str:
        """The Markdown table, for the terminal."""
        return self.toMarkdown()


def _cell(row: BaselineRow, key: str, digits: int = 3) -> str:
    """`mean +/- std` of a number of a row, or `-` when the baseline has none."""
    spread = row.spread.get(key)
    if spread is None or spread.count == 0:
        return "-"
    return f"{spread.mean:.{digits}f} +/- {spread.std:.{digits}f}"


def _flaggedCell(row: BaselineRow) -> str:
    """How many repeats raised a flag, `-` for a baseline that has no model to guard."""
    if not any("num_flags" in run for run in row.runs):
        return "-"
    flagged = sum(1 for run in row.runs if run.get("num_flags", 0.0) > 0)
    return f"{flagged}/{len(row.runs)}"


def _compose(
    settings: BaselineSettings,
    experiment_name: str,
    overrides: Sequence[str],
    output_dir: Path,
    num_epochs: int,
) -> ExperimentConfig:
    """Compose one stage's experiment. The data group is a placeholder, and no callback is kept.

    The callbacks of the shipped configs write checkpoints and monitor a validation loss; the
    baselines hold the validation data nowhere (the held-out pairs must not select anything), and
    save the final checkpoint themselves.
    """
    experiment = loadExperimentConfig(
        experiment_name,
        overrides=[
            f"data.loader_factory={_PLACEHOLDER_LOADER}",
            "data.train_path=unused",
            f"output_dir={output_dir}",
            "training.loggers=[]",
            f"training.device={settings.device}",
            f"training.num_epochs={num_epochs}",
            f"training.reconstruction_reduction={settings.reconstruction_reduction}",
            *overrides,
        ],
    )
    experiment.training.callbacks = {}
    return experiment


def _trainSolo(
    recipe: BaselineRecipe,
    settings: BaselineSettings,
    modality: str,
    batches: list[dict[str, torch.Tensor]],
    seed: int,
    directory: Path,
) -> Path:
    """Run the solo stage of one modality, save its final checkpoint and return its path."""
    experiment = _compose(
        settings, recipe.solo_experiments[modality], [], directory, settings.solo_epochs
    )
    setGlobalSeed(seed)
    model = buildModelFromConfig(experiment.model)
    trainer = buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)
    trainer.fit(batches, num_epochs=settings.solo_epochs)
    path = directory / "checkpoints" / "solo.pt"
    trainer.saveCheckpoint(path)
    return path


def _stage3Terms(
    recipe: BaselineRecipe, settings: BaselineSettings, distance: str
) -> list[LossTermConfig]:
    """The two pair-stage terms of ADR 0054, each ramped from zero."""

    def ramp(end_value: float) -> BetaScheduleConfig:
        return BetaScheduleConfig(
            strategy="linear_warmup",
            kwargs={"warmup_steps": settings.ramp_steps, "end_value": end_value},
        )

    first, second = recipe.pair_spaces
    return [
        LossTermConfig(
            name="latent_alignment",
            schedule=ramp(settings.alignment_weight),
            kwargs={"pairs": [[first, second]], "distance": distance},
        ),
        LossTermConfig(
            name="cross_reconstruction",
            schedule=ramp(settings.cross_weight),
            kwargs={"pairs": [[first, second], [second, first]]},
        ),
    ]


def _pairModel(
    recipe: BaselineRecipe,
    settings: BaselineSettings,
    solo_paths: Mapping[str, Path],
    split: PairSplit,
    seed: int,
    directory: Path,
    distance: str | None,
) -> GlobalVae:
    """Build the pair model from the solo files (baseline 0), then train stage 3 if asked.

    Args:
        recipe: The recipe.
        settings: The settings.
        solo_paths: Modality -> its solo checkpoint.
        split: The split, whose training batches are the stage's data and the collapse probe.
        seed: Seeds the model construction and the training.
        directory: The `output_dir` of the composed config.
        distance: The latent distance of stage 3. `None` returns the warm-started model untrained.

    Returns:
        The model, in evaluation mode.
    """
    experiment = _compose(
        settings,
        recipe.solo_experiments[recipe.modalities[0]],
        [
            *recipe.pair_overrides,
            "training.modality_dropout_p=0.0",
            f"training.beta={settings.beta}",
        ],
        directory,
        settings.pair_epochs,
    )
    experiment.training.warm_start = [
        WarmStartEntryConfig(checkpoint=str(solo_paths[modality]), modules=[modality])
        for modality in recipe.modalities
    ]
    if distance is not None:
        experiment.training.loss_terms = _stage3Terms(recipe, settings, distance)
    setGlobalSeed(seed)
    model = buildModelFromConfig(experiment.model).to(settings.device)
    applyWarmStartFromConfig(
        model, experiment.training, experiment.model, probe_dataloader=split.train
    )
    if distance is not None:
        trainer = buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)
        trainer.fit(split.train, num_epochs=settings.pair_epochs)
    model.eval()
    return model


def _jointModel(
    recipe: BaselineRecipe, settings: BaselineSettings, split: PairSplit, seed: int, directory: Path
) -> GlobalVae:
    """Train the joint model with a shared latent space from scratch, on the training pairs."""
    experiment = _compose(
        settings,
        recipe.solo_experiments[recipe.modalities[0]],
        [*recipe.joint_overrides, f"training.beta={settings.beta}"],
        directory,
        settings.total_joint_epochs,
    )
    setGlobalSeed(seed)
    model = buildModelFromConfig(experiment.model).to(settings.device)
    trainer = buildTrainerFromConfig(model, experiment.training, model_config=experiment.model)
    trainer.fit(split.train, num_epochs=settings.total_joint_epochs)
    model.eval()
    return model


def runAlignmentBaselines(
    pool: PairPool,
    recipe: BaselineRecipe | None = None,
    settings: BaselineSettings | None = None,
    work_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> BaselineComparison:
    """Run the four baselines on `settings.repeats` random splits and seeds of a pool.

    For each repeat: split the pool, train the two solo models (stages 1 and 2), then score
    baseline 0 (the solo spaces in the pair model, no stage 3) and baseline 1 (the Procrustes map
    fitted on the training pairs of the same models), train stage 3 once per distance
    (baseline 2), and train the joint model (baseline 3); every one is scored on the same
    held-out pairs by `evaluateLatentAlignment`. The solo models' active units, measured on the
    held-out pairs, are the reference of the collapse guard of every later model.

    Args:
        pool: The paired samples.
        recipe: The experiment configs of the stages. Defaults to the signal plus image recipe.
        settings: The settings. Defaults to `BaselineSettings()`.
        work_dir: Where the stages write their checkpoints. `None` (default): a temporary
            directory, removed at the end.
        log: Called with one line of progress per stage. Defaults to the module's logger.

    Returns:
        The `BaselineComparison`.

    Raises:
        KeyError: If a distance in `settings.distances` is not registered.
        ValueError: If the pool or a setting is invalid, or a solo model collapses (the warm start
            refuses it: `docs/how-to/choose-beta.md`).
    """
    resolved_recipe = recipe if recipe is not None else BaselineRecipe()
    resolved_settings = settings if settings is not None else BaselineSettings()
    say = log if log is not None else logger.info
    distances = (
        list(resolved_settings.distances)
        if resolved_settings.distances is not None
        else listRegisteredLatentDistances()
    )
    unknown = sorted(set(distances) - set(listRegisteredLatentDistances()))
    if unknown:
        raise KeyError(
            f"Unknown latent distance(s) {unknown}. Available: {listRegisteredLatentDistances()}."
        )
    first, second = resolved_recipe.modalities
    spaces = resolved_recipe.pair_spaces

    rows: dict[str, BaselineRow] = {
        BASELINE_SOLO: BaselineRow(
            BASELINE_SOLO,
            "The two solo models in the pair model, no stage 3: expected at chance, since nothing "
            "made the two spaces say the same thing.",
        ),
        BASELINE_PROCRUSTES: BaselineRow(
            BASELINE_PROCRUSTES,
            "One orthogonal map between the solo posterior means, fitted on the training pairs in "
            "closed form and applied to the held-out pairs.",
        ),
    }
    for distance in distances:
        name = f"{BASELINE_STAGE3} {distance}"
        rows[name] = BaselineRow(
            name,
            f"Stage 3 from the solo models with the `{distance}` latent distance and the cross "
            "reconstruction term.",
        )
    if resolved_settings.include_joint:
        rows[BASELINE_JOINT] = BaselineRow(
            BASELINE_JOINT,
            "One model with a shared latent space and a product-of-experts fusion, trained on the "
            f"pairs from scratch for {resolved_settings.total_joint_epochs} epochs.",
        )

    integrity = {"ok": 0, "suspect": 0, "not_checked": 0}
    counts: dict[str, int] = {}
    labels: list[str] = list(spaces)
    owned = tempfile.TemporaryDirectory() if work_dir is None else None
    root = Path(owned.name if owned is not None else str(work_dir))
    try:
        for repeat in range(resolved_settings.repeats):
            seed = resolved_settings.seed + repeat
            split = splitPool(
                pool, resolved_settings.heldout_fraction, seed, resolved_settings.batch_size
            )
            counts = {
                "train": sum(int(next(iter(b.values())).shape[0]) for b in split.train),
                "heldout": sum(int(next(iter(b.values())).shape[0]) for b in split.heldout),
            }
            folder = root / f"repeat_{repeat}"
            say(f"repeat {repeat + 1}/{resolved_settings.repeats} (seed {seed}): solo stages")
            solo_paths = {
                modality: _trainSolo(
                    resolved_recipe,
                    resolved_settings,
                    modality,
                    split.solo[modality],
                    seed,
                    folder / f"solo_{modality}",
                )
                for modality in resolved_recipe.modalities
            }

            def score(
                model: GlobalVae,
                views: tuple[str, str],
                reference: Mapping[str, int] | None,
                train: list[dict[str, torch.Tensor]] | None,
                split: PairSplit = split,
                seed: int = seed,
            ) -> LatentAlignmentReport:
                return evaluateLatentAlignment(
                    model,
                    views[0],
                    views[1],
                    split.heldout,
                    train=train,
                    distances=distances,
                    ks=resolved_settings.ks,
                    similarity=resolved_settings.similarity,
                    device=resolved_settings.device,
                    reference_active_units=reference,
                    procrustes_center=resolved_settings.procrustes_center,
                    seed=seed,
                )

            say("  baselines 0 and 1: solo spaces and Procrustes")
            solo_model = _pairModel(
                resolved_recipe, resolved_settings, solo_paths, split, seed, folder / "pair", None
            )
            solo_report = score(solo_model, spaces, None, split.train)
            labels = solo_report.labels
            rows[BASELINE_SOLO].runs.append(flattenAlignmentReport(solo_report))
            rows[BASELINE_PROCRUSTES].runs.append(
                flattenAlignmentReport(solo_report, procrustes=True)
            )
            integrity[solo_report.pair_integrity.status] += 1
            solo_units = {label: solo_report.collapse[label].active_units for label in labels}

            for distance in distances:
                say(f"  baseline 2: stage 3 with {distance}")
                model = _pairModel(
                    resolved_recipe,
                    resolved_settings,
                    solo_paths,
                    split,
                    seed,
                    folder / f"pair_{distance}",
                    distance,
                )
                report = score(model, spaces, solo_units, None)
                rows[f"{BASELINE_STAGE3} {distance}"].runs.append(flattenAlignmentReport(report))

            if resolved_settings.include_joint:
                say("  baseline 3: joint training from scratch")
                joint = _jointModel(
                    resolved_recipe, resolved_settings, split, seed, folder / "joint"
                )
                shared = resolved_recipe.joint_space
                joint_reference = {
                    first: solo_units[spaces[0]],
                    second: solo_units[spaces[1]],
                }
                report = score(
                    joint, (f"{shared}@{first}", f"{shared}@{second}"), joint_reference, None
                )
                rows[BASELINE_JOINT].runs.append(flattenAlignmentReport(report))
    finally:
        if owned is not None:
            owned.cleanup()

    for row in rows.values():
        row.finish()
    headline_k = chooseSelectiveK(resolved_settings.ks, counts["heldout"])
    settings_dict = asdict(resolved_settings)
    settings_dict["joint_epochs"] = resolved_settings.total_joint_epochs
    return BaselineComparison(
        labels=labels,
        settings=settings_dict,
        headline_k=headline_k,
        rows=list(rows.values()),
        integrity=integrity,
        num_pairs=pool.num_pairs,
        num_train=counts["train"],
        num_heldout=counts["heldout"],
    )
