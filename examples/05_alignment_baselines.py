#!/usr/bin/env python3
"""The four alignment baselines on a synthetic linked pair, in one command (roadmap P6-6).

A distance between two separately trained latent spaces that falls during stage 3 is not a result:
two collapsed spaces also minimize it (roadmap I-8). This script runs the alternatives side by side
on the same pairs and the same held-out split, so that the learned alignment is read against what
it has to beat (`docs/adr/0056-evaluation-across-two-latent-spaces.md`):

- **0, the solo spaces as they are**: the two solo models in the pair model, no stage 3. Expected at
  chance for retrieval.
- **1, post-hoc orthogonal Procrustes**: one orthogonal map between the solo posterior means,
  fitted on the training pairs in closed form.
- **2, stage 3, once per registered latent distance**: the pair model warm-started from the solo
  models and trained with `latent_alignment` and `cross_reconstruction` (ADR 0054).
- **3, joint training from scratch**: one model with a shared latent space and a product-of-experts
  fusion, trained on the pairs from the start for as many epochs as the staged route spends.

Each is scored by `global_vae.evaluation.latent_alignment.evaluateLatentAlignment` on held-out
pairs (retrieval with its chance level, cross reconstruction next to self reconstruction, active
units against the solo stage, and the Procrustes check that the pairs match), and the table is a
**mean and a spread over repeated random splits and seeds**, never one run: the reference paper uses
five seeds on 72 pairs, which are this script's defaults.

The data is synthetic and linked: the signal is a sine of a random frequency and phase, and the
image is a Gaussian blob whose place is set by that frequency (horizontal) and that phase
(vertical), so the two modalities of a row describe the same two numbers. It is not a measurement
of anything (roadmap I-15), and nothing here says which route is best on real data: it shows that
the harness runs, what the guards catch, and what to look at.

    python examples/05_alignment_baselines.py
    python examples/05_alignment_baselines.py --repeats 2 --pair-epochs 40 \
        --distances cosine mean_mse

For your own pairs, build a `PairPool` of your tensors and call
`global_vae.evaluation.alignment_baselines.runAlignmentBaselines` with a `BaselineRecipe` that names
your solo and pair experiments (spec §6.2: the dataset is the caller's).

Outputs go to `--output-dir` (default `examples/outputs/05_alignment_baselines/`, git-ignored):
`comparison.md` (the table) and `comparison.json` (every number of every repeat). A run of the
defaults trains about eleven models per repeat and takes a long time on a CPU; fewer repeats,
epochs and distances make a quick check.
"""

import argparse
import logging
from pathlib import Path

import torch

from global_vae.evaluation.alignment_baselines import (
    BaselineComparison,
    BaselineSettings,
    PairPool,
    runAlignmentBaselines,
)
from global_vae.losses.distances.registry import listRegisteredLatentDistances

SIGNAL_LENGTH = 256
IMAGE_SIZE = 64
DEFAULT_NUM_PAIRS = 72
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "05_alignment_baselines"


def makeLinkedRows(count: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `count` linked samples: a `(count, 256)` signal and a `(count, 64, 64)` image.

    The signal is a sine of a random frequency and phase. The image is a Gaussian blob whose
    horizontal place is set by that frequency and whose vertical place is set by that phase, so
    the two rows describe the same two numbers.

    Args:
        count: How many samples.
        generator: The private random generator the numbers are drawn from.

    Returns:
        `(signals, images)`.
    """
    time = torch.linspace(0.0, 1.0, SIGNAL_LENGTH)
    axis = torch.linspace(-1.0, 1.0, IMAGE_SIZE)
    frequency = torch.rand(count, 1, generator=generator) * 8.0 + 1.0
    phase = torch.rand(count, 1, generator=generator) * 3.14
    signals = torch.sin(2.0 * 3.14159 * frequency * time + phase)
    center_x = ((frequency - 5.0) / 4.0 * 0.8).unsqueeze(-1)
    center_y = ((phase - 1.57) / 1.57 * 0.8).unsqueeze(-1)
    squared = (axis[None, None, :] - center_x) ** 2 + (axis[None, :, None] - center_y) ** 2
    return signals, torch.exp(-squared / 0.05)


def makeLinkedPool(num_pairs: int, num_unpaired: int = 0, seed: int = 0) -> PairPool:
    """Build the pool of linked pairs, and optionally extra unpaired samples of each modality.

    Args:
        num_pairs: How many pairs.
        num_unpaired: How many extra samples of each modality that have no partner. They are used
            by the solo stages only.
        seed: Seeds the generator: one seed, one pool.

    Returns:
        The `PairPool`.
    """
    generator = torch.Generator().manual_seed(seed)
    signals, images = makeLinkedRows(num_pairs, generator)
    unpaired = {}
    if num_unpaired > 0:
        extra_signals, _ = makeLinkedRows(num_unpaired, generator)
        _, extra_images = makeLinkedRows(num_unpaired, generator)
        unpaired = {"signal": extra_signals, "image": extra_images}
    return PairPool(paired={"signal": signals, "image": images}, unpaired=unpaired)


def buildArgumentParser() -> argparse.ArgumentParser:
    """Build the command-line parser of this script."""
    parser = argparse.ArgumentParser(
        description="Run the four alignment baselines on a synthetic linked pair (roadmap P6-6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", type=int, default=DEFAULT_NUM_PAIRS, help="Pairs in the pool.")
    parser.add_argument(
        "--unpaired", type=int, default=0, help="Extra unpaired samples per modality (solo only)."
    )
    parser.add_argument("--repeats", type=int, default=5, help="Random splits and seeds.")
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0, help="First seed; repeat r uses seed + r.")
    parser.add_argument("--solo-epochs", type=int, default=40)
    parser.add_argument("--pair-epochs", type=int, default=100, help="Epochs of each stage 3.")
    parser.add_argument(
        "--joint-epochs",
        type=int,
        default=None,
        help="Epochs of the joint model (default: solo plus pair epochs).",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        choices=listRegisteredLatentDistances(),
        default=None,
        help="Latent distances to run stage 3 with (default: every registered one).",
    )
    parser.add_argument("--no-joint", action="store_true", help="Skip baseline 3.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def run(argv: list[str] | None = None) -> BaselineComparison:
    """Run the baselines as the command line says, write the outputs and return the comparison.

    Args:
        argv: Command-line arguments (excluding the program name). `None` reads `sys.argv`.

    Returns:
        The `BaselineComparison`, also written to `comparison.md` and `comparison.json` in
        `--output-dir`.
    """
    args = buildArgumentParser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = BaselineSettings(
        repeats=args.repeats,
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
        solo_epochs=args.solo_epochs,
        pair_epochs=args.pair_epochs,
        joint_epochs=args.joint_epochs,
        distances=tuple(args.distances) if args.distances is not None else None,
        include_joint=not args.no_joint,
        device=args.device,
    )
    pool = makeLinkedPool(args.pairs, args.unpaired, args.seed)
    comparison = runAlignmentBaselines(pool, settings=settings)
    print(comparison.summary())
    comparison.saveMarkdown(args.output_dir / "comparison.md")
    comparison.save(args.output_dir / "comparison.json")
    print(f"Wrote the comparison to {args.output_dir}.")
    return comparison


if __name__ == "__main__":
    run()
