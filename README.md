# Global Multimodal VAE

[![CI](https://github.com/Venon282/GVAE/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Venon282/GVAE/actions/workflows/ci.yaml)

A modular multimodal Variational Autoencoder framework. It is not a model for two fixed kinds of
data: it is an architecture where **adding a modality never requires touching the core**. You write
an encoder and a decoder, register them by name, point a YAML file at them, and train.

- **Documentation:** <https://venon282.github.io/GVAE/>
- **Specification:** [`docs/global-vae-project-specification.md`](docs/global-vae-project-specification.md)
- **Roadmap:** [`docs/roadmap.md`](docs/roadmap.md)

## Contents

- [Quick start](#quick-start)
- [Supported modalities](#supported-modalities)
- [Model topologies](#model-topologies)
- [Training and evaluation features](#training-and-evaluation-features)
- [Extending the framework](#extending-the-framework)
- [Development](#development)
- [Releasing](#releasing)
- [Conventions](#conventions)

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
```

The `docs` extra is needed by the test that builds the documentation site. Without it, that test is
skipped.

### 2. Train

```bash
python scripts/train.py \
    data.train_path=/path/to/your/data \
    data.loader_factory=my_project.data:buildSignalDataloaders
```

Data loading is **your** code: `data.loader_factory` points at a `(DataConfig) -> DataloaderBundle`
function (spec §6.2). The contract is in `global_vae/config/data.py`, and the full default config is
`configs/experiment/signal_vae.yaml`. Any value can be overridden from the command line:

```bash
python scripts/train.py training.num_epochs=50 training.optimizer.kwargs.lr=0.0003
```

### 3. Inspect and evaluate

```bash
python scripts/visualize_latent.py \
    --checkpoint outputs/signal_vae/checkpoints/best.pt \
    --model-factory my_project.data:buildSignalModel \
    --dataloader-factory my_project.data:buildSignalDataloaders

python scripts/evaluate.py \
    --checkpoint outputs/signal_vae/checkpoints/best.pt \
    --model-factory my_project.data:buildSignalModel \
    --dataloader-factory my_project.data:buildSignalTestDataloader \
    --output-dir results/
```

`visualize_latent.py` writes the loss curves (and a learning-rate curve when a callback changed the
rate). `evaluate.py` writes a per-modality metrics table (`metrics_table.md` and `.csv`) and, per
latent space, the KL of each dimension and the number of active units. More in
[Evaluate and visualize a checkpoint](docs/how-to/evaluate.md).

## Supported modalities

Every modality below went through the same checklist and was added **with no change to the core**.
Each has a ready-made experiment you can run with `scripts/train.py --config-name experiment/<name>`
and a how-to that shows the part that is yours: the `loader_factory`.

| Modality | Batch tensor | Experiment | Guide |
|---|---|---|---|
| 1D signal | `(batch, length)` | `signal_vae`, `signal_resnet_vae`, `signal_transformer_vae` | [Train a model](docs/how-to/train.md) |
| Image | `(batch, H, W)` | `image_vae`, `image_resnet_vae`, `image_vit_vae` | [Train a model](docs/how-to/train.md) |
| Table | `(batch, num_features)` | `tabular_vae` | [Train on a table](docs/how-to/train-on-a-table.md) |
| Audio waveform | `(batch, samples)` in `[-1, 1]` | `audio_vae` | [Train on audio](docs/how-to/train-on-audio.md) |
| Text | one-hot `(batch, length, vocab)` | `text_vae` | [Train on text](docs/how-to/train-on-text.md) |
| Time series | `(batch, channels, length)`, `NaN` = not observed | `time_series_vae` | [Train on time series](docs/how-to/train-on-time-series.md) |
| Graph | adjacency `(batch, N, N)` | `graph_vae` | [Train on graphs](docs/how-to/train-on-graphs.md) |
| Point cloud | `(batch, N, 3)`, `NaN` row = no point | `point_cloud_vae` | [Train on point clouds](docs/how-to/train-on-point-clouds.md) |

A few things worth knowing before you pick one:

- **A spectrogram needs no new class.** It is an image, so the 2D classes read it.
- **Two modalities have a loss a YAML file cannot name.** Time series with gaps and point clouds
  train through Python with a custom loss function (a masked error and the Chamfer distance).
  Their how-tos show how. The experiment file of a point cloud says so in its header.
- **Each modality has an ADR** with the measurements behind its design
  ([0039](docs/adr/0039-tabular-modality.md) to [0044](docs/adr/0044-point-cloud-modality.md)).

## Model topologies

The routing-graph machinery (`latent/`) supports any encoder -> latent -> decoder topology, and
config files can describe all of it. All 8 configurations of spec §2.1 are built.

| Pattern | What it is | Example config |
|---|---|---|
| One latent space | The default, `EN-L1-DN`: one encoder per modality, fused into one latent space | `configs/model/signal_image_to_image.yaml` |
| Independent latent spaces | One latent space per modality | `configs/model/signal_image_independent_latents.yaml` |
| Shared plus private | A shared latent space and one private space per modality | `configs/model/signal_image_shared_private.yaml` |
| Shared encoder | One encoder over several modalities (`E1-*` rows) | `configs/model/signal_image_shared_transformer.yaml` |
| Shared decoder | One decoder reconstructing several modalities (`*-D1` rows) | see [add a modality](docs/how-to/add-a-modality.md) |

Details: [configure latent spaces](docs/how-to/configure-latent-spaces.md),
[choose a fusion strategy](docs/how-to/choose-a-fusion-strategy.md) and ADRs
[0002](docs/adr/0002-generalize-global-vae-to-routing-graph.md),
[0024](docs/adr/0024-encoder-fan-out-through-latent-heads.md),
[0025](docs/adr/0025-config-schema-for-several-latent-spaces.md) and
[0027](docs/adr/0027-shared-encoders-and-decoders.md).

## Training and evaluation features

Everything in the second table is off by default, so a run that sets none of it is unchanged.

| Topic | Setting | Where to read more |
|---|---|---|
| Reconstruction scale | `training.reconstruction_reduction` (`mean` or `sum`, per modality) | [Choose beta](docs/how-to/choose-beta.md), [ADR 0029](docs/adr/0029-loss-scale-convention.md) |
| Loss and weight per modality | `training.reconstruction_loss`, `training.reconstruction_weight` | [Train a model](docs/how-to/train.md), [ADR 0030](docs/adr/0030-per-modality-reconstruction-loss-and-weight.md) |
| Paired evaluation | `evaluate.py --paired` (cross-modal retrieval and alignment) | [Evaluate](docs/how-to/evaluate.md), [ADR 0038](docs/adr/0038-paired-data-evaluation.md) |
| Streaming metrics | `evaluate.py --streaming` for a test set too large to pool | [Evaluate](docs/how-to/evaluate.md) |

| Opt-in capability | Setting | ADR |
|---|---|---|
| Mixed precision | `training.mixed_precision` (`bf16` or `fp16`) | [0031](docs/adr/0031-mixed-precision-gradient-accumulation-and-compile.md) |
| Gradient accumulation | `training.accumulate_grad_batches` | [0031](docs/adr/0031-mixed-precision-gradient-accumulation-and-compile.md) |
| `torch.compile` | `training.compile_model` | [0031](docs/adr/0031-mixed-precision-gradient-accumulation-and-compile.md) |
| Resume a run | `training.resume_from` (`num_epochs` is the total) | [0032](docs/adr/0032-resume-from-the-training-script.md) |
| Best K checkpoints | callback `top_k_checkpoint` | [0033](docs/adr/0033-top-k-checkpoints-and-learning-rate-schedules.md) |
| Learning-rate schedules | callbacks `linear_warmup_lr`, `cosine_lr` | [0033](docs/adr/0033-top-k-checkpoints-and-learning-rate-schedules.md) |
| Learning rate in the logs | automatic with `reduce_lr_on_plateau` | [0028](docs/adr/0028-learning-rate-in-the-metrics.md) |
| Experiment tracking | `training.loggers`: `csv`, `tensorboard`, `mlflow` | [0035](docs/adr/0035-mlflow-experiment-logger.md) |

Two things to keep in mind:

- **`beta = 1` is not the usual ELBO.** The reconstruction term is a per-element mean by default,
  while the KL is a sum over latent dimensions. Pick the scale with `reconstruction_reduction`.
- **Checkpoints load with `weights_only=True`** ([ADR 0034](docs/adr/0034-safe-checkpoint-loading-by-default.md)).
  A file written by a version before 2.0 needs `upgradeCheckpoint` or `--trust-checkpoint`.

MLflow is an optional extra (`pip install -e ".[mlflow]"`) and needs no server.

## Extending the framework

### A new modality (spec §10 checklist)

1. Subclass `AbstractEncoder` in `encoders/` and decorate it with `@registerEncoder("your_encoder")`.
2. Subclass `AbstractDecoder` in `decoders/` and decorate it with `@registerDecoder("your_decoder")`.
3. Add both imports to `encoders/__init__.py` and `decoders/__init__.py`, so the decorators run.
4. Add a model config that names the two registered classes (start from
   `configs/model/signal_single_latent.yaml`).
5. Add a unit test for the shapes and gradients, and an integration test for the wiring.

> **No core file should need to change.** If one does, the registry pattern is being bypassed
> somewhere: flag it instead of special-casing your modality into `GlobalVae`.

For one encoder over several modalities, or one decoder reconstructing several, subclass
`AbstractSharedEncoder` or `AbstractSharedDecoder` instead. The full guide, with worked examples
for every modality above, is [Add a new modality](docs/how-to/add-a-modality.md).

### A new fusion, assembler or data transform

Same pattern: subclass `AbstractFusion` (`fusion/`), `AbstractAssembler` (`assemblers/`) or
`AbstractTransform` (`data/transforms/`) and register it with `@registerFusion`,
`@registerAssembler` or `@registerTransform`. A transform must stay generic across dimensionality
(spec §6.2): no per-modality or per-dataset logic. See [add a strategy](docs/how-to/add-a-strategy.md).

## Development

### Checks

```bash
ruff check .
ruff format --check .
mypy
pytest --cov=global_vae --cov-fail-under=95
```

CI (`.github/workflows/ci.yaml`) runs the same three blocking jobs on every push and pull request:
`lint` (`ruff`), `types` (`mypy`, Python 3.11) and `tests` (`pytest`, Python 3.11 and 3.13). The
coverage floor is 95%, one point below the measured value.

> **About `mypy`.** With no arguments it checks `src/`, `scripts/`, `examples/` and `tests/` in
> strict mode. Run it on Python 3.11, the version CI uses: on 3.12 or newer, pip installs numpy 2.5
> and its stubs make `mypy` report errors that are not in the code. On a newer interpreter, pass
> `--python-version` (for example `mypy --python-version 3.13`) and treat CI as the reference. A
> function with some annotated and some unannotated parameters is an error: write
> `tmp_path: Path`, not a bare `tmp_path`.

### Test layout and fast loops

| Folder | Holds |
|---|---|
| `tests/unit/` | One component at a time (a class, a function, a registry), from hand-made tensors: no training loop, no files, no subprocess |
| `tests/integration/` | Everything that wires components together: a model from the registries, a `Trainer` run, a Hydra config, a script, an example, the docs build |

A test module that mixes both levels lives in `tests/integration/`. The `slow` marker tags the
tests that run a subprocess, build the docs site or run UMAP. Markers are strict: an unregistered
one is an error.

```bash
pytest tests/unit        # the unit tests only
pytest -m "not slow"     # everything except the slow tests
```

## Releasing

The version is written in one place, `__version__` in `src/global_vae/__init__.py`. `pyproject.toml`
reads it from there, so the package metadata, `global_vae.__version__` and the `global_vae_version`
stored in every checkpoint cannot drift apart.

<details>
<summary>Release checklist</summary>

1. Pick the next version with semantic versioning (breaking change: major, new feature: minor,
   fix: patch).
2. Set `__version__` to it. Do not add a `version` key to `pyproject.toml`.
3. Move the `[Unreleased]` entries of `docs/changelog/CHANGELOG.md` into a new
   `docs/changelog/CHANGELOG_<x.y.z>_<YYYY-MM-DD>.md` under a `## [x.y.z] - <YYYY-MM-DD>`
   heading, list it first in `docs/changelog/SUMMARY.md`, and leave an empty `[Unreleased]`
   section behind.
4. Check that the tag, `pyproject.toml` and the changelog agree: the tag is `v<x.y.z>`, the
   `__version__` that `pyproject.toml` reads is `<x.y.z>`, and the newest changelog file is
   `CHANGELOG_<x.y.z>_<date>.md`. `tests/unit/test_packaging.py` checks the last two.
5. Run the checks above, then in a fresh virtual environment: `pip install .`,
   `python -c "import global_vae.config, global_vae; print(global_vae.__version__)"` and
   `pip check`. The printed version must equal the tag without its `v`.
6. Tag the commit (`git tag v<x.y.z>`) and push the tag.

</details>

The package is typed (`py.typed`, PEP 561) and the licence is MIT (`LICENSE`).

## Conventions

Naming deviates from PEP 8 on purpose. Read this before contributing.

| What | Style | Example |
|---|---|---|
| Classes | `CamelCase` | `GlobalVae`, `OneDCnnEncoder` |
| Variables | `snake_case` | `latent_dim`, `batch_size` |
| Modules and packages | `snake_case`, named after their class | `one_d_cnn_encoder.py` holds `OneDCnnEncoder` |
| Functions and methods | **`camelCase`** | `registerEncoder`, `computeRegularizationLoss` |

- `ruff`'s `N999` is on, so a CamelCase module name fails the lint job.
- `camelCase` functions are intentional (spec §10). The `N802`, `N803` and `N806` rules are disabled
  in `pyproject.toml` so the linter does not "fix" them back. Methods that PyTorch or Python
  require (`forward`, `__init__`, dunder names) keep the name the base class defines.
- Spec §10 also bans em dashes and unicode arrows in code, comments and docs (a test checks it).

## Repository structure

See spec §8 for the target layout. `src/global_vae/` mirrors it, and
[`docs/roadmap.md`](docs/roadmap.md) lists what is done, what is open and the decisions waiting on
the owner.
