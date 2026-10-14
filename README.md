# Global Multimodal VAE

[![CI](https://github.com/Venon282/GVAE/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Venon282/GVAE/actions/workflows/ci.yaml)

## Documentation
https://venon282.github.io/GVAE/

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
```

The `docs` extra is needed by the test that builds the documentation site; without it that
test is skipped.

## Running the signal-VAE milestone (spec §6.1 milestone 1)

```bash
python scripts/train.py \
    data.train_path=/path/to/your/data \
    data.loader_factory=my_project.data:buildSignalDataloaders
```

`data.loader_factory` must point at your own `(DataConfig) -> DataloaderBundle`
callable (spec: data loading stays your own responsibility). See
`configs/experiment/signal_vae.yaml` for the full default config and
`global_vae/config/data.py` for the exact contract, including the generic
per-modality `transforms` pipeline (`log`/`standardize`/`resample`, spec §6.2,
keyed by modality name so a second signal dataset never has to share the
first's steps/statistics) your own `loader_factory` can call via
`buildTransformPipeline(config.data)` if it wants to. Override any
hyperparameter from the command line, e.g.
`training.num_epochs=50 training.optimizer.kwargs.lr=0.0003`.

Inspect the result once trained:

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

## Running checks

```bash
ruff check .
ruff format --check .
mypy
pytest --cov=global_vae --cov-fail-under=95
```

`mypy` with no arguments checks `src/`, `scripts/`, `examples/` and `tests/` in strict mode
(`[tool.mypy]` in `pyproject.toml`). A test function may carry no annotations at all, but a
function with some annotated and some unannotated parameters is an error: write
`tmp_path: Path`, not a bare `tmp_path`. Run it on Python 3.11, the version `python_version`
names and the one CI uses. On Python 3.12 or newer pip installs numpy 2.5, whose stubs make mypy
report errors that are not in the code. On a newer interpreter, pass `--python-version` (for
example `mypy --python-version 3.13`) for a local check; CI stays the reference.

### Test layout and fast loops

`tests/unit/` holds the tests of one component at a time (a class, a function, a registry),
built from hand-made tensors: no training loop, no files on disk, no subprocess.
`tests/integration/` holds everything that wires several components together: a model
assembled from the registries, a `Trainer` run, a config composed through Hydra, a script or an
example, the docs build. A module that mixes both levels stays in `tests/integration/`.

The `slow` marker tags the tests that run a script or an example in a subprocess, build the
docs site or run UMAP. Two fast loops for day-to-day work:

```bash
pytest tests/unit        # the unit tests only
pytest -m "not slow"     # every test except the slow ones
```

Markers are strict (`--strict-markers` in `pyproject.toml`): a marker that is not registered
there is an error, not a warning. CI runs the whole suite, slow tests included.

GitHub Actions (`.github/workflows/ci.yaml`) runs the same checks on every push and pull
request: `lint` (`ruff`), `types` (`mypy`, Python 3.11) and `tests` (`pytest` on Python 3.11
and 3.13). The tests fail under a coverage floor of 95%, set one point below the measured
value; raise it as coverage improves. All three jobs are blocking.

## Releasing

The version is written in one place, `__version__` in `src/global_vae/__init__.py`.
`pyproject.toml` declares it `dynamic` and reads it from there (`[tool.setuptools.dynamic]`), so
the package metadata, `global_vae.__version__` and the `global_vae_version` stored in every
checkpoint cannot drift apart. Release checklist:

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
5. Run the checks of the previous section, then in a fresh virtual environment:
   `pip install .`, `python -c "import global_vae.config, global_vae; print(global_vae.__version__)"`
   and `pip check`. The printed version must equal the tag without its `v`.
6. Tag the commit (`git tag v<x.y.z>`) and push the tag.

`src/global_vae/py.typed` marks the package as typed (PEP 561), so a downstream `mypy` uses its
type hints. It is declared as package data in `pyproject.toml`. The licence is MIT (`LICENSE`).

## Repository structure

See spec §8 for the target layout; `src/global_vae/` mirrors it.

## Naming convention (deviates from PEP8: read this before contributing)

- Classes -> `CamelCase` (e.g. `GlobalVae`, `OneDCnnEncoder`).
- Variables -> `snake_case` (e.g. `latent_dim`, `batch_size`).
- Modules (files) and packages -> `snake_case`, named after the class they hold
  (`one_d_cnn_encoder.py` holds `OneDCnnEncoder`). `ruff`'s `N999` is on, so a CamelCase
  module name fails the lint job. The old CamelCase import paths of the eight CNN encoder
  and decoder modules (`global_vae.encoders.OneDCnnEncoder`) still work and emit a
  `DeprecationWarning` until 2.0 (ADR 0023): import from the snake_case module instead.
- **Functions and methods -> `camelCase`** (e.g. `registerEncoder`,
  `computeRegularizationLoss`), not PEP8's usual `snake_case`. This is intentional
  (spec §10). `ruff`'s `N802`/`N803`/`N806` naming rules are disabled
  in `pyproject.toml` specifically so linting doesn't silently "fix"
  this back to snake_case. Framework-mandated overrides
  (`forward`, `__init__`, and other PyTorch/Python dunder or
  base-class-required names) are the only exception: leave those as
  the base class defines them.

## Adding a new modality (spec §10 checklist)

1. Subclass `AbstractEncoder` in `encoders/`, decorate it with
   `@registerEncoder("your_encoder_name")`.
2. Subclass `AbstractDecoder` in `decoders/`, decorate it with
   `@registerDecoder("your_decoder_name")`.
3. Register both (the decorator does this: nothing else to wire up).
4. Add a config entry referencing the two registry names (see
   `configs/model/default.yaml` for a two-modality example,
   `configs/model/signal_single_latent.yaml` for a single-modality one and
   `docs/how-to/configure-latent-spaces.md` for several latent spaces).
5. Add a test: a unit test for the encoder/decoder shapes (in `tests/unit/`), and
   ideally an entry in the relevant integration test (in `tests/integration/`).

For one encoder over several modalities, or one decoder reconstructing
several, subclass `AbstractSharedEncoder` / `AbstractSharedDecoder` instead
and name them under `encoders` / `decoders` (see
`docs/how-to/add-a-modality.md`).

No core framework file should need to change. If it does, that's a
signal the registry pattern is being bypassed somewhere. Flag it
rather than special-casing the new modality into `GlobalVae`.

## Adding a new fusion, assembler, or data transform strategy

Same pattern: subclass `AbstractFusion` (`fusion/`), `AbstractAssembler`
(`assemblers/`), or `AbstractTransform` (`data/transforms/`), register with
`@registerFusion("name")` / `@registerAssembler("name")` /
`@registerTransform("name")`. A new transform must stay fully generic
across dimensionality (spec §6.2): no per-modality or per-dataset logic.

## Extending beyond `EN-L1-DN`

The routing-graph machinery (`latent/base.py`, `latent/routing_graph_builders/`)
supports arbitrary encoder-latent-decoder topologies, including multiple
independent latent spaces, and `GlobalVae` accepts an explicit `RoutingGraph`
(`docs/adr/0002-generalize-global-vae-to-routing-graph.md`). Two presets are
provided: `buildSingleLatentRoutingGraph` (`EN-L1-DN`, what
`GlobalVae.createSingleLatent` and `configs/model/` build) and
`buildSharedPrivateRoutingGraph` (a shared plus per-modality private latent
space, with a latent head on each encoder -> latent edge). An encoder may feed
several latent spaces through latent heads (`identity`, `linear`;
`docs/adr/0024-encoder-fan-out-through-latent-heads.md`). Config files can
describe all of it (`docs/adr/0025-config-schema-for-several-latent-spaces.md`):
`latent_mode: several` with `latent.spaces` and `latent.decoders_consume`, a
`head` on each `fed_by` edge, and separate `encoders` and `decoders` sections
for names that are not modality names. `configs/model/` has one file per
pattern of spec §9 (`signal_image_independent_latents.yaml`,
`signal_image_shared_private.yaml`) and the model of example 03
(`signal_image_to_image.yaml`), and `configs/experiment/` an experiment for
each, runnable with `scripts/train.py`.

All 8 configurations of spec §2.1 are built. The four `EN-*` rows (one encoder
per modality) include the shared plus private topology, which can be trained
and evaluated, from Python or from a config file. The four `E1-*` rows (one
shared encoder over several modalities) and the `*-D1` rows (one shared decoder
reconstructing several) use `AbstractSharedEncoder` and `AbstractSharedDecoder`
(`docs/adr/0027-shared-encoders-and-decoders.md`): a shared encoder declares
the modalities it ingests and receives them as a `dict[str, Tensor]`, a shared
decoder returns one, and `GlobalVae` flattens it into `"reconstructions"` by
modality, so the losses, the evaluation and the figures are unchanged. No
concrete shared backbone ships yet (roadmap P3-5): the tests build the four
rows with small dummy modules, and `docs/how-to/add-a-modality.md` says what to
implement.

Remaining gaps are tracked in the [roadmap](docs/roadmap.md). See also
`docs/adr/0001-phase1-default-configuration.md`.

## Roadmap

[`docs/roadmap.md`](docs/roadmap.md) lists what is done, what is open and the
decisions waiting on the owner, in priority order.
