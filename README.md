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
value; raise it as coverage improves. The `lint` job is advisory until the ruff clean-up
(roadmap P0-2) lands, the other jobs are blocking.

## Repository structure

See spec §8 for the target layout; `src/global_vae/` mirrors it.

## Naming convention (deviates from PEP8 — read this before contributing)

- Classes → `CamelCase` (e.g. `GlobalVae`, `SignalEncoder`).
- Variables → `snake_case` (e.g. `latent_dim`, `batch_size`).
- **Functions and methods → `camelCase`** (e.g. `registerEncoder`,
  `computeKlLoss`), not PEP8's usual `snake_case`. This is intentional
  (spec §10). `ruff`'s `N802`/`N803`/`N806` naming rules are disabled
  in `pyproject.toml` specifically so linting doesn't silently "fix"
  this back to snake_case. Framework-mandated overrides
  (`forward`, `__init__`, and other PyTorch/Python dunder or
  base-class-required names) are the only exception — leave those as
  the base class defines them.

## Adding a new modality (spec §10 checklist)

1. Subclass `AbstractEncoder` in `encoders/`, decorate it with
   `@registerEncoder("your_encoder_name")`.
2. Subclass `AbstractDecoder` in `decoders/`, decorate it with
   `@registerDecoder("your_decoder_name")`.
3. Register both (the decorator does this — nothing else to wire up).
4. Add a config entry referencing the two registry names (see
   `configs/model/default.yaml` for the shape, once config loading is
   wired up).
5. Add a test: a unit test for the encoder/decoder shapes (in `tests/unit/`), and
   ideally an entry in the relevant integration test (in `tests/integration/`).

No core framework file should need to change. If it does, that's a
signal the registry pattern is being bypassed somewhere — flag it
rather than special-casing the new modality into `GlobalVae`.

## Adding a new fusion, assembler, or data transform strategy

Same pattern: subclass `AbstractFusion` (`fusion/`), `AbstractAssembler`
(`assemblers/`), or `AbstractTransform` (`data/transforms/`), register with
`@registerFusion("name")` / `@registerAssembler("name")` /
`@registerTransform("name")`. A new transform must stay fully generic
across dimensionality (spec §6.2): no per-modality or per-dataset logic.

## Extending beyond `EN-L1-DN`

The routing-graph machinery (`latent/base.py`, `latent/routing_graph_builders/`)
already supports arbitrary encoder-latent-decoder
topologies, including multiple independent latent spaces. `GlobalVae`
currently only *drives* the single-fused-latent case end-to-end; growing
it (or introducing sibling model classes) to cover the other 7
configurations in spec §2.1 is the next milestone. See
`docs/adr/0001-phase1-default-configuration.md`.
