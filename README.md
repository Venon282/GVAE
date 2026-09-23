# Global Multimodal VAE

## Documentation
https://venon282.github.io/GVAE/

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

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
pytest
```

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
5. Add a test — a unit test for the encoder/decoder shapes, and ideally
   an entry in the relevant integration test.

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
