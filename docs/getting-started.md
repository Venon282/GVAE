# Getting Started

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the full pipeline on synthetic data (no setup needed)

```bash
python examples/01_signal_vae_pipeline.py
```

This runs the entire spec §6.1 milestone 1 pipeline end to end — data,
transforms, model, training, evaluation, visualization — on synthetic 1D
signals, no external dataset required. Outputs land in
`examples/outputs/01_signal_vae_pipeline/`.

For the same pipeline driven entirely from `configs/` YAML instead, with a
side-by-side comparison of two named experiment variants, see
`examples/02_config_driven_pipeline.py` (and `examples/README.md` for the
full option list).

## Train on your own data

```bash
python scripts/train.py \
    data.train_path=/path/to/your/data \
    data.loader_factory=my_project.data:buildSignalDataloaders
```

`data.loader_factory` must point at your own `(DataConfig) -> DataloaderBundle`
callable: dataset loading, pairing, and splitting are deliberately outside
this framework's scope, permanently (see the
[project specification](global-vae-project-specification.md), §6.2).
Override any hyperparameter from the command line, e.g.
`training.num_epochs=50 training.optimizer.kwargs.lr=0.0003`.

See [Train a model](how-to/train.md) and
[Evaluate a checkpoint](how-to/evaluate.md) for the full walkthroughs.

## Run the checks

```bash
ruff check .
ruff format --check .
mypy
pytest
```
