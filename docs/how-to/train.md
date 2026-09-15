# Train a model

Training is Hydra-driven: `scripts/train.py` composes
`configs/model/*.yaml` + `configs/data/*.yaml` + `configs/training/*.yaml`
(via a `configs/experiment/*.yaml` file's own `defaults` list) into one
validated `ExperimentConfig`, then builds a real model, trainer, and
dataloaders from it.

## Basic usage

```bash
python scripts/train.py \
    data.train_path=/path/to/your/data \
    data.loader_factory=my_project.data:buildSignalDataloaders
```

`data.loader_factory` must point at your own
`(DataConfig) -> DataloaderBundle` callable: dataset loading, pairing, and
splitting stay entirely your own responsibility — a permanent boundary, not
a gap (project specification, §6.2).

## Overriding hyperparameters

Any field is overridable from the command line with Hydra's dotlist syntax:

```bash
python scripts/train.py \
    training.num_epochs=50 \
    training.optimizer.kwargs.lr=0.0003
```

## Running a different experiment file

```bash
python scripts/train.py --config-name experiment/signal_resnet_vae
```

## Loading a config programmatically

Outside the `scripts/train.py` CLI entry point (e.g. from a notebook or your
own script):

```python
from global_vae.config.experiment import loadExperimentConfig

config = loadExperimentConfig(
    config_name="experiment/signal_vae",
    overrides=["training.num_epochs=5"],
)
```

See the
[`ExperimentConfig` API reference](../reference/global_vae/config/experiment.md)
for the full contract, including every field's default and what it's
forwarded to.
