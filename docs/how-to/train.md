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
splitting stay entirely your own responsibility: a permanent boundary, not
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

Experiment files ship for each pattern of the model schema: `experiment/signal_vae`
(one signal, one latent space), `experiment/signal_image_vae` (signal and image, one latent
space), `experiment/signal_image_independent_latents` and `experiment/signal_image_shared_private`
(several latent spaces). The last three read the data contract `configs/data/signal_image.yaml`, so
your `loader_factory` yields `signal` and `image` batches (plus `image_out` for the independent
latent spaces model). How the models are written is in
[configure encoders, decoders and latent spaces](configure-latent-spaces.md).

## Watch the learning rate

A callback that changes the learning rate, such as `reduce_lr_on_plateau`, is recorded without any
extra setting ([ADR 0028](../adr/0028-learning-rate-in-the-metrics.md)):

```yaml
training:
  callbacks:
    reduce_lr_on_plateau:
      monitor: val/loss/total
      factor: 0.1
      patience: 5
```

- **In the logs.** The epoch line gains `train/lr`, the mean of the rates the epoch's steps trained
  with, written in its shortest form (`train/lr=0.001`, then `train/lr=0.0001`). With several
  parameter groups there is one key per group, `train/lr/group0`, `train/lr/group1`.
- **In the loggers and the history.** `train/lr` is an ordinary epoch metric, so it reaches the CSV
  file, TensorBoard, `Trainer.history` and the checkpoint history with no further setting.
- **In a plot.** `scripts/visualize_latent.py` writes `lr_curve.png` next to `loss_curves.png` when
  the checkpoint history holds a rate, and leaves it out otherwise. `--skip-history` skips both.
  From Python, `plotLearningRate(history)` returns the figure.
- **One epoch of delay.** The rate is read before each optimizer step. A reduction made when epoch
  `n` ends shows from epoch `n + 1`.

A run with no such callback records no rate, so its logs and files are unchanged. A callback of
your own that changes the rate must declare it (`acts_on_learning_rate = True`, see
[add a strategy](add-a-strategy.md#a-note-specific-to-training-callbacks)).

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
