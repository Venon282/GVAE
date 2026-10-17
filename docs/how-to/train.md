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

## Choose the reconstruction reduction

The reconstruction term is a per-element mean by default, while the KL is a sum over the latent
dimensions, so `beta = 1` is not the usual ELBO and the useful `beta` depends on the data shape
([ADR 0029](../adr/0029-loss-scale-convention.md)). `training.reconstruction_reduction` chooses the
scale, for every modality or per decoder:

```yaml
training:
  reconstruction_reduction: mean              # the default: nothing changes
  # reconstruction_reduction: sum             # the per-sample sum, the scale of the KL
  # reconstruction_reduction: {signal: sum, image: mean}   # a decoder left out keeps mean
```

```bash
python scripts/train.py 'training.reconstruction_reduction={signal:sum,image:mean}'
```

An unsupported name, or a decoder the model does not have, stops the run with an error that starts
with `training.reconstruction_reduction`. The validation loss uses the same reduction as the
training loss. Pass the same one to `scripts/evaluate.py --reconstruction-reduction`. How to pick a
`beta` for it, and a sweep recipe, are in [choose beta](choose-beta.md).

## Set the loss and weight per modality

`training.reconstruction_loss` and `training.reconstruction_weight` are one value for every modality,
or a mapping from a decoder name to a value ([ADR 0030](../adr/0030-per-modality-reconstruction-loss-and-weight.md)). Use a mapping when the modalities
need different treatment: a binary target next to a continuous one, or a modality that should count
for more or for less in the reconstruction term.

```yaml
training:
  reconstruction_loss: mse                          # the default, for every modality
  # reconstruction_loss: {signal: mse, mask: bce}     # a decoder left out keeps mse
  reconstruction_weight: 1.0                        # the default, for every modality
  # reconstruction_weight: {signal: 0.5, mask: 0.2}   # a decoder left out keeps 1.0
```

```bash
python scripts/train.py \
    'training.reconstruction_loss={signal:mse,mask:bce}' \
    'training.reconstruction_weight={signal:1.0,mask:0.2}'
```

- The losses are `mse` (the default), `l1`, `bce` and `smooth_l1`. `bce` needs a reconstruction and a
  target inside `[0, 1]`, so the decoder has to end in a sigmoid or an equivalent; PyTorch raises
  an error for a value outside that range, and the config does not check it.
- A weight is a finite number of at least `0`. `0` takes a modality out of the loss: its decoder
  gets no gradient from the reconstruction term. A negative weight, `true` or a string is refused.
- Each modality's term is its loss function, then the reduction chosen above, then its weight. The
  three mappings are independent and may name different decoders.
- A wrong loss name stops the run with a `KeyError` and any other wrong value with a `ValueError`;
  both start with the field's path, and a mapping names the decoder, for example
  `training.reconstruction_weight['mask']: must be a finite number >= 0, got -1`.
- A decoder name that the model does not have also stops the run. The message lists the model's
  decoders and says what the decoder would have kept, because a mistyped name would otherwise change
  nothing and say nothing. (The unmatched keys of `beta_schedules` only log a warning.)
- The validation loss uses the same settings as the training loss. Changing a weight changes what
  the optimizer trades off, so compare runs with different weights on the reconstruction metrics
  and not on the total loss.

To get an evaluation report whose `total reconstruction loss` is on the scale of the training
curves, give `scripts/evaluate.py` the same three settings. Each flag takes a bare value or
`modality=value` pairs separated by commas:

```bash
python scripts/evaluate.py \
    --checkpoint runs/model.pt \
    --model-factory mypackage.models:build_model \
    --dataloader-factory mypackage.data:build_test_dataloader \
    --reconstruction-loss signal=mse,mask=bce \
    --reconstruction-weight signal=1.0,mask=0.2 \
    --reconstruction-reduction sum
```

The per-modality metrics of the report do not depend on any of them. The report records the
reduction and not the loss and the weights, so keep them with the run.

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
