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
latent spaces model). `experiment/image_vae` and `experiment/image_resnet_vae` are the image-only
counterparts of the two signal experiments (the plain and the residual 2D pair): they read
`configs/data/image.yaml`, so the batches carry an `image` of shape `(batch, 64, 64)`, and the
paired loader below serves them too. They train on the `sum` reconstruction loss with weight 0.5,
since the default `mean` collapsed an image model on synthetic data; see
[choose beta](choose-beta.md#choose-a-reduction).

The transformer backbones (ADR 0037) have their own experiments: `experiment/signal_transformer_vae`
and `experiment/image_vit_vae` (the same single-modality contracts as the CNN ones, with a small
transformer over the series and a ViT over the image), `experiment/signal_image_transformer_vae`
(both, fused by `cross_attention`) and `experiment/signal_image_shared_transformer_vae` (both,
through one shared transformer, no fusion). All four set the `sum` loss with weight 0.5 and read
the data contracts above. How the models are written is in
[configure encoders, decoders and latent spaces](configure-latent-spaces.md).

`experiment/tabular_vae` (ADR 0039) is the single-modality VAE of a table: it reads
`configs/data/tabular.yaml`, so the batches carry a `tabular` tensor of shape `(batch, num_features)`,
and it sets the `sum` loss with weight 0.5 like the image and transformer ones. [Train on a
table](train-on-a-table.md) says how to build the rows and set `num_features`.

`experiment/audio_vae` (ADR 0040) is the single-modality VAE of a raw waveform: it reads
`configs/data/audio.yaml`, so the batches carry an `audio` tensor of shape `(batch, 2048)` within
`[-1, 1]`, and it sets the `sum` loss with weight 0.5 too. The clip length is written in the data file and
in the decoder's `output_length`. [Train on audio](train-on-audio.md) has a loader and says how to change
the length, the channels and the stack.

`experiment/text_vae` (ADR 0041) is the single-modality VAE of a text: it reads `configs/data/text.yaml`, so
the batches carry a `text` tensor, one-hot, of shape `(batch, 8, 32)`: a length of 8 tokens over a vocabulary of
32. It sets the `bce` loss with the `sum` reduction and weight 1.0, since the decoder returns probabilities. The
length is written in the data file and in the decoder's `output_length`, and the vocabulary size in the encoder
and the decoder. [Train on text](train-on-text.md) has a loader and says how to change them.

## Train on paired signal and image data

Pairing a signal with its image is your `loader_factory`'s job like the rest of the data
pipeline (spec §6.2). `examples/04_paired_loader.py` is a reference you can copy and adapt: it
pairs two folders of files by sample ID, leaves out the samples that have only one modality,
splits the pairs by sample ID with `data.seed`, runs `data.transforms` and checks the signal
length against `data.sequence_length`.

```bash
PYTHONPATH=. python scripts/train.py --config-name experiment/signal_image_vae \
    data.loader_factory=examples.04_paired_loader:buildPairedDataloaders \
    data.train_path=/path/to/dataset
```

The dataset folder holds `signal/<sample_id>.npy` and `image/<sample_id>.npy`. To read another
layout or format, change the three adaptation points at the top of the example
(`sampleIdFromPath`, `loadSignal`, `loadImage`). If your signals do not have
`data.sequence_length.signal` points, add a `resample` step to `data.transforms.signal` instead
of changing the model. `python examples/04_paired_loader.py --train` writes a synthetic dataset
in that layout and trains on it, end to end.

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

## Learning-rate schedules

`linear_warmup_lr` and `cosine_lr` are callbacks, chosen by name like `reduce_lr_on_plateau`
([ADR 0033](../adr/0033-top-k-checkpoints-and-learning-rate-schedules.md)):

```yaml
training:
  callbacks:
    cosine_lr:
      total_steps: 2000     # optimizer steps until the rate reaches its floor
      warmup_steps: 100     # linear warm-up from 1/100 of the rate to the full rate; 0 for none
      min_lr_ratio: 0.01    # the floor, as a fraction of the starting rate; 0.0 decays to zero
```

```yaml
training:
  callbacks:
    linear_warmup_lr:
      warmup_steps: 500     # the rate of step s is lr * (s + 1) / 500, then it is left alone
```

- **Steps, not epochs.** The rate is a function of the optimizer step. For `E` epochs of `B`
  batches, with `accumulate_grad_batches: N`, a schedule that should end with the run has
  `total_steps = E * ceil(B / N)`.
- **The starting rate is the optimizer's.** Set it with `training.optimizer.kwargs.lr` as before.
  Each parameter group is scaled from its own starting rate.
- **Combining.** `linear_warmup_lr` stops touching the rate when it ends, so it combines with
  `reduce_lr_on_plateau`. `cosine_lr` owns the rate for the whole run: with another callback that
  changes it, the run logs a warning that those changes are overwritten.
- **Resume.** Nothing to save: the rate of the step a run resumes at is computed from the step.
- **Watching it.** Both are recorded as `train/lr` like any rate change (above).

## Mixed precision, gradient accumulation and `torch.compile`

Each is off by default, and a run that sets none of them is unchanged
([ADR 0031](../adr/0031-mixed-precision-gradient-accumulation-and-compile.md)):

```yaml
training:
  mixed_precision: bf16         # null (default), bf16 or fp16
  accumulate_grad_batches: 4    # batches per optimizer step; 1 (default) is a step per batch
  compile_model: true           # run the forward pass through torch.compile
  compile_kwargs:               # passed to torch.compile
    mode: reduce-overhead
```

- **`mixed_precision`.** The forward pass runs under autocast on the trainer's device. `bf16` needs
  nothing else; `fp16` adds a gradient scaler, whose state is kept in the checkpoint. The losses are
  computed in `float32` from the model's outputs, and the parameters stay `float32`, so a checkpoint
  loads into a run with or without it. Expect the loss to agree with `float32` to a few percent, not
  to the last digit. Validation runs under the same autocast.
- **`accumulate_grad_batches`.** The gradients of `N` batches are added (each loss divided by `N`)
  before one optimizer step, for an effective batch `N` times larger. A **step is then an update**:
  `log_every_n_steps`, the warm-up of a `beta_schedules` entry, `warmup_steps` and `total_steps` of a
  learning-rate schedule and the `step` a callback receives all count updates, so scale them by
  `1 / N` from what you used per batch. If an epoch's batches are not a multiple of `N`, the last,
  smaller group is a step of its own, averaged over the batches it has. Clipping applies to the
  accumulated gradient.
- **`compile_model`.** The model is not replaced, so checkpoints keep their keys and load with or
  without it. The first steps are slow while it compiles, and with `modality_dropout_p > 0` each
  new set of kept modalities compiles again. `{backend: eager}` in `compile_kwargs` checks the
  wiring without a compiler.

## Keep the best K checkpoints

`best_checkpoint` keeps one file, the best epoch, at a fixed path. `top_k_checkpoint` keeps the `k`
best, one file each, and deletes the worst when a better epoch comes
([ADR 0033](../adr/0033-top-k-checkpoints-and-learning-rate-schedules.md)):

```yaml
training:
  callbacks:
    top_k_checkpoint:
      directory: ${output_dir}/top_k
      k: 3                              # default 3
      monitor: val/loss/total           # the default; needs a validation set
      mode: min                         # or max
      filename_pattern: top_k_epoch_{epoch:04d}.pt   # must contain {epoch}
```

Each file is a complete training checkpoint, with the optimizer, so it costs as much disk as the
checkpoint of any other callback, `k` times. On a tie the earlier epoch ranks first, and a `nan` is
never kept. The kept list is saved with the checkpoints, so a resumed run goes on ranking against
the same files.

## Resume an interrupted run

Give the same command line plus `training.resume_from`
([ADR 0032](../adr/0032-resume-from-the-training-script.md)):

```bash
python scripts/train.py training.num_epochs=200 \
    training.resume_from=runs/signal_vae/checkpoints/checkpoint_epoch_0099.pt
```

- **`num_epochs` is the total of the run.** The checkpoint above holds 100 epochs, so 100 more are
  trained. A checkpoint that already holds `num_epochs` epochs or more leaves nothing to do: the
  script says so and exits. (`Trainer.fit(num_epochs)` in Python is still an increment.)
- **What comes back.** The model, the optimizer, the step and epoch counters, the history, the
  Python, NumPy and PyTorch random states, the gradient scaler and the state of the callbacks
  (the patience counters of `early_stopping` and `reduce_lr_on_plateau`, the best value of
  `best_checkpoint`, the files of `top_k_checkpoint` and `checkpoint`). The model must be the same:
  the weights are loaded strictly.
- **Files.** A `csv` logger continues its file without a second header. Rows written after the
  checkpoint you resume from stay, so a step can appear twice, and the later row is the resumed
  one. TensorBoard adds an events file.
- **Which checkpoint.** Resume from a file written by the last callback of the list: a
  checkpoint written while the callbacks run holds the callbacks after it as they were one epoch
  earlier. The shipped configs list `checkpoint` last for that reason.
- **The guarantee.** A resumed deterministic run reproduces the loss curve of the uninterrupted
  run: every metric of every epoch to a relative `1e-6` (absolute `1e-9` near zero), and the final
  weights to `1e-6`. Deterministic means the same config and seed, the same data order, and the CPU
  or `deterministic: true`. A `DataLoader` that shuffles with the global PyTorch random state is
  reproduced, since that state is restored; one with its own `generator`, or one that reads
  workers' seeds from elsewhere, is not, and you save its state yourself. A different PyTorch,
  hardware or thread count is outside the guarantee.

## Track a run with MLflow

`mlflow` is a registered logger next to `csv` and `tensorboard`
([ADR 0035](../adr/0035-mlflow-experiment-logger.md)). It is an optional extra, and nothing in the
shipped configs uses it:

```bash
pip install -e ".[mlflow]"
```

MLflow needs no server. A `tracking_uri` that starts with `sqlite:///` keeps the runs in a local
file, and `artifact_location` keeps the figures and images in a local folder. This puts both under
the run's `output_dir`:

```bash
python scripts/train.py \
    'training.loggers=[{name: mlflow, kwargs: {tracking_uri: "sqlite:///${output_dir}/mlflow.db", artifact_location: "${output_dir}/mlartifacts", run_name: baseline}}]'
```

Overriding `training.loggers` from the command line replaces the whole list, so the `csv` and
`tensorboard` entries of the shipped preset are gone. To keep them, list them again, or write the
list in your own training YAML file:

```yaml
training:
  loggers:
    - name: csv
      kwargs:
        path: ${output_dir}/metrics.csv
    - name: mlflow
      kwargs:
        tracking_uri: sqlite:///${output_dir}/mlflow.db
        artifact_location: ${output_dir}/mlartifacts
        experiment_name: saxs            # default: global_vae
        run_name: baseline               # default: a name MLflow picks
        tags: {dataset: saxs_v2}
        params: {beta: 0.5, latent_dim: 16}   # hyperparameters, recorded once at the start
```

Look at the runs afterwards, still with no server of your own to run:

```bash
mlflow ui --backend-store-uri sqlite:///runs/signal_vae/mlflow.db
```

To use a tracking server instead, set `tracking_uri: http://host:5000` (or leave it out and set the
`MLFLOW_TRACKING_URI` environment variable, which MLflow reads). `artifact_location` then belongs to
the server's own setup, and is only used when this run creates the experiment.

What the logger records, per run:

| What | Where in MLflow | Notes |
|---|---|---|
| Step and epoch metrics (`loss/total`, `train/loss/total`, `val/...`, `train/lr`) | Metrics, with the step or the epoch as the x value | One request per step, not one per metric. |
| Figures | Artifacts `figures/<name>/<tag>_<x>.png` | `<tag>` is `step` or `epoch`, so the two never overwrite each other. |
| Images | The image grid of the run page, one image per step | `(C, H, W)` with `C` 1, 3 or 4, `uint8` or floats in `[0, 1]`. Floats outside that range are clipped. |
| `params`, `tags`, `run_name` | Parameters, tags, name | `params` values are converted to strings. |

`scripts/train.py` logs the scalars. Figures and images are logged by calling the logger, which
you can do after `fit` returns: MLflow does not lock a finished run. The plots of
`global_vae.visualization` return matplotlib figures, which `logFigure` takes as they are:

```python
from global_vae.training.loggers.mlflow_logger import MlflowLogger
from global_vae.training.trainer import Trainer
from global_vae.visualization.latent_plot import collectLatentParams, plotLatentSpace
from global_vae.visualization.loss_curves import plotLossCurves

logger = MlflowLogger(tracking_uri="sqlite:///runs/mlflow.db", artifact_location="runs/artifacts")
trainer = Trainer(model, callbacks=[logger])
trainer.fit(train_dataloader, num_epochs=50, val_dataloader=val_dataloader)

mu, _ = collectLatentParams(model, val_dataloader, "z_fused")
logger.logFigure("latent/pca", plotLatentSpace(mu), x=trainer.global_step)
logger.logFigure("loss_curves", plotLossCurves(trainer.history), x=trainer.global_step)
logger.logImage("reconstruction/image", reconstruction_chw, x=trainer.global_step)  # (C, H, W)
```

To log them during training, write a small callback that calls the logger, and list it after the
logger. Neither `Trainer` nor the script logs figures by itself.

- **Status.** The run ends `FINISHED` when training ends normally, `FAILED` when it raises and
  `KILLED` on a `KeyboardInterrupt`. A second `fit` on the same trainer continues the same run.
- **Resuming.** A run resumed with `training.resume_from` starts a new MLflow run in the same
  experiment; its steps continue from the checkpoint's step. (`csv` continues its file; MLflow has
  no file to continue.)
- **The older `./mlruns` folder store** is not the documented route: MLflow 3 refuses it unless
  `MLFLOW_ALLOW_FILE_STORE=true` is set. `sqlite:///` works with every version of the extra.
- **`import mlflow` is slow**, so the logger imports it only when you build one; a run that logs to
  `csv` and `tensorboard` never loads it.

## Checkpoint files and loading them safely

A checkpoint is a pickle file, and loading one can run code. `loadCheckpoint` and
`Trainer.loadCheckpoint` therefore read it with PyTorch's restricted unpickler
(`weights_only=True`) by default, which a file written by this version allows
([ADR 0034](../adr/0034-safe-checkpoint-loading-by-default.md)). Only load files whose origin you
know either way.

A file written by an earlier version, or one whose `config` holds an object that is not plain data,
is refused with a message that names these three ways out, for a file you trust:

```python
from global_vae.training.checkpoint import loadCheckpoint, upgradeCheckpoint

upgradeCheckpoint("runs/old/best.pt")  # rewrite it once; then it loads as usual
upgradeCheckpoint("runs/old/best.pt", "runs/old/best_safe.pt")  # or keep the original

loadCheckpoint("runs/old/best.pt", model=model, weights_only=False)  # or read it as it is
```

`scripts/evaluate.py` and `scripts/visualize_latent.py` take `--trust-checkpoint` for the last one.
A `config` passed to `saveCheckpoint` should be plain data (a dict of numbers, strings and lists);
a dataclass instance is stored as a dict.

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
