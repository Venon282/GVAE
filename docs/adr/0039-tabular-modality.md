# 0039: The tabular modality

**Status:** accepted (roadmap item P3-8, first modality; it adds two classes through the registries
and changes no core module)
**Date:** 2026-10-26

## Context

Roadmap P3-8 lists further modalities (audio, tabular, text, time series, graphs, point clouds),
each through the new-modality checklist of spec §10 and only once a dataset is named. It adds that
tabular data "is the cheapest way to prove the genericity claim a second time": the claim of spec §1
and §6 that adding a modality never touches the core. The signal and the image were the first two
modalities, written together with the core, so neither of them tested it. A table is also the first
modality that is not a grid: it has no length to resample and no neighbourhood for a convolution.

No table has been named, so nothing here is tuned to one. The model is generic, the batch contract
is stated, and the measurements below are on a synthetic table.

Four questions had to be answered, because the batch convention of the model is one tensor per
modality and the reconstruction loss is one function `(reconstruction, target) -> scalar` per
decoder ([ADR 0030](0030-per-modality-reconstruction-loss-and-weight.md)):

- what the single tensor of a table is, since a table mixes numeric and categorical columns;
- whether the encoder and the decoder should know the columns;
- what to do with missing values;
- what loss reconstructs a row.

## Decision

**1. Two registered classes and one module of building blocks.** Nothing else changes:

| Registry name | Class | Reads or writes |
|---|---|---|
| `tabular_mlp_encoder_v1` | `TabularMlpEncoder` | a batch of rows, `(batch, num_features)` |
| `tabular_mlp_decoder_v1` | `TabularMlpDecoder` | a batch of rows of a fixed `num_features` |

`utils/mlp_blocks.py` holds what they share (`buildMlp`, the activation names). The classes are
registered by the imports of `encoders/__init__.py` and `decoders/__init__.py`. `models/`,
`latent/`, `losses/`, `training/`, `evaluation/`, `config/` and `data/` are untouched, and a test
reads their source to check that none of them names the new classes or the `"tabular"` batch key.
The modality name is `tabular` by default, and a model that has two tables passes another
`modality_name` and uses another batch key.

**2. A row is one flat float vector, built by the caller.** The batch key `tabular` holds a
`(batch, num_features)` float tensor. The columns are the numeric ones first, standardized, then
one one-hot block per categorical column. The encoder and the decoder know a width, not columns:
turning a table into that tensor is the `loader_factory`, the permanent boundary of spec §6.2, and
the generic `standardize` transform does the numeric part. This is the one place where a table
differs from a series or an image, and it is a data question: the two classes stay the same for any
table, and a different encoding of the categorical columns (target encoding, an embedding of an
index) is another loader, not another model.

**3. The decoder can be told where the one-hot blocks are.** `categorical_sizes`, the number of
categories of each block (empty by default, which is an all-numeric table), says that the last
`sum(categorical_sizes)` outputs are one-hot blocks, in that order. A softmax is applied inside
each block, so a block is a probability vector, and the numeric outputs stay linear. The size of
each block must be at least 2 and the blocks cannot be wider than the row, and both are checked
when the decoder is built. The encoder needs no such argument, since a one-hot block is just more
input values. `num_features` is data-dependent, so the config layer never fills it in, like a
decoder's `output_length`, and the decoder verifies it and never resizes.

**4. The loss stays `mse`, which is a proper score on both kinds of column.** The mean squared error
between the softmax probabilities and the one-hot target is the Brier score of the column, a proper
scoring rule, and on the numeric columns it is the squared error. So the default loss of
[ADR 0029](0029-loss-scale-convention.md), `mse` with the `sum` reduction at weight 0.5, is the
ELBO of a unit-variance Gaussian decoder on the numeric columns, and a consistent score on the
categorical ones, with nothing added to `losses/`. A per-column loss (a cross-entropy on the blocks
next to a squared error on the numbers) is not available, see the consequences.

**5. Missing values are the caller's.** A `NaN` in a row makes every output of that row `NaN`, which
a test pins. A table has no common recipe (a column mean, a mask column, a model of its own), so
the `loader_factory` imputes, and a mask column is one more numeric column of the row. The
modality dropout of spec §5 drops a whole modality, not a cell, so it does not help here.

**6. The network is a plain MLP, configured by name.** A stack of `Linear` layers, each followed by
an optional normalization (`layer` or `batch`), an activation (`relu`, `gelu`, `silu`,
`leaky_relu`, `tanh`) and an optional dropout, then two linear layers for `(mu, logvar)` in the
encoder and one for the row in the decoder. The widths (`hidden_dims`), the activation, the
normalization and the dropout are arguments with defaults (`[128, 64]` for the encoder and
`[64, 128]` for the decoder, `relu`, none, none), and the activation is a string, unlike the
`Callable` factory of the CNN encoders, so that a YAML file can write it. An empty `hidden_dims`
is a linear VAE. `batch` normalization fails in training on a batch of one row, which the
docstring says. `minimal_input_length` is `num_features`, which is also the only width the
encoder accepts: a row of another width is refused with an error naming the expected shape, as
the decoders here never resize.

**7. One model file, one data file and one experiment.** `configs/model/tabular_single_latent.yaml`
(16 features, a latent of 8, no fusion), `configs/data/tabular.yaml` (the contract, with no
`sequence_length`, which is the length of a 1D signal) and `configs/experiment/tabular_vae.yaml`,
which trains through `scripts/train.py` like the signal and image experiments. The experiment
sets `reconstruction_reduction: sum` and `reconstruction_weight: 0.5`, with the numbers below.

## Measurements

The loss scale of the experiment, on a synthetic table of 1,000 training rows (4 hidden factors
through a `tanh` map plus noise, standardized, so there is a latent structure to find) with a
latent of 8, the warm-up of `training/default.yaml`, one seed:

| Table | Reduction and weight | Epochs | R² on the numeric columns | Active latent units | Category accuracy |
|---|---|---|---|---|---|
| 16 numeric columns | `mean`, 1.0 | 60 | 0.01 | 0 of 8 | |
| 16 numeric columns | `sum`, 0.5 | 60 | 0.23 | 4 of 8 | |
| 16 numeric columns | `sum`, 0.5 | 300 | 0.21 | 4 of 8 | |
| 10 numeric, two one-hot of 3 | `mean`, 1.0 | 60 | 0.00 | 0 of 8 | 0.41 |
| 10 numeric, two one-hot of 3 | `sum`, 0.5 | 60 | 0.46 | 4 of 8 | 0.51 |
| 10 numeric, two one-hot of 3 | `sum`, 0.5 | 300 | 0.44 | 4 of 8 | 0.50 |

R² is against the column means of the validation rows, and the accuracy of a block is the share of
rows whose most probable category is the right one (chance is 0.33). The pattern is that of the
signal and image models ([ADR 0029](0029-loss-scale-convention.md), [ADR 0037](0037-transformer-and-vit-backbones.md)):
under `mean` and `beta = 1` the KL weighs `N / 2` = 8 times more than the ELBO does for a row of 16
values, and the model collapses, while `sum` at 0.5 trains. Training for five times as long did not
improve either table, so the number of epochs is not what limits it. The likely limit is the
unit-variance decoder that this weight assumes (a column that is predicted to no better than a
fraction of its standard deviation is not worth a nat of KL), which was not tested further. These
are numbers from synthetic data, one seed, and no tuning. The choice of beta stays
roadmap decision D-6.

## Consequences

- **The registry design held a second time.** Two classes, one utility module, three config files
  and the tests are all that was added, and no core edit was needed or made. This is stronger than
  for P3-5, which still shared a family with the signal and image: a table has no length and no
  grid, and the model, the losses, the trainer, the evaluation and the config layer did not notice.
- **Finding, not fixed: one loss per decoder.** The reconstruction loss is chosen per decoder, so a
  table with categorical columns cannot take a cross-entropy on the blocks and a squared error on
  the numbers. The softmax and the Brier score of point 4 are the closest thing with no core change.
  A per-column loss would be a change to `config/training.py` and `losses/reconstruction.py`
  (a loss that takes a column mask, or one decoder per column group, which a model can already
  express as several modalities of one table at the price of a loader that splits it).
- **Finding, not fixed: no output variance.** The decoder returns one tensor, the mean, so the
  likelihood is a Gaussian of fixed variance, which is the unit-variance convention of ADR 0029. A
  table whose columns have very different noise levels would want a variance per column, which
  needs a decoder that returns a distribution. That is the single-tensor-decoder item of the
  roadmap audit P4-0 and is not made here.
- **Finding, not fixed: YAML cannot write per-column statistics.** `StandardizeTransform` takes a
  float or a tensor, and a list written in YAML reaches it as a Python `list`, which fails with a
  `TypeError` that does not name the cause (`'>' not supported between instances of 'list' and
  'int'`). A table needs one mean and one standard deviation per column, so the `loader_factory`
  builds the transform with tensors, and `configs/data/tabular.yaml` says so. The fix is to accept a
  sequence in `data/transforms/standardize.py`, a change outside the two classes, so it is flagged.
  `tests/integration/test_tabular_model_configs.py` states the intended behavior in a strict `xfail`
  test, which starts to fail when the transform is changed and then asks for the marker to be
  removed.
- **`num_features` is written twice.** The encoder and the decoder each take it, since the config
  layer knows nothing of the data (the same as `output_length` of a signal decoder and the image
  `output_shape`), and the experiment says how to override both from the command line. The test
  that builds the model from the overrides checks that they have to agree.
- **No dataset was used.** The roadmap asks for a named dataset before a modality, and none was
  named, so the proof of the claim is a synthetic table and the dummy loader of the tests. There is
  no reference `loader_factory` for a real table: its imputation, encoding and splitting are the
  first things a named dataset would decide. Until then this roadmap item stays open for the other
  modalities (audio, text, time series, graphs, point clouds).
- **A model of a table next to a signal or an image** uses the pairs of the config as for any other
  modality (`modalities.tabular` beside `modalities.signal`), a fusion strategy, and one reduction
  and weight per decoder ([ADR 0030](0030-per-modality-reconstruction-loss-and-weight.md)). One
  weight cannot balance a table of 16 values and an image of 4,096, which is the problem of P3-3
  and not a new one.
