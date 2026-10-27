# Train on a table

A table is the tabular modality ([ADR 0039](../adr/0039-tabular-modality.md)): one encoder and one
decoder (`tabular_mlp_encoder_v1` and `tabular_mlp_decoder_v1`) that read and rebuild a row. The model
knows a width, not columns. Everything about your columns happens in your `loader_factory`, as for any
other data (spec §6.2).

## What a batch is

A batch is a dict with one key, `tabular`, holding a float tensor of shape `(batch, num_features)`:

- the numeric columns first, **standardized** with one mean and one standard deviation per column,
  computed on the training split;
- then one **one-hot block** per categorical column, in a fixed order;
- with **no missing value**: a `NaN` makes every output of its row `NaN`, so impute before batching
  (a column mean, which is `0` after standardizing, is the simplest choice, and a mask column is one
  more numeric column if the fact that a value was missing carries information).

`num_features` is the number of numeric columns plus the number of categories of every one-hot block.

## A `loader_factory`

`configs/data/tabular.yaml` describes the contract and leaves `loader_factory` and `train_path` for you
to fill in. This is one that follows it. `loadMyTable` is yours: it returns the numeric columns as a
float tensor (with `NaN` where a value is missing) and the categorical columns as integer codes.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from global_vae.config.data import DataConfig, DataloaderBundle
from global_vae.data.transforms.standardize import StandardizeTransform

CATEGORICAL_SIZES = [3, 5]  # the number of categories of each categorical column


def buildRows(
    numeric: torch.Tensor, codes: torch.Tensor, transform: StandardizeTransform
) -> torch.Tensor:
    """Standardize the numeric columns, impute them, and append the one-hot blocks."""
    standardized = torch.nan_to_num(transform(numeric), nan=0.0)  # a missing value is the mean
    blocks = [
        torch.nn.functional.one_hot(codes[:, column], size).float()
        for column, size in enumerate(CATEGORICAL_SIZES)
    ]
    return torch.cat([standardized, *blocks], dim=1)


def buildTabularDataloaders(config: DataConfig) -> DataloaderBundle:
    numeric, codes = loadMyTable(config.train_path)  # (N, F) with NaN, (N, C) integer codes
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(numeric), generator=generator)
    num_val = int(len(numeric) * config.val_split)
    num_test = int(len(numeric) * config.test_split)
    val = order[:num_val]
    test = order[num_val : num_val + num_test]
    train = order[num_val + num_test :]

    # The statistics come from the training rows only, one per numeric column.
    mean = torch.nanmean(numeric[train], dim=0)
    std = torch.sqrt(torch.nanmean((numeric[train] - mean) ** 2, dim=0)).clamp_min(1e-6)
    transform = StandardizeTransform(mean=mean, std=std)

    def loader(rows: torch.Tensor, shuffle: bool) -> DataLoader:
        data = TensorDataset(buildRows(numeric[rows], codes[rows], transform))
        return DataLoader(
            data,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=lambda items: {"tabular": torch.stack([item[0] for item in items])},
        )

    return DataloaderBundle(
        train=loader(train, config.shuffle_train), val=loader(val, False), test=loader(test, False)
    )
```

The per-column `mean` and `std` are tensors on purpose: `StandardizeTransform` takes a float or a
tensor, and a list written in the YAML file fails (the finding of the ADR). Keep the transform around
if you want to read reconstructions in the original units: `transform.inverse` undoes it for the
numeric columns.

## The model

```yaml
modalities:
  tabular:
    encoder:
      name: tabular_mlp_encoder_v1
      kwargs:
        num_features: 15  # 7 numeric columns, then one-hot blocks of 3 and 5 categories
        hidden_dims: [128, 64]
    decoder:
      name: tabular_mlp_decoder_v1
      kwargs:
        num_features: 15
        hidden_dims: [64, 128]
        categorical_sizes: [3, 5]
```

`configs/model/tabular_single_latent.yaml` is the complete file. Three things to set:

- **`num_features`**, in the encoder **and** the decoder, to the width of your rows. It is never
  filled in for you, and a row of another width is refused, not resized. From the command line:
  `model.modalities.tabular.encoder.kwargs.num_features=15` and the same for the decoder.
- **`categorical_sizes`**, in the decoder only, if the table has one-hot columns: the number of
  categories of each block, in order. They are the last `sum(categorical_sizes)` columns. A softmax is
  applied inside each block, so a block is a probability vector, and the default `mse` loss is the
  Brier score of the column. Leave it out for an all-numeric table.
- **The network**: `hidden_dims`, `activation` (`relu`, `gelu`, `silu`, `leaky_relu`, `tanh`),
  `normalization` (`layer`, `batch` or none) and `dropout`. An empty `hidden_dims` is a linear VAE.
  `batch` normalization fails in training on a batch of one row.

## Train

```bash
PYTHONPATH=. python scripts/train.py --config-name experiment/tabular_vae \
    data.train_path=data/raw/my_table.csv \
    data.loader_factory=my_project.data:buildTabularDataloaders \
    model.modalities.tabular.encoder.kwargs.num_features=15 \
    model.modalities.tabular.decoder.kwargs.num_features=15
```

`experiment/tabular_vae` trains on the `sum` reconstruction loss with weight 0.5, since the default
`mean` collapsed the model on a synthetic table (no latent unit active; see
[choose beta](choose-beta.md#choose-a-reduction)). The numbers behind that are in the ADR and are not
tuned to your data: [choose beta](choose-beta.md) has the sweep to run once you have a real table.

## What it does not do

- **No per-column loss.** The reconstruction loss is one function per decoder, so categorical
  columns cannot take a cross-entropy next to a squared error on the numbers. The softmax with `mse`
  stands in for it.
- **No variance per column.** The decoder returns a mean, so every column is reconstructed with the
  same fixed noise level. Columns with very different scales are the reason to standardize.
- **No handling of missing values**, as above.
