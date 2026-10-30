# Train on graphs

A graph is the graph modality ([ADR 0043](../adr/0043-graph-modality.md)): a dense adjacency matrix, read by the
existing `2d_cnn_encoder_v1` and rebuilt by `graph_cnn_decoder_v1`, which gives the probability of each edge.
Reading files, building the matrices, ordering the nodes and padding all happen in your `loader_factory`, as for
any other data (spec §6.2).

## What a batch is

A batch is a dict with one key, `graph`, holding a **float tensor `(batch, N, N)`**: the adjacency matrix of each
graph, `1` for an edge and `0` for none.

- **`N` is the padded number of nodes**, the same for the whole dataset. It is written in the data file
  (`sequence_length.graph`) and in the decoder (`output_shape`, which has to be `(N, N)`), and nothing fills it in
  for you. A graph with fewer nodes is padded with isolated nodes: the rows and the columns of the nodes it does
  not have are zeros. Padding costs little, so choose `N` as the largest graph, or a round number above it.
- **Padding is zeros and not `NaN`.** The 2D CNN encoder turns a `NaN` into a `NaN` latent vector for that graph.
- **Nodes have to come in a meaningful order.** The model writes the matrix position by position and the loss
  compares it position by position, so the same graph with its nodes numbered differently is a different input and
  a different target. This is the one thing that matters more than the architecture (below).
- **Several kinds of edge** are channels, as for an image: a batch `(batch, C, N, N)`, `in_channels: C` in the
  encoder and `out_channels: C` in the decoder.
- **The number of nodes of each graph** can ride in the batch under another key, such as `num_nodes`. The trainer,
  `evaluate` and the figures restrict a batch to the keys the model encodes before calling it (ADR 0019), so they
  ignore it, and `computeEdgeScores` takes it to score the real pairs only. Calling the model yourself on such a
  batch needs the same restriction, `model.selectEncoderInputs(batch)`, or it raises a `KeyError`.

## A `loader_factory`

`configs/data/graph.yaml` describes the contract and leaves `loader_factory` and `train_path` for you to fill in.
This is one that follows it. `loadMyGraphs` is yours: it reads your files and returns a list of float tensors, each
a square 0/1 adjacency matrix of one graph, of any size up to `N`.

```python
import torch
from torch.utils.data import DataLoader

from global_vae.config.data import DataConfig, DataloaderBundle


def orderByDegree(adjacency: torch.Tensor) -> torch.Tensor:
    """Number the nodes by decreasing degree, ties by the summed degree of their neighbours."""
    degree = adjacency.sum(dim=1)
    key = degree * adjacency.shape[0] ** 2 + adjacency @ degree
    order = torch.argsort(key, descending=True, stable=True)
    return adjacency[order][:, order]


def padGraph(adjacency: torch.Tensor, size: int) -> torch.Tensor:
    """Pad with isolated nodes up to `size` nodes."""
    padded = torch.zeros(size, size)
    padded[: adjacency.shape[0], : adjacency.shape[1]] = adjacency
    return padded


def collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def buildGraphDataloaders(config: DataConfig) -> DataloaderBundle:
    graphs = loadMyGraphs(config.train_path)  # a list of (n, n) 0/1 tensors
    size = (config.sequence_length or {})["graph"]
    largest = max(graph.shape[0] for graph in graphs)
    if largest > size:
        raise ValueError(f"A graph has {largest} nodes but sequence_length.graph is {size}.")
    rows = [
        {"graph": padGraph(orderByDegree(graph), size), "num_nodes": torch.tensor(graph.shape[0])}
        for graph in graphs
    ]
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(rows), generator=generator).tolist()
    num_val = int(len(rows) * config.val_split)
    num_test = int(len(rows) * config.test_split)
    split = {
        "train": order[num_val + num_test :],
        "val": order[:num_val],
        "test": order[num_val : num_val + num_test],
    }

    def loader(name: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            [rows[i] for i in split[name]],
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    return DataloaderBundle(
        train=loader("train", config.shuffle_train),
        val=loader("val", False),
        test=loader("test", False),
    )
```

The data file's `transforms` entry for `graph` stays empty: an adjacency matrix of 0 and 1 is not a quantity to
standardize.

**Ordering the nodes is the part to think about.** If your nodes already have a meaning (atoms in the order of a
file, events in time, a hierarchy from the root), keep that order. If they have none, put every graph in a
canonical order before padding it. `orderByDegree` is a simple one, and it is not a canonical form: nodes of equal
degree and equal neighbourhood stay in the order they came in, so two copies of the same graph with different
labels can still come out different. On synthetic graphs, shuffling the nodes and then sorting them by degree
recovered about half of what the shuffle had lost, and it hurt the families whose nodes all have the same degree,
such as rings ([ADR 0043](../adr/0043-graph-modality.md)). A better order for your data (a breadth-first order
from the highest-degree node, a spectral order, a canonical labelling from a graph library) is worth more than
any change to the model.

## The model

`configs/model/graph_cnn_single_latent.yaml` is the complete file. What to set:

- **`output_shape`** in the decoder, to `(N, N)`, and the same `N` as `data.sequence_length.graph`. This decoder
  checks the exact shape and never resizes, so the rest of its configuration has to land on it. In the shipped
  file, a seed of 4 by 4, two doubling stages and a kernel of 3 with padding 1 give 4, 8, 16. For another size,
  ask the class before building the model:

  ```python
  from global_vae.decoders.two_d_cnn_decoder import TwoDCnnDecoder

  TwoDCnnDecoder.computeOutputShape(
      seed_shape=(8, 8), hidden_channels=(64, 32), kernel_sizes=3, paddings=1
  )  # (32, 32)
  ```

  and set it with `data.sequence_length.graph=32 model.modalities.graph.decoder.kwargs.output_shape=[32,32]
  model.modalities.graph.decoder.kwargs.seed_shape=[8,8]`.
- **`symmetric`** (default `true`) gives an undirected graph, so that the edge from `i` to `j` and the one from `j`
  to `i` are the same probability. `false` keeps a directed graph.
- **`zero_diagonal`** (default `true`) gives a node no edge to itself. `false` allows self-loops.
- **`in_channels`** in the encoder and **`out_channels`** in the decoder, for several kinds of edge (above).
- **`hidden_channels`** of the encoder and of the decoder, and the latent `dim`, are the size of the model. The
  shipped file has 66,816 parameters in the encoder and 36,225 in the decoder.

A dense matrix costs `N^2` per graph and per channel. Sixteen nodes is 256 entries, a thousand is a million, and
this decoder (an image from one latent vector) is not meant for graphs of that size.

## Train

The experiment trains through `scripts/train.py` like the others:

```bash
PYTHONPATH=. python scripts/train.py --config-name experiment/graph_vae \
    data.train_path=data/raw/molecules \
    data.loader_factory=my_project.data:buildGraphDataloaders
```

The same loader serves a Python script that builds the model and the trainer from the experiment:

```python
from global_vae.config.experiment import loadExperimentConfig
from global_vae.config.model import buildModelFromConfig
from global_vae.config.training import buildTrainerFromConfig

config = loadExperimentConfig(
    "experiment/graph_vae",
    overrides=[
        "data.train_path=data/raw/molecules",
        "data.loader_factory=my_project.data:buildGraphDataloaders",
    ],
)
model = buildModelFromConfig(config.model)
bundle = buildGraphDataloaders(config.data)
trainer = buildTrainerFromConfig(model, config.training)
trainer.fit(bundle.train, num_epochs=60, val_dataloader=bundle.val)
```

## The loss

The experiment sets `reconstruction_loss: bce`, `reconstruction_reduction: sum` and `reconstruction_weight: 4.0`.
The decoder returns the probability of each edge, so `bce` against the 0/1 matrix is the Bernoulli likelihood of
the graph, and `sum` makes it one number per graph like the KL term. This is also why the decoder is its own
class: the raw output of `2d_cnn_decoder_v1` is not a probability, `bce` refuses it, and its `mse` measured about
0.01 of AUC behind.

**The weight is the lever.** A weight of `w` on the reconstruction is a `beta` of `1 / w` on the KL term. On
synthetic graphs of 8 to 16 nodes ([ADR 0043](../adr/0043-graph-modality.md)), the area under the ROC curve of the
edges was 0.916 at weight 1 and 0.929 at 4, and the same at 16, so 4 is a middle setting and not a tuned one (the
choice of `beta` is [roadmap](../roadmap.md) decision D-6). Try a few weights on your own data, and watch the
number of active latent units that `scripts/evaluate.py` reports.

The loss reads the padding as a target of `0`, because a YAML loss cannot skip it. That cost about 0.005 of AUC
against a loss that reads only the real pairs, and it keeps everything finite.

## Reading a reconstruction

`computeEdgeScores` scores the edges over the pairs of real nodes only, which is what a graph is scored on. The
metrics of `evaluate` are entry-wise over the whole matrix, padding and diagonal included, so use this for the
edges:

```python
from global_vae.utils.graph_metrics import computeEdgeScores

model.eval()
with torch.no_grad():
    batch = next(iter(bundle.val))
    inputs = model.selectEncoderInputs(batch)  # drops `num_nodes`, which no encoder reads
    reconstruction = model(inputs, use_mean=True)["reconstructions"]["graph"]

scores = computeEdgeScores(reconstruction, batch["graph"], batch["num_nodes"])
# {"bce": ..., "auc": ..., "f1": ..., "density": ..., "num_pairs": ...}
```

`auc` is the chance that a real edge scores above a pair with no edge (0.5 is chance), `f1` is at `threshold=0.5`,
and `density` is the share of the pairs that are edges, so an `auc` or an `f1` should be read against it. For a
directed graph pass `directed=True`.

To draw graphs from the model, decode latent vectors from the prior, choose how many nodes each graph has, and
sample the edges among those nodes:

```python
with torch.no_grad():
    z = torch.randn(8, model.latent_spaces["z_fused"].dim)
    probabilities = model.decoders["graph"](z)  # (8, N, N)

num_nodes = torch.randint(8, 17, (8,))  # or draw them from the sizes of your own graphs
real = torch.arange(probabilities.shape[-1])[None, :] < num_nodes[:, None]
upper = torch.triu(torch.bernoulli(probabilities), diagonal=1)  # one draw per pair of nodes
edges = (upper + upper.transpose(1, 2)) * (real[:, :, None] & real[:, None, :])
```

The decoder does not generate a number of nodes: it writes an `N` by `N` matrix, so the size of each graph is
yours to choose, and the pairs that involve a padded node are dropped. The probabilities are symmetric, so the
sketch draws each pair once and mirrors it, which gives an undirected graph. For a directed graph
(`symmetric: false`) draw every entry with `torch.bernoulli(probabilities)` and keep the diagonal as it is.

## What it does not do

- **A node order of its own.** The model is order-aware and has no loss that matches nodes, so the order is the
  loader's. This is the main thing the measurements found, and a matching loss is a change to the core
  (ADR 0043).
- **Node features.** The batch holds the structure only. Attributes of nodes would be a second modality with its
  own encoder and decoder, and nothing here aligns two modalities node by node.
- **Large graphs.** A dense `N` by `N` matrix and a fixed-size decoder do not scale past a few hundred nodes.
- **One latent vector per graph.** The encoder pools to one vector and the decoder writes the whole matrix from
  it.
- **A named dataset.** No graph collection has confirmed this modality is needed, so the proof is on synthetic
  graphs.
