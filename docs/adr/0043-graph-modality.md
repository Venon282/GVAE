# 0043: The graph modality

**Status:** accepted (roadmap item P3-8, fifth modality after tabular, audio, text and time series; it adds one
decoder class and one scoring module through the registries, reuses an existing encoder, and changes no core
module)
**Date:** 2026-10-30

## Context

Roadmap P3-8 lists graphs among the further modalities, each through the new-modality checklist of spec §10
and only once a dataset is named. No graph collection has been named, so nothing here is tuned to one: the
batch contract is stated, and the measurements below are on synthetic graphs. The tabular
([ADR 0039](0039-tabular-modality.md)), audio ([ADR 0040](0040-audio-modality.md)), text
([ADR 0041](0041-text-modality.md)) and time series ([ADR 0042](0042-time-series-modality.md)) modalities
followed the checklist before it.

What a graph is to this framework decides everything else, so it came first. The framework's unit is one
tensor per modality in a batch, an encoder that maps it to `(mu, logvar)` and a decoder that maps a latent
vector back to a tensor of the same shape, compared position by position by the loss
(`computeTotalReconstructionLoss`). A graph of at most `N` nodes fits that as a **dense adjacency matrix**
`(batch, N, N)`, padded to the largest graph. That is also an image to the 2D convolutions, and the 2D CNN
encoder and decoder are registered. So the question was not "which classes does a graph need" but "what do
the existing ones already do, and what is left over".

Three things are left over, and they were measured before anything was written:

- **Padding.** Graphs have different numbers of nodes. A padded node needs a value, and the registered
  encoder must accept it.
- **Node order.** An adjacency matrix depends on how the nodes are numbered, and a loss that compares two
  matrices position by position (which is the only loss the framework has) does not match nodes. Whether a
  permutation-invariant encoder would remove the problem was the main thing to find out.
- **The loss.** The raw output of `2d_cnn_decoder_v1` is a real number per pixel, so it only fits the `mse`
  of a YAML file. An edge is a yes or no, and the Bernoulli likelihood of it is the natural reconstruction
  term.

## Decision

**1. A graph is a dense, zero-padded adjacency matrix.** The batch key `graph` holds a float tensor
`(batch, N, N)`, `1` for an edge and `0` for none, with `N` the padded number of nodes for the whole dataset
(`sequence_length.graph` in the data file, `output_shape` in the decoder). With several kinds of edge it is
`(batch, C, N, N)`, one matrix per kind, and `C` is set in the encoder (`in_channels`) and in the decoder
(`out_channels`), as the channels of an image are. A padded node is an isolated node: its row and its column
are zeros. Reading files, building the matrices, padding, ordering the nodes and carrying the number of nodes
are the `loader_factory`'s (spec §6.2), and `docs/how-to/train-on-graphs.md` has one.

**2. Padding is zeros, not `NaN`.** The time series modality ([ADR 0042](0042-time-series-modality.md)) marks
a value that was not observed with a `NaN` because a purpose-built encoder reads it. Here the registered 2D
CNN encoder is reused, and **it turns a `NaN` anywhere in a graph into a `NaN` latent vector for that
graph** (a test pins it), so a `NaN` padding is unusable with it. Zeros are a convention that costs
little: against a loss that reads only the pairs of real nodes, a loss that reads the padding as a target of
`0` lost about 0.005 of AUC (0.931 against 0.936, seed 0, below). Zero padding keeps every batch finite and
every loss name of a YAML file usable, and the number of nodes of each graph can still travel in the batch
under another key (`num_nodes`), which the trainer, `evaluate` and the figures ignore, because they restrict a
batch to the encoder's keys before calling the model (ADR 0019). A direct call of the model on such a batch
raises a `KeyError`, and `model.selectEncoderInputs(batch)` is the restriction.

**3. The encoder is `2d_cnn_encoder_v1` as it is, and the decoder is a thin subclass.** The checklist of
spec §10 asks for an encoder and a decoder. Here the encoder already existed and measured better than every
graph-specific encoder tried (below), so no encoder class is added. The decoder is added, and **its
justification is the loss and not the accuracy**: the reconstruction it gives is that of its parent within the
noise of the measurements (0.903 against 0.903 at weight 1, seed 0, below).

| Registry name | Class | Reads or writes |
|---|---|---|
| `2d_cnn_encoder_v1` (existing) | `TwoDCnnEncoder` | an adjacency matrix `(batch, N, N)` or `(batch, C, N, N)` |
| `graph_cnn_decoder_v1` | `GraphCnnDecoder` | the probability of each edge, same shape, in `[0, 1]` |

`GraphCnnDecoder` subclasses `TwoDCnnDecoder` and changes only the end of `forward`: the output is averaged
with its transpose (`symmetric`, on by default, so the graph is undirected), passed through a sigmoid, and
multiplied by `1 - I` so the diagonal is exactly zero (`zero_diagonal`, on by default, so a node has no edge
to itself). Every other argument is the parent's and is passed on, and the output shape must be square. The
class is registered by the import of `decoders/__init__.py`. `models/`, `latent/`, `losses/`, `training/`,
`evaluation/`, `config/` and `data/` are untouched, and a test reads their source to check that none of them
names the new class or the `"graph"` batch key.

**4. The loss is `bce` with the `sum` reduction, and that is why the class exists.** A `bce` of a YAML file is
`F.binary_cross_entropy` on probabilities, and it refuses a value outside `[0, 1]`, which the raw output of
`2d_cnn_decoder_v1` is (a test pins that it is refused). With probabilities, `reconstruction_loss: bce` and
`reconstruction_reduction: sum` give the Bernoulli likelihood of the edges as a per-graph sum, which is the
loss `scripts/train.py` can run. It measured better than the only loss the raw decoder can use (`mse`) by
about 0.01 of AUC, on complete graphs and on padded ones (below). The shipped experiment sets weight `4.0`
on it.

**5. Node order is the loader's, and the model is order-aware.** The decoder writes the matrix position by
position and the loss compares it position by position, so the same graph in two node orders is two different
inputs and two different targets. This is the main finding of the measurements, and it cannot be fixed
inside a model: a permutation-invariant encoder does not help while the loss is not (below). A graph with no
natural order has to be put in a canonical one by the loader before it is padded, and the how-to shows one.
A test pins the order dependence of the encoder: the same graph with its nodes permuted gives another latent
vector.

**6. A scoring helper for the edges.** `utils/graph_metrics.py` holds `computeEdgeScores(probabilities,
adjacency, num_nodes=None, directed=False, threshold=0.5)`, which returns the binary cross-entropy, the area
under the ROC curve (tie-aware, `computeAuc`), the F1 at a threshold and the density over **the pairs of real
nodes only** (`i < j` for an undirected graph, `i != j` for a directed one, and only indices below
`num_nodes`). The metrics of `evaluate` are entry-wise over the whole matrix, so they are not these (see
Consequences).

**7. One model file, one data file and one experiment.** `configs/model/graph_cnn_single_latent.yaml` (graphs
of 16 nodes, a latent of 16, no fusion), `configs/data/graph.yaml` (an empty transform list: an adjacency
matrix is not a quantity to standardize) and `configs/experiment/graph_vae.yaml`, which trains through
`scripts/train.py` like the others. Directed graphs, self-loops, several kinds of edge and another padded size
are YAML values (`symmetric`, `zero_diagonal`, `in_channels` and `out_channels`, `output_shape` with
`seed_shape`) and are tested through the shipped file.

## Measurements

Synthetic graphs (`tests/integration/_graph_corpus.py`), not a collection: four families (a stochastic block
model of two blocks, a ring lattice with a few rewired pairs, a random recursive tree and an Erdos-Renyi
graph), 8 to 16 nodes, padded to 16, in the order the generator writes them. That order is canonical by
construction (the first block comes first, a ring runs in ring order, a tree is in attachment order), so it is
the best case for a model that is order-aware, and a real collection has no such order for free. 4,000
training graphs and 600 held-out ones, a latent of 16, AdamW at 2e-3, batches of 64, 40 epochs, a constant
`beta` of 1 with no warm-up, and the loss `bce` summed over the pairs of a graph with weight `w` on it. Every
score is on the pairs of real nodes of the held-out graphs, from the mean of the posterior. **AUC** is the
area under the ROC curve of the edge probabilities against the true edges (0.5 is chance). Seed 0 unless
stated; the loss reads only real pairs unless the row says it reads the padding as a target.

Two sets of runs are reported. The comparisons between designs, losses and node orders were run in a
throwaway script, in which the padding was marked with `NaN` so that the scoring and the masked loss could find
the real pairs and the registered encoder was given the same matrix with zeros in their place. Its decoder is the
registered 2D CNN decoder used as it is (not symmetrized while training, scored on its raw output; symmetrizing
it afterwards adds at most 0.003 of AUC), except in the first row of the architecture table, which is symmetrized
as the shipped class is. The **shipped model, trained through the shipped experiment** (`experiment/graph_vae`,
whose default optimizer is Adam at 1e-3 with a linear warm-up of 1,000 steps, and whose loss reads the padding as
a target of `0`), was run on the same graphs in the weight table, and it reproduces the script's unmasked run
(0.931 against 0.931 at weight 4, seed 0).

The designs other than the registered pair were written in the same script and are **not in the
repository**: an order-aware transformer encoder with one token per node (its adjacency row, its degree and
whether it is real, learned positions, padding masked); an order-invariant encoder, three rounds of message
passing over the neighbours with a pooled readout and the node count; a transformer decoder with one token per
node and a bilinear edge score; and a three-layer MLP decoder that writes all `N * N` logits. The registered
pair is the 2D CNN encoder and the 2D CNN decoder, with its output symmetrized.

**The architecture** (weight 1):

| Encoder | Decoder | AUC | BCE |
|---|---|---|---|
| **2D CNN (registered)** | **2D CNN (registered)** | **0.903** | **0.344** |
| order-aware transformer | 2D CNN | 0.893 | 0.358 |
| 2D CNN | MLP | 0.893 | 0.359 |
| order-aware transformer | MLP | 0.887 | 0.365 |
| 2D CNN | node-token transformer | 0.886 | 0.374 |
| order-aware transformer | node-token transformer | 0.880 | 0.379 |
| order-invariant message passing | 2D CNN | 0.840 | 0.427 |
| order-invariant message passing | MLP | 0.841 | 0.425 |
| order-invariant message passing | node-token transformer | 0.838 | 0.432 |

Over three seeds the registered pair is 0.903, 0.899 and 0.902 (BCE 0.344 to 0.350), the order-aware
transformer with the node-token decoder 0.880, 0.875 and 0.873, and the order-invariant encoder with the
node-token decoder 0.838, 0.830 and 0.830, so the ranking is not noise. At weight 4 (seed 0) it is the same:
0.936 for the registered pair, 0.892 for the order-aware transformer with the node-token decoder, 0.844 for
the order-invariant one. The registered pair was also among the faster (65 s per run, against 98 to 106 s
for the transformers).

**The weight**, for the shipped model through the shipped experiment (the loss reads the padding as a target of
`0`; the constant guess scores an AUC of 0.70 and a BCE of 0.56):

| Weight | AUC | BCE |
|---|---|---|
| 1 | 0.916 (two seeds: 0.917, 0.915) | 0.325 (0.323, 0.326) |
| **4 (the experiment)** | **0.929** (three seeds: 0.931, 0.928, 0.927) | **0.302** (0.298, 0.301, 0.308) |
| 16 | 0.929 (one seed) | 0.306 |

In the script, with the padding skipped and the decoder not symmetrized, the registered pair gave 0.903 (three
seeds: 0.903, 0.899, 0.902) at weight 1, 0.935 (0.936, 0.935, 0.934) at 4 and 0.936 at 16, so both routes put
the gain between weights 1 and 4 and none above 4.

**The loss, with what a YAML file can say.** The raw `2d_cnn_decoder_v1` can only use `mse`, and a YAML loss
cannot be told to skip padding. So the fair comparison for `bce` on the registered decoder's output is a
`mse` at its best weight, with the padding read as a target of `0` in both:

| Graphs | Loss | Weight | AUC |
|---|---|---|---|
| complete (16 nodes) | `mse` | 1 / 8 / 32 | 0.877 / 0.907 / 0.908 |
| complete (16 nodes) | **`bce`** | 4 | **0.918** |
| padded, padding read as a target | `mse` | 16 | 0.918 |
| padded, padding read as a target | **`bce`** | 4 | **0.931** |
| padded, padding skipped | **`bce`** | 4 | **0.936** |

`bce` is ahead of the best `mse` by about 0.010 on complete graphs and 0.013 on padded ones, and skipping the
padding is worth a further 0.005, which a YAML file cannot ask for. One seed each for `mse` and for the
unmasked rows.

**Node order** (registered pair, weight 4, one seed; the constant guess is the mean matrix of the training
graphs):

| Order of the nodes | AUC | BCE | Constant guess BCE |
|---|---|---|---|
| as generated (canonical) | **0.936** | **0.285** | 0.558 |
| shuffled, then sorted by degree | 0.887 | 0.369 | 0.577 |
| shuffled | 0.836 | 0.438 | 0.613 |

At weight 1 the same shuffle takes the registered pair from 0.903 to 0.743, and it takes the order-invariant
encoder with the node-token decoder from 0.838 to 0.692: the invariant encoder does not escape it, because
the decoder and the loss still write and compare matrices position by position. Sorting by degree recovers
about half of the loss (0.82 at weight 1 for the registered pair), and not for every family: after a degree
sort the ring graphs fall from 0.996 to 0.821 (weight 4) while the trees and the random graphs improve. The
nodes of a ring all have nearly the same degree, so the sort leaves them close to the shuffled order and
scrambles their band; that this is the cause was not isolated. **Node order moved the AUC by 0.10 at weight 4
and 0.16 at weight 1. The whole spread between the registered pair and the weakest design was 0.09 and 0.06,
and between the pair and the best alternative 0.04 and 0.01, so the order is at least as large as every choice
made inside the model.**

**Through the shipped experiment**, with the settings of the tests (1,024 training graphs, 30 epochs, a
learning rate of 3e-3, a warm-up of 20 steps), three seeds: graphs in generation order reach an AUC of 0.897,
0.898 and 0.901 (BCE 0.36, F1 0.73) against 0.700 for the constant guess (BCE 0.556), and the same graphs
shuffled reach 0.731, 0.743 and 0.744 (BCE 0.53). The tests assert a margin of 0.15 of AUC and 0.1 of BCE over
the constant guess, and 0.1 of AUC for ordered over shuffled.

What it says:

- **The existing pair is the answer, and the new class is for the loss.** Every graph-specific design measured
  worse than a 2D CNN encoder and decoder at weight 1 and at weight 4. Telling the encoder which nodes are real
  did not help either (the order-aware transformer, which knew, was 0.01 behind the CNN, which saw zeros).
  That is why no encoder class is added and why the decoder class is justified by the loss alone. Why the
  convolutions suit these matrices better than a transformer over the nodes was not isolated; the four
  families have local structure along the diagonal (a ring is a band, a block model is two squares) that a
  local kernel reads directly.
- **Order matters as much as the whole choice of architecture, and no architecture removes it.** See above. A
  canonical order is the loader's job, and the cost of a bad one is at least as large as any choice made inside
  the model. A loss that matches nodes (a permutation search or a differentiable matching) is the real answer
  and is a change to `losses/`, flagged and not made.
- **The weight is the lever, and D-6 again.** A weight `w` on the summed reconstruction is a `beta` of `1 / w`
  on the KL. Through the shipped experiment the AUC is 0.916 at 1 and 0.929 at 4 and at 16, so from 4 up it is
  flat. 4 is a middle setting for graphs of this size and density, not a tuned value. The choice of `beta`
  stays roadmap decision D-6.
- **Zero padding is cheap.** 0.005 of AUC against a loss that skips it, for a model that never sees a `NaN`.

These are one synthetic family set, three seeds at most for the architectures and one for the loss and order
tables, and no tuning: they show that the model trains, how the registered classes compare with graph-specific
designs and how much order matters, not what a real collection would give.

## Consequences

- **The registry design held a sixth time, with one class.** One decoder, one scoring module, three config
  files and the tests are all that was added, and no core edit was needed or made. The checklist step
  "subclass Encoder" was met by an existing encoder because it measured better, and the step "subclass
  Decoder" by a thin subclass of an existing decoder that earns its place through the loss and not the
  accuracy. That is a deviation from the letter of §10 and not from its point (the core needed no change).
- **Finding, not fixed: node order, and no matching loss.** A loss that compares matrices position by position
  makes the order part of the data, and no encoder removes it. A permutation-matching or invariant
  reconstruction loss is a core change to `losses/`, flagged and not made. Until it exists the loader orders
  the nodes, and a collection with no natural order lost between 0.05 (sorted by degree) and 0.16 (shuffled)
  of AUC in these measurements.
- **Finding, not fixed: the loss table has no logits form.** The raw `2d_cnn_decoder_v1` can use only `mse`
  from a YAML file, since `bce` refuses its output (a test pins it). A `bce` that takes logits in the table of
  `config/training.py` would give the same loss reach to any decoder without a sigmoid of its own. It is a
  core change, flagged and not made, and `graph_cnn_decoder_v1` stands in for it for this modality.
- **Finding, not fixed: the metrics of `evaluate` are entry-wise.** `evaluate` and `scripts/evaluate.py` run
  on a graph model (a test covers them and the figures), but their `mse`, `r2` and `pearson_r` average over
  every entry of the matrix, padding and the diagonal included, and they are not edge scores. Use
  `computeEdgeScores` for those, with `num_nodes` for padded graphs. Edge-level metrics in `evaluation/` are a
  core change, flagged and not made.
- **Finding, not fixed: no `NaN` padding with this encoder.** The registered encoder turns a `NaN` into a
  `NaN` latent vector, so padding is zeros. A mask-aware graph encoder was not needed in the measurements,
  where telling the encoder which nodes were real did not help.
- **Not modeled: the number of nodes, and node features.** The decoder writes an `N x N` matrix of
  probabilities and no node count: the padded rows are pushed toward zero by the loss and are not exactly
  empty, so a generated graph is read by choosing a number of nodes (a prior over `num_nodes` from the data,
  say) and keeping the pairs below it. Node and edge attributes beyond the kind of edge are not covered; node
  features would be a second modality with its own encoder and decoder, and nothing aligns two modalities
  node by node here.
- **Not modeled: large graphs.** A dense matrix costs `N^2` per graph and per channel. Sixteen nodes is 256
  entries, and a thousand is a million, which this decoder (a fixed-size image from one latent vector) is not
  meant for. A sparse or sequential representation would be a different modality.
- **One latent vector per graph.** The encoder pools to one vector, and the decoder writes the whole matrix
  from it. A latent per node would be the single-tensor decoder item of the roadmap audit P4-0.
- **No dataset, no loader.** Reading files, building the matrices, ordering the nodes and padding are the
  loader's, no named collection exists, and the proof is the synthetic graphs and the dummy loader of the
  tests. The remaining modality of P3-8 (point clouds) stays open.
- **Cost.** The shipped file is 66,816 parameters in the encoder and 36,225 in the decoder (103,041 in all),
  and about 65 s for 40 epochs of 4,000 graphs on one CPU thread.
