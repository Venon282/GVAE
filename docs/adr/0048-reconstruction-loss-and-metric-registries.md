# 0048: Reconstruction loss and metric registries

**Status:** accepted (roadmap item P5-3; it replaces the fixed table of loss names of `config/training.py` by a
registry, adds a registry of metrics, and changes no number that a run of 2.0.0 produced)
**Date:** 2026-11-06

## Context

The reconstruction loss was the last pluggable component that was still a fixed table. The four names of a YAML
file (`mse`, `l1`, `bce`, `smooth_l1`) were a dictionary in `config/training.py` that pointed at
`torch.nn.functional`, so a loss that is not one of the four could only be given to a `Trainer` built by hand.
The specification asks for a registry (§10, §12), and the limit showed on four of the six data kinds (findings
I-2, I-3, I-5 and I-13 of the roadmap):

- **Text** has no cross-entropy in the table. `configs/experiment/text_vae.yaml` trained with `bce` as a stand-in.
- **Time series** with gaps give `NaN` with `mse`, and `computeMaskedSquaredError` exists but cannot be named in
  YAML.
- **Point clouds** have `computeChamferDistance`, and `configs/experiment/point_cloud_vae.yaml` carried a placeholder
  `mse` loss and a long warning header telling the reader to train from Python instead.
- **Tabular** data with categorical columns has no loss that treats a column as a category.

Two more things were missing around the loss. The reduction `sum` multiplied the loss by the number of values of a
sample, which is right for an element-wise mean and wrong for a loss that is already a sum over tokens or over
points (`computeTokenCrossEntropy`, `computeChamferDistance`, `computeMaskedSquaredError` each carry a note that
says to use them with `mean`). And `evaluate()` computed the same five metrics for every modality, in `float`
over the pooled dataset, so a series with a gap gave `NaN` everywhere and a point cloud was scored by the squared
error of its points one by one, in whatever order the decoder happened to emit them.

## Decision

**1. `losses/reconstruction/` is a package, with a registry like the other pluggable components.** The module
`losses/reconstruction.py` becomes a package. Everything it exported is still importable from
`global_vae.losses.reconstruction` (`computeTotalReconstructionLoss`, `LossFn`, the reduction helpers), so no
import in the code or in a user's script changes. It gains `AbstractReconstructionLoss`, `registerReconstructionLoss`,
`getReconstructionLossClass` and `listRegisteredReconstructionLosses`, and one file per loss.

**2. `AbstractReconstructionLoss` is an `nn.Module`.** `forward(reconstruction, target)` returns a scalar tensor.
A loss can own parameters, for example a learned noise scale (P7-1). The trainer holds the modules among its
reconstruction losses, puts their parameters in its optimizer, clips them with the model's, saves them in the
checkpoint and restores them when a run resumes, by the mechanism ADR 0047 built for the loss terms (decision 6).

**3. Three declarations, read by the aggregate and the evaluation.** They follow the pattern of
`requires_matching_dims` on the assemblers: class-level, read from the class through the registry, so nothing has
to build a loss to ask.

| Declaration | Meaning | Default |
|---|---|---|
| `requires_matching_shapes` | The reconstruction and the target must have the same shape. The check of roadmap item P0-4(g) stays for every loss that does not opt out, with its message. | `True` |
| `reads_nan_as_unobserved` | A `NaN` in the target is a value that was not observed, or a point that does not exist, and adds nothing to the loss. | `False` |
| `countTerms(reconstruction, target)` | The number of terms one sample's value is the mean of, which the `sum` reduction multiplies by. A method and not a constant, since it depends on the shapes. | the product of the axes after the batch |

The convention of ADR 0029 becomes uniform: **a loss returns a mean over its terms**, and `reduction: sum`
multiplies it by `countTerms`, which gives the per-sample sum averaged over the batch for every loss. A plain
callable that is not an `AbstractReconstructionLoss` (every `torch.nn.functional` loss, a user's own function)
keeps the old rule: the shapes must match, and the terms are the product of the axes after the batch.

A loss may also declare `default_metrics`, the names of the metrics (decision 9) that describe it, and
`decoder_arguments`, the constructor arguments that the config layer fills in from the decoder when the YAML does
not give them (decision 5).

**4. The registered losses.**

| Name | Wraps | Terms of a sample | NaN as unobserved | Same shape |
|---|---|---|---|---|
| `mse`, `l1`, `bce`, `smooth_l1` | `torch.nn.functional`, unchanged | every value | no | yes |
| `bce_with_logits` | `binary_cross_entropy_with_logits` | every value | no | yes |
| `masked_mse` | `computeMaskedSquaredError` | every value | yes | yes |
| `token_cross_entropy` | `computeTokenCrossEntropy` | the tokens (`length`) | no | no |
| `chamfer` | `computeChamferDistance` | 1 | yes | no |
| `tabular_mixed` | new, on the same bricks | numeric columns plus categorical columns | no | yes |

- `smooth_l1` takes `beta` (default `1.0`, the value of `torch.nn.functional`), which makes it the one of the four
  that shows the `{name, kwargs}` form.
- `masked_mse` is the sum of squared errors over the observed values, divided by the batch size, divided by the
  number of values of a sample. With `reduction: sum` it is exactly `computeMaskedSquaredError`. With `mean` it is
  the mean over all positions, an unobserved one counting as zero error, which is the honest reading of "a mean over
  the terms".
- `token_cross_entropy` takes **token ids** `(batch, length)` or a one-hot or soft target `(batch, length,
  vocabulary)` as the target, and the raw scores of a decoder built with `output_activation=None` as the
  reconstruction. It is `computeTokenCrossEntropy` divided by `length`: the mean cross-entropy of a token, so
  `sum` gives the cross-entropy of the text. Ids opt out of the shape check, which is why the check is a
  declaration. The text encoder already reads ids (`TokenEmbedding`), so the batch key can hold the ids and the
  one-hot copy that a batch tensor had to be (`utils/text_blocks.py`) is no longer needed.
- `chamfer` returns exactly `computeChamferDistance`, which is already a sum over points averaged over clouds, so
  its terms are `1`: `mean` and `sum` give the same number, and a weight on it reads like the weight of a summed
  squared error. The two clouds may have different numbers of points, and a row of `NaN` in the target is a point
  that does not exist.
- `tabular_mixed` takes `categorical_sizes` (the sizes of the one-hot blocks, the last columns, in the order of the
  decoder's `categorical_sizes`) and `probabilities` (default `True`: the blocks are what a decoder built with
  `categorical_sizes` returns, a softmax inside each block; `False` for raw scores). It is the mean, over the
  numeric columns and the categorical blocks, of the squared error of a numeric column and of the cross-entropy of
  a block. With `sum` it is the squared error of the numeric part plus the negative log-likelihood of the
  categorical part, which is the scale of the KL term. Its terms are the number of numeric columns plus the
  number of blocks.

A multi-resolution spectral loss for audio is not added: no corpus asks for it, and the registry is the
extension point for the day one does.

**5. `training.reconstruction_loss` resolves through the registry.** A value is one of:

- a **name**: `reconstruction_loss: masked_mse`, as before. The meaning of a bare name does not change;
- an **entry** `{name, kwargs}`: `reconstruction_loss: {name: smooth_l1, kwargs: {beta: 0.5}}`. `kwargs` is
  optional;
- a **mapping from a reconstruction name to a name or an entry**: `{signal: masked_mse, image: {name: l1}}`, as in
  ADR 0030 and ADR 0045, and a reconstruction left out keeps `mse`.

An entry is told from a mapping by its keys: a mapping whose keys are `name` (required) and optionally `kwargs`
and nothing else is an entry. The reading is made without the model, so it does not depend on what the model
produces. One corner is ambiguous: a model that has a reconstruction called `name` and a config `{name: l1}`,
which is an entry for every reconstruction or a mapping that gives `name` the loss `l1`. The config layer
**refuses** it, with a message that gives both ways out (add `kwargs: {}` to say it is an entry, or key the other
reconstructions too to say it is a mapping), because a guess would change the objective without saying so.
`kwargs` that arrive as an OmegaConf container are turned into plain Python containers.

A loss whose constructor needs something the decoder knows lists it in `decoder_arguments`. When the YAML does not
give it, the config layer reads the attribute of the same name from the decoder of that reconstruction:
`tabular_mixed` gets `categorical_sizes` from `tabular_mlp_decoder_v1`, as a decoder gets its `latent_dim`. For a
bare name or entry the decoder is that of the only reconstruction, and for a model with several the arguments
must be in the YAML (the message says so). A bare name or entry gives **one instance** to every reconstruction, so
parameters of a loss are shared by the modalities; a mapping gives each one its own.

`listSupportedReconstructionLossNames` and `resolveReconstructionLossFn` stay, now answered by the registry, and
`validateReconstructionLoss` keeps its messages (`training.reconstruction_loss: unknown loss 'huber'`) and adds the
entry form. `resolveReconstructionLoss` returns instances.

**6. The parameters of a loss reach the optimizer and the checkpoint like those of a loss term.** `Trainer`
collects the `nn.Module`s among its `reconstruction_loss_fn` (one, or those of the per-reconstruction mapping,
without duplicates) in `Trainer.reconstruction_losses`, an `nn.ModuleDict`. Its parameters are added to the
optimizer (a group of their own when the optimizer is an instance), clipped, and moved to the device. The
checkpoint holds them under `reconstruction_loss_state_dict`, a key of its own, written only when the state is
not empty, so a run with the four classic losses writes the file it wrote before. Restoring follows the rules
of `loss_terms_state_dict`: a state the trainer lacks is refused when `strict` is `True`, and a parameter the
checkpoint lacks is refused when the optimizer is restored. A callable that is not a module, `F.mse_loss`
included, is held by nobody and costs nothing.

**7. The aggregate reads the declarations.** `computeTotalReconstructionLoss` checks the shapes only for a loss
that requires them, takes `countTerms` for the `sum` reduction, and for a plain callable does what it did.

**8. Metrics are a name to function table with a declared default set.** `evaluation/metrics.py` gains
`registerMetric(name)` (a function decorator), `getMetric(name)` and `listRegisteredMetrics()`, and
`DEFAULT_METRIC_NAMES`, the five that every report had (`mse`, `rmse`, `mae`, `r2`, `pearson_r`).
`DEFAULT_RECONSTRUCTION_METRICS` stays what it was, a dict of the five. It is a table and not a one-of choice, as
the docstring of the module said, so a registered metric is not exclusive with another, and the registry does
not join `test_registry_contract.py`, whose entries are classes with a base class. Its behaviour (a duplicate
raises `ValueError`, an unknown name raises `KeyError` that lists the names, the list is sorted) is tested in the
tests of the metrics.

**9. The registered metrics.**

| Name | For | Reads |
|---|---|---|
| `mse`, `rmse`, `mae`, `r2`, `pearson_r` | any array | the whole tensor |
| `masked_mse`, `masked_rmse`, `masked_mae`, `masked_r2`, `masked_pearson_r` | arrays with gaps | the values whose target is not `NaN` |
| `chamfer`, `precision`, `recall`, `f_score` | point clouds | `computeCloudScores`, threshold `0.1`, in the units of the data |
| `edge_auc` | graphs | `computeEdgeScores`, every node present, undirected: the area under the ROC curve over the pairs above the diagonal |
| `token_accuracy` | text | the share of positions where the arg max of the scores is the target token |

A masked metric is the plain one computed on the observed values, flattened: the five plain metrics pool every
element in one scalar, so the two are equal on data with no gap (a test compares them). `chamfer`, `precision`,
`recall` and `f_score` each call `computeCloudScores`, which is cheap next to the pass of the model; a caller
who wants another threshold passes a `functools.partial` in a dict, as before. `edge_auc` has no node counts to
leave padding out (a metric is a function of two tensors), so a caller who has them calls `computeEdgeScores`. `token_accuracy`
takes the target as ids or as a one-hot tensor.

**10. `evaluate()` takes a metric set per modality.** `reconstruction_metrics` is now one of: `None`; a dict of
name to function for every modality (the 2.0.0 form, unchanged); a sequence of registered names for every
modality; or a mapping from a reconstruction name to a sequence of names or to a dict of functions, for the
modalities it names. For a modality without an explicit set, the set is the first of:

1. the `default_metrics` of its loss, when the loss given as `reconstruction_loss_fn` is an
   `AbstractReconstructionLoss` that declares them (`chamfer`: the cloud scores; `token_cross_entropy`:
   `token_accuracy`);
2. the five metrics in their masked form, when its loss reads `NaN` as unobserved, **reported under the plain
   names** so that a table keeps its columns;
3. the five default metrics.

So the metrics follow the loss the model was trained with, which `scripts/evaluate.py` already receives
(`--reconstruction-loss`), and a run with the default loss reports what 2.0.0 reported. `streaming=True` supports
the ten array metrics (the masked forms are accumulated by a variant of the streaming classes) and refuses a
metric that needs the pool, with a message that says which; a custom dict is refused with `streaming=True` as
before.

## Alternatives considered

- **Keep the table in `config/training.py` and add names to it.** The next loss would again be an edit of the
  config layer, which is what the registry pattern exists to avoid, and the declarations of decision 3 would
  have no place to live.
- **A loss as a plain function with attributes.** A function cannot own a parameter, so P7-1 would need another
  mechanism, and `nn.Module` is what every other registry here uses.
- **`countTerms` as a class constant.** It depends on the shape of the batch (the tokens of a text, the columns of a
  table), so a constant would be wrong for most of the losses.
- **Losses that return the sum, with `sum` and `mean` dropped for them.** The four classic losses return a mean,
  and ADR 0029 chose that on purpose. One convention for all of them is what lets one YAML line, `reduction: sum`,
  mean the same thing for a text, a table and an image. The helpers that returned a sum stay as functions with
  their notes, and the classes divide by the terms.
- **Treat a mapping with a key `name` as a per-modality mapping, always.** A reconstruction called `name` is rare,
  but an entry that is silently read as a mapping would give every modality `mse` and say nothing. The model-free rule
  with a refusal in the one ambiguous case is stricter than either guess.
- **A flat `kind` field next to the loss name** (`reconstruction_loss: {kind: ...}`). It needs a new reserved word
  where the entry form needs none: `name` and `kwargs` are what `optimizer`, `loss_terms` and the model configs
  already use.
- **Metrics as a class registry.** A metric is a function of two tensors with no state outside the streaming
  variants, and the five existing ones are functions that a test and a user's dict already import. A decorator that
  fills a table is the least change and has the same three functions as the class registries.
- **Always mask `NaN` in the default metrics.** It would make `evaluate()` NaN-free by itself, and also hide a
  corrupt batch of images behind a metric that skips what it cannot read. The `NaN` of a series is a declaration of
  the data (the loss says so), and the metrics follow the declaration.
- **Pick the metrics from the shape of the data** (a `(batch, points, 3)` tensor is a cloud). A heuristic that
  breaks the day a signal has three channels. The loss already says what the modality is.

## Consequences

- **The four classic names give the 2.0.0 numbers.** A test compares `mse`, `l1`, `bce` and `smooth_l1` with
  `torch.nn.functional` on the same tensors, through `computeTotalReconstructionLoss` with both reductions, and
  compares a trainer step built from YAML with one built with the function. The checkpoint of a run with them
  has no new key.
- **`trainer.reconstruction_loss_fn` of a YAML-built trainer is an instance and no longer `F.mse_loss`.** Tests
  that asserted identity with the function now assert the class and the value. Code that passes the function to
  `Trainer` by hand is unchanged.
- **The experiment configs name the right loss.** `text_vae.yaml` uses `token_cross_entropy`, `time_series_vae.yaml`
  `masked_mse`, `point_cloud_vae.yaml` `chamfer` (its placeholder and its warning header go), and `tabular_vae.yaml`
  `tabular_mixed`. Their weights follow the new meaning of `sum` (the weight of the ELBO is `1.0` for the
  likelihood terms).
- **A graph with `bce` keeps working**; `bce_with_logits` is there for a decoder built without a sigmoid.
- **The registry joins `test_registry_contract.py`** (twelve registries). The metrics table has its own tests.
- **Finding, not fixed: `masked_mse` with `mean` shrinks with the share of gaps.** An unobserved value counts as a
  term with no error. It is the reading that keeps `sum` exact, and the config files use `sum`. A mean over the
  observed values only would need a term count that changes from sample to sample.
- **Finding, not fixed: a bare loss entry shares its parameters across the modalities** (decision 5). No loss that
  ships has any. P7-1 decides what a learned scale per modality needs.
- **Specification proposal, for the owner.** The specification names `reconstruction_loss` and `reconstruction_weight`
  as per-modality settings and its lists of registries (§8, §10, §12) do not name the reconstruction loss or the metrics.
  A sentence for each would match the code: the loss is chosen by name from the registry, as a name, an entry
  `{name, kwargs}` or a mapping by reconstruction name, and the metrics are a name to function table with a
  default set of five. It is not edited here, since the specification is the owner's.
