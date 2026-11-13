# Add a fusion, latent head, assembler, regularizer, loss term, reconstruction loss, metric, transform, beta schedule, callback, or logger

Every one of these follows the exact same three-step pattern: subclass an
abstract base class, decorate it to self-register under a name, and import
the new module from its subpackage's `__init__.py` so the decorator actually
runs.

| You're adding a... | Base class | Decorator | Lives in | Import it from |
|---|---|---|---|---|
| Fusion strategy | `AbstractFusion` | `@registerFusion("name")` | `fusion/` | `fusion/__init__.py` |
| Latent head | `AbstractLatentHead` | `@registerHead("name")` | `heads/` | `heads/__init__.py` |
| Assembler | `AbstractAssembler` | `@registerAssembler("name")` | `assemblers/` | `assemblers/__init__.py` |
| Latent regularizer | `AbstractLatentRegularizer` | `@registerRegularizer("name")` | `losses/regularizers/` | `losses/regularizers/__init__.py` |
| Latent distance | `AbstractLatentDistance` | `@registerLatentDistance("name")` | `losses/distances/` | `losses/distances/__init__.py` |
| Loss term | `AbstractLossTerm` | `@registerLossTerm("name")` | `losses/terms/` | `losses/terms/__init__.py` |
| Reconstruction loss | `AbstractReconstructionLoss` | `@registerReconstructionLoss("name")` | `losses/reconstruction/` | `losses/reconstruction/__init__.py` |
| Metric | a function `(reconstruction, target) -> float` | `@registerMetric("name")` | `evaluation/` | `evaluation/__init__.py` |
| Data transform | `AbstractTransform` | `@registerTransform("name")` | `data/transforms/` | `data/transforms/__init__.py` |
| Beta schedule | `AbstractBetaSchedule` | `@registerBetaSchedule("name")` | `training/beta_schedules/` | `training/beta_schedules/__init__.py` |
| Training callback | `TrainerCallback` | `@registerCallback("name")` | `training/callbacks/` | `training/callbacks/__init__.py` |
| Experiment logger | `AbstractExperimentLogger` | `@registerLogger("name")` | `training/loggers/` | `training/loggers/__init__.py` |

A logger is not a training callback for this table's purposes, even though
`AbstractExperimentLogger` happens to subclass `TrainerCallback` so
`Trainer.callbacks` can call its hooks uniformly (ADR 0008): that inheritance
is plumbing, not registry membership. A `Logger` is a journalling service
selected from `TrainingConfig.loggers`/`LoggerEntryConfig`, through its own
`registerLogger`/`getLoggerClass` registry; a training callback (checkpoint,
early stopping, learning-rate scheduling, or your own) is selected from
`TrainingConfig.callbacks` (registry name -> kwargs), through the separate
`training.callbacks` registry. See `docs/adr/0022-callback-registry.md` for
why the two stay apart.

## Steps

1. Create one new file per strategy: one class per file, no god-files (see
   the project's coding standards, spec §10 "Modularity").
2. Subclass the relevant base class above and implement its abstract
   method(s).
3. Decorate the class with the matching `@registerX("your_name")`. This is
   what makes it discoverable by name from config, with nothing else needing
   to import your class directly.
4. Add the one-line import to that subpackage's `__init__.py`, mirroring the
   existing lines there. Without this, `getXClass("your_name")` raises
   `KeyError` even though the file exists on disk.
5. Add a unit test covering registration, lookup, and (where relevant) the
   duplicate-name and unknown-name error paths, following the pattern of the
   existing registry tests.

## A note specific to latent heads

A latent head sits on one encoder -> latent edge and adapts the encoder's
`(mu, logvar)` to the width of the latent space that edge feeds (spec §2.2).
It is how one encoder will feed several latent spaces, a shared one and
private ones for example, instead of giving the encoder several output heads.
The built-ins are `identity` (the default, a no-op that needs
`in_dim == out_dim`) and `linear` (one learned projection for `mu` and
another for `logvar`, Xavier uniform weights and zero biases).

## A note specific to assemblers

An assembler merges the sampled latent vectors of a decoder that consumes several latent spaces
(spec §2.2). `GlobalVae` builds it as `getAssemblerClass(name)(input_dims, **kwargs)`, with
`input_dims` the `dim` of each latent space the decoder reads, in the order of
`RoutingGraph.decoderInputs()`. So a new assembler must:

- accept `input_dims` first, call `super().__init__(input_dims)`, and give every further argument
  a default (they come from `decoders_consume.<decoder>.kwargs`);
- implement `output_dim`, the width of what `forward` returns. The model checks it against the
  decoder's `latent_dim`, or fills `latent_dim` in when the decoder's kwargs omit it;
- implement `forward(latents, present=None)`, calling `self._resolvePresent(latents, present)` first.
  `present` is the position in `input_dims` of each tensor when only some latent spaces are
  present, which a strategy with one parameter per space needs to pick the right ones;
- set `requires_matching_dims = True` if all the input widths must be equal. The base constructor
  then refuses differing widths, and `validateRoutingGraph` reads the flag through the registry, so
  neither needs an edit for your class;
- override `handlesMissingLatents` to return `True` only if `forward` is defined for any non-empty
  subset of the spaces. The default is `False`, and a decoder with an incomplete input to such an
  assembler is left out of that pass.

The built-ins are `concat`, `sum`, `average`, and the learned `weighted_sum` and `attention`.

## A note specific to loss terms

A loss term is one extra objective added to the loss of every step, chosen by name from
`training.loss_terms` (roadmap P5-6, [ADR 0047](../adr/0047-extra-loss-terms.md)). It is an `nn.Module`
whose `forward` receives the model, the batch, what the model produced for the step and the step, and
returns a **scalar**:

```python
@registerLossTerm("latent_alignment")
class LatentAlignment(AbstractLossTerm):
    def __init__(self, pairs: list[list[str]]) -> None:
        super().__init__()
        self.pairs = pairs
        self.head = nn.Linear(32, 32)  # a parameter: the trainer trains and saves it

    def validateModel(self, model: GlobalVae) -> None:  # optional: fail at build time
        for pair in self.pairs:
            for name in pair:
                if name not in model.latent_spaces:
                    raise ValueError(f"latent_alignment: no latent space '{name}'")

    def forward(self, model, batch, outputs, step):
        a, b = (outputs["latent_params"][name][0] for name in self.pairs[0])
        return (self.head(a) - b).pow(2).mean()
```

- **The weight is not the term's business.** It is `weight` (a number, default 1.0) or `schedule` (a
  registered beta schedule, so the term can be ramped) in the config entry, and the trainer applies it. The
  value that is logged, as `train/loss/<name>` and `val/loss/<name>`, is the term after its weight.
- **Return one number.** Reduce over the batch yourself (a mean is the usual choice); the trainer refuses a
  vector, naming the term, since the scale is a choice that belongs to the term.
- **The term runs in `float32`**, outside the autocast context, on `float32` outputs. To encode or decode
  again, call `model.encode(...)` and `model.decode(...)` (ADR 0046), and do not keep the model as an
  attribute: it would become a submodule of the term.
- **Parameters are the trainer's.** They join the optimizer (in the model's single group), are clipped with
  the model's, are saved in the checkpoint under `loss_terms_state_dict` and are restored by `resume_from`.
  A checkpoint can only be resumed with the terms it was saved with.
- **The name is the key.** It is the registry name, a term is listed once in `loss_terms`, and it cannot be
  `total`, `reconstruction` or `regularization`.
- The registry ships empty on purpose: the terms are the objectives of Phases 6 and 7, each with its own ADR.

## A note specific to reconstruction losses

A reconstruction loss compares what a decoder wrote with its target, and it is chosen by name from
`training.reconstruction_loss` (roadmap P5-3, [ADR 0048](../adr/0048-reconstruction-loss-and-metric-registries.md)).
It is an `nn.Module` whose `forward` receives the reconstruction and the target and returns a **scalar that is a
mean over the loss's terms**:

```python
@registerReconstructionLoss("huber_per_channel")
class HuberPerChannelLoss(AbstractReconstructionLoss):
    reads_nan_as_unobserved = True  # a NaN in the target adds nothing
    default_metrics = ("masked_mse", "masked_mae")  # what `evaluate` reports for this loss

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, reconstruction, target):
        seen = ~torch.isnan(target)
        error = functional.huber_loss(
            reconstruction[seen], target[seen], delta=self.delta, reduction="sum"
        )
        return error / target.numel()
```

- **Declare what the generic code needs to know**, as class attributes, so that nothing has to build the loss to
  ask: `requires_matching_shapes` (default `True`: the reconstruction and the target must have one shape, and
  the check of the trainer stays for a loss that does not opt out), `reads_nan_as_unobserved` (default `False`),
  `default_metrics` (names of registered metrics, default `None`, meaning the five default ones) and
  `decoder_arguments` (constructor arguments the config layer reads from the decoder of the reconstruction when
  the YAML does not give them, as `tabular_mixed` does with `categorical_sizes`).
- **Return a mean over the terms, and say how many there are.** `reconstruction_reduction: sum` multiplies the
  value by `countTerms(reconstruction, target)`, the number of terms one sample's value averages over, so
  the same line means the same thing for every loss ([ADR 0029](../adr/0029-loss-scale-convention.md)). The
  default is the product of the axes after the batch. Override the method for a loss whose terms are the tokens
  of a text, the columns of a table, or one per cloud.
- **Parameters are the trainer's.** A parameter of the loss (a learned noise scale, say) joins the optimizer, is
  clipped with the model's, and is saved in the checkpoint under `reconstruction_loss_state_dict`. A loss with no
  parameter writes nothing. A bare name or an `{name, kwargs}` entry gives one instance to every reconstruction,
  so its parameters are shared by the modalities, and a mapping builds one per reconstruction.
- **Keep the metrics in step.** A loss that reads `NaN` as unobserved gets the masked metrics, and one that names
  `default_metrics` gets those. If your loss is for a new kind of data, register a metric for it
  (below) so `evaluate()` says something about the data and not only about a squared error.

## A note specific to metrics

A metric is a function of the reconstruction and the target that returns a float. Unlike the strategies above, a
metric is not one of several exclusive choices: `evaluate()` computes a set of them together, so the registry is
a name to function table (`registerMetric`, `getMetric`, `listRegisteredMetrics`) and not a class registry:

```python
@registerMetric("median_error")
def computeMedianError(reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    return float((reconstruction - target).abs().median())
```

Name it in `evaluate(reconstruction_metrics=["median_error", "mse"])`, or in a loss's `default_metrics`. For a
metric that must ignore a `NaN` target, wrap it with `maskUnobserved(function)`, which is what the `masked_*`
forms of the five default metrics are. The streaming evaluation accumulates the ten array metrics and refuses
any other with a message that names it.

## A note specific to data transforms

Transforms carry one extra constraint (spec §6.2): they must work on a
tensor of *any* shape or dimensionality, and must never encode anything
specific to one dataset or modality. Shape-dependent behavior goes through
an explicit, caller-supplied parameter (like `ResampleTransform`'s
`num_spatial_dims`), never a hardcoded branch on `x.dim()`.

## A note specific to training callbacks

Unlike every other row above, a callback is not chosen one-at-a-time: a
training run typically wants several active together (checkpointing, early
stopping, a learning-rate scheduler, all at once). `TrainingConfig.callbacks`
is therefore a `dict[str, dict[str, Any] | None]` (registry name ->
constructor kwargs), not a single `strategy` field, and `Trainer.callbacks`
is a plain list (loggers from `TrainingConfig.loggers` are combined into the
same list, ahead of these). If your callback's constructor declares a
`config` parameter, `buildCallbacksFromConfig` passes it the full run config
automatically (checked via the parameter's presence, not by special-casing
your class), matching what `CheckpointCallback`/`BestCheckpointCallback`
already do (spec §10: "config snapshotted with every run").

A callback that changes the optimizer's learning rate (a warmup, a cosine schedule, anything that
writes `trainer.optimizer.param_groups[...]["lr"]`) declares it with a class attribute:

```python
@registerCallback("cosine_decay")
class CosineDecay(TrainerCallback):
    acts_on_learning_rate = True
```

`Trainer` then records the rate it trains with each epoch as `train/lr`, so it shows in the log
line, the loggers and the checkpoint history, and `scripts/visualize_latent.py` plots it as
`lr_curve.png` ([ADR 0028](../adr/0028-learning-rate-in-the-metrics.md)). Without the flag the
callback still works, but nothing records the rate.
