# 0047: Extra loss terms at the single loss site

**Status:** accepted (roadmap item P5-6; it adds a registry, a `training.loss_terms` field and a `terms`
mapping on `StepLosses`, and a run that lists no term is unchanged, bit for bit)
**Date:** 2026-11-05

## Context

`Trainer.computeLosses` returns `reconstruction + regularization`, and `StepLosses` has those two fields
(finding I-7). Every objective that Phases 6 and 7 add needs the same place:

- a **latent alignment** between the latent spaces of two modalities (Phase 6);
- a **cross reconstruction**, which decodes the latent sample of one modality with the decoder of another
  (Phase 6, and the reason `GlobalVae.decode` exists, ADR 0046);
- a **latent reconstruction**, which encodes the reconstruction again and compares the two latent vectors (P7-3);
- a **reference alignment** to a space that something else defined (Phase 7).

The single loss site is a rule worth keeping. `fitEpoch` and `evaluate` both call `computeLosses`, so the
training curve and the validation curve are the same function of the batch, and a loss that exists in one and
not in the other cannot happen by construction. Each of these objectives would otherwise be written as a
subclass of `Trainer` that overrides `computeLosses`, or as a callback that adds a gradient after the fact, and
both break the rule.

Two things had to be settled besides the place. Some terms have **parameters** (a projection head in front of
an alignment, a learned weighting), and a parameter that is not in the optimizer, not in the checkpoint and not
restored on resume is a run that cannot be reproduced (ADR 0032). And the **scale** of a term needs the same
care as the others (ADR 0029, ADR 0031): computed in `float32`, divided by the accumulation like the total.

## Decision

**1. A registry of loss terms, `losses/terms/`.** It mirrors the registries of the regularizers and of the
beta schedules (`losses/terms/registry.py`):

| Function | Role |
|---|---|
| `registerLossTerm(name)` | Class decorator. A name registered twice raises `ValueError`. |
| `getLossTermClass(name)` | Lookup. An unknown name raises `KeyError`, and the message lists the available names. |
| `listRegisteredLossTerms()` | The registered names, sorted. |

`AbstractLossTerm` (`losses/terms/base.py`) is an `nn.Module` and an ABC, like `AbstractLatentRegularizer`, so
that a term with parameters and a term without are interchangeable through the same registry:

```python
def forward(
    self, model: GlobalVae, batch: Mapping[str, Tensor], outputs: Mapping[str, Any], step: int
) -> Tensor: ...
```

- `model` is the `GlobalVae` the trainer trains, never the compiled wrapper. A term that has to encode or decode
  again calls `model.encode` and `model.decode` (ADR 0046). It is an argument and never an attribute, so the
  model does not become a submodule of the term and its parameters are not counted twice.
- `batch` is the whole batch on the trainer's device, targets included (a key that is not an encoder input is
  still there).
- `outputs` is what `GlobalVae.forward` returned for this step: `reconstructions`, `latent_params` and
  `latent_samples`. It is the pass of the training step, so it went through modality dropout when that is on,
  and its tensors are `float32` under mixed precision (Decision 5).
- `step` is the global step, the one the beta schedules read. `evaluate` passes the current step without
  advancing it.
- It returns a **scalar** (a tensor with no dimension). A term that returns a vector raises a `ValueError` that
  names it, since a silent sum or mean would be a choice about the scale that the term has to make.

An optional hook, `validateModel(model)` (a no-op by default), is called once when the trainer is built, so that
a term which names latent spaces or modalities in its kwargs fails at build time on a typo and not at the first
step (the lesson of ADR 0045).

**The registry ships empty.** The terms are the objectives of Phases 6 and 7, and each of them lands with its
own ADR and its own evidence. A built-in invented here would be an objective with no use. The contract test of
the registries (`tests/unit/test_registry_contract.py`) has a flag for a registry that has no built-in yet; every
other clause of the contract applies to this one.

**2. `training.loss_terms`, a list of `{name, weight, schedule, kwargs}`.**

```yaml
training:
  loss_terms:
    - name: latent_alignment            # a registered term
      kwargs: {pairs: [[z_signal, z_image]]}
      schedule:                         # the weight follows a registered beta schedule
        strategy: linear_warmup
        kwargs: {warmup_steps: 500, end_value: 0.1}
    - name: latent_reconstruction
      weight: 0.5                       # or a constant weight (default 1.0)
```

- `name` is the registry key, and it is the term's name everywhere else: in `StepLosses.terms`, in the metrics,
  and in the checkpoint. Two entries with the same name raise a `ValueError`, and so does a name that is
  `total`, `reconstruction` or `regularization` (the keys of the three built-in terms) or that has a dot in it.
- `weight` is a number of at least zero. `schedule` is `{strategy, kwargs}`, resolved through the **beta
  schedule registry** (`constant`, `linear_warmup`, `cyclical_annealing`), and its value at the step **is** the
  weight, as a beta schedule is the beta of its latent space and not a factor of it. Giving both is a
  `ValueError`: a product of two numbers that the reader has to multiply to know the weight is worse than one.
  Giving neither means a weight of `1.0`.
- `Trainer` takes the same through two arguments, `loss_terms` (name -> term) and `loss_term_weights` (name ->
  a number or an `AbstractBetaSchedule`), in the way it takes `beta` and `beta_schedules`. A weight for a name
  that is not a term raises, since a mistyped name would otherwise leave the term at its default weight.

**3. `StepLosses.terms`, one entry per term, empty by default.** Each entry is the term **after** its weight,
the quantity that was added to the total, as `regularization` is already after its beta, so that
`total = reconstruction + regularization + sum(terms)` holds on the numbers that are logged. `asMetrics()`
gives `loss/<name>` for each, after the three it has, so the keys of a run without terms are the keys of
2.0.0. `fitEpoch` and `evaluate` prefix them, and the metrics are `train/loss/<name>` and `val/loss/<name>`:
the family of `train/loss/total`, which the callbacks that monitor a value (`val/loss/total`) and the loggers
already read. The CSV, TensorBoard and MLflow loggers take a flat dict of floats and need no change; a test
runs the CSV logger and reads the column.

**4. The parameters of a term belong to the trainer.** `Trainer.loss_terms` is an `nn.ModuleDict`, moved to the
trainer's device, and:

- **Optimizer.** When the trainer builds the optimizer from a class, the parameters of the terms are appended
  to those of the model in **one** parameter group, so a run has the same single learning rate key as before
  (`lr`, not `lr/group0` and `lr/group1`, ADR 0028). When the caller passes an optimizer instance, the
  parameters of the terms it does not already hold are added with `add_param_group`, which takes the defaults
  of that optimizer.
- **Clipping.** `grad_clip_norm` clips the gradient of the model and of the terms together, the one norm the
  optimizer step uses.
- **Checkpoint.** `saveCheckpoint` writes the `state_dict` of the terms under `loss_terms_state_dict`, **a
  separate key from `model_state_dict`**: the parameters of an objective are training-only, so a model that is
  loaded for evaluation or export does not need them, and does not carry them. The key is omitted when the
  terms hold no state, so the file of a run without a term is the file of 2.0.0. It holds tensors only and is
  read in the safe mode (ADR 0034).
- **Resume.** `loadCheckpoint` restores the terms before the optimizer, with the `strict` flag of the model:
  a state with names the trainer's terms do not have is a `ValueError` that lists both sets. A trainer whose
  terms have parameters, with an optimizer to restore, and a checkpoint that has no term state, is a
  `ValueError` that says the checkpoint was saved by a run with other terms (the optimizer state could not be
  loaded either, with a message about parameter groups that points nowhere). With `restore_optimizer=False` a
  term the checkpoint does not know keeps its initial parameters, and a warning says so. The optimizer state
  of the term parameters, the RNG state and the step are restored by the code that already restores them, so
  the guarantee of ADR 0032 covers a run with terms: the same curve to a relative `1e-6`, and the same
  weights.
- **Mode.** `fitEpoch` puts the terms in training mode and `evaluate` in evaluation mode, with the model, so a
  term with a dropout or a normalization behaves as the model does.

**5. `float32` and the accumulation.** A term runs outside the `torch.autocast` context, which covers the
forward pass of the model only, and its `outputs` are cast to `float32` the way the reconstructions and the
posterior parameters already are (`_castOutputsToFloat32` now also casts `latent_samples`, which only a term
reads, so no number of an existing run changes). A term is added to the total, and the total is divided by
`accumulate_grad_batches` before the backward pass, so a term is scaled exactly like the other two; the step
metrics are the mean over the batches of the step, term included. The `GradScaler` of `fp16` scales the total,
and unscales every parameter the optimizer holds, the terms' included.

**6. Nothing changes when no term is listed.** `computeLosses` adds nothing, `StepLosses.terms` is empty,
`asMetrics()` has three keys, the optimizer is built from `model.parameters()` as before, the checkpoint has no
new key, and the clipping covers the same list of parameters. A test keeps a copy of the 2.0.0 `computeLosses`
and compares a full run with it, history and weights, to the last bit, with mixed precision, accumulation,
clipping and modality dropout.

## Alternatives considered

- **A subclass of `Trainer` per objective that overrides `computeLosses`.** It works and it is what most
  projects do, but each objective then has its own loop, the validation pass has to be overridden again, and two
  objectives cannot be combined without a third subclass. The registry is what lets Phase 6 and Phase 7 terms be
  chosen and combined from a YAML file.
- **A callback.** A `TrainerCallback` sees the step after the backward pass (`onStepEnd`), and it cannot change
  the total that was differentiated, nor take part in the validation loss.
- **Terms as part of `GlobalVae`, like the regularizers.** A regularizer is a function of the latent parameters
  and lives in the model because a learned prior is part of the model (P4-2). A term is a function of the batch
  and the step as well, and its parameters are an artefact of training. Put in the model they would travel with
  every evaluation checkpoint and every export, and `state_dict` would change for every run that has a term.
- **A plain function registry, without `nn.Module`.** It cannot hold a parameter, so a term with a projection
  head would keep it outside the trainer, and the three things of Decision 4 would be the caller's to do. The
  regularizers are modules for the same reason (ADR 0003).
- **The weight multiplies the schedule** (`weight * schedule(step)`). It reads well for a ramp to a given
  height, but it makes `constant` and a plain weight two ways to say one thing and puts two numbers where the
  reader needs one. The ramp has its own `end_value`.
- **Log the raw value of a term and its weight separately.** It would show a ramp better, and it would double
  the columns of every run with terms. The weighted value is the one that adds up to the total, and the weight
  at a step is a function of the schedule that the config states.
- **One optimizer group for the model and one for the terms.** It would allow another learning rate for the
  terms, and it would change `lr` into `lr/group0` and `lr/group1` for every run with terms. A caller who wants a
  second rate passes an optimizer instance with its own groups, and the trainer adds only what is missing.
- **Reading the metric name literally from the roadmap, `train/<name>`.** It would put a term outside the
  `loss/` family that the three other terms and the monitoring callbacks use, and a term named `lr` or `epoch`
  could clash with a metric of another kind. Changing it is one line in `StepLosses.asMetrics`.

## Consequences

- **A run without `loss_terms` is unchanged.** A test compares it with a copy of 2.0.0's `computeLosses`, bit
  for bit, in four configurations (plain, `bf16` with accumulation, clipping with modality dropout, a beta
  schedule), and checks that the optimizer has one group of the model's parameters, that the metrics have three
  loss keys, and that the checkpoint has no `loss_terms_state_dict`.
- **A term with one parameter trains.** A dummy term with one parameter is in the optimizer, moves, shows up as
  `train/loss/<name>` and `val/loss/<name>` in the history and in the CSV file of the existing logger, is saved
  in the checkpoint and is restored, and a run resumed after half of its epochs gives the curve and the weights
  of the uninterrupted run to the tolerance of ADR 0032. The `bf16`, accumulation and `torch.compile` paths
  run with a term, and the term sees `float32` outputs.
- **The registry joins the contract test, and ships empty.** The contract has eleven registries, and one flag
  tells it that this one has no built-in. The first terms come with Phase 6.
- **Finding, not fixed: a term that decodes runs in `float32` under mixed precision.** It is outside the
  autocast context, so a cross reconstruction in a `bf16` run decodes in full precision. The result is right and
  the step is slower than it could be. A term can enter `torch.autocast` itself, and whether the trainer should
  offer a way to run it in the reduced precision is decided when the first such term exists.
- **Finding, not fixed: a term needs a pass of its own to see another pass.** `outputs` is the pass of the
  step, with its modality dropout, and a term that wants the pass with every modality, or the pass of one
  modality, calls `model.encode` and `model.decode`, and pays for it.
- **Not decided:** another learning rate for the terms (an optimizer instance does it today), a label so that
  one term can be listed twice with different kwargs (the name is the key and a duplicate is refused, so a
  term that takes several pairs in its kwargs is the answer for now), terms that apply to the validation pass
  only, and a way to put a term's parameters in the exported model.
- **The specification is not edited here.** Its lists of registries (spec §8, §10 and §12) do not name this
  one, and the specification is the ground truth: this ADR is the proposal for the owner, as the spec sync of
  P5-2 left it.
