# 0054: Pair-stage loss terms

**Status:** accepted (roadmap item P6-4; it adds the first two built-in loss terms, one hook on `AbstractLossTerm`
and two read-only methods on `GlobalVae`, and a run that lists neither term is unchanged, bit for bit)
**Date:** 2026-11-12

## Context

Stage 3 of the staged recipe (ADR 0051, ADR 0052) trains two modalities that each have a latent space of their own
and asks the two spaces to agree on the same sample. Finding I-9 is that no objective in the framework does that. The
pieces it needs are in place:

- a **distance** between two latent spaces, chosen by name (ADR 0053, P6-3);
- a **single loss site** where an extra objective is one more registered term, with a weight and a ramp (ADR 0047);
- a way to **decode the sample of one latent space with the decoder of another**: pass it to `GlobalVae.decode` under
  the other's name (ADR 0046);
- a **per-modality reconstruction loss, reduction and weight**, keyed by the name of the reconstruction (ADR 0045,
  ADR 0048).

What was missing is the two terms, and four questions that the roadmap leaves to this item: what a term may know
about the trainer that holds it (a cross reconstruction has to score a modality with the loss that the trainer uses
for it, and `forward(model, batch, outputs, step)` has no way to say what that is), what a pair stage refuses and
when, what "stop gradient" means for each term, and what the value of a term is when it has several pairs.

## Decision

**1. Two terms, `latent_alignment` and `cross_reconstruction`, in `losses/terms/`.** One class per file, registered
with `@registerLossTerm`. They are the first built-ins of the registry. `losses/terms/pairs.py` holds the checks that
the two share (it is not a term and registers nothing). Both are configured in `training.loss_terms`, so their weight
`lambda` and a ramp from zero are the `weight` and `schedule` of the entry, as for every term (ADR 0047), and a ramp
protects the solo solutions at the start:

```yaml
training:
  modality_dropout_p: 0.0
  loss_terms:
    - name: latent_alignment
      kwargs:
        pairs: [[z_signal, z_image]]
        distance: {name: wasserstein2_diag, kwargs: {squared: true}}
      schedule: {strategy: linear_warmup, kwargs: {warmup_steps: 100, end_value: 0.3}}
    - name: cross_reconstruction
      kwargs:
        pairs: [[z_signal, z_image], [z_image, z_signal]]
      schedule: {strategy: linear_warmup, kwargs: {warmup_steps: 100, end_value: 0.3}}
```

**2. `latent_alignment`: the distance of each pair, averaged over the batch, summed over the pairs.**

- `pairs` is a non-empty list of `[first, second]` latent space names. A pair of one name with itself, a pair listed
  twice, and a name that is not a string are refused with the entry named. The two orders of a pair are one pair when
  the distance is symmetric (`is_symmetric`, ADR 0053), and two when it is not.
- `distance` is **required**: the name of a registered latent distance, or `{name, kwargs}`. Which distance stage 3
  trains with is roadmap decision D-13, which stays open until the measurements of P6-6, so the config states the
  choice and no default makes it silently. The how-to says which to try first (`wasserstein2_diag`, with `mean_mse` as
  the ablation).
- The distance acts on the posterior parameters of **one sample from each modality**, not on two independent samples:
  the term reads `outputs["latent_params"]` of the step, where row `i` of one space and row `i` of the other describe
  the same sample. Nothing is encoded again, so the term costs one distance per pair.
- The value is `sum over pairs of mean over the batch of distance(first, second)`. A sum over pairs, and not a mean,
  because each pair is one more thing asked of the model, as each latent space is one more KL term, and a mean would
  make the weight depend on how many pairs were listed. With a distance that sums over the latent dimensions (every
  built-in does by default) the value is on the footing of the KL term (ADR 0029), so `lambda` reads next to `beta`.
- `stop_gradient` is `none` (default), `source` or `target`: the gradient does not reach the first, or the second,
  space of every pair, which makes it an anchor that the other space is trained toward.

**3. `cross_reconstruction`: each target modality rebuilt from another modality's latent sample.**

- `pairs` is a non-empty list of **ordered** `[source, target]` latent space names. List both orders for both
  directions. A pair listed twice in the same order is refused.
- For each pair the term takes the sample that the step drew in the `source` space (`outputs["latent_samples"]`),
  passes it to `GlobalVae.decode` under the name of the `target` space, and compares what the decoders that read
  `target` produce with the batch tensors of the modalities they reconstruct.
- **The comparison is the trainer's own.** It is `computeTotalReconstructionLoss` with the trainer's per-modality loss,
  weight and reduction (ADR 0045, ADR 0048), so a modality is scored the same way whether it was rebuilt from its own
  space or from another's, and the sum reduction multiplies it by the same number of terms. The term does not carry a
  loss of its own to configure a second time and get out of step.
- The value is the sum over the ordered pairs. Its weight and ramp are the entry's.
- `stop_gradient` is `none` (default), `source` or `target`. `source` detaches the sample, so only the decoders learn
  to read the other space and the source encoder is left as it is. `target` keeps the parameters of the decoders and
  of the assemblers out of this term's gradient, so only the encoders learn to write what the decoders read: it
  switches their `requires_grad` off for the duration of the decoding and puts every flag back as it was, a parameter
  that was already frozen included. The gradient still reaches the sample through the decoders' activations.

**4. The rules.**

- **Every batch must carry every modality of a pair.** A batch that lacks one raises a `ValueError` that says that
  the term needs paired batches, names each missing modality with the reason it is needed (which pair, and whether it
  feeds the source space or is what the target space decodes to), and lists what the batch has. The term raises and
  does not skip the pair, because a skipped term changes the objective without a trace in the numbers, and the likeliest
  cause is a loader that does not pair its samples. The modalities that a pair needs come from the routing graph
  (`GlobalVae.modalitiesEncodedInto`, Decision 6), so a latent space fed by several encoders needs all of them. The check
  runs in training and in validation, since both go through the single loss site.
- **The two spaces of a pair must have the same width (roadmap D-16),** checked when the trainer is built, in
  `validateModel`: the message gives both widths and names the field to set. A name that is not a latent space of the
  model is refused there too, with the real ones listed (the lesson of ADR 0045). D-16 is a build check and not a
  conversion: nothing projects one space onto the other.
- **A cross reconstruction needs a decoder that can run from the target space alone,** also checked at build time
  (`GlobalVae.modalitiesDecodedFrom`): a space that no decoder reads, or that is read only through an assembler that
  needs the other spaces too (`concat`), is refused with the pair named, instead of decoding nothing at the first step.
- **Modality dropout is rejected.** A pair stage has no fusion to be robust for, and a term that compares two spaces
  needs both encoded in every step. A trainer with `modality_dropout_p` above zero that holds either term raises a
  `ValueError` when it is built, with the setting to change (`training.modality_dropout_p=0`).

**5. A hook, so that a term can know what it needs about the trainer.** `AbstractLossTerm.bindTrainerSettings(settings)`
is called once for every term when the trainer is built, right after `validateModel`, with a frozen
`TrainerSettings`: the trainer's `reconstruction_loss_fn`, `reconstruction_weights`, `reconstruction_reduction` and
`modality_dropout_p`. The default does nothing, so no existing term changes. It raises a `ValueError` for a setup the
term cannot run with, which is how Decision 4 refuses modality dropout. A term keeps the settings as a plain attribute,
never as a submodule: a reconstruction loss that is a module is shared with the trainer, and registering it a second
time would count its parameters twice (the reason ADR 0047 passes the model as an argument). The settings are what the
trainer was given, the same objects.

**6. Two read-only methods on `GlobalVae`.** `modalitiesEncodedInto(latent_name)` returns the batch keys that the
encoders feeding a latent space ingest, and `modalitiesDecodedFrom(latent_names)` returns the reconstructions that
`decode` gives when only those spaces are passed. Both read the routing graph and run nothing. The second applies the
rules of `decode` (ADR 0046) without calling a decoder; the loop of `decode` is unchanged, so the bit-for-bit guarantee
of that ADR is untouched, and a test compares the method with the keys that `decode` returns on every subset of the
latent spaces of several models, so that the two cannot drift apart.

**7. Nothing else changes.** The terms have no parameters (the distance is a module with no state), so
`loss_terms_state_dict` stays absent from the checkpoint and the resume path of ADR 0047 has nothing new to restore;
`mixture_jsd` draws from the global generator, which the checkpoint restores (ADR 0032). The terms run outside the
autocast context in `float32` (ADR 0031). The metrics are `train/loss/<name>` and `val/loss/<name>` for each, the
weighted value that was added to the total (ADR 0047). The registry contract test no longer has a registry that ships
no built-in.

## Alternatives considered

- **Give `forward` the trainer.** The term could then read every setting at the step. It also makes the term able to
  change anything, and it makes the trainer a dependency of every term that a test has to build. A frozen record of
  the four settings that a term has needed so far, given once, is the narrower interface, and the next setting a term
  needs is one more field.
- **Fill the reconstruction settings in through the config layer, as `decoder_arguments` does for a loss.** It works
  only for a term built from a config. A `Trainer` built in Python, which is how most of the tests and notebooks use
  it, would have to pass the same three arguments by hand and keep them equal to its own.
- **A loss, a reduction and a weight on the term itself.** It is configurable and it is the way to have a modality
  scored one way when it is rebuilt from itself and another way when it is rebuilt from the other, which is the thing
  to avoid.
- **Unordered pairs and a `both_directions` flag.** It is shorter to write for the common case, and it hides that
  the two directions are two terms with two gradients. Listing both orders is one more line and says what is done.
- **A `stop_gradient` for each pair.** It allows an anchor in one direction only. The name of a term is its key, and a
  term cannot be listed twice (ADR 0047), so a mixed setting would need a second name for the same term; it is left for
  the case that needs it.
- **Skip a pair whose modality is absent from the batch.** It is the behaviour of the other terms of the loss, where a
  missing modality means a missing reconstruction. In a pair stage a missing modality is a bug of the loader, and a
  silent skip is the failure that the stage's own check is there to catch.
- **A default distance.** It saves a line of YAML and it decides D-13 in the code.
- **A mean over the pairs.** It keeps the weight of a term from depending on how many pairs it has. It also makes a
  third space halve the weight of each pair, and the sum is what the KL terms do.
- **Cut the target side by `torch.func.functional_call` with detached parameters.** It leaves the flags alone. It
  calls `forward` of a module and not the method `decode`, so it would need its own copy of the decoding rules, which
  is the loop that ADR 0046 made one.
- **Cross reconstruction through `forward` with a swapped input.** `forward` encodes, and the point of the term is to
  use the latent sample that the step has already drawn, in the same pass as the self reconstruction.

## Measurements

The synthetic pair is a signal (a sine of a random frequency and phase) and an image (a Gaussian blob whose place is
a function of that frequency and phase), so the two modalities describe the same two numbers and an aligned pair has
something to align. The pair model is the one of `configs/model/signal_image_pair.yaml`, warm-started from two solo
models trained for 15 epochs (ADR 0052). Stage 3 is 100 epochs of 4 batches of 16, the sum reduction, `beta` 0.1,
modality dropout off, and the two terms with a ramp from zero over 100 steps to 0.3. The control is the same stage with
no term. Each figure is read on two validation batches from the posterior means, with the 2-Wasserstein distance
(squared) for the distance and the sum of squared errors per sample for the reconstructions, so that training noise is
not in them. Four seeds (the control varies much more than the run with the terms, and the range is the range of the
four):

| Measure, validation | Start (after the warm start) | Control, end | Terms, end |
|---|---|---|---|
| Latent distance | 138.7 | 9.7 to 16.5 | 1.5 to 3.0 |
| Signal self reconstruction | 139.3 | 52 to 65 | 55 to 68 |
| Image self reconstruction | 84.5 | 74 to 304 | 62 to 76 |
| Signal to image cross decoding | 103.9 | 592 to 897 | 38 to 43 |
| Image to signal cross decoding | 135.0 | 160 to 175 | 83 to 124 |

What the table says: the terms bring the two spaces together (the distance, and above all the cross decoding, which is
the content of "aligned": a signal's latent sample decoded as an image gives the image), and neither self
reconstruction ends above where stage 3 started. What it does not say: the control's distance also falls, because both
spaces move toward the prior under the KL term, so a falling distance on its own shows little, and the cross decoding
is the measure that separates a run with the terms from one without. A constant weight of 1 on both terms at the
default `beta` of 1 was worse (image self reconstruction 84 to 130 in 120 steps), and the sum reduction is needed to keep
the models from collapsing (finding I-4), as in every stage of the recipe. These are the numbers of a small synthetic
pair, not of a real one, and they are not a recommendation for a weight (finding I-15).

## Consequences

- **A run with neither term is unchanged, bit for bit.** The hook defaults to a no-op, the two model methods are new,
  and a test runs the existing loss-term suite and the contract suite unchanged.
- **The registry ships two built-ins.** The contract test keeps its flag for a registry with none, and no registry sets
  it any more; the two checks that were skipped now run on this one.
- **On the synthetic pair, stage 3 lowers the latent distance and does not raise a self reconstruction.** The test
  `tests/integration/test_pair_stage.py` warm-starts the pair model from two solo checkpoints, trains it with the terms
  and without, and checks, with the tolerances stated in it: the distance falls to a tenth of its start and to half of
  the control's, neither self reconstruction ends more than 10 % above its start, and the cross decoding is better than
  the control's. Each term is in the history and in the CSV file as `train/loss/<name>` and `val/loss/<name>`, and a
  run interrupted and resumed gives the curve and the weights of the uninterrupted one (ADR 0032), with the
  deterministic distance and with `mixture_jsd`, whose draws the checkpoint restores.
- **Finding, not fixed: the logged value of `latent_alignment` is the weighted one.** With a ramp the curve starts at
  zero when the distance is largest, and the distance itself has to be measured on its own, as the tests do, or read
  with a constant weight. ADR 0047 chose the weighted value because it is the one that adds up to the total, and
  logging the raw value next to it is not decided.
- **Finding, not fixed: the cross decoding runs the decoders in training mode.** Their `BatchNorm` layers update their
  running statistics with activations that come from the other modality's latent samples, so the running statistics
  after stage 3 describe a mixture of the two, and the self reconstruction in evaluation mode is computed with it. The
  measurements above are in evaluation mode and the self reconstructions did not rise, but a stage that trains for long
  with a small `momentum` should look at it. The freeze policy of P6-5 puts a frozen module in evaluation mode, which
  is the tool for it.
- **Finding, not fixed: `stop_gradient: target` flips `requires_grad` for the duration of one decoding.** It is not
  safe to call from several threads on the same model, which a training step does not do, and the term runs on the
  trainer's own model and not on a compiled wrapper, so `torch.compile` does not see it. A cross decoding in a `bf16`
  run is in `float32`, as ADR 0047 recorded for every term that decodes.
- **Not decided:** a per-pair `stop_gradient`, a group of more than two spaces for a distance that takes them
  (`mixture_jsd` does), a second logged value that is the raw distance, a cross reconstruction that goes through a
  head to reconcile unequal widths (D-16 refuses them for now), and the distance that stage 3 uses (D-13, after P6-6).
- **The specification is not edited here.** It describes the loss through its terms of reconstruction and
  regularization and does not list the terms of the registry, as ADR 0047 noted; this ADR is the proposal for the
  owner.
