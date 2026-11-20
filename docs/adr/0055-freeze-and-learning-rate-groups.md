# 0055: Freeze modules and learning-rate groups

**Status:** accepted (roadmap item P6-5; it adds one callback, one trainer argument and one check at resume, and a run
that uses neither is unchanged, bit for bit)
**Date:** 2026-11-13

## Context

Stage 3 of the staged recipe starts from two solo solutions (ADR 0051, ADR 0052) and trains the pair model on a few
pairs. The reference paper trains both VAEs together, from the first step, at the same rate. With few pairs and small
batches that can undo what the solo stages learned: the optimizer is free to move every weight, and the `BatchNorm`
layers of the CNN blocks (`nn.BatchNorm1d` and `nn.BatchNorm2d` in `utils/conv_blocks.py`) move their running
statistics at every forward pass in training mode, whether the weights train or not. ADR 0054 flagged the second
effect for the cross reconstruction, which decodes in training mode.

Three later items want the same mechanism: a one-sided alignment (one modality is an anchor, the other moves toward it),
a frozen reference encoder (P7-5) and a post-hoc prior (P4-2). Roadmap decision D-14 asks what is frozen in stage 3;
the proposed answer is "nothing, as in the paper", with the gentler variants available.

What was missing: a way to name modules, a way to keep them fixed that is correct for `BatchNorm` and not only for the
weights, a way to give some modules a smaller learning rate, and the answer to what each of these does to a checkpoint
and to a resume.

## Decision

**1. A registered callback `freeze_modules`, with `modules` and an optional `until_epoch`.**
`training/callbacks/freeze_modules.py`, configured in `training.callbacks` like every callback. `modules` is a list of
name patterns (item 2). `until_epoch` is the first epoch in which the modules train: they are frozen in epochs
`0 .. until_epoch - 1` and released at the start of epoch `until_epoch`; `None` (default) never releases them; a value
below 1 is refused (it would freeze nothing, and "leave the callback out" says that). Frozen is three things, done in
`onEpochBegin`, which runs after `Trainer.fitEpoch` has put the model in training mode:

- `requires_grad` is switched off, and a gradient that a parameter still holds is dropped, so that no optimizer moves
  it, weight decay included;
- the module is put in **evaluation mode**, so that its `BatchNorm` statistics and its dropout stand still. This is the
  part that a `requires_grad_(False)` alone does not do, and it has to be repeated every epoch because the trainer
  puts the whole model back in training mode at the start of each one;
- it **stays in the optimizer**. Nothing is added to or removed from the groups, so the checkpoint has the same layout
  frozen or not, and a run can be resumed across the epoch at which a module is released. The optimizer skips a
  parameter whose gradient is `None`, so a frozen one costs nothing but its place in the state.

Whether an epoch is a frozen one is a function of the epoch number alone, and `Trainer` hands the callback the real
number after a resume, so the callback has **no state to save**: a resumed run is frozen, or released, exactly as the
uninterrupted one. A parameter that was not trainable before the run is left alone (it is not frozen a second time, and
the release does not make it trainable), and when the run ends every `requires_grad` goes back as it was found.

**2. Modules are named by patterns that match `named_modules()`.** `training/module_patterns.py`:
`encoders.signal`, `decoders.*`, `encoders.signal.conv.0`. The match is `fnmatch.fnmatchcase` against the dotted name
of every module, so `*` stands for any characters, dots included, and a pattern is case sensitive. A module stands for
its own parameters and those of every module inside it. The root module (name `""`) matches only `*`. A pattern that
matches no module is an **error** that lists some real names, at the start of the run for the callback and when the
trainer is built for the groups, because a typo that freezes nothing looks like a stage that works. The module is
shared by the two features so that they cannot disagree on what a name means.

**3. `training.param_groups`: a learning-rate factor per set of modules.** A list of entries with `modules` (patterns),
`lr_factor` (a finite number above 0; 0 is refused with a message that points to `freeze_modules`, and a `bool` is not
a number) and an optional `name`. `Trainer` takes them as `param_groups` (a sequence of `ParamGroupSpec`,
`training/param_groups.py`) and builds the optimizer from the class it is given: one group for each entry, in order,
then a group named `default` with every remaining parameter of the model and the parameters of the loss terms and of the
reconstruction losses, at factor 1. The first entry that matches a parameter takes it, so a narrow entry goes before a
wide one. An entry that ends with no parameter is an error that says which earlier entries took them. Each group holds
`name` and `lr_factor` next to `params`, and `lr` is the optimizer's own `lr` times the factor, set once the optimizer is
built (`applyLrFactors`), so it works for any optimizer class that has an `lr`. **Without `param_groups` the optimizer is
built exactly as before, one group with no extra key.** An optimizer *instance* together with `param_groups` is refused:
the groups can only be built for a class.

**4. The schedules keep the factor.** `linear_warmup_lr` and `cosine_lr` (ADR 0028) scale each group from its own
`initial_lr`, which the optimizer holds under that key, so the group at 0.1 stays at a tenth of the default group along
the whole schedule, and `lr/group<i>` logs each one. No schedule needed a change.

**5. The groups reach the checkpoint, and a resume with other groups is refused.** PyTorch saves every key of a group
but `params`, so `name`, `lr_factor` and the scaled `lr` are in `optimizer_state_dict` already. They are also what a
resume **restores**: `load_state_dict` overwrites a group's `lr` with the saved one, so a run resumed with a config that
says 0.5 where the file says 0.1 would carry on at 0.1 and ignore the config without a word, and one with another number
of groups would stop with PyTorch's message about sizes, which does not say why. `loadCheckpoint` therefore compares the
`(name, lr_factor)` of each group, in order, just before `load_state_dict`, and refuses a difference with a message that
names `training.param_groups` and `restore_optimizer=False`. It does nothing when neither side has a named group, which is
every checkpoint written before this item and every run that does not use the groups.

**6. A clear error when everything that trains is frozen.** If, in an epoch, no parameter the optimizer holds can train,
`freeze_modules` raises before the first step, and the message says what to change. It also says that keeping both VAEs
frozen and training only an alignment head needs a head that the model does not have yet (the adapter of P6-8, roadmap
D-16; since ADR 0060 the message points at `latent_alignment`'s `adapters`). A batch that only a frozen modality can use (no trainable parameter takes part in the loss) gives, in
`Trainer._backward`, a `RuntimeError` that says so, where PyTorch says "element 0 of tensors does not require grad".

**7. D-14 is implemented as opt-in.** The default is nothing frozen and one group, as in the paper. The options of the
proposal are a line of configuration each: the decoders frozen (`modules: [decoders.*]`), one side frozen
(`[encoders.signal, decoders.signal]`), a gentler rate for the pretrained modules (`param_groups` with `lr_factor`), and
a release after some epochs (`until_epoch`). "Both VAEs frozen, an alignment head trained" is item 6's message until P6-8.

## Alternatives considered

- **`requires_grad_(False)` and nothing else.** It is what the word "freeze" usually means and it leaves `BatchNorm`
  running statistics moving. The frozen decoders of the measurements below would then drift under the cross
  reconstruction, and a bit-identity test of the buffers would fail on the real pair model.
- **Eval mode through a hook on `Module.train`.** It survives a `model.train()` at any place. It also changes what a
  module does for every caller, a validation loop included, and a hidden override is harder to reason about than one
  line in `onEpochBegin` that runs after the call it undoes.
- **Take frozen parameters out of the optimizer.** It saves their state. It also gives a checkpoint a layout that
  depends on the freeze, which makes a resume across a release epoch impossible without migrating the optimizer state.
- **Freeze through a learning rate of 0.** The weights would stay where they are (Adam, AdamW and SGD with momentum
  and weight decay all scale their step by the rate, which was checked). Nothing else would: the `BatchNorm`
  statistics keep moving, the gradients are still computed and stored, and they count in the norm that `grad_clip_norm`
  clips to, which changes the step of the parameters that do train. `lr_factor: 0` is refused and points to the
  callback.
- **State in the callback (`frozen: bool`, saved with the checkpoint).** It is more general and it is one more thing to
  keep in step with the epoch. The epoch alone decides, so there is nothing to restore.
- **One `until_epoch` for each pattern.** A callback is one instance per name in `training.callbacks`, a mapping from
  name to arguments, so two patterns with two release epochs would need a second name for the same callback. Not built;
  the case that needs it can ask the registry for a second name, or release in two runs.
- **Per-group `lr` instead of `lr_factor`.** It is explicit, and it breaks when the base rate is changed in one place
  and not in the other, and the schedules multiply the group's own `initial_lr`. A factor keeps the ratio the thing that
  is written down.
- **Regular expressions for module names.** They are more powerful. A dotted name with a `*` is what a person writes
  without looking at a manual, and the error that lists real names is enough to correct a pattern.
- **Silently ignore a pattern that matches nothing.** It is the permissive behaviour of many frameworks. A stage that
  was meant to freeze the decoders and froze nothing gives a plausible curve, and nobody sees it.

## Measurements

The setting is that of ADR 0054, shorter: the linked synthetic pair (the image is a function of the signal's frequency
and phase), the pair model of `configs/model/signal_image_pair.yaml` warm-started from two solo models trained for 15
epochs, 4 batches of 16 pairs, the sum reduction, `beta` 0.1, modality dropout off, `latent_alignment` and
`cross_reconstruction` with a ramp from zero over 100 steps to 0.3, and **30 epochs** of stage 3. Every figure is read on
two validation batches from the posterior means (the squared 2-Wasserstein distance; the sum of squared errors per
sample for the reconstructions). Three seeds, which vary the sampling and the order of the data, not the two solo
checkpoints, which are the same for all; the figure is the mean, and the range of the three is in brackets where it
matters. "Drift" is the relative distance of the parameters of the encoder and the decoder of a side from where the
warm start put them. At the start of stage 3: distance 138.7, signal self reconstruction 139.3, image self
reconstruction 84.5, signal to image cross decoding 103.9, image to signal cross decoding 135.0.

| Policy | Distance | Signal self | Image self | Signal to image | Image to signal | Drift signal / image |
|---|---|---|---|---|---|---|
| Nothing frozen (the paper) | 7.1 (3.2 to 11.1) | 74.1 | 205.9 (56.7 to 385.0) | 54.6 | 145.1 | 0.19 / 0.22 |
| Decoders frozen | 6.0 (4.8 to 8.4) | 125.7 | 79.8 (79.4 to 80.0) | 83.8 | 130.1 | 0.12 / 0.08 |
| Decoders frozen until epoch 15 | 8.2 (5.6 to 11.0) | 79.8 | 121.6 (87.6 to 145.5) | 60.7 | 220.0 (168.3 to 253.1) | 0.18 / 0.17 |
| Signal side frozen (the anchor) | 38.7 (25.0 to 49.7) | 139.3 | 81.0 (58.7 to 92.2) | 70.1 | 132.0 | 0.00 / 0.22 |
| Image side frozen (the anchor) | 27.8 (27.0 to 28.6) | 69.0 | 84.5 | 83.7 | 132.7 | 0.19 / 0.00 |
| Both sides at 0.1 of the rate | 12.7 (12.5 to 13.0) | 121.6 | 73.5 | 75.1 | 128.5 | 0.05 / 0.05 |
| Signal side at 0.1 of the rate | 11.3 (6.7 to 17.9) | 120.6 | 85.1 (54.9 to 116.6) | 59.6 | 128.5 | 0.05 / 0.22 |

What the table says:

- **The paper's setting is the most erratic one for the image.** Its image self reconstruction ends anywhere from 57 to
  385 for a start of 84.5, with the same solo checkpoints and the same loss; two of three seeds end above the start. The
  decoders frozen, and both sides at a tenth of the rate, end within 1 of each other on the three seeds and
  at or below the start. This is the effect the item is for, seen on a toy: a few pairs, an optimizer free to move
  everything, and the solution of a solo stage undone.
- **The protection has a price, in the figures of this toy.** The signal side of the solo model was trained for 15 epochs
  on 4 batches, and is far from converged (139 at the start); the free run keeps training it, to 74, and every policy
  that holds it back ends between 120 and 139. A real solo stage that has converged has less to lose, but this is not
  measured here.
- **A frozen anchor does not align as far in 30 epochs.** With one side frozen the distance ends at 28 to 39, against 7
  for the free run, because only one encoder moves toward the other, and the frozen side's self reconstruction is what
  it was to the bit (139.3 and 84.5, the same on all seeds). The anchor is the point of the option, not a fault, and it is
  also the setting where the distance alone says least about the alignment. The cross decodings are the figures to read.
- **The decoders frozen resolve the P6-4 finding about `BatchNorm`.** The cross decoding runs the decoders in training
  mode, and in the free run their running statistics move; with the decoders frozen they are bit-identical after the
  stage (`tests/integration/test_freeze_and_param_groups.py`, the real `BatchNorm1d` and `BatchNorm2d`).
- **A release in the middle undoes part of it.** Frozen until epoch 15 and free after, the stage ends between the two,
  with the worst image to signal cross decoding of the seven policies (168 to 253). One run per setting does not
  say whether another release epoch does better.
- **A policy for a pair of unequal solo stages is not read from this.** The signal side at a tenth of the rate leaves the
  image at the full one, and the image figure is as erratic as the free run's (55 to 117).

What it does not say: that any of these is the right setting for a real pair. The data are synthetic (finding I-15), the
solo stages are short, three seeds are three seeds, and 30 epochs on 64 pairs is not 72 pairs of the paper. D-14 stays
at "nothing frozen", and the table is there to show that the options work and what each one costs.

## Consequences

- **A run with no `freeze_modules` and no `param_groups` is unchanged, bit for bit.** The optimizer is built as before,
  the groups have no new key, the checkpoint has the same optimizer layout, and a test pins the groups of a default
  trainer.
- **The parameters and the buffers of a frozen module are bit-identical after training.** The unit tests check it on a
  small model with `BatchNorm`, and `tests/integration/test_freeze_and_param_groups.py` checks it on the real pair model,
  for either side and for the decoders, with the cross reconstruction on and with a control that shows the same modules
  change without the callback. Eleven deliberate breakages of the mechanism (no eval mode, no `requires_grad` switch, a
  gradient kept, a release one epoch late, factors not applied, the layout check removed, and others) are each caught by
  at least one test.
- **An unfreeze epoch works across a resume.** Interrupted before, at and after the release epoch and resumed, a run gives
  the loss curve and the weights of the uninterrupted one to a relative `1e-6`, under the guarantee of ADR 0032.
- **The groups reach the optimizer and the checkpoint**, and a Hydra override such as
  `training.param_groups=[{modules:[encoders.signal],lr_factor:0.1,name:signal}]` or
  `+training.callbacks.freeze_modules.modules=[decoders.*]` builds them (`configs/training/default.yaml` and
  `several_latents.yaml` document both).
- **A resume with other groups is an error, not a silent carry-on.** The cost is that a run cannot change its
  `param_groups` at a resume; `Trainer.loadCheckpoint(..., restore_optimizer=False)` loads the weights and starts a new
  optimizer.
- **Limit: one release epoch per run.** All the patterns of one `freeze_modules` share an `until_epoch`.
- **Limit: `torch.compile`.** The callback flips `requires_grad` and the mode of modules between epochs. A compiled model
  guards on both, so the first epoch of each state is expected to recompile the graph once. It was not measured, and
  `compile_model` is off by default (ADR 0031).
- **Limit: both VAEs frozen with only an alignment head trained** needs the adapter of P6-8. Until then the run stops
  with a message that names it (ADR 0060 delivered the adapter, and the message now says to give `latent_alignment`
  `adapters`).
- **The cross reconstruction's `BatchNorm` finding of ADR 0054 has its tool:** freeze the decoders, or all the modules
  whose statistics must not move. It is still a finding for a run that does not.
- **A frozen module keeps its optimizer state.** An Adam state for a parameter that never trains is a few tensors of the
  size of the parameter. It is the price of the checkpoint layout in item 1.
