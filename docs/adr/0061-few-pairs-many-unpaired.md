# 0061: Few pairs, many unpaired samples: a mixed stream and a frozen flow prior

**Status:** accepted (roadmap item P6-8, extension "few pairs, many unpaired samples"; it adds `on_unpaired` to the two pair
loss terms, a batch stream, a flow prior with its term and its fitting script, and changes nothing for a config that has none
of them)
**Date:** 2026-11-18

## Context

The staged recipe (ADR 0051, 0052) trains each modality's VAE on all of its own samples, then aligns the spaces on the
samples that carry every modality. When pairs are scarce (a few dozen) next to thousands of unpaired samples per modality,
two things go wrong in the pair stage as it is:

- a pair stage that sees the pairs only stops seeing the unpaired samples, so each VAE is refined on a few dozen points and
  forgets what its solo stage learned from the rest;
- a pair term refuses a batch that lacks a modality (`needs paired batches`, ADR 0054), so the unpaired batches cannot even
  be put in the stream, and the refusal is deliberate: a skipped term would hide a loader that does not pair its samples.

The roadmap names two ways out: a pair loss **interleaved with unpaired reconstruction batches**, or a **frozen flow prior
fitted on the other space** (Li et al., "Aligning Latent Spaces with Flow Priors",
[arXiv 2506.05240](https://arxiv.org/abs/2506.05240)). They are not alternatives for each other, so both are built, and the
measurement below says what each one does.

## Decision

**1. `on_unpaired` in `latent_alignment` and `cross_reconstruction`.** `"raise"` (the default, the behavior of every
earlier config) or `"skip"`. With `skip`, a batch that lacks a modality of a pair gives a scalar zero (`zeroLoss`, on the model's device, not attached
to the graph: the trainer's total still holds the reconstruction and the regularization of the batch, so the step trains
the rest as it would without the term) and counts in the term's `num_skipped`; `num_applied` counts the others. The default stays `raise`: a skipped term changes the objective
without a trace in the numbers, so a stage turns it on by name, and the message of the refusal says where.
The sample-id check (ADR 0058) runs on the batches that are computed and not on the skipped ones.

**2. `interleaveBatches` (`data/interleave.py`).** The stream that `Trainer.fit` takes in place of a loader:

```python
stream = interleaveBatches(
    paired_loader,                                      # batches with every modality
    {"signal": signal_loader, "image": image_loader},   # batches with one modality each
    unpaired_per_paired=2,
)
trainer.fit(stream, num_epochs=150)
```

An epoch is one pass over the paired loader, and each of its batches is followed by `unpaired_per_paired` unpaired ones
(a fraction is a credit that accumulates: `0.5` is one after every second batch), taken from the unpaired loaders in turn.
The unpaired loaders are **not** restarted at an epoch's start: their position is kept, and a loader that runs out starts
again, so every unpaired sample is seen over the epochs even when one epoch uses a few. The loaders and what they put in a
batch stay the caller's (spec section 11); the stream only orders them.

**3. A frozen flow prior (`losses/priors/flow_prior.py`).**

- `FlowPrior(dim, hidden, layers, time_features, source_dim)` is a velocity network `v(x, t)` over a space of width `dim`,
  with fixed preprocessing in buffers: the mean and one scalar scale of the target points, and, when the target space has
  another width than the space that will be pulled, a **fixed random projection** (the paper's choice) instead of a
  trainable adapter, which would bring back the shrinking of ADR 0060.
- `fitFlowPrior(points, ...)` fits it by flow matching on the target points (`x0 ~ N(0, I)`, `t ~ U[0, 1]`,
  `x_t = (1 - t) x0 + t x1`, the net predicts `x1 - x0`), holds 10 % of the points out when there are at least 100, then
  **freezes** it (`requires_grad_(False)`, gradients dropped). The initialization uses a forked random state, so a fit is
  reproducible from its `seed` and leaves the global generator where it was. It returns the prior with the losses it
  reached.
- The loss of a latent `y` is `E || v((1 - t) x0 + t y, t) - (y - x0) ||^2` with the flow frozen: a latent that looks like a
  target point has the velocity that a target point would have. It needs no likelihood and no ODE solve.
- `saveFlowPrior` / `loadFlowPrior` keep one in a file of tensors and numbers (`torch.load(weights_only=True)`; a `format`
  entry, `global_vae.flow_prior`, and a `version`, so another torch file is refused by name).
- `FlowPrior.alignmentGap(y)` is the loss of `y` minus the loss that the target's own points got (`reference_loss`, stored by
  the fit). The raw value has no meaning alone, since its minimum is not zero: a target sample's velocity is not determined
  by `x_t`.

**4. The term `flow_prior_alignment` (`losses/terms/flow_prior_alignment.py`).**
`FlowPriorAlignmentTerm(space, prior, *, read="mu", num_draws=1, on_missing="raise")`, registered like the others:

```yaml
training:
  loss_terms:
    - name: latent_alignment
      kwargs: {pairs: [[z_signal, z_image]], distance: wasserstein2_diag, on_unpaired: skip}
    - name: flow_prior_alignment
      kwargs:
        space: z_image                 # the space that is pulled
        prior: priors/z_signal.pt      # fitted on the z_signal means of the unpaired signals
        on_missing: skip               # a stream whose batches may not carry the image
      weight: 0.01
```

- It acts on every batch whose forward pass reached `space`, paired or not. `on_missing: raise` (default) refuses a batch
  that did not, `skip` returns the zero loss.
- `read: mu` pulls the posterior mean (the default: it is what the prior was fitted on and what an evaluation reads),
  `sample` the reparameterized sample of the step.
- The prior is a **submodule of the term** with frozen parameters, so it is in `loss_terms_state_dict` (ADR 0047) and a
  resumed run uses the prior it began with, whatever the file now holds. The optimizer holds the parameters and never moves
  them.
- `validateModel` refuses an unknown space and a prior whose width is not the space's, with the `--dim` advice.

**5. Fitting it (`scripts/fit_flow_prior.py`, `evaluation/space_means.py`).** `collectSpaceMeans(model, view, dataloader)`
encodes the batches that carry a view's inputs, skips the others, and returns the posterior means of its space (it refuses an
empty result). The script loads a checkpoint, takes a model factory and a dataloader factory as `evaluate.py` does, fits the
prior to one space and writes it with a `<output>.json` of what the fit did (`--space`, `--dim`, `--steps`, `--hidden`,
`--layers`, `--seed`, `--max-samples`, `--trust-checkpoint`, ...). A missing space, a width that cannot be projected and an
empty loader stop the script with a message and no traceback.

## Measurement

`tests/integration/test_few_pairs_measurement.py` reproduces it (60 seconds, slow marker). Two views of one 3-dimensional
factor `u` (`signal = u W`, `image = tanh(u V)`, a little noise), two spaces of width 3, `beta = 1e-3`. **16 pairs**, two
pools of **2000 unpaired samples** (one per modality, none of them a pair), 512 held-out pairs, three seeds. Each VAE is
first trained 6 epochs on its own pool, then the pair stage runs 150 epochs: **A** the pair term on the 16 pairs alone;
**B** the same with two unpaired batches after each paired one; **C** B plus `flow_prior_alignment` on `z_image` with a prior
fitted on the `z_signal` means of the unpaired signals; **D** no pair term, the unpaired batches and the prior only.
Retrieval is recall@1 of the true partner among 512 (mean of both directions, chance about 0.002), `signal MSE` is the
reconstruction error of the held-out signals from their own posterior means, `spread` is the variance over samples of the
`z_image` posterior mean, averaged over dimensions, and `gap` is `alignmentGap` of the held-out `z_image` means:

| run | recall@1 per seed | signal MSE per seed | `z_image` spread per seed | gap |
|---|---|---|---|---|
| A: 16 pairs only | 0.040, 0.044, 0.095 | 0.51, 0.42, 0.13 | 1.3, 2.0, 1.6 | |
| B: + unpaired batches | 0.081, 0.023, 0.093 | 0.17, 0.24, 0.09 | 1.2, 1.8, 1.8 | |
| C: B + flow prior, weight 1.0 | 0.030, 0.021, 0.031 | 0.19, 0.23, 0.10 | 0.27, 0.18, 0.27 | -1.8, -2.0, -1.8 |
| D: prior only, no pair term, weight 0.1 | 0.003, 0.000, 0.001 | 0.09, 0.10, 0.05 | 0.18, 0.18, 0.21 | -1.8, -2.1, -1.8 |

A second world where `u` is a mixture of 6 separated clusters (`cluster accuracy` is the share of held-out images whose
nearest signal posterior mean is of the same cluster, chance 1/6; the retrieval of one sample among 512 is then 0.01 to 0.04
for every run, and says little):

| run | cluster accuracy per seed | `z_image` spread per seed | gap |
|---|---|---|---|
| B: 16 pairs + unpaired batches | 0.898, 0.990, 0.920 | 1.8, 2.2, 2.0 | |
| C: B + flow prior, weight 0.01 | 0.895, 0.988, 0.920 | 1.7, 2.2, 1.9 | 1.71, -0.03, -0.16 |
| C: B + flow prior, weight 1.0 | 0.723, 0.943, 0.900 | 1.8, 0.9, 0.8 | 0.13, -0.55, -0.91 |

A wider scratch run of the same script (not in the repository; the weights 0.01 and 0.1, and 64 pairs instead of 16) gave the
same picture: at 16 pairs the mean recall@1 was 0.060 (A), 0.066 (B), 0.066 / 0.051 / 0.027 (B with the prior at weight
0.01 / 0.1 / 1.0); at 64 pairs 0.115 (A), 0.144 (B), 0.139 / 0.111 / 0.044; on the clustered world the mean cluster accuracy
at 16 pairs was 0.928 (A), 0.936 (B), 0.934 / 0.932 / 0.855 (prior at 0.01 / 0.1 / 1.0).

What it says, on a toy problem and three seeds (roadmap I-15, no real data):

- **The pairs alone align the spaces** far above chance (0.04 to 0.095 against 0.002), and the unpaired batches **cut the
  reconstruction error of the held-out signals** to about half (mean 0.35 to 0.17) at a retrieval that is not lower (mean
  0.060 to 0.066, a difference smaller than the spread between seeds). That is the case for the stream: each VAE keeps seeing
  all of its own data in the stage that adjusts it to the pairs.
- **A distribution term alone does not find the pairing**: D retrieves at chance (0.000 to 0.003). A cloud has symmetries
  (a rotation of it is the same cloud) and the prior has no way to tell which latent goes with which. Pairs are what break
  them; the prior can only say where latents may lie.
- **The flow prior shrinks the space it moves.** The loss is a surrogate, not a likelihood: a cloud concentrated toward the
  middle of the target scores *below* the target's own points (the gap is negative in every run with the prior on this world:
  -1.8 to -2.1). The spread of `z_image` falls from 1.2 to 1.8 to 0.18 to 0.27, and **the active-unit guard does
  not flag it** (three active units in every run, no collapse flag). The cost grows with the weight (the wider run): mean recall@1
  0.066, 0.051, 0.027 at 0.01, 0.1, 1.0 and, on clustered data, the cluster accuracy fell from 0.936 to 0.855 at weight
  1.0. On the clustered world the gap is not always negative (-0.9 to +1.7 across seeds and weights), so the sign is a
  property of the world and not a rule.
- **At a small weight it changes nothing measurable here.** At 0.01 the recall and cluster accuracy equal those of B within
  noise. The flow prior did **not** improve retrieval or cluster accuracy in any world or setting that was tried, the
  clustered world (a target with structure) included. What the paper's setting has and these worlds lack, features of a
  pretrained model with many unpaired samples, was not tried, so this is a cost measured on toy data and not a verdict on
  the idea.

## Alternatives considered

- **Skip unpaired batches silently, always.** Rejected: it hides a loader that does not pair its samples, which is the
  commonest way for stage 3 to train on strangers. `skip` is a named choice of the stage.
- **A second trainer for the unpaired batches, alternating with the pair trainer.** Rejected: two optimizers over the same
  modules, two checkpoints, two schedules. One stream through one trainer gives one objective, one `loss_terms_state_dict`
  and one resume.
- **A pair-aware sampler inside the data loader.** Rejected: loaders are the caller's (spec section 11). The stream only
  orders the loaders the caller already has.
- **The flow prior trained jointly with the latent space.** Rejected: a trained prior can follow the latent to wherever the
  latent drifts, and the paper's point is a **frozen** one that the latent has to move toward.
- **A trainable adapter between spaces of different widths for the prior.** Rejected: ADR 0060 measured two trainable maps
  shrinking their output. The fixed random projection of the paper cannot.
- **Fitting the prior from the data, not from the model.** Rejected: the prior has to be over the space's own posterior
  means, which only the model gives. `collectSpaceMeans` and the script do that and nothing else.
- **A likelihood or an ODE solve for the prior.** Rejected: the loss is the paper's tractable surrogate, and its caveat is
  stated and measured instead.

## Consequences

- No change for a config without `on_unpaired`, `interleaveBatches` or `flow_prior_alignment`: the terms, their state and
  their messages are the same.
- A project with few pairs can keep both solo VAEs learning from all of their data during the pair stage, with `on_unpaired:
  skip` on the pair terms and `interleaveBatches` over the loaders it already has. This is the recommended first step.
- The flow prior is available, with its caveat. Start at a small weight (0.01), read the `z_image` spread against the stream
  without it, the retrieval and the gap, and do not trust the active-unit guard for what it does. A weight that moves the
  spread is a weight that moves the space.
- `on_unpaired: skip` and `on_missing: skip` mean that a batch can run with no term active. `num_applied` and `num_skipped`
  of each term say how many it was.
- Limits: the stream interleaves at one fixed ratio and does not weight by modality; the prior is fitted once and does not
  follow a solo space that keeps moving; random projection of a wider target loses the directions that the projection drops;
  the measurements are those of two toy worlds and three seeds, which show costs and not a gain for the prior.
- Not done here: the **multi-way loss** (P7-4) and the correction of Achara et al. (GCPA), as in ADR 0059.

Tests: `tests/unit/test_pair_loss_terms.py` (`on_unpaired` of both terms, the zero loss, the counters, the config route),
`tests/unit/test_interleave.py` (the stream: the ratio, the credit, the restart, the errors), `tests/unit/test_flow_prior.py`
(the network, the fit and its determinism, the loss, the gap, the file and its refusals), `tests/unit/test_flow_prior_alignment.py`
(the term, its reads, its state and resume, its refusals), `tests/unit/test_space_means.py`,
`tests/integration/test_fit_flow_prior_script.py` (the script, its files and each refusal) and
`tests/integration/test_few_pairs_measurement.py` (the tables above, as inequalities).
