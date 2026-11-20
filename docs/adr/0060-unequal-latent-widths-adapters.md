# 0060: Unequal latent widths: adapters read by the alignment term only

**Status:** accepted (roadmap item P6-8, extension "unequal latent widths"; it adds `adapters` to `latent_alignment`, a
reader for the adapters in the evaluation and one script flag, and changes nothing for a config that has none)
**Date:** 2026-11-17

## Context

D-16 says that the two spaces of a pair have the same width: a distance compares two posteriors coordinate by
coordinate, and a cross decoding hands the sample of one space to the decoder of the other. Two models that were each
trained at the width that suits their modality (16 for a spectrum, 8 for an image) cannot then be aligned without
retraining one of them at the other's width, which throws away what a solo stage settled.

The roadmap asks for an adapter head used by the alignment term only, and notes that a `linear` head already exists in
`heads/`. The question is what "by the alignment term only" has to mean for the rest of the framework.

## Decision

**1. `adapters` in `latent_alignment`.** The term takes `adapters`: latent space name -> a width (a `linear` head from the
space's width to it) or `{name, out_dim, kwargs}` (another registered head). The term reads the posterior `(mu, logvar)`
of that space through the head and compares the adapted one. The spaces, their encoders and their decoders keep the
widths they have, and nothing else reads an adapter: not the decoders, not the cross reconstruction, not the evaluation
of the model alone.

```yaml
training:
  loss_terms:
    - name: latent_alignment
      kwargs:
        pairs: [[z_signal, z_image]]        # widths 16 and 8
        distance: wasserstein2_diag
        stop_gradient: target               # z_signal stays; the adapter and z_image move
        adapters:
          z_image: 16                       # linear, 8 -> 16
```

- **Built when the term is checked against the model.** The head's input width is the width of the space it reads, which
  is the model's to say, so the adapters are built in `validateModel` (which the trainer calls before it builds its
  optimizer), on the model's device. A name that is not a latent space of the model, a head that is not registered or one
  that refuses its widths (the identity head, when they differ) is refused there, by name.
- **A parameter of the term.** The adapter goes in the trainer's optimizer, in `loss_terms_state_dict` under
  `<term>.adapters.<space>` and back when a run resumes (ADR 0047). A resume with other adapters is refused by the
  existing check of the loss-term layout.
- **The widths compared are the adapted ones.** `validatePairsAgainstModel` takes the width of the adapter's output for a
  space that has one, so a pair whose adapters leave two widths is refused with the advice to set the `out_dim` of the
  adapters. `cross_reconstruction` takes no adapters: it still refuses two widths, and says that no adapter of
  `latent_alignment` can change that.
- **The gradient is cut before the adapter.** With `stop_gradient: target` the target space does not move, and its own
  adapter, if it has one, still learns. An adapter in a pair is therefore always trained by the term.
- **Only spaces of a pair.** An adapter that names a space that is in no pair is refused.

**2. Adapt one space toward the other.** Two trainable adapters have a trivial way to make a distance small: shrink both
outputs. The config documents the recommendation (one adapter on one space of a pair, `stop_gradient` on the other), and
the measurement below is why the recommendation is not left as a matter of taste.

**3. The evaluation reads the adapters back.** `evaluateLatentAlignment(adapters=...)` takes the heads and
`scripts/evaluate.py --latent-pair A B --alignment-adapter SPACE=[HEAD:]WIDTH` (repeatable, `--alignment-term` for the
term's name, `latent_alignment` by default) builds them from the checkpoint's loss-term state
(`readLossTermState` in `training/checkpoint.py`, `loadAdapters` in `losses/terms/adapters.py`); the head and the width
are the ones of the training config, and a width that the checkpoint does not hold is refused with what the state lacks.
With adapters:

- the retrieval, the alignment metrics and the distances read the **adapted** posteriors;
- the collapse guard reads the posteriors as the model has them and the adapted ones as `<view> (adapted)`;
- the Procrustes baseline and the pair-integrity check read the **raw** posteriors, which are what the model has: for two
  widths the orthogonal map is a partial isometry (a map with orthonormal rows or columns) that embeds the narrower space
  in the wider one;
- the **cross reconstruction** is not computed when the raw widths differ, since it hands a code of one width to the
  decoders of the other, and the report says so in its `notes`, with the `widths`, the `adapters` it read and, for each,
  its `spread_ratio`;
- a pair that is read without adapters and has two widths is refused, with the way out in the message.

**4. A measure that the active-unit guard does not give.** The `spread_ratio` of an adapter is the per-dimension spread
of the posterior mean after the adapter over before it (mean over dimensions of the variance over samples of each
dimension of `mu`, so that a narrower and a wider space compare). The report flags an adapter below
`min_adapter_spread` (0.1 by default, `--min-adapter-spread`), as "possibly shrinking its output to make the distance
small: read the retrieval first".

## Measurement

The test `tests/integration/test_adapter_measurement.py` reproduces it (13 seconds). Two views of one 3-dimensional
factor `u`: `signal = u W` and `text = tanh(u V)` (8 features each, a little noise), a `signal` space of width 3 and a
`text` space of width 5, a small KL weight (`beta = 1e-3`: with the default one the VAEs of such a toy problem collapse on
their own, which an earlier attempt showed and which says nothing about adapters), 400 steps, three seeds. Retrieval
recall@1 on 384 held-out pairs (mean of both directions; chance is 1/384, about 0.003); `spread` is the `spread_ratio` of
each adapter:

| run | recall@1 per seed | spread of each adapter, per seed |
|---|---|---|
| control: widths 3 and 3, no alignment term | 0.000, 0.007, 0.003 | none |
| one adapter on `z_text` (5 to 3), `stop_gradient: target` | 0.366, 0.473, 0.224 | 0.63, 1.12, 0.40 |
| one adapter on `z_text`, no stop gradient | 0.263, 0.499, 0.547 | 0.49, 1.00, 0.98 |
| two adapters (3 to 4, 5 to 4), no stop gradient | 0.093, 0.065, 0.053 | 0.03 / 0.01, 0.06 / 0.06, 0.07 / 0.05 |
| two adapters (3 to 4, 5 to 4), `stop_gradient: target` | 0.047, 0.211, 0.336 | 0.03 / 0.06, 0.05 / 0.10, 0.20 / 0.23 |

What it says, on a toy problem and three seeds (roadmap I-15, no real data):

- One adapter aligns the spaces: retrieval far above the control, with the spread of what it reads kept.
- Two trainable adapters retrieve little and shrink their output by one to two orders of magnitude. **The active-unit
  guard does not flag them** (no adapted space has fewer than one active unit), and the distance of mismatched pairs falls
  from 3 to 12 (one adapter) to 0.3 to 1.0, since a small scale makes every distance small. The retrieval, which does not
  change with the scale, and the spread ratio are what show it.
- A stop gradient on one space with two adapters sits in between and varies by seed. The floor of 0.1 flags every
  adapter of the second-to-last row and three of the six of the last: it is where to start reading, not a verdict.

## Alternatives considered

- **Project both spaces into a shared width inside the model.** Rejected: it changes the model that the other stages
  use, and the point is that the spaces keep the widths of their solo stages.
- **A `linear` adapter that is also read by the decoders.** Rejected: "by the alignment term only" is the roadmap's
  condition. A decoder that read the adapted code would no longer be the decoder of the solo stage.
- **Orthogonal adapters.** Rejected for now: they need equal widths, and the case here is that they differ. A head
  registered under another name can be given as `{name, out_dim, kwargs}` without a change to the term.
- **Freezing the adapter's output scale, or normalizing the adapted posterior.** Rejected: it hides the shrinkage that
  the measurement is about, and a normalized `logvar` changes what the distance means. The recommendation (one adapter,
  a stop gradient) and the measure are the response.
- **A cross reconstruction through the adapters.** Rejected: an adapter goes one way (wide to narrow, or the reverse),
  and a decoder needs the code of its own space, which an adapter cannot give back.

## Consequences

- No change for a config without `adapters`: the term, its state and its error messages are the same.
- A project with two solo models of different widths can run stage 3 without retraining one of them, with the adapter as
  the only new trained module, and read the result with the same report as before.
- The spaces stay unequal after the alignment: whatever needs equal widths (a cross decoding, a shared space) still does,
  and says so.
- Limits: a linear adapter can only align what a linear map of the posterior aligns; the retrieval of a toy problem is not
  the retrieval of real data; the Procrustes baseline between two widths is a partial isometry and reads the posterior
  means only; the spread floor is a heuristic taken from three seeds.

Tests: `tests/unit/test_alignment_adapters.py` (parsing, building, the term's gradients and state, the checkpoint
round trip, the widths check), `tests/unit/test_latent_alignment_adapters.py` (rectangular Procrustes, `adaptPosteriors`,
the report with adapters, the spread ratio and its flag), `tests/integration/test_adapter_script.py` (the flags, their
files and each refusal) and `tests/integration/test_adapter_measurement.py` (the table above, as inequalities).
