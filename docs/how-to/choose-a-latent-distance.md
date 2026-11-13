# Choose a latent distance

Stage 3 of the [staged recipe](train-in-stages.md) trains two modalities, each with its own latent
space, and asks the two spaces to agree on the same sample. "Agree" needs a distance. This page says
what the six that ship compute, what each one ignores, which to try first, and how to call one. The
reasoning is in [ADR 0053](../adr/0053-latent-distance-registry.md) and in
[Phase 6 of the roadmap](../roadmap.md).

!!! note "Which distance stage 3 trains with is still open"
    The loss term `latent_alignment` (roadmap P6-4, [ADR 0054](../adr/0054-pair-stage-loss-terms.md))
    applies one of these distances to the pairs of latent spaces you list, and it has **no default
    distance**: you name one in `distance`, so the choice is in your config and not in the code. Which
    one to train with is roadmap decision D-13, which stays open until the measurements of P6-6. The
    page holds no number from a real run. [Train in stages](train-in-stages.md#pull-the-two-latent-spaces-together)
    shows the term in a stage 3 config.

**On this page**

- [Which to try first](#which-to-try-first)
- [The six distances](#the-six-distances)
- [Call one](#call-one)
- [The scale](#the-scale)
- [What each one needs to know about](#what-each-one-needs-to-know-about)
- [What a distance refuses](#what-a-distance-refuses)
- [Add your own](#add-your-own)

## Which to try first

Start with **`wasserstein2_diag`**, in its default squared form, and keep **`mean_mse`** as the
ablation.

- `wasserstein2_diag` compares the two posteriors, not only their means, in closed form, with a
  gradient everywhere. It is the form the CADA-VAE line of work uses for the same job, the
  alignment of the latent spaces of two modalities.
- `mean_mse` is the simplest thing that could work: the squared gap of the two posterior means. If
  it does as well as `wasserstein2_diag` on your pair, the variances were not what mattered, and
  you have one fewer thing to explain.

That order is a starting point and not a result: no distance has been measured on a real pair yet.
Roadmap P6-6 compares all six on held-out pairs, and D-13 is settled from that.

## The six distances

| Name | What it compares | Closed form | Blind to | Symmetric |
|---|---|---|---|---|
| `mean_mse` | the posterior means | yes | the variances | yes |
| `wasserstein2_diag` | the two diagonal Gaussians | yes | nothing | yes |
| `symmetric_kl` | the two diagonal Gaussians | yes | nothing, but explodes if one variance is tiny | yes |
| `cosine` | the direction of the means | yes | the scale of the means, and the variances | yes |
| `mixture_jsd` | each posterior against the mixture of all of them | no, Monte Carlo | nothing | yes, in expectation |
| `info_nce` | each sample against the other samples of the batch | no | the variances | yes |

Every distance takes the `(mu, logvar)` pair of each space for one batch, and returns **one value
per sample**, shape `(batch,)`, in float32 or float64. The two spaces describe the same samples row
by row: row `i` of one is the posterior that modality gives to the sample that row `i` of the other
describes.

| Name | Options (defaults) |
|---|---|
| `mean_mse` | `dimension_reduction` (`"sum"`) |
| `wasserstein2_diag` | `squared` (`True`), `dimension_reduction` (`"sum"`) |
| `symmetric_kl` | `dimension_reduction` (`"sum"`), `min_logvar` (`-10.0`), `max_logvar` (`10.0`) |
| `cosine` | `eps` (`1e-8`) |
| `mixture_jsd` | `num_samples` (`1`), `dimension_reduction` (`"sum"`), `sum_over_spaces` (`False`), `min_logvar` (`-10.0`), `max_logvar` (`10.0`) |
| `info_nce` | `temperature` (`0.1`), `eps` (`1e-8`) |

## Call one

The registry has the three functions every registry in the framework has. They live in
`global_vae.losses.distances.registry`, and importing `global_vae.losses.distances` registers the
six.

```python
import torch

import global_vae.losses.distances  # noqa: F401
from global_vae.losses.distances.registry import (
    getLatentDistanceClass,
    listRegisteredLatentDistances,
)

print(listRegisteredLatentDistances())
# ['cosine', 'info_nce', 'mean_mse', 'mixture_jsd', 'symmetric_kl', 'wasserstein2_diag']

distance = getLatentDistanceClass("wasserstein2_diag")()
z_signal = (torch.zeros(8, 16), torch.zeros(8, 16))  # (mu, logvar) of the signal space
z_image = (torch.ones(8, 16), torch.zeros(8, 16))  # (mu, logvar) of the image space

per_sample = distance(z_signal, z_image)  # shape (8,), 16.0 for every sample
loss = per_sample.mean()  # the batch average, taken by the caller
```

A distance is an `nn.Module` with no parameters and no state, so `.to(device)` is harmless and it
adds nothing to a checkpoint. It returns the per-sample vector and leaves the batch average to the
caller, as the regularizers do.

To read the posterior of a latent space from a model, use the same `(mu, logvar)` the model
exposes to its regularizers; [Configure encoders, decoders and latent
spaces](configure-latent-spaces.md) says where they come from.

## The scale

`mean_mse`, `wasserstein2_diag` and `symmetric_kl` sum over the latent dimensions by default, and
the result is averaged over the batch by the caller. That is the footing of the KL term
([ADR 0029](../adr/0029-loss-scale-convention.md)), so a weight `lambda` on a distance reads next to
`beta`: a `lambda` of 1 and a `beta` of 1 weight a unit of distance and a nat of KL the same way.

`dimension_reduction="mean"` divides by the number of latent dimensions instead, which makes the
value, and so the right `lambda`, independent of the width of the space. It exists for a sweep over
widths. `cosine` and `info_nce` have no such option: their value does not grow with the width, since
they act on the direction of a vector. `mixture_jsd` is the divergence of the whole posterior
and not a sum of per-dimension ones, so it does not grow with the width either; its
`dimension_reduction="mean"` divides that value by the width anyway, for the same sweep.

The bounds that a reader needs to put `lambda` in context:

- `cosine` is in `[0, 2]`: 0 for the same direction, 1 for orthogonal means, 2 for opposite ones.
- `mixture_jsd` is at most `log(K)` for `K` spaces (`log 2` for a pair, in nats, per sample,
  whatever the width), and `K` times that with `sum_over_spaces`.
- `info_nce` is strictly positive; with nothing to tell the samples of a batch of `N` apart it is
  `log(N)`, and it falls toward 0 as matched pairs become distinguishable from the rest.
- `mean_mse`, `wasserstein2_diag` and `symmetric_kl` have no upper bound.

## What each one needs to know about

**`wasserstein2_diag`.** The value is the squared gap of the means plus the squared gap of the
standard deviations, summed over the dimensions. The paper this recipe follows takes the square root
of that. The root has no gradient at zero (it is a `0/0`, a `nan` in autograd), which is exactly where a
trained pair is heading, so the default is the squared form and the root is `squared=False`. The root is taken after
the dimensions are combined, and where the distance is exactly 0 the gradient is 0 and finite and
not `nan`.

**`mean_mse`.** It ignores the variances, so nothing stops the two posteriors from agreeing on the
mean and disagreeing on how sure they are. If you see that happen, move to `wasserstein2_diag`.

**`symmetric_kl`.** `KL(a||b) + KL(b||a)` divides by the variance of each side, so a posterior with
a tiny variance makes it explode. The two log-variances are clamped to `[min_logvar, max_logvar]`
(`[-10, 10]` by default) before the division, and a clamped log-variance gets no gradient, which is
the price of a value that stays finite. Widen the range only if you know the encoders stay inside it.

**`cosine`.** It compares directions, so it is blind to the scale of the means: a space that is a
constant multiple of the other has distance 0. That is the point when the two encoders have no reason
to agree on a scale, and a defect when they should. A mean of exactly zero has no direction and is
counted as orthogonal to everything, at distance 1.

**`mixture_jsd`.** Each posterior is compared with the mixture of the posteriors of all the spaces,
and the result is their average, the Jensen-Shannon divergence. The rate term of MMVM (Sutter et al.,
NeurIPS 2024) is `M` times this divergence; `sum_over_spaces=True` returns that form. It takes more
than two spaces: `distance(a, b, c)`. There is no closed form, so the value is an unbiased
Monte Carlo estimate from `num_samples` draws per sample and per posterior, and:

- it draws from the global generator, so seed it (`torch.manual_seed`) when you want a reproducible
  value, and a run that must be reproducible already does;
- one value can fall slightly below 0 even though the divergence cannot, and the batch mean is what
  to read;
- more draws lower the variance and cost memory in proportion.

**`info_nce`.** The only distance in which a sample's value depends on the other samples of the
batch: the similarity of a matched pair is scored against the similarity of that sample to every
other row of the other space, in both directions, averaged. It needs at least two samples per batch
and refuses one, since a batch of one has no negatives. A small `temperature` makes the negatives
count more; start from the default `0.1` and sweep it. It is not zero on identical inputs, by design.
It is the objective that [roadmap item I-9](../roadmap.md) found missing: the one that asks not only
for two spaces to be close, but for each sample to be close to its own partner and far from the
others.

## What a distance refuses

A distance checks its inputs and says what it found, rather than broadcasting a mismatch into a
number.

- **Spaces of different widths.** The two latent spaces must have the same `dim` (roadmap D-16), and
  the message names both shapes and says that the widths must be equal. Equal widths are a choice
  of the pair model file, and `configs/model/signal_image_pair.yaml` makes it.
- **Different batch sizes**, a `logvar` of another shape than its `mu`, tensors that are not
  `(batch, dim)`, or that are not floating point.
- **More spaces than it compares.** A distance of two spaces refuses a third; only `mixture_jsd`
  takes any number.
- **Too small a batch**, for `info_nce` (fewer than 2).
- **A bad option**: a `dimension_reduction` that is neither `"sum"` nor `"mean"`, a `temperature` or
  `eps` that is not positive, a `num_samples` that is not an integer of at least 1, a clamp range
  that is empty.

## Precision

A distance runs outside the autocast context and in float32 at least: half precision inputs
(`float16`, `bfloat16`) are cast up, `float32` and `float64` are kept, and the value is the same
inside and outside `torch.autocast`. `info_nce` is a matrix product, which autocast would otherwise
run in half precision. The gradient reaches the inputs in their own dtype. This is
[ADR 0031](../adr/0031-mixed-precision-gradient-accumulation-and-compile.md) applied to a new place.

## Add your own

A distance follows the same three steps as every other strategy in
[Add a fusion, latent head, ...](add-a-strategy.md): subclass, decorate, import from the package.
The base class does the checking and the cast; you write the value.

```python
from collections.abc import Sequence

import torch

from global_vae.losses.distances.base import AbstractLatentDistance, LatentParams
from global_vae.losses.distances.registry import registerLatentDistance


@registerLatentDistance("mean_l1")
class MeanL1Distance(AbstractLatentDistance):
    """The absolute gap of the posterior means, summed over the dimensions."""

    def computeDistance(self, posteriors: Sequence[LatentParams]) -> torch.Tensor:
        (mu_a, _), (mu_b, _) = posteriors
        return (mu_a - mu_b).abs().sum(dim=-1)
```

`computeDistance` receives the posteriors already validated (a list of `(mu, logvar)` pairs, all of
shape `(batch, dim)`, floating point, in at least float32) and returns a tensor of shape `(batch,)`.
Declare in class attributes what is true of yours, so that the tests and the loss term know it:

| Attribute | Default | Set it to |
|---|---|---|
| `is_symmetric` | `True` | `False` if `d(a, b) != d(b, a)` |
| `is_zero_on_identical` | `True` | `False` if identical inputs can give a positive value (a contrastive loss) |
| `is_stochastic` | `False` | `True` if the value is an estimate drawn at random |
| `max_spaces` | `2` | the number of spaces it compares, or `None` for any number |
| `min_batch_size` | `1` | the smallest batch it can score |

Then import the module in `losses/distances/__init__.py`. The registry contract test
(`tests/unit/test_registry_contract.py`) picks the class up from there; the properties that the six
share (non-negative, zero on identical inputs, symmetric where declared, finite gradients at 0,
float32 under mixed precision) are in `tests/unit/test_latent_distances.py`, and a new class can join
the lists at the top of that file.
