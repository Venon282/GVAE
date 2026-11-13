# 0053: Latent distance registry

**Status:** accepted (roadmap item P6-3; it adds a package and a registry that nothing calls yet, the loss terms of P6-4 will,
and changes no number that any run produced)
**Date:** 2026-11-11

No number from a real run appears in this ADR. Which distance stage 3 trains with is roadmap decision D-13, which stays open
until the measurements of P6-6; this ADR makes the candidates available and says which to try first.

## Context

Stage 3 of the staged recipe (roadmap Phase 6) minimizes "the distance between the latent spaces" of two modalities. The
text of PairVAE read for the roadmap does not name its distance, and several are known, cheap and different in what they
ignore: the variances, the scale, the direction, the other samples of the batch. A registry is how the framework holds a family
of interchangeable strategies (spec §10), and the loss term of P6-4 (`latent_alignment`) should name its distance in YAML
the way a latent space names its regularizer.

What the distance acts on is already fixed by the roadmap. A latent space is described, for a batch, by the pair
`(mu, logvar)` of its posterior (the convention of encoders, fusion and regularizers), and a distance compares the posteriors
that two modalities give to the **same sample**, not two independent draws. Three constraints come from earlier decisions:

- **The scale of ADR 0029.** The regularization term is a sum over the latent dimensions, averaged over the batch, times
  `beta`. A distance that is a sum over the dimensions has the same footing, so that its weight `lambda` reads next to `beta`.
- **Mixed precision (ADR 0031).** A loss term runs outside the autocast context, but a distance is also called by hand, in an
  evaluation, in a notebook, and `info_nce` is a matrix product, which autocast would run in half precision.
- **Equal widths (D-16).** Stage 3 requires the two spaces of a pair to be as wide as each other.

## Decision

**1. A package `losses/distances/`, the thirteenth registry.** It holds `AbstractLatentDistance` (`base.py`), the registry
(`registry.py`: `registerLatentDistance`, `getLatentDistanceClass`, `listRegisteredLatentDistances`, the three functions of
every registry, so `tests/unit/test_registry_contract.py` covers it), a small module of shared numerics (`numerics.py`) and
one file per distance. `losses/distances/__init__.py` imports every built-in so that the decorators run (spec §10).

**2. The contract.** `AbstractLatentDistance` is an `nn.Module` (a distance may own parameters, such as a learned temperature,
and the trainer then holds them like those of a regularizer). Its `forward(first, second, *others)` takes the posterior
`(mu, logvar)` of each latent space, every tensor of shape `(batch, dim)`, and returns **one value per sample**, shape
`(batch,)`, in `float32` or wider. It does not average over the batch: the term that calls it does, as the regularizers leave
that to `computeTotalRegularizationLoss`.

`forward` is written once, in the base class, and does what is common to every distance: it checks the inputs (a pair of
tensors per space, `logvar` of the shape of `mu`, equal shapes across spaces, at least `min_batch_size` samples, a count of
spaces the class accepts), casts half precision to `float32` (a wider type is kept), switches autocast off around the
computation, and checks that what came back has shape `(batch,)`. A subclass implements `computeDistance(posteriors)` on
checked `float32` tensors. The five rules are therefore not repeated six times, and a seventh distance cannot forget one.

Five class-level declarations say what a caller may assume, read through the registry without building anything (the pattern
of `requires_matching_dims` on assemblers and of ADR 0048):

| Declaration | Default | Meaning |
|---|---|---|
| `is_symmetric` | `True` | Swapping the two spaces changes nothing. A directional distance (a KL to a frozen reference, P7-5) sets `False`. |
| `is_zero_on_identical` | `True` | Two equal posteriors give 0. `info_nce` sets `False`: its minimum is above 0. |
| `is_stochastic` | `False` | The value is a Monte Carlo estimate that draws random numbers. Only `mixture_jsd`. |
| `max_spaces` | `2` | How many spaces one call compares. `None` means any number of at least two (`mixture_jsd`). |
| `min_batch_size` | `1` | The fewest samples a call needs. `info_nce` needs 2. |

**3. Six built-ins, one class per file.**

| Name | Value per sample, `a` and `b` the two posteriors | Closed form | Options |
|---|---|---|---|
| `mean_mse` | `sum_d (mu_a - mu_b)^2` | yes | `dimension_reduction` |
| `wasserstein2_diag` | `sum_d [(mu_a - mu_b)^2 + (sigma_a - sigma_b)^2]`, `sigma = exp(logvar / 2)`; the square root of it when `squared: false` | yes | `dimension_reduction`, `squared` (default `true`) |
| `symmetric_kl` | `KL(a, b) + KL(b, a) = 1/2 sum_d [(var_a + (mu_a - mu_b)^2) / var_b + (var_b + (mu_a - mu_b)^2) / var_a - 2]` | yes | `dimension_reduction`, `min_logvar`, `max_logvar` |
| `cosine` | `1 - cos(mu_a, mu_b)`, in `[0, 2]` | yes | `eps` |
| `mixture_jsd` | `1/K sum_k KL(q_k, m)` with `m = 1/K sum_j q_j`, the Jensen-Shannon divergence of the `K` posteriors | no: Monte Carlo | `dimension_reduction`, `num_samples`, `sum_over_spaces`, `min_logvar`, `max_logvar` |
| `info_nce` | `1/2 (CE(row i) + CE(column i))` of the matrix `cos(mu_a_i, mu_b_j) / temperature` | no | `temperature` |

The choices behind the rows:

- `mean_mse` ignores the variances: it is the simplest distance and the ablation of the others.
- `wasserstein2_diag` is the closed-form 2-Wasserstein distance between two diagonal Gaussians, the distribution alignment of
  CADA-VAE. The published form takes the square root, which has an infinite derivative at zero, so the squared form is the
  default (D-13) and `squared: false` gives the root. The root is computed with a safe square root (the derivative is taken as
  0 at 0), so that its gradient is finite on identical inputs as well. The 2-Wasserstein distance between `N(0, 1^2)` and
  `N(1, 2^2)` is `sqrt(2)`, its square is 2, and the tests pin both.
- `symmetric_kl` is the **sum** of the two directions (the Jeffreys divergence), not their average: the logarithms of the
  variances cancel, which leaves the form above. It explodes when one variance is tiny (`1 / var`), so both log-variances are
  clamped to `[min_logvar, max_logvar]` (default `[-10, 10]`); a clamped value has no gradient, which is the intended effect
  on a variance that is out of range.
- `cosine` compares the direction of the means and is blind to their scale and to the variances. A zero mean has no
  direction and counts as orthogonal to everything, distance 1, so two identical zero means are **not** at distance 0; the
  distance is defined with `max(norm, eps)` so that its gradient is finite there (and large: it is the usual convention).
- `mixture_jsd` is the Jensen-Shannon divergence of `K` diagonal Gaussians, each posterior against the mixture of all of
  them, which has no closed form. It is estimated with `num_samples` reparameterized draws from each posterior:
  `KL(q_k, m) = E_{z ~ q_k}[log K - logsumexp_j (log q_j(z) - log q_k(z))]`, so the gradient flows through both the samples
  and the densities. The rate term of MMVM (Sutter et al., NeurIPS 2024) is `M` times this divergence, and
  `sum_over_spaces: true` returns that form (`sum_k KL(q_k, m)`). It accepts any number of spaces. The estimate is **unbiased
  and not clamped**: a single sample can come out slightly below zero when the posteriors nearly coincide, and the mean over
  the batch is what estimates the divergence. Clamping at zero would make the estimate non-negative but biased upward, with a
  floor that never reaches 0 on aligned spaces. On identical posteriors it is exactly 0, whatever the draws are, since every
  log-density ratio is 0. It is the one distance that draws random numbers (`is_stochastic`), so a run that uses it still
  resumes exactly (ADR 0032), the random-number state being in the checkpoint.
- `info_nce` is the contrastive objective that finding I-9 found missing. Row `i` of the first space and row `i` of the second
  are a positive pair, and the other rows of the batch are the negatives. It uses the posterior means, normalized, with a
  `temperature` (default 0.1), and averages the two directions (first against second, second against first) so that it is
  symmetric. It needs at least 2 samples and a `ValueError` says so, and its negatives are the other samples of the same
  call: with gradient accumulation, each micro-batch has its own. Its minimum is above 0 (`is_zero_on_identical = False`).

**4. The scale follows ADR 0029.** Every distance that is a sum over the latent dimensions (`mean_mse`, `wasserstein2_diag`,
`symmetric_kl`) returns that sum per sample, so that averaged over the batch it is on the footing of the KL term.
`mixture_jsd` is on the same footing in another way: it is the divergence of the whole posterior, whose log-density is a sum
over the dimensions, so it is one number per sample that is at most `log K` whatever the width (not a sum of per-dimension
divergences, which the mixture, not being a product, does not factor into). `dimension_reduction: mean` divides the value by the
width instead, for a weight that does not depend on the width of the space.
`cosine` and `info_nce` have no sum over dimensions (a cosine is dimension-free, and a cross-entropy is over the batch), take
no such option and are on their own scale: `cosine` in `[0, 2]`, `info_nce` near `log(batch)` at chance. Their weight is not
comparable to `beta`, and the how-to says so.

**5. Numerics.** The computation runs in `float32` (a `float64` input stays `float64`), with autocast switched off, so a
distance returns the same value inside and outside a mixed precision region. Every distance has a finite gradient on identical
inputs at the origin (`mu = 0`, `logvar = 0`). The tests check each of these for all six.

**6. What this ADR does not do.** It adds no loss term, no config field and no experiment: `latent_alignment` and
`cross_reconstruction` are P6-4, and they will build a distance from a `{name, kwargs}` entry the way the reconstruction losses
are built. It does not choose the distance of stage 3 (D-13 stays open, the six are compared in P6-6); the how-to proposes
`wasserstein2_diag`, with its default squared form, as the first to try, and `mean_mse` as the ablation.

## Alternatives considered

- **A table of functions, like the four reconstruction losses of 2.0.0.** A distance may own parameters (a learned temperature
  for `info_nce`, a projection in front of the comparison), and its declarations are read from the class. The reconstruction
  losses made the same move for the same reasons (ADR 0048). Rejected.
- **An abstract `forward` in each subclass, as the regularizers have.** It repeats the input checks, the cast and the autocast
  guard in every file, and the mixed-precision rule is exactly the kind a new class forgets. The base class owns `forward`
  and the subclass owns the mathematics. Rejected.
- **Compare two samples, `z_a` and `z_b`, instead of the posteriors.** Two independent draws add noise that the posterior
  parameters do not have, and a closed form is lost. The roadmap says the distance acts on the parameters of one sample from
  each modality. Rejected.
- **Reduce over the batch inside the distance.** It would remove the per-sample value that a mask, a sample weight or a
  per-sample log needs, and the regularizers keep that choice outside. Rejected.
- **The root of the 2-Wasserstein distance as the default.** It has no gradient at zero, and a pair that starts aligned (a
  warm start) is near zero. Rejected as a default, kept as an option.
- **The average of the two KL directions for `symmetric_kl`.** It is half the Jeffreys divergence: the same shape, a factor of
  2. The sum is the usual name for the symmetric KL, and `lambda` can absorb the factor. Rejected.
- **Clamp `mixture_jsd` at zero, or use a closed-form bound.** The clamp biases the estimate upward, and the known closed-form
  bounds of a Gaussian mixture entropy are not the divergence. Rejected.
- **Extend the pairwise distances to `K` spaces by averaging over the pairs.** It is a choice about weighting that the loss
  term makes (`pairs` is a list there), and only `mixture_jsd` is a divergence of `K` distributions by itself. Rejected here.

## Consequences

- Importing `global_vae.losses.distances` registers six names. Nothing else imports it yet, so no run, no checkpoint and no
  default changes.
- `tests/unit/test_registry_contract.py` covers a thirteenth registry. A distance added later joins by adding a file, an import
  line and, if it needs one, a declaration.
- Every distance returns a `float32` vector of one value per sample, finite on its gradient at the origin, the same inside and
  outside autocast, and every refusal of a malformed input names the class and the shapes.
- A caller that needs the shape of the problem before building anything reads the class declarations: `max_spaces` to refuse a
  pair list of the wrong length, `min_batch_size` to refuse a batch of one, `is_zero_on_identical` to know that a value of
  `info_nce` is never 0.
- `info_nce` and `cosine` are not on the footing of `beta`; their weights are tuned on their own.
- The specification's lists of registries (§8, §10) do not name the reconstruction losses, the loss terms or the distances.
  Updating the specification is a proposal for the owner, not made here.
