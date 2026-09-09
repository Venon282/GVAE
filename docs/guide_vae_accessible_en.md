# Understanding VAEs: complete and accessible guide

This document explains VAEs (Variational Autoencoders) from scratch. Every term is defined when it first appears, including those that seem "obvious" to someone already familiar with the subject: the VAE literature is often poorly explained for beginners, with notation that assumes years of statistics already. This document goes through everything in order, with direct links to the project code (`global-vae-project-specification.md` and the files in `src/global_vae/`) at each step.

---

## 1. The starting point: the classic autoencoder

Before the "V" in VAE, there is the plain autoencoder (AE). The idea is simple:

- An **encoder** takes an input datum `x` and compresses it into a small vector `z`, called the **latent code** (or **latent space**, for the set of all possible `z` values).
- A **decoder** takes this `z` and tries to reconstruct `x` identically, producing a reconstruction denoted `x̂` (x-hat).

Analogy: summarize a book on a 10-line sheet (`z`), then ask someone else to rewrite the entire book from that single sheet (`x̂`). If the summary is good, the reconstructed book looks like the original.

**The problem with a classic AE**: nothing forces the `z` values of two similar inputs to be close to each other in the latent space, nor does it force the latent space to be "filled" continuously. Result: by picking a random point in this space (to *generate* a new, never-before-seen datum), there is a high chance of landing in a "hole" that the decoder has never learned to interpret, producing nonsense. A classic AE knows how to compress and reconstruct, but does not know how to *generate*.

This is exactly the problem that the V (Variational) solves.

---

## 2. The idea of a VAE: replacing a point with a cloud

A VAE (Variational Autoencoder) changes one thing, but it has enormous consequences: instead of the encoder producing one single, precise point `z`, it produces a **probability distribution**: a kind of "cloud" of possible values for `z`, with a more probable central region and less probable edges.

A **probability distribution** is a rule that says, for each possible value, how likely it is. The VAE uses the simplest and best-understood distribution in statistics: the **normal distribution** (or **Gaussian**, named after the mathematician Carl Friedrich Gauss), the famous "bell curve." A Gaussian is fully described by two numbers:

- the **mean** (`mu`, the Greek letter μ): the center of the bell, the most probable value.
- the **variance** (often written σ², sigma squared): how wide or narrow the bell is. A small variance = narrow bell = we are almost certain of the value. A large variance = wide bell = a lot of uncertainty. The square root of the variance is called the **standard deviation** (σ, sigma), in the same unit as the datum itself.

Concretely, for a given input `x`, the encoder no longer produces a single `z`, but two vectors: `mu` (the center of the distribution) and something related to the variance (see just below why it is not the variance directly). This can be seen exactly as-is in `AbstractEncoder.forward`:

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Retourne (mu, logvar), chacun de forme (batch, latent_dim)."""
```

### Why `logvar` instead of the variance directly?

The variance must always be positive (a negative "width" makes no sense). But a neural network, at the output of a simple linear layer, can produce any number, positive or negative. If we asked the network to produce the variance directly, nothing would prevent it from outputting a negative number at some point during training, which would break everything.

The solution: the network produces `logvar` (the logarithm of the variance), which can be any real number without a problem, and then we convert it back to the actual variance with `exp(logvar)` (the exponential "cancels" the logarithm and guarantees a result that is always positive, whatever `logvar`). This is exactly what `LatentSpace.reparameterize` (`src/global_vae/latent/base.py`) does:

```python
std = torch.exp(0.5 * logvar)   # écart-type = exp(logvar / 2)
```

(Le `0.5 *` vient du fait qu'on veut l'écart-type, pas la variance : écart-type = racine carrée de la variance = exp(logvar)^0.5 = exp(0.5 * logvar).)

---

## 3. The prior: the plan we impose on the latent space

**Prior** means "previous," "before" (from the Latin *a priori*, "before experience"). In Bayesian statistics, the prior is the belief we have about a quantity *before even looking at the data*. Here, it is the shape we decide, in advance, to give to the entire latent space, regardless of any particular input `x`.

In a VAE, we almost always choose the same prior: the standard normal distribution, denoted `N(0, I)`:

- mean = 0 in every latent dimension
- variance = 1 in every dimension
- the dimensions are independent of each other (the `I`, for "identity matrix," means that there is no correlation between the dimensions)

This is a choice, not an absolute mathematical necessity, but an excellent practical choice for three reasons:

1. **It is trivial to sample.** Drawing a random vector according to `N(0, I)` simply means drawing independent "normal" random numbers, a basic operation in any numerical library (`torch.randn_like(...)`).
2. **It is mathematically well understood**, which gives us a simple formula for comparing a distribution to this prior (section 8).
3. **It gives generation "for free"**: once the model is trained, to create an entirely new datum, all you need to do is draw a random `z` from this same prior `N(0, I)`, then give it to the decoder. No input `x` is needed at all.

The prior is therefore literally the "mold" we want the latent space to adopt, so that we can then pick from it blindly and always get something that the decoder knows how to interpret.

---

## 4. Posterior: the plan that we really have of the latent space

Once we have defined the prior (the distribution we *want*), we need a word to designate the distribution that the encoder *actually* produces for a specific input `x`. This is the **posterior** ("posterior," "after," that is, after observing the data).

- Prior `p(z)`: the initial belief about `z`, before observing anything. Always `N(0, I)`, fixed, independent of any input.
- Posterior `q(z|x)`: the updated belief about `z`, once we have observed a specific input `x`. This is exactly `(mu, logvar)` that the encoder computes for that `x`. `q(z|x)` is read as "the distribution of `z` given `x`."

Each input `x` therefore has its own posterior (its own `mu`, its own variance), but there is only one prior shared by the whole model. The entire goal of training is to make these posteriors, taken together, look as much as possible like the prior, while remaining sufficiently distinct from one input to another so that the decoder can still tell them apart. This tension is at the heart of every VAE (section 9).

---

## 5. The reparameterization trick: how we can still do gradient descent

Once we have `(mu, logvar)`, we need to *sample* a concrete `z` from this distribution so that we can give it to the decoder. The problem: "sampling randomly" is a non-differentiable operation. Yet training a neural network relies entirely on **gradient descent**: we calculate how much each network weight contributed to the final error by tracing the computations backwards (the **backpropagation**). If a raw random draw sits in the middle of the computation chain, we can no longer propagate the gradient through it: the computation is "broken" at that point.

The trick, the **reparameterization trick**, consists in taking the random part out of the path that we need to differentiate:

```
epsilon ~ N(0, 1)          # un tirage aléatoire "pur", sans rapport avec le réseau
z = mu + epsilon * std     # combinaison différentiable de mu, std (calculés par le réseau) et epsilon
```

`epsilon` is simply external noise, with no parameter to learn. All that matters for the gradient is `mu` and `std`, which are computed by the network and are therefore differentiable normally. This is exactly the code of `LatentSpace.reparameterize`:

```python
def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)   # epsilon, tiré au hasard, hors du graphe de calcul appris
    return mu + eps * std
```

This is the little trick that makes everything else possible: without it, we simply could not train a VAE with standard gradient descent.

---

## 6. KL: the regalurisateur

### Where the name comes from

**KL** is simply the initials of two surnames: **Solomon Kullback** and **Richard Leibler**, two American statisticians who formalized this measure in 1951. The "Kullback-Leibler divergence" (or "KL divergence") measures how much one probability distribution differs from another.

It is not a true "distance" in the strict mathematical sense (it is not symmetric: the divergence from A to B is not necessarily equal to that from B to A), but it behaves intuitively like a distance:

- if the two distributions are identical, the KL divergence is **0**
- the more they differ, the **larger** the divergence

### What it measures in a VAE

We use it to compare the posterior `q(z|x)` (what the encoder produced for this specific input) to the prior `p(z) = N(0, I)` (what we would generally like it to look like). It is written `KL(q(z|x) || p(z))`.

### The formula, term by term

Since the prior and posterior are both Gaussians, there is an exact formula (no approximation needed) for this divergence. It is the one implemented in `KlStandardNormalRegularizer` (`src/global_vae/losses/regularizers/kl_standard_normal.py`):

```python
def forward(self, mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
```

Let's break it down with concrete values:

- **If `mu = 0` and `logvar = 0`** (so variance = `exp(0) = 1`): this is exactly the prior `N(0, 1)`. The calculation gives `-0.5 * (1 + 0 - 0 - 1) = 0`. Makes sense: if the posterior is identical to the prior, the divergence is zero.
- **If `mu` moves away from 0**: the `mu²` term grows, making the result inside the parentheses more negative, and therefore (because of the `-0.5 *` in front) making the final KL larger. In other words: *moving away from the center of the prior is costly.*
- **If `logvar` becomes very negative** (so the variance becomes tiny, and the encoder becomes "too confident" about an almost point-like `z`): the `logvar` term decreases and `exp(logvar)` tends toward 0, which also makes the KL grow. In other words: *becoming too precise or too narrow is costly too, not just moving away from the center.*

This formula therefore penalizes two things at once: moving away from the center of the prior, and being too confident (variance too small). It "pulls" each posterior toward `N(0, I)` in both ways.

### Why we need this constraint

Without any KL term at all, nothing would stop the encoder from doing exactly what a classic autoencoder does (section 1): placing each input `x` at a completely arbitrary location in the latent space, with a tiny variance (almost a point). The model would then learn a kind of directory saying "this exact input goes exactly here," which reconstructs known inputs very well but leaves the latent space full of unorganized holes: exactly the problem we were trying to avoid in section 1.

The KL term is the "glue" that forces all posteriors to stay close to one and the same shared mold (`N(0, I)`), and therefore keeps the latent space continuous and usable as a whole.

---

## 7. Reconstruction: the other half of the loss

The decoder receives `z` and must produce something that looks like `x`. To measure "how similar it is," we use a reconstruction loss function, the simplest and most common being **MSE** (*Mean Squared Error*): we calculate the difference between each value of `x` and `x̂`, square it (so that positive and negative errors do not cancel each other out, and to penalize larger errors more strongly), then take the average. This is `losses/reconstruction.py` in this project, with `torch.nn.functional.mse_loss` as the default choice.

---

## 8. The total loss: reconstruction + KL, and the tension between the two

The total loss of a VAE is simply the addition of the two parts:

```
loss_totale = loss_reconstruction + beta * loss_KL
```

This is exactly `StepLosses` in `training/trainer.py`:

```python
total_loss = reconstruction_loss + regularization_loss
```

These two terms pull in opposite directions, and this is precisely what makes training a VAE delicate:

- Minimizing **only** reconstruction would push the encoder back toward being a classic autoencoder (tiny variance, memorization): good reconstruction, bad latent space.
- Minimizing **only** KL would push the encoder to completely ignore `x` and always produce `mu = 0`, `logvar = 0` (the prior itself, which has zero KL by definition): a well-organized latent space, but one that no longer contains any useful information about `x`. The decoder, deprived of all information, can then only produce an average, blurry reconstruction that is identical for everyone.

A good VAE finds a compromise between the two. In the literature, this compromise has a formal name: the **ELBO** (*Evidence Lower BOund*). This is the exact mathematical name of this quantity (reconstruction minus KL, up to a sign depending on conventions), which we seek to maximize. Its full derivation is not necessary to use the framework, only the fact that "loss = reconstruction + beta * KL" *is* literally a practical reformulation of the ELBO.

### The role of `beta`

`beta` (β) is a simple multiplicative coefficient that weights the importance of the KL term relative to reconstruction. It is the "β" in **β-VAE**, a widely used variant. The larger `beta` is, the more we force the latent space to resemble the prior (at the risk of ignoring `x`); the smaller it is, the more we favor reconstruction fidelity (at the risk of falling back to a poorly organized latent space).

In this project, `beta` can even vary during training, via `training/beta_schedules/`:

- `ConstantBetaSchedule`: `beta` fixed from start to finish.
- `LinearWarmupBetaSchedule`: `beta` starts at 0 and gradually increases.
- `CyclicalAnnealingBetaSchedule`: `beta` rises and falls several times during training.

---

## 9. The classic pitfall: "posterior collapse"

If `beta` is too high, or rises too quickly, right at the very beginning of training (before the decoder has even learned anything useful from `z`), the encoder finds the easiest way to quickly reduce the loss: produce `mu = 0`, `logvar = 0` for absolutely every input, that is, copy the prior exactly, regardless of `x`. The KL drops to 0, the encoder has "won" its half of the game, but it has completely stopped using `z` to transmit any information about `x`. This is **posterior collapse** (the posterior collapsing onto the prior).

The symptom: reconstruction remains poor indefinitely, regardless of how much model capacity (more layers, more parameters) is added, because the problem is not a lack of capacity, but a latent representation that no longer carries any information to exploit.

This is exactly what the project documents in `examples/01_signal_vae_pipeline.py` (the "On regularization" section of its docstring): a first version, with the default `kl_standard_normal` regularizer and a warm-up that was too short, remained stuck at an R² of about 0.3 (a reconstruction quality score close to "barely better than predicting the mean everywhere"), regardless of the capacity added to the model: the typical signature of a collapse.

Two countermeasures, already present in this framework:

- **Beta warm-up** (`LinearWarmupBetaSchedule`): keep `beta` very low at the beginning, giving the decoder time to actually learn to use `z`, before gradually increasing the regularization pressure.
- **Free bits** (`FreeBitsKlRegularizer`, `losses/regularizers/free_bits_kl.py`): give each latent dimension a small KL "budget" that it is never penalized for using. Concretely, below a certain threshold (`free_bits`, often around 0.5 to 1.0), the KL of that dimension no longer helps reduce the loss, so the encoder has no incentive to completely squash it toward 0. This is the solution that `01_signal_vae_pipeline.py` ultimately uses, raising the R² from about 0.3 to about 0.97.

---

## 10. Overview: the complete path through `GlobalVae`

Putting everything we have just seen together, here is the exact path that a batch of inputs follows in the code (`GlobalVae.forward`, `src/global_vae/models/global_vae.py`):

1. `x` (the raw input) passes through the **encoder**, which produces `(mu, logvar)`. (`AbstractEncoder.forward`)
2. `(mu, logvar)` passes through **reparameterize**, which produces a concrete, sampled `z`. (`LatentSpace.reparameterize`)
3. `z` passes through the **decoder**, which produces `x̂`, the reconstruction. (`AbstractDecoder.forward`)
4. We compute the **reconstruction loss** between `x` and `x̂` (`losses/reconstruction.py`), and the **KL loss** between `(mu, logvar)` and the prior `N(0, I)` (`losses/regularizers/kl_standard_normal.py`, or another registered strategy).
5. `total_loss = reconstruction + beta * KL`, and we take a gradient descent step on it (`training/trainer.py::computeLosses` then `.backward()`).

At inference time (once the model is trained), there are two distinct uses, both already present in this code:

- **Reconstructing an existing input**: go through steps 1 to 3 normally (`evaluate()` even uses `use_mean=True`, meaning that we directly take `z = mu`, without random sampling, to obtain a deterministic and reproducible result from one execution to another).
- **Generating an entirely new datum**: skip steps 1 and 2 entirely. Directly draw `z ~ N(0, I)` (the prior), and only perform step 3 (`decoder(z)`). This is where all the effort put into the KL term pays off: because the entire latent space has been trained to resemble this prior, any randomly drawn `z` from it gives the decoder something plausible, even if that exact `z` was never produced by a real `x` during training.

---

## 11. Bonus: what about multiple modalities (signal + image)?

A very quick overview, for the day when Fusion (`fusion/poe.py`) comes into play: when multiple encoders (one per modality) feed the **same** latent space, each produces its own `(mu, logvar)`, like independent "experts" on the same variable `z`. The **Product-of-Experts (PoE)** strategy combines these experts into a single fused posterior, *before* sampling, giving more weight to the most confident experts (those with the smallest variance). Everything else (reparameterize, decoder, losses) then works exactly as described above, on this already fused posterior. This is why the spec clearly distinguishes Fusion (combining distributions, before sampling) from the Assembler (combining already sampled vectors, after): two different steps in the pipeline.

---

## Small glossary, to quickly find a word

| Term | Short definition |
|---|---|
| **Autoencoder (AE)** | Network that compresses an input into a code, then reconstructs it from that code. |
| **VAE** | Autoencoder whose encoder produces a distribution (not a point) and whose latent space is regularized to resemble a chosen prior. |
| **Encoder** | Network that transforms the input `x` into `(mu, logvar)`. |
| **Decoder** | Network that transforms a `z` into a reconstruction `x̂`. |
| **Latent space** | The set of possible `z` values; the compressed representation. |
| **z** | A concrete latent vector, sampled from `(mu, logvar)`. |
| **mu (μ)** | The mean of the distribution produced by the encoder: the most probable center. |
| **logvar** | The logarithm of the variance; allows the network to output any number while guaranteeing a positive variance after `exp()`. |
| **Standard deviation (σ)** | Square root of the variance; measures the width of the distribution, in the same unit as the datum. |
| **Probability distribution** | A rule that gives the probability of each possible value. |
| **Gaussian / normal distribution** | The "bell-shaped" distribution, fully defined by its mean and variance. |
| **Prior `p(z)`** | The distribution we impose in advance on the latent space, fixed, `N(0, I)` by default. |
| **Posterior `q(z\|x)`** | The distribution that the encoder actually produces for a specific input `x`. |
| **Reparameterization trick** | `z = mu + epsilon * std`, the technique that makes sampling differentiable. |
| **KL divergence** | Measure (due to Kullback and Leibler) of the difference between two distributions; 0 if identical. |
| **ELBO** | *Evidence Lower BOund*, the formal quantity that "reconstruction + KL" practically reformulates. |
| **Beta (β-VAE)** | Coefficient that weights the contribution of the KL term in the total loss. |
| **Posterior collapse** | Failure mode where the encoder ignores `x` and always produces the prior; `z` no longer carries information. |
| **Free bits** | A small KL budget, per dimension, that is never penalized, to avoid collapse. |
| **Reconstruction loss** | Measure of the difference between `x` and `x̂` (often MSE). |
| **Fusion (PoE, etc.)** | Combines multiple `(mu, logvar)` values (one per modality) into a single posterior, before sampling. |
| **Assembler** | Combines multiple already sampled `z` values into a single vector, for the decoder. |
