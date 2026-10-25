# Choose beta

`beta` sets how strongly a latent space is pulled toward the prior, against how well the decoders
reconstruct. This page says what it multiplies in this framework, how the reconstruction
reduction changes that, how to read the latent diagnostics of an evaluation report, and how to
run a sweep. The reasoning is in [ADR 0029](../adr/0029-loss-scale-convention.md).

It holds no value of `beta` from a real run. Which values to use, and how long the warm-up should
be, is roadmap decision D-6, and it waits for the first real run (P3-1). Everything below is
either arithmetic on the objective or a procedure.

## What beta multiplies

```
total = sum over decoders m of  w_m * R_m   +   sum over latent spaces s of  beta_s * KL_s
```

`KL_s` is a sum over the latent dimensions of a sample, averaged over the batch. `R_m` is the
reconstruction loss of decoder `m`, scaled by its **reduction**
(`training.reconstruction_reduction`):

| Reduction | `R_m` | Where the usual ELBO sits (Gaussian decoder of unit variance, `reconstruction_loss: mse`) |
|---|---|---|
| `mean` (default) | the mean over every element of the batch, as `torch.nn.functional` returns it | `beta = 2 / N` |
| `sum` | the sum over the `N` values of a sample, averaged over the batch | `beta = 1`, with `reconstruction_weight: 0.5` |

`N` is the number of values one sample of the modality reconstructs. For a 256-point signal
`N = 256`, so the ELBO is at `beta = 2 / 256 = 0.0078125` under `mean`, and the default
`beta = 1` weights the KL 128 times (`N / 2`) more than the ELBO does. For a 32 by 32 image
`N = 1024` and the factor is 512. These are positions on the scale, not recommended values: a
`beta` below or above the ELBO is a beta-VAE, and choosing it is what the sweep below is for.

Two more rows of the same reading, for the other losses of `training.reconstruction_loss`: with
`sum`, `bce` is the negative log-likelihood of a Bernoulli decoder and `l1` that of a Laplace
decoder of scale 1, so `beta = 1` is the ELBO for both, with `reconstruction_weight: 1.0`.

With several modalities the two reductions differ in one more way. `mean` gives each modality a
term of the same size whatever its `N`, which keeps a 256-point signal and a 1024-pixel image on
the same footing, but then the ELBO-equivalent weight of one KL differs per modality (`N_m / 2`).
`sum` gives each modality its own negative log-likelihood, so one `beta` has one meaning, and a
modality with more values weighs more. `reconstruction_weight` is how you rebalance them: one number
for every modality, or one per decoder (see
[set the loss and weight per modality](train.md#set-the-loss-and-weight-per-modality)).

## Choose a reduction

- **Keep `mean`** if you have already tuned a `beta` and a warm-up, if you want to compare with
  earlier runs, or if you want modalities of different sizes balanced without a weight.
- **Use `sum`** if you want `beta` to keep one meaning across data shapes, or to read it as the
  weight of an ELBO.
- **Mix them** with a mapping when modalities call for different treatment. A decoder you leave
  out keeps `mean`.
- **Images** are where `mean` fails first, because `N` is large (4096 for a 64 by 64 image, so
  `beta = 1` weights the KL 2048 times more than the ELBO does). The image experiments
  (`experiment/image_vae`, `experiment/image_resnet_vae`) therefore set `sum` with weight 0.5. On
  the synthetic images of `examples/04_paired_loader.py`, 30 epochs under `mean` left no latent
  unit active, and under `sum` all 16 were. That is a measurement on synthetic data, not a tuned
  value: the numbers are D-6's. The transformer and ViT experiments of ADR 0037
  (`signal_transformer_vae`, `image_vit_vae`, `signal_image_transformer_vae`,
  `signal_image_shared_transformer_vae`) set `sum` with weight 0.5 for the same reason, and `mean`
  collapsed each of them on the same data.
- **A shared decoder takes one value.** For a decoder that reconstructs several modalities
  (`shared_transformer_decoder_v1`), give `reconstruction_reduction`, `reconstruction_weight` and
  `reconstruction_loss` a single value, which applies to every modality. A mapping is not usable
  there yet: one keyed by modality is refused, and one keyed by decoder name is not applied by the
  loss (a weight or a reduction) or raises `KeyError` (a loss function), see the finding of ADR 0037.

```yaml
training:
  reconstruction_loss: mse
  reconstruction_weight: 0.5        # the unit-variance Gaussian, for mse with sum
  reconstruction_reduction: sum     # or mean, or {signal: sum, image: mean}
  beta: 1.0
```

From the command line, a mapping is one quoted Hydra override:

```bash
python scripts/train.py 'training.reconstruction_reduction={signal:sum,image:mean}'
```

A name that is not `mean` or `sum`, or a key that is not a decoder of the model, stops the run
with an error that starts with `training.reconstruction_reduction`. It is an error and not a
warning because a mistyped decoder would silently stay on `mean`.

Changing the reduction changes the meaning of `beta` and the size of every loss. Do it before the
sweep, not during it, and do not read a `sum` curve against a `mean` one.

## Read the latent diagnostics

An evaluation report has, for each latent space, the KL of every dimension and the number of
**active units**: dimensions whose posterior mean varies over the evaluation set by more than a
threshold (the variance of `mu` above 0.01 by default, `--active-unit-threshold` to change it).
They come from the encoder's `(mu, logvar)`, so they mean the same whichever regularizer trained
the model.

```bash
python scripts/evaluate.py \
    --checkpoint runs/model.pt \
    --model-factory mypackage.models:build_model \
    --dataloader-factory mypackage.data:build_test_dataloader \
    --reconstruction-reduction sum \
    --reconstruction-weight 0.5 \
    --output-dir results/
```

The summary gains a block, and `results.json` the full lists under `latent_diagnostics`:

```
  latent diagnostics:
    [z_fused] active units: A/D (variance of mu > 0.01)
      kl per dimension: k_1, k_2, ..., k_D
```

A latent space wider than 16 dimensions prints the minimum, median and maximum KL in the summary;
the JSON always has every value. Pass the reduction, the weight and the loss the model was trained
with (`--reconstruction-reduction`, `--reconstruction-weight`, `--reconstruction-loss`, each a bare
value or `modality=value` pairs), as above, so that `total reconstruction loss` is on the scale of
the training curves. With the defaults (`mse`, weight `1`) only the reduction matters. The
reconstruction metrics (`mse`, `r2`, ...) do not depend on any of them.

| What the report shows | What it says |
|---|---|
| Few active units, KL near zero on the rest, reconstruction poor | Posterior collapse on those dimensions. Lower the effective `beta` (a smaller `beta`, a longer warm-up, or `sum`, which weights the reconstruction `N` times more), or the latent is wider than needed. |
| Few active units, KL near zero on the rest, reconstruction as good as you need | The latent is wider than the data requires. Nothing is wrong. |
| Every dimension active, reconstruction as good as you need | Raising `beta` compresses the latent. Do it until the reconstruction starts to degrade (see the sweep). |
| KL above zero on a dimension whose mean is flat | The dimension spends its KL on an offset of the mean or on a variance that is not 1, not on the input. |
| KL near zero on every dimension, reconstruction no better than predicting the mean | Full collapse. See the first row. |

`scripts/visualize_latent.py` draws the same per-dimension KL as a bar chart, from the same
function, so the bars and the report agree.

The active-unit count is a sample statistic. On a small evaluation set it is noisy, and
`--max-samples` truncates the data it is computed from.

## Sweep recipe

The aim is a plot of reconstruction quality against KL, one point per `beta`, and a choice read
from it.

1. **Fix everything else.** Seed, data split, number of epochs, optimizer, warm-up, reduction. One
   difference between runs: `beta`.
2. **Pick a grid.** Log-spaced, one decade apart (or a factor of 3 for a finer look), over at least
   two decades, around the position of the ELBO in the table above. With `sum` and
   `reconstruction_weight: 0.5` that is `1`; with `mean` it is `2 / N`.
3. **Train one run per value.** Mind which `beta` is in force: a `beta_schedules` entry takes
   priority over `training.beta` for the latent space it names. `configs/training/default.yaml`
   has one for `z_fused`, so for that file the value to sweep is the schedule's `end_value`.
   `configs/training/several_latents.yaml` has none, so there you sweep `training.beta`.

    ```bash
    betas=(0.1 1 10)          # an example for `sum`, whose ELBO position is 1: use your own grid
    for beta in "${betas[@]}"; do
        python scripts/train.py \
            data.train_path=/path/to/your/data \
            data.loader_factory=my_project.data:buildSignalDataloaders \
            training.reconstruction_weight=0.5 \
            training.reconstruction_reduction=sum \
            training.beta_schedules.z_fused.kwargs.end_value="$beta" \
            output_dir="runs/beta_$beta"
    done
    ```

4. **Evaluate every run on the same held-out set,** with the reduction and weight it was trained
   with.
   `best.pt` is chosen by the validation loss of that run, which depends on its `beta`; use a
   fixed epoch's checkpoint instead if you want the runs treated identically.

    ```bash
    for beta in "${betas[@]}"; do
        python scripts/evaluate.py \
            --checkpoint "runs/beta_$beta/checkpoints/best.pt" \
            --model-factory mypackage.models:build_model \
            --dataloader-factory mypackage.data:build_test_dataloader \
            --reconstruction-reduction sum \
            --reconstruction-weight 0.5 \
            --output-dir "results/beta_$beta"
    done
    ```

5. **Tabulate.** Compare the reconstruction metrics, the KL and the active units. Do not compare
   the losses: they weigh the terms differently in every run, by construction.

    ```python
    import json
    from pathlib import Path

    for path in sorted(Path("results").glob("beta_*/results.json")):
        report = json.loads(path.read_text())
        mse = report["reconstruction_metrics"]["signal"]["mse"]  # your modality's name
        for latent, diagnostics in report["latent_diagnostics"].items():
            kl = report["regularization_metrics"][latent]["kl_standard_normal"]
            active = f"{diagnostics['active_units']}/{diagnostics['num_dimensions']}"
            print(path.parent.name, latent, f"mse={mse:.6f}", f"kl={kl:.3f}", f"active={active}")
    ```

6. **Choose.** Take the largest `beta` whose reconstruction still meets what you need, on the held-out
   data and not on the training set. A larger `beta` gives a more compact, smoother latent, which is
   what you want from it, and the reconstruction is what you pay. The bend of the curve, where the
   reconstruction starts to degrade quickly for a small gain in KL, is where to look. The active
   units tell you whether the latent is still in use at that value or has collapsed.
7. **Then the schedule.** Keep that `beta` and sweep the warm-up the same way, changing
   `training.beta_schedules.<latent>.kwargs.warmup_steps` only, and compare the active units at the
   end of each run: the warm-up exists to keep dimensions from collapsing early, so that is the
   number it should move.

Write the result down with the data it was measured on. When the first real run of P3-1 has been
made, its values belong in roadmap decision D-6, not in this page.
