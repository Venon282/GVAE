# Evaluate and visualize a checkpoint

Two standalone scripts cover this, neither needing a `Trainer` — only a
`GlobalVae` and a dataloader.

## Full evaluation report

```bash
python scripts/evaluate.py \
    --checkpoint runs/model.pt \
    --model-factory mypackage.models:build_model \
    --dataloader-factory mypackage.data:build_test_dataloader \
    --output-dir results/
```

`--model-factory` and `--dataloader-factory` are `"module.path:function_name"`
references to your own code, the same convention as `data.loader_factory` in
the training config: the framework never guesses your architecture or data
format. This prints reconstruction metrics (mse / rmse / mae / r2 /
pearson_r) and per-latent-space regularization values to the console, and
(with `--output-dir`) saves a JSON report plus reconstruction / latent-space
figures.

## Quick latent-space inspection

```bash
python scripts/visualize_latent.py \
    --checkpoint runs/model.pt \
    --model-factory mypackage.models:build_model \
    --dataloader-factory mypackage.data:build_dataloader \
    --output-dir results/latent/
```

Faster than a full evaluation pass when you just want to look at the latent
space and training curves: saves a latent-space scatter plot and a
per-dimension KL bar chart per latent space (for spotting posterior
collapse), plus the training-curve plot if the checkpoint carries a history.
Supports `--label-key` (color the scatter by a batch field),
`--latent-names` (restrict which latent spaces to plot), and
`--use-samples` (plot realized samples instead of the posterior mean).

Run either script with `--help` for the full option list.
