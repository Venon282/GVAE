# Evaluate and visualize a checkpoint

Two standalone scripts cover this, neither needing a `Trainer` — only a
`GlobalVae` and a dataloader. Both restrict a raw batch to the keys
naming one of the model's encoders before the forward pass
(`GlobalVae.selectEncoderInputs`,
`docs/adr/0019-decouple-encoder-inputs-from-decoder-targets.md`), so a
decoder whose reconstruction target is not itself an encoder input (a
translation-style model) works with no special-casing on your side.

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

If your own data pipeline applied a `data.transforms` step (`log` /
`standardize` / `resample`, spec §6.2) before training, reconstruction
figures are plotted in that transformed space by default. Pass
`--inverse-transform-factory mypackage.data:build_inverse_transforms`
(a factory returning `dict[str, Callable[[Tensor], Tensor]]`, e.g. built
straight from a `DataConfig` via `buildTransformPipeline`) to have them
plotted in the original, physical-units space instead.

## Quick latent-space inspection

```bash
python scripts/visualize_latent.py \
    --checkpoint runs/model.pt \
    --model-factory mypackage.models:build_model \
    --dataloader-factory mypackage.data:build_dataloader \
    --output-dir results/latent/
```

Faster than a full evaluation pass when you just want to look at the latent
space and training curves: saves a latent-space scatter plot (PCA, the
default projection, is an exact, deterministic decomposition, so this and
`scripts/evaluate.py`'s own latent scatter plot agree on the same
checkpoint and dataloader) and a per-dimension KL bar chart per latent
space (for spotting posterior collapse), plus the training-curve plot if
the checkpoint carries a history — split by default onto two
independently-scaled axes (reconstruction/total loss vs. regularization
loss, spec §2.3; `--history-no-twin-axis` to disable, `--history-log-scale`
/`--history-twin-log-scale` for either axis). Supports `--label-key` (color
the scatter by a batch field), `--latent-names` (restrict which latent
spaces to plot), and `--use-samples` (plot realized samples instead of the
posterior mean).

Run either script with `--help` for the full option list.
