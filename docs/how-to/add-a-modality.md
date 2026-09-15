# Add a new modality

Adding a modality should never require touching core framework code
(`models/global_vae.py`, `latent/`, `losses/`). If it does, that's a sign the
registry pattern is being bypassed somewhere — flag it rather than
special-casing the new modality in.

1. **Subclass [`AbstractEncoder`](../reference/global_vae/encoders/base.md)**
   in `encoders/`, mapping your modality's raw input to a `(mu, logvar)`
   pair. See `encoders/OneDCnnEncoder.py` for a worked example
   (length-agnostic, via adaptive pooling).
2. **Subclass [`AbstractDecoder`](../reference/global_vae/decoders/base.md)**
   in `decoders/`, mapping a latent vector back to a reconstruction of your
   modality.
3. **Register both** with `@registerEncoder("your_name")` /
   `@registerDecoder("your_name")`, and add the import to
   `encoders/__init__.py` / `decoders/__init__.py` so the decorator actually
   runs — a registered class only exists once its module has been imported.
4. **Add a config entry** referencing the two registry names. See
   `configs/model/signal_single_latent.yaml` for the shape.
5. **Add a test**: a unit test for your encoder/decoder's forward-pass
   shapes and gradient flow, plus an entry in the relevant integration test
   if your modality changes which of the 8 architecture configurations
   (spec §2.1) are exercised end to end.

See the [project specification, §6](../global-vae-project-specification.md)
for the full modality roadmap (1D signals and images are Phase 1; anything
else is intentionally open-ended).
