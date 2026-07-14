# Global Multimodal VAE — Project Specification

**Status:** living document, v0.1 — refine as decisions evolve.
**Purpose:** this document is the ground-truth context for the Global VAE project. Any Claude conversation working on this project should treat it as authoritative. When something here is ambiguous or missing, ask rather than assume.

---

## 0. Language & conventions

- Conversations about the project may happen in any language (typically French).
- All code, docstrings, comments, config keys, commit messages, and technical documentation **must be written in English** (research/industry standard, needed for eventual publication and collaboration).

---

## 1. Vision

Build a **modular, extensible multimodal Variational Autoencoder framework** capable of ingesting and reconstructing arbitrary data modalities — not a model for two fixed data types, but an architecture designed from day one so that adding a new modality never requires touching the core.

Context: research project with both industrial and academic ambitions. No artificial ceiling on complexity — the framework should be able to evolve toward richer generative architectures (hierarchical/sequential latents, discrete codes, autoregressive priors) if the research direction moves toward world-model-like or LLM-adjacent systems. We are not building that now, but the core interfaces must not make assumptions that would block it later.

**Starting modalities (Phase 1):**
1. 1D signals — a generic modality (any scalar series indexed by one variable). The first concrete dataset is SAXS (Small-Angle X-ray Scattering — scattering intensity vs. scattering vector *q*), but the encoder/decoder must be designed for the "1D signal" modality in general, not hardcoded to SAXS specifics.
2. Images

SAXS is a dataset/instance of the signal modality, not a modality of its own. Other signal sources (spectroscopy, sensor time series, etc.) should slot into the same "signal" encoder/decoder family later, with only preprocessing differing, not the architecture.

Everything beyond this is added later through the same extension mechanism — no special-casing.

---

## 2. Core architectural principle: three independent axes

The model is not one fixed architecture — it's a **configuration space** along three independent binary axes, instantiated from config at model-creation time.

| Axis | Option A | Option B |
|---|---|---|
| **Encoder(s)** | Several — one specialized encoder per modality | Single — one shared/unified encoder ingesting all modalities |
| **Latent space(s)** | Single — exactly one latent code for the whole model | Several — independent latent spaces connected to encoders and decoders through a configurable routing graph (see §2.2) |
| **Decoder(s)** | Several — one specialized decoder per modality | Single — one shared decoder producing all outputs |

This gives 8 valid configurations. None is "the" architecture — all must remain selectable via config.

### 2.1 Configuration matrix

| Code | Encoders | Latent | Decoders | Notes |
|---|---|---|---|---|
| E1‑L1‑D1 | single shared | single | single shared | Fully unified: modalities tokenized/concatenated into one stream in and out. |
| E1‑L1‑DN | single shared | single | per‑modality | Shared trunk, specialized reconstruction heads. |
| E1‑LN‑D1 | single shared | several | single shared | One backbone fans out to several independent latent heads; the single decoder consumes them through an assembler (§2.2). |
| E1‑LN‑DN | single shared | several | per‑modality | One backbone, several independent latents; each decoder consumes whichever subset the routing graph assigns it. |
| EN‑L1‑D1 | per‑modality | single (fused) | single shared | Fusion (§4) combines specialized encoders into one latent; generic decoder. |
| **EN‑L1‑DN** | per‑modality | single (fused) | per‑modality | **Recommended Phase‑1 default** — classic multimodal VAE family (MVAE/MMVAE‑style). |
| EN‑LN‑D1 | per‑modality | several | single shared | Specialized encoders feed several independent latents; the single decoder assembles all of them (§2.2). |
| EN‑LN‑DN | per‑modality | several | per‑modality | Most expressive / most flexible — a full routing graph between encoders, latent spaces, and decoders. |

The table above is a simplified overview. For any "several latent spaces" row, the actual topology — which encoder feeds which latent space(s), which decoder consumes which latent space(s), and with what assembler — is a further configuration choice, detailed in §2.2.

### 2.2 Latent routing graph

"Several latent spaces" does not mean a fixed shared/private split. It means the model can have **any number of independent latent spaces**, each with its own posterior, its own prior, and its own KL term, wired to encoders and decoders through a **configurable routing graph**:

- **Encoder → latent.** An encoder can feed exactly one latent space, or fan out to several — e.g. one shared trunk with several independent projection heads, each producing its own `(mu, logvar)`. When *several encoders* feed the *same* latent space, that's the Fusion problem from §4 (PoE / MoE / concat+MLP / cross-attention) — Fusion combines distribution parameters, before sampling.
- **Latent → decoder.** A decoder can consume exactly one latent space, or several. When it consumes several, an **Assembler** combines the already-realized latent vectors into one input tensor. Candidate assemblers: `concat`, `sum`, `average` — a pluggable registry following the same pattern as Fusion, but operating on realized vectors rather than distribution parameters (no probabilistic machinery needed here — the vectors already exist, this is just merging them for the decoder's input layer).

This **generalizes** the earlier "shared + private" idea rather than replacing it — shared+private is just one specific routing graph: all encoders feed a `z_shared` via Fusion, each encoder also feeds its own untouched `z_private_m`, and each modality's decoder consumes `{z_shared, z_private_m}` via `concat`. The framework supports that topology as *one configuration among others*, not as the hardcoded meaning of "several latent spaces."

**Constraints to validate at model-construction time:**
- Every latent space must have at least one encoder feeding it and at least one decoder consuming it — no orphan latent spaces.
- `sum` and `average` assemblers require all their input latent spaces to share the same dimensionality; `concat` has no such restriction.

---

## 3. Glossary (keep consistent everywhere)

- **Modality** — a data type/source (1D signal, image, ...). A modality may have several concrete datasets (e.g. SAXS is one dataset within the "1D signal" modality).
- **Encoder** — network mapping one modality's raw input to distribution parameters (or to features destined for fusion).
- **Fusion module** — combines outputs of multiple encoders into a single set of latent distribution parameters.
- **Latent space** — the space of `z`. Either a single shared space, or a factorized space (`z_shared` + `z_private_m`).
- **Decoder** — network mapping latent(s) back to a modality-specific reconstruction.
- **Configuration** — the tuple {encoder cardinality, latent cardinality, decoder cardinality, fusion strategy, residual flag} that fully determines a model instance.

---

## 4. Fusion strategies (encoder → latent)

No single fusion strategy is hardcoded. The framework implements **several interchangeable strategies**, selected via config at model initialization, each optionally with residual connections (also a config flag).

| Strategy | Idea | Handles missing modalities natively? |
|---|---|---|
| **Product-of-Experts (PoE)** | Each modality contributes a Gaussian "expert"; experts are multiplied together (MVAE-style). | Yes — simply drop the missing expert's term. |
| **Mixture-of-Experts (MoE)** | Each modality proposes its own latent estimate; a mixture combines them (MMVAE-style). | Yes — natively subset-tolerant. |
| **Concatenation + MLP** | Concatenate encoder outputs, project through an MLP. Simple baseline. | No — needs an explicit imputation/masking strategy if modalities can be missing. |
| **Cross-attention / transformer fusion** | Treat each modality's encoding as a token; a transformer block fuses them. | Yes — missing modalities are simply omitted tokens; also the most natural bridge toward the long-term "world model" direction. |

This is implemented as a **strategy pattern**: an `AbstractFusion` interface with each strategy as a subclass, registered by name (e.g. `poe`, `moe`, `concat_mlp`, `cross_attention`) so a new one can be added without touching existing code.

---

## 5. Missing-modality robustness

Explicitly desired, but achieved as a **side effect of the fusion strategy choice** rather than as separate machinery:

- Using PoE, MoE, or cross-attention fusion, the model can be trained and queried with any subset of available modalities.
- Recommended training technique: **modality dropout** — randomly hide one or more modalities per training step so the model learns to be robust to their absence at inference.
- Concatenation-based fusion is the one strategy that does *not* get this for free; if used with missing modalities it needs an explicit masking/imputation scheme. This should be documented as a known limitation, not silently patched.

---

## 6. Data modality roadmap

| Phase | Modality | Notes for encoder/decoder design |
|---|---|---|
| 1 | 1D signal (generic) — first dataset: SAXS (intensity vs. *q*) | Variable-length series. Preprocessing is dataset-specific (e.g. log-scale intensity is standard for SAXS) and must live outside the architecture, in the data pipeline. Candidate encoders: 1D CNN or small transformer over the series. |
| 2 | Images | Candidate encoders: CNN (ResNet-style) or ViT, depending on resolution/dataset size. |
| 3+ | Open-ended (audio, tabular, text, time series, graphs, point clouds, ...) | Adding a modality = writing one new `Encoder` + `Decoder` subclass and registering them. Zero changes to core framework code. |

---

## 7. Long-term direction (not built now, but not blocked either)

Noted so that early interface design doesn't foreclose these later:
- Hierarchical or sequential/temporal latents (world-model-style state prediction).
- Discrete/VQ latent codes as an alternative to continuous Gaussian latents.
- Autoregressive or learned priors over the latent space instead of a fixed standard normal.
- Scaling toward larger backbones (transformer-heavy encoders/decoders/fusion).

These are **future directions**, listed to keep core abstractions (encoder/decoder/fusion interfaces, latent representation) general enough to extend without a rewrite — not a commitment to build them now.

---

## 8. Repository structure

```
global-vae/
├── pyproject.toml
├── README.md
├── configs/                     # Hydra/OmegaConf configs
│   ├── model/
│   ├── data/
│   └── experiment/
├── src/
│   └── global_vae/
│       ├── encoders/
│       │   ├── base.py          # AbstractEncoder interface
│       │   ├── registry.py
│       │   ├── signal_encoder.py
│       │   └── image_encoder.py
│       ├── decoders/
│       │   ├── base.py          # AbstractDecoder interface
│       │   ├── registry.py
│       │   ├── signal_decoder.py
│       │   └── image_decoder.py
│       ├── fusion/
│       │   ├── base.py          # AbstractFusion interface
│       │   ├── registry.py
│       │   ├── poe.py
│       │   ├── moe.py
│       │   ├── concat_mlp.py
│       │   └── cross_attention.py
│       ├── latent/
│       │   ├── single.py
│       │   └── factorized.py
│       ├── models/
│       │   └── global_vae.py    # assembles encoders + fusion + latent + decoders from config
│       ├── losses/
│       │   ├── reconstruction.py
│       │   └── kl.py
│       ├── data/
│       │   ├── datamodule.py
│       │   └── transforms/
│       ├── training/
│       │   └── trainer.py
│       └── utils/
├── tests/
│   ├── unit/
│   └── integration/              # end-to-end test for each of the 8 configurations
├── notebooks/
├── scripts/
└── docs/
```

---

## 9. Illustrative config example

Not final — shows how the registry + config-driven pattern is meant to operate in practice:

```yaml
model:
  name: global_vae
  modalities:
    signal:                    # first concrete dataset for this modality: SAXS
      encoder: signal_cnn_v1
      decoder: signal_cnn_v1
    image:
      encoder: resnet_encoder_v1
      decoder: resnet_decoder_v1
  latent:
    mode: single            # single | factorized
    dim: 128
  fusion:
    strategy: poe            # poe | moe | concat_mlp | cross_attention
    residual: true
  training:
    beta_schedule: linear_warmup
    modality_dropout_p: 0.15  # trains robustness to missing modalities
```

---

## 10. Coding standards

- **Language/runtime:** Python 3.11+, PyTorch (latest stable).
- **Formatting/linting:** `ruff` (lint + format), consistent import ordering.
- **Naming convention (custom — overrides PEP8 default for callables):**
  - Classes → `CamelCase` (e.g. `GlobalVae`, `SignalEncoder`, `ProductOfExperts`).
  - Variables → `snake_case` (e.g. `latent_dim`, `batch_size`).
  - Functions and methods → same rule as classes but starting lowercase, i.e. `camelCase` (e.g. `computeLoss`, `encodeSignal`, `registerEncoder`) — **not** PEP8's usual `snake_case` for callables.
  - Since this deviates from PEP8, disable/adjust `ruff`'s naming rules (`N802`, `N803`, `N806`) in `pyproject.toml` and note the exception in the contributor docs, so linting doesn't silently "fix" it back to snake_case later.
- **Typing:** type hints mandatory everywhere; `mypy` run in CI.
- **Docstrings:** Google-style, mandatory on every public class/function — purpose, `Args`, `Returns`, `Raises`.
- **Modularity:** one responsibility per file; one class per file for encoders/decoders/fusion strategies. No god-files.
- **Interfaces:** `AbstractEncoder`, `AbstractDecoder`, `AbstractFusion` as ABCs. Every concrete implementation subclasses one of these and self-registers via a decorator (`@registerEncoder("signal")`) — this is what makes "add a modality without touching the core" actually true, not just aspirational.
- **Config management:** Hydra + structured dataclasses (or Pydantic) for validated, composable configs. No magic strings/dicts scattered through the code.
- **Testing:** `pytest`. Unit tests per module, plus an integration test that instantiates **each of the 8 architecture combinations** end-to-end on dummy tensors (shape and gradient sanity checks). Coverage should target core logic (fusion math, loss correctness, forward/backward pass), not a vanity percentage.
- **Experiment tracking:** Weights & Biases or MLflow — log losses, latent-space visualizations, reconstructions per run.
- **Reproducibility:** global seed management, deterministic-mode flag documented, config snapshotted with every run.
- **Logging:** standard `logging` module — no bare `print`.
- **Version control:** Conventional Commits, semantic versioning, maintained `CHANGELOG.md`.
- **CI:** GitHub Actions — lint, type-check, tests on every push.
- **Documentation:** `mkdocs` + `mkdocstrings` built from docstrings; major architectural choices (e.g. "why PoE + MoE + cross-attention", "why factorized latent") recorded as short ADRs (`docs/adr/NNNN-title.md`).
- **New-modality checklist:** adding a modality means (1) subclass `Encoder`, (2) subclass `Decoder`, (3) register both, (4) add a config entry, (5) add a test. This keeps the "no limit on data types" promise operational rather than aspirational.

---

## 11. Open questions (deliberately deferred)

- Exact latent factorization scheme for "several latent spaces" (shared + private vs. fully independent) — see assumption flagged in §2.1.
- Precise loss weighting / β-VAE annealing schedule.
- Whether training uses raw PyTorch loops or PyTorch Lightning.
- First concrete joint signal+image dataset/task to validate the pipeline end-to-end (using SAXS as the initial signal dataset).
- Production/serving requirements, if any arise later (currently out of scope — research framework first).

---

## 12. How future Claude conversations should use this document

- Treat the terminology in §3 as canonical — don't invent new names for the same concepts.
- Any new component (encoder, decoder, fusion strategy) must follow the registry pattern in §10, not be hardcoded into the model class.
- If a request would violate the "no fixed fusion strategy" or "no fixed modality set" principles, flag it rather than silently narrowing the design.
- When a decision in §11 is needed to proceed, ask — don't guess and move on.
