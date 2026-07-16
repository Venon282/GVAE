# Global Multimodal VAE: Project Specification

**Status:** living document, v0.1. Refine as decisions evolve.
**Purpose:** this document is the ground-truth context for the Global VAE project. Any Claude conversation working on this project should treat it as authoritative. When something here is ambiguous or missing, ask rather than assume.

---

## 0. Language & conventions

- Conversations about the project may happen in any language (typically French).
- All code, docstrings, comments, config keys, commit messages, and technical documentation **must be written in English** (research/industry standard, needed for eventual publication and collaboration).

---

## 1. Vision

Build a **modular, extensible multimodal Variational Autoencoder framework** capable of ingesting and reconstructing arbitrary data modalities: not a model for two fixed data types, but an architecture designed from day one so that adding a new modality never requires touching the core.

Context: research project with both industrial and academic ambitions. No artificial ceiling on complexity: the framework should be able to evolve toward richer generative architectures (hierarchical/sequential latents, discrete codes, autoregressive priors) if the research direction moves toward world-model-like or LLM-adjacent systems. We are not building that now, but the core interfaces must not make assumptions that would block it later.

**Starting modalities (Phase 1):**
1. 1D signals: a generic modality (any scalar series indexed by one variable). The first concrete dataset is SAXS (Small-Angle X-ray Scattering, scattering intensity vs. scattering vector *q*), but the encoder/decoder must be designed for the "1D signal" modality in general, not hardcoded to SAXS specifics.
2. Images

SAXS is a dataset/instance of the signal modality, not a modality of its own. Other signal sources (spectroscopy, sensor time series, etc.) should slot into the same "signal" encoder/decoder family later, with only preprocessing differing, not the architecture.

Everything beyond this is added later through the same extension mechanism, with no special-casing.

---

## 2. Core architectural principle: three independent axes

The model is not one fixed architecture: it's a **configuration space** along three independent binary axes, instantiated from config at model-creation time.

| Axis | Option A | Option B |
|---|---|---|
| **Encoder(s)** | Several: one specialized encoder per modality | Single: one shared/unified encoder ingesting all modalities |
| **Latent space(s)** | Single: exactly one latent code for the whole model | Several: independent latent spaces connected to encoders and decoders through a configurable routing graph (see §2.2) |
| **Decoder(s)** | Several: one specialized decoder per modality | Single: one shared decoder producing all outputs |

This gives 8 valid configurations. None is "the" architecture: all must remain selectable via config.

### 2.1 Configuration matrix

| Code | Encoders | Latent | Decoders | Notes |
|---|---|---|---|---|
| E1‑L1‑D1 | single shared | single | single shared | Fully unified: modalities tokenized/concatenated into one stream in and out. |
| E1‑L1‑DN | single shared | single | per‑modality | Shared trunk, specialized reconstruction heads. |
| E1‑LN‑D1 | single shared | several | single shared | One backbone fans out to several independent latent heads; the single decoder consumes them through an assembler (§2.2). |
| E1‑LN‑DN | single shared | several | per‑modality | One backbone, several independent latents; each decoder consumes whichever subset the routing graph assigns it. |
| EN‑L1‑D1 | per‑modality | single (fused) | single shared | Fusion (§4) combines specialized encoders into one latent; generic decoder. |
| **EN‑L1‑DN** | per‑modality | single (fused) | per‑modality | **Recommended Phase‑1 default**: classic multimodal VAE family (MVAE/MMVAE‑style). |
| EN‑LN‑D1 | per‑modality | several | single shared | Specialized encoders feed several independent latents; the single decoder assembles all of them (§2.2). |
| EN‑LN‑DN | per‑modality | several | per‑modality | Most expressive / most flexible: a full routing graph between encoders, latent spaces, and decoders. |

The table above is a simplified overview. For any "several latent spaces" row, the actual topology (which encoder feeds which latent space(s), which decoder consumes which latent space(s), and with what assembler) is a further configuration choice, detailed in §2.2.

### 2.2 Latent routing graph

"Several latent spaces" does not mean a fixed shared/private split. It means the model can have **any number of independent latent spaces**, each with its own posterior, its own prior, and its own KL term, wired to encoders and decoders through a **configurable routing graph**:

- **Encoder → latent.** An encoder can feed exactly one latent space, or fan out to several, e.g. one shared trunk with several independent projection heads, each producing its own `(mu, logvar)`. When *several encoders* feed the *same* latent space, that's the Fusion problem from §4 (PoE / MoE / concat+MLP / cross-attention). Fusion combines distribution parameters, before sampling.
- **Latent → decoder.** A decoder can consume exactly one latent space, or several. When it consumes several, an **Assembler** combines the already-realized latent vectors into one input tensor. Candidate assemblers: `concat`, `sum`, `average`: a pluggable registry following the same pattern as Fusion, but operating on realized vectors rather than distribution parameters (no probabilistic machinery needed here: the vectors already exist, this is just merging them for the decoder's input layer).

This **generalizes** the earlier "shared + private" idea rather than replacing it. Shared+private is just one specific routing graph: all encoders feed a `z_shared` via Fusion, each encoder also feeds its own untouched `z_private_m`, and each modality's decoder consumes `{z_shared, z_private_m}` via `concat`. The framework supports that topology as *one configuration among others*, not as the hardcoded meaning of "several latent spaces."

**Constraints to validate at model-construction time:**
- Every latent space must have at least one encoder feeding it and at least one decoder consuming it: no orphan latent spaces.
- `sum` and `average` assemblers require all their input latent spaces to share the same dimensionality; `concat` has no such restriction.

---

## 3. Glossary (keep consistent everywhere)

- **Modality**: a data type/source (1D signal, image, ...). A modality may have several concrete datasets (e.g. SAXS is one dataset within the "1D signal" modality).
- **Encoder**: network mapping one modality's raw input to distribution parameters (or to features destined for fusion).
- **Fusion module**: combines outputs of multiple *encoders* into a single set of latent distribution parameters, for one given latent space. Operates on distribution parameters, before sampling.
- **Latent space**: an independent probabilistic code `z_i`, with its own posterior, its own prior, and its own KL term. A model has one or several; when several, they are wired to encoders and decoders through the routing graph (§2.2), not a fixed shared/private split.
- **Assembler**: combines several already-realized *latent vectors* into one decoder input (`concat`, `sum`, `average`, ...). Distinct from Fusion: Fusion acts on distribution parameters before sampling; the Assembler acts on realized vectors after.
- **Routing graph**: the bipartite wiring {encoders → latent spaces} and {latent spaces → decoders}, set via config, that determines connectivity whenever there is more than one latent space.
- **Decoder**: network mapping latent(s) back to a modality-specific reconstruction.
- **Configuration**: the tuple {encoder cardinality, latent cardinality, decoder cardinality, routing graph, per-latent-space fusion strategy, per-decoder assembler strategy, residual flag} that fully determines a model instance.

---

## 4. Fusion strategies (encoder → latent)

No single fusion strategy is hardcoded, and fusion is not a single model-wide choice: it is selected **per latent space**, for whichever latent spaces are fed by more than one encoder (see the routing graph, §2.2). A latent space fed by exactly one encoder needs no fusion strategy at all. Each fusion assignment can optionally use residual connections (a config flag alongside it).

| Strategy | Idea | Handles missing modalities natively? |
|---|---|---|
| **Product-of-Experts (PoE)** | Each modality contributes a Gaussian "expert"; experts are multiplied together (MVAE-style). | Yes: simply drop the missing expert's term. |
| **Mixture-of-Experts (MoE)** | Each modality proposes its own latent estimate; a mixture combines them (MMVAE-style). | Yes: natively subset-tolerant. |
| **Concatenation + MLP** | Concatenate encoder outputs, project through an MLP. Simple baseline. | No: needs an explicit imputation/masking strategy if modalities can be missing. |
| **Cross-attention / transformer fusion** | Treat each modality's encoding as a token; a transformer block fuses them. | Yes: missing modalities are simply omitted tokens; also the most natural bridge toward the long-term "world model" direction. |

This is implemented as a **strategy pattern**: an `AbstractFusion` interface with each strategy as a subclass, registered by name (e.g. `poe`, `moe`, `concat_mlp`, `cross_attention`) so a new one can be added without touching existing code.

See §2.2 for the symmetric decoder-side mechanism (the Assembler) used when a decoder consumes more than one independent latent space.

---

## 5. Missing-modality robustness

Explicitly desired, but achieved as a **side effect of the fusion strategy choice** rather than as separate machinery:

- Using PoE, MoE, or cross-attention fusion, the model can be trained and queried with any subset of available modalities.
- Recommended training technique: **modality dropout**, randomly hiding one or more modalities per training step so the model learns to be robust to their absence at inference.
- Concatenation-based fusion is the one strategy that does *not* get this for free; if used with missing modalities it needs an explicit masking/imputation scheme. This should be documented as a known limitation, not silently patched.

---

## 6. Data modality roadmap

| Phase | Modality | Notes for encoder/decoder design |
|---|---|---|
| 1 | 1D signal (generic), first dataset: SAXS (intensity vs. *q*) | Variable-length series. Preprocessing is dataset-specific (e.g. log-scale intensity is standard for SAXS) and must live outside the architecture, in the data pipeline. Candidate encoders: 1D CNN or small transformer over the series. |
| 2 | Images | Candidate encoders: CNN (ResNet-style) or ViT, depending on resolution/dataset size. |
| 3+ | Open-ended (audio, tabular, text, time series, graphs, point clouds, ...) | Adding a modality = writing one new `Encoder` + `Decoder` subclass and registering them. Zero changes to core framework code. |

---

## 7. Long-term direction (not built now, but not blocked either)

Noted so that early interface design doesn't foreclose these later:
- Hierarchical or sequential/temporal latents (world-model-style state prediction).
- Discrete/VQ latent codes as an alternative to continuous Gaussian latents.
- Autoregressive or learned priors over the latent space instead of a fixed standard normal.
- Scaling toward larger backbones (transformer-heavy encoders/decoders/fusion).

These are **future directions**, listed to keep core abstractions (encoder/decoder/fusion interfaces, latent representation) general enough to extend without a rewrite, not a commitment to build them now.

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
│       ├── assemblers/
│       │   ├── base.py          # AbstractAssembler interface
│       │   ├── registry.py
│       │   ├── concat.py
│       │   ├── sum.py
│       │   └── average.py
│       ├── latent/
│       │   ├── routing_graph_builders/
│       │   │   ├── single.py        # preset: one latent space feeding every decoder
│       │   │   ├── shared_private.py # preset: shared + private latent spaces
│       │   └── base.py          # LatentSpace, RoutingGraph, validateRoutingGraph
│       ├── models/
│       │   └── global_vae.py    # assembles encoders + fusion + latent + decoders from a routing graph
│       ├── losses/
│       │   ├── reconstruction.py
│       │   └── kl.py            # aggregates KL across latent spaces
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

## 9. Illustrative config examples

Not final: these show how the registry + config-driven pattern is meant to operate in practice, matching `GlobalVae`'s two constructors (`__init__` with an explicit `RoutingGraph`, and the `createSingleLatent()` convenience wrapper; see `models/global_vae.py` and ADR 0002).

**Single latent space** (`GlobalVae.createSingleLatent`, the `EN-L1-DN` Phase-1 default):

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
    mode: single              # single | several
    dim: 128
    fusion:                   # only meaningful because this latent space is fed by 2 encoders
      strategy: poe            # poe | moe | concat_mlp | cross_attention
      residual: true
  training:
    beta_schedule: linear_warmup
    modality_dropout_p: 0.15  # trains robustness to missing modalities
```

**Several independent latent spaces, no encoder fan-out** (`GlobalVae.__init__` with an explicit `RoutingGraph`; this pattern works end to end today, verified against ADR 0002):

```yaml
model:
  name: global_vae
  modalities:
    signal:
      encoder: signal_cnn_v1
      decoder: joint_decoder_v1        # shares a decoder with image: its own key, not "signal"
    image:
      encoder: resnet_encoder_v1
      decoder: joint_decoder_v1
  latent:
    mode: several
    spaces:
      z_signal:
        dim: 128
        fed_by: [signal]                # exactly one encoder: no fusion needed
      z_image:
        dim: 128
        fed_by: [image]                 # exactly one encoder: no fusion needed
    decoders_consume:
      joint: {spaces: [z_signal, z_image], assembler: concat}
  training:
    beta_schedule: linear_warmup
    modality_dropout_p: 0.15
```

**Shared plus private latent spaces** (`latent/shared_private.py`'s preset; illustrative only, does not run yet, see ADR 0002):

```yaml
model:
  name: global_vae
  modalities:
    signal:
      encoder: signal_cnn_v1
      decoder: signal_cnn_v1
    image:
      encoder: resnet_encoder_v1
      decoder: resnet_decoder_v1
  latent:
    mode: several
    spaces:
      z_shared:
        dim: 128
        fed_by: [signal, image]        # more than one encoder: needs a fusion strategy
        fusion: poe
      z_signal_private:
        dim: 32
        fed_by: [signal]                # same encoder as z_shared: this is encoder fan-out
      z_image_private:
        dim: 32
        fed_by: [image]                  # same encoder as z_shared: this is encoder fan-out
    decoders_consume:
      signal: {spaces: [z_shared, z_signal_private], assembler: concat}
      image:  {spaces: [z_shared, z_image_private],  assembler: concat}
  training:
    beta_schedule: linear_warmup
    modality_dropout_p: 0.15
```

`fed_by: [signal]` for `z_signal_private` looks identical in shape to the working example above, but `signal` is *also* listed under `z_shared`'s `fed_by`. That is the encoder fan-out case: the `signal` encoder would need to produce two independent `(mu, logvar)` pairs (one per latent space it feeds) instead of one. Building the `RoutingGraph` succeeds, but `GlobalVae` currently rejects it with `NotImplementedError` rather than silently using the same pair for both spaces.

All three examples are illustrative, not final: the actual schema still needs validation logic and a Hydra/dataclass binding (§11).

---

## 10. Coding standards

- **Language/runtime:** Python 3.11+, PyTorch (latest stable).
- **Formatting/linting:** `ruff` (lint + format), consistent import ordering.
- **Naming convention (custom, overrides PEP8 default for callables):**
  - Classes → `CamelCase` (e.g. `GlobalVae`, `SignalEncoder`, `ProductOfExperts`).
  - Variables → `snake_case` (e.g. `latent_dim`, `batch_size`).
  - Functions and methods → same rule as classes but starting lowercase, i.e. `camelCase` (e.g. `computeLoss`, `encodeSignal`, `registerEncoder`), not PEP8's usual `snake_case` for callables.
  - Since this deviates from PEP8, disable/adjust `ruff`'s naming rules (`N802`, `N803`, `N806`) in `pyproject.toml` and note the exception in the contributor docs, so linting doesn't silently "fix" it back to snake_case later.
- **Typography:** no em dashes (`—`) in code, comments, docstrings, commit messages, or project documentation. Use a period, a colon, parentheses, or two sentences instead. This is a house style rule, not a technical one, so there is no linter for it; review for it like any other style note.
- **Typing:** type hints mandatory everywhere; `mypy` run in CI.
- **Docstrings:** Google-style, mandatory on every public class/function: purpose, `Args`, `Returns`, `Raises`.
- **Modularity:** one responsibility per file; one class per file for encoders/decoders/fusion strategies/assemblers. No god-files: a base class, its registry, and every concrete strategy each get their own file (see `fusion/` and `assemblers/`).
- **Interfaces:** `AbstractEncoder`, `AbstractDecoder`, `AbstractFusion`, `AbstractAssembler` as ABCs. Every concrete implementation subclasses one of these and self-registers via a decorator (`@registerEncoder("signal")`); this is what makes "add a modality without touching the core" actually true, not just aspirational.
- **Registry population:** a class decorated with `@registerX(...)` is only registered once its module has actually been imported. Each registry-based subpackage's `__init__.py` must import every concrete implementation for that side effect (`import global_vae.assemblers.concat  # noqa: F401`, one line per file), or `getXClass(name)` raises `KeyError` even though the file exists on disk.
- **Routing graph validation:** `validateRoutingGraph` (`latent/base.py`) must run at model-construction time for every configuration, not just the Phase-1 default. It rejects orphan latent spaces, rejects a decoder that consumes more than one latent space without an assigned assembler, and checks dimensional compatibility for `sum`/`average` assemblers.
- **Config management:** Hydra + structured dataclasses (or Pydantic) for validated, composable configs. No magic strings/dicts scattered through the code.
- **Testing:** `pytest`. Unit tests per module, plus an integration test that instantiates **each of the 8 architecture combinations** end-to-end on dummy tensors (shape and gradient sanity checks). Coverage should target core logic (fusion math, loss correctness, forward/backward pass), not a vanity percentage.
- **Experiment tracking:** Weights & Biases or MLflow, logging losses, latent-space visualizations, and reconstructions per run.
- **Reproducibility:** global seed management, deterministic-mode flag documented, config snapshotted with every run.
- **Logging:** standard `logging` module, no bare `print`.
- **Version control:** Conventional Commits, semantic versioning, maintained `CHANGELOG.md`. Architectural decisions get a new ADR when they change, rather than an old ADR being edited in place (see `docs/adr/0002-*.md` for an example of one ADR superseding part of another).
- **CI:** GitHub Actions running lint, type-check, and tests on every push.
- **Documentation:** `mkdocs` + `mkdocstrings` built from docstrings; major architectural choices (e.g. "why PoE + MoE + cross-attention", "why a routing graph instead of a fixed shared/private split") recorded as short ADRs (`docs/adr/NNNN-title.md`).
- **New-modality checklist:** adding a modality means (1) subclass `Encoder`, (2) subclass `Decoder`, (3) register both, (4) add a config entry, (5) add a test. This keeps the "no limit on data types" promise operational rather than aspirational.

---

## 11. Open questions (deliberately deferred)

- Exact schema (YAML/dataclass) for expressing the routing graph in config (`fed_by`, `spaces`, `decoders_consume` in §9's second example are illustrative, not final) and how it is validated before being turned into a `RoutingGraph`.
- Which assembler operators to implement beyond `concat` / `sum` / `average` (e.g. a learned or attention-weighted assembler); not needed for Phase 1, but the registry should stay open to it.
- How `AbstractEncoder` should expose one `(mu, logvar)` pair per latent space it feeds, for encoders assigned to more than one latent space (needed for `E1-*` rows and for the shared-plus-private preset in `latent/shared_private.py`; see ADR 0002). Until this is decided, `GlobalVae` rejects that case with `NotImplementedError` rather than guessing.
- Precise loss weighting / β-VAE annealing schedule, including whether β is shared across latent spaces or set per latent space (`losses.kl.computeTotalKlLoss` already accepts either; the actual values/schedule are still undecided).
- Whether training uses raw PyTorch loops or PyTorch Lightning.
- First concrete joint signal+image dataset/task to validate the pipeline end-to-end (using SAXS as the initial signal dataset).
- Production/serving requirements, if any arise later (currently out of scope, research framework first).

---

## 12. How future Claude conversations should use this document

- Treat the terminology in §3 as canonical: don't invent new names for the same concepts.
- Any new component (encoder, decoder, fusion strategy, assembler) must follow the registry pattern in §10, not be hardcoded into the model class, and must be added to its subpackage's `__init__.py` imports so it actually registers.
- Latent spaces are independent by construction: never hardcode a shared/private split as *the* meaning of "several latent spaces"; always go through the routing graph (§2.2). Fusion is chosen per latent space, not once for the whole model.
- Build routing graphs through `RoutingGraph` directly or through a preset in `latent/` (`single.py`, `shared_private.py`, or a new one); never re-derive the same construction inline in a model class.
- If a request would violate the "no fixed fusion strategy" or "no fixed modality set" principles, flag it rather than silently narrowing the design.
- When a decision in §11 is needed to proceed, ask: don't guess and move on.
