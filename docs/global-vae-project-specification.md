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
1. 1D signals: a generic modality (any scalar series indexed by one variable). 
2. Images

Other signal sources (spectroscopy, sensor time series, etc.) should slot into the same "signal" encoder/decoder family later, with only preprocessing differing, not the architecture.

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

"Several latent spaces" does not mean a fixed shared/private split. It means the model can have **any number of independent latent spaces**, each with its own posterior, its own prior, and its own egularization term (§2.3), wired to encoders and decoders through a **configurable routing graph**:

- **Encoder -> latent.** An encoder produces one `(mu, logvar)` output; that same output can feed one latent space directly, or several. When it feeds several, each encoder -> latent edge may apply an optional **Latent Head** (identity by default, or a small linear/MLP projection when the target latent space's dimensionality or purpose differs) to adapt the shared encoder output before it becomes that latent space's own parameters. When *several encoders* feed the *same* latent space, that's the Fusion problem from §4 (PoE / MoE / concat+MLP / cross-attention), applied to the (optionally head-adapted) outputs. Fusion combines distribution parameters, before sampling.
- **Latent -> decoder.** A decoder can consume exactly one latent space, or several. When it consumes several, an **Assembler** combines the already-realized latent vectors into one input tensor. Candidate assemblers: `concat`, `sum`, `average`, `weighted_sum` (learned per-space weights), and `attention` (cross-attention over latent vectors, no fixed-dimensionality requirement) . a pluggable registry following the same pattern as Fusion, but operating on realized vectors rather than distribution parameters (no probabilistic machinery needed here: the vectors already exist, this is just merging them for the decoder's input layer).

This **generalizes** the earlier "shared + private" idea rather than replacing it. Shared+private is just one specific routing graph: all encoders feed a `z_shared` via Fusion, each encoder also feeds its own untouched `z_private_m`, and each modality's decoder consumes `{z_shared, z_private_m}` via `concat`. The framework supports that topology as *one configuration among others*, not as the hardcoded meaning of "several latent spaces."

**Constraints to validate at model-construction time:**
- Every latent space must have at least one encoder feeding it and at least one decoder consuming it: no orphan latent spaces.
- `sum` and `average` assemblers require all their input latent spaces to share the same dimensionality; `concat` has no such restriction.

### 2.3 Latent regularization
 
KL divergence to a standard normal prior is the default regularization term for a latent space, but it must not be hardcoded as the only option. Each latent space's regularization is a **pluggable strategy** (`AbstractLatentRegularizer`, registered like Fusion / Assembler / Latent Head), so alternatives . Maximum Mean Discrepancy (WAE-style), free-bits KL, or eventually a learned/autoregressive prior (§7) . can be added without touching the model class.
 
The weight applied to a latent space's regularization term (`beta`) is likewise flexible. The three common patterns below are meant to be expressible through the same config mechanism, not mutually exclusive code paths:
- a single constant shared by every latent space (simplest, no annealing);
- a global schedule (e.g. linear warm-up, cyclical annealing) shared by every latent space;
- a per-latent-space value or schedule, letting different latent spaces (e.g. a shared vs. a private code) be regularized differently.
`losses/regularizers/` holds the registry of regularization strategies; the schedule (constant, or a function of training step) is a separate, orthogonal config concern from *which* regularizer is used for a given latent space.

---

## 3. Glossary (keep consistent everywhere)
 
- **Modality**: a data type/source (1D signal, image, ...). A modality may have several concrete datasets (e.g. SAXS is one dataset within the "1D signal" modality).
- **Encoder**: network mapping one modality's raw input to distribution parameters (or to features destined for fusion).
- **Latent head**: optional per-edge module adapting an encoder's `(mu, logvar)` output to a specific latent space's dimensionality/purpose, used when that encoder feeds more than one latent space. Identity by default.
- **Fusion module**: combines outputs of multiple *encoders* into a single set of latent distribution parameters, for one given latent space. Operates on distribution parameters, before sampling.
- **Latent space**: an independent probabilistic code `z_i`, with its own posterior, its own prior, and its own regularization term (not necessarily KL divergence, see §2.3). A model has one or several; when several, they are wired to encoders and decoders through the routing graph (§2.2), not a fixed shared/private split.
- **Assembler**: combines several already-realized *latent vectors* into one decoder input (`concat`, `sum`, `average`, `weighted_sum`, `attention`, ...). Distinct from Fusion: Fusion acts on distribution parameters before sampling; the Assembler acts on realized vectors after.
- **Routing graph**: the bipartite wiring {encoders -> latent spaces} and {latent spaces -> decoders}, set via config, that determines connectivity whenever there is more than one latent space.
- **Decoder**: network mapping latent(s) back to a modality-specific reconstruction.
- **Transform**: a generic, invertible tensor operation applied as preprocessing/postprocessing (`log`, `standardize`, `resample`, §6.2). Distinct from an Encoder/Decoder: a transform never has learnable parameters and knows nothing about latent spaces; distinct from a Dataset/DataModule: a transform is a pure function of a tensor, with no knowledge of files, pairing, or splits.
- **Configuration**: the tuple {encoder cardinality, latent cardinality, decoder cardinality, routing graph, per-latent-space fusion strategy, per-decoder assembler strategy, residual flag} that fully determines a model instance.


---

## 4. Fusion strategies (encoder -> latent)

No single fusion strategy is hardcoded, and fusion is not a single model-wide choice: it is selected **per latent space**, for whichever latent spaces are fed by more than one encoder (see the routing graph, §2.2). A latent space fed by exactly one encoder needs no fusion strategy at all. Each fusion assignment can optionally use residual connections (a config flag alongside it).

| Strategy | Idea | Handles missing modalities natively? |
|---|---|---|
| **Product-of-Experts (PoE)** | Each modality contributes a Gaussian "expert"; experts are multiplied together (MVAE-style). | Yes: simply drop the missing expert's term. |
| **Mixture-of-Experts (MoE)** | Each modality proposes its own latent estimate; a mixture combines them (MMVAE-style). | Yes: natively subset-tolerant. |
| **Concatenation + MLP** | Concatenate encoder outputs, project through an MLP. Simple baseline. | No: needs an explicit imputation/masking strategy if modalities can be missing. |
| **Cross-attention / transformer fusion** | Treat each modality's encoding as a token; a transformer block fuses them. | Yes: missing modalities are simply omitted tokens; also the most natural bridge toward the long-term "world model" direction. |

This is implemented as a **strategy pattern**: an `AbstractFusion` interface with each strategy as a subclass, registered by name (e.g. `poe`, `moe`, `concat_mlp`, `cross_attention`) so a new one can be added without touching existing code.

See §2.2 for the symmetric decoder-side mechanism (the Assembler) used when a decoder consumes more than one independent latent space, and for the Latent Head, the analogous mechanism on the encoder side when a single encoder feeds more than one latent space.

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

### 6.1 Near-term milestones
 
The genericity described above (§1, §6) is a design constraint on the architecture, not the order in which things get built. The concrete build order is:
 
1. **A working single-modality signal VAE.** One encoder, one latent space, one decoder . no fusion, no second modality, just `signal -> z -> signal` . trained end to end on SAXS data, with the ability to inspect training curves and visualize the latent space. This validates the base building blocks (encoder, latent, decoder, losses) in isolation before any multimodal machinery is exercised.
2. **A paired signal + image setup.** Two datasets (signal, image) that are currently separate get associated into (signal, image) pairs. The exact pairing mechanism (matching by filename/sample-ID convention, or otherwise) is still open (§11). This is what exercises Fusion (§4) and the `EN-L1-DN` default (§2.1) for the first time.
3. Everything else in the roadmap table above (further modalities, richer latent topologies) comes after milestones 1 and 2 are working end to end.
Future Claude conversations should default to whichever milestone is currently active, rather than jumping straight to the fully general multimodal machinery.

### 6.2 Generic data transforms

Spec §6 already anticipated that preprocessing differs per dataset/modality
while the architecture does not ("SAXS-specific preprocessing... belongs in
transforms, not in the encoder"; other signal sources "should slot into the
same... family later, with only preprocessing differing"). `data/transforms/`
is where the framework draws the line between the two:

- **Dataset loading, matching/pairing samples across modalities, and
  train/val/test splitting stay entirely the caller's own responsibility**
  (`DataConfig.loader_factory`, §9), permanently, not merely pending an open
  question. There is no reusable structure to extract from this: how to read
  a file, how to associate a signal with an image, how to split — all of it
  is inherently specific to one dataset/user, and a framework-provided
  `datamodule.py` would either be a no-op wrapper or would have to guess at
  dataset-specific behavior. `datamodule.py` therefore does not exist and is
  not planned.
- **Generic, invertible tensor operations belong in the framework**, because
  they carry no dataset-specific structure at all: an elementwise log, a
  normalization by known statistics, a resampling to a fixed grid, are the
  same operation whether applied to a SAXS curve, a sensor reading, or a
  pixel value. `data/transforms/` holds exactly these, following the same
  `AbstractTransform` + registry pattern as every other pluggable strategy in
  this codebase (`apply`/`inverse`, self-registered via
  `@registerTransform(name)`, e.g. `log`, `standardize`, `resample`), plus
  `ComposeTransform` for chaining several into one invertible pipeline.
  `DataConfig.transforms` (§9) is a list of these, by registry name;
  `config.data.buildTransformPipeline` resolves it into a single composed
  callable, whose `.inverse` is what
  `visualization.reconstruction_plot`'s own `inverse_transform` hook expects.
  Nothing in this framework calls `buildTransformPipeline` automatically —
  a caller's `loader_factory` may use it, or preprocess data its own way
  entirely; the framework only provides the reusable operation.

**Hard requirement: every transform in `data/transforms/` must be fully
generic across dimensionality.** Data in this framework can be 1D, 2D, 3D,
multi-channel, or anything else; a transform's behavior with respect to
shape is expressed only through explicit, caller-supplied parameters (e.g.
`ResampleTransform`'s `num_spatial_dims`), never by hardcoding a
dimensionality (no separate "1D version" / "2D version" of a transform) or
by encoding anything specific to one dataset or modality (no "SAXS",
no modality name, anywhere in this subpackage). A transform whose logic
cannot be written this way does not belong in `data/transforms/`; it belongs
in the caller's own data pipeline, exactly like dataset loading already does.
This mirrors, and is held to the same standard as, the existing
architecture-level rule that adding a modality's Encoder/Decoder must never
require touching the core (§10, §12) — here applied to preprocessing instead
of model architecture.

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
├── examples/                     # runnable, self-contained pipeline walkthroughs on
│   │                             # synthetic data (distinct from scripts/, which need a
│   │                             # user-supplied factory/config; see examples/README.md)
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
│       ├── heads/
│       │   ├── base.py          # AbstractLatentHead interface
│       │   ├── registry.py
│       │   ├── identity.py      # default: no-op passthrough
│       │   └── linear.py        # small learned projection between an encoder output and a latent space
│       ├── assemblers/
│       │   ├── base.py          # AbstractAssembler interface
│       │   ├── registry.py
│       │   ├── concat.py
│       │   ├── sum.py
│       │   ├── average.py
│       │   ├── weighted_sum.py
│       │   └── attention.py
│       ├── latent/
│       │   ├── base.py          # LatentSpace, RoutingGraph, validateRoutingGraph
│       │   ├── single.py        # preset: one latent space feeding every decoder
│       │   └── shared_private.py # preset: shared + private latent spaces
│       ├── models/
│       │   └── global_vae.py    # assembles encoders + heads + fusion + latent + decoders from a routing graph
│       ├── losses/
│       │   ├── reconstruction.py
│       │   └── regularizers/
│       │       ├── base.py      # AbstractLatentRegularizer interface
│       │       ├── registry.py
│       │       └── kl_standard_normal.py  # default strategy
│       ├── data/
│       │   ├── datamodule.py    # NOT built, and not planned: see §6.2 (permanent scope boundary)
│       │   └── transforms/      # AbstractTransform interface + registry: log, standardize, resample (§6.2)
│       ├── training/
│       │   └── trainer.py       # raw PyTorch loop for now (see §10); Lightning/Fabric later
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
    beta: linear_warmup       # constant | schedule name | per-space dict (see §2.3)
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
    beta: linear_warmup       # constant | schedule name | per-space dict (see §2.3)
    modality_dropout_p: 0.15
```
 
**Shared plus private latent spaces** (`latent/shared_private.py`'s preset . the encoder fan-out case, resolved via the Latent Head, §2.2):
 
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
        head: linear                    # projects the signal encoder's 128-dim output down to 32
      z_image_private:
        dim: 32
        fed_by: [image]                  # same encoder as z_shared: this is encoder fan-out
        head: linear                    # projects the image encoder's 128-dim output down to 32
    decoders_consume:
      signal: {spaces: [z_shared, z_signal_private], assembler: concat}
      image:  {spaces: [z_shared, z_image_private],  assembler: concat}
  training:
    beta: linear_warmup            # constant | schedule name | per-space dict (see §2.3)
    modality_dropout_p: 0.15
```
 
`signal` feeds both `z_shared` (through Fusion, combined with `image`) and its own `z_signal_private`. This is the encoder fan-out case from §2.2: the `signal` encoder produces one `(mu, logvar)` pair, and the `head: linear` on the `z_signal_private` edge adapts that shared output down to this latent space's own dimensionality (128 -> 32), so the two latent spaces stay genuinely independent instead of accidentally sharing the same values.

**Data preprocessing pipeline** (§6.2, `DataConfig.transforms`, resolved by `config.data.buildTransformPipeline`):

```yaml
data:
  loader_factory: my_project.data:buildSignalDataloaders
  train_path: data/raw/train
  transforms:
    - name: log                 # data.transforms registry key
      kwargs:
        eps: 1.0e-6
    - name: standardize
      kwargs:
        mean: 0.42               # computed from the caller's own training split;
        std: 1.13                # never guessed by this framework (see StandardizeTransform)
```
 
All of these examples are illustrative, not final: the actual schema still needs validation logic and a Hydra/dataclass binding (§11).
 
---

## 10. Coding standards

- **Language/runtime:** Python 3.11+, PyTorch (latest stable).
- **Formatting/linting:** `ruff` (lint + format), consistent import ordering.
- **Naming convention (custom, overrides PEP8 default for callables):**
  - Classes -> `CamelCase` (e.g. `GlobalVae`, `SignalEncoder`, `ProductOfExperts`).
  - Variables -> `snake_case` (e.g. `latent_dim`, `batch_size`).
  - Functions and methods -> same rule as classes but starting lowercase, i.e. `camelCase` (e.g. `computeLoss`, `encodeSignal`, `registerEncoder`), not PEP8's usual `snake_case` for callables.
  - Property not calculated -> snake_case
  - Since this deviates from PEP8, disable/adjust `ruff`'s naming rules (`N802`, `N803`, `N806`) in `pyproject.toml` and note the exception in the contributor docs, so linting doesn't silently "fix" it back to snake_case later.
- **Typography:** no em dashes (`—`) and in code, comments, docstrings, commit messages, or project documentation. Use a period, a colon, parentheses, or two sentences instead. This is a house style rule, not a technical one, so there is no linter for it; review for it like any other style note. Same for the arrows (`→`) you can use `->` instead
- **Typing:** type hints mandatory everywhere; `mypy` run in CI.
- **Docstrings:** Google-style, mandatory on every public class/function: purpose, `Args`, `Returns`, `Raises`.
- **Modularity:** one responsibility per file; one class per file for encoders/decoders/fusion strategies/assemblers/heads/regularizers/transforms. No god-files: a base class, its registry, and every concrete strategy each get their own file (see `fusion/`, `assemblers/`, `heads/`, `losses/regularizers/`, `data/transforms/`).
- **Interfaces:** `AbstractEncoder`, `AbstractDecoder`, `AbstractFusion`, `AbstractAssembler`, `AbstractLatentHead`, `AbstractLatentRegularizer`, `AbstractTransform` as ABCs. Every concrete implementation subclasses one of these and self-registers via a decorator (`@registerEncoder("signal")`); this is what makes "add a modality without touching the core" actually true, not just aspirational.
- **Registry population:** a class decorated with `@registerX(...)` is only registered once its module has actually been imported. Each registry-based subpackage's `__init__.py` must import every concrete implementation for that side effect (`import global_vae.assemblers.concat  # noqa: F401`, one line per file), or `getXClass(name)` raises `KeyError` even though the file exists on disk.
- **Routing graph validation:** `validateRoutingGraph` (`latent/base.py`) must run at model-construction time for every configuration, not just the Phase-1 default. It rejects orphan latent spaces, rejects a decoder that consumes more than one latent space without an assigned assembler, checks dimensional compatibility for `sum`/`average` assemblers, and rejects an encoder -> latent edge whose `head` output dimension does not match the target latent space's declared `dim`.
- **Latent regularization:** never hardcode `LatentSpace.klDivergence` (or an equivalent KL-only computation) as the only regularization path inside the model class; route it through the `AbstractLatentRegularizer` registry (§2.3) so alternative strategies (MMD, free-bits, learned priors) can be swapped in via config.
- **Data transforms:** never hardcode a preprocessing step (log, normalization, resampling) inline in an encoder, a decoder, or the data pipeline; route it through the `AbstractTransform` registry (`data/transforms/`, §6.2) instead, and keep every transform fully generic across dimensionality (§6.2's hard requirement).
- **Training loop:** a raw PyTorch loop (`training/trainer.py`) for now, favoring transparency while the architecture (routing graph, registries) is still actively changing; migrate to PyTorch Lightning (or `Lightning Fabric` as an intermediate step) once the model design stabilizes and multi-GPU/scaling needs become concrete.
- **Config management:** Hydra + structured dataclasses (or Pydantic) for validated, composable configs. No magic strings/dicts scattered through the code.
- **Testing:** `pytest`. Unit tests per module, plus an integration test that instantiates **each of the 8 architecture combinations** end-to-end on dummy tensors (shape and gradient sanity checks). Coverage should target core logic (fusion math, loss correctness, forward/backward pass), not a vanity percentage. This includes, at minimum :
  - Unit tests for every self-registration registry (encoders, decoders, fusion, assemblers, regularizers, beta schedules, **data transforms**), covering registration, lookup, and the duplicate/unknown-name error paths.
  - Unit tests for the beta-schedule strategies' value correctness (constant, linear warm-up, cyclical annealing, and any further schedule).
  - Unit tests for the data transforms (§6.2), **notably their invertibility**: `apply` then `inverse` must round-trip to floating-point precision for exact transforms (`log`, `standardize`, any composition of them via `ComposeTransform`), and to a documented, tested tolerance for lossy ones (`resample`); exercised across more than one dimensionality per transform, per §6.2's genericity requirement.
  - **No test is required for `datamodule.py`**: per §6.2, it is a permanent scope boundary, not deferred code, so there is nothing to test.
  - A **smoke test for the trainer**: a handful of optimizer steps on dummy (non-real) data, verifying that the loss decreases over that short run and that no parameter's gradient is left `None` after a step.
  - An **integration test with the real modules** for spec §6.1 milestone 1 specifically: `OneDCnnEncoder` + `OneDCnnDecoder`, assembled via `GlobalVae.createSingleLatent` with no fusion strategy (single modality), covering forward-pass shapes, gradient flow, and a short trained-end-to-end run where the loss decreases. This is distinct from, and in addition to, the dummy-encoder/decoder `EN-L1-DN` integration test already required above: dummy modules validate the assembly/routing machinery, this test validates that the actual Phase-1-milestone modules work together correctly.
- **Experiment tracking:** Weights & Biases or MLflow, logging losses, latent-space visualizations, and reconstructions per run.
- **Reproducibility:** global seed management, deterministic-mode flag documented, config snapshotted with every run.
- **Logging:** standard `logging` module, no bare `print`.
- **Version control:** Conventional Commits, semantic versioning, maintained `CHANGELOG.md`. Architectural decisions get a new ADR when they change, rather than an old ADR being edited in place (see `docs/adr/0002-*.md` for an example of one ADR superseding part of another).
- **CI:** GitHub Actions running lint, type-check, and tests on every push.
- **Documentation:** `mkdocs` + `mkdocstrings` built from docstrings; major architectural choices (e.g. "why PoE + MoE + cross-attention", "why a routing graph instead of a fixed shared/private split") recorded as short ADRs (`docs/adr/NNNN-title.md`).
- **New-modality checklist:** adding a modality means (1) subclass `Encoder`, (2) subclass `Decoder`, (3) register both, (4) add a config entry, (5) add a test. This keeps the "no limit on data types" promise operational rather than aspirational.
- **Flexibility:** The code should be as flexible and generalist as possible so that it can be adapted to suit all users. In other words, avoid hard-coded values that may be significant to the user; instead, define default values. Another example: the signal encoder is not general-purpose, whereas the 1D CNN encoder and the ResNet 1D encoder are

---

## 11. Open questions (deliberately deferred)

- Exact schema (YAML/dataclass) for expressing the routing graph, the Latent Head assignments, and the regularizer/beta configuration . §2.2, §2.3, and §9 are illustrative, not final; the validation logic (`validateRoutingGraph` and its Hydra/Pydantic binding, §10) still needs to be written.
- Which assembler operators to implement first: `concat`, `sum`, `average`, `weighted_sum`, and `attention` are all wanted (§2.2) . order of implementation is open, but none is deferred indefinitely.
- Which regularizer strategies to implement beyond the default `kl_standard_normal` (§2.3) . MMD and free-bits KL are candidates; not needed immediately, but the registry should stay open.
- Exact pairing mechanism for the first paired signal+image dataset (§6.1) . matching by filename/sample-ID convention is the likely approach, but the concrete scheme isn't decided yet. (Note: this is a *pairing* question, i.e. a `loader_factory` concern, distinct from and not blocking §6.2's generic transforms, which are already implemented and do not depend on how pairing is eventually done.)
- Precise β schedule hyperparameters (warm-up length, cyclical period, per-space values) . the mechanism supports all of this (§2.3); only the actual numbers are still to be tuned empirically once training starts.
- Concrete production/serving target (API serving, batch inference, edge deployment, ...) . in scope longer-term (§1), but sequenced after the model is functionally complete; specific requirements aren't known yet.

**Resolved, no longer open:** whether any part of the data pipeline belongs inside the framework (§6.2). Generic, invertible preprocessing (`data/transforms/`) does; dataset loading, pairing, and splitting (`datamodule.py`) permanently does not, for the reasons given in §6.2. This was previously bundled as one deferred item; it is now a settled design decision, not an open question.

---

## 12. How future Claude conversations should use this document

- Treat the terminology in §3 as canonical: don't invent new names for the same concepts.
- Any new component (encoder, decoder, fusion strategy, assembler, latent head, regularizer, **data transform**) must follow the registry pattern in §10, not be hardcoded into the model class (or, for transforms, into an encoder/decoder or the data pipeline), and must be added to its subpackage's `__init__.py` imports so it actually registers.
- Latent spaces are independent by construction: never hardcode a shared/private split as *the* meaning of "several latent spaces"; always go through the routing graph (§2.2). Fusion is chosen per latent space, not once for the whole model.
- Encoder fan-out to several latent spaces goes through a Latent Head (§2.2), never by giving `AbstractEncoder` multiple output heads or duplicating the encoder.
- Regularization is never hardcoded to KL-to-standard-normal inside the model class; it goes through the `AbstractLatentRegularizer` registry (§2.3).
- Data transforms (`data/transforms/`, §6.2) must stay fully generic across dimensionality (1D/2D/3D/other) and must never encode anything specific to one dataset or modality (e.g. SAXS); modality-specific preprocessing decisions (which transforms, in what order, with what parameters) stay in the caller's own data pipeline, exactly as they always have. `datamodule.py` is a permanent scope boundary (§6.2), not a pending milestone: do not build it, and do not treat its absence as a gap to fill.
- Build routing graphs through `RoutingGraph` directly or through a preset in `latent/` (`single.py`, `shared_private.py`, or a new one); never re-derive the same construction inline in a model class.
- Respect the milestone order in §6.1: don't build the fully general multimodal machinery before the single-modality signal VAE (encoder -> latent -> decoder, training + latent visualization) actually works end to end.
- If a request would violate the "no fixed fusion strategy" or "no fixed modality set" principles, flag it rather than silently narrowing the design.
- When a decision in §11 is needed to proceed, ask: don't guess and move on.
