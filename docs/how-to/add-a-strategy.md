# Add a fusion, assembler, regularizer, transform, beta schedule, or logger

Every one of these follows the exact same three-step pattern: subclass an
abstract base class, decorate it to self-register under a name, and import
the new module from its subpackage's `__init__.py` so the decorator actually
runs.

| You're adding a... | Base class | Decorator | Lives in | Import it from |
|---|---|---|---|---|
| Fusion strategy | `AbstractFusion` | `@registerFusion("name")` | `fusion/` | `fusion/__init__.py` |
| Assembler | `AbstractAssembler` | `@registerAssembler("name")` | `assemblers/` | `assemblers/__init__.py` |
| Latent regularizer | `AbstractLatentRegularizer` | `@registerRegularizer("name")` | `losses/regularizers/` | `losses/regularizers/__init__.py` |
| Data transform | `AbstractTransform` | `@registerTransform("name")` | `data/transforms/` | `data/transforms/__init__.py` |
| Beta schedule | `AbstractBetaSchedule` | `@registerBetaSchedule("name")` | `training/beta_schedules/` | `training/beta_schedules/__init__.py` |
| Experiment logger | `AbstractExperimentLogger` | `@registerLogger("name")` | `training/loggers/` | `training/loggers/__init__.py` |

## Steps

1. Create one new file per strategy — one class per file, no god-files (see
   the project's coding standards, spec §10 "Modularity").
2. Subclass the relevant base class above and implement its abstract
   method(s).
3. Decorate the class with the matching `@registerX("your_name")`. This is
   what makes it discoverable by name from config, with nothing else needing
   to import your class directly.
4. Add the one-line import to that subpackage's `__init__.py`, mirroring the
   existing lines there. Without this, `getXClass("your_name")` raises
   `KeyError` even though the file exists on disk.
5. Add a unit test covering registration, lookup, and (where relevant) the
   duplicate-name and unknown-name error paths, following the pattern of the
   existing registry tests.

## A note specific to data transforms

Transforms carry one extra constraint (spec §6.2): they must work on a
tensor of *any* shape or dimensionality, and must never encode anything
specific to one dataset or modality. Shape-dependent behavior goes through
an explicit, caller-supplied parameter (like `ResampleTransform`'s
`num_spatial_dims`), never a hardcoded branch on `x.dim()`.
