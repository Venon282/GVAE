# 0035: An MLflow experiment logger, as an optional extra

**Status:** accepted (roadmap item P2-4; decision D-8 taken at its proposed default, so the spec needs no amendment)
**Date:** 2026-10-20

## Context

Spec §10 asks for experiment tracking with "Weights & Biases or MLflow, logging losses, latent-space
visualizations, and reconstructions per run". [ADR 0008](0008-experiment-loggers.md) built the seam
(`AbstractExperimentLogger`, a registry, `TrainingConfig.loggers`) and shipped CSV and TensorBoard,
and said that a third backend is "purely additive: one new file implementing `logScalar` (and
optionally `logImage`/`logFigure`/`close`), registered in `training/loggers/__init__.py`". The
roadmap (P2-4) left the choice to decision D-8 and proposed MLflow as an optional extra, since it is
open source and easy to self-host, unless the team already uses Weights & Biases. Nothing in the
repository says that it does.

Three things about MLflow decided the shape of the adapter:

- Its fluent API (`mlflow.start_run`, `mlflow.log_metric`) keeps one active run per process. A
  logger built on it would conflict with a second logger, or with the user's own `mlflow` calls.
- A tracking server is a network round trip per request, and `Trainer` logs several metrics at
  every step.
- MLflow 3 moved the local backend. The `./mlruns` directory of files is in maintenance mode, and
  `MlflowClient(tracking_uri="file:...")` raises unless `MLFLOW_ALLOW_FILE_STORE=true` is set (seen
  with 3.17.1). A SQLite file (`sqlite:///...mlflow.db`) is the supported way to run with no
  server, and it is the default of MLflow 3 when no tracking URI is set.

## Decision

**1. `MlflowLogger`, registered as `mlflow`, in `training/loggers/mlflow_logger.py`.**
`MlflowLogger(tracking_uri=None, experiment_name="global_vae", run_name=None,
artifact_location=None, tags=None, params=None)`. It subclasses `AbstractExperimentLogger`,
implements `logScalar`, `logImage`, `logFigure` and `close`, and is imported by
`training/loggers/__init__.py` like the other two, so `getLoggerClass("mlflow")` and a
`LoggerEntryConfig(name="mlflow", kwargs=...)` work. No change to `Trainer`, to the config schema or
to the other loggers.

**2. The import of `mlflow` is deferred, as for TensorBoard.** `MlflowClient` and the entity
classes are imported inside `MlflowLogger.__init__`, which raises an `ImportError` that names the
package and the install command when it is missing. Importing `global_vae.training.loggers` never
loads MLflow (a test checks it in a fresh interpreter, and `import mlflow` is slow, which is a
second reason). `pyproject.toml` gains an `mlflow` extra (`mlflow>=3.9`) and lists it under `dev`,
so the test suite runs against the real package, as it does for TensorBoard.

**3. An explicit `MlflowClient` and a run id, never the fluent API.** The logger creates its run
at construction (`create_run`, in an experiment found or created by name; if another process
creates it between the lookup and the creation, as parallel runs of a sweep can, it takes that
one) and passes the run id to every call. Two loggers, or a logger and the user's code, do not
share state. `run_id` is a public property, to query the run or attach more artifacts.

**4. What is logged.**

- *Scalars.* `logScalars` is overridden so that all the metrics of one call travel in one
  `log_batch` request (split above MLflow's limit of 1000), with `x` as the step. `logScalar` calls
  it. The `tag` does not change a metric's name, as in `TensorBoardLogger`: the step-level and the
  epoch-level names already differ (`loss/total`, `train/loss/total`).
- *Figures.* `logFigure` saves a matplotlib figure as the artifact
  `figures/<name>/<tag>_<x>.png`, with `x` zero-padded to eight digits. The tag is in the file
  name because a figure is a file, and epoch 3 and step 3 of one name must not overwrite each
  other. The figure is not closed.
- *Images.* `logImage` takes the `(C, H, W)` convention of the base class (`C` in 1, 3 or 4; an
  `(H, W)` tensor is read as grayscale), `uint8` or floating point, and calls
  `MlflowClient.log_image(key=name, step=x)`, which the MLflow UI shows in its image grid, one
  image per step. A float image is clipped to `[0, 1]` first: left alone, MLflow rescales an
  out-of-range image by its own minimum and maximum (with a warning), so two reconstructions of one
  run would end up on different scales, and the unbounded output of a decoder is common. Other
  integer dtypes and other shapes raise `TypeError` and `ValueError`.
- *Parameters, tags and the run name* are constructor arguments, recorded once at the start.
- *Names.* A name that becomes a key or an artifact path is sanitized: `/` stays a separator,
  characters other than letters, digits, `_`, `.`, `-` and space become `_`, and empty and dot-only
  segments are dropped, so a name cannot leave the run's artifact folder.

**5. Offline mode is a SQLite file.** `tracking_uri="sqlite:///runs/mlflow.db"` and
`artifact_location=Path("runs/artifacts")` (a `Path` is resolved to an absolute `file:` URI; a string
goes through, so `s3://...` works) keep everything on the machine, and
`mlflow ui --backend-store-uri sqlite:///runs/mlflow.db` reads it later (checked: it serves the
metrics, figures and images of a run). The tests of this logger run that way, one database for the
module (creating the schema costs seconds) and one experiment and artifact folder per test. The
file store is not used or documented, for the reason above. `tracking_uri=None` leaves the choice to
MLflow (`MLFLOW_TRACKING_URI`, then its own default), which is how a tracking server is used.

**6. A finished run is not locked.** `Trainer.fit` calls `onTrainEnd` from a `finally` block, and
the logger sets the status there from `sys.exc_info()`: `FINISHED`, `FAILED` for an exception,
`KILLED` for a `KeyboardInterrupt`. A crashed run is therefore not left `RUNNING` in the UI, and not
marked as a success. It then keeps accepting scalars, images and figures (MLflow allows it), for two
reasons: the figures of the final model are naturally logged after `fit` returns, and a second `fit`
on the same trainer (the notebook case of `Trainer.fit`'s docstring) continues the same run, where
`CsvLogger` fails on a closed file. `close()` sets `FINISHED` and may be repeated.

**7. A resumed run is a new MLflow run.** `buildCallbacksFromConfig` passes `append=True` only to a
logger whose constructor has an `append` parameter (ADR 0032), and this one has none: there is no
file to continue, and no run to continue unless the caller names one. A resumed run starts a new
run in the same experiment, with its steps continuing from the checkpoint's step.

## Alternatives considered

- **Weights & Biases.** It is the other backend the spec names. It needs an account and a hosted
  service by default (an offline mode exists, and its runs are synced afterwards), against a store
  that is a file on disk. D-8 asked for one adapter, and the recipe above is the same for the
  second: nothing here stops a `wandb` logger from joining later.
- **Amend the spec to name only CSV and TensorBoard.** It would close the item with no code, and
  leave a run without a way to compare runs across a project; the spec's sentence is about that.
- **The fluent API (`mlflow.log_metric` and the active run).** Shorter, and wrong for the reason in
  the context.
- **One `log_metric` per scalar** (the default of `AbstractExperimentLogger.logScalars`). Correct,
  and a request per metric per step.
- **The `./mlruns` file store**, as in older MLflow tutorials. Refused by MLflow 3 by default.
- **`mlflow-skinny`.** Smaller, but it does not carry the database layer the SQLite store needs, so
  the offline mode would need more packages named by hand.
- **Logging figures and reconstructions from the training loop.** The spec's sentence ("per run")
  is met by a logger that can store them, as TensorBoard's can; when to compute a latent projection
  or a reconstruction grid (every epoch? every ten?) is a training-loop decision with a cost, and
  not part of choosing a backend. A callback that calls `logFigure` and `logImage` can be written
  against any logger, and is not done here.

## Consequences

- Spec §10's tracking bullet is met by CSV, TensorBoard and MLflow. The spec names the backends as
  examples ("Weights & Biases or MLflow"), so it is not amended; its §8 layout line lists the new
  logger.
- `pip install -e ".[dev]"` now installs MLflow, which brings a web server and a database layer
  (Flask, SQLAlchemy, alembic, pandas, pyarrow, scikit-learn, among others) into the dev
  environment and the CI jobs. The runtime dependencies of the package are unchanged, and a user
  who wants MLflow installs the `mlflow` extra alone.
- `mlflow` ships type hints, so `mypy` (strict) checks the calls and needs no override.
- The floor of the extra is `mlflow>=3.9`, the oldest release the tests passed on with the
  dependencies pip resolves today: MLflow 3.8 and older fail to import their SQLite store against
  SQLAlchemy 2.1 (`cannot import name 'FallbackAsyncAdaptedQueuePool'`), whatever this logger does.
  The suite passes on 3.9.0, 3.10.0 and 3.17.1, on Python 3.11 and 3.13. Raise the floor if a
  release is found to break it, and add a ceiling only when MLflow 4 does.
- The tests are `tests/integration/test_mlflow_logger.py` (about 70 tests in about 15 seconds, all
  offline: scalars, batching and the 1000-metric split, images of every shape and dtype, the
  clipping, figures and the file names, name sanitization, the experiment race, the three final
  statuses through a real `Trainer.fit`, logging after `fit`, a second `fit`, construction from
  `LoggerEntryConfig`) and one subprocess run of `scripts/train.py` with the logger chosen on the
  command line (`test_train_script.py`).
- The `FAILED` and `KILLED` statuses read `sys.exc_info()`, which holds the exception being
  handled. A `fit` call made inside the caller's own `except` block, after an earlier failure,
  would be read as failed whatever happens. It is rare, and the status is a label in a UI.
- MLflow's SQLAlchemy store prints a `SADeprecationWarning` (`noload`) the first time a store is
  opened. It comes from MLflow, and is shown in the pytest summary.
- `.gitignore` gains `mlflow.db` and `mlartifacts/`, which MLflow writes in the working directory
  when it is given no location.
- Not decided here: continuing the same MLflow run on resume (a `run_id` argument would be enough);
  a callback that logs latent figures and reconstructions during training; logging checkpoints or
  models as MLflow artifacts; a Weights & Biases adapter.
