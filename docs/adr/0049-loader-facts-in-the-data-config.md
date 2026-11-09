# 0049: Loader facts in the data config

**Status:** accepted (roadmap item P5-7, finding I-12; it adds one optional field to `DataConfig`, one module to
`config/`, one optional argument to the two audio classes, and it lets `standardize` take a list; it settles the
three findings "no vocabulary in `DataConfig`", "the sample rate has no place in the data contract" and "YAML
cannot write per-column statistics" of [ADR 0039](0039-tabular-modality.md), [ADR 0040](0040-audio-modality.md)
and [ADR 0041](0041-text-modality.md), which are left as written)
**Date:** 2026-11-08

## Context

Three numbers are fixed by the loader and repeated by the model:

| Fact | Fixed by the loader when it | Repeated by the model as |
|---|---|---|
| the sample rate of an audio modality | resamples every clip to one rate | nothing: the model knows a number of samples |
| the vocabulary size of a text modality | builds the vocabulary and the one-hot rows | `vocab_size` of `text_transformer_encoder_v1` and `text_transformer_decoder_v1` |
| the categorical block sizes of a table | one-hot encodes the categorical columns | `categorical_sizes` of `tabular_mlp_decoder_v1` |

The framework never fills them in, on purpose: they are data-dependent, and a value that is not written is a value
guessed (spec §10, §12). So two files each carry a copy, the data file says nothing, and nothing compares them. The
three modality ADRs each recorded it as a finding. A model written for 16 kHz trains with no error on clips at 22.05
kHz (the same 2048 samples are 0.128 s in one case and 0.093 s in the other), a vocabulary rebuilt from a larger
corpus changes the width of the rows and not `vocab_size`, and a decoder that forgot `categorical_sizes` treats every
column of a table as numeric. Roadmap finding I-12 adds a fourth gap of the same family: a per-column mean and
standard deviation are lists in YAML, and `StandardizeTransform` took only a float or a tensor, so it refused a list
with a `TypeError` that did not name the argument (a strict `xfail` test pinned it).

Two constraints shape the answer. `config/model*.py` knows nothing about the data and `config/data.py` nothing about
the model, on purpose (their docstrings say so), so the comparison cannot live in either. And a run that sets none of
the facts must be the run it was.

## Decision

**1. `standardize` takes a list.** `mean` and `std` may be a float, a tensor, or a list or tuple of numbers (nested
lists make a matrix), which becomes a `float32` tensor, as a YAML file writes one value per column. A float or a
tensor is kept as it is, so nothing that worked changes. An error names the argument: `std must be positive
everywhere` for a non-positive element (the old message), and `mean must not be an empty list`, `std must be a list
of numbers` (ragged, or not numbers) and `mean must be a float, a tensor or a list of numbers` (any other type) for
the rest. The strict `xfail` test of `test_tabular_model_configs.py` loses its marker and passes.

**2. The facts live in the data config.** `DataConfig` gains `loader_facts: dict[str, LoaderFactsConfig] | None`,
default `None`, keyed by modality name like `transforms` and `sequence_length`. `LoaderFactsConfig` has three
optional fields: `sample_rate` (an integer in hertz), `vocab_size` and `categorical_sizes` (a list of integers, each
at least 2, like the decoder's). The loader owns the value, so the data file is where it is written; the model
repeats it as a declaration that can be checked, never the other way round. The values are checked on their own
(`validateLoaderFacts`): a positive integer, a non-empty list of integers of at least 2, and the message starts with
the path (`data.loader_facts.audio.sample_rate: expected a positive integer, got -1.`).

**3. The model declares through the arguments of its classes.** `vocab_size` and `categorical_sizes` already are
arguments. The two audio classes gain `sample_rate: int | None = None`, the last argument, kept as a read-only
property and read by nothing else: no layer, no output and no parameter name changes (a test compares them), so a
model that declares it is the model that does not. `None` means "not declared".

**4. The comparison is `checkDataContract(model, data)`, in `config/data_contract.py`.** It is the one module that
reads both config domains. Neither domain imports the other or it. It runs where a config becomes
whole and before anything is built: at the end of `loadExperimentConfig`, and in `scripts/train.py` after the
config is materialized and before the seed, the model and the data loaders (the script composes through Hydra and
does not call `loadExperimentConfig`). A run that disagrees therefore stops before the model is built, so before the
first step. A config built by hand in Python calls the function itself.

**5. What it compares.** For each modality of `loader_facts` and each fact given there, every encoder and decoder
whose instance name is that modality (from `modalities`, `encoders` or `decoders`) is compared with its argument of
the same name:

- a value in the config's `kwargs` is compared as it is (a list and a tuple are the same blocks, `16000.0` is
  `16000`); a `null` means "not declared" and is skipped;
- an argument left out is compared on the **class default** when the class has one that is not `None`, so the
  decoder of a table with blocks and no `categorical_sizes` (default `()`, every column numeric) is a disagreement,
  and the message says it leaves the value unset. An argument with no default (`vocab_size`) is left to the class,
  which raises its own error, and one whose default is `None` (`sample_rate`) is "not declared";
- a fact the data block does not give is not compared, and with `loader_facts` at `None` or empty the function
  returns at once, before it reads the model: **an experiment that sets neither value is unchanged**, and not one
  shipped experiment sets either.

**6. The error convention is that of ADR 0025: the path of the offending field comes first.** The offending field is
the model's, since the loader owns the value: `model.modalities.audio.encoder.kwargs.sample_rate: the model declares
16000, but data.loader_facts.audio.sample_rate says the loader gives 22050. The loader owns this value: set the
model to 22050, or change the loader and this block together.` A modality of `loader_facts` that no encoder or
decoder of the model is named after is refused with the data path first and the names of the model, since an entry
that matches nothing would compare nothing and hide the typo it comes from. The paths of fields under `encoders` and
`decoders` are `model.encoders.<name>.kwargs.<fact>` and `model.decoders.<name>.kwargs.<fact>`.

**7. The shipped files carry the block as a comment, not as a value.** `configs/data/audio.yaml`, `text.yaml` and
`tabular.yaml` show the block commented out, and the three model files and the three how-to pages
(`train-on-audio.md`, `train-on-text.md`, `train-on-a-table.md`) say where to write the value and what the error
looks like.

## Alternatives considered

- **Put the facts in `ModelConfig`.** The model does not own them, and `config/model*.py` would have to name data
  kinds (an audio rate) it deliberately does not know. The model already repeats them as arguments.
- **Fill the model's arguments from the data block** (`vocab_size` copied into the encoder and the decoder). It would
  remove the drift and also the check: a loader that rebuilt its vocabulary would silently resize the model, and a
  checkpoint trained with another width would fail at load time and not at the start of the run. The framework's rule
  is that a data-dependent value is written by the user, never guessed (spec §10, §12), and a fill-in would make
  `config/model*.py` read the data domain.
- **Check inside `buildModelFromConfig`.** It takes a `ModelConfig` and would need the data to compare with, which
  breaks the separation above for no gain over a function called where both configs exist.
- **A new top-level block of `ExperimentConfig` (`contract:`).** The facts describe what the loader yields, which is
  what `DataConfig` is for (`sequence_length` is the precedent), and a block outside it would be one more place to
  forget when a data file is swapped.
- **A float `sample_rate`.** A rate in hertz is a whole number in every format that carries one (8000, 16000, 22050,
  44100, 48000), and OmegaConf cannot type a union of `int` and `float`. A float that is a whole number written in a
  model's `kwargs` still compares equal.
- **Refuse a fact that no class of the modality takes.** A third-party audio encoder with no `sample_rate` may still
  want the rate recorded next to the data, and the check has nothing to say about it. The block says what the loader
  gives; the check compares what both sides state.
- **Warn for a modality that matches nothing, as `beta_schedules` does.** That warning exists because
  `configs/training/default.yaml` names a latent space that other single-latent models do not have. A data file is
  chosen per data kind, so an entry whose name matches no encoder or decoder is a typo or a data file used with the
  wrong model, and the error costs one line to fix.
- **Check `sequence_length` against `output_length` in the same function.** It is the same kind of gap, and it is not
  in the roadmap item. Doing it would make every shipped experiment that overrides one and not the other fail where it
  passed before, so it is a change to decide on its own.

## Consequences

- **A 16 kHz model and a 22.05 kHz data block fail before the first step**, through `loadExperimentConfig` and
  through `scripts/train.py` (a test runs the script and checks that no epoch starts and no metrics file is written),
  with the field of the model first. The same holds for a vocabulary and for the blocks of a table.
- **Nothing changes for a run that sets neither value.** Every shipped experiment composes with `loader_facts` at
  `None` (a test composes each); a declared `sample_rate` changes no weight, no name in the state dict and no output; a run on the shipped
  audio experiment trains as before.
- **`standardize` accepts what YAML can write.** A list becomes a `float32` tensor. A `float64` batch is promoted by
  the ordinary rules of tensor arithmetic, and a half-precision one is promoted to `float32` as with any `float32`
  tensor statistic.
- **Finding, not fixed: an omitted model value with no class default is not compared.** A text model that leaves out
  `vocab_size` raises its own error at build time, which is the right place; the check adds nothing before it.
- **Finding, not fixed: the `tabular_mixed` loss has its own `categorical_sizes`.** An entry of
  `training.reconstruction_loss` can carry sizes that differ from the decoder's and from the block (it must, for a
  model with several modalities, ADR 0048). Those live in the training config, a third domain, and are not compared.
- **Finding, not fixed: `sequence_length` against `output_length`**, as above.
- **Specification proposal, for the owner.** Section 9 describes `DataConfig` (paths, batch size, splits,
  `transforms`, `sequence_length`) and does not name `loader_facts`. A sentence would match the code: an optional
  block, keyed by modality, of the numbers the loader fixes and a model must agree with (`sample_rate`, `vocab_size`,
  `categorical_sizes`), compared with the model by `checkDataContract` before the model is built. It is not edited
  here, since the specification is the owner's.
