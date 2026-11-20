# 0058: Pair integrity: optional sample ids in a batch

**Status:** accepted (roadmap item P6-8, extension "pair integrity"; it adds one module, one argument to each pair loss
term, a check in the pair evaluation and one field of its report, and changes nothing for a batch that carries no ids)
**Date:** 2026-11-16

## Context

A paired batch is a dict in which row `i` of every modality is the same sample, and nothing in the tensors says so
(roadmap finding I-8). A loader that joins two sources can break that quietly: each side sorted differently, one side
filtered, one side shuffled on its own. Stage 3 then trains `latent_alignment` and `cross_reconstruction` on strangers,
and the only symptom is a retrieval at chance, which reads the same as an alignment that did not work. P6-6 added a
statistical check after the fact (the Procrustes map fitted on the training pairs must recover the partner above
chance, `checkPairIntegrity`, ADR 0056), which says "suspect" and cannot say where. The roadmap item asks for the
direct one: a loader that knows where each row came from says so, and the terms compare.

## Decision

**1. One id entry per modality, named `sample_id.<modality>`.** An integer tensor of shape `(batch,)` holding the id that
*that modality's own source* gave to row `i` (`data/sample_ids.py`: `sampleIdKey`, `isSampleIdKey`). One shared id
cannot do the job, since a join that is wrong gives the same shared id to different rows: only two ids that come from
two sources can differ. The entries are tensors because a batch is moved to the device tensor by tensor
(`Trainer._moveBatchToDevice`) and integers because they are compared for equality. They are not modalities:
`GlobalVae.selectEncoderInputs` keeps only the names of encoders, the reconstruction loss walks the decoders, and a
test pins both. `hashSampleIds(names)` turns file names or record keys into 63-bit ids from a BLAKE2b digest, so the
same name is the same id in every process (Python's `hash` of a string changes from run to run).

**2. `checkSampleIds(batch, modalities, where)`.** The ids of the modalities asked for are compared with each other.
It returns `None` when no modality has an entry (the ids are optional, and a batch without them is trusted as before)
and the number of rows compared otherwise. It raises a `ValueError`, which names who checked (`where`), when:

- some of the modalities have ids and others do not (it cannot be checked, and a half-labelled loader is a mistake
  to hear about, not to skip);
- an entry is not a one-dimensional integer tensor with one id per row of its modality;
- the ids of two modalities differ. The message gives how many rows differ, the first three with the id each modality
  gave them (`row 0 (signal 17, image 3)`), and what to look at: a sort, a filter or a shuffle applied to one
  modality only.

**3. The two pair terms call it.** `requireSameSamples` in `losses/terms/pairs.py` runs after `requireModalitiesInBatch`
on the modalities the pairs of the term need, in `latent_alignment` and `cross_reconstruction`, in training and in
validation. It is on by default and `check_sample_ids: false` turns it off for a term (a loader whose ids are known
to be of another kind). The check costs one comparison of two `(batch,)` tensors per step. A batch that lacks a
modality is still refused for that first, by the existing rule.

**4. The pair evaluation calls it too.** `collectPairPosteriors` and `collectPairStatistics` compare the ids of every
batch they read, so `scripts/evaluate.py --latent-pair` stops on a mismatched loader. `PairPosteriors.sample_ids_checked`
counts the rows compared, `PairIntegrity.sample_ids_checked` sums the held-out and the training rows, and the report
says in one sentence whether the pairs were checked by id (`72 rows agreed across the modalities`) or trusted
(`the batches carry no sample_id entries, so the pairs were not checked by id`). The statistical check of P6-6 stays: it
needs no ids and catches a join that gives the same ids to wrong rows.

## Alternatives considered

- **One shared `sample_id` entry.** Rejected: nothing to compare, as above. It would only test that the loader numbers
  its rows.
- **A nested dict `sample_id: {signal: ..., image: ...}`.** Rejected: the trainer, the evaluation and the visualization
  collectors move a batch with `tensor.to(device)` per key, and a nested dict would need every one of them to learn it.
- **A string id.** Rejected: a tensor cannot hold one, and a list of strings in a batch does not move to a device or
  survive the default collate of every loader. `hashSampleIds` is the bridge.
- **The check in the trainer, for every term.** Rejected: only the pair terms assume that rows are paired. A solo stage
  has no such rule, and an unpaired stage (ADR 0061) does not.

## Consequences

- No change for a run that carries no ids: the terms and the evaluation do what they did, and the report has one more
  sentence. A loader that gives ids to some modalities and not to others now fails, by design.
- A loader that gives the same ids to different rows (a hash of the position, say) passes the id check, and the
  statistical check of P6-6 is still what catches it.
- Limits: the ids prove what the loader says, not that the tensors are right; a loader that takes both ids from one
  list is trusted by this check as much as before.

Tests: `tests/unit/test_sample_ids.py` (the helper), `tests/unit/test_pair_loss_terms.py` (both terms and the trainer),
`tests/unit/test_latent_alignment.py` (the evaluation and the report) and `tests/integration/test_sample_ids_loader.py`
(a `Dataset` that joins two sources, through the default collate: a correct join trains, a join in another order stops
the first step and the evaluation, and the same wrong join without ids goes through, which is what the ids add).
