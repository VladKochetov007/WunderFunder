# Fold0 spread auxiliary builder handoff

Status: implemented and independently checked; proposed screen must still be
frozen by root/critic after clean controls finish. No GPU launch, candidate
prediction/scoring, live integration, model/feature changes, or edits to frozen
SSM/scorer/scheduler/global ledger. Original clean control artifacts remain intact.
The original future-proxy spread is retained as the matched supervision control:
its extra future-tail information may be useful even if it differs from current
spread. Neither a physical interpretation nor an improvement is established.

## Frozen TRAIN map and fitting-row cache

`selection.json` was written before reading target values. It selects512groups,
one per permitted TRAIN component, by the salt/hash ordering in preregistration.
No labels or scoring masks participate in selection. All selected components
are outside fold0 evaluation components; raw dataset identities use frozen byte
size/mtime/footer hashes from the clean split. Source and parent checkpoint hashes
were frozen before the map build.

Only selected TRAIN `i0_p0..i0_p21,t0,t1` were read to fit the map. Target-lattice
class identifies offline TRAIN supervision. Votes are restricted to nonpositive
interior target pairs with implied spread>=1, sampled at raw-code plateau starts
then every7rows. Current float32 code is rounded4 with the same pooled key grammar.
Accepted modal `(class,spread)` votes require support>=3/confidence>=.95. Class is
not supplied to the lookup or model. `all_code_votes.json` preserves all supported
and rejected counts/conflicts:33063votedcodes,6300accepted, zero observed class or
spread conflicts. The map/table SHA is frozen before application.

Cache applies that frozen map using prices only on all10535FIT TRAIN and1405FIT
VALID groups. No VALID/heldout targets, masks, evaluation rows or external records
were read for map/cache construction. One uint8 array/split encodes spread bins
0..11;255 denotes unknown. Files use compact FIT-group indices with exact ordered
row-group lists, so missing/evaluation groups raise rather than using a fallback.
Array bytes238800256, with no duplicated main feature cache.

| Cache split | Known rows | Unknown rows | Total rows | Known coverage |
|---|---:|---:|---:|---:|
| FIT TRAIN |208239240|2460760|210700000|98.8321%|
| FIT VALID |27782178|317822|28100000|98.8690%|

These are fitting-row coverage diagnostics, not independent accuracy or scores.
Unknown rows and full denominators are retained. No unseen-market guarantee follows.

## Unapplied trainer patch and loss identity

`integration.patch` adds only a supervision-cache/loss helper module and trainer
options/loading/provenance/loss plumbing. It is unapplied and `git apply --check`
passes. `integration_manifest.json` records original/proposed trainer and support
hashes. Support/source/cache commitments are in `source_freeze.json`.

CLI: `--spread-aux-mode proxy_known|current_known --spread-cache-root <this_dir>`.
Default `original` retains the original loss numerically and consumes no cache.
Both arms append the same cached uint8 supervision tensor to the batch after
existing propensity. It is not appended to x/y/features/hidden states. Pin/nonpin
batch handling keeps propensity separate. Candidate replaces only the spread
labels; control retains original proxy labels. Known-code mask is identical.

Spread CE numerator is multiplied by `(encoded!=255)` in both arms, using the
original `sw` weights and **original sw.sum denominator**. Branch/class terms keep
their exact original labels/numerators/denominator. Main, auxiliary-next, EMA and
all other losses are unchanged. Parent's actual N_SP_AUX12 and n_out23 are checked;
no parameters/head dimensions are added. All label bins use the original top-bin
clipping. Unknown spread gradients are exactly zero in both arms.

Validator checks complete/exact files, source-scoped group membership/order,
unique groups, fitVALID/evaluation disjointness, split/table/selection/array hashes,
file names, shapes, uint8 type, head dimension and full-array bin range0..11/255.

## CPU and independent evidence

All expensive phases used durable systemd services in `wnn-campaign.slice`,
CPUQuota100%, one thread, MemoryMax2GiB, no swap, runtime300s, CUDA hidden. Exact
commands/properties/exits are recorded in `durable_job_provenance.json`; job reports
and raw logs are retained. Map18.99s/peak1075032KiB; cache127.51s/peak1139128KiB;
actor proof2.18s/peak1572016KiB, exit0/CUDAuninitialized. No300s phase overrun.

`cpu_proof.json` verifies bitwise original-mode loss/gradient identity, unchanged
branch/class gradients, shared known mask, zero unknown spread gradients, original
denominator, all-known/all-unknown behavior, cache lookup/current-price agreement
on6FIT groups, exact float32 parquet alignment, sparse-group rejection and unchanged
synthetic dataset inputs/targets/scored tensors with/without propensity.

Independent primary report: `../audit/spread_aux_independent_checks_v1.json`, pass.
It checks6synthetic loss/gradient cases using an independent logsoftmax oracle,
11cache failures, rederived512group selection, reaggregation of33063recorded code
vote records to6300accepted, six raw-price cache replays, and full-array hashes/ranges.
It does not claim a raw512group target-vote replay. Peak1.674GiB/3.32s in a verified
hard2GiB/one-core scope; CUDA uninitialized. Independent builder/code audit remains
distinct from empirical competitive confirmation.

## Proposed root-owned screen and FT path

Both arms initialize from **the same exact** `c20260929_since_s3_f0@ep3` bytes,
with original normalization/features and n_out. One fixed continuation epoch0,
seed3, full clean fold0train+valid, batch32/chunk2000, fresh matched AdamW/OneCycle
and fresh matched EMA, original WMSE/struct0.3/next0.3/teacher1/propensity/since
recipe. Freeze empty, fixed checkpoint/no evaluation during fitting. Preregistration
contains proposed exact settings; root/critic still freeze budget and launch design.

After continuation, run **both** arms through separately predeclared identical
two-stage GCLP FT, seed3, fixed epoch0 per stage/no evaluation during either stage.
The label/cache options intentionally are not inherited: FT defaults original
future-proxy labels in BOTH arms. Existing GRU head-stage freeze is `inp,gru`, then
empty freeze for full FT. Cache/known-mask options are for the continuation only.
Root must declare the exact FT options, parents, epochs, bytes and budget before
any fit/evaluation; do not compare a continued raw model to an unchanged FT model.

Replacement pairs:0.5unchanged seed4FT +0.5new seed3FT (candidate/control), versus
unchanged0.5seed4FT +0.5seed3FT. Same complete frozen evaluator rows/mask/metric and
component bootstrap. Continue only if candidate pair improves unchanged pair by
>=+.0005 GlobalWP and is positive versus matched control pair, plus existing class
and uncertainty gates. Unknown masking differs from the unchanged pair; the matched
control isolates the supervision-label change conditional on that shared mask.

Architecture/native inference stays the original GRU; existing auxiliary-head
slicing discards training-only labels. No map/cache/class/known flag is deployed,
no third member is added, and fixed replacement weights remain0.5/0.5. Runtime
and packaging still require the normal verification gates if a model survives.
