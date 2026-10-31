# Batch/TBPTT follow-up, 31 October 2026

The user approved the [quality diagnosis](../quality-diagnosis-r1/README.md).
This is a separately bounded continuation after the completed original wave;
it does not reopen the old Optuna study or resume stopped fits. The five new
fits below and six fixed mixtures are the complete admitted scientific scope.
Results and weights are delivered through [PR53](https://github.com/VladKochetov007/WunderFunder/pull/53)
as SemyonEpanov.

## Frozen comparison

| Arm | Batch | TBPTT rows | Accumulation | Seeds | Four-epoch updates |
|---|---:|---:|---:|---|---:|
| A | 256 | 256 | 4 | 1301 reused, 1302 fresh | 3,280 |
| B | 64 | 256 | 4 | 1301, 1302 fresh | 12,960 |
| C | 64 | 1,024 | 1 | 1301, 1302 fresh | 12,960 |

All use frozen Source4 GRU384x2, raw112 inputs, fitting-only prefix32
normalization, strict FP32, AdamW LR0.001, weight decay0.01, clip1, and the same
legacy warmup/cosine5% recipe. Every endpoint is configured epoch4, regardless
of the monitored validation curves. A/1301 is the exact existing checkpoint
`6087d52714c8020cbc60ea1c69343a32ad7c5c6187b39fb5813050ec8b0c86b6`;
it is not a new fit or an independent replication of itself.

A versus B jointly changes batch, update frequency and optimizer trajectory at
fixed epochs; it does not isolate batch at matched optimizer steps. B versus C
matches update count and nominal row windows while changing gradient history
and accumulation. Two seeds are a small screen, not a significance result.

Exactly six no-fit comparisons add 5% or10% of the previous full-primary GRU,
LSTM or Transformer to fixed TL3. The additional final panel contains three
equal two-seed arm means and their six fixed5%/10% TL3 mixtures. No coefficients
are learned on primary validation. The final composite panel was frozen after
early curves, the cheap mixture results, and A/1302's native epoch4 result were
already known, before the new Source6 exports. It is not a fresh confirmation.

## Validation and interpretation

Official boundary:10,325 eligible train fitting members, zero valid fitting
members, all1,873 valid sequences and3,528,273 scored rows after99 warmup rows.
The282 train sequences connected to valid remain excluded. No shuffled seq_ix
ordering is treated as chronology. See [ADR0044](../../../../specs/adr/0044-chronological-primary-validation.md).
Normalization, model ancestry and teachers do not fit valid. Existing valid is
development-exposed; no public/private transfer claim follows from this table.

Epoch plots compare a fixed16-sequence train diagnostic using all rows after
warmup with full official valid using is_scored. Their masks and populations
differ; the absolute train/valid gap is not an overfitting estimate.

## Execution and delivery

Remote root: `/data/epanov_s/tabdata/runtime/quality-horizon-20261031-r1`.
ROOT's bound is14:42:33-18:42:33 UTC, at most five fresh fits,7200seconds per
fit,10 engineering profile attempts, and15minutes reserved for collection.
Five original config-parser failures happened before CUDA; five repaired native
Source4 profiles passed actual50-update runs and independent engineering review.
No fitting code or old frozen source was changed by that representation repair.

GPU0/1/2/6/7 host the five independent fits;3/4/5 replayed the three existing
TL3 members and exited. Existing NIER/MPS co-runs are preserved. All owned GPU
jobs have detached process identities and external timeouts. The collector waits
for all five terminal process receipts and empty owned process groups before
starting the final export. Only complete exit0 epoch4 checkpoints enter the
final table and model archive; missing/partial cells remain explicit.

The native journal publisher and local mirror expose five `qh-*` runs at
http://127.0.0.1:18777/ without overwriting the previous40 views. Durable collection
recomputes canonical global WP and checks independent raw-label arithmetic,
then publishes the table, standalone curves and a SHA-verified checkpoint release.
The finalizer checks that no owned GPU-producing command remains and removes
only its inode-qualified staged GitHub credential. It never signals NIER/MPS.

Published [fixed-mixture results](fixed-mixtures/REPORT.json),
[execution contract](operator/CONTRACT_R2.json), source/config/reviewer receipts
and the final comparison panel preserve the actual receiving artifacts.
Final scoring and checkpoint delivery status must be read from the final
receipts; preparation of these workers is not completion evidence.
