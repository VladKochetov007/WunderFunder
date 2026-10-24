# Independent review R2: F0-only online grade adapter

## Verdict

**Pass for root-controlled admission of the fixed exploratory grade, subject to
the root's actual resource receipt.** The seven F0/seed1001 endpoints and
four primary contrasts match the frozen question; the 11 original contrasts
remain descriptive. No fit is added, and the old all-fold panel remains
incomplete. This is exposed F0 evidence conditional on fixed predictions, not
fresh confirmation or a seed/fold estimate.

## Identity and evidence checked

- Recipe R2 SHA-256:
  `0d6cf458a895c322f94dd25ec1d228c5b3bfc5320f0da10c55027f58f84fd059`.
- Source manifest R2 SHA-256:
  `55b6ded533703d3430ecfbdc60d28ccd62d6812b249029028cea549c602d9553`.
- Label-free preflight R3 SHA-256:
  `f8fd01dab2226fdc46660e72a1cc25d1da3457429f404d0d0f036e18f01d7692`.
- The receipt reports `passed_label_free_preflight`, seven `(468, 20000, 2)`
  little-endian FP32 prediction arrays, 468 evaluation groups, 141 components,
  four primary and 11 secondary contrasts, ten ancestry runs and 27 ancestry
  edges. The recipe and scorer contrast lists agree on static inspection.
- I checked adapter source, small JSON/provenance metadata, source/recipe
  identities and the preflight JSON. I did not open target, prediction or
  sequence-cache payload arrays, run the scorer, or read labels.

## Earlier findings rechecked

The first adapter review is preserved in [REVIEW.md](REVIEW.md). Its blockers
are resolved in the current source snapshot: terminal source pins are read
from the inventory's actual `source_pins` schema; parent, initializer,
distillation, resume and propensity ancestry is traversed recursively with
source-qualified membership; the grade admission reader caps worker memory at
400,000,000 bytes and CPU at one core; and each contrast now includes the
fixed time-bin deltas. Historical cache payloads remain explicitly unhashed.

The target-cache integrity gap is also closed in the current scorer. It hashes
the frozen target archive before opening ZIP members, compares that digest
with the recipe pin, hashes again after endpoint scoring, and rejects either
a pre-score mismatch or a change during scoring. The result records both
digests and the two verification points. The preflight itself did not hash or
open target contents (`content_hash_checked=false`, `labels_read=false`); that
is expected for the label-free gate and is not grade evidence.

The preflight binds prediction headers, C-order/byte coverage, full-file hashes,
sidecar/provenance/checkpoint identities and source-qualified group order.
Ancestry checks every recorded checkpoint and provenance hash and checks
source-qualified valid-fit membership against the F0 evaluation groups. The
recipe fixes `global_wp_clip2_v1`; scoring uses the canonical global
accumulator, mask and clipping contract, and component bootstrap intervals are
conditional on the seven fixed endpoints.

## Limits at admission and interpretation

- A code-level 400,000,000-byte cap is not a measured peak. Root should bind
  the worker cgroup, zero swap, physical scratch path and live marimo/shared
  slice, and record peak memory/OOM counters in the execution receipt.
- The target pre/post hashes establish stable archive identity during a
  successful grade under the normal local-filesystem assumption. They do not
  make the already exposed F0 evaluation set independent.
- The NPY dtype and producer source pin support a claim about stored FP32
  predictions; they do not independently reproduce endpoint inference or
  prove every arithmetic operation used FP32. The result must remain a fixed
  prediction grade, not an inference-parity claim.
- The result must report point estimates and conditional component intervals
  without treating them as fit, fold, seed or public-transfer uncertainty.
- Root retains admission and execution authority. This review is not a grade
  approval or execution receipt.
