# Whole-endpoint storage, precision and runtime policy — R1

Independent analysis, based on MAIN `d2c82811c53f047585678dad52624136f117b5e9`.
Owner: `compression_endpoint_policy`; integration/admission owner: root.
Latest effective scope: requested Sol 6.1/xhigh analysis, isolated worktree,
no fits/provider operations/spend, no MAIN/source/grader/queue mutation.
Support execution: absolute MAIN `scripts/limited`, `wnn-campaign.slice`,
1 CPU, 700,000,000 bytes, 120 seconds per invocation; no full parquet/cache loads.
This card recommends tests; it does not admit their execution or change a frozen recipe.

Prefer a quality-supported broader Q8 endpoint using FP16 **disk transport**
before spending engineering effort on lower-bit inference. The actual selective
FP16 storage probe creates sufficient size headroom for three h384 teachers plus
one h256 student under the existing array schema. A wider blend must still earn
its place through its complete quantized score and measured runtime. Retain Q4
as a bounded composition test: a loss for a fixed pair does not exclude a larger
Q4 endpoint that gains enough conditional signal.

The binding final envelope remains ZIP ≤20,000,000 bytes, offline Python 3.11,
one CPU, 16 GB and 60 minutes; all members have own independent weights and
causal sequence reset. [ADR0032](../../../../specs/adr/0032-package-available-full-data-blends-now.md)
allows packaging now and a bounded Q4 screen. Clean ancestry, fold exposure,
distinct-seed accounting and exact delivered-byte checks remain binding.

## Source findings

| Representation | What the current code actually does | Consequence |
|---|---|---|
| Current “Q8” ZIP | `package_submission.py:187` writes dense NPZ arrays; `solution_native.py:59–72` restores every tensor to contiguous FP32 and passes `precision.txt` to `EnsemblePredictor`. | Q8 denotes inference, not one-byte disk tensors. A precision filename alone is insufficient after the R1 wrapper failure. |
| Q8VNNI | `rnn_fast.rs:799–835` folds normalization into an I16 projection; `853–862` packs recurrent/head matrices into `dq`; `924–948` splits head.0 into hidden/projection blocks; `968–970` preserves FP32 biases/final head. | Distinguish projection quantization, dynamic activations and matrix quantization from transport loss. State remains FP32. |
| Selective FP16 NPZ | Existing `_load` restores FP16 arrays to FP32 without a native API change. | Half only GRU matrices and head.0 first. Preserve input projection, normalizers, biases and head.2 exactly. This changes model values and can change Q8 rounding; it is not lossless. |
| Existing preserved Q8 export | `quantization_export.py:13–62,65–101` stores/restores codes/scales; `python.rs:104–132` imports through `PackedQ8`; `rnn_fast.rs:1058–1059,1087–1104` access only `q8`, not `dq`. | This path supports q8r/q8x presets, not the Q8VNNI dynamic store. Do not feed it to Q8VNNI or switch backend and call the output equivalent. `quantization_codes()` can return empty mappings for dynamic Q8; that is not zero quantized storage. |
| Exact dynamic codes | `vnni.rs:101–168,189–199` already supports `DynQ8::from_rows`, `pack` and `to_rows`. | A small independently written named `dq` import/export extension is possible. It still requires API/loader changes, fresh ABI-safe native builds and actual equality tests. |
| Current Q4 diagnostic | Dense matrices rounded rowwise to levels −7…7, reconstructed as FP32, then run on Q8VNNI. | Fifteen-level weight rounding, not native INT4 or a modern calibrated quantizer. Smaller compressed transport is possible; dense Q8 kernels retain their dimensions and byte layout, so a runtime speedup is unestablished. |

Code citations: [loader](../../../../submission/solution_native.py#L59),
[exporter](../../../../scripts/package_submission.py#L160),
[native construction](../../../../native/src/rnn_fast.rs#L799),
[preserved export](../../../../src/wnn/quantization_export.py#L13),
[Python import](../../../../native/src/python.rs#L104),
[dynamic representation](../../../../native/src/vnni.rs#L101).
The native callback [advances every row before checking need_prediction](../../../../native/src/python.rs#L717),
resets features/norm/all members on sequence changes, and the
[loader disables the governor](../../../../submission/solution_native.py#L67).
Runtime must count all rows, including warmup and rows returning `None`.

## Bounded evidence checked

[TRANSPORT_STORAGE_PROBE_R1.json](TRANSPORT_STORAGE_PROBE_R1.json) binds source
receipt and ZIP hashes and records every converted tensor. It loads only small
NPZ weight payloads, converts arrays in memory and writes only this receipt.
No native callbacks, labels, held-out predictions or complete datasets were used.

| Exact source endpoint | Original ZIP bytes | Selective-FP16 ZIP bytes |
|---|---:|---:|
| H388 h384 seeds4/5 CP2 pair, R3 | 17,272,496 | 9,590,917 |
| H390 h256 seed3 CP5 + H388 h384 seed5 CP2 | 12,914,019 | 7,364,001 |

The same-shape raw NPZ estimate for three h384 teachers plus one h256 student
is 15,523,040 bytes; unchanged native libraries and metadata add 2,908,393
uncompressed bytes, giving **18,431,433 before outer archive headers**. This
includes every NPZ array and its existing NPY/NPZ headers: unchanged tensors
are retained, not dropped. Allow another conservative24 KiB for outer ZIP
headers/deflate overhead and the enlarged members manifest, giving18,456,009
bytes under the same array schema. This is a capacity estimate, not an exact
independently exported third-teacher ZIP or a quality result. Its ~1.54 MB
slack warrants the exact build instead of assuming
the larger blend needs Q4. Converted pairs change approximately 0.7% of the Q8
codes under separately computed NumPy max/127, ties-even algebra; many scales
also change. Native equality and scored fidelity remain untested.

The preserved independent precision reviews establish only the ten-group
**in-sample** H390 diagnostic: Q8−FP32 = −0.00007090; rounded-Q4−Q8 =
−0.00180125. The Q4 ZIP is 3,357,848 bytes. These do not measure another
endpoint, clean-fold transfer or native INT4 speed.
[Q4 review](../diversity_stack_20261009/ops_continuation_20261011/h390_submission_review_r3/Q4_PRECISION_DIAGNOSTIC_REVIEW_R1.json),
[FP32/Q8 review](../diversity_stack_20261009/ops_continuation_20261011/h390_submission_review_r4/FP32_Q8_INDEPENDENT_REVIEW_R1.json).

Existing exposed-fold1 teacher-trio score .67254162 and student-half/trio-half
.67264009 imply only a .00009847 addition from the student, conditional on those
selected fitted models; component interval [−.00065936,.00089053]. Adding teacher
signal to that student gives .00160862. These asymmetrical comparisons show why
standalone quality or disagreement cannot substitute for complete endpoint WP.
Adding weaker teachers previously hurt that particular trio. The new all-data
H390 student is not the exact evaluated F1 student recipe, so these numbers
cannot be attached to its final ZIP.
[Frozen comparisons and paired results](../diversity_stack_20261009/ops_continuation_20261011/QUALITY_DIVERSITY_PAIRED_R1.jsonl).

## Four discriminating tests, in priority order

Each recipe/configuration and membership manifest is committed before reading
scores. No new training is needed for these tests. Use small complete-sequence
chunks in the existing aggregate slice; cap each invocation at 1 CPU/700 MB/120 s,
persist partial accumulators/predictions, and admit a finite total chunk budget
before execution. Root supplies execution/grading owners; this analyst does not
self-approve a release. Larger future jobs need their own explicit allocation.

1. **Transport fidelity at fixed members.** For the exact wide pair and compact
   pair, compare dense-FP32 transport/Q8VNNI with selective-FP16 transport/Q8VNNI.
   Start with the existing ten complete sequences, one endpoint/replay invocation
   at a time; record actual native precision, all active members, coefficients,
   finite outputs, reset replay, scored WP/t0/t1, errors and package bytes. The
   first comparison is in-sample precision evidence only. Repeat on source-qualified
   excluded-component counterparts as they exist. Stop expanding the transport
   arm if its measured loss consumes the complete blend's gain; do not use a
   universal per-tensor error threshold as the selection objective.

2. **Broader Q8 endpoint.** Root's immediate final-package shortlist is H388
   own h384 seeds3/4/5 CP2, equal1/3, and H390 own h256 seed3 CP5 at1/2 plus
   those teachers at1/6 each. Pin the separately exported third teacher and
   final source receipts; this is not the old F1 student's exact recipe. All-data
   final panels are in-sample serving fidelity, with no held-out endpoint claim.
   For local recipe evidence, freeze three strong eligible teachers at weights1/3,
   and their half-weight blend with the eligible student at weights1/6,1/6,1/6,1/2.
   Compare both selective-FP16/Q8 endpoints with the best complete feasible Q8
   comparator on identical rows. Keep an unconstrained dense-transport/Q8 version
   of each broader endpoint as a research reference to separate composition gain
   from transport loss; it need not meet serving size. Use one fixed64-component
   panel per available source-qualified fold, feature-only membership and complete
   causal sequences, then full-fold scoring for the selected finalist. Record all
   folds, endpoint seed sets and missing clean counterparts; existing folds remain
   exposed. No coefficient sweep. If added signal is tiny and time grows sharply,
   a smaller endpoint can remain strongest; imprecise positive effects may justify
   bounded confirmation rather than a universal CI rejection.

3. **Q4 compensation test on the complete endpoint.** Compare the broader frozen
   endpoint with rounded-Q4 matrices on Q8VNNI directly to its feasible smaller
   Q8 comparator on the same two panels. Include one selective variant that keeps
   the comparator's anchor teachers at original precision and rounds only added
   complementary members. Report every tested configuration together. Dense
   transport/Q8 of the broader mixture separates diversity from rounding cost.
   Advance Q4 only if its *delivered* complete score and measured capacity are
   better under the constraints; its same-member regression alone is not a veto,
   and a lower ZIP size alone is not a promotion. If FP16/Q8 already fits and wins,
   it removes the size rationale for this Q4 endpoint but not every lower-bit arm.

4. **Runtime calibration of finalists.** On the same host, Python3.11, pinned
   one-core affinity and native ISA, interleave the exact previously uploaded
   old pair and at most two finalists on the same complete10-group panel with
   fixed repeats. Include cold import, feature construction, all callbacks and
   returned-array/scorer overhead; record warm and cold separately and cgroup
   peak/events. Test forced q8vnni_avx2 against available VNNI for the selected
   finalist; portable fallback requires separate capacity evidence if needed.
   Use the exact old-pair reported portal runtime (~27 min, user report) only as
   a provisional calibrated ratio: candidate portal time ≈ old portal time ×
   matched host endpoint time ratio, with an explicit slowdown/overhead margin.
   It is not a guaranteed 60-minute result. Disk FP16 and rounded Q4 cannot be
   credited with a kernel speedup. If the complete runner's row count is unknown,
   record it unknown: `expected_rows=39_400_000` is a native governor default,
   not a verified official denominator. Preserve deterministic fixed membership;
   no wall-clock pruning to rescue an oversized endpoint.

For any smaller endpoint C and larger endpoint L, compare directly:

`WP(L delivered) − WP(C delivered)`

The useful decomposition is composition gain between high-fidelity endpoints
plus the **difference** in their precision/transport losses on those same rows.
Neither marginal gains nor weighted Pearson scores are additive across members,
and the H390-pair Q4 loss is not the loss of L. Paired whole-component resampling
describes fixed-prediction uncertainty; it does not provide seed variation or a
public forecast. Final recipe ranking uses all valid matched folds and distinct
fitting seeds available, never a best-F1-only selection.

If selective FP16's measured fidelity is inadequate, exact dynamic-code
transport is the next focused engineering route: serialize the *actual native*
named `dq` codes/scales for recurrent and both head.0 blocks, retain original
projection/normalization/bias/final-head floats, import through `DynQ8::from_rows`
without re-quantization, and prove bitwise callback equivalence on long complete
sequences including reset across ISA variants. Decoding codes×scale and packing
again is not an exactness proof because scales can change by FP32 rounding.
Exact codes reduce transport size; they do not speed the unchanged Q8 kernels.

No public score or numerical public gain is claimed. No fits, provider calls,
billing changes, queues, MAIN source or frozen submission artifacts were changed.
