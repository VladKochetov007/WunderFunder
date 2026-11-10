# CPU submission: fixed canonical TL3 + 10% PR53 LSTM

Prepare a genuinely different CPU submission from four available team checkpoints,
with a real 14,011,657-byte ZIP and a two-stage Linux receiver. No local tests,
inference, training, server job or competition upload was performed.

1. **Status, lineage and owner.** Packaging/receiver implementation only; continuation
   of PR53's fixed mixtures, not a new scientific fit. Base research-team7e407431;
   source report at PR53f98543d2. Exact checkpoint/source/archive hashes are in
   INPUTS.json and BUILD_RECEIPT.json. Status PACKAGED_NOT_EXECUTED.
2. **Mechanism.** Test whether the published LSTM complement survives CPU serving.
   Alternative explanation: the tiny exposed-validation difference is noise, and
   INT8/fast activations/streaming-feature differences remove it. The matched
   served TL3 is measured alongside the actual candidate callback.
3. **Compute advantage.** Zero B300 compute. Reuse published weights; spend bounded
   CPU validation before considering a portal upload. New teacher/student fits
   remain a separate GPU research decision.
4. **Frozen comparison.** 0.9 canonical train-only W18/tiedlead5/tiedlead6 TL3 plus
   0.1 raw112/H192x1 LSTM, fixed epoch8/seed1302. Baseline weights .34/.33/.33.
   Native dynamic INT8/fast-activation GRUs; FP32 NumPy LSTM; FP64 mixture then
   one FP32 cast. No weight search or retraining. All recurrent states reset by
   seq_ix and advance on every row, including warm-up.
5. **Budget.** Zero new fits. Server profile uses eight complete sequences plus
   state/reference checks, timeout30minutes; full validation uses all1873
   sequences, timeout2hours. One logical CPU,16GB, no swap/GPU/network. Existing
   GPU jobs are unchanged. Full validation is explicitly separate from profile.
6. **Validation and uncertainty.** Historical unquantized PR53 WP .69250206 vs
   .69237742 (+.00012464) is not a score for this ZIP. Full official global WP,
   both targets, served-base difference, predictions and timings are returned.
   Repeated exposed validation and selection are disclosed; no significance or
   public-score claim. Missing official test cardinality leaves timing pending.
7. **Leakage/coverage.** The four pinned checkpoints declare the same10325 train
   fitting groups and disjoint1873 official-valid groups; saved train normalizers
   are kept. Legacy consumed teacher-cache ancestry is incomplete. No new future
   features are introduced. The native tiedlead streaming semantics can differ
   from the chunked research evaluator and require delivered-predictor scoring.
8. **Decision.** Require finite/reset/parity checks, positive served-base point
   delta, and defensible CPU envelope before release review. A negative result
   rejects this deployment recipe, not the entire LSTM direction. No automated
   coefficient tuning or upload.
9. **Run.** RUNBOOK.md and launch_server.sh provide portable server-only Docker
   commands. The exact image ID, ZIP, recipe, INPUTS and passing profile receipt
   are bound before full validation. GPU host/access is not needed for CPU checks.
10. **Return.** profile.json/full.json, console logs, package/prediction hashes,
    both prediction arrays, CPU model/affinity/quota, peak memory, process CPU
    time, Python/library versions, score and conditional whole-test projection.
11. **Trace.** DEDUP.json records108 visible submissions and byte/member checks
    against downloaded archives. The prior PR49 blend used fbc1k, and November4
    submitted full-corpus W18 alone; neither is this recipe. STATIC_REVIEW.md
    separates source review from outstanding numerical/server acceptance.

Code/manifests only are committed. The actual ZIP and portable receiver handoff
are delivered as local artifacts, with their hashes recorded here; no weights
are added to Git. Local verification is Ruff, AST/JSON/shell syntax and byte/CRC
inspection only. No model construction, forward pass or native load occurred.
