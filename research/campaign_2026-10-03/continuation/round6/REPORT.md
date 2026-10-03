Frozen landed evidence index for controller ledger/backup support.

Only landed_evidence_index.json, REPORT.md and HASHES.json are new. Original
reports, states, manifests, failed jobs and supplements remain untouched.
No ledger/frontier/mainSTATE edit, GPU action, data read or capture payload rehash.

Exact source declarations (from frozen metadata; no live-source reread):
scripts/train_rnn.py: 1d6b059d9880e0befb7b2c6251c7ae6ac9791d0bb0f5cd35dd8e4930cf189ef5
src/wnn/nn.py: 388af04570f0a3944159bf86c1a234dc55c78b951804c38ef59fa314f92f3db3
src/wnn/optim.py: 0e476a54d1fdfeed77970816f91668e271f7e2ed72e13d0c047a01d0ff74dae3

| Evidence | Terminal interpretation | Exact primary report/diagnosis path and SHA256 |
| --- | --- | --- |
| v4_builder | FROZEN_BUILDER_CPU_ONLY_30_TARGETED_TESTS | research/campaign_2026-10-03/continuation/round4/live_capture_v4/REPORT.md ; b7b1844273e8593a9482bb262144ebe64ceb2f59a97e92f90651e83c69393445 |
| v4_independent_builder_review | PASS_FOR_BOUNDED_UNCHANGED_DEFAULT_CUDA_MEASUREMENT_WITH_MANDATORY_GATES | research/campaign_2026-10-03/continuation/round4/capture_review_v4/REPORT.md ; 3ffef486d9ecec822cdbe9684698569c6078f7af29e838dfe387efe2b748a396 |
| completed_capture_independent_review | PASS_MEASURED_PARITY_FOR_RECORDED_WARM_V4_DEFAULT_TRAJECTORIES_ONLY | research/campaign_2026-10-03/continuation/round6/capture_measurement_review/REPORT.md ; 923e6fab6e2a5d5265816f9cc8a6bb55819241a28bad221967ad9dbc4a175f4b |
| false_timeout_2112 | DIAGNOSED_REPAIRED_SIX_PREDICATE_FIXTURES_PRE_ADMISSION_FAILURE_PRESERVED | research/campaign_2026-10-03/continuation/runtime_incidents/false_timeout_2112/REPORT.json ; 2b2a159b6192cd2d8c7696b22cb9272ece6d6b339c7fc30d416df2c9d44dbc2e |
| R5_mechanism_instrument_v2 | BLOCK_ACTUAL_CAPTURE_SCIENCE_PENDING_APPROVED_V4_AND_INDEPENDENT_DECOMPOSITION_POLICY | research/campaign_2026-10-03/continuation/round5/mechanism_instrument_v2/REPORT.md ; eabb289d9b79c3ca52ac30799c1cbfcce1710c9c44c7752db3910dad85834f1a |
| R6_state_drift_builder | FROZEN_CPU_INSTRUMENT_PASS_REAL_UNAPPROVED | research/campaign_2026-10-03/continuation/round6/state_drift/REPORT.md ; 56d8dfd2c2e7258443b93fc29be9feb0a60bd7673a6c30e6819a56c8fac45353 |
| R6_state_drift_blind_critic | BLIND_PREMORTEM_FROZEN_WITH_ADDITIVE_MANIFEST_COMPLETE_PRIOR_EXIT75_PRESERVED | research/campaign_2026-10-03/continuation/round6/state_drift_critic/REPORT.txt ; 5c09c698aaeb8a84a8c9c3dff7851a05a25030b35a9f8bc4b4e5629467c993f8 |
| R6_state_drift_final_review | BLOCK_CURRENT_REAL_ADMISSION_CONDITIONAL_ON_INDEPENDENT_DEFAULT_GATE | research/campaign_2026-10-03/continuation/round6/state_drift_review/REPORT.txt ; 242901783fc42482c21f061cfe9cb191a0efa51e4e703edf45465deca5014ee3 |
| R6_amp_linearity_audit | FROZEN_TESTED_PENDING_INDEPENDENT_REVIEW_AND_CONTROLLER_AUTHORIZATION | research/campaign_2026-10-03/continuation/round6/amp_linearity_audit/REPORT.md ; 53725803676ce4712b0f25b803a96df24cd204cb7a74c2ac3996b8047a093ab7 |

V4: builder30 tests plus independent32checks PASS. The independent builder
review is conditional default-measurement support, not scientific approval.
Job c20261003_capture_v4_default_2120r finished exit0 in31.28510255800211s,
peak observed VRAM3,709,861,888bytes. Recorded warm unchanged/default original,
repeat, normal and deliberate-skip maxima/floors are0 across synthetic actual
architecture and fixed fit-mixture panel. Measurement review PASS is narrow;
reviewer did not replay CUDA/GRU or independently validate private CUDA engines.
The reviewer streamed40files/2,380,024,182bytes exactly once; this index hashes
its small audit/index/report/state manifests and retains declared payload hashes
without opening/rechecking any payload. No new payload integrity audit is claimed.

False_timeout_2112: unanchored timeout predicate matched a successful Started
command containing --timeout600, falsely publishing an admission barrier.
Controller diagnostic records6 repaired fixtures, CPUlauncher exit0, GPUstarted
false and no new OOM. Historical parentOOM11 is not a new failure. Preserved
c20261003_capture_v4_default_2113 remains failed before GPU admission; the later
2120r result is a distinct successful fresh job, not a rewrite of that failure.

R5mechanism_instrument_v2 is frozen/tested but actual scientific analysis BLOCKED.
Its32 adversarial guards and fp16 sum counterexample do not provide an actual
operation-specific AMP decomposition bound. Whole-step floor0 cannot certify
component-sum linearity. Descriptive component self-gradients remain UNCERTIFIED.

R6state_drift retains the original prereg plus separate warm-skip clarification;
four attempted updates with skips are an amended estimand, not four completed
updates. Its13 tiny CPU controls pass; no actual state drift was measured.
Blind critic STATE/REPORT retain historical exit75/pending-manifest wording;
HASHES_SUPPLEMENT.json and MANIFEST_ADDENDUM.txt complete metadata hashing only,
with no rerun, candidate/capture read or approval. Final independent review BLOCK:
finite stale-hidden/prediction/final-metric guards, new runtime/invocation/deadline
binding and explicit original decimal4.8GB allocator cap are still required.
That review did not read/accept the measured default gate; later narrow capture
review PASS does not retroactively approve this different scientific entrypoint.

R6amp_linearity_audit is frozen/tested pending independent review and explicit
controller authorization. Exactly two fixed normal cases;20 adversarial fixtures
pass. No actual gradients read or discrepancy measured. Descriptive measurement
does not supply a calibrated allowance or authorize conflict/update attribution.

Fresh panel normalization/trajectory are not recovered mature full-fit states.
No eval/public score, estimator intervention, optimization or scientific promotion.
All exact state snapshots, metadata/source declarations, receipt/job paths/hashes
and per-entry missing gates are in landed_evidence_index.json. Manifest references
are declared evidence; only selected consumed small files are locally hash checked.

Metadata-only build: 63 files, 413220 input bytes;
streamedSHA256/64KiB chunks,1MiB per-file bound, originals rechecked before write.
run_cpu.sh round4-capture enforces1core/2GiB/900s. HASHES.json binds both new
outputs plus consumed metadata; excludes itself to avoid a circular checksum.
