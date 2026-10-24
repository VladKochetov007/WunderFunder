# B300 bounded 24-slot short-screen amendment

This new cohort explicitly supersedes the old CP30 launch scope for this allocation only. It does not resume the historical controllers or complete the original full-data PR26/PR27 matrices.

The user authorizes all eight existing B300 devices, three owned actors per GPU, a strict four-hour node wall-time limit, partial work, and results/cleanup within that limit. Common T0 is **2026-10-22 03:02:11 UTC**; the absolute deadline is **07:02:11 UTC / 10:02:11 Moscow**. Actual systemd startup settings are captured in [the startup receipt](receipts/STARTUP_UNIT.json): RuntimeMaxSec=13500s, TimeoutStopSec=120s, KillMode=control-group, no restart, and an ExecStopPost final publisher. The two bounded stop/finalizer allowances leave the all-in service ceiling at 13740s, below 14400s. No foreign service or process is stopped.

## Question and comparison

PR26 tests per-field versus grouped causal ages at the same 323-input width. PR27 tests causal innovations versus the same-width moments/zeroed-innovation control, 291 inputs. Both use cold H512x1, source a266b036b02e8e012c6eb31072291dad618a169d, labels only, strict FP32, autocast and TF32 off, AdamW/OneCycle, six epochs and fixed epoch5. All incomplete endpoints remain partial and unscored under this exact endpoint.

The finite panel is two mechanisms × two fitting-data fidelities × three matched fold/seed cells × two arms = **24 new scientific fit attempts**. Cells are F0/seed1201, F2/seed1202 and F3/seed1203. Fidelities select 10% and 25% of complete fitting components, nested by a label-independent SHA ordering. Actual sequence counts differ from component fractions and are explicit in [PLAN.json](packet/PLAN.json). Every sequence retains its full 20000 rows, while official held-out membership/scoring is unchanged.

This is a deliberate data-fidelity comparison for two mechanisms, not four independent discoveries. There are six fits per mechanism/fidelity panel, twelve per mechanism across both fidelities. Controls are new subset-matched fits; no old full-data control or unqualified partial checkpoint is reused. The larger subset tests whether a cheap screen's direction persists with more data. It is not full-data confirmation or a public-score forecast.

## Evidence and lifecycle

[Packet manifest](packet/MANIFEST.json), [source](packet/code/recipe.py), [finite execution](packet/code/run_cohort.py), [fit validation](packet/code/validate_fit.py), and [publisher](packet/code/publish.py) preserve exact recipes, request hashes, input membership, source pins and deadlines. All fitting uses the existing canonical trainer and original process envelope; all scoring uses the unchanged original grader.

[CPU preflight](receipts/CPU_PREFLIGHT_V2.json) checked slot counts, paired inputs, nested complete components, permitted CPUs and owned process timeout. [Independent review acceptance](receipts/SOURCE_REVIEW_ACCEPTANCE.json) records the fixed exact-arm/request validation gap; later user permission for partial work is explicit in the final plan. [All 24 profiles passed](receipts/PROFILES.json), each with100 successful updates and finite losses/gradients. [Actual admission](receipts/ADMISSION.json) admits24 fits. Profiles are engineering work, not additional scientific replications.

Fit processes stop at T0+11100s, export at T0+12900s, and grading at T0+13140s. Every subprocess is owned and bounded; the outer service kills its full cgroup and invokes the final publisher even on timeout/failure. Accepted pairs publish updates as SemyonEpanov to PR26 and PR27. The final publisher records an actual owned-CUDA-client cgroup check before posting terminal results. A transport failure is retained in the private outbox/failure receipt; successful authentication and both comment writes were verified before science.

Live artifacts and later terminal receipts remain at /data/epanov_s/tabdata/state/short4h-20261022-r1. The committed receipts are startup/profile evidence, not final model scores. Keep lifecycle counts separate: attempted, complete-epoch records, integrity accepted, exported and officially scored. Component bootstrap intervals condition on fixed models; three paired cells cannot separate all fold/seed effects or establish public transfer.

- [PR26 live result/status comment](https://github.com/VladKochetov007/WunderFunder/pull/26#issuecomment-6331586358)
- [PR27 live result/status comment](https://github.com/VladKochetov007/WunderFunder/pull/27#issuecomment-6331586689)

All negative, partial and failed work is retained. No automatic promotion, best-checkpoint selection, quantization, extra teacher training, new rental or source-recipe retry is admitted.
