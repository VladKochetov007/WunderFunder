# E530: online learning and adaptive sequence state

User-requested new family, now registered as `family:online-learning`. Initial controller proposals are committed separately from the native scout and requested external-provider prompts. [Brief](BRIEF.md), [controller proposals](CONTROLLER_PROPOSALS_V1.md), [provider access evidence](provider_runs/ACCESS.json).

| Arm | Smallest useful competition experiment | Current evidence |
|---|---|---|
| H240: adaptive feature dynamics | Same fixed head/parent, fixed vs warm-up-only vs continuously updated small latent dynamics | Numerical CPU reference passed; task fit not run |
| H241: learned fast weights | Same-scale recurrent/no-write/learned-write controls | Source-grounded proposal |
| H242: task-aligned self-supervised adapter | Same joint offline model without updates vs fixed updates vs meta-trained updates | Source-grounded proposal |
| H243: adaptive expert reliability | Fixed vs adaptive vs stale gates on a new expert cohort | Interaction proposal; proxy usefulness unproven |

The inference interface has no target feedback. Learning from completed feature transitions is available: row t reveals the next-feature outcome predicted from t−1. The prototype observes that transition, updates small sequence-local weights and forecasts the next input. Its innovation/forecast still needs a separately fitted map to the competition targets. A lower feature-prediction error alone is not success. Reset every adaptive value, including covariance, pending forecasts, optimizer moments and normalizers, at sequence changes. Sequence IDs and hidden scoring masks cannot enter the predictor.

## Completed bounded protocol

Own implementation: [online_dynamics.py](../../../../src/wnn/online_dynamics.py), with [five focused checks](../../../../tests/test_online_dynamics.py). Each prefix agrees with directly solved weighted ridge for forgetting factors 1 and 0.98; tests also cover predict/update ordering, future-prefix invariance, reset/replay, invalid input and a 20,000-row constant sequence. [Independent review](reviews/ONLINE_DYNAMICS_REVIEW_V1.md) passed this narrow protocol. Non-unit forgetting still needs a full-length ill-conditioning test before use on real data; only forgetting=1 has a 20,000-row check here.

[Reference timing receipt](ONLINE_DYNAMICS_PROTOCOL_V1.json): three synthetic 20,000-row trials per width, one CPU and approximately 48 MB peak RSS. Width 16 measured 28.05 microseconds per row; width 32 measured 34.58. Stored numerical arrays, including the fixed prior, were approximately 6.3 and 24.8 KB. These are float64 NumPy reference measurements, excluding projection, parent inference and readout. They do not establish final serving feasibility or a speedup. A native implementation is a later option if the task result justifies it.

## Model diversity and next action

Grok 4.7, Freebuff's GLM 5.3 Flash/max picker route and OpenCode Muse were checked for this assignment but stopped before inference on read-only local CLI state/log directories. Kimi's exact route is unresolved and was not guessed. No external-model proposal is claimed. [Separate prepared prompts and invocation requirements](provider_runs/FUTURE_COMMANDS.md) preserve the requested diversity for an execution-capable environment. A fresh-context Luna/max scout completed five proposals from primary sources and a seeded random-matrix library passage. [Synthesis](SYNTHESIS_V1.md) merges the overlapping adapter idea and adds H244–H247 for eight distinct arms/interactions total, preserving the originals and correcting resource/allocation assumptions.

Root owns the next comparison: a stream-generated latent-dynamics/readout recipe with fixed membership, own parent, explicit update schedule and matched controls. First task fit belongs to the protected exploration allocation, alongside independently progressing H230/H232/H217 work. No GPU job or online-learning competition result has yet run. Larger online-learning teachers and compression remain eligible even if their inference would exceed submission limits. Record each descendant and evaluate the complete endpoint; keep numerical priors unknown until measured.
