# H232 AMP4096 conditional V4 packet assembly

Created 2026-10-07T23:13:31Z. The identified packet is assembled and frozen for independent root review; no provider push or fit occurred.

Packet: `research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_conditional_v4/packet`  
Kernel: `vladkochetov07/h232-full-feature-gru384-b8-amp4096-conditional-profile-fit-v4`  
Status: `assembled-awaiting-independent-root-review-and-fresh-quota-admission`

The successor recipe changes FP16 GradScaler initial scale from the historical trainer default 65,536 to explicit 4,096 for both profile and fit. The original recipe and its default remain preserved; this attempt will not be described as an exact reproduction of that historical numeric policy. All other scientific recipe fields remain fixed. The packet embeds the exact successor recipe and binds its SHA-256.

The only source-tree member changed relative to H232 Profile V2 is `scripts/train_rnn.py`, carrying the independently tested Path-to-string JSON serialization repair. Its hash is `cc0821a144fbea572dfaa03832fbbf31cf1ea605a917b0943c3db5da6b763e1a`. The source archive contains 54 files, SHA-256 `afde296e1c5e2f0c47a6fb352c2dea8e0a3df75900f53c26f103b3921df1f565`; its member manifest is `e9ca52c7247ebc944bac086261715879f37006033c556c9c9890445ac22bbb01`.

The profile gate uses three warmup batches and 20 timed outer batches (30 warmup and 200 timed optimizer updates). The fit runs in the same worker only if finite loss/gradient/update evidence, zero timed AMP skips, stable T4 identity, memory/disk headroom, exact read-only data stat tuples, actual train/valid shape probes and projected lifetime all pass. It then verifies the fixed epoch-3 checkpoint hashes and exact fold-0 memberships. Prediction export, grading and scoring are outside this packet.

The embedded ZIP digest is `e5db8d4675b8d7e2cd132e5c363f83f79f8ba61198d70e56932e1f34a1bfd3a0` and the Kaggle code-file digest is `a2f7937b37f7d7a4e885701e18a665fba19d5733c9523512173e25421d519746`. All 11 embedded members were rechecked byte-for-byte. The exact packet reviewed hash map is in the JSON receipt.

CPU-only gate tests, compile checks and Ruff F checks passed under the 1-CPU/1.5-GB support limit. One test exposed and corrected a mismatch between the gate’s expected trace summary keys and the trainer’s actual `attempts`/`completed_updates` schema.

V4 has no actual provider timing yet. The preceding E520 V12 mount required 544.533 seconds for raw train/valid SHA-256; whether V4 can reuse the prior digest depends on exact read-only path and stat-tuple equality. The worker records startup, per-file hash or stat reuse, profile, shape probe, fit/checkpoint and output-validation timings separately.

This is one diagnostic profile and at most one full fit; it is not a completed candidate/control screen. The separate same-session diverse-approach follow-up needed for five screens per hour remains unadmitted. Root must independently review the exact files, issue fresh quota admission and replace the pending review receipt before rebuilding and pushing.
