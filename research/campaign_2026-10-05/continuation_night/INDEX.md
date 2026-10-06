# October 5–6 continuation

Public anchor: **0.6660**. Research target: **0.7000**, unmet.

- [Contract](CONTRACT.md): authorization, independent roles, budgets and cutoff.
- [Coverage](COVERAGE.md): 66 registered mechanisms; queued designs are not results.
- [Runner admission](admission/README.md): reviewed aggregate 40 GB RAM, 36 GB pressure threshold, 12 CPU, no worker swap; original budget transferred once.
- E491: actual systemd runner smoke passed.
- E489 recovered frozen FP32 GPU probe: failed strict clipping-coverage gate after 19.666 device seconds. Peak RAM 2.429 GB, VRAM 0.422 GB; no new OOM events and teardown verified. Full fits remain unadmitted. [E499 actual diagnostic](activation_transfer/e489_clipping_diagnostic/actual_v1/report.json) finished in 26.174 charged seconds: 2.577 GB RAM, 0.422 GB VRAM; one seed stayed below clipping and failed AdamW update parity. [Measured diagnostic plot](activation_transfer/e489_clipping_diagnostic/eda/v1/clipping_and_update_parity.png).
- [E492 raw event-clock EDA](event_semantics/eda/v1/report.json): completed on 24 fit components in 5.664 CPU seconds, peak RAM 508 MB; [independent recount v2](event_semantics/reproduction/v2/receipt.json) passed: 24 fit components, 384 component rows and 64 aggregate rows matched all seven metrics (exact integer counts; target mass tolerance1e-9). Prior failed comparator v1 is preserved.
- [Event-clock figure](event_semantics/eda/v1/event_alignment.png) and [joint target support](event_semantics/eda/v1/joint_support.png); numerical CSVs alongside figures.
- E493 activation mixture and E494 label reanchor: source prepared. Actual full-shape control probe passed, peak RAM 7.688 GB and VRAM 3.611 GB; candidate probes and fits remain unadmitted.
- Kaggle: W384 v2 failed compatibility preflight (24 failed, 12 passed), with no OOM or surviving children. W384 v3 passes 36 staged legacy tests and 16 transport tests, but independent review is NO-GO because the packet omits the new checkpoint/provenance/history from its artifact manifest. A packet revision is in preparation; no push. TPU v3 remains queued. W256 has not been pushed.
- Independent preparation reviews: E495 GO for unsealed preparation; E497 GO for isolated prototype only; E496 has hash-bound independent preparation GO. Its resolver validates measured mass vectors and the actual epoch-3 checkpoint; 23 tests passed at 1 CPU/2 GiB. Seed3/fold0 fit-only calibration is now running at1CPU/6GiB/zeroGPU; no supervision fits or scores yet; unresolved templates remain unsealed.

Local admission closes October 6 at 10:00 Europe/Kyiv. Admitted jobs may drain; Kaggle continues within its separate measured quotas and cumulative campaign budget. Subagents use only GPT-6 Luna/max; Sol and Astra are excluded by the latest user instruction.

- [Further experiment code](FURTHER_EXPERIMENTS.md): E495 cadence, E496 supervision frontier, E497 causal memory prototype, E498 dependency inventory and E499 diagnostic. Combined bounded CPU validation: **36 passed**. Source hashes verified; **201 draft jobs and 8 unresolved calibration templates**, no new scientific fits launched.
- [Preparation receipt](checks/PREPARATION_RESULT.json) and [test receipt](checks/preparation_tests.xml).
- [Background monitoring](kaggle/MONITOR.md): enabled five-minute timer; failures recorded and diagnostic downloads bounded.

- [Open untested ideas](NEXT_UNTESTED_IDEAS.md): prioritized review proposals, their actual experiment status and decisive transfer tests.

- [E496 rebound review](admission/E496_INDEPENDENT_REBOUND_REVIEW.md) and [W384 v3r1 packet review](admission/W384_V3R1_INDEPENDENT_REVIEW_GO.md) are hash-bound preparation clearances only.

- Latest user resource override for **future runs across local and Kaggle**: aggregate **5 CPU cores / 3 decimal GB VRAM**, with 40 GB RAM unchanged. Running jobs keep their original envelope. Older larger drafts require revised profiling/admission: [override](admission/FUTURE_RESOURCE_OVERRIDE.json).

Latest user override: **5 CPU cores and 3 decimal GB VRAM for new runs**, enforced by the campaign parent CPUQuota=500%; existing admitted jobs continue. [Active loop](ACTIVE_LOOP.md) records current continuation.

- [Next causal event/quiet-state test](event_semantics/NEXT_TEST_CONTRACT.md): registered design based on independently reproduced fit-panel event counts; predictive experiments still required.

- E496 fit-only mass visualization code: [plot_calibration.py](supervision_frontier/plot_calibration.py) verifies the recorded float64 observation hash and saves ordered masses, distribution, cumulative mean and derived provenance; actual plot awaits the running calibration report.
