---
kind: review
status: complete
created_utc: 2026-10-15T10:59:29Z
updated_utc: 2026-10-15T10:59:29Z
hypothesis_ids: [H-CPU-POLICY-20261014, H-STUDENT-LADDER-20261014]
experiment_ids: [E-VAST-A-CPU-FEASIBILITY-20261015-R1]
evidence_paths:
  - ../../campaign_2026-10-15/vast_a_cpu_probe_r1/PROTOCOL_AND_MARKET_R2.md
  - ../../campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/portable_cpu_benchmark_r1/BENCHMARK_PACKET_R1.json
  - ../../campaign_2026-10-14/cpu_student_frontier_r1/portable_cpu_benchmark_r1/smoke_local_20261015_r1/SMOKE_PAIR_R1.json
---

# Vast-A CPU policy timing packet review

## Findings

The frozen fixture and runner are suitable for an **exploratory, runtime-only A/B/C/D panel** once the root controller has an allowed host. The package binding fixes the original xteam_h4 members, weights, order, coefficients, preprocessing and Q8 representation while asserting the selected `q8vnni`, `q8vnni256`, `q8vnni_blocked` or `q8vnni_fused` policy. The worker imports the package's unchanged callback, checks the selected policy and loaded native library hash, and times complete raw-feature callbacks after an untimed sequence reset. The four-field callback adapter matches the pinned `utils.py` contract. The native `PredictionModel.predict` returns an `f32` NumPy array, so the runner's float32 staging array preserves the frozen callback output type.

The fixture uses eight complete, source-qualified F0 seed-3 fitting sequences, one from each of eight distinct components. It binds the split and sampler manifest, selected sequence identities and feature hashes, plus the complete compact feature/request payload hashes. Its generated request vector—false for rows 0–98 and true for rows 99–19,999—matches the documented callback schedule in `docs/data_overview.md`; it omits `is_scored`, which the callback does not receive. There are no targets, scoring masks, teachers or cache values in the fixture. The source Parquet body was not rescanned: identity is limited to its footer hash, size and modification time, while selected features are individually hash-bound. This is adequate for the selected runtime fixture, not a full source-file integrity claim or a clean evaluation panel.

The panel pairs every candidate with A on the same eight fixed sequences, brackets each candidate trio with A, randomizes sequence order, and balances all six B/C/D orderings once per slot in each six-block cycle. The reported interval is a descriptive Student-t interval over block means of sequence-paired log runtime ratios. It is conditional on this fixed sequence panel and one host; within-sequence repeat spread and between-sequence mean spread are also reported separately. Adaptive stopping is unadjusted and all blocks are explicitly exploratory. A winner claim still needs a separately preregistered untouched confirmation panel.

One operational guard is needed before any future metered-host run: the runner checks its `--time-budget-s` only after a panel/block finishes, and its pilot estimate cannot enforce a hard wall-clock cap if callbacks slow down. The root controller must wrap it in a stricter external deadline with collection and teardown reserve. The protocol's optional shape ladder and minGRU pair should remain separate follow-up panels after A/B/C/D collection; they should not be added to the first panel's time budget.

The local smoke is correctly limited to compatibility evidence. Its sealed receipt records two complete, reset-separated replays each for A and B on one source-qualified sequence, with bitwise equality between A and B and within-policy replay. On a single pinned Intel i9-11900H logical CPU, callback wall times were A 2.168/2.081 seconds and B 2.879/2.998 seconds. This is not a policy ranking, an AMD EPYC 9224 measurement, a variance estimate, or a quality result. Current exact-CPU search receipts establish no runnable CPU-only host; no provider action or spend is part of this review.

## Recipe comparison

| Requirement | Frozen packet | Review |
|---|---|---|
| Fixed served endpoint | Same xteam_h4 members, weights, coefficients, preprocessing and Q8 codes across A/B/C/D | Bound by package manifest and asserted policy/native-library identities |
| Runtime input | Eight complete 20,000-row F0 fit sequences; raw 112-feature rows; documented warmup/request schedule | Source-qualified, deterministic selected panel; labels and `is_scored` excluded |
| Reset and timed path | Full callback loop after a sequence-ID sentinel; startup, construction, hashing and sentinel reset separate | Matches the callback's sequence-state reset and preserves causal history; warmup is still processed on every timed replay |
| Comparison design | Five-callback pilot, minimum six paired blocks, maximum twelve, A bracket controls, balanced candidate order | Appropriate for exploratory host-conditional timing; optional stopping and no multiplicity adjustment preclude confirmatory interpretation |
| Runtime admission | 1,000-second internal panel budget and one-CPU affinity; CPU quota observed rather than asserted | Require a controller-enforced outer deadline and teardown reserve before metered execution |

## Sources, checks and limits

Reviewed the exact-CPU protocol in `research/campaign_2026-10-15/vast_a_cpu_probe_r1/PROTOCOL_AND_MARKET_R2.md`; fixture manifest and producer source in worktree `/home/vlad/development/wunder/worktrees/vast-a-cpu-probe-r1-20261015` at commit `e7f9561c8624582dc9154c645be89bdaf17acfe8`; and the portable benchmark worktree `/home/vlad/development/wunder/worktrees/cpu-portable-benchmark-20261015` at final commit `337a3a9d40b331959f83395d56941b885796d577`. The runner SHA-256 is `be2697340cc90d676745f52d33b04e22794dc80c338c34a6bfb30527b671825b`; package-binding SHA-256 is `61a11ab4ea6bf89df3f2789ae0a64db6e5064c12e1d554150ea55f118a0f2367`; fixture-manifest SHA-256 is `941b656d8408be6f5c8f5816c1e32f3c8b24e59040bb4ed84bb895078a9f9f05`; and smoke receipt SHA-256 is `afc48e63a62e7b51ada060cb04e67abddbec3e04444c790516fe63ecaffc6821`.

The recorded four pure-protocol tests cover order balance, minimum-block admission, paired intervals and separation of timing variation levels. I independently checked the frozen source and receipt hashes and inspected the smoke fields; I did not rerun the harness or read the fixture payloads. The full timing panel remains unrun. The exposed F0 sequence panel supports runtime measurement only; it cannot establish model quality, performance on unseen sequences, cross-host generalization, or competition-serving capacity.

## Decision and next step

Pass for a future **exploratory** exact-CPU panel after the root verifies a runnable CPU-only EPYC 9224 host and admits a hard external time limit. Current availability evidence does not satisfy that host condition, so the panel remains pending without a provider action. Preserve the smoke as compatibility evidence only. If a panel is eventually run, preregister separate confirmation before making a winner claim and report both host and sequence-panel limits.
