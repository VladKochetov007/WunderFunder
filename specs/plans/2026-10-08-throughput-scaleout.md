# Scale toward five to six evaluated idea screens per hour

Effective user instruction, 2026-10-08 Kyiv: when Kaggle and Vast are up, scale to at least five to six ideas per hour. Continuous execution remains authorized. This is an allocation target, not a measured capacity claim or a requirement to shorten every scientifically useful experiment.

## Count useful results

An idea screen resolves a declared mechanism comparison with valid predictions, the fixed official metric, matching fitting/evaluation membership and an identified control. Seed repeats, profiles, uploads, failed startup attempts and several views of the same comparison do not become additional ideas. Record them as runs and infrastructure work separately. A control may be reused only where recipe, membership, ancestry and training budget make it appropriate; otherwise include its fit cost.

Report completed distinct screens per wall-clock hour and a rolling three-hour rate. Also report full fits, GPU-hours per completed screen, startup/hash/load/fit/export/evaluation time and failed attempts. Separate initial cold-start time from sustained warm throughput. No warmed-data claim without an actual cache/session hit.

## Capacity and allocation

Measured local compact fits currently take approximately 33–36 minutes; a larger local fit can take 85–90 minutes. Cloud production fit throughput is not established yet. Do not multiply local timing by advertised TFLOPS to manufacture a cloud rate.

For N independent GPU worker slots and T minutes of total accelerator work per completed idea, ideal capacity is N × 60 / T ideas per hour. This excludes queueing and CPU, I/O or evaluation bottlenecks. Five active slots require roughly 50–60 accelerator-minutes per idea to reach five to six per hour. If each idea needs two new fits, that means approximately 25–30 minutes per fit. Longer recipes require more slots or a lower-fidelity screen; full training retains its own ETA.

First obtain actual fit/profile measurements from local, Kaggle and one owned Vast cohort. Then size additional owned Vast rentals from observed cycle time and useful prepared recipes, rather than repeatedly renting machines that only stage data. Consider both Kaggle T4s when actual host RAM, data loading and per-device profiling support it. TPU capacity counts only after real-device numerical and full-shape checks succeed. Follow ADR0015's existing-credit ceiling and strict ownership rules.

Keep heterogeneous-method confirmation and public-finalist training moving while establishing a separate screening lane. Screen model-family changes, losses/target treatments, causal representations and feature interactions ahead of seed expansion. Reserve full-scale follow-ups for plausible scale-dependent ideas even if their small-data rank is weak.

## Remove repeated overhead

Use identified multi-recipe cohorts with verified data staged once and reusable metadata/hash handoffs inside the same worker. Prefer profile → conditional fit → export in one session over a profile-only kernel followed by another cold startup. Keep exact independent recipe/checkpoint/prediction receipts for every fit. Mount identity changes still require verification; a cache from a different Kaggle worker is not automatically valid.

Apply the [small-data screening protocol](2026-10-08-small-data-screening.md): nested fitting components and whole causal histories, fixed full validation, paired budgets and explicit full-data teacher ancestry. Pilot two fidelities across several families before relying on subset ranks. Training subset results are allocation evidence, not public-score estimates.

Prepare the next diverse cohort during current fits. Keep memory headroom measured across concurrent processes; use one local GPU worker. Do not sacrifice healthy informative jobs merely to hit an idea count. Add parallel cloud capacity only with reviewed runnable successors and bounded all-in reservations.

## Ownership and next decisions

- Cloud operator: H232 same-session conditional profile/fit/export packet, then measured two-T4 cohort feasibility and reusable-data timings.
- Vast lifecycle actor: corrected bounded staging/setup/collection for the owned H217 pair, actual device timing, then a warmed diverse successor cohort and capacity/cost quote.
- H230 actor: finish healthy matched control, export and deliver to independent grading; audit the independently implemented public-finalist recipe and identify separate subset screens.
- Root controller: review exact packets, admit useful fits, compare measured cycle times, allocate extra owned capacity and update the tree/notes. Reassess the allocation every two hours.

Status at plan creation: local matched H230 control fitting; Kaggle H232 V2 failed before profiling due to Path JSON serialization and corrected same-session packet is being prepared; TPU queued; Vast successor prepared but not yet rented. Five to six completed screens per hour is not achieved yet.
