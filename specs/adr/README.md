# Architecture and research decision records

ADRs explain why a consequential research or orchestration decision was made. They are not claims that its hypothesis worked. Link the actual evidence and alternative costs; declare consequences and conditions for revisiting it.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-long-term-winning.md) | Optimize long-term winning with exploration and deepening | Accepted |
| [0002](0002-layered-evidence-memory.md) | Keep compact specifications, indexed episodes and immutable evidence | Accepted |
| [0003](0003-independent-diverse-ideation.md) | Use diverse agents and random/directed literature components | Accepted |
| [0004](0004-independent-training-and-target-variants.md) | Train own models; explicitly distinguish raw/clipped target recipes | Accepted |
| [0005](0005-result-ownership-and-independent-arms.md) | Assign end-to-end result owners and arm-specific dependencies | Accepted |
| [0006](0006-resource-safe-autonomous-execution.md) | Bound resources and autonomously repair identified execution failures | Accepted |
| [0007](0007-specialized-research-skills.md) | Specialize research workflows around concrete failure modes | Accepted |
| [0008](0008-research-scale-beyond-serving-envelope.md) | Pursue top-1/>0.700 with training scale beyond submission limits when rules permit | Accepted |
| [0009](0009-uncertainty-is-not-a-kill-switch.md) | Treat uncertainty as evidence, not a universal stop rule | Accepted |
| [0010](0010-continuous-experiment-delivery.md) | Keep authorized experiment owners active through repairs and measured results | Accepted |

For a new decision, copy [the template](template.md), choose the next unused number and add it here. If superseding a decision, preserve its text/history and link both records. User instructions take precedence; record their effective scope rather than quietly editing an old scientific specification.

- [ADR0011: measured pipeline utilization](0011-measured-pipeline-utilization.md): verifieddigestreuse, completecohortcost and bounded loaderprofiling.
- [ADR0012: independently implement teammate methods](0012-independent-teammate-method-implementation.md): read code for ideas and implementation details; independently write training and serving code in our own project conventions.
- [ADR0013: bounded paid compute and owned instances](0013-owned-paid-compute-and-continuous-queue.md): finite useful-run queues, $4 initial/$20 cumulative Vast spend, and no actions on pre-existing or unrelated machines.
- [ADR0014: continuous useful compute](0014-continuous-useful-compute.md): keep local/Kaggle/authorized Vast lanes supplied with useful work until explicit user stop; own repairs, terminal artifacts and prepared successors.
- [ADR0015: flexible credit-bounded parallel compute](0015-flexible-credit-bounded-parallel-compute.md): latest user spending instruction removes the first-cohort release gate, retaining existing credit and a conservative $20 cumulative operating ceiling with reviewed bounded cohorts.
- [ADR0016: exact teammate-source diagnostic training](0016-exact-reference-diagnostic-training.md): latest user instruction permits isolated exact-source reference training with independently trained weights; production submissions remain independently implemented.

- [ADR0017: continuous research and five-core morning cap](0017-continuous-research-five-core-morning-cap.md): research continues until explicit stop; local CPU drops to five cores at10:00Kyiv with unchanged memory/VRAM and separate cloud allocation.

- [ADR0018: attribute OOM to the resource boundary](0018-attribute-oom-to-resource-boundary.md).

- [ADR0019: common-parent final student pair](0019-common-parent-final-student-pair.md): bounded paired endpoint now, separate all-data teacher-history follow-up; shared ancestry is explicit.

- [ADR0020: explicit precision in paired representation tests](0020-explicit-precision-in-paired-representation-tests.md): preserve the FP16 control and skipped-update evidence; independently test both representations under an audited FP32 numerical treatment.

- [ADR0021: actual producer/consumer compatibility](0021-validate-consumers-against-producer-artifacts.md): source-qualified identities, real serialized fixtures and header/payload accounting; preserve completed clean fits during delivery repairs.

- [ADR0022: prepared successors and paid warm cohorts](0022-prepared-successors-and-paid-warm-cohorts.md): keep accelerator-specific executable queues ready before completion, overlap preparation/review, amortize data loading and terminate our paid workers when useful admitted work is exhausted.
- [ADR0023: faithful method reconstruction before variants](0023-faithful-method-reconstruction-before-variants.md): reproduce teacher training, seed-specific parents and student blends with independent code; make optimizer, baseline, normalization and schedule deviations explicit comparisons.

- [ADR0024: AMP adaptation and empirical result delivery](0024-amp-adaptation-and-result-delivery.md): preserve legitimate skipped-update evidence and finite artifacts instead of a universal zero-skip rejection gate.

- [ADR0025: bounded Sol continuation delivery](0025-sol-continuation-delivery.md): one authorized Sol lead, independent Luna verification, exact Z3 comparisons, serialized support, and signed-input capacity cohort with capped staging/spend.
- [ADR0026: Luna-only unattended execution](0026-luna-only-unattended-execution.md): withdraw premium-agent exception; transfer context to Luna/max and deliver a bounded persistent experiment manager.
- [ADR0027: diverse models and second-level learning](0027-diverse-models-and-second-level-learning.md): prioritize diverse own teachers/students and learned combinations for24h; one new explicitly authorized Astra/xhigh ideator, independent Luna/max investigations and requested external providers within existing caps.
- [ADR0028: Sol medium diversity delivery](0028-sol-medium-diversity-delivery.md): latest explicit user authorization replaces the Luna lead with one Sol6.1/medium research/implementation lead; two operators remain Luna/max and no Astra task is authorized.
- [ADR0029: Luna max only](0029-luna-only-native-diversity-delivery.md): latest user supersedes the Sol exception; transfer its completed native-device draft to an isolated Luna/max role and keep all subordinate work on Luna/max.

- [ADR0030: local evidence selects submissions](0030-local-evidence-selects-submissions.md): research and local validation select models and fixed blends; public leaderboard confirms transfer.

- [ADR0031: cross-fold top-tier release](0031-cross-fold-top-tier-submission-release.md): diversified clean cross-validation ranks complete endpoints; deliver only the strongest supported submissions after exploration lands, then exploit measured serving headroom.

- [ADR0032: package available full-data blends now](0032-package-available-full-data-blends-now.md): latest user supersedes ADR0031's waiting requirement; deliver a small reviewed set now, retain honest cross-validation/provenance, and measure Q4/Q8 endpoint value.

- [ADR0033: complete endpoint compression and Sol analysis](0033-complete-endpoint-compression-and-sol-analysis.md): optimize the actual served blend under size/runtime constraints; the user's explicit Sol6.1/xhigh exception covers analysis roles, while implementations and artifact reviews remain Luna/max.

- [ADR0034: resumed strategy execution and full GPU allowance](0034-strategy-execution-full-gpu-allowance.md): implement every actionable strategy-review idea, use all Kaggle allowance and actual local GPU capacity, preserve CPU/RAM/no-paid-spend constraints, and allow at most.001 mean-CV sacrifice for private robustness.

- [ADR0035: official team code and weight sharing](0035-official-team-code-and-weight-sharing.md): direct reuse of official teammates' code, native libraries, weights, teachers and initializers is authorized; source/fit ancestry, independent criticism and valid evaluation remain required. Complete joint-model research before new ZIPs.
- [ADR0036: QAT research with renewable capacity](0036-qat-renewable-capacity.md): implement the indexed W4/W8, activation-aware and learned-scale plan with no campaign GPU-hour ceiling; retain actual provider/physical limits, clean ancestry and complete-endpoint validation.

- [ADR0037: CPU student frontier and asynchronous probes](0037-cpu-student-frontier-and-asynchronous-probes.md): fixed-weight native comparisons, architecture/precision/initializer search and continuous theory/training while user uploads later;5GB observedVRAM for new admissions.

- [ADR0038: Vast-A exactCPU benchmark budget](0038-vast-a-exact-cpu-benchmark-budget.md) records explicit$3authority on`vastai`, exactEPYC9224requirement, CPU-only access uncertainty and no-rental status.

- [ADR0039: two local CPU cores and physical-disk scratch](0039-two-core-local-budget-and-disk-scratch.md) supersedes five-core allocations after laptop responsiveness/storage concerns; preserves GPU research, model recipes and checkpoints.

- [ADR0040: research-team draft experiments](0040-research-team-draft-experiments.md) records the October17 user protocol: per-idea draft PRs,30 paired cells per arm, strictFP32, English execution/evidence handoffs and unified merge trace; local preparation before push, no queue or new fits.

- [ADR0041: modular training code](0041-modular-training-code.md) makes the semantic package/thin CLI, uv/Ruff, explicit run state and300/200-line structure checks the shared main/research-team policy; preserves frozen source and numerical recipes.

- [ADR0042: local/Kaggle plateau-review budget](0042-local-kaggle-plateau-review-budget.md) records the October 19 user revision: 15 local GPU-hours, verified available Kaggle quota in parallel, a scoped waiver of the 30-cell matrix, unchanged 2-CPU/5-GB/no-swap bounds, preparation now and experiment start October 20. Full-data fits remain allowed; the canonical plan and frontier preserve continuation after compaction.

- [ADR0043: fast screens and large Transformer teachers](0043-fast-screens-and-transformer-teacher-scale.md) replaces mandatory CP30 for future drafts with4–6 total new fits per initial screen, explicit cached-control and teacher/student accounting, and targeted follow-up. Sets the8×B300 causal-Transformer→GRU priority while preserving FP32, ancestry, historical protocols and existing compute/start boundaries.
