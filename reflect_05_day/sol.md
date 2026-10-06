My diagnosis: **the campaign has enough ideas; it lacks decisive tests against the complete winning predictor, and several conclusions were broader than their evidence.**

I wrote the [full audit and prioritized plan](/home/vlad/development/wunder/wnn_connectome_starterpack/research/RESEARCH_DIRECTION_AUDIT_2026-10-05.md), tracing claims to saved results and code.

The biggest elephants:

- **Prototype success is replacing transfer testing.** Most of the 224 activation fits used small models and the same exposed panel. The surviving fixed GELU/LeakyReLU mixture deserves a GRU256 comparison through the final-head procedure.
- **The “overlooked 0.70404 winner” is an invalid holdout comparison.** `ab85_final_ema6_s4_gclpall` explicitly trained on all validation rows, including the claimed holdout. Its apparent advantage cannot justify promotion.
- **The event definitions differ.** Target-mechanics diagnostics use *either* top-price code changing; production’s event flag requires *both*. Existing inputs already contain event information. The opportunity is changing state updates, rather than adding another event feature.
- **Stopped bootstrap acquired an unjustified theoretical veto.** Its public success is real. The older “unbiased” claim rests on a small linear-projection diagnostic and does not establish that original-target teachers or richer temporal states have no room.
- **The public class mix is inferred, not observed.** PUB143 is a sensitivity scenario. The local/public score gap does not measure a recoverable distribution-shift loss.

I would prioritize:

| Order | Direction | Decisive next test |
|---|---|---|
| **1 — immediate** | Fixed GELU/LeakyReLU readout | Two matched GRU256 seeds, unchanged control, identical final-head procedure. |
| **1 — structural** | Persistent event state plus quiet-row belief updates | Compare with an equally sized unrestricted residual branch and stale-event control. Preserve updates between events. |
| **2** | Origin-aligned privileged teacher | Teacher sees through `t+8` but predicts **the original `Y_t`**, with original weights; compare against a causal teacher and test student transfer. |
| **3** | Target-specific history assimilation | Give both heads both clocks; compare learned asymmetric adapters, swapped assignments, and age/hazard controls. |
| **4** | Actual distilled normalizer and cadence | Finish the prepared pooled-accumulation/credit comparisons; they remain scientifically untested. |
| **5 — conditional** | W384 and quantization-enabled diversity | Establish real memory/runtime feasibility, then replicate the score comparison. |

The ideas worth reopening are **dynamic event/quiet state allocation, original-target teachers, target-specific input clocks, and structured heads with a competitive regression control**. W384 is resource-censored. Joint codebook heads have a narrow negative with a losing fitting control. Equal-target reweighting has stronger counterevidence and should stay low priority.

I would start with **one readout transfer branch and one event/posterior branch**, reserving confirmation compute before selecting a winner. Best public remains **0.6660**; expected public **0.7000** remains unsupported.

This was an evidence review; no training or uploads were launched.