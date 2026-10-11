Now I have a thorough understanding of the context. Let me propose 6 distinct distillation methods tailored to this specific data.


Done. Wrote 6 distinct distillation methods to `DISTILL_PROPOSALS.md`. Each addresses the specific data characteristics:

1. **Per-Run Level-Offset Distillation** — targets the 94% run-level offset residual
2. **Hidden-State Alignment with 1-Row Lead** — exploits label-leads-features structure
3. **Multi-Teacher Disagreement-Weighted** — downweights teacher noise in class 3 via disagreement variance
4. **Quantile/Rank Target Distillation** — aligns with metric's clipped Pearson weighting
5. **Class-Conditioned Distillation** — specialized blends per regime (class 3 = 69% residual SS)
6. **Counterfactual Event Intervention** (unconventional) — forces event-invariant label levels