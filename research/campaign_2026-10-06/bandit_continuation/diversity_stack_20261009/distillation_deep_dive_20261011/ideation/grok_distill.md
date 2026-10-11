I'll read the three distillation notes and the campaign's current evidence so the six methods sit on this dataset's actual constraints.Distillation can recover part of the blend, and the current +0.0008 is a weak test of that, not the ceiling. A realistic target for one causal GRU-256 is about **+0.001 to +0.002** over continuation (~0.6686), toward the diverse-pair blend (~**+0.003**, roughly 0.672–0.673). Beating the blend needs a target no causal teacher output contains. Seed noise is ~0.001, so anything under that needs three seeds before it counts.

The blend gain is the small slice of error that is **not** shared: members share 96–99.9% of the residual, 94% of residual sum of squares is a per-run level, and class 3 holds 69% of that residual at correlation 0.44. Row-wise MSE to the mean of two similar teachers, mixed 30% with the raw stepped labels, for 3 epochs at 1e-4, spends its gradient on the shared part the warm start already has. Wider teachers already failed this way (h384 distill 0.66836 vs continuation 0.66861). Scratch runs that keep the 1e-3 one-cycle past epoch 3 memorize (0.667 → 0.649). Every method below uses a long low-LR tail instead of a longer high-LR cycle: AdamW from the same parent, 2 epochs at 1e-4, then 8–12 epochs cosine from 3e-5 to 1e-5. Label weight stays ≤0.1 during the tail so the stepped labels do not paint the ensemble correction back out.

## 1. Event-run level distillation

**Mechanism.** Labels are piecewise constant (mean run ~9.5 rows) and change at book events. The metric pools rows, so a wrong level on a long run dominates weighted Pearson. Distill the ensemble’s level on each event-run, and match shape only inside the run.

**Loss.** Segment runs with the existing event flags. Let `L̂` be the ensemble mean over the run. Loss = `0.8 · WMSE(run_mean(student), L̂) + 0.2 · WMSE(student − run_mean(student), teacher − L̂)`, label weight 0.1, zero extra weight on duplicated intra-run rows. Same long low-LR tail. Aux head: predict the run level from the GRU state at the event row and hold it until the next event.

**Signal.** Ensemble outputs, collapsed to a per-run level, plus within-run residual.

**Data.** Train split only. Runs from train sequences.

**Falsifier.** On cached fold-1 predictions, replace each model’s within-run mean with the ensemble’s run mean and rescore. If that oracle is under +0.001, there is no level headroom. If it is ~+0.002–0.003, this is the whole prize and a real fit should land +0.001–0.002.

**Pointer.** Kim & Rush, Sequence-Level Knowledge Distillation (EMNLP 2016): supervise the sequence statistic, not every token.

## 2. Orthogonal residual of diverse teachers

**Mechanism.** Same-recipe seeds buy ~+0.001; an h256+h384 pair buys ~+0.003. The useful target is `ensemble − parent`, a few percent of the residual. MSE to the mean ignores that and the raw h384 target already transferred nothing.

**Loss.** `WMSE(student, y)` with weight 0.3, plus `β · a_r · MSE(student − parent, ensemble − parent)`, `β` from