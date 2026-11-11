# Independent design review

Two independent read-only ideation roles examined representation/Transformer and
objective/state directions before seeing each other's suggestions. A separate
read-only audit role reviewed PR60–80. The representation role then criticized the
selected portfolio. These are agent reviews from related model families, not
independent empirical evidence. Root owns the final synthesis. No reviewer ran a
model or test, and no benchmark result was independently reproduced here.

| Finding | Resolution in this packet |
|---|---|
| Plain sums of groupwise linear maps equal a flat linear map. | Card02 requires nonlinear within-row token mixing; it remains a capacity-confounded package comparison. PR8 is named as its ancestor. |
| Linear late adapters followed by a linear head can collapse algebraically. | Card03 uses GELU in rank8 private branches and explicit zero-initialized residual outputs. |
| PLE can add capacity and collapse on duplicate quantiles. | Card01 preserves raw bypass, specifies duplicate/constant-feature behavior, and calls capacity attribution unresolved. |
| Tied parameter blocks must not share logical-layer KV state. | Card05 specifies24 caches,12 weight sets and optimizer deduplication. Fresh control copies initial block values but unties parameters, matching the initial function. |
| Deep heads can improve themselves without helping the final predictor. | Final fixedE4 remains the primary head/endpoint; no valid-selected head or coefficient. |
| Ranking is not weighted Pearson. | Card06 freezes bounded same-chunk lag32 pairs, margin, weights and coefficient; concordance is diagnostic only. Train has no is_scored column. |
| Future targets can accidentally create lookahead and are row offsets, not verified clock horizons. | Card07 allows a loss-only label buffer, sequence-end masks and no future features/teacher forcing. The three-head package is one contrast, not three attributed effects. |
| Noise regularization may destroy useful memory or increase compute. | Card08 fixes fork position, stride, sigma,32-row suffix and clean-state continuation. It claims empirical sensitivity only, with all overhead reported. |
| “NLL transfer” could hide prerequisite parent fits and overstate older evidence. | Card09 starts both arms from scratch; only the idea transfers. Old one-panel result and its zero-crossing interval are recorded. No parent/teacher fits are assumed. |
| Tail events disappear if classes are made from clipped targets. | Card10 requires raw train labels, fixed support thresholds, no class balancing or threshold retuning; infeasible support means no fit for that recipe. |
|40 fits is not40 equally costly jobs or a launch budget. | Only an inventory; selected fits=0, receiver/timeout unassigned, no scheduler or runnable command. |

Outstanding work is explicit: separate mechanism implementation, source integration,
data/lineage binding, remote correctness and profile receipts, fit admission and
numerical evidence. Equal-parameter nonlinear controls for cards01–03, alternate
group maps for02, and per-horizon ablations for07 require separate allocation if
the initial package result makes those questions useful.
