# Independent review of Grok 4.7 reframe

Reviewer: native Luna/max critic, read-only review against the project contract and `src/wnn/gmetric.py`. The raw prompt and response remain unchanged.

## Blocking metric correction

The Grok prompt and response incorrectly call the official objective fixed globally weighted squared error and claim leaderboard ranking is monotone in that loss. The official `global_wp_clip2_v1` instead clips targets and predictions to `[-2, 2]`, weights by absolute clipped target, pools rows for each channel, computes weighted Pearson, and averages channels. Weighted MSE can be a training surrogate; it cannot substitute for endpoint scoring. Also, weighted squared row loss can reach 32 before normalization (`|clip(y)| ≤ 2`, squared clipped residual `≤ 16`), not 16. The official formula is in `src/wnn/gmetric.py`.

Thus the report's card ordering and numerical claims are not treated as metric-grounded evidence. Raw source ideas are retained as hypotheses, not allocated rewards.

## Proposal disposition

- Cards 2, 3, 4, 6, 7 and 8 survive as hypotheses if all final comparisons use global clipped weighted Pearson. Card 1 can only motivate diagnostics; there is no additive per-row official score contribution for CVaR.
- Card 4's weighted-MSE optimum is a target-weighted conditional mean, not an ordinary conditional mean, and neither is automatically optimal for Pearson. Begin with rail-mass diagnostics and matched heads scored by global WP.
- Card 5 gates on a true residual that is unavailable at inference. Defer unless it is replaced by a legal, learned inference-time gate with cross-fitted residual labels. It overlaps E-177/H-184 and has lower priority.
- Card 6 merits a cheap audit of legal, simultaneously observed inputs before any model work. Card 7 is distinct from the rejected pure-TCN recipe, but lower priority. Card 8 needs evidence that its explicit time-gap mechanism differs from existing two-clock/event/state arms.

None of these proposals displaces current C1 grading or the E508/E510/E512 cloud dependency chain. The current folds are already exposed; changing group partitions does not make them fresh. No numerical priors or expected public gains are inferred.
