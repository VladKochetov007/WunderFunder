# Next research cycle: 30 local GPU-hours plus Kaggle

Accepted user plan, October 21, 2026. Owner: root. This is a new allocation,
superseding the earlier cycle's prospective priorities, not rewriting its costs
or results. Local and Kaggle only; B300, paid compute and external competition
upload are not authorized by this plan.

## Objective and diagnosis

Produce a stronger complete submission through informative comparisons.
The user reports the latest package at 0.66667 / 53m47s versus TL3 at
0.66832 / 39m12s. This is user-supplied public-transfer evidence. H512 reproduced
its B300 result, but neither that replication nor the tested local blend
established a profitable TL3-member replacement. The local blend trailed WMSE.

## Frozen priorities and allocation

| Stage | Local GPU-hour ceiling | Deliverable |
|---|---:|---|
| Saved-artifact recovery | 2 | Kaggle CP5/SWA, fixed checkpoint mean, corrected actual-package comparisons |
| i1 temporal-memory screen | 5 | A/B/C × F0 seed1001 and F2 seed1002; six fits and full-fold predictions |
| Future-input dynamics | 3 | Weight 0/0.05 × same two cells; four fits and full-fold predictions |
| Strong-parent follow-up | 8 | Conditional matched control/treatment continuations on F0/F1; four fits |
| Production/export/recovery reserve | 12 | Conditional final training and one fully evaluated package |
| Total | 30 | Profiles, exports and failed attempts count |

One GPU worker, two aggregate CPU cores, 5,000,000,000 bytes aggregate task RAM,
5 GB VRAM, no task swap, physical scratch at
`/home/vlad/development/wunder/research_tmp`, strict FP32 with autocast/TF32 off.
Reserve each complete panel only after sustained actual-shape profiling; account
for thermal slowdown. Defer a panel that cannot fit, without truncating one arm
or borrowing the production reserve. Recheck Kaggle quota and other-agent
reservations before use. The plan's approximately 39-minute balance and October24
refresh are historical snapshots, not admission authority. Keep paired arms on
the same device and recipe. Never spend another agent's reservation.

## Comparisons

1. **Checkpoint mean: zero new fits.** Execute the existing frozen equal mean of
   parent CP5 and WMSE continuation CP0/CP1, requiring identical tensor topology
   and normalization buffers. One candidate, no coefficient search or averaging
   of independently initialized models. Compare the fixed 75/25 oldZ4/mean
   endpoint against oldZ4/WMSE CP1. Recover existing Kaggle CP5/SWA evidence.
2. **i1 memory: two fits per arm, six total, zero teachers or reused fit controls.**
   Preserve 367 slots. A masks all raw/derived i1 and history; B uses standard
   inputs with zero history; C adds causal previous-EMA residuals at half-lives
   32/512 and slow change energy. Cold H512×1, 10% fitting components with
   complete sequences, three epochs, fixed CP2, prepared AdamW/OneCycle recipe,
   fitting-only normalization. F0/1001 and F2/1002 are matched within each cell.
3. **Dynamics: two fits per arm, four total, zero teachers or reused controls.**
   Same small-screen backbone/subsets. At each causal hidden state predict
   52 training-only summaries: mean standardized change and log change energy
   for 13 input groups at future horizons 16/256. Standardize targets on fitting
   data only. Both arms have the same head and initialization; mean Huber weight
   0 versus 0.05 beside unchanged WMSE. Future inputs are targets only. Remove
   auxiliary heads for serving. Improved auxiliary prediction alone is failure
   to earn a larger run if target quality and fixed ensemble contribution do not
   improve. This adapts CPC's predictive-representation principle, not CPC itself.
4. **Strong parents: two fits per arm, four total, zero new teacher fits.**
   Conditional on screens, use clean F0/F1 parent/cache catalogs. Match initializer,
   targets, sampling, optimizer and two-epoch schedule. Test one selected mechanism,
   not a history/auxiliary combination. Compare paired deltas, not absolute scores.
   Prefer cheaper serving when evidence is comparable. Establish benefit over the
   strong continuation control before production; a weak screen winner is not
   automatically a submission.

## Evaluation and release

Keep `global_wp_clip2_v1`, full source-qualified evaluation folds, fixed checkpoints,
both targets, fixed sequence-position bins and paired component uncertainty.
Report every paired delta. All folds are exposed; two fold/seed cells cannot
separate those axes or establish significance. +0.0003 mean official improvement
is a practical follow-up target, not a universal cutoff. Smaller consistent gains
or useful fixed ensemble contribution can earn follow-up. Mixed results require
diagnosis; an interval crossing zero does not close a direction. Use the prepared
fixed 90/10 control/candidate mixtures for the small screens, without tuning.

Grade the actual native complete endpoint against its unchanged reference before
release. TL3 comparisons remain ancestry-qualified diagnostics. Preserve
`PredictionModel.predict`; verify prefix causality, chunk/state resets, semantic
masks, checkpoint restore, head removal, exact cache coverage including NPY
headers, native quality/runtime and source/teacher/initializer ancestry.

## Deferred work

Defer more circle/geometry bundles, output Kalman smoothing, warmup routers,
leaderboard blend sweeps, width/depth grids, unchanged Gaussian NLL/correlation
sweeps, online weight/RLS/reset adaptation, Transformers/DDP, quantization and
kernel work. Preserve negative evidence without rejecting materially different
descendants. Per-field age is the next unresolved representation alternative
if i1 fails informatively.

## Durable execution

Reuse canonical training components and optional auxiliary defaults off. Preserve
historical recipes and unrelated dirty work. Record measured phases, costs and
failures in [the cycle frontier](../../research/campaign_2026-10-21/next_cycle_r1/FRONTIER.md)
and amend [the previous frontier](../../research/campaign_2026-10-20/plateau_review_r1/FRONTIER.md).
Index hypotheses, public receipt, failures and artifacts through unified READMEs.
Draft publication remains authorized under the existing research-team workflow.

## Primary-source scope

Root verified the primary abstracts on October21: [SWA](https://arxiv.org/abs/1803.05407v3) averages points on cyclical/constant-SGD trajectories; the present AdamW checkpoint proposal is an adaptation. [CPC](https://arxiv.org/abs/1807.03748v2) predicts future latent information with a contrastive objective and negative sampling; our52summary Huber objective borrows predictive supervision, not that method. [Test-time training](https://arxiv.org/abs/1909.13231v3) adapts parameters using self-supervision on test samples; it motivates only deferred research cards here. None establishes a gain on this dataset. No external weights or data were imported.
