# Local NLL adaptation

Frozen two-arm recipe against the H256×2 seed-3 F0 parent. The WMSE and NLL arms share initialization, strict-FP32 update schedule, target cache, and one appended zero-initialized logvariance head; only the objective flag differs. Exact commands and all known pins are in [recipe.json](recipe.json).

Status: both bounded profiles passed; the matched two-fit sequence is running under `wnn-overnight-nll-fit-20261020.service` (invocation `d2a82026557c4f5b915a361b8f8e375a`). Profile uses 2 warmup plus 4 measured full batches per arm (60 chunk updates total) and omits `--save-training-state`; fit uses two full epochs and fixed zero-based checkpoint epoch 1. Root owns launch and monitoring.

Accounting: 2 new scientific fits total, one per arm; no new teacher fits. The target cache stores 0.8125 times the mean of three fixed MainT3 teacher epoch-2 predictions; keep cache scale 1.0. Parent/cache artifact names and target semantics are pinned in the JSON. A positive small CPU MLP NLL pilot is only a directional prior and does not establish GRU transfer.

This is a strict-FP32 adaptation from a historical FP16 parent, not a fresh-FP32 replication. The one-seed/one-fold comparison is directional and F0 is exposed through ancestry. Root's grader should score parent, each child and the fixed 50/50 parent/NLL prediction blend without coefficient search. Prior parent FP16 time was 932–1086 seconds per epoch; the new strict-FP32 profiles measured about1.68seconds per full batch, projecting42minutes per two-epoch fit. This is a short-profile projection, not a measured fit duration.


## Execution evidence and navigation

- [Original preparation](recipe.prepared.json) and [admitted recipe](recipe.json): same scientific parameters; root reduced allocator cap to4.5GB and bound unchanged Parquets by size/mtime/footer identity. The recipe retains its original preparation status; the dated admission is authoritative.
- [Input verifier](verify_inputs.py) and [actual input verification](input_verification.json): source, checkpoints, target and propensity caches verified before profiles. Four propensity digest placeholders in the recipe are resolved in this receipt.
- [Profile admission](profile_admission.json), [literal profile plan](profile_plan.json), [raw profile log](profiles.log), [sequence report](profile_plan_report.json) and [profile result](profile_result.json):60/60updates per arm, finite losses and preclip gradients, no AMP skips;41.30seconds total,3.49GB peakRSS, zero swap. Final parameter/optimizer-state finiteness and CUDA peak were not emitted by this generic profile path.
- [Fit admission](fit_admission.json) and [literal fit plan](fit_plan.json): one new fit per arm, no new teachers, three-hour maximum for the pair. Practical screen effect0.0002WP is a follow-up guide, not a significance threshold. Latest progress is in [fit log](fits.log) and [sequence report](fit_plan_report.json); existence of a running worker is not a completed result.

The NLL profile clipped every update (mean clip factor0.176), compared with11.7% for WMSE. Nearly98% of squared gradient mass was in the readout, initially learning variance from zero logvariance. This is an observed optimization difference to track, not a demonstrated gain or an automatic rejection. The declared objective comparison includes this interaction with common global clipping; changing variance initialization or per-head clipping would be a separate follow-up.
