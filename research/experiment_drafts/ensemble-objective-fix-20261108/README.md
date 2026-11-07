# Four-fit ensemble-objective package, 8 November2026

The [explicit receiving contract](CONTRACT_R1.md) authorizes exactly four fresh
matched4E teachers: labels WMSE control and fixed0.9 cleanTL3+0.1z package,
seeds101101/101102. Official train10325 only, zero valid fitting ancestry,
full valid1873/3,528,273; validation is repeatedly exposed development evidence.
Primary endpoint is each fixed4E mixture, with standalonez diagnostics from the
same evaluation forwards. Two seeds establish no significance or public gain.

The package uses0.5 unscaled mixture WMSE +0.5 attenuated weighted correlation
loss. WMSE is sum(abs(clip(y))*squared(raw mixture-clip(y))) divided by total
original-label weight across both targets and the complete actual A4 window.
Correlation clips the FIXED MIXTURE and y to[-2,2], uses each target's original
abs(clippedy) weight, global centered FP32 moments, and averages targets equally.
The ROOT-authorized pretrial denominator is std_y*max(std_mixture,.01), in
clipped-prediction std units. This equals Pearson*min(1,std_mixture/.01), not exact
WP in the nearconstant regime. Target std<=1e-8 yields zero correlation; selected
zero-weight/constant targets have loss1/zero correlation gradient; globally empty
row masks have loss0. Selected raw nonfinite values reject before clipping;
excluded NaNs are sanitized before multiplication. No official grader changes.

Differentiable SUM backward replication cancels DDP parameter-gradient averaging;
no extra world-size factor is added to the correlation term. Correlation losses
are averaged within each actual microchunk accumulation window, including the
short final window. This is a declared microbatch surrogate, not dataset Pearson
or Pearson computed across the entire accumulation window. Both curvature
removal, correlation and numerical regularization are package factors; no gain
may be attributed to just one. The original no-floor nearconstant oracle failure
is preserved as numerical evidence; tolerances were not widened.

Preserved numerical recipe: T768/L24/H12, B128/A4/world2, context1024/chunk256,
BF16/BF16x9 with FP32 masters/grads/loss/state, original unbatched Muon3e-4,
decay.01/clip1, cosine5%warmup/minLRratio.1, fixed4E/3,280updates. All4 source-
qualified configs are in [configs](configs); exact POSIX native50-update commands,
environment and pair assignments are in [profile-plan.json](profile-plan.json).
The max4h clock began at ROOT's first actual new GPU qualification, with only3h
available for fits and1h reserved for evaluation/collection; it cannot reset.

Current local evidence:46 bounded CPU checks pass, modular114-file check has zero
errors; actual2-rankGloo gradient oracle passes8 cases at unchanged atol3e-6 /
rtol8e-5, maximum absolute error1.192e-7. ROOT's actual B300/NCCL8-case oracle
also passes with the same maximum error and frozen loss source hashes. These
checks qualify arithmetic, not full-shape BF16/BF16x9 or model quality.

The training source is frozen separately from the receiving adapter. Every
new native profile additionally round-trips actual-shaped model, optimizer,
scheduler and host/CUDA RNG with strict container/dtype/shape/stride/byte checks,
coordinated over both ranks. Successful temporary engineering state is removed;
no resumable scientific profile schema is written. These native checks, memory,
finite updates, ROOT approval and safe retirement of the old partial30E attempts
remain fit-admission gates. The old model capsules/configs/results remain intact.
ROOT exclusively owns SSH, hardware admission and process signals.
