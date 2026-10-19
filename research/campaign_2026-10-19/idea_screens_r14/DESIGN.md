# R14 design: same-span exponential filter coordinates

## Question and scope

H-NF19-18 predicts easier optimization from an orthonormal response basis while preserving poles, state count, information and fitted objective. This bounded linear full-fit test addresses that question directly; it is not another model-family/EMA-information comparison, ridge sweep or GRU quality claim. Two coordinate systems describe the SAME448-feature affine predictor and SAME regularized weighted squared risk. No new neural/GPU/paid fit. R3's existing exposed64fit/32eval panel remains fixed.

## Frozen representations

Build causal histories on each complete20,000-row raw112 sequence before original masks/stride40. Use half-lives8/32/128, p_i=2**(-1/HL_i), zero filter states at sequence start, delta[0]=0 and delta[t]=raw[t]-raw[t-1]. Fitting stride rows ONLY determine raw112 mean/std in FP64; std floor1e-6. Current raw coordinates=(raw-mean)/std. Filter inputs=delta/std. Three impulse-unit-norm EMA states e_i[t]=p_i*e_i[t-1]+sqrt(1-p_i**2)*delta[t]/std. ArmA features are current112 plus e8/e32/e128 blocks of112, with intercept1:449coefficients per target.

Let C_ij=sqrt((1-p_i**2)*(1-p_j**2))/(1-p_i*p_j). This is the exact infinite impulse-response Gram of the three normalized exponentials, independent of data/labels. Cholesky C=L L.T, T=inv(L), so T C T.T=I. ArmB replaces the three EMA blocks by T@e channelwise; current112 and intercept are unchanged. With column feature vectors zB=S zA, S=diag(I112,T kron I112,1), the function map is betaA=S.T betaB. No separate B normalization, covariance whitening, changed pole, extra state or truncation. Common raw scaling preserves the shared normalized-delta input to all three filters. FP64 filtering/moments/linear algebra throughout; only official scorer predictions castFP32. This is an analytic CPU diagnostic, not strictFP32 campaign evidence.

## Same objective, controlled optimization

For each target y=clip(label,±2), w=abs(y), R_A(beta)=(sum(w*(X_A beta-y)**2)+100*||beta_nonbias||²)/sum(w). Intercept unpenalized. For B use the MAPPED regularizer100*betaB.T*S*R*S.T*betaB, not isotropic ridge in B coordinates. Thus H_B=S H_A S.T, b_B=S b_A and the underlying function objective is identical. Report congruence and mapped optimum checks; no model-selection alpha/CV/grid.

Both arms start beta=0 and perform exactly128 deterministic steepest-descent iterations using full-fit sufficient statistics. Per target gradient g=2*(H beta-b), Hessian2H, exact line step eta=(g.T g)/(g.T*(2H)*g). This removes an arbitrary global learning-rate choice; changing coordinates still changes gradient geometry, which IS the treatment. Record eta/gradientnorm/objectivegap/finite status, actual nonzero steps vs converged no-op iterations; never invent optimizer completions. No Torch optimizer, clipping, stochastic batches or epochs. Reference beta*=solve(H_A,b_A) is ONE exact optimal function for both arms, not a third independently fitted recipe.

Primary: per-target and mean relative objective gap at iteration128, gap=(betaA-beta*)T H_A (betaA-beta*) divided by beta*T H_A beta* (initial gap), with finite/zero-gap guards. Also fixed checkpoints0/1/2/4/8/16/32/64/128, spectral min/max/condition numbers H_A/H_B, and mapped optimum agreement. Prospective optimization spending gate: B/A terminal relative gap <=0.8 for BOTH targets (20%reduction), A gap positive; not significance, generalization or campaign admission. Exact line search is part of this fixed algorithm; no post-outcome iteration/pole/lambda/window retuning. Both arms128 attempted joint-target iterations; count actual nonzero target steps/noops separately. Moment algebra reuses rows, so do not claim128*N physical row exposures or equal walltime/FLOPs.

## Prediction and validity

After both fixed128 endpoints finish, score A128, mapped B128 and common optimal reference ONCE on all original59,436scoredrows/32evalcomponents using canonical global_wp_clip2_v1/FP32 prediction input. Construct common A features once per eval sequence; B predictions use mapped coefficients, so no change in supplied information. Preserve pooled/per-target/component/drop-one six-sums; drop-one is sensitivity, not32replications. Outer scores are SECONDARY, never used for steps/poles/lambda/selection. Faster fit optimization can hurt validation; no automatic promotion follows. No comparison to stored R3 absolute scores because the shared span now includes three poles.

Predeclared numerical gates: synthetic/direct-row objective congruence and history reset/causality/impulse-Gram tests; on actual fit64 fixed first rows, direct B features/native beta vs mapped A prediction maxabs<=1e-6, and mapped exact optima relative coefficient discrepancy<=1e-6. Symmetric H positive definite/finiteness required. Any failed invariant makes result INCOMPLETE, not a scientific negative; no runtime-cap expansion/restart/replacement seed. Keep all source/normalizer/coefficient payloads on B, transport only compact JSON/log metadata and producer archive SHA/stat.

## Execution boundary

One serial Beleriand unit, intended computer-vision-homework-eda19-r14-v1.service, source B/support-source/eda19-r14-v1, output B/research_tmp/nf19-r14-v1.0.5CPU/high1.4GB/max1.5GB/swap0/600s, hiddenCUDA/BLAS1/physicalB/research_tmp. Fresh scoped ownership/allpopulatedchildren/source/sample/outputabsence/disk/headroom gates before root admission. Parent2CPU/high18GB/max20GB/swap0,target17GBcommitted. Existing R1–R13/F3 untouched/no replay. Preparation/source review alone is not actual execution. Do not stage an admission before root review and exact transport checks.

Sample SHA b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64, membership e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217.64fitcomponents/31808stride40rows;32eval/59436rows; allF0 exposed/adaptive/cost-capped, notfresh/unbiasedfullfold. Same-span parameterization under one convex optimizer does not establish GRU conditioning, new information, public or CP30 gain.
