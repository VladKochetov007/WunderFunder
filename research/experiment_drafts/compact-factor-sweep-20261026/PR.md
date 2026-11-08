# Compact Transformer: isolate dropout, decay and LR with live train/val curves

1. **Status/lineage/owner.** E-COMPACT-FACTORS-20261026 and E-COMPACT-LR-20261026;
   teacher continuation of PR43/42/32, ownerSemyonEpanov. User authorized eightB300;
   preparation/resource profiles are distinct from actual scientific-fit startup.
2. **Mechanism.** Separate dropout and decay after their joint package failed to
   improve WP; test whether lower/higher LR changes five-epoch quality/overfit.
3. **Compute advantage.** Four disjoint twoGPU fits per wave, two finite waves.
   Reuse exact completed controls instead of paying for redundant baseline fits.
4. **Frozen recipe.** Same source0a70b4ab,113.5M Transformer, batch128/rank/W2,
   fiveepochs/1150updates, Muon/AdamW,warmup5%+cosinefloor.1, BF16/BF16x9 with
   FP32 masters/states. Only declared dropout/WD/LR changes; no quantization.
5. **Panel.** Regularization screen: dropout.1only andWD.1only, eachF0/F2=fourfits.
   LRscreen:5e-5 and2e-4,eachF0/F2=fourfits. Total8newteacherfits,0students;
   two exact PR43controls reused, two historical joint fits plotted as context.
6. **Validation.** Every safe epoch evaluated immediately, identical fixed64train
   probe/full component-heldout284/287seqs. Perarm common epoch maximizing mean
   F0/F2WP;earliesttie. Full paired deltas,targets/curves/epoch5, no significanceclaim.
7. **Leakage.** Same causal/component/fitting-only normalization and clean initialization
   ancestry. Already exposed development folds; no pristine confirmation implied.
8. **Decision.** MeanWPgain>=.001 andpositivebothfolds motivatesconfirmation;
   smaller/mixed effects uncertain. LRscreensconfiguredoptimizer includingcoupleddecay;
   shortbudget does not settle asymptotic quality. No publicscore forecast.
9. **Run.** stage.prepare→bounded12updateprofiles→inspectedapproval→nativecron
   execute(twofourfitwaves). Same measuredbatch/GPUallocation; about2.5–3h plus
   startup.3h/wavetimeout,ownedprocesscleanup,mandatoryvalidationfailureguard.
10. **Evidence.** Exact configs/plans/profile/source/environment/weights/indices,
    train/val metrics,LR/gradients/clipping/timing/memory. Independently verified
    PNG/SVG/CSV/JSON provenance plots and auto-updated authorized GitHubcomment.
11. **Results.** Pending actual execution; all historical PR43 results retained.
    See packetREADME for scope, comparison/falsifiers, artifact paths and limitations.
