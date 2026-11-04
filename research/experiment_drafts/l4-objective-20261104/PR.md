# E-L4-OBJECTIVE-20261104: relax clipping and test late objective interference

1. **Status and owner.** User-authorized follow-up, SemyonEpanov; preparation, not a measured gain. Maximum six new fits: four teachers and at most two students. Current PR61 execution remains unchanged.

2. **Mechanism.** Clip1 affects virtually all scratch GRU updates. Main, structural and next-change objectives have different reductions and can interact through the shared recurrent trunk. Test a larger clip threshold and one diagnostic-selected optimization change.

3. **Compute advantage.** Use the currently free existing L4 lanes, one fit per GPU. Reuse exact controls, existing data and baseline artifacts. Historical teacher cost is about34minutes plus startup; current actual profiles must qualify the estimate. No new rental or invented gain.

4. **Frozen teacher comparison.** GRU256x2, native211/23 outputs, seeds1001/1002, four epochs/fixed zero-basedep3. Arm1 changes only clip1→5. Arm2 is selected by the frozen fitting-only diagnostic rule: late active-auxiliary anneal or half peak LR. Constant-head layout, initializer RNG and strictFP32 are preserved.

5. **Student transfer.** At most two W18 GRU256x2 students use the predeclared selected family and unchanged eligible FBCep3/base/family cache, alpha.7/family.1, AdamW1e-4, B16/C2000, two epochs/fixedep1. Direct streaming builds the new target without persistent full prediction banks.

6. **Validation.** Exact official train10325/0valid fitting ancestry, full official valid1873/3528273 scored rows, unchanged pooled clipped weighted Pearson. Same matched seeds; no fabricated temporal folds. Validation is exposed development.

7. **Leakage.** Normalizers, ancestors and features remain fitting-only and causal. The diagnostic reads fitting sequences, computes gradients without an optimizer, and verifies model state unchanged. Original timestamps beyond the supplied train/valid boundary are not inferred from sequence IDs.

8. **Decision.** Report every fixed-epoch paired delta. The selected teacher family must meet the frozen +.001 mean-WP spending criterion for optional students. Useful fixed-blend improvement is +.0005; this is an allocation criterion, not a promise/significance statement. Negative arms remain recorded.

9. **Execution.** Existing8×L4; new teachers onGPU2–5, original jobs retained. Three-hour receiving authority includes45minutes collection; minimum3GB physical free space,2.5GB new-output cap, reserve external outputs. The newly derived target is transient under the declared retention policy; older caches are untouched.

10. **Evidence.** Source/data/split/parents, exact argv, finite profiles, all checkpoints, train/valid plots, per-loss diagnostics, resource telemetry, canonical prediction grades and GitHub updates. Actor/critic checks precede fitting; actual outcomes cannot be inferred from code readiness.

11. **Results.** Pending actual execution. [Contract](CONTRACT_R1.json) and [packet](README.md) hold the proposed rules; no scientific fit or public improvement is claimed here.
