# [Experiment] B300 causal Transformer and distillation runtime

1. **Status, lineage and owner.** Implements a CUDA/torchrun teacher → fitting-only
   cache → GRU/Transformer student → official score path, on research-team
   `4e0626c8`. E/H-B300-SEQUENCE-RUNTIME-20261020 connects PR28 transfer, PR29
   representation hints and PR30 streaming students. Implementation and static
   review only: no local tests, model execution, fits or B300 profiles.
2. **Why it should work.** A causal attention teacher can expose dependencies
   absent from a compact recurrent student. Compare label-only with same-model
   KD; compare aligned versus component-permuted hints to separate correspondence
   from auxiliary regularization. Teacher quality and student benefit are separate
   observations; neither is assumed.
3. **Compute advantage.** Large teachers use DDP over the allocated eight B300s;
   students start with one GPU per fit. Dimension ladder is 768×24, 1280×32 and
   1792×40; real parameter counts, FP32 memory and throughput are recorded by the
   runtime. No advertised hardware throughput or scaling law is treated as measured.
4. **Frozen comparison.** Explicit amendment: raw112, fitting-only normalization,
   sliding per-layer KV, fixed chunk256/accumulate4, full sequences, strictFP32,
   AdamW/cosine, fixed final three epochs. This is not the old anchored4096 or
   anchored211/Muon recipe. Label/KD are additive weighted squared errors;
   teacher labels/outputs clip at2, weight is absolute clipped truth. Hints
   use independent masked normalization. Full details in README/configs.
5. **Bounded fitting panel.** F0/seed1001 and F2/seed1002. First panel costs six
   fits including two teachers. Hints costs four new fits with exact transfer
   control/cache reuse; streaming costs four with the same prerequisites.
   These are separate panels: fourteen total if all three are admitted. Optional
   width and scale follow-ups each cost four new fits and are not automatic.
   Historical CP30 recipes remain unchanged.
6. **Validation and uncertainty.** Canonical pooled global_wp_clip2_v1 on exact
   held-out component membership. Report both targets and every paired delta.
   Two exposed cells confound fold and fitting-seed effects; no significance
   claim or automatic 30-run expansion. Outer validation never selects checkpoints.
7. **Leakage and coverage.** Full raw causal rows, first99 warm-up, reset per
   sequence, component-disjoint canonical manifests. Teacher cache validates the
   entire same-fold ancestry and source-qualified membership; output exports bind
   every file and checkpoint hash. Unsupported legacy native211 weights fail
   explicitly. Existing fold exposure and unknown temporal regimes remain limitations.
8. **Decision-changing result.** A descriptive mean student gain ≥0.001 with both
   cells nonnegative is a useful follow-up signal, conditional on compute/serving
   cost; inconsistent or smaller effects motivate targeted diagnosis. A negative
   screen narrows this exact recipe. Interval crossing zero alone does not reject
   the family. Teacher scale alone does not establish transfer.
9. **How to run.** README provides receiving-host dependencies, data/split binding,
   real-shape profile, approved-receipt admission, fit, cache, evaluation and score
   commands. Safe resume is epoch-boundary only with exact world size and identity.
   No queue manager, GPU rental or job launch is included. B300 acceptance is pending.
10. **Evidence to return.** Source/config/data/cache identity, hardware/runtime,
    rank logs, losses, LR, preclip norm, attempted/successful updates, elapsed time/
    peak memory, final/resume checkpoints, predictions and official grades.
    Separate hash-bound weights/evidence archives; retain failures. CPU submission
    export/size/runtime acceptance remains separate.
11. **Results and trace.** No empirical results. Unified ideas index links this
    packet; the existing PRs receive explicit implementation amendments. Remote
    unit/DDP/real-data/resume acceptance must precede a ready-to-run claim.
