# [Experiment] B300 bilinear i0×i1 inputs and entmax attention

Implements two architecture screens through the existing PR75 training lifecycle,
with MIDAS preparation/evidence handoff. The input study tests cross-stream products
against a parameter-matched additive branch. The attention study tests fixed
alpha1.5 entmax against softmax using the same explicit attention arithmetic.
No shared trainer, optimizer, scheduling, precision or batching changes.

1. **Status, lineage and owner.** H/E-B300-BILINEAR-20261112 and
   H/E-B300-ENTMAX-20261112; standalone Transformer/teacher-representation ideas.
   Preparation owner: Ivanhacker13 / Codex. Receiver owner, host and fresh budget
   are not assigned. Canonical runtime PR75 `8626000a4b965763a70011a0291a891aa6535d7e`;
   packet adapters derive from PR90 `34b2beaf165eb7a224b792c17b99e41b5fd7384a`.
   Eight prepared fits; no measured results or claim of GPU acceptance.
2. **Why it should work.** A linear input projection cannot directly express
   cross-instrument products. A rank32 residual W(U(i0)⊙V(i1)) may expose useful
   synchronous interactions early. The additive control W((U+V)/sqrt2) has the
   same trainables and zero-output initialization. A gain from generic extra
   parameters alone should appear in both arms. Entmax may exclude irrelevant
   historical keys that dense softmax retains; alpha1.5 is fixed. Both arms share
   explicit FP32 normalization arithmetic so backend differences do not define
   the comparison. Useful observations are full-E4 paired WP and target deltas;
   neither sparsity nor a lower loss alone establishes predictive benefit.
3. **Compute advantage.** B300 permits full-data four-epoch paired screens rather
   than extrapolation from tiny data. A promising teacher could later be distilled
   into the CPU envelope under a separate matched experiment. Dense entmax sorting
   can be expensive; no throughput gain or scaling law is claimed. These eight
   fits compete with further investigation of the existing stronger GRU baseline.
4. **Frozen comparison.** Two seeds206201/206202, same supplied split and fitting-only
   normalization. Raw112; H768/L24/12heads/FF4, context1024, chunk256, horizon256,
   batch128/rank, accumulation4, world2, epochs4. Muon/AdamW LR3e-4, WD.01,
   clip1, warmup5%, cosine floor.1, native role factors and optimizer grouping.
   BF16 autocast/BF16x9 with FP32 master/gradient/loss/state under the accepted
   precision amendment. Compile blocks on; dropout0; no activation checkpointing.
   Fixed final E4 is primary; retain all epochs. Native causal cache/reset,
   WMSE labels objective and resume semantics remain intact. No KD or quantization.
5. **Bounded fitting panel.** Each study has two new control fits and two new
   treatment fits: four per idea, eight total. Zero reused controls, zero separate
   teacher-cache/student fits. Profiles and receiver probes are engineering work.
   Each stage is timeout-bounded, owned and source-bound; no automatic dispatch,
   retry or approval. A 50-update full-size profile is mandatory for every cell.
6. **Validation and uncertainty.** Official `global_wp_clip2_v1`, 1,873 complete
   valid sequences and 3,528,273 scored rows; report WP,t0,t1 and each paired delta.
   This supplied valid has been used repeatedly and is not clean new holdout.
   Two seeds on one source-qualified split do not establish significance or robust
   generalization. No per-arm best-epoch selection or public-LB retuning. Independent
   source-qualified confirmation would need a separately frozen split/ancestry plan.
7. **Leakage and coverage.** i0[0:52], i1[52:104], a[104:112] are bound to canonical
   `FEATURE_COLUMNS`. Products use only the current normalized row; attention
   enforces past/current positions within the native window. No future/target
   feature, train-valid normalization, validation-label online update or teacher
   target is added. The exact inherited split preserves provenance but cannot
   prove economic-regime independence or eliminate prior validation exposure.
8. **Decision-changing result.** Mean paired WP≥+0.0005 with both seeds positive
   nominates an idea for confirmation, subject to target drops larger than.001
   and resource cost review. This is an allocation heuristic, not significance
   or submission approval. Mixed seeds/small gains retain uncertainty; two clearly
   negative deltas deprioritize this exact recipe, not the entire family. Preserve
   OOM, compile failures, incomplete endpoints and negative scores explicitly.
9. **How to run.** See `B300_COMMANDS.md` and `midas/README.md`. Use a clean pinned
   PR75 runtime and a separate new packet checkout, provisioned Linux/Python/CUDA
   environment, supplied data/split, two actually free B300s, persistent storage,
   receiver owner and fresh budget. Receiver small-shape checks do not establish
   full-shape fit. Human profile approval precedes fitting. Existing jobs are untouched.
10. **Evidence to return.** Source/config/split/data/runtime/owner identities;
    loss/LR/preclip gradients/update counts; profile/reload/memory/time receipts;
    terminal status, retained epoch index, final predictions and complete official
    scores. Preserve all source-bound logs; archive weights separately with hashes.
    MIDAS metadata export labels external checkpoint claims correctly and does not
    independently verify GPU execution or score recomputation.
11. **Results and trace.** No numerical result is available. Static source review
    is recorded separately; local tests, training, inference and profiling were
    not run. MIDAS returned `READY_LOCAL_HANDOFF` for preparation only. Publication
    does not mean receiver acceptance, a live job or a competition-ready ZIP.

Primary entmax source: [Peters et al., ACL2019](https://aclanthology.org/P19-1146/).
This motivates the normalizer; its NLP results do not predict competition gains.
