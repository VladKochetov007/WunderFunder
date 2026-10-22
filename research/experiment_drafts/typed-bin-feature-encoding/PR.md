# [Experiment] Train-fitted numerical bases for a causal Transformer

1. **Status, lineage and owner.** `H-TYPED-BIN-ENCODING-20261022` /
   `E-TYPED-BIN-ENCODING-20261022-R1`; teacher representation, proposal only.
   Parents: PR #8 input encoding and PR #31 numerical-basis ladder. Receiving
   researcher owns integration and execution; no compute is admitted here.
   Base `c452e8cf4f8ef57e5c07c6801a6e10fc7616251d`; trainer dependency is
   PR #32 at `d7559e94008c0d658898b59290785ae457b1b7bb`, not yet on this base.

2. **Why it should work.** A linear scalar projection may spend capacity learning
   uneven feature resolution. Fixed quantile knots allocate resolution where
   observations occur; uniform knots are the matched alternative. Keep the raw
   scalar to preserve amplitude and outliers. Test whether the quantile arm beats
   the identical uniform arm, not whether a larger encoder beats a smaller one.
   [Numerical embeddings](https://arxiv.org/abs/2203.05556) motivate this basis;
   [TradeFM](https://arxiv.org/abs/2602.23784v1) motivates testing bin placement.
   Neither paper reports a Connectome improvement. Do not recreate TradeFM's
   physical price ratios, event semantics or composite trade vocabulary from
   anonymized fields.

3. **Compute advantage.** Additional B300 time buys three controlled full-membership
   fits per cell and later a larger teacher only if the representation is useful.
   One B300 per fit initially, subject to actual FP32 memory profiling. Keep the
   same context, updates and batch across arms. No throughput or scaling-law
   claim; distillation is a separately budgeted follow-up, not part of these six fits.

4. **Frozen comparison.** Arms: `raw`, `uniform16`, `quantile16`. All retain the
   same normalized raw-112 projection and canonical 99 event features. The two
   basis arms add the same per-field 16-component piecewise-linear branch; only
   knots differ. Knots and raw normalizers use fitting components only. For
   quantiles use probabilities 0,1/16,...,1; for uniform use 17 equally spaced
   knots between the same fitting min/max. For basis element k use
   `clip((x-edge[k])/(edge[k+1]-edge[k]),0,1)`; zero-width intervals output zero,
   including constant fields. No division is evaluated for a zero-width interval.
   Preserve raw scalar and shared finite/missing indicator; never sort columns or
   merge their identities. Record knot support and out-of-range rates.
   Binned arms have identical trainable shape/count; comparison with `raw` is a
   predictive-value contrast with extra capacity, not a pure mechanism ablation.
   Starting temporal model: 4 layers, width256, 4 heads, FFN1024, causal window256,
   one row/token, shared positional treatment, dropout0.05, direct two-output head.
   Common recipe: FP32 throughout, autocast/TF32 off; AdamW LR2e-4, decay0.01,
   preclip norm1, 12 epochs, 2% linear warmup, 20% linear cooldown to0.1×LR.
   Global batch32 complete-sequence streams, chunk256, common detach/carry rules.
   Final epoch is the endpoint; outer scores never select epochs or knots.
   Receiver must seal exact loss/target normalization, positional mode, event
   mapping, successful-update count and context/cache semantics before admission;
   these are explicit blockers, not choices to tune after seeing outcomes.

5. **Bounded fitting panel.** F0/seed1001 and F2/seed1002, all eligible fitting
   components and full fold evaluation. Three arms × two cells = **six new
   teacher fits**, two per arm; zero student fits and zero reused controls.
   Random initialization, no pretrained weights. Cells are prospective and not
   resource reservations. Failure/retry consumes recorded execution cost and
   does not silently expand the scientific panel.

6. **Validation and uncertainty.** Primary contrast `quantile16 - uniform16` in
   official `global_wp_clip2_v1`, with unchanged scored mask and clipping.
   Report each cell, equal-cell mean, both target scores and secondary contrasts
   against raw. Paired component bootstrap:10,000 draws, seed221022, common
   multiplicities. Intervals condition on fixed fitted predictions; two cells
   confound seed/fold and establish neither significance nor fresh validation.
   All current folds are exposed. Audit source/age/activity and out-of-range
   strata as diagnostics without selecting a winning subgroup.

7. **Leakage and coverage.** Fit knots/scales only on the cell's fitting rows;
   never use outer labels, masks or entire evaluation sequences to fit a transform.
   Current-row scalar encoding is stateless, while event features and temporal
   state must advance on every row and reset by sequence. Preserve warmup0–98;
   no labels/`is_scored`/sequence ID as inputs. Check prefix and full/chunk/row
   parity on the receiving host. Stable field IDs are not physical depth order.

8. **Decision-changing result.** A primary mean gain≥0.0003 with consistent signs
   and no major operational penalty supports a limited follow-up; this is an
   allocation threshold, not a forecast or significance test. Also require useful
   quality versus raw before promoting this encoder into the teacher programme.
   Mixed signs, a wide interval or different effects by field family remain
   uncertainty; uniformly inferior matched results narrow this exact basis.
   If both binned arms beat raw equally, retain basis expansion as a possibility
   without claiming quantile placement caused the gain.

9. **How to run.** There is no runnable fit command in this documentation PR.
   Integrate with `scripts/train_sequence.py` and the canonical sequence model
   factory from the pinned PR #32. Implement fold-only knot export/load, exact
   encoder/config/checkpoint schema and row inference; seal concrete commands
   and environment in a receiver amendment. No copied trainer, new queue manager,
   local tests, model imports, fits, rentals or submissions in this task.

10. **Evidence to return.** Source/environment/data/component hashes, knots,
    normalizers, initialization and all arm configs; attempted/successful steps,
    loss/LR/preclip norm, memory/time, final checkpoints, full predictions and
    official grades for every cell. Return separate evidence and weight archives.
    Large-teacher quality alone is not CPU submission readiness; a later student
    or native encoder needs exact-ZIP Python3.11 offline/one-core acceptance.

11. **Results and trace.** No measured gain, GPU profile, ZIP or public forecast.
    [recipe.json](recipe.json), [README.md](README.md) and the linked source triage
    form the handoff. Independent static criticism will be retained in REVIEW.md;
    it cannot substitute for empirical acceptance. Preserve failed/negative cells
    and publish the actual PR link after creation.
