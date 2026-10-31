# Full-corpus TL3 + LSTM submission preparation

The user requests fitting the strongest supported solution on all supplied
labeled data for submission. [ADR0045](../../../specs/adr/0045-submission-only-all-labeled-final-fit.md)
scopes this explicitly to final fitting; primary research validation remains
unchanged. No full fit or platform submission has started at this packet's
initial preparation state.

1. **Owner and lineage.** Root; new namespace `tl3final_` plus
   `final-lstm192x1`. Preserve the completed train-only PR49/52/53 evidence and
   source/weights. New final weights must never replace those controls.
2. **Mechanism.** Reuse the fixed model/optimization recipes with more supplied
   labels. There is no guaranteed quality gain from adding the later data.
   The prior best tested fixed mixture is90%TL3 +10%LSTM WP0.692502063 versus
   TL30.692377425 on development-exposed valid; this is not a final-fit score.
3. **Compute.** Canonical TL3 has3 inference members but17 training dependencies
   (54 configured epochs,312,000 updates on12,480 sequences). One additional
   fixed8E LSTM brings the final-fit inventory to18, with seven TL3 roots and
   LSTM initially ready for eight B300 lanes. No new search grid. Measure actual
   profiles/critical path before freezing the full execution deadline.
4. **Frozen recipe.** Same TL3 GRU256x2 W18/tiedlead graph, objectives, epochs,
   auxiliary heads, optimizers, schedules, neutral propensity and FP32 compute.
   LSTM192x1 retains its exact Source4 recipe. Nested FP64-to-FP32 blending is
   preserved at both TL3 and outer-mixture levels; no premature clipping.
5. **Budget and scope.** Exactly17 TL3 fits plus one LSTM; no automatic extra
   seeds, fine-tuning or new teachers. Actual source/plan/argv identities and
   profile receipts must be independently reviewed before costly fitting.
   ROOT selects bounded wall/child limits from measured profiles. Reuse the one
   SSH connection42104; no foreign process signals, new rentals or billing.
6. **Evaluation.** All10,607 train and1,873 valid groups fit; independent eval0.
   Keep complete sequences and fixed final epochs. No validation selection,
   learned primary weights or claim that same valid remains independent.
   Final-fit plots show training diagnostics; old train/valid curves stay separate.
7. **Data and ancestry.** [Fresh receiver proof](receiving/SOURCE_VALIDATION_R1.json)
   hashes both entire Parquets and verifies every20,000-row group. Group528
   decodes all117 columns with no nonfinite values on this source. Final schema
   names exact row-group ordinals; no inferred sequence-ID chronology. All new
   normalizers, teachers, initializers and caches bind final-mode source/member
   identities; historical train-only descendants cannot be relabeled final.
8. **Decision.** Deliver a complete deterministic package that passes size,
   offline ABI, reset/causality and representative one-CPU timing gates. Do not
   promote an incomplete teacher graph or fabricate a full-data valid score.
9. **Implementation.** `final_submit_recipe_builder.py` generates the17-node
   plan; the explicit final-fit trainer/export/cache branch leaves ordinary
   chronological guards intact. `train_final_lstm.py` reuses the native sequence
   runtime with complete fitting membership and no evaluation. The native
   callback adds raw112/event211/tiedlead263 routing and tiedlead nonlinear
   state; the old native ensemble cannot serve this candidate unchanged.
10. **Evidence.** Retain source/config/plan hashes, every actual launch/profile/
    exit/checkpoint/export, loss/LR/gradient/resource traces, package hashes,
    independent critique and exact GitHub download verification. Tool/bootstrap
    failures are engineering evidence, not negative quality results.
11. **Current state.** [Draft PR54](https://github.com/VladKochetov007/WunderFunder/pull/54)
    is published as SemyonEpanov. Both actual R2 profiles pass the
    [independent checker](INDEPENDENT_PROFILE_REVIEW_R2.json): native LSTM50
    optimizer updates and legacy GRU60 updates (five measured whole-sequence
    batches plus one warmup). The GRU profile measures9.06sequences/second,
    approximately92minutes for its fixed four epochs, not a timing of every
    specialized descendant. Full scientific fitting has not started.
    Linux builds for v3/AVX2/v2 and the focused residual-scratch Rust regression
    pass. Short synthetic CPU parity, reset and future-prefix checks pass.
    However, the current residual F32 inference path measures858microseconds
    per callback on one pinned Xeon6767P core: an extrapolated9.4hours for39.4M
    rows, which fails the official60-minute budget. This is an engineering
    blocker, not a delivered or eligible ZIP. A faster residual kernel, actual
    full-sequence parity, Python3.11 offline package checks and final weights
    remain pending. Preserve failed pre-CUDA R1 and original panic evidence.

The plan keeps the17-TL3 subtotal54epochs/312,000updates separately from explicit
aggregate totals18fits/62epochs/319,840updates. Root's detached execution bound
is12node-hours from the first profile, maximum6hours per scientific fit,
600seconds per engineering profile and30minutes reserved for collection.
Each actual fit still needs its exact-recipe engineering admission. These bounds
do not declare a fit running or promise that the complete DAG will finish.

Official sources checked31October2026: [FAQ](https://wundernn.io/connectome/docs/faq)
permits supplied train/valid labels; [submission guide](https://wundernn.io/connectome/docs/submission_guide)
specifies Python3.11, one CPU,16GB RAM,60minutes,20MB ZIP, no GPU/network.
The selected references do not override our stricter research validation policy.
