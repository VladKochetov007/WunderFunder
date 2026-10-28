# [Experiment] Source-separated compact Transformer to GRU distillation

1. **Status, lineage and owner.** Implementation handoff, not a result or a launch.
   H-SOURCE-HOLDOUT-TRANSFORMER-KD-20261028 /
   E-SOURCE-HOLDOUT-TRANSFORMER-KD-20261028-R1; connection/teacher/student experiment.
   Based on `research-team` at `7e4074314a62d0907c9608f9220cf72958708a2d`, including
   the merged #32/#42 canonical trainer. The preparation owner is Ivan's research
   packet; a receiving B300 worker must claim the actual cells before launching.
   Adds a full-source split producer/loader, six complete configurations, explicit
   profile/admission/fit/cache/evaluation commands and a paired results report.
   Local tests, model imports, training, inference and GPU profiles were not run.

2. **Why it should work.** #43 establishes that a compact causal Transformer can
   learn the task, but its component-held-out development protocol fitted some
   official valid-source rows. It does not answer whether its predictions help a
   fresh GRU when all official valid data are excluded from fitting. A same-seed
   teacher may suppress noisy label variation while transferring temporal
   structure. The decisive comparison is KD versus an otherwise identical
   labels-only GRU. Both use the same initialization seed, training membership,
   normalization, update schedule and loss scale. A normalized label/teacher WMSE
   removes the trivial doubled-gradient-scale explanation; teacher errors,
   capacity limits and training-target overfitting remain alternative explanations.
   Teacher predictions on fitting rows are in-sample; this is not out-of-fold KD.

3. **Compute advantage.** Two fresh Transformer fits buy new training-only
   teachers; four GRU fits measure whether their benefit survives compression.
   This answers a deployment-relevant question before funding another 1.54B
   ladder. More GPU memory permits the existing compiled causal context1024
   teacher; more compute does not establish a scaling law. The GRU is a future
   CPU candidate, with package size, latency and quantization deliberately left
   to a later measured submission phase. No serving or leaderboard gain is claimed.

4. **Frozen comparison.** Raw112 inputs, full eligible train-source data,
   all valid-source sequences evaluated, five fixed epochs, no early stopping or
   full-valid checkpoint selection. Whole graph components touching valid are
   excluded from train. Fit-only normalization, official scoring mask with first99
   rows excluded, causal inputs and sequence-state reset are shared. Teacher:
   Transformer768×16, 12heads, FF4, context1024, dropout0, efficient attention,
   compiled blocks, Muon+AdamW LR1e-4, batch128/rank. Students: GRU256×2,
   AdamW LR1e-3, batch32/rank. All use W2, chunk512, accumulation8, warmup5%,
   cosine floor0.1, weight decay0.01, clip1. Explicit BF16/BF16x9 follows the
   user-approved precision amendment; weights/gradients/loss/optimizer state are
   FP32, and this is not strict IEEE-FP32 compute. Labels-only has KD weight0;
   KD has weight1 and `(WMSE_label + WMSE_teacher)/2`. Optional normalization
   preserves legacy additive behavior/config hashes when omitted. Only the fixed
   final epoch5 becomes the scientific endpoint. Retention is `final_only`, with
   resumable safe state. Exact settings are in the six committed JSON configs.
   Teacher versus GRU compares useful recipes, not isolated architecture at equal
   compute. Batches are candidates until newly profiled on the receiver.

5. **Bounded fitting panel.** Six TOTAL new scientific fits: two teachers, two
   labels-only GRUs and two KD GRUs at seeds1001/1002. This is one split and two
   seeds, not two folds. Zero reused controls. #43 checkpoints have invalid
   ancestry for this protocol; #46's event211 GRU, optimizer and adaptive horizon
   are not an exact control. Their published metrics are context only. Each cell
   first requires a disposable 12-update engineering profile and explicit review;
   profiles are not scientific replications or permission to change batch/LR.
   No automatic queue, retries, extra seeds or scale sweep.

6. **Validation and uncertainty.** Fixed final endpoints are evaluated once on
   all valid-source sequences with canonical pooled `global_wp_clip2_v1`, including
   t0/t1, selected-row and component counts. Report all six endpoints and each
   seed's teacher-minus-labels and KD-minus-labels differences. Equal-weight seed
   means summarize the panel; two seeds cannot establish significance, independent
   fold robustness or training-variance coverage. Official valid has already been
   exposed to team development. It is source-separated from these new fits, not
   pristine confirmation. No comparison against #43's differently selected
   inner-valid scores is interpreted as a gain/loss. Source files do not prove
   chronological ordering; `seq_ix` is an identifier, not a verified timestamp.

7. **Leakage and coverage.** A new feature-only graph and thin source-qualified
   indexes cover every physical row group, including train528. Legacy graph caches
   silently omit that group and are not reused. Missing/corrupt data fail preparation;
   there is no automatic exclusion. Every train component linked to any valid
   component is purged; no valid rows enter loss, normalizer, initialization or KD
   cache. Valid features are used only for graph-based overlap exclusion, which
   is a transductive preprocessing choice. Approximate feature-derived components
   are not proof against every unknown duplicate or temporal linkage. Exact
   data/index/graph/split identities are recorded. Teacher caches cover only fit
   members and must match the designated same-seed fresh teacher, source, config
   and terminal epoch5 checkpoint. No old checkpoint, ensemble or foreign cache
   substitution is accepted. Receiver causal-prefix and data checks remain pending.

8. **Decision-changing result.** Predeclared follow-up signal: mean KD-minus-labels
   WP at least +0.001 and nonnegative on both seeds, with no invalid/nonfinite
   endpoints. This justifies a separately allocated robustness/serving study, not
   a submission or significance claim. Mixed signs, target disagreement or a smaller
   positive gain are uncertain and require inspecting all cells and cost. Negative
   results narrow this five-epoch single-teacher recipe, not all Transformer/KD
   directions. A weak standalone teacher suggests improving teacher learning before
   scaling distillation. No confidence-interval sign alone closes the direction.

9. **How to run.** See the packet [README](README.md), `receiver.env.example` and
   `run.sh`. Use a clean checkout at a full pinned commit in an already provisioned
   Linux B300 runtime. Prepare complete source caches/split first. Choose exactly
   two free GPUs, profile one explicit cell, review its receipt, admit, then fit.
   Export each fixed-final teacher's fit-only cache before profiling that seed's
   KD student. Evaluate fixed endpoints after fitting; compare only when all six
   complete. Fresh roots and exact identity checks protect existing jobs/artifacts.
   All actual environment/data/numerical acceptance and profiles are pending.

10. **Evidence to return.** Exact Git/source/config/environment/data/split hashes,
    prepared-source membership/exclusions, six original/reviewed profiles, run
    provenance, losses, LR, preclip gradient norms, attempted/successful updates,
    nonfinite events, elapsed time and memory, terminal statuses, final checkpoint
    hashes, teacher cache manifests, prediction manifests, canonical grades and
    the complete paired comparison. Return failures and exclusions too. Archive
    compact receipts/logs separately from weights/cache arrays, with hashes and
    storage locations; do not upload private data or model payloads to Git.

11. **Results and trace.** Six fits remain unrun. No quality, throughput, hardware
    fit or CPU-submission result exists for this packet. See [static validation](VALIDATION.json),
    [independent review](REVIEW.md), [unified ideas](../../unified/ideas/README.md),
    and append-only H/E ledger entries. Publication and merge are distinct from
    source-bound run evidence. The draft PR URL and published commit are recorded
    on publication; receiver results must reference the actual training commit.
