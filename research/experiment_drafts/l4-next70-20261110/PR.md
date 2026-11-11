# [Experiment] L4-NEXT70: shared and target-specialist GRU256x2

1. **Status, lineage and owner.** Approved preparation and fit admission. The
   canonical L4 sequence trainer is reused; the target-selective objective is
   implemented in core commit `a61902c3ca5ade5c4071e2b1a8e06aef7f4d9423`.
   ROOT froze the native source capsule at commit
   `9a5d2e7ca2549c24ff8217d5062df6de49cce9b5`, source identity
   `b05a73fc8a8a7a4f3f46962e31fd81ee6b401db715109e36d4f0d6e32bf35140`.
   ROOT owns native staging, fit admission, process groups, resource monitoring,
   grading and artifact collection. The six initial supervised fits are admitted;
   two SSL lanes remain authorized and wait for a separate core extension.

2. **Why it should work.** The shared GRU control trains both target heads under
   one weighted objective. The specialists retain the same two-output model but
   update the numerator for one selected target while retaining the original
   two-target denominator. This tests whether dedicated target training improves
   the specialist head enough to justify combining it with the other target's
   shared-control head. The strongest alternative is that both heads benefit from
   shared gradient information and specialization lowers their useful scale. The
   paired seeds and fixed output combinations distinguish these outcomes.

3. **Compute advantage.** Eight existing L4 GPUs are assigned to eight lanes.
   GPUs0/1 run the two shared controls, GPUs2/3 target0 specialists, GPUs4/5
   target1 specialists, and GPUs6/7 are reserved for the approved SSL mechanism.
   The first six fits run independently of SSL preparation. Shared controls are
   reused in both target-wise comparisons. The clock allows12h from the first
   actual profile, closes fit admission after11h, and reserves the final hour for
   full validation and collection. No new host, rental, dataset copy, or paid
   resource is part of this packet. Preserve the unrelated processes on GPU4.

4. **Frozen comparison.** All six initial configs use raw112 inputs, GRU256x2,
   FP32, batch32, chunk1000, accumulation1, AdamW LR3e-4, weight decay0.01,
   clip1, cosine schedule with5% warmup and minimum-LR ratio0.1, six fixed epochs,
   all-fit normalization over the10,325 official train sequences, and the
   all-after-warmup training mask. Diagnostics are recorded every20 updates and
   the observer samples every10 seconds. The0.1 minimum-LR floor is a dated
   amendment shared across all arms; it is not described as an exact replay of
   the historical floor-0 control. Every config requires one L4 rank, uses the
   standalone weighted-MSE objective, retains every epoch, and names epoch6 as
   its fixed endpoint. No teacher, reference, KD, hint, blend, or TL3 input is
   used. Raw config hashes and cell/GPU identities are frozen in
   [profile-plan-r1.json](native/profile-plan-r1.json); ROOT's native admission
   adds the normalized config hashes and approved profile-receipt hashes.

5. **Bounded fitting panel.** The initial8 worker lanes contain six supervised
   fits plus two SSL workers. The complete approved plan contains10 unique
   training stages: six6E supervised fits, two2E self-supervised pretraining
   stages, and two4E supervised fine-tunes initialized from their same-backbone
   pretraining checkpoints with fresh optimizer state. Exact SSL configs and
   parent-receipt rules are pending the scoped core extension; those lanes do not
   block the six executable configs. There are two shared controls (seeds203201
   and203202), two target0 specialists and two target1 specialists, using those
   same matched seeds. No historical checkpoint is resumed.

6. **Validation and uncertainty.** Primary evaluation uses the supplied
   chronological train/valid boundary, never random row or sequence folds.
   Each fit uses only10,325 earlier train sequences for fitting and normalization.
   Every epoch reports on all1,873 later validation sequences and all3,528,273
   selected rows. The official full-validation metric is repeatedly exposed
   development evidence, not a pristine holdout. The three predeclared complete
   outputs are `(s0.t0,c.t1)`, `(c.t0,s1.t1)`, and `(s0.t0,s1.t1)`, with scores
   computed as the mean of their two selected per-target official weighted-Pearson
   components: `0.5*(WP_t0(left model)+WP_t1(right model))`. No weights are learned
   from validation. Two seeds do not establish significance or public transfer.

7. **Leakage and coverage.** The native split SHA-256 is
   `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.
   Whole source-qualified sequences remain in the supplied official partitions.
   Sequence IDs are shuffled and are not used as temporal order. Normalizers use
   only fitting members; teachers, initializers, references, and learned blends
   are absent. Core provenance binds the two-output architecture and selected
   numerator. Native profile receipts bind actual source, config, GPU, data
   membership and reload state. Source input parquet footer/size/mtime and the
   source audit were qualified during ROOT's prepared-project receipt; the
   project uses the existing shared dataset through a link and copied only small
   cache metadata, not dataset payloads.

8. **Decision-changing result.** The comparison is useful if either specialist
   pairing improves the fixed shared-control score on its selected target across
   both matched seeds while preserving the other target through its declared
   control head. Report every per-seed score and both target components. Negative
   or conflicting changes remain evidence; they do not imply that one target's
   model is universally weaker. Follow-up allocation requires a separate review
   of full endpoint results and uncertainty. No public-score increase is inferred.

9. **How to run.** ROOT uses the existing L4 owned-command executor and observer
   with the pinned Python
   `/data2/lc-epanov_s/tabdata/.venv/bin/python`, source capsule
   `/data/lc-epanov_s/tabdata-next70-20261110-r1/source-specialist-r2`, prepared
   data root `/data/lc-epanov_s/tabdata-next70-20261110-r1/project`, and output
   root `/data/lc-epanov_s/tabdata-next70-20261110-r1/runs`. The exact trainer
   child argv for a profile is
   `python scripts/train_sequence.py --config <cell.json> --data-root <project> --split-manifest <project/chronological_primary.json> --output <runs/cell/profile> --profile-updates 30`.
   After a separate ROOT approval, the fit argv is the same without
   `--profile-updates 30`, adding
   `--profile-receipt <control/approved/cell.json>`. Each child sees only its
   assigned GPU UUID and uses8 CPU threads for BLAS/OpenMP. ROOT starts only
   approved cells; a missing approval does not block another cell. No automatic
   retry or resume is allowed. Keep outputs on `/data`, preserve at least5GB
   free there and3GB on `/data2`, cap aggregate output at6GB, owned RSS at128GiB,
   and require at least64GiB host available memory.

10. **Evidence to return.** For each cell preserve raw and normalized config
    hashes, source capsule identity, native approved profile and reload receipt,
    fit command/process birth identity, observer/resource samples, every-epoch
    train and full-valid metrics, LR and preclip gradient diagnostics, completed
    updates, model/checkpoint SHA-256, and terminal status. Fixed6E model hashes
    and both full-valid target metrics bind the three complete output scores.
    Keep plots and summaries alongside the run evidence and record the storage
    path and SHA-256 when ROOT archives them.

11. **Results and trace.** All six30-update L4 profiles completed and passed the
    strict model/optimizer/scheduler/host/CUDA-RNG reload gate. ROOT reported
    3.62–3.79 seconds per profile and peak reserved memory of1,801,453,568 bytes
    (about1.68GiB). These are engineering results, not quality evidence. ROOT
    has approved the six profile receipts for fresh fit admission. At this
    handoff, no fixed6E quality result is recorded in this packet. Independent
    critic review and terminal artifacts remain ROOT's active handoff items.
