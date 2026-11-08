# [Experiment] GALAM-L4: observable recurrent and branch comparison

1. **Status, lineage and owner.** H-GALAM-L4-20261028 / E-GALAM-L4-20261028 descend
   from H-GALAM-20261027. Root owns delivery. This implements local instrumentation
   and explicit variants of the source-traced schematic; no scientific fits,
   quality score, GPU profile or exact historical identity are claimed.
2. **Why it should work.** Recurrent memory may complement attention/ConvMixer
   on noisy causal inputs. GRU may improve the useful quality/throughput frontier,
   while the global branch may be redundant. Different parameter counts and3W
   versus4W decoders are competing explanations; the initial screen tests complete
   practical endpoints, not an isolated claim about gate counts.
3. **Compute advantage.** Use independent one-GPU fits before DDP for these small
   networks. Measure complete rows/hour and final evaluation cost. Memory is per
   L4 device and PCIe/NUMA communication can dominate small DDP jobs. No scaling
   law, speedup factor or public-score forecast is invented.
4. **Frozen comparison.** [PLAN.json](PLAN.json) and the [six configs](configs/README.md)
   declare FP32/AdamW, lr1e-4, wd0.01, warmup/cosine, hidden/context/chunk64,
   batch16,5epochs and fixed-final primary score. Preserve parent recipes; no KD
   teacher/initializer or valid-fitted preprocessing. Runtime profiles must match
   actual source/config/data/environment before admission.
5. **Bounded fitting panel.** Three arms x2matched seeds1001/1002 =6new fits,
   two per arm, zero reused controls and zero teacher/student fits. No Cartesian
   grid or30-run requirement. Engineering profiles are separately accounted.
   The panel is prepared, not admitted or automatically queued.
6. **Validation and uncertainty.** Use the mandatory official chronological
   train/valid boundary, full pooled global_wp_clip2_v1 and original score masks.
   Report target-specific scores, both paired seed deltas, conditional component
   sensitivity and exposure. Two seeds cannot establish reliable significance.
   Intermediate epochs and the fixed train-eval cohort are diagnostic; neither
   is a pristine test. Do not average per-batch/rank correlations.
7. **Leakage and coverage.** Whole source-qualified sequences/component exclusions,
   train-only normalizers/ancestry, past-only attention/convolutions and clean
   reset/TBPTT semantics remain mandatory. seq_ix is shuffled, not a timestamp.
   Report current timestamp evidence limits; no arbitrary extra temporal split.
8. **Decision-changing result.** The plan records illustrative prospective gates:
   mean paired WP gain>=0.0005, or <=0.0003 WP loss with>=1.25x measured throughput.
   These are spending/selection rules, not significance claims or guaranteed
   private utility. Keep mixed/partial/infrastructure-censored outcomes explicit.
9. **How to run.** Local dashboard and receiver commands are in README. Actual
   training is blocked by [live preflight](L4_PREFLIGHT.json): sparse partial train,
   full data disks, missing audited indexes and unaccepted host/source runtime.
   Existing unrelated GPU4 services remain untouched. No data install or fit is
   implied by this handoff; get exact data/headroom/runtime/profile first.
10. **Evidence to return.** Immutable config/source/runtime/membership; complete
    per-step logs and per-epoch train-eval/full-valid metrics; sampled diagnostics,
    timings and resource measurements; resumable checkpoints, predictions and
    official grades; separate evidence/weights archives with hashes. Live
    dashboard retention is bounded and must not be mistaken for the full archive.
11. **Results and trace.** CPU/API/browser checks and independent review are
    engineering evidence only. All scientific results remain unrun. Links from
    unified methods/reviews and this packet preserve source assumptions, the
    Predictorium chronology decision and the concrete receiving-host blockers.
