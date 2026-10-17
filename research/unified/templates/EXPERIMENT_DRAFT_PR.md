# [Experiment] <ID>: <mechanism>

1. **Status, lineage and owner.** Proposal/implementation/profile/fit/grade status; H/E IDs; teacher, student or connection; parent nodes; owner and source commit. State what is actually implemented.
2. **Why it should work.** Current deficiency, mechanism, predicted observation if true, strongest alternative explanation and discriminating ablation.
3. **Compute advantage.** Where additional B300 compute buys quality; scale/context/optimization ladder; downstream student or blend path; opportunity cost. No invented scaling law or LB estimate.
4. **Frozen comparison.** Candidate/control, common ancestry, full FP32 settings, data/splits/normalization, targets, optimizer, LR/scheduler, batch/accumulation, context, updates/epochs, clipping, checkpoint policy, reset and resume.
5. **Thirty paired runs.** Exact 8/8/7/7 fold-seed matrix per arm; source-qualified fold membership and clean teacher/initializer per cell. Distinguish resource profiles from scientific fits.
6. **Validation and significance.** Official metric and mask; paired component evaluation, fitting-seed variability, fold exposure, multiplicity and power limits. State primary statistic and interval scope.
7. **Leakage and coverage.** Future-label availability, past-only inputs, fitting-only transforms, component/duplicate boundaries, ancestry, representative sources/regimes, known limitations.
8. **Decision-changing result.** Predeclared useful effect and cost, success, uncertainty, negative-result interpretation and next action. Zero-crossing CI alone is no kill rule.
9. **How to run.** Exact working directory, dependencies, GPU count/topology, setup/environment, data/cache mounts, command(s), expected output paths, resume and failure handling. Explicitly label missing trainer/profile work.
10. **Evidence to return.** Config/source/env/provenance; every cell's losses/LR/preclip gradient norm/step counts/validation/time/memory; prediction/grade manifests; separate evidence and weights archives with SHA256 and storage location.
11. **Results and trace.** All cells and exclusions, unified note links, archive hashes, independent critique, PR/merge/source SHAs. Unrun means no measured gain; leaderboard impact unknown until observed, and never inferred by adding a local delta.
