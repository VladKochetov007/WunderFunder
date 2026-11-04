# Independent Reporting Adapter Review R1

Date: 2026-11-04  
Disposition: **no scoring blocker remains in the current grader; the dashboard needs one nonblocking exposure label clarification.** This was a read-only review of the reporting script, dashboard adapter, canonical scorer/loader and focused fixtures. No fits, GPU access, scoring, tests or source edits were performed by this reviewer.

## Current status

I rechecked the updated reporting files after the author addressed the two root-found grader blockers:

1. Grader now treats the project symlink root as the canonical data root, so datasets and cache/valid_index.npz resolve under the frozen project path.
2. Canonical reference checkpoints now use the exact sidecar checkpoint_file, restricted to model.pt or model_ep{epoch}.pt, with the frozen panel checkpoint SHA enforced. New cohort candidates remain bound to model_ep1.pt at fixed zero-based epoch 1.

No remaining issue changes numerical scoring or blocks export grading. The panel and numerics are not modified by these adapters.

## Numerical and artifact checks

- Row order: expected_row_order reads the actual valid_index.npz row_group list, requires 1,873 unique integer IDs, and canonical load_contract checks the set against all chronological evaluation groups. Canonical load_bank maps row_group ID to prediction-bank position; score_groups requests every evaluation group through this mapping. This is a real ID join, not an assumed positional match.
- Artifact identity: panel SHA is pinned to the Linux R3 bytes. Sidecars require run, valid split, fixed epoch, checkpoint filename, official split hash, FP32 inference, and exact row_groups. Canonical pinned_file recomputes actual SHA256 for prediction banks, checkpoints, metadata and provenance; load_bank then checks producer/provenance digests, fixed checkpoint args, and chronological fitting membership. Candidate prediction and checkpoint bytes therefore do not rely only on the sidecar's declared digest.
- Blend arithmetic: fixed_ensembles preserves TL3 weights 0.34/0.33/0.33 and creates exactly eight predeclared TL3 0.9 / candidate 0.1 blends. blend_predictions composes raw FP32 outputs without clipping. The canonical GlobalAccumulator applies official clipping and global weighted Pearson after blending. The scorer uses the official is_scored mask, removes the 99-row warmup, covers all 1,873 sequences and checks 3,528,273 selected rows.
- Paired reporting: matched_comparisons reports candidate and control member scores for each configuration at both seeds 1001/1002, plus the candidate fixed-blend delta versus TL3 and versus the same-seed control blend. Missing scores remain null and are labeled incomplete.
- Endpoint: the grader requires epoch index 1 from a two-epoch frozen panel and collects candidate model_ep1. The dashboard records this fixed endpoint separately from best-observed epoch; best-observed is explicitly provisional until additional valid records arrive.
- Exposed validation: canonical grade report text marks official valid as development-exposed, and train-eval metadata states it is not comparable to primary valid. The dashboard metadata and per-evaluation event identify “official full validation” but omit the exposed-development qualifier. Add an explicit exposure field such as “previously inspected development validation; not independent confirmation” to dashboard validation metadata/events before presenting them. This is a reporting-label issue, not a scoring or fit blocker.
- Matched-table missing values use an em dash in the current grader. The earlier mojibake observation is no longer present in the updated file.

## Paths reviewed

- scripts/w18_l4_quality_grade_20261104.py — SHA256 9d38f7946ca4879aaf791fcfb2c4a38196b68d7d89dc1a5a23536b5bc232c100
- scripts/w18_l4_quality_dashboard_20261104.py — SHA256 ded4647f143bc18bdae4e8e61aab27e61e28dcacafd411c0b3b5faf39ffc30cb
- scripts/heterogeneous_ensemble_contract.py — canonical load_bank and actual pinned-byte validation
- scripts/heterogeneous_ensemble_score.py — raw blend, full-valid row loop and fixed scorer
- src/wnn/gmetric.py — global_wp_clip2_v1 label/prediction clipping and pooled accumulator
- tests/test_w18_l4_quality_grade_20261104.py — fixture coverage for exact blends, row-order mismatch, pending artifacts, project root and reference checkpoint filenames

The author reports five focused fixture tests, Ruff and py_compile passed after the fixes; I did not rerun them. Root’s reporting-only patch should remain staged separately from frozen numerics. No candidate scores exist in these scripts until fit exports arrive.