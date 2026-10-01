# Frozen clean-panel scorer

Run with one thread and a 2GiB scope in `wnn-campaign.slice`:

```bash
WNN_SLICE=wnn-campaign.slice WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 scripts/limited .venv/bin/python research/campaign_2026-09-30/audit/campaign_score.py --spec path/to/spec.json --output path/to/new_report.json
```

`campaign_score_freeze.json` records the tested source/dependency hashes. The
spec refers to this manifest by path and SHA256; the scorer validates every
frozen source before reading evaluation targets. Actors must not modify this
grader or its identities. Tests in `scorer_checks_v3.json` use only synthetic
arrays and temporary synthetic provenance. Existing grader files remain intact.

The spec contains `scorer_freeze`, `preregistration`, and `targets` identities,
each `{ "path": "repository/relative/path", "sha256": "..." }`. Targets are
the immutable `cache/valid_targets.npz`; its ZIP_STORED NPY members are mapped
directly, so the full target tensor is not decompressed/copied into RAM.

The prior preregistration JSON contains a `design` object. The post-fit spec
contains the identical object as `design`; changing any field fails. Freeze
design before model fitting/evaluation. A baseline report can contain only a
`reference` role and an empty comparison list. For an experimental screen, use
candidate/control/reference roles and fixed derived blends:

```json
{
  "metric_id": "global_wp_clip2_v1",
  "n_boot": 2000,
  "bootstrap_seed": 271828,
  "comparisons": [["candidate", "control"], ["candidate_blend", "control_blend"], ["candidate_blend", "reference"]],
  "derived_predictions": {
    "candidate_blend": {"reference": 0.8, "candidate": 0.2},
    "control_blend": {"reference": 0.8, "control": 0.2}
  },
  "folds": [
    {
      "fold": 0,
      "manifest": {"path": "research/campaign_2026-09-29/artifacts/splits/fold0.json", "sha256": "..."},
      "roles": {
        "reference": [
          {"run": "c20260929_ft_since_s3_f0", "epoch": 0, "weight": 0.5, "fit_source": "valid", "checkpoint": "artifacts/runs/c20260929_ft_since_s3_f0/model_ep0.pt", "expected_args": {"seed": 3}},
          {"run": "c20260929_ft_since_s4_f0", "epoch": 0, "weight": 0.5, "fit_source": "valid", "checkpoint": "artifacts/runs/c20260929_ft_since_s4_f0/model_ep0.pt", "expected_args": {"seed": 3}}
        ]
      }
    }
  ]
}
```

The example illustrates field names only; it omits actual candidate/control
roles and is not a runnable screen. Each member has fixed run, epoch, weight,
checkpoint path, `fit_source` (`train+valid` or `valid`), optional `job_name`,
`expected_args` for declared recipe invariants, and optional `source_sha256`
for required training-source identities recorded by the durable job. Controls
whose parent seed differs still have fine-tune seed3. FT checkpoint snapshots
must match prediction sidecars exactly; `model.pt` and `model_ep0.pt` are not
interchangeable identities.

The spec's `artifacts` is keyed by fold string, then run name. Each run has
`prediction`, `checkpoint`, `sidecar`, `provenance`, `job` identities. If its
recorded args use `init`, include `initial_checkpoint` identity; the scorer
matches the recorded init path and hash. An exported model trained in a job
with a different name must declare `job_name` before evaluation. Member arrays
must be complete float32 `[1873,20000,2]` validation exports with the target
cache's exact row order (1873 is read from the actual cache, not hard-coded).
The sidecar's run/split/epoch/checkpoint/prediction/split hashes and row groups
are all validated. Fit provenance must have fixed epoch/no evaluation selection,
matching split and allowed fitting groups. The job must be a successful bounded
GPU fit. Requested expected arguments and source hashes must match.

Mixtures are accumulated in float64 and rounded to float32 before clipping.
Every raw member and mixed prediction must be finite, including warmup; the
scorer rejects nonfinite values before either blending or clipping. Primary
rows are index>=99 AND is_scored. The target cache encodes the fixed 99-row
warmup format; no scoring-mask-dependent memory/prediction operation occurs.
Output includes pooled two-target WP, each fold, class diagnostics, softened
lambda0.02/0.05 diagnostics, eligible sequence means, paired component bootstrap
SD/95% interval and full provenance identities. Shared components across folds
are grouped in pooled resamples. The script never tunes weights or promotes a
candidate, and it never calls historical folds untouched.

Output creation is exclusive. Declare a new report path for every invocation.
The scorer intentionally refuses compressed target NPZ files or incomplete
evaluation-only prediction exports; create independently verified complete NPY
exports before changing the scorer contract if such support becomes necessary.

## Four-fold reference execution after controls finish

`reference_fourfold_design.json` freezes equal seed3/seed4 final FT snapshots on
all four folds. The design is reference-only. After every dependency finishes,
the root coordinator can bind the actual immutable files:

```bash
WNN_SLICE=wnn-campaign.slice WNN_THREADS=1 WNN_CPU_QUOTA=100% WNN_MEM_MAX=2G CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 scripts/limited .venv/bin/python research/campaign_2026-09-30/audit/build_reference_spec.py --design research/campaign_2026-09-30/audit/reference_fourfold_design.json --spec research/campaign_2026-09-30/audit/reference_fourfold_spec.json
.venv/bin/python scripts/campaign_job.py launch --name c20260930_reference_fourfold_score_v1 --kind cpu --phase independent-grade --timeout 900 --threads 1 -- env WNN_MEM_MAX=2G CUDA_VISIBLE_DEVICES='' scripts/limited .venv/bin/python research/campaign_2026-09-30/audit/campaign_score.py --spec research/campaign_2026-09-30/audit/reference_fourfold_spec.json --output research/campaign_2026-09-30/audit/reference_fourfold_report.json
```

The builder refuses missing final artifacts and verifies backward lineage:
final sidecar checkpoint snapshot, stage1 warmup checkpoint, exact parent
epoch3, fold propensity checkpoint, clean fitting memberships, and FT seed3.
Both FT stages belong to the final FT's one durable job; its command must bind
the parent@ep3, final run, fold and FTseed3. Distinct source identities are
retained. Completed fold0/1 lineage already passed a provenance-only rehearsal,
saved in `reference_f0_f1_provenance_check.json`, with no real prediction scoring.
The builder never alters historical snapshots or sidecars.
