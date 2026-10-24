# Resume R1 feasibility scout

- [F1 mean-only follow-up](f1_mean_shrink_r1/README.md) — the fixed additional 0.1 correction is essentially flat versus its parent, −0.000004696 WP, after raw A loses 0.001567413. The independently reviewed metadata repair and full preflight permit a 15.036-second bounded grade; all target/prediction hashes remain stable. This exposed second fold/seed does not reproduce F0's point gain or complete the original panel.

This folder records the saved-artifact feasibility comparison and the
completed exploratory F0-only grade of six saved online heads. Root admitted
and ran this new seven-endpoint grade; the historical 12-fit panel and its
grader remain unchanged and incomplete.

The source-only [control interpretation note](CONTROL_INTERPRETATION_R1.md)
clarifies that A is a slope-zero, mean-only relation estimator with continuing
causal state updates and the full 419-wide raw/event/statistics input. It
distinguishes the actual A/B/C arms from the untouched parent and a hypothetical
419-wide zero-stat head, which has not been evaluated. No arrays, labels, or
model imports were used for that interpretation.

The comparison and exact artifact paths are documented in
[REPORT.md](REPORT.md). The [R2 recipe](F0_EXPLORATORY_RECIPE_R2.json)
freezes six existing F0 heads, their parent, the official metric, primary and
secondary contrasts, resource limits, exposure and recursive ancestry pins.
It adds pre-score and post-score target digest checks; R1 is preserved as the
reviewed earlier draft. R2 was independently reviewed, admitted, and executed;
the original 12-fit recipe and grader were not changed.

The separate [label-free preflight adapter](grade_online_f0.py) uses
[f0_grade_preflight.py](f0_grade_preflight.py),
[f0_grade_ancestry.py](f0_grade_ancestry.py), and
[f0_grade_scoring.py](f0_grade_scoring.py). It streams targets to physical
scratch-backed memory maps only after root admission, checks the target file
hash before its first label read and again after scoring, and reports per-contrast time-bin deltas and
conditional component-bootstrap intervals. The source manifest
([SOURCE_MANIFEST_R2.json](SOURCE_MANIFEST_R2.json)) records all 42 historical
runtime/grader pins, the four adapter modules, and the synthetic-check receipt.
The exact pre-postcheck R1 scorer and its synthetic test are retained in the
[R1 source snapshot](r1_source_snapshot/README.md), matching both R1 source
hashes.

The bounded preflight passed with no label reads. Its create-only receipt is
`/home/vlad/development/wunder/research_tmp/resume4h-r1/f0-online-preflight-r3.json`
(SHA-256 `f8fd01dab2226fdc46660e72a1cc25d1da3457429f404d0d0f036e18f01d7692`).
It confirms seven full prediction payloads, source-qualified coverage for 468
evaluation groups and a recursive ten-run ancestry with zero valid-fit/eval
overlap. Four [synthetic invariant checks](test_f0_grade_synthetic.py) pass
under the 400 MB, one-core cap; Ruff format and check pass. The admitted R2
grade completed in 32.71 seconds with a 399,998,976-byte peak, zero OOM events,
and zero swap. See the [grade output](F0_EXPLORATORY_GRADE_R2.json),
[admission](F0_GRADE_ADMISSION_R2.json), and
[execution receipt](F0_GRADE_EXECUTION_R2.json). No fit or profile was run.
Root owns result interpretation and shared-index updates.

The next no-target clock EDA is frozen separately in
[clock_eda/README.md](clock_eda/README.md). It uses the same 96 valid row groups
and notebook field/change definitions; the script and synthetic checks are
ready at that preparation snapshot; the [completed findings](clock_eda/FINDINGS_R1.md) now record both admitted target-free scans.

The root controller owns the resume frontier and unified research indexes.
The preflight and proposed recipe are described here so the folder can be
indexed by root without implying scientific admission or a quality result.

The separate [fixed correction-shrink R3 packet](correction_shrink_r1/README.md)
asks whether an additional fixed 0.1 scale on the five unique saved F0
corrections improves their endpoints relative to the parent and raw predictions.
Its [recipe](correction_shrink_r1/CORRECTION_SHRINK_RECIPE_R3.json),
[62-file source manifest](correction_shrink_r1/CORRECTION_SHRINK_SOURCE_MANIFEST_R3.json),
and [synthetic validation receipt](correction_shrink_r1/CORRECTION_SHRINK_SYNTHETIC_R3.json)
were frozen and admitted for one exposed F0 grade. The [result](correction_shrink_r1/CORRECTION_SHRINK_GRADE_R3.json)
reproduces all raw endpoints exactly; its separate [independent result review](correction_shrink_r1/RESULT_REVIEW_R3.md)
finds that the fixed 0.1 shrink gives positive point changes over parent but
all five conditional 95% intervals cross zero. It does not establish
relationship/mean-only equivalence or an optimal coefficient. The original
12-fit panel remains incomplete.

The separate [F0/F1 batch-mass audit](batch_mass_audit_r1/README.md) freezes
the original five-column in-sample fitting suffixes and producer/membership
pins for both folds. Its metadata/header and synthetic checks passed without
reading suffix values. The separately admitted full stream now completes;
[measured findings](batch_mass_audit_r1/RESULTS_R1.md) show material batch
reweighting and reduced component concentration without claiming model-quality
harm. The [independent result review](batch_mass_audit_r1/RESULT_REVIEW_R1.md)
recomputes the component and loss aggregates from JSON and cross-checks the
verified suffix identities against their transport pins.

The [completed fixed direction probe](direction_probe_r1/RESULTS_R2.md) separates ordinary calibration from the common A correction on saved F0/F1 endpoints. The independent [result review](direction_probe_r1/RESULT_REVIEW_R2.md) confirms moment-derived point derivatives and projection ranks; its numerical bootstrap interval reproduction remains incomplete. B/C residual slope intervals in the actor result span zero, and F1's fixed endpoint is essentially flat. The [packet](direction_probe_r1/README.md) preserves the failed R1 startup, repaired actual-producer fixture, bounded R2 run, moment artifact, and review limits. No new fits or serving coefficients.
