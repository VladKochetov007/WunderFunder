# Submission audit — October 5, 2026

## Decision

There is no new public best beyond the already-scored seed-4 combo at **0.6660**. It reached the campaign’s former 0.6660 target at the portal’s displayed precision. The new activation experiments do not produce a deployable package; E488 is graded, but it is a fixed-panel mechanism study rather than a submission candidate. No submission is recommended from the q8 repair, seed bag, activation pilots, or W384 work.

The public rows below come from the saved user-pasted portal record, not a live portal query. I recomputed the listed archive SHA-256 values and ran `unzip -tq` over these local ZIPs in a one-CPU, 2 GiB limited scope. All eight archives tested cleanly. This verifies ZIP integrity and identity; the separate runtime, parity, and portal evidence is described below.

## Publicly scored submission ranking

| Rank | Submission | Portal WP | Portal runtime / memory | Local ZIP and SHA-256 | Evidence |
| ---: | --- | ---: | --- | --- | --- |
| 1 | Seed-4 equal parent/head combo, `2e5j6bnt` | **0.6660** | 37m07s / 626 MB | [`c20261004_combo_s4_i16.zip`](../../submissions/c20261004_combo_s4_i16/c20261004_combo_s4_i16.zip) — `c4dc258b529a52267a5e108ef0936f7919221cbdd17913270c8eb3482557f94a` | User-pasted portal record; E450 package/parity report |
| 2 | Seed-3 equal parent/head combo, `10tq248b` | **0.6655** | 37m47s / 625 MB | [`submission.zip`](../../submissions/c20261004_combo_s3_i16/submission.zip) — `21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb` | User-pasted portal record; E462 compatibility checks |
| 3 | Seed-3 fine-tuned head endpoint, `0np7c273` | **0.6654** | 34m01s / 624 MB | [`c20261004_combo_head_s3_i16.zip`](../../submissions/c20261004_combo_head_s3_i16/c20261004_combo_head_s3_i16.zip) — `715958e25a12cf75b105bfe80668798aa3a4a1153902e0dd28bfa3b53e049a73` | User-pasted portal record; E462 compatibility checks |
| 4 | Seed-3 parent endpoint, `0cawau9i` | **0.6641** | 33m57s / 624 MB | [`c20261004_combo_parent_s3_i16.zip`](../../submissions/c20261004_combo_parent_s3_i16/c20261004_combo_parent_s3_i16.zip) — `a908efd5903396070b7e0b2aef98454b91fea31316e3621ce5c6ec5f1b62ae46` | User-pasted portal record; E462 compatibility checks |

The source portal rows and score provenance are recorded in [`PORTAL_RESULTS_20261004_LATE.json`](../campaign_2026-10-04/PORTAL_RESULTS_20261004_LATE.json). The portal timezone was not explicit in the supplied evidence. The portal score/runtime data should be treated as user-reported, not independently fetched.

The seed-4 package also has E450 actual-native parity evidence: 20 complete streams (400,000 rows), v2/v3 bitwise agreement, reset and future-suffix causality checks, and an FP32 comparison. It completed in 164.15 seconds with 580,894,720-byte peak RAM and verified teardown. Its actual public run supplies the full deployment runtime. The seed-3 head and parent have actual public runtimes and scores. All four packages are already submitted; the seed-4 combo is the current best.

## Pending packages and non-candidates

| Candidate | Evidence and exact artifact identity | Status |
| --- | --- | --- |
| E462 q8r Python 3.11 repair | [`E462_q8r_python311_repaired_v1.zip`](../campaign_2026-10-04/E462_python311_repair_v1/q8r/E462_q8r_python311_repaired_v1.zip), SHA-256 `c325c570b293af3165e9340a936ac8c97610dea51920a3365494692226d53bdd` | **Do not upload yet.** The clean Python 3.11.16 compatibility report passed its static ELF/import/callback checks, full two-sequence streams, reset, and bitwise archived-output equivalence. It has no successful quiet-host timing approval and no public score. |
| E462 q8x Python 3.11 repair | [`E462_q8x_python311_repaired_v1.zip`](../campaign_2026-10-04/E462_python311_repair_v1/q8x/E462_q8x_python311_repaired_v1.zip), SHA-256 `fde32cb046518c9d7eac9cd35342b959321f5acd46a867421b43bcbad5cd125b` | **Do not upload yet.** Same compatibility scope and missing timing approval as q8r. |
| q8r/q8x original Python 3.13-linked ZIPs | [`q8r original`](../campaign_2026-10-04/E462_public_q8_build_v2/q8r/E462_public_q8_build_v2q8r.zip), SHA-256 `7edb26d8903bafa2f3c38bf31b1a5d0dad1fd6f79e1f4d5a0ea13eaf3ca044fd`; [`q8x original`](../campaign_2026-10-04/E462_public_q8_build_v2/q8x/submission.zip), SHA-256 `b435c04c0b047c72aeaca93a0be5aa2436a885d3abce5da483bd7fa43bfa87e7` | **Withdrawn; never upload.** Both use native libraries requiring `libpython3.13.so.1.0`. q8r submission `1yv88h4s` failed prerun; q8x was withdrawn before submission. The repaired ZIPs are distinct artifacts and do not overwrite these failures. |
| Fixed seed-3/seed-4 equal bag | No public ZIP/submission. The saved exposed-panel score is 0.701420130 versus 0.701694557 for seed 4. | **Not ranked for submission.** Its local exposed-panel point is below seed 4; there is no untouched public-transfer evidence or runtime approval. |
| Activation studies, including E488 | Experimental checkpoints/results only; E488 is now graded and independently reproduced. | **Not a submission candidate.** No package, runtime proof, or public forecast is ready. Do not promote these pilots based on proxy scores. |
| W384 dual-T4 study | Transport/runtime failures; no valid completed full-fit result. | **Not a submission candidate.** A resource/transport attempt is not evidence of a new best. |

The repaired-q8 compatibility report is [`compatibility_report.json`](../campaign_2026-10-04/E462_python311_repair_v1/compatibility_report.json). The q8 local paired score report used the original ZIPs and a repeatedly inspected 368-sequence panel: i16 0.699821638, q8r 0.699816586 (difference -0.000005052; paired interval [-0.000043098, +0.000030580]), and q8x 0.699804134 (difference -0.000017505; interval [-0.000069563, +0.000022038]). The separate full-stream parity report is [`E462_FULLSTREAM_PARITY.md`](../campaign_2026-10-04/E462_FULLSTREAM_PARITY.md). Neither report supplies timing clearance or public-transfer calibration for the repaired archives.

The only q8 timing attempt is recorded as failed in [`e462-public-q8-timing-v1.json`](../campaign_2026-10-04/runner/attempts/e462-public-q8-timing-v1.json), with a preserved quarantine receipt. It stopped because the host was outside the preregistered quiet-load window after one timing trial began; it did not finish the paired timing panel. This is not a negative q8 speed result and not a timing pass.

## Expected public score: no defensible new forecast

For any as-yet-unscored candidate, no calibrated central public-score estimate or prediction interval is justified from the saved pair score. The most relevant seed comparison is E450’s seed-4-minus-seed-3 difference of +0.001872398 on 368 previously exposed sequences, with a paired 95% component-bootstrap interval of [-0.000975354, +0.006931454]. The actual public difference was +0.0005. That is one successful transfer in direction, not a calibration curve or a training-seed distribution; the interval conditions on the fitted models and the local panel was repeatedly inspected.

The q8 paired panel is also exposed and explicitly descriptive. Its small differences cannot be added to seed-3’s 0.6655 public point to forecast a repaired-q8 public score. The earlier `0.664–0.666` seed-3 endpoint heuristic in the 20:41 shortlist was explicitly an uncalibrated illustration and is superseded by the actual portal outcomes. Do not present it as a confidence interval, calibrated bound, or expected value.

The best actual public score is therefore the fixed seed-4 combo’s **0.6660**, not a forecast. The newer 0.7000 aspiration is unsupported by these submissions. When a new package later receives a real portal score, record that observed score and exact archive identity; until then list its public score as unmeasured rather than filling in a local proxy.

## Research wrap and next decision

E488’s fixed mixture minus GELU at epoch 3 was +0.0089295, +0.0043077, and +0.0086897 across three seeds (two intervals excluded zero; one crossed zero). Recurrent-context minus static-mixture was +0.0000838, -0.0000127, and +0.0001658, with all intervals crossing zero; the three epoch-1 recurrent-minus-static intervals were negative. Duration interactions were positive in all seeds, with one interval excluding zero. The results support a fixed-mixture effect on this exposed panel more than an added context-gate effect; they are not a public-score forecast. E488’s complete 24-snapshot/27-global/54-target report is [`REPORT.md`](context_mixture_duration/REPORT.md).

The October 4 reflection consolidates the seed-4 transfer, endpoint comparison, local validation limitations, and deployment lesson in [`PUBLIC_RESULTS_REFLECTION_20261004_LATE.md`](../campaign_2026-10-04/PUBLIC_RESULTS_REFLECTION_20261004_LATE.md). The October 5 activation retrospective is [`ACTIVATION_RETROSPECTIVE_2026-10-05.md`](ACTIVATION_RETROSPECTIVE_2026-10-05.md). The 224 completed activation fits now have official grades; E488’s 24 snapshots, 27 global contrasts, 54 target contrasts, and all bootstrap draws were independently reproduced. These remain mechanism and optimization studies on historically exposed components, not public-prediction evidence. See the [E488 report](context_mixture_duration/REPORT.md).

Current submission action: retain the seed-4 combo as best; keep the four portal-scored archives and their receipts immutable; keep both repaired q8 archives pending a completed timing gate; retain the two failed original q8 archives as incident evidence; and do not promote any activation pilot based on its local proxy result.
