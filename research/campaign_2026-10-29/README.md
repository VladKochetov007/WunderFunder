# Campaign 2026-10-29 — TabM-style recurrent ensemble and custom Muon variant

User (2026-10-29): "Read updated current status and continue in interesting directions. think
about custom optimization, about TabM like architecture but for this competition with
recurrency", plus a reflection request on idea triage. Resources: up to 30 GB RAM, 5 GB VRAM,
all Kaggle. Local slice raised from the stale Oct-24 1 GiB/1-core cap to 28G high / 30G max /
10 cores (it had stalled the first smoke run in memory reclaim).

- [TabM-GRU and NorMuon screen](tabm_gru_r1/README.md) — bounded exploration on the fast cold
  Muon recipe; ceiling stated before fitting.
- [Idea-triage reflection](../unified/methods/IDEA_TRIAGE_REFLECTION_20261029_R1.md) — what
  separates useful ideas here (ceiling, mechanism presence, transfer, detectability, decision)
  and the matching skill change.
- [Fresh-context reframe](reframe_r1/README.md) and [view-mask screen](view_mask_r1/README.md).
- [Teacher-quality track](teacher_track_r1/README.md) — train-only (ADR0044) blend and
  distillation of PR49 teachers into the fixed Muon GRU child; plan and ≤10-fit budget declared
  before launch. Parent P 0.68671, labels-only child C0 overfits to 0.67562; PR49 teachers
  reproduced locally (fin256_s3 0.68780 vs posted 0.68826). D1/D2 pilots landing.
- [Validation audit](validation_r1/README.md) — official valid is a regime-shifted block
  (adversarial AUC 0.984, 1% cross-split time links); shift is multivariate (max univariate AUC
  0.70); replicates E-094/E-110/E-161. Protocol: [robust validation](../unified/methods/ROBUST_VALIDATION_20261029_R1.md).

- [PR49 TL3C submission ZIPs](submissions_pr49_r1/README.md) — two checked uploadable ZIPs (blend 0.692336 / single 0.691434 served full valid).

## Status: paused by user (14:10Z)

"Finish with current ideas. pause research loop." Only the teacher track's declared fits are
landing; see [pause receipt](PAUSE_R1.json). Final results are written here on landing.

## Stopped by user (17:25Z)

All jobs killed; see [stop receipt](STOP_R1.json). D1 (α0.9) finished at 0.68997 (+0.0033 over parent P, below teacher fbc1k 0.69142); D2 interrupted and needs a rerun. Resume steps are listed in the receipt.
