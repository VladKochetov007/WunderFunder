# Independent source review of E427 production smokes

This review covers only the new `staging/e427_smoke_review` copies and draft
specs. The actor's E427 trainer/model and the root trainer/model remain
unchanged. I inspected sources and ran the one lightweight stdlib packet test;
no CUDA, model import, fit, export, or score was run.

The full-shape parity gate uses the actual first batch on CUDA and carries both
GRU states over all 20 chunks. The E427 model has a 264-wide first recurrent
input whose added eight update-gate columns start at zero, so the full output
and hidden-state comparison against the loaded 256-wide parent exercises the
intended zero-extension equivalence. The tolerance is `1e-4` for both tensors.
The separate prefix probe changes only raw input suffix rows 512 onward and
requires exact equality of earlier cached context features and stale-source
indices. That checks causal closure using actual CUDA input values without
reading any target or holdout sequence.

After the two full batches, the gate checks exactly two completed loader
batches and 40 optimizer attempts, plus the student and EMA projection audit
counts. AMP skips and completed updates are reported separately. Aligned and
stale require an actual completed step and nonzero finite gradients/weights in
the added path; no-op requires the added weights and gradients to remain zero.
The trainer applies the existing post-attempt projection after every Muon
attempt, including skipped scaler steps, and applies it to EMA after every EMA
update; the smoke counts those audits. Both host RSS and allocated/reserved
CUDA peaks are checked against the requested 24-GiB, 4.2-GB, and 4.8-GB limits.

One trainer behavior matters when reading the counters: the existing loop calls
the EMA update and advances the scheduler after each optimizer attempt,
including a GradScaler-skipped update. The smoke reports skipped attempts and
completed parameter updates separately; `ema_updates == 40` therefore means 40
EMA update calls, not 40 successful parameter updates.

The three drafts are identical in every inherited fixed-recipe field except
the declared arm, run name, and smoke-only profile/no-evaluation controls.
They retain the batch-32, 20,000-row shape and chunk-1,000 windows. The
one-batch shortcut was not used. `--no-eval-during-fit` prevents creation of
the evaluation dataset/loader. The trainer reads validation sequence IDs to
form the fixed `eval-source` group list, but it does not load holdout feature or
target sequences, calculate a score, or use a score for selection. The exact
teacher lineage and prior score exposure still mean that a later E427 result
cannot be called untouched generalization.

The queued smoke records are diagnostic only. A later epoch-0 export must
load the exact fixed `model_ep0.pt`, validate output width 23 and the saved
parent normalization, and bind sidecar hashes to the E427 checkpoint,
provenance, parent checkpoint/provenance, fit-only context-statistics file,
and code-source closure. It must predict all 1,873 VALID groups in canonical
typed order, with FP32 shape `[1873,20000,2]`; scoring remains a separately
declared component-clean fold screen. No prediction or score artifact has been
staged by these smoke specs.

Lightweight test result: `test_smoke_packet.py` passed (1 test). The three
specs are drafts only and require root review/admission after queued local GPU
work. Nothing here was sealed or launched.
