# H232 V7B delivery-only review correction

V7B repairs the review-receipt packaging error that stopped V7A before any
profile or fit work. It carries the V7A science and execution packet unchanged,
binds the approved conditional-fit review to the worker's actual validator,
and uses a distinct canonical Kaggle kernel title and slug. V7B is built for
root review; its receipt keeps provider push unauthorized.

## V7A terminal diagnosis

The only V7A push was version 1 of
`vladkochetov07/h232-gru384-b8-amp4096-v7-proactive-cache-recovery`. Kaggle
reported `KernelWorkerStatus.ERROR`. The downloaded phase record and provider
logs show `ProfileAbort: external review receipt lacks conditional fit
authorization`; the pushed packet still carried the pending placeholder
review. The abort occurred in the review gate, before input identity, GPU
profiling, fitting, checkpoint creation, or prediction export.

The full terminal download and hash manifest are in
[`delivery/h232_v7_proactive/version1`](../../../../../delivery/h232_v7_proactive/version1).
The terminal receipt SHA256 is
`9d60eb07b79ae2de6a2afa2d1d7b331d4bd3435caedcde2b0d0759e3ea1715db`.
It records all 19 downloaded output files (1,102,373 bytes), the raw provider
execution log, the kernel log, the phase file, and the empty provider file
listing. No checkpoint or prediction artifact was returned.

## V7B correction

The V7B kernel is
`vladkochetov07/h232-gru384-b8-amp4096-v7b-proactive-cache-recovery`, with
title `H232 GRU384 B8 AMP4096 V7B Proactive Cache Recovery`. The experiment ID
and all training run names remain the V7A values because the scientific and
execution contract did not change.

The external review receipt now has the exact `approved`,
`conditional_fit_authorized`, `same-worker-only-after-profile-gate`, reviewer,
and reviewed-hash fields required by `validate_external_review`. Its 10
reviewed hashes match the already root-reviewed V7A inputs, and the packet
metadata carries the same map and the new review digest. The receipt binds the
qualified independent 24-test review, the previous V7A root admission and
push receipt, the V7A terminal failure, and root's V7B delivery-only
authorization. It explicitly leaves `provider_push_authorized` false; root
must provide a fresh V7B push admission.

Only these provider packet files differ from V7A: `external_review.json`,
`packet_meta.json`, `kernel-metadata.json`, and the regenerated transport
wrapper. The exact metadata diff is
[`V7B_V7A_DELIVERY_DIFF.patch`](V7B_V7A_DELIVERY_DIFF.patch). The source
archive and manifest remain byte-identical to V6 at SHA256
`f1333c63eca38e69149e14dbc94c6a701ba97e113d6679b8939d7c860f1a95e4` and
`6ee80717ab4cec6da8b9ef3f124439a7598c2dfde2fa25de9fb35bf47a0adc42`.
All 14 provider packet file hashes and the regenerated bundle hashes are in
[`V7B_BUILD_RECEIPT.json`](V7B_BUILD_RECEIPT.json).

The review preserves the 28% proactive trigger, 25% and 6.5 GB hard gates,
512 MB recovery margin, 768 MB per-event cap, 320-event/245.76 GB cumulative
caps, 20-second interval, and identity-bound recent-rowgroup advice. The
16,000-second worker hard wall and full four-epoch/fixed-epoch-3 export
contract are unchanged. Root's noted zero-second CUDA-cleanup accounting and
pre-cleanup projection remain a nonblocking unresolved reporting limitation;
the old duration estimate is not promoted.

## Validation and admission boundary

The focused V7 recovery, fit-gate, rowgroup, and review-preflight tests passed:
25 tests. Ruff passed. The preflight invokes the actual worker validator,
checks every reviewed hash and root-authorization binding, and does not import
Torch or PyArrow. The generated Kaggle wrapper parses successfully. The source
archive builder confirms byte identity with V6.

V7B has not been pushed and its provider status has not been queried. No fit,
checkpoint, prediction, metric, or score is claimed. Root's independent review
and fresh push admission remain required.
