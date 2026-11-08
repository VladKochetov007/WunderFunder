# Eight-B300 multistage campaign

The user authorized the four-stage ensemble research campaign on31October2026.
The [experiment packet](../../experiment_drafts/multistage-optuna-b300-20261031/README.md)
contains the bounded contract, exact input/config identities, independent reviews
and original receiving evidence. This note indexes the campaign; execution status
comes from its launch and terminal receipts.

Primary validation follows ADR0044:10325 eligible official-train fitting
sequences, zero fitting valid sequences, all1873 official-valid evaluation
sequences. The inner8260/2065 whole-component train/control partition is an
auxiliary nonchronological proxy, with no official is_scored mask. No official
valid fitting or learned blend weights are permitted.

The six original GRU/LSTM batch16/32/64 profiles each completed50 real updates in
41–42seconds. Two batch256/prefetch1 profiles completed50 updates in23.2–23.4s,
processing12.86million real rows each. This is throughput evidence, not quality
evidence. Their VRAM reservation was6.57GiB and peak host RSS17.35–17.38GiB.
The original16GiB host admission ceiling is exceeded; an explicit resource-plan
revision and independent profile review precede full fits. Preserve the earlier
six failed Python-bootstrap attempts (zero updates) and all original receipts.

Eight unrelated NIER workers and the existing MPS server remain untouched.
The native source snapshot is immutable; controller/evaluation changes have
separate source identities. Keep NVML capacity and Torch allocator-visible
capacity distinct while joining them by exact device UUID/model. SSH disconnects
must not stop admitted fits, and independent timeout watchdogs bound owned work.

At every stage retain raw metrics and actual-data PNG/SVG: paired train/control
loss and WP, per-target scores, objective terms, LR, preclip gradients, clipping,
nonfinite counts, resources and throughput. Keep official-valid reporting
separate, checkpoint roles explicit, and results traced to exact source/config/
split/parent/weights/prediction identities. No new quality score is asserted here.
