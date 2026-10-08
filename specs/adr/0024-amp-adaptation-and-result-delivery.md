# ADR0024: AMP adaptation and empirical result delivery

Status: Accepted

Date: 2026-10-08

## Context and evidence

The own strong Muon parent completed with explicitly counted AMP skips. The S4 head-control diagnostic observed one initial scale backoff, followed by stable scaling. Its fixed fit subsequently delivered 870 attempts, 869 completed updates and one skip, with finite tensors and unchanged frozen parent parameters. These observations do not support treating every AMP skip as persistent instability.

The frozen capacity delivery checker nevertheless required zero skips in every epoch. This could discard finite checkpoints after expensive training without obtaining the intended scientific comparison. Historical recipes and failures remain immutable.

## Decision

For new identified successors, retain the declared AMP configuration, optimizer, objective, sampler and fixed epoch. Verify exact attempted-update budgets, nonnegative skip counts, positive completed updates, and the identity `completed + skipped = attempts`. Persist epoch and cumulative counts. Verify finite checkpoints and required predictions, declared membership and checkpoint selection, and actual resource/exit status.

AMP skips are numerical evidence to interpret, not a universal scientific rejection rule. Sustained inability to update, nonfinite artifacts, contradictory counters, OOM, or incorrect membership require diagnosis before a valid-result claim. Sampled finite loss telemetry is not proof that every loss was finite. An experiment with skips must disclose its actual update budget and precision when compared with controls.

Root owns admission and any explicit supersession of an older zero-skip gate. Builders cannot silently relax frozen recipes or alter the grader. The V5R3 capacity successor must preserve V5R2 and change delivery validation under a new ID.

## Alternatives and consequences

A universal zero-skip gate is easy to automate but confuses normal scaler adaptation with failed training. Automatically restarting every such run wastes compute and changes training variation. Lower-scale or different-precision retraining remains an eligible separately recorded experiment; it is not required before evaluating an otherwise valid finite result.

## Revisit when

Counters or optimizer-state audits reveal incomplete accounting, recurrent numerical problems affect validity, or controlled comparisons show a different precision treatment is preferable.
