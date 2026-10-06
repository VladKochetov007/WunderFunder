# Proposed causal event/quiet update falsifier

Status: **design proposal only**. No model code, fit, score, or admission is
authorized by this note.

## What E492 supports

E492 v2 independently reproduced 384 per-component and 64 aggregate rows across
24 fit-role components, with no evaluation components loaded. The report binds
the event analyzer to SHA-256
`36c82e18d4a9b15b613f629ce78ab29d7b96c1e2b4f0fd2dcd84a063ffd57388`; the v2
receipt is `91994bc286880968263ef4650d7f9fea5a39541767b8cdc1ef88b34646808b0a`.
For the production bid-AND-ask clock, eligible fit-panel `t0` target changes
were nearly all accompanied by an event (0 misses in diagnostic classes 0, 2,
and 3; 4 of 10,837 missed in class 1). Events were much more common than
target changes: for example, class 0 had 46,045 events and 20,854 target
changes, with 25,191 event rows lacking a `t0` change. These counts are exposed,
descriptive fit-panel evidence. They motivate a mechanism test but establish
neither conditional predictability nor model gain. Class labels are diagnostic
metadata only.

## Falsifiable screen

Freeze one H256 two-layer parent checkpoint and fit fold, seed, epoch, teacher
cache, row order, normalization, optimizer schedule, losses, and masks. Freeze
the parent weights for this first screen so the comparison measures the added
state mechanism. Use only origin labels `Y_t` and their historical origin-row
weights/masks as supervision on training components. Do not expose future
labels, target-transition indicators, evaluation labels, class diagnostics,
`is_scored`, or any label-derived event to the candidate, its state, its
normalizer, or checkpoint selection. Held-component labels may be read only by
the frozen scorer after every model and checkpoint is fixed.

For every row, derive the production event causally from raw features:

```text
e_0 = 0
e_t = (i0_p0[t] != i0_p0[t-1]) AND (i0_p11[t] != i0_p11[t-1])
```

Reset all event history, elapsed age, rolling activity, and residual state at
each sequence boundary. Feed every row after reset through the parent and
residual, including quiet rows and the usual warm-up; emit a prediction for
every required row. Never hold the previous output when no event occurs.

Add a 16-dimensional residual belief state `b_t`, initialized to zero, using
only current/past features and the frozen parent state. Its update is

```text
q_t = [x_t, h_parent_t, b_(t-1), age_since_AND_t,
       event_fraction_(last 32 rows), event_fraction_(last 256 rows)]
delta_t = e_t * f_event(q_t, b_(t-1))
        + (1 - e_t) * f_quiet(q_t, b_(t-1))
b_t = 0.99 * b_(t-1) + tanh(delta_t)
prediction_t = prediction_parent_t + W_out * b_t
```

Zero-initialize `W_out`, so the candidate begins exactly at the parent. Compute
the age and activity covariates from the causal event prefix; give the same
covariates to every arm.

Run one preregistered seed-3/fold-0 screen with these paired arms:

1. **AND event/quiet update:** the equation above.
2. **Unrestricted residual:** the same state size and parameter count, but a
   context-driven learned gate selects its two innovation branches; it receives
   age/activity covariates but no AND bit. Update it on every row.
3. **Stale-event control:** the same event/quiet update with `e_(t-16)` as its
   routing bit, defined as zero when the history is shorter than 16 rows. It
   receives the same true age/activity covariates, and still updates and emits
   on every row.

Use identical initialization streams, batches, data order, optimizer updates,
and fixed checkpoint selection for all arms. Score the same held components,
rows, and masks once with the settled primary metric `global_wp_clip2_v1`.
Report mean per-sequence weighted Pearson as a secondary diagnostic. Use 2,000
paired component-bootstrap draws and show both targets.

The mechanism screen passes only if the AND arm improves pooled score by at
least `+0.0005` over **both** controls, with both target deltas positive, and no
diagnostic class loses more than `0.0005`. Also report the parent separately.
A gain over the parent that does not beat the unrestricted or stale-event
control is a failure of the event-specific hypothesis. A positive screen only
earns fresh seed-4/fold-0 and seed-3/fold-1 confirmation; it does not support a
public-score claim.

Before any fit, require independent source review, a full-shape one-worker
resource profile under the current 3,000,000,000-byte observed-VRAM stop and
aggregate CPU/RAM limits, exact checkpoint/data/split/cache hashes, and root
admission. Keep E492's original EDA and the failed v1 recount immutable.
