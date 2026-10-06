# Independent elephants review — October 5, 2026 (second pair of eyes)

Read-only audit launched after the October 5 reviews. I did not start fits,
runs, uploads or admissions. Everything below comes from existing source,
cached predictions, the experiment/submission ledgers and the raw portal
record. This supplements [ELEPHANTS_REVIEW_2026-10-05.md](ELEPHANTS_REVIEW_2026-10-05.md)
and [SECOND_PAIR_OF_EYES_2026-10-05.md](SECOND_PAIR_OF_EYES_2026-10-05.md); it
does not replace or amend them.

I re-derived the numbers I cite with small scripts over `cache/valid_targets.npz`
and `cache/pred_*_valid.npy` (the 368-sequence `holdout368` domain defined by
[`src/wnn/splits.py`](../src/wnn/splits.py)), and cross-checked two of them
against the campaign's own saved artifacts. Where a claim is mine it is marked
**measured here**; otherwise it is linked.

## Why the reasoning machine is stuck

The prior agent is not out of ideas. It has run out of *resolvable* ones.

1. **Every remaining local knob moves the metric by about as much as the
   validation→public transfer noise.** The submission-transfer fit in
   [`research/lb`](lb/lb_probe.py) has leave-one-out RMSE of `11.2e-4`
   (`pub143` mix) with family-dependent residuals from `+16e-4` to `-13e-4`
   (re-run of `lb_probe.py fit`, unchanged inputs). Seed-4 minus seed-3 was
   `+0.0019` locally and `+0.0005` publicly. A `+0.001` local screen therefore
   does not identify a `+0.001` public change.
2. **The only unbiased instrument (the public LB) is barely used.** The saved
   record has 20 scored submissions; the rules allow 5/day. Selection is being
   made on the biased instrument and confirmed on the sparse one.
3. **The interesting model families were tested as *replacements*, and a
   replacement that loses is discarded even when it would have value as
   diversity.** That is the single largest methodological error in the
   campaign. I checked whether it actually cost anything, and the answer is
   split — see below. For the E464/468 heads it did not; for the older `ab84/ab85`
   generation it did.

## The elephants

### E1. The public-target milestone (0.7000) has no path in the evidence, and the gap is a transfer problem

Local clean-holdout is ~0.7017 and public is 0.6660: a ~0.036 transfer loss,
larger than every real lever found all week. The historical taxonomy
([game_changers.md](game_changers.md)) is explicit that wins came from
removing a deterministic scorer mismatch (type A, +0.057) or lowering
estimation variance (types C/D, ~0.001 each). Type A is spent. Nothing in the
finished work supports 0.034 of further gain. **The honest recommendation is to
reframe the milestone** (e.g. "public ≥ 0.668 with a measured, calibrated path")
rather than to keep spending compute against an unsupported number. Report the
0.7000 aspiration as unmet; do not manufacture it from local deltas.

### E2. An entire finished generation with the best clean holdout was abandoned and is absent from every current review

**Measured here**, on `holdout368` (`global_wp_clip2_v1`, pooled):

| Candidate | holdout368 | public-mix proxy¹ | class 0/1/2/3 |
|---|---:|---:|---|
| Combo seed-4 (incumbent) | 0.70169 | 0.67478 | 0.8176 / 0.7667 / 0.7460 / **0.4326** |
| `ab85_final_ema6_s4_gclpall` | **0.70404** | **0.67744** | 0.8199 / 0.7656 / 0.7442 / **0.4394** |
| `ab85_final_ema6_s3_gclpall` | 0.70160 | 0.67471 | 0.8180 / 0.7663 / 0.7432 / 0.4332 |
| `ab85_final_onorm_s5_gclpall` | 0.69995 | 0.67301 | 0.8150 / 0.7657 / 0.7444 / 0.4305 |
| blend `0.5·s4 + 0.5·ab85_ema6_s4_gclpall` | 0.70422 | 0.67750 | 0.8201 / 0.7674 / 0.7466 / 0.4383 |

¹ public-mix proxy = pooled WP with classes re-weighted to the recorded public
mix `PUB143 = [0.183, 0.16, 0.198, 0.459]`. It is a diagnostic, not a forecast;
the fitted-transfer `shift b` is `-0.007` (`lb_probe.py fit`).

The incumbent's own `history.json` records `split: holdout368`; the `ab85`
history records the same split and `weighted_pearson 0.7040365`, matching my
independent recomputation `0.70404`. So these are clean-holdout numbers under
the campaign's own evaluation domain. `ab85_..._gclpall` is a **single-model**
artifact (`artifacts/runs/ab85_final_ema6_s4_gclpall/model.pt`), so it fits the
~34-minute one-forward CPU budget — unlike a two-parent blender.

Its siblings that *were* submitted (`ab83_final_ema6_*_gclp`, `ab84_final_ch500`
in [`registry.json`](lb/registry.json), public `0.6633` ensemble) score *below*
seed-4 on the same holdout (0.69496, 0.69995), and public rank agreed. The
`gclpall` variants were never packaged, never in the registry, and are not
mentioned in any October 4–5 report I searched. The advantage is concentrated
in class 3 (the public-important class, E4). Two live caveats: the `gclpall`
name suggests a global-correlation fine-tune; if these variants were *selected*
by holdout368, the 0.7040 is selection-optimistic, and the family may carry a
worse transfer offset than seed-4. The only way to settle it is a submission.

### E3. Ensembling was evaluated only for near-copies, never for the diverse completed finals

The current [submission audit](campaign_2026-10-05/NEW_SUBMISSION_AUDIT_2026-10-05.md)
correctly rejects the same-recipe seed-3/seed-4 bag (I confirm: holdout
0.70145 < 0.70169). But it never tests the dozens of *other* full-data finals
(`ab83/ab84/ab85`, `since`, `onorm`, `d256a07`, `te07`) as ensemble members.
**Measured here:** greedy blending over those members lifts the holdout from
0.70169 to `0.70437` (`s4 + 0.5·ema6s4all + 0.15·ema6s3all + 0.10·onormall`),
and all residuals are correlated ~0.986–0.990, not 0.999.

Two honest qualifications:

- The effect is barely above the ~0.001 transfer noise; treat it as a
  submission candidate, not a proven gain.
- **Deployability caps it.** The complete combo already runs 37m07s of the
  60-minute budget as a single merged model, so a second full GRU256 parent
  does not fit. This is the real reason to revisit **QAT/INT8 as an *enabling*
  technology, not a packaging chore**: a quantized parent is what makes a
  2-member ensemble fit inside 60 minutes. The campaign has tested QAT
  accuracy preservation but never treated it as the ensemble enabler.

### E4. The evaluation under-weights exactly the class the public over-weights

The held-out class mix is `[0.220, 0.160, 0.250, 0.370]`; the recorded public
mix is `[0.183, 0.160, 0.198, **0.459**]`. Class 3 is both the weakest class
(~0.43) and ~46% of public weight, and **the strong-target report** already
shows ~92% of class-3 residual energy sits in the target-difference mode. Yet
most reports advertise pooled WP, and most designs are evaluated pooled. A
class-3 gain of +0.01 is worth more publicly than a +0.01 class-2 gain, and
that is not how the campaign has been ranking. Every future comparison should
report the public-mix proxy alongside pooled and per-class numbers.

### E5. Budget misallocation: 224 activation fits, zero completed width fits

The activation campaign spent 224 graded fits on a fixed 24-fit/16-exposed-
component panel ([retrospective](campaign_2026-10-05/ACTIVATION_RETROSPECTIVE_2026-10-05.md));
its largest surviving effect (fixed 50/50 GELU–LeakyReLU) is ~+0.005 on that
panel. Meanwhile width (W384) — the one architecture axis with an original
signal and a plausible capacity mechanism — is still 0-for-N, censored by a
self-imposed 30 GB local RAM cap, while ~20 GPU-hours sit unused. The fix is
not a bigger declared limit; it is moving the width test to the GPU path or a
genuinely streamed loader. One width replication is worth more knowledge than
another activation grid.

### E6. The causal event clock is treated as a diagnostic, never as a model input

The strong report shows `3,761/3,763` (fitted) and `3,841/3,842` (row-excluded)
scored target changes coincide with a *current* `i0_p0`/`i0_p11` change. That
makes the target behave as a jump process with an almost-observable clock, and
the current-row event is a legal causal feature. The campaign tested only a
post-hoc *hard output hold* on an untrained-in-that-structure model (which
loses, correctly). It never trained an event-gated level tracker with a
quiet-run posterior-update path — the smaller, better-specified hypothesis.
The early target-law note also flagged a feature-only `a6` sentinel as an event
marker; whether `a6` is the same clock as `i0_p0/p11` is unverified and cheap
to check. This is the strongest *structural* lead, and it is the same one the
prior review ranks #2, so treat it as confirmed rather than novel.

## Rejections that should stay rejected (I checked)

I specifically tried to resurrect the "losing candidate as ensemble diversity"
folk hypothesis. It does **not** hold for the recent architecture work:

- E464 eight architecture heads: holdout 0.6949–0.6988, correlation with the
  incumbent 0.9917–0.9946, **zero greedy ensemble gain**. Reject as replacements
  *and* as diversity.
- Same-recipe seed bag: correctly below seed-4 on both the excluded validation
  slice and `holdout368`.
- E468 joint codebook: loses its own regression anchor by −0.000555; no ensemble
  value measured.

So the campaign's negatives on these arms are sound. The failing is not that it
rejected them; it is that it never ran the *survey* (E3) and never looked
outside the recent generation (E2).

## New directions, prioritized

| P | Direction | Changed mechanism | Cheapest decisive test | Falsifier / promotion bar |
|---|---|---|---|---|
| **P0** | Instrument the public LB + settle the abandoned generation | Use the allowed daily submissions as a transfer-calibration experiment, not a final exam | Package and submit `ab85_final_ema6_s4_gclpall` (single model, fits runtime) and one diverse blender; log portal score with exact archive hash. Reserve 2–3 subs/day for pre-registered probes. | Portal score ≤ 0.6660 closes the candidate; a positive score updates the transfer model. |
| **P0.5** | Quantized ensemble feasibility | INT8/QAT parent to fit 2 members in 60 min | Measure native INT8 runtime on one parent; if ≤ ~25 min, package a 2-parent equal blend of seed-4 + the best diverse member | Runtime > 60 min or accuracy parity loss > transfer noise → drop. |
| **P1** | Class-3 programme | Report and target the public-dominant class | Re-rank all existing finals by public-mix proxy and per-class; pick class-3-specialised heads for the ensemble; difference-mode + long-age + quiet-run readouts | No class-3 gain on a second component block/seed → stop. |
| **P2** | Event-gated level tracker | Trained jump-size head at observed events + quiet-run posterior updates | Zero-init residual adapter on strong states vs parameter-matched no-event control; `a6`-vs-`i0_p0/p11` clock identity check first | No gain over matched control on held-out components; oracle-only features → reject. |
| **P3** | Width replication | Capacity, not activation | One W384 fit on the GPU path (or streamed loader), same recipe, frozen scoring | No gain at matched recipe on two seeds → close width. |
| **P4** | Shift robustness | Augmentation / random origins (H-079) / aggressive EMA for the 0.036 transfer loss | One matched-recipe run with random-origin augmentation vs control | No gain on the public-mix proxy and no public improve → close. |

### Ordering logic

P0 and P0.5 are cheap and act on the actual bottleneck (transfer uncertainty),
so they come first even though they may not raise the ceiling. P1 improves the
*ranking* of everything already trained and reuses the binaries we have. P2 is
the only new mechanism with a strong prior. P3/P4 are capacity/robustness bets
that require infrastructure the campaign currently lacks.

### What must not happen next

Do not add another 100 tiny-panel fits. Do not promote a candidate from a
`+0.001` holdout move. Do not spend an admission on a mechanism that cannot be
resolved above transfer noise. Do not restate 0.7000 as reachable without a
measured, submitted path.

## Reproduce my checks

```
# holdout368 pooled / per-class / public-mix scores for cached final models
OMP_NUM_THREADS=4 python3 research/lb/indep_holdout_scan.py
# transfer fit and per-submission residuals
OMP_NUM_THREADS=4 python3 research/lb/lb_probe.py fit
```

Exact artifacts: `artifacts/runs/ab85_final_ema6_s4_gclpall/{model.pt,history.json}`,
`artifacts/runs/c20261004_combo_s4_final/{model_ep3.pt,history.json}`,
`cache/pred_ab85_final_ema6_s4_gclpall_valid.npy`,
`cache/pred_c20261004_combo_s4_final_fp32_valid.npy`,
`research/lb/registry.json`, `research/lb/holdout_stats.json`.
