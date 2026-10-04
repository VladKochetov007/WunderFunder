# Second review: why the campaign stalled, and what to do next

Written 2026-10-03 (post-wrap). Read-only review of the paused checkpoint on
`main`. No experiment, training, queue unpause, or upload is performed or
authorized by this document. Companion to `SECOND_PAIR_OF_EYES.md`.

## Why the idea machine stopped

Facts from the checkpoint, not speculation:

1. **After the 10-03 morning screens, no agent had idea generation as a job.**
   The last forecasting experiments are the morning screens (Muon/chunk1000/SBD/
   combo, W384, c3on/off) and the fold2/3 combo confirmation (gate passed
   20:31, E-381). Every round-6 agent task in `round6/agents.json` is an
   instrument or a review: AMP, ARTBP premortem, ARTBP oracle, state-drift
   review, state response. The ledger tail E-382..E-405 contains no new
   forecasting candidate at all.
2. **The process ratchet inverted the cost structure.** A new empirical idea
   now needs prereg + provenance + byte-identity + independent review before
   GPU time, while pure measurement work (capture v4, state drift, AMP audit,
   ARTBP) proceeded freely. `round6/month_research_options.md` proposes up to
   80 GPU-hours with weeks 1-2 spent on identities/projections/guards. Its own
   text concedes "no public score follows". The agent did not run out of
   compute; it ran out of *licensed* hypotheses.
3. **"No untouched data" became a veto.** The strongest candidate's decisive
   follow-ups (full-data training, W384 fold1) were not blocked by evidence —
   only by pause markers and the 23:00 deadline. Meanwhile the fold2/3
   confirmation proves that declared exposure on explored folds is perfectly
   usable evidence.
4. **Session fragmentation.** Usage-limit pauses ate most of 10-01 and 10-02
   wall time; the 23:00 stop cut the evening where synthesis normally happens
   (10-01 and 10-02 both produced their best ideas in written wrap-ups).

Net effect: the campaign ended with a confirmed recipe it never shipped, a
half-run significant result (W384), and a research plan whose remaining items
are all measurement.

## Elephants in the room

### E1. The best confirmed recipe was never trained on full data or submitted
The Muon + chunk1000 + stopped-bootstrap (h8) combo is the only candidate with
gates passed on four folds: +0.00185 pooled bag (folds 0/1, CI [0.00042,
0.00341]) and +0.00179 (folds 2/3, CI [0.00092, 0.00275]), public-like
diagnostic +0.00187. `WRAP_COMPLETED.md` line 30: "No new full-data Muon/
chunk1000 submission exists." The four packaged archives are the 0.6641
reference, an untested teacher variant, a precision variant, and an old
chunk500 ensemble. With the measured 0.8 public/local transfer slope (10-01
REPORT; range 0.63-0.92, "not identified for new families"), expected public
is ~0.6655, not 0.666 — but it is the highest-EV action available and it is
blocked only by the pause, not by evidence.

### E2. Class 3 carries ~46% of public metric weight and idea generation for it stopped
Class WP is 0.43 vs 0.81/0.76/0.74 for classes 0/1/2. Known structure (10-01
E-302/303, H-002, H-042/043): targets are piecewise constant on a6 endpoint
rows, 9 distinct values, ~35% zeros; interior t0+t1 = -spread exactly on known
codes; an oracle branch readout reaches 0.968. Yet after H-008..H-016, H-022,
H-028 were all refuted (09-27 era), later rounds stopped proposing class-3
mechanisms entirely — including in cohorts where class3 was the dominant
reweighted slice. The combo's fold2 class3 delta is -0.0003 (fold3 +0.0068):
the current best recipe is at slight class-3 risk, and the packaging gates
watch class3 but nothing *targets* it. The one class-3 idea family never
tested is H-046 (branch/U categorical head, causal inputs, analytic rational
readout) — E-066's refutation was an oracle-spread leakage artifact, and the
structural S/D head E-303 was a different, code-based construction.

### E3. The runtime gate, not modeling, decides the submission shape
2x GRU256 members = 464 kcyc/row = ~57 public min > 52-min gate (E-336), so
every width-256 blend is banned; the 0.6641 winner is a single trunk. Engine
levers exist and were shelved as "engineering": integer-madd and an L2-resident
weight split, estimated -15%, putting a 2x GRU256 pair bag at ~48 min. A pair
bag historically adds +0.0004..+0.0008 (H-064, E-333). Conversely, one w384
trunk (~1.5x compute = ~43 min) fits where two 256s do not. The engine is a
scoring lever and should be prioritized as one.

### E4. W384: a significant positive result left half-run
`c1_report.json`: W384 (hidden 384, otherwise the d256 control recipe) beat
d256_s3 by +0.0020 on fold0, CI [0.00072, 0.00332], p~0.002. Only
`c20261003_c1_w384_s3_f0` exists — fold1 was never launched. Width320's flat
pooled result (+0.00007) was taken as "width does not help", but w320 ran a
different context (AdamW, chunk2000), and the Muon x width interaction is
untested. Seed noise on fold0 is ~+/-0.001 (S100s4 vs S100s3 = +0.0010,
p=0.065), so W384's fold0 gain is roughly 2x seed noise with a positive CI —
not conclusive, but it is a three-fit question, not an open research program.

### E5. The teacher-ensemble conflict was never resolved
E-330/337: teacher ensemble (since pair + alpha-0.7 student) +0.0011 on two
folds, CI > 0. E-339: the final student contradicted it on holdout (-0.0008).
The declared arbiter — public score of upload 3 (TE07s3) — was packaged but
never submitted: the 10-02 registry ends at D07s4 = 0.6639 and has no TE07
row. Distillation quality multiplies every future student; this uncertainty
caps generation 3 (the 10-02 "next steps #4" gen-3 teacher ensemble was also
never executed).

### E6. Metric dual-reporting quietly dissolved
All 10-03 screens report only `global_wp_clip2_v1` (pooled). The frozen
contract (skill/contract-2026-09-28) requires both pooled and block-average
for serious candidates until the official metric is resolved. The
`legacy_sequence_mean_not_official` blocks show rankings occasionally differ
(e.g. w320 legacy +0.0006 vs pooled +0.00007). Public calibration supports
pooled, so this is low-risk — but final packaging checks should re-include
the block metric rather than inherit the shortcut.

### E7. The instrument stack was pointed at hypotheses with ~zero score content
The finished instruments keep returning "no lever": state drift is
DETECTABLE_SUBTHRESHOLD (prediction RMS 1.6e-5, five orders below any
score-relevant scale); AMP component residual L2 is 4.5e-4 with "no
decomposition bound or score benefit"; the ARTBP oracle only distinguishes
under clipping/Adam transforms, i.e. it fails on the actual optimizer. The
correct response after each null was to stop. Instead the month plan budgets
finishing them (state_drift_response, metadata projection, macrostep pilot
<=8 GPU-h + <=16 confirmation). None of these items names a forecasting
decision it unblocks.

## Underestimated ideas and ideas rejected for nothing

| # | Idea | Ledger status | Why it deserves another look |
| --- | --- | --- | --- |
| U1 | W384 width scaling + Muon x width | fold0 only, +0.0020 CI>0 (p~0.002), then abandoned | Orphaned, not rejected. Three fits (f1/f2/f3 of the existing `dcurve_fold.sh` recipe) decide it. |
| U2 | Chunk-mass alignment (month option 4) | designed, never forked | Main-mass CV 15.16% is measured on the actual live recipe; the aligned-vs-permuted fork is ~2 GPU-h on frozen states. The only month option that targets the confirmed recipe's own loss normalization. |
| U3 | Common-macrostep credit, actual test (option 3) | instrument proven (58/58 synthetic), actual comparison never run | If connected credit matters, chunk length is re-opened; if null, chunk1000's +0.0014 is optimization noise. Either result steers the next recipe. ~8 GPU-h pilot. |
| U4 | Late Muon-to-AdamW handoff + module clipping (opt2) | implemented, 81 CPU tests + parity pass, GPU screen never run | The natural follow-up to the confirmed Muon win; opt2_review names these two as the surviving recommendations. Blocked only on GPU/fused/EMA parity checks. |
| U5 | Teacher generation 3 | proposed 10-02 "next steps #4", never executed | Gen-2 teacher ensemble was +0.0011 on two folds. Add the alpha-0.7 seed-4 student and TE07 student (and later a Muon student) as teachers. Resolve E5 first via upload 3. |
| U6 | LSTM h128 member for diversity (H-058) | queued 09-29, never run | Seed partners are 0.985-0.999 residual-correlated (H-064), so seed bags add ~+0.0005; architecture diversity attacks exactly that. Deployment-compatible trunk. |
| U7 | Branch/U categorical head, rational readout (H-046) | proposed 09-27, never fit | Oracle branch knowledge gives 0.968 (H-043). E-066 failed on oracle-spread leakage; E-303 was a different code-based head. A causal CE branch head is the one structurally new readout family left. High risk, highest ceiling; the reserved "structurally novel arm". |
| U8 | Propensity-gated mixture heads (H-032); residual decomposition class x rows-to-break x |y| (H-024) | proposed, never fit | H-023 (mask-weighted predictor = WP-optimal) is supported, so gate/mixture structure is principled, not ad hoc. |
| U9 | Common increment coordinate (H-053) | `blocked_by_frozen_fit_gate_failure` | Blocked by an instrument incident, never evaluated on evidence — literally rejected for nothing. Un-block or close explicitly. |
| U10 | c3on/c3off class-3 distillation screen | marginal pooled (+0.0015, p=0.054), inconsistent folds, dropped silently | Either close it with the predeclared rule or run the deciding fold; an ambiguous orphan in the ledger is how class-3 work died the first time. |

Properly rejected (do not reopen): bilinear cross-member head (E-270, -0.0057
with matched controls — the skill's "priority mechanism 1" was actually tested
and failed honestly); weight soup (E-332); class routing parent/FT (E-335);
DTW family (E-334); width320 in its tested context; teacher-ensemble-final
holdout contradiction stands until upload 3 resolves it.

Unfit siblings worth noting: H-048/H-049/H-050/H-051/H-052 (structured state,
content memory, specialist mixture, content retrieval, current-spread aux)
are all `proposed_not_fit`/pending from the 09-29 campaign. One sibling of
that batch (bilinear) was tested and failed; the rest were never run. They
rank below U1-U10 because their mechanisms overlap the failed recurrence and
calibration families, but H-052 (current-spread aux with proper caching) is
cheap and directly probes the supported H-042 target law.

## Priorities

### P0 — unblock and ship the confirmed recipe (days, ~2-4 GPU-h + upload)
1. Train full-data combo d256, seed 3 (add seed 4 only if budget allows — a
   two-member bag does not fit the 52-min gate at width 256 anyway, so the
   second seed is for gen-3 teaching and insurance, not for the submission).
   Exact frozen flags: `--optim muon --chunk 1000 --distill-boot-h 8`,
   teachers = incumbent since parents, alpha 0.5, propensity `c20260929_prop`,
   fixed epoch 3 (FOLD23_CONFIRMATION.txt). Package single trunk int16,
   full packaging gates, both metrics recorded, upload.
2. Expected public ~0.6655 (0.8 x +0.0018), honest PI roughly 0.6645-0.6660.
   Do not advertise 0.666 before the draw.
3. In parallel on CPU-queue time: W384 fold1 (then folds 2/3 if positive),
   three fits from the existing script. Decides E4/U1 and, if width holds,
   opens the muon+w384 single trunk (~43 min, fits the gate).

### P1 — this week
4. Upload 3 (TE07s3) as the public arbiter of the teacher question (free,
   already packaged and verified). Resolves E5 before any gen-3 GPU spend.
5. Teacher gen-3 distillation (U5) once the Muon student exists.
6. Engine: integer-madd + L2-resident weight split to bring 2x GRU256 under
   52 min (~48). Enables pair bags (+0.0004-0.0008) on top of Muon — the most
   credible remaining route to >=0.666 without new science.
7. opt2 late-handoff + module clipping GPU screens (U4), after GPU/fused/EMA
   parity, one fold each, same no-op controls as planned.
8. Month options 4 then 3 (U2, U3): chunk-mass alignment fork (~2 GPU-h) and
   the single macrostep-credit actual comparison (~8 GPU-h). These are the
   only instrument-ready items that touch the live recipe.

### P2 — the reserved structural arm
9. Causal branch head (U7): CE on branch + U-magnitude head from causal
   inputs only, propensity-weighted, rational readout; matched-budget control
   = the trunk's own wMSE head. Promotion requires a class-3-under-pub143 win,
   not just pooled. Predeclare the branch feature source; no oracle spread,
   no target-derived inputs at inference.

### P3 — close or freeze
10. Keep frozen: state_drift_response, AMP metadata projection, ARTBP
    training proposal, and the 80-GPU-h month lab as scoped. The ARTBP blind
    critic cross-review is CPU reading and can be done anytime. Formally close
    U9 and U10 with one-line ledger entries stating on what grounds.

## Process fixes so this does not recur
- One agent owns a living *candidate queue* (the thing that stopped). Idea
  generation is a role, not a by-product.
- Any instrument task must name the forecasting decision it unblocks and its
  expected effect size; otherwise it queues behind candidates.
- Declared-exposure confirmation on explored folds is acceptable evidence
  (the combo proves it); "no untouched data" may slow claims, not experiments.
- Reserve the 10-20% structural arm in the queue, not in a document.

## Caveats on this review's own claims
- All combo deltas come from previously explored folds; gains are fold0-heavy
  (+0.0030..0.0037 there vs +0.0005..0.0007 on fold1). Public transfer could
  land at +0.0008..0.0014 rather than +0.0018. The 0.8 slope was estimated on
  the GRU128-since family, not on distill/Muon models.
- Muon, chunk1000, and SBD deltas are correlated measurements on the same
  panels; do not sum them. Combo minus Muon is +0.00056 (p=0.10) — the recipe
  is adopted as a bundle, not as three stacked wins.
- Class3 fold2 -0.0003 is small but sits in the 46%-weight class; the
  packaging class guard matters more for this recipe than for past ones.
- The four ready archives: only #1 is a known quantity; #2 (TE07) is the
  cheapest new information per upload; #3/#4 are controls, not candidates.
- Everything here is read from local evidence; no public score was observed
  beyond the registry, and no forecast above is calibrated to LB noise.
