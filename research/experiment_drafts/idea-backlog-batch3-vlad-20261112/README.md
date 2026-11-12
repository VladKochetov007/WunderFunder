# Idea backlog batch 3: untested mechanisms from our October reframes, aimed at the event clock and the refit gap

**Status: proposal, no fits executed.** Preparation owner: Vlad / Claude Code. Receiving owner:
whoever has capacity (Semyon's L4 contract by default). This PR collects the ideas from our
October reframes and ledger that are **untested, not closed by prior evidence and not already
in a team PR** (#1–#92). Contract as in [#82](https://github.com/VladKochetov007/WunderFunder/pull/82):
PR80 recipe, fixed E6, train 10,325 / full valid 1,873, seeds 203301/203302, mean paired gain
≥ +0.0005 with both seeds positive. The default control is the same-code **since32** cell
(+0.0033 over K16), the current best single change.

Sources: Oct 29 reframe (Opus + Grok), `research/campaign_2026-10-29/reframe_r1/` on branch
`experiment/next-cycle-recovery-20261021` (draft PR #83 publishes it);
`research/unified/ideas/FRESH_REFRAME_20261023_R1.md`; ledger rows E-078, E-079, E-127, H-326,
H-330.

Excluded on purpose (prior evidence): output hold/smoothing of a per-row model (H-326 refuted),
per-class blend weights and scales (E-127: −0.0012 to −0.0027 cross-fitted), quadratic/Koopman
output correction (Oct 29: +0.0003 mean), bid/ask mirror augmentation (S38: maps break the input
marginals), per-sequence online z-score inputs (E-118), burstiness (already in PR #38).

---

## C01. Gap ledger and data-scaling slope (decides where fits go)

**Why.** Train-only models score 0.6636 public vs 0.6683 for full-corpus TL3; the final refit
looks like the largest remaining lever, yet nobody has measured how WP scales with training
data on this contract. Opus (Oct 29) proposed splitting the remaining error into: (a) teacher
ensemble − student, (b) best single − cross-fitted oracle blend, (c) score vs data size.
**Test.** (a), (b) zero-fit from existing valid predictions. (c) since32 recipe on 50% and 75%
of the fitting sequences (whole chain components, fixed random subset, seed 203301): **2 fits**,
plus the existing 100% cell. **Decision.** A steep slope (≥ +0.002 per doubling) says the
train+valid refit (+18% sequences) and any augmentation matter more than architecture; a flat
slope says spend on teachers and representation. **Falsifier.** All three gaps < 0.002.

## C02. Two-clock GRU: a slow cell that updates only on book events

**Why.** Labels live on an event clock (E-079: class-3 label = mean mid over the next four
book events); since32's gain says rows between events are mostly label noise. Event-triggered
control and Phased LSTM (arXiv 1610.09513) update an estimator only on triggers. **Change.** A
fast per-row GRU feeds a slow GRU cell whose update is gated by a causal i0 book-event
indicator (soft gate initialized near 1 on events, near 0 otherwise); the head reads both.
**Control.** since32. **Falsifier.** No gain over since32, or the learned gate ignores events.
Differs from batch 2 B06 (GRU-D decay of one cell).

## C03. Event-row training with zero-order hold at inference

**Why.** Grok (Oct 29, card 2): if the target is held between events, a model trained and run
on event rows only, with its last prediction held in between, concentrates capacity where the
label changes and shortens BPTT to the event count. H-326 refuted holding a *per-row-trained*
model's output; this tests a model trained for the hold. **Change.** Loss on rows with since = 0
only (i0 book event rows); the cell steps on every row (state must still see all inputs) but the
output is held from the last event row. **Control.** since32 (same step budget). **Falsifier.**
WP ≤ since32 − 0.002, or ρ to since32 > 0.995 with ΔWP < 0.001.

## C04. Frozen-trunk switching readouts (zero backbone fits; precheck for #82 card 09)

**Why.** One linear head serves four classes with WP 0.81/0.78/0.74/0.44. **Test.** One forward
of the since32 E6 model over train and valid; save h_t. Fit four |y|-weighted ridge heads on
**train** h_t: (i) oracle routing by true class (upper bound), (ii) a causal class rule (attainable
bound). Score on valid. **Decision.** Oracle Δ < 0.002 closes class-routed readouts (and lowers
#82 card 09); ≥ 0.002 justifies the trained version. Also report within-class kNN on a PCA of
h_t: kNN ≈ net WP means the class-3 residual is aleatoric in this representation. **Cost.** CPU
ridge on a dump; zero fits.

## C05. Residual-boosting small GRU

**Why.** Distillation copies the teacher; a student in the blend slot moved public only ≈ 0.0006.
A small GRU trained on the residual `y − p_since32` is one boosting step aimed at what the main
model misses; a held-out line search sets its weight to 0 if the residual is unlearnable.
**Change.** GRU H64×1, same WMSE and clip, target = residual of the frozen since32 model on train
(in-sample residuals are optimistic; use 2-fold cross-fitting on train, so **2 fits of the
small GRU** per seed fold, cheap). Line-search α on a train inner fold; never on valid.
**Falsifier.** α ≈ 0 or valid Δ < 0.0005.

## C06. Shared pair encoder for i0/i1 channel correspondences, with raw bypass

**Why.** 52 inputs have proposed i0↔i1 counterparts. A shared small map per pair plus pooling
(Deep Sets) gives a cheap cross-instrument interaction basis while the raw 112 inputs stay as a
bypass. First proposed Oct 23 (H-395), never run. **Arms.** Proposed map vs a deranged pairing of
the same size (controls for added parameters). **Fits.** 2 per arm = **4**, control since32
reused. **Falsifier.** True map ≤ deranged map or ≤ since32. Note: adding i1 lead inputs
directly hurt (−0.0049/−0.0016, Oct 25); the bypass and pooling are the difference.

## C07. TensorSketch of same-row channel interactions

**Why.** A GRU must learn instantaneous pairwise interactions from hidden capacity; a fixed
32/64-dim degree-two TensorSketch (Pham & Pagh, 2013) appends a cheap interaction basis without
6,216 explicit pairs. Proposed Oct 23 (H-397), never run. **Change.** Append a seeded,
hashed 64-dim sketch of the standardized row; raw path unchanged. **Control.** since32.
**Falsifier.** No gain, or the gain flips under a second sketch seed.

## C08. Delayed-label test-time training: ceiling first

**Why.** In Jane Street's real-time forecasting competition, one SGD step per day on newly
revealed labels gave the winning-class GRU +0.008 CV. Here labels are never revealed, but our
E-078 showed that knowing true labels with a 50-row lag would add +0.011 via simple
recalibration, while class-3 labels rebuilt causally (90% exact, 4-event lag) **hurt** (E-080).
**Test (zero-fit).** (i) Replay E-078 with *head-only SGD* instead of recalibration on true labels
at lags 50/200/1000; (ii) the same with rebuilt labels for all classes. **Decision.** Only if (ii)
is positive is a serving implementation worth building (CPU budget: 42 min portal limit for a
3-member blend). **Falsifier.** (ii) ≤ 0, which we expect given E-080; the card records why online
learning does not transfer here unless label reconstruction improves.

---

## Budget

C01 2 fits, C02 2, C03 2, C05 2–4 small, C06 4, C07 2: **14–16 new fits**; C04 and C08 zero-fit.
Run order by information per GPU-hour: C04 and C08 (zero-fit), then C01, then the event-clock
pair C02/C03, then C05–C07.
