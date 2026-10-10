Longer training is puzzling — I'll dig into your pipeline and curves to design the run set.# Round 4: eight long-run regimes (10–30 epochs)

Baseline you are beating on fold 1: `0.641` (ep0-p123) → `0.653` (ep0) → `0.660` (ep1) → `0.663` (ep2) → `0.6671` (ep3 peak) → `0.662` (ep4) → `0.649` (ep5). Mid-epoch ≈ epoch-end (±0.001). So epoch 3 is a cliff, not a plateau: train loss keeps falling while WP falls 0.005 then 0.013. That plus the facts — 20k highly-autocorrelated rows per sequence (same sequences repeated every epoch), multi-rate clocks (most rows are stale repeats: blocks update on 12/15/31/40/70/100% of rows), label piecewise-constant (mean run 9.5 rows) defined by a **long future path** — says the mechanism is **sequence-fingerprint memorisation**: the slow GRU state learns "which stream/chunk this is" and its label mean, which minimises train WMSE but destroys weighted-Pearson on disjoint components. A teammate's "more epochs = better" is only compatible with your curve if their regime suppresses the fingerprint path. All eight runs below are fingerprint suppressors, not generic "tune lr/wd".

Overview:

| # | name | epochs | core trick | P(>0.670 fold-1) |
|---|---|---|---|---|
| 1 | WSD anchor | 20 | low lr + high wd + WSD + EMA | ~0.25 |
| 2 | Muon-everywhere | 20 | Muon on proj+GRU+head, WSD-sqrt, dropout 0.15, SWA | ~0.20 |
| 3 | Noisy short-chunk | 15 | chunk 250, batch 64, constant+cooldown | ~0.20 |
| 4 | Stale-row thinning | 18 | drop no-event rows + block-drop, WSD | ~0.28 |
| 5 | ★ Amnesia + shrink-perturb | 20 | inter-chunk state noise + 5-epoch shrink-restart | ~0.15 |
| 6 | ★ Self-distil long tail | 20 | freeze nothing; blend in own ep3 predictions α=0.5 after ep6 | ~0.18 |
| 7 | ★ Timescale cycling | 20 | chunk 500↔2000 alternation + LR sawtooth | ~0.15 |
| 8 | Constant-LR SWA harvest | 20 | constant lr, SWA from ep8, endpoint ignored | ~0.22 |

Common fixed settings unless stated: same 2×256 GRU, same 211 inputs, same WMSE (|clip(y,2)| weights) + aux 0.3/0.3, same propensity/mask weighting, fp16+GradScaler, seed s3 (match `r3_ctrl_f1_s3_e3`), save 3 snapshots/epoch and score them offline exactly like `learning_curve_6epochs.jsonl`. All probabilities are single-run, single-fold P(WP>0.670); 0.670 is +0.0036 over the 4-run default mean (0.6664) and +0.0029 over your ep3 peak, i.e. ~1.5–1.8σ of the one-run one-fold comparison sd (~0.0022). Nothing here is >0.35. Value is information, not P alone.

---

## 1. WSD low-LR anchor (the Rosetta Stone)

**Exact:** `MuonAdamW` split as today (Muon on `weight_ih/hh` only) but lr **4e-4** both groups (Muon mult 1.0), wd **0.08** (decoupled, applied to AdamW group + Muon), schedule **WSD: 1-epoch linear warmup → stable 4e-4 to epoch 16 → linear decay to 4e-6 over epochs 16–20**, batch 32, **chunk 1000** (unchanged → 7,460 steps/epoch, ~150k steps total), dropout **0.1** (embedding+GRU variational, same mask across chunk), no augmentation, **EMA decay 0.999** scored alongside raw endpoint.
**Why on THIS data:** your one-cycle keeps LR near max exactly at epochs 3–4 where the cliff happens — maximum plasticity at maximum repetition. WSD holds a low stable LR through the repeated exposures so the fast fingerprint fit cannot spike, and the short decay isolates "did stability help" from "did decay position help". High wd specifically punishes large identity-mapping weights the fingerprint needs; dropout 0.1 is the smallest dose that breaks exact stale-row-repeat matching. This is the run that tells you whether the teammate result is just "lower + longer".
**Kill signature:** if snapshot WP has not exceeded 0.666 by end of ep6 **and** last 3 snapshots are monotonically down while train loss still falls >2%/epoch → stop; fingerprint won. Also kill if ep2–ep4 WP < your 6-epoch curve at same epoch by >0.004 (under-plastic, lr too low — raise, don't extend).
**P(>0.670): ~0.25.** Highest of the conventional set because it changes the failure phase, not just the peak.

## 2. Muon-everywhere + WSD-sqrt + SWA

**Exact:** Muon on **every 2-D weight** (input `Linear 211→256`, both GRU layers, head `512→256` and `256→2`; 5 NS steps, bf16) + AdamW on biases/vecs only; lr **5e-4 Muon / 3e-4 AdamW**, wd **0.05**, schedule **WSD-sqrt: 1-ep warmup, stable to ep15, sqrt-decay to 0 over ep15–20**, batch 32, chunk 1000, dropout **0.15**, GradScaler on, **SWA: average snapshots from ep12 onward (every 248 batches), score SWA separately**.
**Why:** your current split leaves proj+head on AdamW — exactly where a linear sequence-ID readout can live cheaply while Muon regularises only the recurrence. Teammate "more epochs better" is what Muon-everywhere often produces (orthogonalised updates resist rank-1 fingerprint growth in all matrices). Tests whether overfit lives in proj/head, not GRU.
**Kill signature:** if SWA trail is *below* raw endpoint at ep12–ep14 by >0.002 (SWA normally above endpoint in overfit regimes) → the run is underfitting, not overfitting; stop and redeploy budget to #4/#5. If raw WP drops >0.005 ep10→ep14 with no SWA recovery → stop.
**P(>0.670): ~0.20.**

## 3. Noisy short-chunk (4× steps, 1/4 horizon)

**Exact:** AdamW all-params (control for Muon confound) lr **3e-4**, wd 0.05, **constant 3e-4 to ep13 → linear cooldown to 3e-6 ep13–15**, **batch 64 seqs, chunk 250** (≈160 steps/seq, ~4× optimizer steps per epoch vs today; truncated BPTT horizon 250 not 1000), dropout 0.1, no augmentation, EMA 0.999.
**Why:** an epoch today is only 373 batches of the *same* 20k-row tapes — few noisy steps, easy to memorise. Chunk 250 quadruples steps/epoch and shortens the gradient horizon below the stale-repeat timescale, so the net cannot use a 1000-row exact-match template; each step is noisier (short chunk, larger batch partially compensates). Directly tests "data economy, not optimisation" from Round-3 notes: if step-count/noise is the cure, this wins without any new code beyond chunk/batch.
**Kill signature:** if ep0–ep2 WP trails the 6-epoch curve by >0.006 at same step-count (not same epoch) → horizon too short for the future-path label; stop. If train loss oscillates >10% batch-to-batch with WP flat <0.660 to ep8 → noise too high; stop.
**P(>0.670): ~0.20.**

## 4. Stale-row thinning + block-drop (clock-invariant training)

**Exact:** Muon-RNN (as today) lr **5e-4**, AdamW rest 5e-4, wd 0.05, **WSD linear (warmup 1, stable to ep14, decay ep14–18)**, batch 32, chunk 1000, dropout 0.1, **augmentation: (a) thin stale rows — independently drop each row with no price/volume/trade event w.p. 0.25, carry hidden state across the gap, keep surviving rows' labels/targets unchanged, never drop rows ≤2 after a label change; (b) block-drop — with p=0.1 per sequence zero a whole update-block (one instrument's price *or* volume block) for the chunk**, EMA 0.999.
**Why:** multi-rate clocks mean most rows are bit-identical repeats; the fingerprint is literally "count the repeats". Thinning destroys repeat-count identity while preserving every event and every label change (label-preserving by construction, cf. R3-5); block-drop forces cross-instrument redundancy. This is the augmentation analogue of your chunk-length observation, aimed at the exact spurious feature.
**Kill signature:** if ep0 WP < 0.640 (vs 0.653 today at ep0) → thinning too aggressive, broke causality/labels; stop and halve thin prob. If WP peaks ≤ep4 then falls with the same slope as the 6-epoch cliff → augmentation insufficient; stop by ep10.
**P(>0.670): ~0.28.** Highest single-run P in my list: attacks the memorised feature, not the optimiser.

## 5. ★ Inter-chunk amnesia + shrink-and-perturb (unconventional)

**Exact:** same optimiser/lr/wd/schedule as #1 (lr 4e-4, wd 0.08, WSD-20) so #1 is the clean control; **plus: (a) add ε~N(0, 0.04²) to the detached carried hidden state at every chunk boundary (both layers), scaled by state RMS (i.e. 4% state noise); (b) every 5 epochs: `w ← 0.85·w + 0.15·w_init_random` (shrink toward fresh init, not zero) then 0.5-epoch LR re-warmup to stable value**. Batch 32, chunk 1000, dropout 0.05 only, no other augmentation, score pre- and post-perturb snapshots.
**Why:** the fingerprint must survive the chunk boundary in the slow hidden state across 20 chunks/sequence. 4% carry noise is negligible for genuine forecasting features (they re-establish from events within tens of rows) but fatal to a 20,000-row identity code. Shrink-perturb periodically deletes late-epoch memorisation while keeping the general forecaster — the long-run version of "epoch 3 was good, epoch 5 memorised". If this curve rises after each perturb while #1 falls, you have proven where the memory lives.
**Kill signature:** sawtooth with downward envelope: post-perturb WP recovers less than 50% of the drop for two cycles in a row → noise is destroying signal too; stop. If WP never reaches 0.660 by ep8 → noise too high from the start; stop.
**P(>0.670): ~0.15.** Low P, very high diagnostic value.

## 6. ★ Self-distillation long tail (unconventional)

**Exact:** epochs 0–6 exactly default recipe (Muon-RNN, lr 1e-3, wd 0.01, constant — *not* one-cycle — to ep6); snapshot ep3 (`own-teacher`). **Epochs 7–20: target = 0.5·true label + 0.5·own-ep3-prediction (frozen, precomputed cache, same WMSE weights), lr cut to 3e-4, wd 0.05, WSD-stable→decay ep16–20**, batch 32, chunk 1000, dropout 0.1, EMA 0.999. No external teachers.
**Why:** your student gains +0.003 from external-teacher blending; this asks whether the gain is "teacher knowledge" or just "stop fitting the raw label". Blending in your *own* ep3 predictions after ep6 caps how hard the net can chase idiosyncratic label noise on repeated tapes — a Born-again regulariser that costs no new families. On piecewise-constant labels with 9.5-row runs it specifically prevents sharpening onto exact change-point rows of the train tapes. Distinguishes "teammate trains longer because their targets are smoother" from optimiser stories.
**Kill signature:** if WP at ep7–ep8 (just after blend switch) drops >0.004 below ep6 and does not recover by ep10 → blend collapsed plasticity (α too high); stop. If curve reproduces the ep3→ep5 cliff shape delayed by ~4 epochs → self-targets don't substitute for independent teachers; stop at the second consecutive drop.
**P(>0.670): ~0.18.**

## 7. ★ Timescale cycling (unconventional)

**Exact:** Muon-RNN lr **6e-4 base**, wd 0.05, **sawtooth: 2-epoch cosine cycles (warm restart to 6e-4, decay to 1e-4) ×10 cycles = 20 epochs**, **chunk alternates 500 (even cycles) ↔ 2000 (odd cycles)**, batch 32, dropout 0.1, SWA over cycle-ends only.
**Why:** chunk length sets both step-count and BPTT horizon; every fixed choice bakes in one truncation artifact the fingerprint can exploit (1000-row template). Alternating horizons forces the recurrence to be consistent across timescales — a cheap timescale-invariance prior matched to "label depends on a long future path" plus multi-rate clocks. Restarts prevent settling into the ep3 memorisation basin. If cycle-end SWA beats every within-cycle endpoint, the served answer is "average over horizons", directly servable in one net.
**Kill signature:** if odd (2000) and even (500) cycle-ends diverge by >0.006 persistently → horizons learn incompatible solutions, averaging is invalid; stop. If no cycle-end exceeds 0.665 by cycle 4 (ep8) → cycling prevents any fit; stop.
**P(>0.670): ~0.15.** Riskiest, but the only run that tests horizon-artifact memorisation.

## 8. Constant-LR SWA harvest (endpoint will overfit; ship the average)

**Exact:** AdamW all-params, lr **constant 4e-4 all 20 epochs (no decay at all)**, wd 0.05, batch 32, chunk 1000, dropout 0.1, **SWA from ep8: average every epoch-end; also keep EMA 0.9995; the scored artifact is SWA, endpoint is diagnostic only**.
**Why:** mimics the most likely boring explanation of the teammate claim: they kept LR up, endpoints overfit like yours, but their *averaged/selected* checkpoint improves with epochs. Constant LR + late SWA tests "more epochs = better posterior mean even as MAP collapses". If SWA rises ep8→ep18 while endpoint falls ep3→ep5 exactly like your curve, you have reconciled both observations for ~1 line of code and the long-run policy becomes "never ship the endpoint".
**Kill signature:** if SWA < endpoint at ep10 **and** ep12 (averaging hurts even mid-run) → basin is non-convex at this LR; stop. If endpoint diverges (NaN/GradScaler skip-rate >5%/epoch) → LR too high for constant; stop.
**P(>0.670): ~0.22.**

---

## Start first with #4 (stale-row thinning); run #1 in parallel as its control

If only one GPU: **start #4**. It has the highest P(~0.28), directly exploits your two strongest facts (multi-rate stale repeats + repeat-the-same-tapes memorisation), is label-preserving so it cannot be dismissed as "changed the task", and its kill signature is visible by epoch 4 (compare ep0 WP to 0.653: <0.640 = bug, not result). Pair it with #1 at the first free slot — #1 is the only run that makes #2–#8 interpretable (without the WSD anchor you cannot tell whether any win came from stability, Muon scope, steps, augmentation, or averaging). Do not start #5/#6/#7 before #1+#4 have passed epoch 6; they are conditioned on what those two reveal.
