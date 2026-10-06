# Slow/fast decomposition and test-composition probes (2026-10-02)

Public: e6a (V1 + wm + ow, 6 members) = 0.6641 in 36m31s == V1. Ensemble breadth saturated on public too.

decompose.py (d256 fold OOF, 936 seqs): base 0.6906; within-class WP c0 .805 c1 .777 c2 .734 c3 .455.
Seq-level weighted means: pred vs target corr 0.88/0.90; between-seq share of weighted target variance ~0.19.
means2.py: class-level target means shift between blocks (class 3 wmean train -1.26 vs valid -0.85).
offsets.py / offsets_full.py: oracle per-class offset/affine recalibration headroom on valid: +0.0002..0.0004,
also for the TRAIN-ONLY model (toref). REFUTED: "test loss comes from per-class level miscalibration".
mix.py: d256 local WP valid-mix 0.6906, train-mix 0.6468, class3 x1.5 0.6608. Public 0.664 ~ class-3 share ~0.46 if no difficulty shift.
probe_table.py: expected public of full / c3zero / c3only by class-3 share (rest in valid proportions):
 share .37: .6915 .5538 .4253 | .45: .6674 .5094 .4419 | .55: .6357 .4535 .4550
LB probes built on midpoint members + detector (99.8% acc): ../2026-10-02-big-ensemble/probe/p_c3zero.zip, p_c3only.zip.
Compare against midpoint public 0.6639.

## Public probe results (2026-10-02, midpoint members)
full 0.6639 | c3zero 0.5473 | c3only 0.4017.  Local valid-mix (d256 OOF): 0.6906 | 0.5522 | 0.4260.
Gap valid->public: classes 0-2 -0.005, class 3 -0.024: the transfer loss is concentrated in class 3.
invert.py exact fit: share3 0.556, k012 1.23, k3 0.81. alpha.py: under the exact fit the class-3 output scale optimum is ~0.7-0.8
(predicted 0.672 at 0.7); misfit alternatives (residual ~0.004) put the optimum near 0.9-1.0. Two-block analytic form has two
branches (alpha* ~0.65 or ~1.6). Direct LB probes built: probe/p_c3x070, x085, x055, x120 (verified scale applied only on class 3).

## Probe p_c3x070 (class-3 outputs x0.7): public 0.6556 (-0.0083). REFUTES the exact-fit prediction (0.672).
Quadratic through alpha 0/0.7/1 (0.5473/0.6556/0.6639) peaks at alpha~0.96 (0.6641): class-3 scale is already ~optimal on test.
The test loss is within-class-3 ranking quality, consistent with the quiet-regime finding. No further scale probes.

## Quiet-regime screens (h128 fold recipe, fold 0+1, tailrank.py)
| run | all | c012 | c3 | c3quiet | c3active |
|---|---|---|---|---|---|
| qw0 ctrl s3 | .6879 | .7924 | .4502 | .3992 | .5631 |
| qw0s4 ctrl s4 | .6887 | .7935 | .4510 | .3990 | .5632 |
| qw1 beta1 | .6874 | .7919 | .4495 | .3997 | .5593 |
| qw2 beta2 | .6878 | .7917 | .4512 | .4038 | .5568 |
| adrop (a0..a7 zeroed) | .6870 | .7913 | .4494 | .4027 | .5575 |
| adem (a minus causal running mean) s3 | .6879 | .7919 | .4514 | .4020 | .5615 |
| adems4 s4 | .6888 | .7934 | .4514 | .4047 | .5614 |
Seed noise on c3quiet for the control is 0.0002; demean replicates (+0.0028/+0.0057 quiet, all-neutral). SUPPORTED (2 seeds, 2 folds).
horizon.py: in quiet c3 scored rows sit ~25 rows before the next target change vs ~9 in active: part of the gap is intrinsic.
position.py: no WP decay along the sequence for our models (c012 flat ~.795); competitor "state reset" lever does not apply.
Scouts: public notes of #11 (0.6569) and 0.6581 competitors; nothing from leaders. Possible hard timeout ~70 min.
Next: final-data demean h128 members (final_dm.sh: findm128_s3/s4, findmc500_s3 + h2), package = midpoint with the three h128 members swapped (solution_dm.py), public contrast vs midpoint 0.6639.

## mid_dm3 package (2026-10-02 19:30)
Members: wm_s3h2 (raw view, 1/4) + findm128_s3h2, findm128_s4h2, findmc500_s3h2 (demean view, 1/4 each; final_dm.sh, epoch-1 H2 checkpoints).
Parity engine(python demean)+int8 vs torch findm128_s3h2: corr 0.99996/0.99989, max|d| 0.06; negative control without demean corr 0.9992/0.9997, max|d| 0.19.
Harness 54.9 local-min (midpoint 38.5 -> 21.1 public) => ~30 min projected. Deterministic. Public contrast: midpoint 0.6639.

## Closed today without compute (prior evidence)
- Online per-sequence recalibration from rebuilt matured labels: E-080 (class 3 rebuilt 90.5% exact) hurt 0.672 -> 0.647. Competitor "+0.047 oracle per-seq calibration" is in-sample; no causal version.
- Disagreement shrinkage (disagree.py, 6 cached members): null (members corr 0.98-0.99, sd/|mu| 7%).
## Running
- auxn1 (--aux-next 1.0), auxk (--aux-next-k 2,4), both demean, fold 0/1 seed 3; control adem.
- qexp: class-3-only quiet expert (seqw_q3: exp(-3z), non-c3 weight 0), demean. Gate: c3quiet >= adem + 0.01.
- RESULT aux arms vs adem (s3): auxn1 all .6882 c3quiet .4009 (-0.0011); auxk all .6880 c3quiet .3966 (-0.0054). REFUTED: far-horizon aux targets do not help quiet class 3.
- RESULT qexp (class-3-only, quiet x20, demean): c3 .3521, c3quiet .2615 (adem .4020), c3active .5114. REFUTED strongly: specialization loses the shared representation learned from classes 0-2; class 3 is data/representation-limited, not capacity-limited.

## Data scaling + augmentation (2026-10-02 night)
- half (train-half 0, demean): all .6843 c3 .4425 c3quiet .3892 (vs adem .6879/.4514/.4020): class 3 is data-limited (~-0.009 c3 per halving).
- c3only (train class 3 + all valid folds, demean): c3 .4469 c3quiet .3921: classes 0-2 data helps class 3.
- Mirror by bid/ask block swap (gateA.py): NOT a mirror (corr(f(Mx)t0,f(x)t1) ~0). True mirror needs code decoding (deferred).
- i1 role swap (scripts/i1swap_aug.py, cache/i1swap_train_*.npy via build_i1swap_cache.py): in class-3 seqs i1 shares the i0 code table
  (82% transitions known); i1 is a quiet large-tick instrument (price events 6.5% rows vs 38.5%). Label rebuild exact 90% on i0; reliable rows on i1 54%.
  i1sA (+4557 swapped seqs): all .6848 c3 .4469 c3quiet .3925; i1sB (equal steps): .6817/.4413/.3833 vs ref .6884/.4512/.3990. HURTS.
  Running: i1sF = i1sA + --domain-flag (extra input 1 on swapped seqs, 0 at test).
- i1sF (i1sA + --domain-flag): all .6851 c3 .4463 c3quiet .3923: flag does not rescue.
- i0rb (same 4557 class-3 train seqs, NO swap, labels replaced by rebuilt ones (90% exact), reliable-mask weights): all .6822 c3 .4434 c3quiet .3905. Also HURTS.
  => the harm is not i1-domain specific: either 10% label noise or the reliable-mask weighting (uniform over reliable rows instead of propensity^2 toward scored-like rows). Not separated.
  Implication for a true mirror: it would keep exact labels (t0<->t1 swap) and could keep propensity weighting; the i1/i0rb failures do not refute it.
- Event clock (--arch dual, demean): evdual all .6877 c3 .4483 c3quiet .3946; evonly (--no-row-gru) .6802/.4369/.3859, quiet-minus-c3 gap -0.051 unchanged.
  REFUTED: the quiet-regime loss is not a row-clock vs event-clock representation problem.
## Synthesis (2026-10-03)
Five mechanism families failed to lift quiet class 3 beyond ~+0.005: reweighting (capacity trade), specialization (expert -0.14),
far-horizon aux targets, label-synthesizing augmentation (i1 swap / rebuilt labels), event-clock architectures. Only demean (+0.003..0.006 quiet, neutral overall) survives.
Data scaling is real (train-half -0.009 c3), but every synthetic-label route hurts. Best reading: quiet class 3 is information-limited for causal models.
- level_oracle.py: oracle sequence level (scored-row weighted mean of y) would add +0.031 overall, +0.082 on class 3.
  level_persist.py: the residual level is NOT persistent (corr 1st vs 2nd half 0..0.1; past-2000 vs next-2000 on c3 0.05-0.15):
  the oracle gain is unpredictable within-sequence drift; consistent with E-080. CLOSED.

## v1_dm / v1_dm5 (2026-10-03)
v1_dm = findm256_s3h2 + findm128_s3h2 + findm128_s4h2 + findmc500_s3h2 (all demean, 1/4): V1 recipe with demean. Harness 62.7 local-min (~26-35 public).
v1_dm5 = + findm256_s4h2 (1/5 each). Harness 95.3 local-min (~40-52 public). findm256 parity vs torch corr 0.99995. Both deterministic.
Public contrast: v1_dm vs V1 0.6641 (and v1par).
- d256dm (fold d256 recipe + demean, seed 3): all .6895 c3 .4535 c3quiet .3995 vs d256 .6906/.4554/.4048 (d256s4 .6911/.4561/.4071).
  demean HURTS the distilled 2x256 member (quiet -0.0053, beyond seed noise ~0.0023). demean is h128-specific, NOT universal.
  => v1_dm / v1_dm5 (demean d256) deprioritized; keep raw-view d256 in packages.
- i0ty (same 4557 class-3 dup seqs, TRUE labels, same reliable-row weighting): all .6872 c012 .7887 c3 .4543 c3quiet .4047.
  vs i0rb (rebuilt labels) c3 .4434: rebuilt-label noise costs -0.011 on c3; the weighting is not the culprit. Exact extra class-3 data: c3 +0.003, quiet +0.006, c012 -0.004 (class-share shift) -> all -0.0012.
- fail_map.py: rebuild errors 9% of rows but 25% of metric weight; concentrated where the 4th next event is >=32 rows away (15-20% vs 1-3%).
- law_clock.py: far rows: 4-event law 0.818 match, best clock-based definition 0.563 -> labels ARE event-based. law_k.py: failures not explained by K-event truncation (any K: 20%).
  hidden_shift.py: no label changes without visible code events (no hidden level shifts). Far-horizon law residual unexplained (real-time window or move-table errors). PARKED: low value (exact extra c3 data nets ~0 without class balancing).

## Valid FT sweep on d256 fold parents (ft_sweep.sh), fold 0+1
parent .6906 (c3 .4554) | ftA H2 head 2ep lr5e-4 .6907 (.4526) | ftB head 4ep .6902 | ftC H2+full 1ep 5e-5 .6901 | ftD full 2ep 1e-4 .6910 (.4520).
All within +-0.0005; FT moves quality from class 3 to classes 0-2 locally; public showed FT +0.0011. Keep standard H2.
## Mixed-view ensemble (ens_mix.py, fold OOF)
raw4 .6907 (quiet .4047) | dm4 (h128 demean) .6908 (.4071) | dm4+d256s4 .6912 (.4077) <- best.
## FINAL candidate 2026-10-03: final_mix5 = find256_s3h2 (raw) + find256r_s4h2 (raw, new) + findm128_s3h2 + findm128_s4h2 + findmc500_s3h2 (demean), 1/5 each.
Rationale: d256 without demean (demean hurts d256 locally), h128 with demean (helps locally, 2 seeds), second d256 seed (fold analog best). Expected public ~0.6641-0.6646.
final_mix5.zip built (sha256 f92aacba02f705d9...): new member find256r_s4h2 parity vs torch corr 0.99993-0.99997; full package equals weighted member sum (2.4e-7);
deterministic; harness 99.0 local-min (v1_dm5 95.3 -> 39m12s public) => ~41 min projected. Copied to Mac submissions/.
PUBLIC final_mix5: 0.6642 (38m47s) — new best (+0.0001 vs V1 0.6641), inside the predicted 0.6641-0.6646 band.

## Day 2026-10-03 local campaign
- C noisy student on i1 (pseudo_i1.py, --i1swap-pseudo): falsified before training: fold teachers d256 reach only WP 0.21 vs rebuilt labels on reliable rows of swapped i1 (vs ~0.45 on i0 class 3). i1 is a different domain; teacher soft labels too weak. CLOSED.
- Running: A dropout 0.1/0.2 (drop1, drop2), B distillation h128 student from 2 vs 4 teachers (dist2, dist4).
- A dropout: drop1 .6882, drop2 .6885 vs ref .6884 (refs4 .6893). B distillation teachers: dist2 .6887, dist4 .6892. All within seed noise (0.0009). CLOSED.
- sumdiff.py: t0+t1 is NOT -spread when F leaves the quotes (sum = unsigned move magnitude); corr(sum) .51 all, .06 class 3; oracle-sum +0.032, oracle-diff -> 0.976.
  The direction (t0-t1) is the crux; the magnitude is unpredictable on class 3 (MSE already yields E[sum|x]). No lever.
- ftT (head FT with gcorr on train+valid folds, propensity weights): all .6837 (-0.007 vs parent), c012 .7851. The FT gain comes from VALID DATA (true scoring masks), not from the gcorr loss. Supports valid-bagging (kf_*) LB tests.
- ftNM (valid head FT WITHOUT mask weighting): all .6901 c012 .7941 vs ftA (with true mask) .6907/.7959, parent .6906/.7946.
  The local FT gain on classes 0-2 needs the true scoring mask. Mechanism: training exactly on scored rows (only valid has them).
## Nested simulation: k-fold valid bagging vs full valid (nested.sh, nested_eval.py, h128, pseudo-test folds 3 and 2)
T3: full1 .6991 | full 2 seeds .7009 | nested members .6993-.6998 | k-fold avg3 .7004.  T2: full1 .6773 | full2 .6788 | members .6763-.6769 | k-fold avg3 .6775.
Members trained on less valid = full-valid single; k-fold average gains only via ensembling and loses to a 2-seed full-valid ensemble on both pseudo-tests.
REFUTED: valid bagging. kf_* LB pairs not worth submissions.
- Muon (hybrid, match_rms_adamw, lr mult 1, momentum 0.95): all .6877 c3 .4499 c3quiet .3955 vs ref .6884/.4512/.3990: slightly worse. Running muon2 (lr mult 2).
- HP sweep (h128 fold recipe): lr2e-3 .6890, beta2 .99 .6888, clip .3 .6882, warmup 20% .6880 (all within seed noise vs ref .6884 / refs4 .6893);
  batch 64 .6858 and lr 5e-4 .6860 clearly worse. Current HPs are at the optimum. CLOSED.
- muon2 (Muon lr mult 2, seed 3): all .6897 c012 .7936 c3 .4538 c3quiet .4008 vs ref .6884/.4512/.3990: +0.0013 (> seed noise 0.0009). Replicating (seed 4) and lr x3/x4.
- Muon scan: muon2s4 .6898 (vs refs4 .6893, c3 .4540 vs .4532); muon3 .6890; muon4 .6906 (c3 .4563, +0.0051 vs ref). Small consistent gain, noisy in lr. Replicating x4 on seed 4 + x6.
- ens_muon.py: muon2 pair .6907 (c3 .4554) vs adam pair .6896 (.4532): +0.0011; d256 + muon pair .6914 vs d256 + adam pair .6908. Mixing adam+muon adds no diversity.
- MUON CONFIRMED: h128 muon4 s3 .6906 / s4 .6906 (vs .6884/.6893: +0.0022/+0.0013); muon6 s3 .6909 (c3 .4572, c3quiet .4090 = +0.010);
  d256mu (x2) .6920 vs d256 .6906 (+0.0014). Gain grows with Muon lr. Scanning x8/x12 (h128), x6 seed 4, d256 x4/x6.
- MUON SCAN DONE: h128 x6 s3 .6909 / s4 .6908, x8 s3 .6910, x12 diverged (NaN). d256 x2 .6920, x4 .6911, x6 .6921 (c3 .4590, quiet .4114). Chosen: Muon x6 for all members.
- bip.py v1 (pure bidirectional) undertrained: AUC .878 vs causal .922; v2 = causal logit + backward correction (zero init), rerunning.
- MASK SMOOTHER (bip.py v2: frozen causal prop + backward-GRU correction, zero init): fold 0 AUC .9218 vs causal .9217, Brier .0515 vs .0521 after 4 epochs.
  Future rows add ~nothing to scoring-mask prediction. REFUTED; stopped.
- GLOBAL HEAD (gh_extract.py + gh_fit.py): joint weighted-ridge readout over 5 fold members (897 dims) toward the equal-weight ensemble, chain-half cross-fit:
  mean delta lam1e3 -0.025, 1e4 -0.017, 1e5 -0.008, 1e6 -0.002; never positive on average. REFUTED: readout refit on valid chains does not transfer across chains.
## final_mu5.zip (2026-10-03 evening): fmu256_s3h2 + fmu256_s4h2 (distilled d256, Muon x6) + fmu128_s3h2 + fmu128_s4h2 + fmuc500_s3h2 (Muon x6), all raw, 1/5 each;
parents Muon x6 (train+valid), H2 head FT AdamW on valid with true mask. Parity corr .99985-.99994; equals weighted member sum (2.4e-7); deterministic; harness 84.4 local-min (~35 public).
Public contrast: final_mix5 0.6642 (same roles, AdamW parents, demean h128).
- Muon refinements: 6 epochs .6879 (worse, overfits); wd 0.05 .6918 (+0.0009 vs muon6, c012 up, quiet down: noise-level); momentum flag is a NO-OP
  (OneCycleLR cycle_momentum overwrites momentum/beta1 every step for all runs; consistent across arms).
- Muon round 4: wd .05 s3 .6918 / s4 .6916 (+.0008 all, c012 +.0015, c3quiet -.002/-.005: trade-off, keep wd .01); muon on inp .6907 (no gain); d256 x8 .6001 (diverged -> x6 is the edge).
  final_mu5 config (x6, wd .01, inp AdamW) confirmed.
- DATA SELECTION by gradient alignment (ds_score.py, muon6 fold models; cos with true-mask valid gradient: median .39-.42, only ~5% negative):
  dsU (uniform sampler control) .6866 | dsSoft (rank weights .25-1.75) .6847 | dsDrop20 .6832. REFUTED (both below matched control).
  Side finding: sampling with replacement costs -0.004 with Muon (dsU .6866 vs shuffle muon6 .6909); final_mu5 members use plain shuffle.
- CONTEXT DISTILLATION (ctx_teacher.py: teacher muon6_fK cold vs warm = state carried from the chain predecessor; 9191 seqs have one):
  students Muon x6 + distill alpha .5: ctxC .6915 | ctxW .6921 (+.0006). Rows 99-2000: all .6931 -> .6977 (+.0046), c012 .8014 -> .8081, c3 .4287 -> .4319; later rows equal.
  Mechanism-consistent (gain only where context is missing). Self-distillation from the Muon teacher itself +.0006 vs muon6. Replicating seed 4.
- Context distillation seed 4: ctxWs4 .6908 vs ctxCs4 .6907 (rows 99-2000: .6864 vs .6846). Mean over seeds +.0003 overall (early rows +.003): direction holds, size within noise. Self-distillation from Muon teacher: +.0006 (s3), .0000 (s4). Not worth a retrain alone.
- Queue results: muema (EMA .9995) .6879 (-.003 vs muon6); mu4k (chunk 4000) .6897 (-.0012); h256mu (non-distilled 2x256 Muon x6) .6880 < h256 AdamW .6890 -> no Muon teachers.
  final_mu5 config stays best. Pending: tfmmu vs tfmad (d128 4-layer transformer).
- Small transformer d128x4 (tfm): AdamW .6881, Muon .6897 (Muon helps +.0016) but below GRU Muon .6909; corr with muon6 .9873 (vs seed pair .9884);
  muon6+tfmmu .6918 < muon pair .6921; d256mu6+muon6+tfmmu .6927 vs +muon6s4 .6928. No ensemble value -> no C++ transformer port. CLOSED.
## LB distribution-knob pairs (2026-10-04 night), each = 2 x h128 Muon x6 full-data (s3, s4) + H2; only the train weighting differs (by repetition, plain shuffle):
pair_base (fmu128) | pair_quiet (rep_quiet: class-3 exp(-2z)) | pair_classmix (train class mix -> valid mix) | pair_advval (train-vs-valid odds^0.3, class-neutral).
All parity 0.0, deterministic, ~14-16 local-min (~6 public). Pending: P1 (prop-power 1), S0 (since-weight 0), V0 (valid-true-mask 0).
## Fold knob screens (Muon x6, seed 3, repetition weighting): muQ .6922 (c3 .4602, quiet .4155, active .5653) = +.0013 all, +.0065 quiet, no active loss;
muK .6906; muV0 .6911; muP1 .6911; muS0 .6888 (-.002); muA .6891 (-.0018). Quiet emphasis now works without trade-off (Muon + no-replacement repetition).
- Ensembles: muQ+muon6s4 .6930 vs muon pair .6921; d256mu6+muQ+muon6s4 .6935 (quiet .4155) vs d256mu6+muon pair .6928. Q adds +.0007..9 in ensembles.
  Queued: muQ3 (beta 3), muQs4 (seed 4), d256muQ (folds); final fmuQ256_s3/s4 + fmuQc500_s3 for final_muQ5.
- Q REPLICATION FAILED: muQs4 .6891 vs muon6s4 .6908 (-.0017, quiet .4017 vs .4062); d256muQ .6905 vs d256mu6 .6921 (-.0016); muQ3 (beta 3) .6885.
  muQ s3 +.0013 was seed noise. Seed noise under Muon is ~+-.0015: single-seed screens are not evidence. Q NOT recommended; pair_quiet/final_muQ5 are informational only.
## SEED NOISE CALIBRATION (h128 Muon x6, 4 seeds): .6909 .6908 .6884 .6897 -> mean .6900, sd ~.0012. AdamW seeds .6884 .6893 (mean .6889).
Muon true gain ~+.001 (not +.002; seeds 3/4 were lucky), quiet tail +.004 on average. Single-run deltas below ~.002 are noise. Consistent with public +.0002 for final_mu5.
Methodology from now: >=2 seeds per arm, compare means.
- 2-seed knob means vs base (.69085): classmix .6898 (-.0011), vtm0 .6903 (-.0005), prop1 .69055 (-.0003). No local lever; LB pairs decide.
## LB (2026-10-04): pair_base .6644 (= final_mu5!), pair_fullft .6649 (+.0005), pair_vtm0 .6637, pair_v2 .6626 (-.0018).
## Better local validation, step 1: calib_val.py: class-share + quiet-tilt weights of valid cannot reproduce public probe levels (residuals ~.008-.009, params at bounds).
Test differs in difficulty, not only composition -> need different holdout data (train-holdout).
- train-holdout (half_f0 on train half 1, propensity mask): .613 vs valid fold .684 vs public ~.664 -> public lies between; composite validation calibration running (comp_arm.sh, 6 arms x 2 seeds, eval comp_eval.py).
- Built LB pairs ffA (full FT lr 2e-4), ffB (4 ep), ffD (Muon FT) to push the full-FT direction (+.0005 on LB).
## COMPOSITE VALIDATION CALIBRATION (comp_arm.sh/comp_eval.py; parents on train half 0 + valid folds!=0, FT on valid folds!=0; eval valid fold 0 true mask + train half 1 propensity mask; 2 seeds)
| arm | public | d_valid | d_trainHO | d_composite |
| fullft | +.0005 | +.0005 | .0000 | +.0001 |
| v2 | -.0018 | -.0082 | -.0062 | -.0065 | (1 seed)
| vtm0 | -.0007 | +.0006 | -.0007 | -.0005 |
| trainonly | -.0024 | -.0005 | -.0014 | -.0012 |
| adam | -.0002 | -.0011 | -.0028 | -.0025 |
Composite gets 5/5 signs (magnitudes inflated); valid fold misses vtm0; train-HO misses fullft. Rule: accept only if neither valid fold nor train-HO is negative.
Levels: valid fold .70, train-HO .60 (models trained on half train). Vlad recipe (Muon+chunk1000+stopped bootstrap h8) ported (--distill-boot-h) and screening.
- Vlad recipe screens (d256 Muon x6): +stopped bootstrap h8 seed4 .6933 vs base s4 .6916 (+.0017; c3 +.005). +chunk1000 (+boot) seed3 .6750 (collapse; chunk 1000 doubles optimizer steps, too aggressive with our Muon x6). Waiting d256boot s3.
- STOPPED BOOTSTRAP replicated: d256boot s3 .6925 (+.0004), s4 .6933 (+.0017): mean +.0010 vs Muon d256 base. Launching full-data d256 Muon x6 + boot8 finals (s3,s4) -> full valid FT -> LB pair vs pair_fullft (.6649).
- Built LB pairs pair_boot_h2 / pair_boot_fullft / pair_boot_combo (full-data d256 Muon x6 + boot8, s3 s4). Times ~59/58/123 local-min.

### Night 10-05
- (6) seg_smooth.py: labels constant within i0-px segments (139 exceptions / 3.19M changes) but freezing output = -0.28 c3, any segment-EMA negative (a=0.5 -0.0016). Within-segment output motion is anticipation, not noise. CLOSED.
- (7) ceiling.py (d256boots4 OOF, c3 343 seqs, scored rows, 2-fold crossfit HGB): decoded price-state-only lookup WP c3 .050 / quiet .017 / active .088 vs GRU .460/.413/.567; GRU+HGB residual on state worse (-.05, overfit). Price state alone carries ~no info; Markov-Bayes per-sequence engine (agent 4 flagship) dropped.
- (12) organizer baseline (converted, parity vs ONNX corr .99998) blended into cmp_fullft: (1-w)P+w*0.25*B: w=.15 dv +.0005/+.0008 dHO +.0018/+.0018; w=.25 dv +.0005/+.0010 dHO +.0026/+.0026; w=.35 dv +.0001/+.0008 dHO +.0030/+.0031; w=.5 dv neg. scale .4 w=.25 dv +.0001/+.0008. Optimum eff. coef ~.06. STRONG, both seeds, both parts positive.
- (1) WiSE (comp): a=.5 dv -.0003/+.0003 dHO +.0023/+.0020; a=1.5 dv -.0009/-.0015 dHO -.0041/-.0038. surgA (freeze gru, lr1e-4) dv -.0002/+.0003 dHO +.0008/+.0008. lam02 dv -.0004/+.0002 dHO +.0008/+.0007.
- (9) int8 vs fp32 (pair_fullft engine, 100 valid seqs, 189k scored rows): WP .71036 vs .71041 (-.00005), corr .99999. CLOSED.
- (8) rank_split.py (300 seqs each, every 4th row): train/valid marginals NOT identical (a2 q01 -2.47 vs -2.04, i1_v1 q01 -1.96 vs -1.54, std ratio valid/train 1.05 avg) -> rank transform fit globally, not per split. Agent-3 idea 13 premise refuted. a5-a7 have sentinel -5.199 (13-43% rows).
- (10) mirror_graph.py: class-3 code graph 895 states/1594 edges (1 component); exact mirror automorphism impossible (edge-label counts differ slightly: (0,-1) 84 vs (1,0) 75); label marginals nearly symmetric. Inconclusive (empirical graph incomplete); approximate matching deferred.
- (11) i1 audit: swap_instruments leaves a0..a7 (i0-clock features) unswapped -> tested a-drop model: WP i1-swapped vs rebuilt (reliable rows, 200 seqs) qw0 .181 / adrop .174 vs i0 same seqs .298. a-feature inconsistency is NOT the cause; i1 domain gap real. CLOSED.
- (12b) baseline blend on d256 fold OOF (valid folds 0+1, pre-FT parents): boots4 w=.1 all -.0002 (c3 +.0001), w=.25 all -.0010; mu6 w=.25 all -.0005 c3 +.0006. On valid-only strong models ~0/neg; composite gain comes mostly from train-HO. Historical: E-086 public +.0021 while local holdout was negative -> local underestimates baseline. Candidate with risk.
- WiSE/baseline LB packages built+verified: pair_wise05 (8a08a8dd, 10.8 local-min), pair_fullft_base (18ebb624, 37.8), pair_wise05_base (959e87f3, 40.1); baseline member half_weights fp32 path (corr .999998 vs ONNX), pair members int8.
- (5) ffB (4ep lr1e-4): dv -.0014/-.0015 dHO -.0026/-.0027 -> pair_ffB NOT recommended. surgB (freeze gru lr3e-4): dv -.0004/-.0005 dHO -.0011/-.0013. ftsince dv 0/+.0003 dHO 0/+.0003. ffA s3 dv -.0002 dHO -.0016.
- (16) gglob995 (gcorr with running global moments, decay .995) s4: dv .0000 dHO +.0022 (like WiSE .5).
- LB projection (lstsq public ~ a*dv + b*dHO on 5 calib points, a=-.15 b=.48, poor fit): vs pair_fullft wise05 +.0010, fullft+base +.0011, wise05+base +.0018, gglob995 +.0011, surgA/lam02 +.0004. Weak guide only.
- (17) gate-wise Muon gwhh s3 .6908 (c3 .4554 quiet .4053) vs muon6 s3 .6909 -> null so far; gwall + s4 pending.
- (rawcorr) composite arm h128 --raw-only --loss mix (Muon x6) + raw FT: alone dv -.0053 dHO -.0116; as blend w=.2 into fullft: dv +.0007/+.0012 dHO +.0004/+.0007. 3-way fullft+.25*(.25 base)+.15*rawcorr: dv +.0009/+.0018 dHO +.0027/+.0029 (best). raw-only native export (zero ev cols, fold const into bias) parity corr .99995. Training full-data frawc_s3/s4 (final_raw.sh). Ablation arms rawwmse/fullmix queued.
- WiSE grid (comp): a=.25 dv -.0004 dHO +.0026; a=.5 dv .0000 dHO +.0021; a=.75 dv +.0001 dHO +.0013. gglob98 dv +.0001 dHO +.0021; gglob995 dv -.0002 dHO +.0024.
- (17) gate-wise Muon CLOSED: gwhh s3 .6908 s4 .6896 vs muon6 .6909/.6908 (avg -.0006); gwall s3 .6908 (quiet .4018 vs .4090). gwalls4 f0 OOM, not rerun.
- VLAD 10-04 late: seed-4 combo (d256 Muon chunk1000 boot8, equal parent/FT-head mix, i16) public 0.6660 (37m07s); seed-3 combo 0.6655; seed-3 parent-only 0.6641, head-only 0.6654 (mix +.0001 over head). Public seed noise single model ~.0005. Excluded-368 local seed4-seed3 +.0019 vs public +.0005. Research paused on his side.
- (13) d256bc1k (boot8 chunk1000 Muon x3) fold0 ep3 .7049 vs d256boot .7045 / boots4 .7047 / mu6 .7033 (fold-0 only): no collapse at x3, +.0003 ~ noise; f1 + s4 pending.
- VLAD s4 zip (c4dc258b) weights run exactly in our engine (fp32 corr .9999999995, int8 .99992). His holdout (sha256(seq_ix)%5==0) ∩ our folds0+1 = 180 seqs: vlad .7096, our d256boots4 OOF .7065; blends within ±.001 (vlad+pair .5 fold0 .7328 vs .7323; +base -0.001..-0.002 on valid only). Built vlad_br (c3f1e61c), vlad_pair (8f49d2c4), vlad_pair_br (4479fd54); pair_fullft_br (a2961780) / pair_wise05_br (d67567ad) 65 local-min, deterministic.
- TIMING CALIBRATION: check_e7-style projection of Vlad s4 zip = 75.2 local-min vs public 37m07 -> public ~= 0.49 x local. vlad_pair 61 (~30), vlad_br 83 (~41), vlad_pair_br 103 (~51, too risky), pair_*_br 65 (~32), pair_boot_fullft 58 (~29). Raw step: vlad in our int8 30 local-min vs his i16 77; baseline half 14.
- (14) d256bh32 (boot h32) fold0 .7041 vs h8 .7045/.7047 -> no gain (f1 pending).
- (14) CLOSED boot horizon: d256bh32 .6915 (quiet .4058) vs boot h8 .6925 / no-boot .6921 (same seed 3). Long horizon hurts. DataSphere auto-launch armed: noise_s5/s6 (h128 full + fullft, gt4.1), fbc1k_s5/s6 (d256 Vlad recipe Muon x3 chunk1000 boot8 + fullft, gt4i.1).
- (13) d256bc1k s3 (boot8 chunk1000 Muon x3) both folds .6933 (c3 .4602 quiet .4084) vs d256boot s3 .6925 (+.0008), = boots4 .6933. Best seed-3 d256; s4 running; full-data fbc1k_s5/s6 queued on DataSphere.
- (13) d256bc1k s4 .6935 (vs boots4 .6933); s3 .6933 (vs boot .6925): chunk1000+Muon x3 avg +.0005. 
- DIVERSITY ABLATION (blend 0.2 into cmp_fullft): rawwmse (raw-only, wmse) dv +.0005/+.0010 dHO +.0011/+.0012; rawcorr dv +.0007/+.0012 dHO +.0004/+.0007; fullmix (all feats, mix loss) dv +.0002..8 dHO +.0004..7. => raw-only view is the main diversity source (like organizer baseline). 3-way fullft+base(.25)+rawwmse(.25): dv +.0007/+.0018 dHO +.0033/+.0035 (best so far). Training full-data frawW_s3/s4. FT-teacher students d256ftt queued.
- PUBLIC 2026-10-05 03:11: vlad_bw.zip (Vlad s4 .5 + raw-wMSE frawW s3/s4 .25 + organizer baseline .0625, our engine) = 0.6664 (NEW BEST; vs Vlad s4 alone 0.6660, our pair_fullft 0.6649). Runtime 32m02s / 640MB; local projection 79.7 -> ratio 0.40.
- PAIR-LEVEL composite (base = cmp_fullft s3+s4 mean): extra seeds control +.0002 dHO; subset members (pairs, .25): za dv+.0006 dHO+.0003, zvol -.0001/+.0006, zpxev +.0003/+.0005, zi1 +.0002/+.0006, rawwmse(s3) +.0003/+.0005, rawcorr +.0005/-.0002; baseline .25*.25 +.0003/+.0019 (main lever); base+za+zvol+zpxev +.0007/+.0027 (best); base+rawwmse +.0003/+.0019. Single-model composite overstated diversity (seed blend control gives +.0012).
- (15) FT-teacher students d256ftt (teachers h256 fully FT on valid folds!=k; bc1k recipe) .6932 (quiet .4036) vs d256bc1k .6933/.6935 -> null on valid folds. CLOSED (no train-HO view available for fold models).
- PUBLIC 2026-10-05 03:56: blend_pair_rw.zip (d256 combo s4 .3 + our pair_fullft .2 + frawW s3/s4 .25 + baseline .0625) = 0.6666 (NEW BEST, +.0002 vs blend_rw). Runtime 37m51s, local proj 93.8 -> ratio 0.40.
- PUBLIC 05:01 blend_pair_sub (combo .25 + pair .15 + za/zvol/zpxev s3 .1 each + base) = 0.6664 (-.0002 vs blend_pair_rw .6666, noise). Runtime 41m07s (proj 97.8 -> 0.42). Subset members do not add on public.
## AUTONOMOUS DAY PLAN 10-05 (bold): A org-style members (raw, plain mse all rows, no prop, adamw) to replicate baseline effect with multiple seeds; B less valid-specialized base in blend (wise05/gglob98 pairs); C stacking/per-class weights fit on train-HO, check valid fold0; D online-norm member as blend view; E DataSphere second d256 (bc1k full).
- (B) pair-level: wise05 pair alone dv -.0002 dHO +.0018; wise05+base -.0002/+.0030 (vs fullft+base +.0003/+.0019); +za/zvol/zpxev +.0004/+.0033. gglob98 ~ wise05; wise025+base+subs +.0002/+.0034. NOTE: LB today: blend_rw .6664, blend_pair_rw .6666, blend_pair_sub .6664 -> composite dHO deltas of ~.0008 did not show on public (noise ~.0005): composite overstates blend gains.
- (A) org-style members (raw, plain mse all rows, adamw, no prop; composite, no FT): org pair alone dv -.0231; in blend .25 dv -.0004 dHO +.0001; +base+org dv -.0006 dHO +.0014 (< base alone +.0019); orgall/orgw similar. Organizer baseline effect NOT reproducible by style -> baseline weights themselves carry it (hypothesis: trained on other/larger data). onorm composite arms OOM (rerun later).
- Started: baseline FT on valid (true mask, gcorr, fp32 to avoid shift-16 precision loss; new --precision fp32, WNN_PREDICT_FP32) comp arms baseft4 (lr1e-4) s3/s4, baseft35 (3e-5) s3; fp32 baseline preds orgbase32.
- DataSphere: dataset disk read ~3.7 MB/s (27GB concat took 2h). Current 4 jobs continue; no more heavy-data DS jobs.
- baseline leakage check: orgbase valid fold0 .6287 train-HO .5377 ratio .855 = same ratio as all our models (.855) -> no sign baseline saw train; its diversity is genuine.
- baseline FT (converted orgbase, fp32): full FT valid WP .63 -> .18 (lr3e-5) / .004 (1e-4); even head-only FT (freeze inp,gru,head.0) t0 .63 -> .33. Training-path vs eval-path mismatch for the converted baseline (unresolved). PARKED.
- DataSphere jobs done (2h concat + ~1h train): fbc1k_s5/s6 (+fbc1kft) d256 Vlad-recipe full data, fmu128/ffull s5/s6 imported. blend_2d_rw (combo .25 + fbc1kft s5/s6 .25 + rawW .25 + base) 137 local-min (too slow); blend_2d_lite (combo .35 + fbc1kft_s5 .2 + rawW s3 .15 + base .0625) 96.7 local (~40 public), deterministic.
- (C) per-class baseline weights: fit on trainHO -> [.25,.2,.2,.25], eval valid .7019 = global .25 (.7019) vs pair .7016; fit on valid -> all 0 (eval trainHO .6019 vs global .6038). Per-class stacking adds nothing. CLOSED. Launched comp arms: onorm s3/s4 (rerun), lstm s3/s4, dual-clock, two-stream (as blend members).

## Revival audit 2026-10-05 11:27 UTC
DataSphere chain-bootstrap completed: `../2026-10-05-revival/report.json`; detailed update and 15 ideas in `../2026-10-05-revival/research_update.txt` and `ideas.json`. Two-stream seed3 fixed-artifact blend delta +.000685 valid / +.000931 train-HO on chain-clean subsets. Replication tstream s4 + matched raw s4 and onorm s3/s4 running, with bounded finish supervisor. These are exploratory composite results, not new public scores.

## Wave2 completed 2026-10-05 12:27 UTC
Two-stream replicated (pair vs raw: +.000679 valid-clean / +.000606 prior train-clean). Full-data twostream seed3 exported exactly via block-diagonal GRU256; two tested ZIPs prepared, not submitted: `2026-10-05-wave2/blend_twostream.zip` and `blend_twostream_lite.zip`. Relative runtime projections vs scored incumbent: 43.4 / 36.3 min; older conservative upper estimates 56.1 / 46.8. DataSphere native audit SUCCESS. Sentinel paired trial no robust gain; complement residual FT positive train-HO but no gain over direct FT on valid. Existing bootstrap already respects label-change boundary; duplicate proposal withdrawn. Full report and source hashes in `research/iterations/2026-10-05-wave2/`. Recorded public remains .6666. All wave2 GPU services and finish controllers completed successfully.

## Autonomous continuation, 2026-10-05 14:36 UTC

User target: top-1 = 0.6711. Recorded public remains 0.6666. No new submissions.

All 15 directions received an experiment or feasibility check, not exhaustive validation of every implementation. Exact mirror failed full feature-map verification; future/race/retrieval started as limited prototypes. Current registry: AUTONOMOUS_STATUS.json.

Main result: heterogeneous distillation into h256, two seeds. On 425 true-mask valid sequences / 123 chains excluding ALL train chains, student pair 0.7181503 vs teacher 0.7153250 (+0.0028253, conditional CI [0.0019575, 0.0038363]); single seeds 0.7176578 / 0.7174401. Two-stream teacher 0.7161344. Adding that teacher lowers the student pair score. Train-disjoint composite replication and final all-data training are active. These are NOT public/private score estimates.

Retrieval revived: 65536-row chain-balanced bank, GRU-only PCA64, k128, correction .05. Stride32 pair: train +0.0002378, valid +0.0002250; conditional CIs positive. Causal MLP student retains about +0.00016 on both, but intervals cross zero. Full-stream replay active.

Negative followups: capped32 bootstrap loses guarded h8 by .000524. Event8 adaptive compute loses matched all-update control. Warmup summary, quiet adapter, moment-ratio head, SSL pretrain and joint objective show no reliable gain. H512 distillation beats h128 but loses h256.

Baseline: old FP32 path kept cuDNN TF32 enabled; +16 shifts impaired optimizer conditioning. Corrected FT trains, but head FT loses .00133 on train diagnostic, removing baseline loses .00235. Keep baseline in robust candidate rather than select on valid alone.

Native: skip exactly zero GRU blocks while preserving global int8 activation scale. Bit-exact on all scalar/AVX2/VNNI backends and v2/avx2/v3 builds; reset/step parity pass. Relative projected public runtime: sparse single two-stream package 36.8 min, with linear int8 baseline 31.3 min; two-stream pair ~40.7 min. Not actual scorer time. Final seed4 training completed.

## Public .6674 and resumed research, 2026-10-05 20:35 UTC

User reported `distilled_h256_pair.zip`, submission `26n9pg4y`: public .6674, 24m57s, 640MB. New best +.0008 vs .6666. User leader .6711, remaining gap .0037. Daily counter 4/5 as reported; no automatic submission performed. Archive SHA256 86020dcf384430ae5fd3a848c8a8b4533cd81031caad3b0feec7647c4a5491f9.

Train-disjoint replication completed: distilled pair vs its teacher +.0017429 valid, +.0016844 train diagnostic (estimated scoring mask); both conditional chain-bootstrap intervals positive. New two-stream teacher yields another +.0002137 on full-train-model valid cohort; full-data final counterparts training as w21final_distilled_s5/s6.

Direct-student blending rejected: full-train-model valid has a small gain, but composite holdout loses .0003169 to .0027839 on train as direct weight rises .1 to .5, and valid also declines.

Wave22: h256 -> h384 function-preserving GRU/head widening with zero-sum outgoing jitter; two seed parity passes, all-head output error <3.6e-5. Both widths receive identical 2-epoch continuation against stronger ensemble teacher. h384 seeds3/4 train on GPUs5/6, matched h256 controls sequentially on GPU1. Independent DataSphere CPU audit passed; single h384 roughly 28 min, mixed h384+h256 roughly 40.5 min relative to newly scored h256 pair. This is a projection, not actual scoring-host timing or a quality result.
