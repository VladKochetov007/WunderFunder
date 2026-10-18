# Autonomous research 2026-10-13 (after cross-team blend xteam_h4 was submitted)
Eval: ~/xteam/cmp.py on our strict425 (fold0, 123 chains), paired chain bootstrap. Baselines: ftt pair (d256ftt_f0+dftt_s4) .71859; w19 pair .71836; ours4 .71904.
## Arms
| arm | where | idea | status |
|---|---|---|---|
| dftt_p0_s3/s4 | bel GPU0/2 | ftt recipe, prop_power 0 (teammate lever +.0016..+.0035) | running |
| w19p0_s3 / w19p0_s4 | bel GPU1 / DS | champion student, prop_power 0 | running |
| dftt_ncl_s3 / s4 | bel GPU6 / DS | negative-correlation member: target .5y+.2T1+.2T2-.1*W19 (anti-champion) | running |
| dftt_p0mi_s3 | bel GPU6 after ncl (DS failed INTERNAL x2) | prop0 + Muon on input projection (~muon-all) | running |
| t384_s3/s4 -> t384ft -> dftt_t384_s3/s4 | bel GPU4/1 | stronger teachers: h384 Muon+inp scratch 3ep, valid FT, ftt student distilled from them | running |
| w19a9e6_s3/s4 | bel GPU0/2 after wave1 | champion student alpha .9, 6 epochs (teammate +.0017 lever) | queued |
## CLOSED: cross-sequence warm state (chain predecessor)
warm/warm.py: concat full predecessor (valid chain, tail->head NN) + current, score current part. strict425 seqs with a valid predecessor: 303.
w19distilled_s3 cold .73820 warm .73829 (+.0001 [-.0003,+.0006]); d256ftt_f0 .73773 -> .73803 (+.0003 [-.0001,+.0008]).
Upper bound (perfect linking, predecessor always seen); test would be ~half -> ~+.0001. Not worth serving complexity.
DataSphere: all 4 jobs + retry fail with INTERNAL after ~5 min (before code runs) on 2026-10-13 -> likely budget/quota.
## Results wave 1 (strict425)
- w19p0_s3 (student prop0) .71820 vs w19distilled_s3 .71790 (+.0003, p .87); swap into ours4 +.00005.
- dftt_p0 pair .71890 vs ftt pair .71859 (+.0003, p .71); per seed s3 +.00004 vs ctl s3, s4 +.0007 vs dftt_s4. ours4 with p0 ftt pair .71930 (+.00026 vs .71904); with p0 student too .71935.
- NCL anti-champion member dftt_ncl_s3 .71524 (-.0029 vs ctl s3): too weak; ours4 swap +.00009. CLOSED at this strength.
=> prop0 in our lineage ~+.0003 (much smaller than teammate's +.0016..+.0035 on his scratch lineage).
## CLOSED: is y0+y1 (common mode) determined by current prices?
Valid rows: 22 i0 price codes take only ~23k distinct tuples, fully determined by (i0_p0, i0_p11). Within identical tuples 64.5% of sum variance and 95% of diff variance remain -> sum depends on the future, not just on the current spread. Linear R2 .32. No decoding shortcut.
## h384 Muon(+inp) teachers on fold0 (strict425)
raw pair t384_s3/s4 .71528 vs old ftt teachers h256ft pair .71441 (+.0009). Any valid FT hurts h384: full FT .71173, head-only FT .71440. Students dftt_t384r_s3/s4 distilled from RAW t384 teachers (FT-teacher students killed).
## Results wave 2
- w19a9e6 (champion student alpha .9, 6 ep): pair .71706 vs w19 pair .71836 (-.0013, ci [-.0022,-.0004]); both seeds -.0013. ours4 swap .71860 (-.0004). CLOSED: teammate's alpha .9/e6 lever does not transfer to our w19 student.
- BUG: first dftt_t384r students trained on NaN teacher caches (predict_any_env without `full` fills only fold0 rows) -> loss nan, discarded (bad_cache/). Relaunched as dftt_t384f_s3/s4 with *_full caches.
- wave4: t384_s5 (3rd h384 teacher), w19p0_s4 (prop0 student replicate).

## 2026-10-13 results (strict425)
- h384 raw teachers -> ftt students (dftt_t384f_s3/s4, full caches nan 0): pair .71848 vs ftt pair .71859 (-.0001); s4 -.0011. CLOSED.
- dftt_p0mi_s3 (prop0+muon_inp ftt) solo .71836 vs ctl .71816 (+.0002).
- w19p0 pair (prop0 champion students s3+s4) .71835 vs w19 pair .71836 (0). prop0 neutral for us.
- E4 .71904; E4p0 .71916; E4p0mi .71926 (+.0002, noise-level); E6 (6 members) .71929.
- Post-hoc transforms on E4p0mi blend, 2-fold chain CV x20: time-bin gains -.0006, power terms -.0002, causal running-std norm -.0003, all -.0011. Models already calibrated. CLOSED.
- xteam_h4 public 0.6681, 50m16s (pair_ftt2 0.6682, 44m42s): cross-team blend gives nothing on public.

## 2026-10-13 wave 5-9: reopened / new hypotheses (strict425, ftt recipe from d256ftt_f0 args)
Trainer copy plew19/train_rnn_rs.py (env flags; flag-off identical): RS_RUNNING_NORM (EMA weight-sum normaliser in the distill branch; old --loss-norm running was a no-op there), RS_SYNTH (spliced synthetic sequences, online teachers h256ft_f0+h256s4ft_f0 as targets; teacher-vs-cache corr .9987), RS_VOL28 (teammate volume28 run-relative features, +28 inputs; predict via predict_rs.py).
- #9 hold-between-events (CPU, E4p0mi blend): when all i0 unchanged y never changes (702,917 rows, 0 changes), but holding the prediction costs -.0017 (prices-only hold -.085). CLOSED.
- #13 per-chunk W_c CV measured in training = .19-.20 (>5% threshold) -> dftt_rn_s3/s4.
- #5/#4 width: dftt_h384_s3 solo +.0002 vs ctl (null); raw h384 teachers solo .713-.715 (train-only, below valid-FT ftt students). Wide student from 3 h384 teachers: dftt_w384t3_s3.
- #2 data ladder: dftt_d50_s3 (5300 train seqs), dftt_d75_s3 (7950).
- #1 synthetic splice distillation: dftt_syn_s3/s4 (frac .5, w 1).
- #6 volume28: dftt_vol_s3/s4.  #8 dual clock (event GRU 128): dftt_dual_s3.
### wave 5-9 results (solo vs ctl d256ftt_f0 .71816; pair .71859; E4 .71904)
- running norm: rn_s3 -.0004, rn_s4 -.0009, pair .71809 (-.0005 vs pair). CLOSED.
- synth splice distill: syn_s4 .71589 (-.0023, p>0 .00). s3 OOMed (GPU shared), rerun as wave7b. Likely CLOSED.
- data ladder: 50% train .71687 (-.0013), 75% .71817 (0.0000). Curve flat from 75% to 100% -> NOT data-limited for more same-distribution data; lowers prior on synth/MAE/pretraining ideas.
- wide student h384 from 3 h384 teachers: .71792 (-.0002). CLOSED.
- volume28: s3 -.0008, s4 -.0002, pair .71823 (-.0004 vs pair). CLOSED.
- dual clock (event GRU 128): .71855 (+.0004 solo, best single); as 5th member E5dual .71924 vs E5 same-family .71903 (+.0002). Seed s4 running (wave10). Engine lacks dual -> needs real gain to justify.
- wave10: dftt_dual_s4, dftt_lowlr_s3 (lr 2e-4, 12 epochs; #14).
### wave 7b/10/11 results
- synth s3 rerun .71620 (-.0020): synthetic splice distillation CLOSED (both seeds -.002).
- dual clock s4 .71759 (-.0006); dual pair .71862 = ftt pair .71859; E4dual .71913 vs E4 .71904. CLOSED (not worth engine work).
- EMA multi-timescale inputs (hl 128/1024/8192, 36 feats): s3 -.0003, s4 -.0011, pair -.0006 vs pair. CLOSED.
- low lr 2e-4 x 12 epochs: .71831 (+.00015, one seed, ci +-.002). Null.
Verdict 2026-10-13: 11 of 15 hypotheses tested, all null/negative. Family ceiling ~.719 strict425 / ~.668 public.
### 2026-10-13 evening
- lookahead(1,2) teacher (vfold BEST .7146 vs causal ~.703) -> ftt students lk_s3/s4 pair .71814 (-.0005 vs ftt pair). CLOSED.
- need_prediction == step>=99 (no info); is_scored not served. CLOSED.
- 9-member mega ensemble M9 .71928 (+.00024 vs E4): family ensemble ceiling.
- h192 ftt: solo .71819/.71685/.71792 (mean -.0004 vs h256); h192x3 .71835 < ftt pair; 7-member E ~ E4. h128 .71681. Rust q8vnni: h192 16us/row vs h256 37us (cpp h192 is SLOWER 70us), 6xh192 175us < 4xh256 200us, but no quality gain. CLOSED (fttall192/d192 champion killed).
- 5th-member screen on E4 (0.8 E4 + 0.2 m): lowlr +.00033, t384f +.00026, w384t3 +.00024, dual +.0002, p0mi +.00016; others <=0.
- Swap inside 4 members: W(w19 pair) + lowlr_s3 + t384f_s3 = .71954 (+.0005, selection-biased); W + lowlr_s3 + ftt s4 +.0003.
- lowlr vfold curve plateaus at ep6-7 (.7059).
- Running: wave16 replication dftt_lowlr_s4/s5, dftt_lowlr_t3_s3 (lowlr + h384 trio teachers); wave17 speculative all-data fttall_lowlr_s5/s6.
### 2026-10-14 night: LR-policy zoo + per-teacher students (literature-driven)
Lit: LREnsemble (Jin & Wu 2024, arXiv 2410.07564) and Snapshot Ensembles (Huang 2017): different LR policies -> different basins -> diverse members; Fort et al. 2019: ensemble diversity mostly from different basins; Nam et al. NeurIPS 2021 (arXiv 2110.14149): distillation from an ensemble mean collapses student diversity.
- wave18: dftt_lladam_s3 (AdamW lr3e-4 12ep), dftt_ll8_s3 (lowlr 8ep).
- wave19 zoo (seed 3): zc constant 3e-4 8ep; zcos cosine 1e-3 6ep; zhi onecycle 2e-3 3ep; zb16 batch16 2e-4 8ep.
- wave20 (after wave16): per-teacher students dftt_tA_s3 (h256ft_f0 only), dftt_tB_s4 (h256s4ft_f0 only), dftt_llB_s4 (lowlr 8ep, teacher B only).
- wave21: dftt_aux_s3/s4 (--aux 100,1000 weight .3; JS2024 8th place aux future targets +.001)

## 2026-10-14 EDA: target structure (CPU only)
- Metric weight by class (strict425, E4): c0 .311 (WP .814), c1 .180 (.780), c2 .090 (.766), c3 .420 (.471). WP without class 3 = .805. Class 3 is the lever.
- S=t0+t1, D=t0-t1: S quantized (class tick quanta .1255/.2413/.4686/.65687), var share of S 17/15/8/5%; our corr on S .34/.31/.30/.02, on D .79/.75/.76/.51.
- Oracle true S -> WP .7527 (+.034) but causal table book-code->S (fit on other valid seqs) -> .6864 (-.033): S variation = future stop outcomes, not current spread. CLOSED as shortcut.
- Label law check (tick paths via wnn.ticks, F=round(mean mid over next H book events), stop sp+st): exact match c3 .72-.77 (H 4-6, st 2), c2 .57, c1 .36, c0 .16 (c0 13% unknown transitions). Row-time horizons fit worse (c3 .54 at 100 rows). -> label ~ event-time horizon (likely wall-clock), law near-exact only in class 3.
- Next: H-RB3 (Rao-Blackwellised class-3 labels) and event-horizon privileged teacher for class 3.
### 2026-10-14 imbalance / quant / event-time (CPU, strict class-3, chain-split cross-fit)
- volume columns 22-43 are continuous mixtures (~15k uniques/seq), not codes -> exact-code table useless (coverage 0).
- next-mid-move direction AUC: ensemble D .8205, MLP(D) .8441, vol-only .8421, D+vol .8588 (imb2.py).
- but label-space correction on E4 (imb3-5): ctl MLP(p) -.0014, MLP(p+vol) -.0450, MLP(p+dvol) -.0368, linear ctl -.0007, linear p+dvol -.0076. Direction signal does not transfer to metric. CLOSED.
- event-time feats (mid vs last 1-4 event mids, time since change; ev1.py) linear: -.0013 vs ctl -.0007. CLOSED.
- per-class data ladder, class 3: 50% .4661, 75% .4695, 100% .4692/.4673 -> class 3 not data-limited; class-3 synthesis of same-distribution data low prior.
- RB3 labels (dftt_rb_s3/s4): solo -.0020/-.0018, class 3 .4639/.4663 vs .4692/.4673, pair -.0009 vs ftt pair. CLOSED. Label noise is not the class-3 bottleneck.
- launched lowlr2.sh: dftt_lowlr_s4/s5 (lr 2e-4, 12 ep) on GPU5, replication of lowlr_s3 swap gain.
- low-lr replication (lr 2e-4, 12 ep): swap for dftt_s4 in E4: s3 +.0004, s4 +.0003, s5 +.0007 (all p>0 >=.96). E4lo45 (both ftt -> lowlr s4,s5) .71992 +.0009 ci [+.0001,+.0018]. CONFIRMED.
- launched falow.sh: fttall_lowlr_s5/s6 on GPU5 for submission (swap ftt members of pair_ftt2).
- package pair_lowlr.zip (champ s5/s6 .25 + fttall_lowlr_s5/s6@ep11 .25): parity corr .99993, deterministic, timing = pair_ftt2 (4.09 vs 4.10 s/seq). sha 113ab142...
- 2026-10-15: pair_lowlr public 0.6680 (45m07s) vs pair_ftt2 0.6682: local +.0009 did NOT transfer (2nd time after xteam_h4). Sub-0.001 local gains are below public resolution.
- position-in-seq WP (E4): no late-sequence degradation (16-20k best .742); long context not a lever.
- launched c3sp.sh: class-3 specialist FT (d256ftt_f0 init, c3-only seq weights, 2 ep lr 2e-4) + uniform control, GPU5.
- causal per-row scale exp(theta.z) (running |p|, event rate, p0-p1, log t; scale1.py) fitted on metric, 5x2 chain cross-fit: -.0018 sd .0007, all 10 negative. Models already scale-calibrated. CLOSED.
- class-3 specialist FT (c3-only, 2 ep lr 2e-4 from d256ftt_f0): class3 solo .4689 vs parent .4692 vs uniform-FT ctl .4719; E4 swap on c3 seqs +.0001 (sp) / +.0002 (ctl). Specialisation does not help class 3; CLOSED.
- zoo screen (zoo.py, 793 cached preds as 5th member of E4): every member >= +.001 is contaminated (f1/f2/f3 folds, all-valid h2, finals). Best clean: dftt_lowlr_s5 +.0008@.3. No clean diverse family exists in cache. Saturation confirmed.
- continued FT (2-4 ep lr 2e-4 from ftt parents): E4 swap +.0003..+.0004 (beta 0 and 1.05). Below public resolution. CLOSED.
- qeval.py: beta-weighted (quiet test, rate^-1.05) strict425 compare. E4lo +.0009 (b0) / +.0013 (b1.05). Launched dftt_q105_s3: FT from d256ftt_f0 with seq weights rate^-1.05 (ctl = dftt_c3ctl_s3 uniform).
- quiet-composition FT dftt_q105_s3 (seq w rate^-1.05) vs uniform ctl: solo -.0006 (b0), -.0009 (b1.05, ci<0). CLOSED: reweighting training toward test composition hurts even on test-weighted eval.
- WP by rate quintile (E4): .685/.661/.699/.727/.761; test weight share (b1.05) 24/22/21/18/15% vs local 9/15/22/25/29%. Quiet seqs weakest and doubly weighted on public.
- H3 (opus_micro, never run): tied i0/i1 book encoder (enc=tied in src/wnn/nn.py, zero-init residual MLP on raw 52-col instrument blocks + row diffs, shared weights). dftt_tied_s3/s4 = ftt recipe + enc tied; ctl d256ftt_f0/dftt_s4. Predicted +.001..003 on classes 0-2 (opus_micro N4: i1->i0 lead corr .25-.34 in class 2).
- tied encoder result: class 2 +.0015/+.0023 (both seeds), classes 0/1 +0..+.0006, class 3 flat; pair +.0001 (b0) / +.0004 (b1.05); E4 swap +.00015/+.0003. Direction right, size small. Next: tiedlead (enc + RS_LEAD: i1 block at last i0 px event, encoded i1 change since last i0 move), dftt_tlead_s3/s4.
- tiedlead result: pair +.0004 (b0)/+.0008 (b1.05, p.91); E4tlead +.0003/+.0005; E6tt (w19x2 .2 + tied/tlead .15x4) +.0003/+.0006 p.99. Per class: class 3 slightly better than tied (.4688/.4685 vs .4679/.4668), class 2 +.0006..+.0008. Consistent but < public resolution. Next: tiedlead2 (RS_LEAD=2: + i0 at last i1 event, + log ages since last i0/i1 event), dftt_tl2_s3/s4.
- A5 instrument recurrence (ident2.py, fair baseline = held-out train seq max-Jaccard vs 120 other-chain train seqs): valid ~= train in every class (c2 .53-.90 vs .55-.91; c3 all 1.0 both). No valid-specific recurrence; code identity already in inputs. CLOSED.
- tiedlead2 (both-way lead + ages): pair +.0003/+.0006 = tlead, no extra. Family blend E6t2 (w19x2 .2 + tl2/tlead .15x4) +.0009 (b0, p.98) / +.0010 (b1.05). Scaling: fttall_tlead_s5/s6 (ARGS_FROM fttall_s5 ep3, enc tiedlead, RS_LEAD=1) + C++ engine support.
- C++ engine tied support: ~/rs1013/native-tied (copy of wave16/native-sparse + tied MLP, lead block, int8 tied matrices padded 104->128). Parity vs torch = plain members (corr .99991-.99993). Cost per d256 member: plain 29us/row, tied 38, tlead 40. Package script ~/pkgwork/build_tlead.py (tl3 = s5 .34 + fttall_tlead_s5/s6 .33; tl4 = s5,s6,tl5,tl6 .25). Dry run with fold models: deterministic, tl3 projected 42.5 min, tl4 54.5 (projections run ~15% high vs scorer). Local blends: E3tl (w19s3 + tlead pair) +.0003 b0 / +.0007 b1.05 vs E4; E4tl +.0003/+.0005.
- 2026-10-15 evening: fttall_tlead_s5/s6 trained; packages tl3.zip (sha d80e69d3, proj 43.7 min -> ~38 real) and tl4.zip (sha 659ad2e6, proj 55.5 -> ~48 real) built, engine-vs-torch corr .99988/.99990, deterministic. Copied to Mac submissions/. Upload tl3 first.
- PUBLIC 2026-10-15: tl3 0.6683 (39m12s), tl4 0.6683 (50m7s) vs pair_ftt2 0.6682 / pair_lowlr 0.6680. Tied family: +.0001 public (within ~.0005 resolution), but new best and first architecture-diverse member. Timing projections ran 10% high.
- Next: dftt_tlbig_s3/s4 = tiedlead with tied MLP 104->256->128 (RS_TIED_DIMS=256,128).
- tlbig (tied 256/128): pair -.0004 (b0) / -.0006 (b1.05) vs tlead. Bigger encoder worse. CLOSED.
- E3tl + 3rd tied member (tl2/tied): +.0001, saturated. Next: w19tl_s3/s4 = champion student recipe (ARGS_FROM w19distilled_s3: 2ep FT lr1e-4, w19teacher distill .7, family head) but init dftt_tlead_sX@ep3 + enc tiedlead (CompressModel now passes enc).
- w19tl (champion student recipe on tlead init): pair +.0007/+.0009 over w19 pair solo (strongest pair so far, .71909), but in blends = E3tl (E4w +.0000/-.0001, E4x +.0001/-.0000). Same-lineage, no diversity. Package gain none -> CLOSED for packaging.
