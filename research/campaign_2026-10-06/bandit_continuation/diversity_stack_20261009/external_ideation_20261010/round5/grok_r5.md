I'll read the folder notes first, starting with the HPO record and the previous-competition tricks, then the rest, so the ten ideas stay distinct from what already ran.## 1. Row-level linear stacker of two banked members — about 0.006 WP per GPU-hour

**Mechanism.** Pairwise predictions correlate 0.95–0.99, and a biased bank screen still gained about +0.002 from a second member before saturating at three. A frozen linear map can use disagreement and clock state, which a global average ignores.

**Change.** Members: `h230_f1_s3_ep3` and the best diverse banked peer (3-epoch h384 scratch, else the gated run). Fit, on fold 0 only, a per-target ridge from `[p1, p2, p1−p2, |p1−p2|, fast-clock age, rows-since-event]` to `y`, weights `|clip(y,2)|`, rows ≥ 99. Freeze coefficients. Serve both q8vnni GRUs plus that map.

**Expected.** +0.0012 versus 0.6692 (about −0.001 to +0.003). The +0.002 bank figure is selection-biased. One-run fold-1 sd is 0.0022, so one seed above 0.670 is not evidence.

**Cost.** 0.2 GPU-hour (export and one-core timing). Fit is CPU on the bank.

**Falsifier.** Two-seed mean fold-1 ≤ 0.6692, or loaded one-core replay > 91 µs/row.

## 2. Per-sequence adaptive weights — about 0.005 per GPU-hour

**Mechanism.** Sequence-level error shares are uneven, so a single scalar per sequence can move weight toward the member that is stable in that regime. The served weight is a function of causal inputs; labels are used only while fitting on fold 0.

**Change.** Same two members. Online state after row 99: EWMA of `|p1−p2|`, EWMA of `|p|`, and fast-clock age (blocks that update together: columns 0–21, 44–47, 52–69). `w = sigmoid(a + b·disagreement + c·|p| + d·age)`, one `w` per target, clipped to [0.15, 0.85], held constant on rows 0–98. Fit `(a,b,c,d)` by fold-0 weighted Pearson. Start from the stacker solution so this arm tests the sequence scalar.

**Expected.** +0.0013 versus 0.6692 (about −0.001 to +0.003). Incremental gain over the stacker is about +0.0003 and may be zero because the members share most errors.

**Cost.** 0.25 GPU-hour.

**Falsifier.** Fold-1 WP ≤ the frozen stacker on the same pair, or replay > 91 µs/row.

## 3. Early-sequence hard routing, one forward — about 0.004 per GPU-hour

**Mechanism.** Market class is recoverable from the first 500 rows (99.4% balanced accuracy). Routing picks one existing checkpoint per sequence from the unscored prefix, so the blend costs one GRU.

**Change.** From rows 0–98 only, compute change rates of the fast blocks above. Eight quantile bins fit on fold 0, crossed with the warmup class prediction. For each bin, choose the banked member with the best fold-0 weighted Pearson (student, h384 scratch, gated, prop-power-0 if that seed replicate holds). Sequences in empty bins use the student. Serve the chosen q8vnni member only.

**Expected.** +0.0007 (about −0.001 to +0.002). High member correlation caps specialization. This is the blend that stays inside the timing budget even if two forwards do not.

**Cost.** 0.15 GPU-hour.

**Falsifier.** Fold-1 routed WP ≤ the best single member in the pool.

## 4. HPO trial 52 through the student chain — about 0.0010 per GPU-hour

**Mechanism.** Scratch HPO prefers a short WSD schedule, SWA, and light dropout over 4-epoch OneCycle (0.6660). Trial 52’s objective is 0.6690. That recipe has not been the distillation parent or a teacher family.

**Change.** Scratch teacher, fold-1 fit split: WSD schedule, `swa_frac` 0.31, dropout 0.04, 2 epochs, weight decay 0.0035, loss-weight floor 0.18, class-3 weight 1.19, Muon multiplier 0.70, averaging setting 2, loss weighted MSE, propensity on, valid rows use the true mask. Warm-start the student from that teacher’s SWA weights; distill 3 epochs, AdamW 1e-4, one-cycle, alpha 0.7, replacing the strong-family slots with two seeds of this recipe. Keep the other families. Repeat once with trial 35 (dropout 0.16, SWA fraction 0.23) as a second seed family, not a mixed schedule.

**Expected.** +0.0015 (about −0.002 to +0.004). The student already sits at 0.6692, so much of the HPO gain may already be in the blend.

**Cost.** 1.5 GPU-hours (two short teachers plus one student).

**Falsifier.** Student two-seed mean fold-1 ≤ 0.6692.

## 5. Eight-row privileged teacher, distilled — about 0.0008 per GPU-hour

**Mechanism.** The same GRU with 8 future rows scores 0.711 versus 0.666 causal. A teacher that sees those rows only while writing distill targets can move the causal student if part of that future is predictable from the past.

**Change.** Teacher inputs append the next 8 rows of both instruments’ price blocks (known offline). Loss stays weighted MSE on the current-row labels, clip ±2. Fit only on the fold-1 training split, with 2-way out-of-fold predictions inside that split. Student is the current architecture, causal inputs, warm-started from `h230`, alpha 0.7, the privileged out-of-fold prediction added as a fifth family with coefficient chosen on fold 0. Serve the student alone.

**Expected.** +0.002 (about −0.003 to +0.008). Most of the +0.045 oracle gap is not a function of the past, so the transferred slice is small.

**Cost.** 2.5 GPU-hours.

**Falsifier.** Causal student fold-1 ≤ 0.6692. The teacher’s own score is an oracle, not a result.

## 6. Second-stage residual, then one served student — about 0.0007 per GPU-hour

**Mechanism.** Labels are piecewise constant (mean run 9.5 rows) and only weakly co-move (both targets change together 2.9% of rows). A second model fit to the frozen student’s residual can catch run-level structure the first fit averaged over.

**Change.** Freeze `h230_f1_s3_ep3`. Train a 2-layer GRU-128 on `r = clip(y,2) − p_student`, same weighted MSE and propensity, 2 epochs, Muon on the recurrent matrices, one-cycle, fold-1 training split only. Distill a new GRU-256 whose target is `0.3·y + 0.7·(p_student + stopgrad(p_residual))`, 2 epochs, AdamW 1e-4. Serve that one student. A side measurement adds the h128 residual at inference only to see whether the sum beats the distilled net.

**Expected.** +0.0012 (about −0.002 to +0.0035).

**Cost.** 1.6 GPU-hours. The live sum needs a separate timing check (h256 q8vnni is 42 µs/row; h128 is much smaller).

**Falsifier.** Distilled student fold-1 ≤ 0.6692, and the live sum ≤ 0.6692.

## 7. Next-step book pretrain, then short fine-tune — about 0.0007 per GPU-hour

**Mechanism.** The label is a long-horizon price forecast. A next-step loss on the book, run before any weighted MSE, spends the early epochs on the dynamics. The joint forecast-head arms were an auxiliary on the PnL label during the supervised run; this is a separate stage whose target is the next input.

**Change.** Pretrain the 2-layer GRU-256 to predict the next row’s instrument price and volume blocks (104 channels) from the causal 211 inputs. Loss is MSE on those channels, 2 epochs, Muon, WSD, no label and no propensity. Streams: `train.parquet` plus valid components in folds 0, 2, and 3. Fold-1 streams stay out. Then reset the head and fine-tune 2 epochs, weighted MSE, propensity power 2, one-cycle, lr 1e-3, on the fold-1 training labels only. Score fold 1.

**Expected.** +0.0015 (about −0.002 to +0.005).

**Cost.** 2.2 GPU-hours.

**Falsifier.** Fine-tuned two-seed mean fold-1 ≤ 0.6660 (scratch control). A gain that appears only after fold-1 streams are added to pretraining is contamination.

## 8. LightGBM teacher family from frozen GRU state — about 0.0006 per GPU-hour

**Mechanism.** The GRU state is a causal summary; a tree can add threshold interactions on book imbalance and event clocks that a linear head will not. Trees are a different error, which is what the four-family blend needs.

**Change.** Dump the best teacher’s 256-d hidden state on the training split, causally. Add a small causal table: top-of-book imbalance per instrument, fast-clock age, last nonzero price difference, lags 1, 3, 13. Fit LightGBM, two targets, objective weighted MSE with `|clip(y,2)|`, 2-way out-of-fold inside the fit split, row subsample so the fit stays on the order of a few million scored rows. Insert the out-of-fold prediction as a fifth distill family, coefficient fit on fold 0, student otherwise identical to `h230` (alpha 0.7, 3 epochs, AdamW 1e-4). Serve the student.

**Expected.** +0.0009 (about −0.002 to +0.003). Extra hand-built flow features were flat (+0.0004), so the tree has to earn its keep through the state and the interactions.

**Cost.** 1.4 GPU-hours for the dump and the student. The tree itself is CPU.

**Falsifier.** Student fold-1 ≤ 0.6692, or the tree’s out-of-fold WP ≤ 0.63, in which case skip the student.

## 9. Masked reconstruction pretrain — about 0.0004 per GPU-hour

**Mechanism.** A masked reconstruction forces the GRU to represent held-out past channels before the short supervised budget memorizes labels. Inputs only, so valid labels never enter.

**Change.** Same stream rule as idea 7 (no fold-1 components). On spans of 8–32 rows, zero 20% of input channels and reconstruct them with MSE; masks are functions of the row index and a fixed seed. 2 epochs, GRU-256, Muon, WSD. Fine-tune exactly as idea 7 (new head, 2 epochs, weighted MSE). This arm does not use the next-step head from idea 7.

**Expected.** +0.0012 (about −0.002 to +0.004). Sibling gains came with the eval streams inside pretraining; this screen does not do that.

**Cost.** 3.0 GPU-hours (reconstruction is a wide output).

**Falsifier.** Fine-tuned fold-1 ≤ the idea-7 fine-tune if both run, else ≤ 0.6660.

## 10. AutoGluon plus a GluonTS trajectory teacher — about 0.0002 per GPU-hour

**Mechanism.** A stacked tabular ensemble and a global sequence model (DeepAR-style) fit the same causal covariates with different inductive bias. Their average is another distill family, not a served model.

**Change.** Same features and out-of-fold split as idea 8, subsampled rows. AutoGluon weighted-MSE ensemble (GBM, extra trees, and one small neural net) on the row table. Separately, a GluonTS global model on whole sequences with those covariates, prediction length 1, rolling forward causally, trained to the clipped targets. Average the two out-of-fold predictions, then distill as in idea 8. Serve only the GRU student.

**Expected.** +0.0008 over 0.6692 (about −0.002 to +0.003), and it should beat idea 8 by about +0.0004 if the stack is worth the hours. Sibling TFT, Informer, and DeepAR-class models lost to a tuned recurrent net, so the prior is weak.

**Cost.** 5 GPU-hours, including the student.

**Falsifier.** Out-of-fold teacher WP ≤ the LightGBM teacher, or the student fold-1 ≤ the idea-8 student.
