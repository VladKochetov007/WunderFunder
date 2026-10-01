mo.md(f"""
# What limits the model, and what to do about it

Clean out-of-fold predictions, 4 chain-excluded folds. **SOTA** = R4D analog (classes 0–2: fine-tuned since s4 + distilled
GRU256; class 3: since s4 with merged parent+FT heads + distilled GRU256). **Reference** = incumbent recipe (two-stage FT
since pair). Pooled Global WP: SOTA {SUMMARY[4]['WP SOTA (R4D analog)']:.4f} vs reference {SUMMARY[4]['WP reference (incumbent recipe)']:.4f}.

### 1. The loss is one place: long class-3 runs  (F3)
Half of all lost metric sits in class-3 rows of runs lasting 64–1,024 rows (segment WP 0.25–0.44); by distance to the
next quote move, rows 32–512 rows away hold ~44% of the loss at WP 0.09–0.28. Classes 0–2 together are under 40%.

### 2. In those runs the outcome is not knowable at the start  (F1, F2)
For runs lasting ≥ 64 rows, corr(prediction, run target) at the run's first row is 0.02 in class 3 (0.07 class 2,
~0.2 classes 0/1) and rises only in the last ~32 rows before the next move. F1 shows it directly: the target sits
at e.g. −1.3 for hundreds of rows while the prediction hovers near 0 until the book starts to move. This is the
future-path part of the label, not a modelling bug — and why every post-hoc fix we tried failed.

### 3. Ensembling is saturated  (F6)
The since member and the distilled GRU256 have residual correlation 0.985–0.999 in every segment. Averaging them
gains ≤ 0.0005 and sometimes loses. Gains must come from better single models, not more members.

### 4. Why distillation works  (F5)
Against the since member, the α 0.5 student is worse far from the next move (class 3, 128+ rows: −4 to −7 ×10⁻³)
while the α 0.7 student is better everywhere (+0.5 to +4 ×10⁻³). Stronger teacher weight denoises labels that are
dominated by unknowable future paths — exactly the high-loss region. Fine-tuning shows the opposite trade: the
un-fine-tuned parent is better early in long runs for classes 1–2 (F2).

### 5. Dead ends confirmed here
Prediction scale per segment looks misallocated (F7: far-from-move rows "want" 0.1–0.75× scale), but a causal
24-parameter scale correction loses on every fold (E-322). Calibration is on the diagonal (06). Sequence-level
dispersion and residual autocorrelation are future information (E-314, E-317).

### Next experiments this points to
1. **Noise-adaptive distillation (H-059):** teacher weight rising with the training row's distance to label
   resolution (known at train time; inputs stay causal), against constant α 0.7. Targets the 44–50% loss region.
2. **Push α (0.85 queued) and teacher quality** (GRU256 teachers in the team recipe).
3. **Head-only FT or lighter FT for the since trunk** in classes 1–2, which keeps early-run information.

Appendix (descriptive): target distributions, run structure, event-time ACF/PACF of target and residual, ARCH/GARCH
(residual volatility persistence α+β ≈ 0.82–0.86, class 3 ≈ 0.99), calibration, per-sequence and between/within tables.
""")
