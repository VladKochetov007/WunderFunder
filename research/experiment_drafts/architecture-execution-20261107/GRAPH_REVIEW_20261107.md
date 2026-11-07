# Online curve review, 7 November 2026

The user explicitly requested a fresh check of learning curves and training
health. This is an actor-side diagnostic review, not independent approval of a
candidate. No model/optimizer/fit recipe, endpoint, membership or GPU allocation
was changed. Native116/L4 and117/B300 were observed20:50UTC; native118 checked
the actual retained checkpoint bytes. Actual PNG/SVG curves were opened and
compared with their source epoch journals.

## Confirmed generalization degradation

| Family, two seeds | Best observed valid WP | Latest checked valid WP | Epochs |
|---|---:|---:|---|
|plain GRU |0.681928 /0.682716 |0.666475 /0.669900 |best6/5, latest10 |
|GRU plus local attention |0.682724 /0.681294 |0.668074 /0.667245 |best6/5, latest9 |
|Mamba2 |0.680358 /0.682011 |0.651482 /0.660485 |best6/6, final12 |
|EMA/custom TabM-style |0.672715 /0.671281 |0.672695 /0.671266 |best11/11, final12 |

Plain GRU seed101001, epoch6→10: train diagnostic WMSE1.263953→1.165074 while
valid WMSE0.969962→1.009600 and valid WP0.681928→0.666475. Mamba2 seed101001,
epoch6→12: train WMSE1.260433→1.101592 while valid WMSE0.974665→1.047920 and
WP0.680358→0.651482. Both fitting seeds show degradation in each recurrent
family. These are strong overfitting/generalization-loss symptoms, despite healthy
processes. The curves alone do not identify whether regularization, objective
sampling, regime shift or another cause dominates.

The deterioration begins before the GRU infrastructure continuation at epoch8;
Mamba2 and local-GRU never used that continuation and show the same broad pattern.
There is no basis to attribute this pre-existing multi-family pattern to resume.
Train diagnostic uses a32-member fitting cohort/all-after-warmup mask; full valid
uses is_scored. Absolute curve separation is not a comparable train/valid gap.
The evidence above is each curve's own temporal deterioration/improvement.

Observed earlier peaks remain development-exposed selection evidence, distinct
from the frozen fixed final endpoint. Late12E weights should not automatically
become a submission candidate. Earlier weights and a separately declared
checkpoint/ensemble follow-up remain useful; weak final endpoints do not erase
that evidence or prove no combination benefit.

## B300 remains an early fixed-mixture test

Only three complete epochs were available at the native check. Labels mixtures
give0.692618919/0.692482021 versus clean TL3 reference0.692377425. Complement
mixtures give0.692117543/0.691319487. Each output is fixed90%TL3+10%Transformer;
these are not standalone Transformer scores. The dominant reference can mask a
weak component. A small mixture delta is not proof of a strong teacher or
distilled student, statistical significance, final30E improvement or public gain.

At epoch3 complement WMSE0.942259/0.943497 is below labels WMSE0.949035/0.948103,
but complement WP is lower. Better MSE is not sufficient evidence of improvement
in the actual WP objective. There is not yet a sustained B300 overfitting pattern
comparable to the L4 recurrent curves.

## Numerical/resource checks and clipping

All eight active fits had fresh native heartbeat ages below4seconds; four L4
fits were completed. Last100 native recorded loss/gradient samples per cell were
finite. Both GRU external safe-state archivers were live. Existing monitors do
not record a separate lifetime NaN/Inf counter; these are bounded tail checks.

The last200 recorded rank samples (logging cadence, not every update) show
99.5–100% clipping for GRU/local/Mamba. Median preclip norms were about2.1–2.8
for GRUs and4.0–4.2 for Mamba, against threshold1. Complement B300 had100%
clipping with medians17.8–19.0; labels about90% with medians2.7–2.8. This warrants
a separate clipping/objective-scale ablation, not a claim of exploding gradients
or a proven local minimum. Muon/AdamW make preclip scaling different from a
simple proportional effective-learning-rate statement. No live clipping change
was made from exposed validation. LR is finite and decays on the declared
schedule; completed12E L4 jobs ending at zero LR is expected.

## Retained weights and a repaired plotting gap

Native118 verified all eight best-observed checkpoint files exist and match
their epoch-index SHA256, including GRU/local epochs5/6 and Mamba epoch6. New
GRU full resume archives contain epochs8/9/10. Original indexes and old sources
remain unchanged. No fitting or model evaluation was needed for these checks.

The export watcher admitted only host labels l4/b300 and incorrectly rejected
the new b300-opt panel. Native metrics and dashboard mirroring were intact, but
standalone B300 epoch exports were missing. The exact known b300-opt host is now
admitted; unknown hosts remain rejected, as tested. Six focused CPU tests,
including the real plotter smoke test and unknown-host rejection, passed; Ruff
passed. Only the idle local exporter was restarted. Actual B300 PNG/SVG epoch
exports now exist, and the exporter continues without loading a model.

Primary validation remains complete1873 sequences/3528273 scored rows, official
train10325 only, zero valid fitting/ancestry, unchanged split
0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c.
Current numerical/operational health does not justify saying that all scientific
training is going well. Preserved curves show substantial L4 quality regression.
