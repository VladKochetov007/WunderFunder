# Learning dynamics and training side effects: literature -> our regime -> tests (2026-09-26)

## 1. Which regime are we in? (our evidence first)
- Label noise dominates: targets are piecewise constant between i0 price events, ~30 independent
  outcomes per class-3 sequence (physicist F3), direction near chance far from moves (class 3
  far-row WP 0.03-0.31). Heavy label concurrency (AFML ch. 4 "average uniqueness").
- Early-learning signature: best at 2 epochs, 3 epochs +0.0006, width/depth flat, SWA over the tail
  flat (E-052), seed residual corr 0.994, distillation +0.0013 alone but 0 on top of the recipe.
- Fine-tuning on valid rows with true masks transfers to the LB (+0.0014, E-144). Weight-space
  interpolation between parent and fine-tuned child (alpha 0.5) keeps ~90% of the gain with half the
  variance; extrapolation (alpha 1.5) hurts (E-147).
=> This is the noisy-label "early learning then memorization" regime (Arpit et al. 2017; Liu et al.
2020 ELR, arXiv:2007.00151), not the algorithmic-grokking regime. Grokking needs a hidden
low-complexity rule, near-zero train loss and weight decay pulling the norm into the "Goldilocks"
zone (Omnigrok LU mechanism, arXiv:2210.01117). Our train loss never approaches zero, so classic
grokking is unlikely; what transfers is the weight-norm view (weight decay / norm control) and the
epoch-wise double-descent risk under label noise (Stephenson & Lee 2021, arXiv:2108.12006: only
above a noise threshold; arXiv:2601.08316).

## 2. Directions, each mapped to a test
| id | idea (source) | why it might matter here | test | status |
|---|---|---|---|---|
| L1 | Long training + weight decay, look for delayed generalization / 2nd descent (Omnigrok 2210.01117; Grokfast 2405.20233; epoch-wise DD 2108.12006) | 2-epoch optimum may be the first dip of a double descent, or a norm effect | fold 0, 12 epochs, wd 0.01 vs 0.1, per-epoch valid (q13 B) | running |
| L2 | WiSE-FT weight interpolation (Wortsman 2109.01903) and task-vector extrapolation (Ilharco 2212.04089) | fine-tune is our only LB-transferring lever; interpolation hedges against shift for free | alpha 0.5 / 1.5 / 2 / 3 on fold models (q13 A) | alpha 0.5 ~= full FT with lower variance; >1 hurts |
| L3 | LP-FT: head first, then all (Kumar 2202.10054) | full FT distorts features when shift is large (test = new instruments) | head-only and LP-FT on fold models (q13 D) | running |
| L4 | Diverse weight averaging of FT children from one parent (DiWA 2205.09739; model ratatouille 2212.10445; soups) | zero inference cost; averaging needs a shared parent (our E-119 soup failed on different normalizations) | 4 FT children with different seeds / data orders / aux tasks, average weights | planned |
| L5 | Shrink & perturb before FT or before the final refit (Ash & Adams 1910.08475; plasticity: Dohare, Nature 2024) | warm-started training generalizes worse; our final = warm-started FT | theta <- 0.8 theta + 0.2 theta_init, then FT | planned |
| L6 | Self-distillation with teacher weight near/above 1 under strong label noise (Das & Sanghavi 2301.13304; born-again networks) | the theory says the optimal teacher weight grows with noise; we only ran alpha 0.5 | distill alpha 0.7 / 0.9 / 1.0 with a same-recipe teacher (fold models) | planned |
| L7 | Early-learning regularization: temporal ensemble of own predictions as a target regularizer (ELR 2007.00151) | prevents memorizing noise; could allow more epochs | ELR term on the main head, 3-4 epochs | planned |
| L8 | Label-uniqueness row weights and sequential-bootstrap sampling (AFML ch. 4) | long constant-label stretches dominate gradients with near-unpredictable rows | weight rows by 1/stretch length (causal variant: time since last change) — beware the bias E[w y|x]/E[w|x] | planned (bias check first) |
| L9 | Flat minima / domain generalization: SWAD dense averaging (2102.08604), SAM / DGSAM per-domain (2503.23430), V-REx risk variance over chains (2003.00688) | test = unseen instruments; chains are natural domains | SWAD over the whole run; V-REx penalty over chain groups in each batch | planned |
| L10 | RNN regularizers never tried: zoneout (1606.01305), variational dropout | dropout was always 0.0 | zoneout 0.1 / 0.2 on both GRU layers | planned |
| L11 | Self-supervised auxiliary tasks / next-event prediction (NEP foundation models 2607.09955; LOBERT 2511.12563) | the aux next-change head was our only shipped idea from scouts; more SSL heads = representation + ratatouille diversity | heads: next book-event type, next trade sign, next i1 move (links H-P2) | planned |
| L12 | Test-time training inside a sequence (TTT layers, Sun 2407.04620; TS test-time adaptation PETSA 2506.23424) | each sequence is a new instrument; within-sequence state is allowed; SSL targets are observable online | small TTT-linear layer on next-event SSL; check 1-vCPU cost first | long shot |
| L13 | Regression mixup (C-Mixup 2210.05775) | OOD robustness in regression; label-similar mixing | mix hidden states of sequences with similar |y| profile | low prior |
| L14 | minGRU/minLSTM parallel scan (Feng 2410.01201) | 10-100x faster training -> more seeds/arms; a different inductive bias for diversity | port a minGRU member behind the residual-corr gate | engineering |

LOB generalization literature: LOBCAST benchmark (2308.01915) finds every published LOB deep model
drops sharply on new data; hybrid conv + recurrent models generalize best. Consistent with our
train/valid AUC 0.99 and the value of the baseline blend as a shift regulariser (E-129).

## 3. Kept from the panel plan (not excluded)
Baseline-member FT (X1), prop_v2 selection model (H-P3), i1-lead probe (H-P2), seed-sigma ladder,
native engine speedup, gcorr objective member, rows-to-next-event slice.
