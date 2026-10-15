---
kind: method
status: proposed
created_utc: 2026-10-15T16:29:33.670324+00:00
updated_utc: 2026-10-15T16:29:33.670324+00:00
hypothesis_ids: [H-FROZEN-PARENT-ANCHOR-20261015]
experiment_ids: []
evidence_paths: []
---

# Sources for preserving inherited predictions during continuation

## Question or claim

Can teacher matching help by preserving an inherited function, without adding
information from a different teacher? These primary sources motivate controls;
they do not establish usefulness for our GRUs or competition.

## Mechanism and method

[Li and Hoiem, Learning without Forgetting, ECCV2016/journal revision2017](https://arxiv.org/abs/1606.09282)
uses predictions of an earlier network to retain prior capabilities while
learning from new-task inputs. The transferable component is a fixed prediction
reference. Our same-task sequence regression/MSE proposal is an adaptation,
not a reproduction of their classification/multitask experiment.

[Li, Grandvalet and Davoine, Explicit Inductive Bias for Transfer Learning with
Convolutional Networks, ICML2018](https://proceedings.mlr.press/v80/li18a.html)
studies retaining proximity to pretrained parameters and recommends a parameter
penalty centered on the starting model. This motivates an inexpensive parameter
anchor as a distinct control; it does not imply equivalence to output matching
or transfer of their vision results to recurrent competition models.

## Evidence and interpretation

Both primary pages were opened on October15. No code from either source was
executed. Our local anchor hypothesis remains untrained. The early alpha0/.9
comparison cannot distinguish informative new targets from inherited-function
preservation, because it removes both simultaneously.

## Limits and validity

Same-task regression, long causal state, fixed fitting membership and exposed
folds differ from the source settings. Function and parameter proximity impose
different constraints. Neither paper predicts a local score or public gain.

## Decision and next step

[Freeze a parent-output control](../ideas/FROZEN_PARENT_ANCHOR_CONTROL_20261015_R1.md)
after the current six-epoch trajectory; retain capacity/age fits independently.

## Source records

Primary links above; [current trajectory](../../campaign_2026-10-15/student_quality_tradeoff_r1/README.md).
