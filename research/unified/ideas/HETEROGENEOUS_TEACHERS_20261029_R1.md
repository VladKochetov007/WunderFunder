---
kind: idea
status: proposed
created_utc: 2026-10-28T21:11:23Z
hypothesis_ids: [H-HETEROGENEOUS-TEACHERS-20261029]
experiment_ids: []
evidence_paths:
  - ../../experiment_drafts/galam-ensemble-diagnostics-20261028/RESULTS.md
  - ../../experiment_drafts/tl3-chronological-20261028/README.md
---

# Heterogeneous teachers before scaling seeds

## Question or claim

The user proposes GRU/LSTM/GALAM/other teacher families, followed by low-LR
correlation fine-tuning and distillation. Test whether complementary families
improve the fixed-capacity GRU student on full official valid. This is a proposed
successor, not an amendment to the frozen17-fit TL3 graph or permission to start
28 teachers. No code, GPU task, resource allocation or remote publication starts
from this note. Owner: root/tabdata.

## Mechanism and method

Architectural names and distinct seeds do not establish useful diversity. Inspect
weighted residual correlations, causal-regime coverage and the actual marginal
pooled WP of a predeclared combination. Prediction disagreement can be noise.
Use family-level weights so adding seeds to one family does not increase its
total weight. Preserve each base checkpoint when testing fine-tuning.

Candidate pool: reuse existing clean GRUs and schematic GALAM; initially add a
plain LSTM on the same causal211 inputs and a causal dilated TCN. Subsequent
families can include bounded-window causal attention and causal EMA/random-
projection weighted ridge. Layer/input preprocessing must not inspect future
rows. The TCN's receptive field, causal padding and the LSTM adapter need code
and receiving checks before a runnable recipe is claimed.

Proposed first screen: **6 total new scientific fits**:

1. One fresh LSTM teacher and one fresh causal TCN teacher:2 fits.
2. Two GRU students from the existing GRU-only teacher pool:2 matched seeds.
3. Two identical GRU students from the heterogeneous pool:the same2 seeds.

Existing GRU/GALAM ancestors are reused and are not counted as new fits. Their
prediction exports, alignment and support compute must still be budgeted.
Teacher and student endpoint epochs, input/optimizer/sampler/target treatment
must be frozen before their new valid results. In the paired students, change
only the teacher mixture. A proposed fixed family mixture is70% existing GRU
family mean,10% GALAM-LSTM128 epoch3,10% new LSTM and10% new TCN. This is an
explicit heuristic arm, not fitted or optimal weights. Student architecture,
distillation coefficient and bootstrap rule stay identical across each pair.

The existing complete scientific recipe is reused where applicable; new LSTM/
TCN source, exact hyperparameters, measured profiles, disk headroom and numeric
runtime bounds remain admission prerequisites. No runtime or score gain is
invented for unprofiled architectures. The current budget covers a proposal,
not admission of these6 fits. A blanket28-base-plus28-fine-tune allocation would
already require56 fits before new students; it is a later scaling decision.

Do not make GCorr fine-tuning compulsory for every family. First retain the
current base/FT evidence; isolate a future fine-tune ablation if useful. Ridge
and tree models require their own fitting procedure. This six-fit screen adds
no new correlation-fine-tuning runs.

## Evidence and interpretation

The predeclared80/20 clean-GRU/GALAM-LSTM128 blend gaveWP0.681219679 versus
0.677856123 for the GRU; prediction correlations were0.956–0.962. Thus a useful
combination can exist even with high prediction correlation. This does not show
that GALAM improves the newer, stronger TL3 students.

The [PR49 terminal report](https://github.com/VladKochetov007/WunderFunder/pull/49#issuecomment-6484849146)
reports16 completed fits and W18 blocked on cache_unavailable. Fixed endpoints:
fbc1k_s5 WP0.691355; tiedlead_s5 WP0.691415; tiedlead_s6 WP0.690949. These are
reported artifact-verified development scores, not a completed three-member
TL3 score or public results. The current status request could not reach the node
after the old SSH connection reset and one new connection timed out; no live
GPU/process status or underlying cache-error log was inspected.

All6 fixed-endpoint correlation-fine-tune branches score below their respective
base parents in that report. This is evidence about the whole fine-tune recipe:
loss, optimizer/LR, EMA and since-event weighting also differ. It does not isolate
GCorr as the causal failure. It argues against mechanically attaching this
unchanged stage to every new architecture.

## Limits and validity

Use ADR0044:10325 fitting train members,0 fitting valid members, all1873 valid
sequences and the official mask. No seq_ix chronology. All normalizers, teacher
weights, initializers and cached fitting targets must exclude valid ancestry.
Do not fit mixture weights on primary valid. Any learned weighting requires
fitting-only/group-cross-fitted training evidence; random training OOF is not
an additional chronological primary holdout.

The primary endpoint is student pooled WP, not teacher pool WP or a small error
correlation. Report both paired student deltas and train/valid curves. Two
student seeds do not establish significance or estimate teacher-training variance.
Extra teacher compute is intentional; this first screen does not isolate
architectural diversity from additional teacher compute. Uncertain/negative
results concern this composition and weighting, not every heterogeneous pool.

## Decision and next step

Prepare the6-fit matched screen as a separate draft after preserving the frozen
TL3 results and diagnosing its cache handoff. A useful working threshold is
student delta+.0005; repeat/extend when the paired results justify it. Teacher
pool gain without student gain directs attention to capacity or target fusion,
not automatic expansion to7 seeds per family. No new job has been admitted.

## Source records

- [Existing combination evidence](../../experiment_drafts/galam-ensemble-diagnostics-20261028/RESULTS.md).
- [Current TL3 contract](../../experiment_drafts/tl3-chronological-20261028/README.md).
- [Chronological primary policy](../../../specs/adr/0044-chronological-primary-validation.md).
- [Ding et al., AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/view/29746):
  more/stronger teachers were not universally better in their NLP/CV experiments;
  transfer to this regression task remains a hypothesis.
- [Du et al., NeurIPS2020](https://proceedings.nips.cc/paper_files/paper/2020/hash/91c77393975889bd08f301c9e13a44b7-Abstract.html):
  conflicting ensemble-teacher signals motivate testing the fusion rule and
  student endpoint, rather than assuming uniform averaging transfers every gain.
