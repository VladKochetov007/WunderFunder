# Column view-mask (SpecAugment-style) members — cold-recipe screen (R1)

**Question.** Can a train-only view augmentation make a *separate* network more diverse
(lower ρ to the standard model) without making it weaker — the regime the blend law rewards and
TabM failed to reach? Source: [reframe Opus #4](../reframe_r1/opus_out.md), SpecAugment
([arXiv 1904.08779](https://arxiv.org/abs/1904.08779)).

**Intervention.** `--view-mask-p p` (train-only): each input column is set to its mean for a whole
sequence with probability p; serving unchanged (test in `tests/test_tabm_gru.py`).

**Fits.** Kaggle T4, cold Muon recipe, F0/s2001 and F2/s2002, p ∈ {0.15, 0.30} (4 fits);
controls reused: R3 `krepro_*_muon_r3`. Dataset `vladkochetov07/wnn-vmask-20261029-r1`
(READY, 741,995 B), kernel `…-r1-runner` v1 pushed 02:47 UTC. [Specs](specs/), [builder](build_specs.py).

**Decision rule (declared before results).** Report standalone Δ, ρ to control and the pair
(masked + control, 0.5/0.5) minus control. A masked member is a useful diverse member if, on both
folds, pair − control > +0.0010 (a typical seed partner gives +0.0005–0.0006 over the better
member on this panel) and standalone Δ ≥ −0.0015. Then: diverse-view teachers for distillation
and as blend members.

## Results

Pending.
