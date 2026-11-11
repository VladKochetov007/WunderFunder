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

## Results ([F0 grade](grade_kaggle_f0_s2001.json), [F2 grade](grade_kaggle_f2_s2002.json), [local grade](grade_local_f0.json))

| Cell | p | standalone Δ [95% CI] | ρ to control | pair − control |
|---|---|---|---:|---|
| Kaggle F0/s2001 | 0.15 | −0.00140 [−0.00386, +0.00052] | 0.9941 | +0.00025 |
| Kaggle F2/s2002 | 0.15 | **−0.00298 [−0.00457, −0.00143]** | 0.9949 | −0.00067 |
| Local F0/s1001 (control `tabm29_f0_s1001_muon_local_r1`, [runner](run_local_p15.sh)) | 0.15 | **−0.00257 [−0.00558, −0.00042]** | 0.9947 | −0.00045 |
| Kaggle F0/s2001 | 0.30 | **−0.00694 [−0.01390, −0.00217]** | 0.9866 | −0.00146 |
| Kaggle F2/s2002 | 0.30 | **−0.00676 [−0.00930, −0.00430]** | 0.9890 | **−0.00173** |

**Decision rule failed in every cell.** p=0.15 weakens the model without adding diversity beyond a
seed (ρ ≈ 0.994–0.995); p=0.30 buys seed-beating diversity (ρ ≈ 0.987–0.989) at a 0.007 strength
cost. H-VMASK29-R1: **screen_failed** for column masking on this recipe. Not tested: time-span
masking, masking in the distillation-teacher role (no reason to expect it to escape the same
trade-off).

**Cross-family view ([plot](../tabm_gru_r1/plots/diversity_sources_tradeoff.png), marimo
[notebook](../tabm_gru_r1/ensemble_eda.py)).** 21 candidate members vs their controls: only seeds
and small recipe variants (Muon rms 0.3, H512×2) sit below the break-even line
deficit < 2·0.168·(1−ρ); AdamW sits on it; TabM members, view masks and AdEMAMix sit above it.
On this panel every structural diversity source so far has cost at least as much strength as its
diversity is worth. Seeds remain the only "free" diversity, and they are worth ≈ +0.0005–0.001.

Fits: 5 (4 Kaggle, 1 local).
