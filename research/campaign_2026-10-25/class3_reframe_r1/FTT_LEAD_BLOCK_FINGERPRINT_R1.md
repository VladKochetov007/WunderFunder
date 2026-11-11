---
kind: analysis
status: partial
created_utc: 2026-10-25T07:40:00Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - submissions/tl3_two_parents_20261022/tl3_two_parents_20261022.zip
---

# What are FTT's 52 "lead" inputs? Zero-fit fingerprint from the served ZIP

**Question.** TL3's score mass sits in the two `fttall_tlead` members (263 inputs = our 211
+ 52 "lead" features); their training source was never located (Oct-24 value review).

**Served architecture** (from `tl3_two_parents_20261022.zip`, members `fttall_tlead_s5/s6@ep3`):
docstring "all-data tied-encoder parents (shared i0/i1 book MLP, i1 lead block), dynamic
int8"; tensors `mean/std (263)`, `inp (256×263)`, `tied.0 (128×104)`, `tied.2 (64×128)`,
`tied_out (256×128)`, `lead_out (256×64)`, GRU 256×2, head 512→256→2. Native library
strings: `TransitionFeatures::update/reset`, "tiedlead model input width mismatch",
"lead_out without tied encoder", "C++ twin of wnn.nn.RNNModel + event features". The
feature code lives only in the teammate's native source (not in this repository).

**Fingerprint.** Normalization stats of inputs 211:263 match the raw i1 block (columns
52:104): mean correlation 0.9986, per-column mean shift ≤ ~0.003; std ratio extra/i1 =
0.92–1.00 (corr of std vectors 0.926). On 60 valid sequences no causal EMA reproduces the
ratios (half-life 1: RMSE 0.070, corr 0.861; longer half-lives worse), and first
differences are far smaller (ratio 0.3–0.6). So the block is **i1 raw values transformed
in time with almost no smoothing** — most consistent with i1 values lagged or held on an
event clock (e.g. sampled at i0 book events), not an EMA and not differences.

**Implication.** Our `i1_history` research features (367 slots; i1 mixture +0.0003,
below noise) are the same family, so the FTT advantage is plausibly the **tied i0/i1
encoder** (a shared MLP over the two books plus a separate lead path) as much as the
features. Concrete teammate ask: the `TransitionFeatures` lead-block definition (which
clock/lag) and a matched ablation (tied encoder with vs without the 52 lead inputs).
Limits: stats-only inference; FTT stats come from all-data training, my comparison from
valid sequences; no member was re-run.
