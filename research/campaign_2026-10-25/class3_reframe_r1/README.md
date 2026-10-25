# Class-3 fresh-context reframe — 2026-10-25 R1

Fresh-context reframe (research-reframe) of the real bottleneck: class 3 (largest tick,
~46% of public metric weight, corr ~0.44) where the hidden quantity is F's move relative
to the quotes. [Brief](external/BRIEF_R1.md) states only objective, target law and
decisive observations.

- [GLM 5.3 flash](external/oc_glm_R1.md): seven mechanisms — virtual-label state machine,
  branch-first classification with exact jump constants (locked-book identity), queue
  depletion race + Hawkes clustering, i1 lead-lag conditioning, change-vs-carry regime
  split, iceberg refill signature, discrete Bayesian filter over branch/level.
- [Grok 4.7, R2](external/grok_4.7_R2.md) (R1 [stalled](external/grok_4.7_R1_attempt1_stalled.md)
  after loading repo skills; rerun from an isolated scratch directory): absolute-F lattice
  projection, latch forecast across runs, Cont–de Larrard depletion-race formula, i1 as
  fine-grained twin of F, reload state, event-key codebook, Hawkes surprise. Flags a
  possible sp+1 vs sp+2 off-by-one in the outside-branch law.
- Root triage: "latch/copy-forward" is already falsified (causal segment-hold 0.607 vs
  0.669, Oct-11 residual EDA). Lattice projection replaces an expectation with a point
  and is expected to hurt Pearson unless very confident. The shared decisive gate for the
  remaining information claims (depletion, refill, Hawkes, i1 lead) is whether causal
  hand features explain the strong GRU's held-out residual:
  [residual_info_audit.py](residual_info_audit.py) (running).
