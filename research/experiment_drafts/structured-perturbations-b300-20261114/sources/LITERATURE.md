# Primary motivation, checked 2026-11-14

- Ghiasi, Lin, Le, [DropBlock](https://arxiv.org/abs/1810.12890): correlated features
  can retain information despite independent dropout; structured omissions were
  studied on image tasks. We borrow the comparison principle, not its image
  results, schedule or numerical gains. Our past-only time-series mechanisms are
  different implementations and are unmeasured.
- Zhou et al., [Scheduled DropHead](https://aclanthology.org/2020.findings-emnlp.178/):
  structured head removal is a Transformer regularizer. Here heads remain intact;
  only old temporal keys are intermittently unavailable. This is motivation for
  testing structural granularity, not a reconstruction of DropHead.

Neither source establishes label invariance, market robustness, profitability or
an expected challenge score. Team masking negatives and existing implementations
are the closer evidence; see [PR_AUDIT.md](PR_AUDIT.md). No third-party model code
or weights were copied from these papers.
