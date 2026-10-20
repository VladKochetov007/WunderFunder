---
kind: method
status: complete
created_utc: 2026-10-19
hypothesis_ids: [H-LARGE-DYNAMICS-TEACHER-20261019-R2]
experiment_ids: [E-LARGE-DYNAMICS-TEACHER-SCALE-20261019-R2]
evidence_paths: []
---

# Primary sources for the large B300 dynamics teacher

These sources were inspected through primary arXiv, NVIDIA, PyTorch and author
repository pages on October 19, 2026. They inform the
[design](../ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md); none supplies a
measured result on our dataset. No external data or pretrained weights were added.

| Source | What was checked and how it informs this plan | Limit |
|---|---|---|
| [One-Step Gradient Delay is Not a Barrier for Large-Scale Asynchronous Pipeline Parallel LLM Pretraining,2606.30634](https://arxiv.org/abs/2606.30634), user-supplied [PDF](https://arxiv.org/pdf/2606.30634) | Abstract, delayed-update definition, optimizer comparison and setup in the primary PDF. Studies fixed one-step delay, Muon and a correction for asynchronous pipeline updates. | LLM evidence; not a demonstration that async training or DDP improves this time-series task. We propose synchronous DDP first. |
| [NVIDIA DGX B300 hardware](https://docs.nvidia.com/dgx/dgxb 300-user-guide/introduction-to-dgxb 300.html) | Specifies eight B300s and 288 GB per GPU for DGX. Informs conditional capacity accounting. | Does not establish actual operator allocation, free memory, node topology or strict-FP32 throughput. |
| [PyTorch Muon API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) | Hidden2D matrix use, shape-dependent LR conventions including match_rms_adamw. | Our time-series LR values are proposals; semantic input/readout groups must be chosen explicitly. |
| [PyTorch Muon source](https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py), inspected [raw file](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/_muon.py) | `_zeropower_via_newtonschulz` explicitly converts the working matrix to bfloat 16. | Mutable upstream branch, not a runtime pin. The receiver must inspect/pin its installed version; unmodified inspected source violates our strict-FP32 contract. |
| [Muon is Scalable for LLM Training,2502.16982](https://arxiv.org/abs/2502.16982) | Primary abstract supports attention to weight decay and update-scale conventions. | Published language-model efficiency is not a local gain or runtime forecast. |
| [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) | Math reference/backend selection and differing backend support. | Actual FP32 forward/backward and memory must be checked at the intended shape. |
| [FlexAttention API](https://docs.pytorch.org/docs/2.14/nn.attention.flex_attention.html), [FP32 handling discussion](https://pytorch.org/blog/flexattention-for-inference/), [FA4 integration](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) | Custom causal/local masks and block-sparse computation are relevant engineering options. | API/prototype and backend-dependent precision; FA4 examples in BF16 do not certify a strict-FP32 path. |
| [PyTorch DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html) | Replica semantics, gradient synchronization and accumulation context. | No actual eight-card speedup measured; our weighted-loss normalization is an explicit derivation. |
| [Numerical feature embeddings,2203.05556](https://arxiv.org/abs/2203.05556) | Piecewise-linear and periodic scalar bases motivate a numeric-encoder arm. | Tabular evidence does not establish causal dynamics or a Transformer gain here. |
| [RoFormer,2104.09864](https://arxiv.org/abs/2104.09864) | Rotary relative-position mechanism. | Row order is not physical elapsed time. |
| [Query-Key Normalization,2010.04245](https://arxiv.org/abs/2010.04245) | Normalized queries/keys with a learned score scale motivate an attention-stability arm. | Must specify its scaling correctly; language-translation results do not establish market-regression improvement. |
| [Gated Attention,2505.06708](https://arxiv.org/abs/2505.06708) | Query-dependent attention-output gates motivate a small noisy-write suppression arm. | Published LLM improvements are not local empirical evidence. |
| [Transformer-XL,1901.02860](https://arxiv.org/abs/1901.02860) | Segment recurrence is a precedent for longer causal memory. | Our completed-input-block slow stream is a separate proposal, not a reproduction of Transformer-XL. |
| [Timer,2402.02368](https://arxiv.org/abs/2402.02368) | Next-observation pretraining is a time-series precedent. | No reason to assume its data/patch normalization recipe transfers unchanged to our multivariate per-row targets. |
| [MiniCPM / WSD,2404.06395](https://arxiv.org/abs/2404.06395) | Stable training with later decay motivates extendable teacher horizons. | The1500-update warmup, cosine cooldown, horizons and LR in our plan are locally proposed settings. |
| [Scaling Data-Constrained Language Models,2305.16264](https://arxiv.org/abs/2305.16264) | Repeated-data training has diminishing returns; useful caution for large teachers on fixed data. | Do not transfer language-token scaling coefficients or an epoch cutoff to correlated market rows. |
| [On the Efficacy of Knowledge Distillation,1910.01348](https://arxiv.org/abs/1910.01348) | Teacher capacity/accuracy need not order student performance. | Vision evidence motivates a local teacher/student test; it does not predict a GRU capacity ceiling. |

## Search and inspection provenance

The root used web search with arXiv primary queries for Muon scaling, numerical
embeddings, QKNorm, WSD, causal time-series pretraining, data-limited scaling and
distillation. It opened the supplied 2606.30634 PDF/abstract and primary NVIDIA,
PyTorch and raw optimizer sources. No local arxiv-search helper command is claimed.
The arxiv-search skill guided primary-paper discovery; API/documentation inspection
was necessary to verify dtype and parallelism details. Titles and links above
identify the actual inspected sources, not hypothetical suggested readings.

Independent Luna/max scout `transformer_scale_scout` checked the major hardware,
optimizer, attention and teacher-transfer sources in a read-only task. The
independent specification critic inspected the original proposal and repository
contracts. Their public findings and synthesis limits are recorded in the
[review](../reviews/LARGE_TEACHER_DESIGN_REVIEW_20261019_R2.md).

The parameter, memory, DDP-weighting and runtime equations in the design are our
analytical accounting. Hardware support and model quality still require actual
source-bound profiles and matched teacher/student endpoints.
