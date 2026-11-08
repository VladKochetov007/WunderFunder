# Fresh compact architecture cohort

Eight recipes implement the authorized four-family/two-seed comparison. Seeds
101001 and101002 are shared fitting seeds, not eight independent replications.
Every recipe has12epochs and fixed final checkpoint selection; intermediate
epoch checkpoints/curves are descriptive. There is no early stopping, KD,
initializer, teacher cache or inherited preprocessing.

All arms use raw112, the canonical train-only all-fit normalizer over the frozen
10,325 fitting sequences, all training rows from row99, and the full1,873-member
official validation boundary. Every context row advances state; only official
need_prediction AND is_scored rows enter validation scoring. No validation row
enters normalizer, model or parent ancestry. Official valid remains exposed.

The shared frozen optimization policy is AdamW, LR0.0003, beta(.9,.999), eps1e-8,
weight decay.01 on every parameter, clip1,5%warmup and warmup_cosine_v2 to zero.
Precision is strictFP32 and each recipe requires one actual NVIDIA L4. Numerical
runtime candidates currently use batch8/chunk2000/accumulate1 and evalbatch8;
these values are provisional until ROOT profiles all families. Freeze any
matched change in new recipe/source identities before scientific fitting.

| Family | Declared graph | Persistent state |
| --- | --- | --- |
| gru | Fresh canonical GRU256x2 and its canonical encoder/readout | Two hidden vectors |
| gru_local | GRU256x2 parallel with d128 two-block RoPE local attention,4heads/window256, gated residual fusion, direct linear current-input head | Canonical GRU state plus each block's255-row KV cache and absolute row position |
| mamba2 | d128 six pre-norm residual genuine Mamba-2 blocks, expand2/state16/head32/conv4/SSD chunk256, final LayerNorm and linear head | Exact convolution history and SSD state per layer |
| ema_tabm_style | Learned32-channel input projection, fixed causal half-lives32/256/2048/8192 rows, current input/first difference/EMA features, two-value scalar embeddings, three256-wide rank-one ensemble layers with4members | Four32-channel EMA states and previous standardized112-vector |

GRU/local attention is a new adaptation, not an exact GALAM or Denis Tomin
reconstruction. Both its stems receive the same once-standardized input using
the adapter's fitting-only registered statistics; branch normalizers are identity.
Each component is pointwise or causal; no time-axis normalization is performed.
The local memory field is bounded but the GRU/Mamba/EMA states span the prefix.
TBPTT detaches architecture-owned state only at canonical trainer boundaries;
state resets only at whole-sequence starts.

EMA/TabM-style is deliberately named as an adaptation. Its custom scalar GELU
embeddings and shared dense weights with member-specific input/output rank-one
factors are Torch-only. It does not use the official tabm dependency or claim
official TabM initialization/architecture equality. The canonical supervised
objective trains the mean of the four predictions, unlike official TabM's
memberwise training objective. The causal EMA parallel prefix scan avoids
division by tiny decay powers. Fixed decay constants are row half-lives, not
physical timestamps or parameters tuned on validation.

Mamba-2 preserves the inspected historical backend pin: mamba-ssm2.2.4 from Git
commit95d8aba8a8c75aedcaa6143713b11e745e7cd0d9. Install with the receiving host's
existing Torch/CUDA toolchain, without changing its shared environment:

```text
mamba-ssm @ git+https://github.com/state-spaces/mamba.git@95d8aba8a8c75aedcaa6143713b11e745e7cd0d9
```

The backend requires Git direct_url.json metadata matching that complete commit.
The wrapper uses official Mamba2 parameters/initialization and official
mamba_chunk_scan_combined with differentiable initial/final states. Its exact
causal depthwise Conv1d uses Torch so causal-conv1d is optional. Upstream's full
forward with use_mem_eff_path=False is retained as the independent oracle.
No CPU substitute is supplied: actual CUDA parity/backward and finite-update
profile receipts remain mandatory. All-parameter AdamW intentionally also
decays the dynamics/vector parameters, matching the declared shared recipe.

Implementation reuses build_sequence_model/load_config and the canonical
train_sequence.py runtime; there is no copied trainer. Generic state detachment
recognizes an architecture-owned detach() method. Source/config/runtime/data
identities remain bound by canonical receiving admission.

CPU verification uses tests/test_compact_sequence_models.py for current/future
perturbation, full/chunk/one-row parity, reset, detached/attached gradients,
normalization parity and the eight-cell recipe contract. The optional real GPU
test additionally checks every Mamba block against official full forward. Local
CPU results cannot establish GPU throughput, final score or CPU serving cost.
Independent review and receiver-owned actual device profiles precede fits.
