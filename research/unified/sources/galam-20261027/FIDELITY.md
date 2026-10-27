# GALAM reconstruction fidelity

Status: **source-traced schematic reconstruction, not exact reference replication**.
Reference image: [unchanged supplied diagram](galam-original-diagram.jpg).
The user explicitly requires exact reconstruction. Missing source details remain
unknown; this requirement cannot be satisfied by silently inventing them.

## Source-to-code map

| Diagram observation | Code | Fidelity and limits |
|---|---|---|
| Parallel global/local memories | `src/wnn/galam/model.py` | Both branches implemented. Number of repeated blocks is unspecified in the source. |
| Local BatchNorm1D → LSTM | `mixer.py`, `model.py` | Per-time, across-batch normalization avoids future-time leakage. Author's BN axis/statistics convention is unknown. |
| Temporal self-attention on LSTM hidden history | `attention.py`, `model.py` | Current and cached past only; LSTM hidden/cell also have separate Linear→GELU paths. Exact attention history and recurrent depth unknown. |
| Query/key RMSNorm | `attention.py` | Applied after Q/K projections and before scaled dot product. Head count, norm epsilon and parameter-sharing conventions unavailable. |
| Relative timestamp bias: Linear→Tanh→Linear; add after scaled dot product | `attention.py` | Learned additive bias before softmax. Timestamp source and scale are explicit assumptions; ordinal mode uses row position, never seq_ix. |
| Global projection → attention residual → Post-LayerNorm | `model.py` | LayerNorm is distinct from local/decoder RMSNorm and must match the picture. |
| ConvMixer: Conv1D→GELU→BatchNorm1D twice | `mixer.py` | Order is read bottom-to-top along the image arrows. Causal padding is required. Kernel, expansion channels and grouping unknown and must be declared. |
| ConvMixer residual → Post-LayerNorm | `model.py` | Same source distinction from RMSNorm. |
| Fusion Gate: Linear→SiLU→Linear→sigmoid | `fusion.py` | Both branch values and projected hidden/cell enter fusion. Exact gate operands, bypass multiplication and reduction equation are not recoverable from the diagram. The chosen equation is an explicit assumption. |
| Elementwise product after Fusion Gate | `fusion.py`, `model.py` | Visible multiply implemented. Four-channel concatenated bypass and its ordering are mandatory declared assumptions; the chosen equation is not verified against original code. |
| Decoder Post-RMSNorm, FFN Linear→GELU→Linear with residual, Post-RMSNorm, final Linear | `model.py` | Visible operations are preserved; FFN dimensions and detailed residual/fusion operands remain unknown. |
| 32 feature next-state prediction | historical config | 32→32 template. All hidden widths and context choices are illustrative, not the original hyperparameters. |
| Current Connectome inputs/outputs | adaptation config | 112→2 is a separately labelled adaptation, not literal historical identity. |

## Not established by these sources

Original weights, exact hidden/FFN/bias/ConvMixer dimensions, head counts, block
counts, normalization eps/momentum, context window, precise fusion arithmetic,
optimizer/scheduler/initialization recipe, SWA epochs/weights and four-model FiLM
architecture are missing. No public source found in the bounded search resolves
them. The author's Discord report states the original recipe was lost after an
SSD failure. A correct implementation of the visible graph is therefore not proof
that the historical numerical model has been restored.

The Discord text calls the global branch GeGLU/MLP-Mixer, while the supplied image
shows GELU/ConvMixer. This implementation follows the supplied image. A GeGLU
variant, if added, must have its own declared configuration and comparison.

## Checks and claim boundary

Deterministic CPU tests check prefix causality, state reset, streaming/chunk
consistency, explicit TBPTT detachment, config validation, source-visible operation
order and adapter compatibility. They do not establish historical numerical parity,
training quality, CPU submission runtime or B300 throughput. A receiving-host
data/ancestry audit and a separately authorized profile are required before fits.
