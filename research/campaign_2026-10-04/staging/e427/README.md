# E427 isolated implementation draft

This folder is an overlay only. It does not modify the sealed root trainer, model, or scorer. `scripts/train_e427.py` is a staged copy of the current trainer with the E427 architecture, parent-state transfer, full-sequence side-context dispatch, and post-optimizer structural projection hooks. Its import path puts this folder's `src/wnn` first, then extends to the unchanged project `src/wnn` modules. The copied trainer anchors `ROOT` at the project root.

The parent model is the exact public E424 seed-3 fixed epoch-3 checkpoint. E427 reads its `n_in` and strict-copies its state; the observed parent width is 211 (112 raw + engineered inputs), hidden 256, 2 GRU layers, and 23 outputs. The original input projection, feature construction, normalization, head, teachers, labels, and loss remain intact. Only layer-0 GRU `weight_ih_l0` is widened by 8 context coordinates. Context is computed from the first 112 raw values of the original input tensor and never uses targets or scored masks.

For production dimensions, each arm has exactly the same parameter layout:

| Count | Parameters |
|---|---:|
| Parent ordinary model trainable | 980,759 |
| E427 allocated extra coordinates | 6,376 |
| E427 reset/candidate coordinates projected to zero | 4,096 |
| E427 effective added learnable coordinates | 2,280 |
| E427 optimizer-visible trainable total | 987,135 |
| E427 effective trainable total after fixed coordinates | 983,039 |

The widened `nn.GRU` remains a fused two-layer GRU. Structural `r`/`n` context columns are checked as exact zeros after every optimizer attempt and after EMA updates. This is essential for the Muon recipe; the gradient hook is only supplementary. The student and EMA teacher share a pure, input-keyed aligned feature / stale-row-index cache so the full `[32,20000,28]` feature stream is materialized once per batch, not once per model. The main feature array is about 71.7 MB (68.4 MiB); temporary FP64 cumulative sums and the row-index stream have not yet been measured in the bounded worker. Root reported a recent baseline peak of about 4.49 GB with a 4.8-GB stop threshold, so this arithmetic estimate does not establish headroom. The full-batch smoke must measure actual allocator peak before any fit admission.

`prepare_context_stats.py` scans only raw `x` columns for the exact parent TRAIN+fit-VALID group lists, rejects forbidden TRAIN row group 528, binds the stats to the exact parent checkpoint and provenance plus both Parquet hashes, and writes the preregistered deterministic endpoint-window sample. It requests no labels or evaluation groups.

Drafts are under `specs/`: one bounded 1-CPU/2-GiB synthetic CPU test job, one 1-CPU/2-GiB fit-context stats job, and three matched one-epoch fit recipes (`aligned`, `stale`, `noop`). They are unsealed and not launch authorization. No model imports, tests, fits, exports, or launches were run locally. The parent must seal source/input hashes and run the bounded CPU tests and a full-batch throughput/memory smoke before deciding on fit admission.

Known follow-up: E427 checkpoint scoring/export requires a wrapper that reconstructs this staged model and recomputes the same causal volume context on full VALID sequences. The root `predict_run.py` is deliberately untouched, so native/deployment readiness is not claimed here.
