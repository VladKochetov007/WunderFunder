# Blackwell precision amendment — 24 October 2026

The user confirmed that mandatory strict-FP32 model computation was withdrawn.
The active four-teacher successor uses **BF16 autocast plus process-wide cuBLAS
BF16x9 emulation**. R4 now adopts that explicit numerical arm; the earlier
strict-FP32 R1/R2/R3 artifacts remain historical references, not current launch
instructions.

## Current team execution and provenance

[PR #40](https://github.com/VladKochetov007/WunderFunder/pull/40) implements the
selected runtime. Its scientific source is
`f1beeb4c210a9da01735e03b1ff6e27be878b873`; later operator/documentation revisions
are distinct. [The measured batch amendment](https://github.com/VladKochetov007/WunderFunder/pull/32#issuecomment-6377750553)
records four fresh 30-epoch teachers on F0/F2, two B300s each: 630M at 96/rank
(global192) and 1.54B at 64/rank (global128). These are the team's active four
fits, not executions of the conditional R4 B–H campaign. Track evaluated
endpoints separately from optimizer progress in the
[live report](https://github.com/VladKochetov007/WunderFunder/pull/32#issuecomment-6377032958).

The R4 integration ports the canonical runtime and its prepared tests from that
scientific source (18 source/test files). It adds a FP32 cast at the GRU recurrent
state boundary for BF16 streaming, precision-aware ensemble ancestry checks,
and R4 configuration/admission compatibility. This combined source is new:
PR40's remote test/profile receipts do not admit it automatically. No active
host processes, controllers, running source pins or checkpoints are modified.

## Actual arithmetic contract

| Part | Selected contract |
|---|---|
| Config | `precision_mode=bf16_bf16x9` |
| Transformer compute | BF16 forward autocast; exact local-window efficient SDPA |
| Transformer execution | `compile_blocks=true`, `checkpoint_blocks=false` |
| GRU compute | BF16 forward autocast; no Transformer compiler/attention flags |
| Master weights, gradients, loss, optimizer and carried state | FP32 tensors |
| KD target export and arithmetic ensemble | Finite FP32 arrays and FP32 mean |
| TF32 / AMP scaler | Disabled |
| Process environment | `CUBLAS_EMULATE_SINGLE_PRECISION=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

BF16x9 is a process-wide cuBLAS request and can affect Muon matrix operations.
FP32 tensor storage and five FP32 Newton–Schulz iterations do **not** establish
IEEE-identical FP32 arithmetic. Neither a finite loss nor higher throughput
establishes quality equivalence. This is mixed-precision research, not submission
quantization. The generic loader retains its legacy `fp32` default for old
explicit recipes; every new R4 cell writes its precision explicitly.

## Scope of the R4 continuation

Both candidate and reference in G use the amended precision/backend/compiler
settings. The historical R3 JSON is copied then explicitly amended, never edited
in place. Historical FP32 results or the running PR40 W2 results cannot be silently
reused as new R4 W8 controls. B–H retain 30 epochs, their seeds/splits, Muon or
student AdamW, 5%-baseline warmup/cosine policy, and their declared hypotheses.
New configs retain batch2/rank and eight ranks as **unprofiled starting settings**;
this correction does not transfer measured W2 batches96/64 or their 10% reserve
to 3.15B/context4096/chunk2048 experiments. R4 still requires its conservative
20% profile reserve until a separately measured amendment chooses a new batch.

Create a new campaign root at the reviewed successor commit. Source/config
hashes deliberately reject existing materializations, strict-FP32 receipts and
checkpoints as resume inputs. Run receiver correctness checks and exact-shape
one/eight-rank and peak-LR profiles before any fit. Legacy FP32 correctness and
DDP checks run in separate subprocesses with cuBLAS emulation disabled; BF16
runtime tests and fresh emulation-enabled profiles cover the new path. Full
one/eight-rank export parity remains a required measured cache prerequisite.

Local preparation runs only source/AST/JSON/shell/lint checks. Tests, model
imports, training, inference, GPU profiles, remote jobs and submission actions
are not executed. See [R4 commands](high_compute_r4/README.md) and its validation
record for the outstanding receiver checks.
