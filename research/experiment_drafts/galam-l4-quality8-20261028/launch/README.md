# Authorized fixed eight-fit launch

The user explicitly authorized full execution of the allocation in
[v3](../L4_ALLOCATION_V3.md), without changing hypotheses. [PANEL.json](PANEL.json)
materializes exactly eight fits; [configs](configs/README.md) are complete runtime
recipes. Old v1 files remain untouched. Batch candidates will be profiled and
frozen with exact config hashes before the scientific fits.

All use a common20-epoch maximum, strict FP32, fresh seed1001, official earlier
train fitting and complete official valid reporting. No pretrained initializer,
teacher, propensity model, KD or quantization is introduced. GRU211 differs from
raw112 GALAM as a whole-system benchmark. Shared warmup-cosine-v2 is an explicit
adaptation from the historical short OneCycle GRU recipe. All AdamW groups use
beta1.9/beta2.999/eps1e-8/wd.01, including hybrid non-Muon parameters; only the
declared matrix update rules differ in the hybrid arm. Muon momentum.95,5NS
steps, epsilon1e-7 and strict FP32 state/matrix work are explicit.

Keep every epoch snapshot plus strict-best/last pointers. Full official valid
WP determines the separately declared development selection; early stop has
min_epoch5/patience4/min_delta.0005. Strict-best improvement and the patience
threshold are distinct and recorded. Train-eval uses audited all-after-warmup
rows; valid uses official is_scored. These masks are explicitly labelled.

Engineering synthetic shape probes are discarded and provide no model quality.
Full scientific admission also requires actual-source profiles with diagnostics,
normalization/coverage verification, independent recipe review and matched
source/config/runtime receipts. Receiver software is the existing isolated
Torch2.8+cu128 environment; it must pass receiving checks, rather than installing
the incompatible cu130 lock onto driver570. No other service belongs to us.

[prepare_data.py](prepare_data.py) is the bounded receiving-side preparation
owner: it waits for the transfer's finalized train filename, then runs the pinned
source audit, thin indexes, complete source-qualified component graph and primary
manifest. It prewarms train-only raw112/event211 normalization once. Each phase
has a durable receipt/log; it starts no scientific fit and never promotes a
partial file by logical size.
