# E489 clipping-coverage diagnostic: preregistered before implementation

Status: a new non-scoring diagnostic, not a retry or repair of the frozen E489
gate. Root assigns its ledger identity. Frozen v2 probe failed before writing
all seed records because its clipping-exercise requirement was false on an
unknown seed. Preserve that attempt, source, tolerances, seeds and all gates.
No CUDA work is admitted by this document.

Question: for each original seed271836/271837/271838 and its original canonical
first fitting batch, is the failed prerequisite due to a naturally sub-threshold
gradient, or do raw/clipped native/custom updates disagree numerically?

Reuse the immutable baseline_snapshot E489 model, alpha-zero custom recurrence,
fit-only normalization, initial bytes, canonical B4/T256 inputs and FP32 IEEE
optimizer. Load only the24 fit-role groups; never load evaluation-role groups,
score predictions, train full trajectories or change the scientific treatment.

For each seed compute native/custom prediction,state,input-gradient and every
parameter-gradient comparison. Record preclip norms and exact historical
clip_grad_norm_(max_norm=1,error_if_nonfinite=True) factors/status. Save raw
gradients. Starting from the identical initial bytes and fresh optimizer states,
make four one-step diagnostic updates: native/raw,custom/raw,native/clip1,
custom/clip1. Raw means no clipping,not rescaling loss to force a gate pass.
Compare native versus custom updates separately for raw and clip1; compare
raw versus clip1 within each implementation to reveal clipping consequences.
Use the unchanged original four-Muon-matrix plus AdamW-family partition and
record RMS/max errors and original tolerances as descriptive booleans.

No failing coverage/numerical comparison aborts collection of later seeds,
provided values remain finite and resource/identity guards pass. Nonfinite
values,failed provenance or resource breach abort and preserve partial evidence.
Write each complete seed's JSON and raw numeric arrays before moving on; a
later failure must not erase earlier records. Final report explicitly says
it neither passes E489 nor authorizes its full fits. Do not select a new
seed,loss scale or threshold from these results.

Resource proposal:one GPU worker,one CPU,3,000,000,000-byte host cap,
2GiB allocator,3,000,000,000-byte observed-device stop,180-second attempt
ceiling,zero swap and common parent guards. The prior same-shape attempt used
2.429GB host/0.422GB device; this is a feasibility hint,not an admission proof.
Measure peaks and preserve charged seconds/OOM counters/exit status in runner.
Independent source review plus root admission precede any execution.

CPU-only unit checks may exercise diagnostic arithmetic on tiny synthetic
tensors and schema/branch bookkeeping under1CPU/2GiB. They must not run this
actual-data diagnostic or be represented as actual-device evidence.
