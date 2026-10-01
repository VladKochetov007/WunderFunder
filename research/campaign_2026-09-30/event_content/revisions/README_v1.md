# D1 endpoint content memory: CPU implementation handoff

Status: standalone research implementation and unapplied integration patch.
No competition data, targets, masks, scores or GPUs were read. Shared live code,
trainer, SSM implementation, ledger and scheduler remain untouched.

## Frozen model

Input first112 columns retain raw float32 data grammar. `a6` is index110;
`i0_p0..21` are columns0:22. Extract endpoint equality before standardization
and autocast. Prereg v1 mistakenly mixed instruments; source-only grammar
inspection corrected v2 before any check. Root no-origin-event instruction
produced v3 before checks. Both earlier registrations are preserved.

Encoder GELU Linear(input,96), GRU1 width96, four query/key8/value16 memories,
nonlinear shared head with encoder/GRU skip. Original Card A rowheads update every
row, half-lives64/512; eventheads update/decay only own arm gate, half-lives2/16.
Endpoint row0 writes only if source endpoint; raw-code row0 has unknown previous
price and never writes. Empty event history S/Z remainszero. First actual write
has ordinal0, subsequent writes increment ordinal; state writecount counts writes.
All rows, including warmup, update GRU/rowheads. External caller resets each
sequence using None, carrying flat states across TBPTT/inference chunks.

Three arms: endpoint-content, endpoint-position, raw-code-content. Same seed,
parameters/dimensions, values/readout, states, arithmetic; source and eventgate
change. C/R changes BOTH write and decay policy. C/P addresses content versus
period128 Walsh positions, not unrestricted time addressing. Rowheads use row
ordinal, eventheads use mostrecent write ordinal, aliasing n and n+128. Values
tanh and projected q/k clamps[-8,8] are explicit numeric amendments; clamp
saturation has zero gradient. Positive ELU+1 retrieval is normalized with floor1e-6.

Parallel implementation uses local <=64x64 causal weight matrices, bounded
exp(logprefix_i-logprefix_j)<=1 and carried S/Z, with no inverse powers or full
20k-square attention. FP32 memory under outer autocast, FP64 mode for oracle
checks. `reference.py` implements direct independent row/head recurrence,
constructing Walsh positions itself and using separate one-row GRU calls.
It shares learned tensors and serialized log-decay buffers, not scan helpers.

## Actor evidence

See `results_v1.json` and exact rerun `results_v2.json` with logs/start/end/exit.
All actor synthetic cases pass fixed prereg tolerances. Independent critic grade
is still required. This proves engineering checks, not competition performance,
physical quote movement or causal variable identities.

The actor checks cover all3 arms FP64/FP32 output, six state fields and gradients
of inputs/initial floating states/parameters; nonzero carried state; chunks1/17/64/
2000; future perturbation, reset and batch isolation; noevent20k, firstactualevent,
64/128 boundary events; extreme finite projections, constant-value convex-hull
check; outerCPU BF16 memory dtype; uncentered Walsh Gram128I rank96; all q/k
columns and value/readout gradients. Constant-value bound is narrower than an
independent arbitrary-varying-value convex-hull audit. Query gradients are small
in the synthetic stationary workload; report minima in JSON, not just pass status.

No actualTRAIN Walsh exposure or freshcomponent endpoint replay was run here.
Those independent preregistered feature gates remain mandatory before any fit.
No model quality, GPU cost, long20k eventful streaming parity, native/runtime or
public score evidence is supplied by this CPU proof.

## Interfaces and replay

`EventContentModel(n_in,n_out,arm,microblock=64)(x,state)` returns output and
flat `(GRUh,S,Z,rowcount,writecount,previous_raw22)`; FP32 state size2664bytes per
sequence at actual input211/output23, trainable count121687. Existing generic
trainer forward_chunk and tuple detach need no state dispatch changes.
`integration.patch` only proposes module/factory/CLI/metadata imports; unapplied.
It is a separate patch from SSM; root resolves their factory contexts if both earn
integration. Strict arm checkpoint loading must use metadata, not shape guesses.

Replay via `run_proof.sh` ONLY with CUDA hidden, WNN_THREADS=1,
WNN_SLICE=wnn-campaign.slice,WNN_CPU_QUOTA=100%,WNN_MEM_MAX=2G,
WNN_RUNTIME_MAX=300 and a unique WNN_SCOPE_UNIT. Durable command in command_v2.txt.
Script captures explicit exit and timestamps. Raw artifacts/hashes in manifest.
Rerun retains original resultsv1 and writes resultsv2. No fit is launched.
Read GPU_PROFILE_PLAN.md for root-owned conditional profiling/fit requirements.

## Normalized retrieval bound

From zero history, each head's effective key contribution has nonnegative weight
`gate_j * exp(sum future logalpha) * dot(phi(q),phi(k_j))`. Their sum is D>=0.
For D>=1e-6, retrieval is the convex combination of written values with these
weights divided by D. For 0<=D<1e-6, retrieval adds zero as a convex-hull vertex
with weight 1-D/1e-6. Thus every coordinate lies in the hull of written values and
zero, and absolute retrieval <=1 under bounded tanh values. Arbitrary incoming
S/Z need not represent such a history; parity tests intentionally allow nonzero
carried tensors without claiming this bound for arbitrary invalid state.
