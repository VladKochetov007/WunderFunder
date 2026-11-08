# Bounded L4 receiving handoff

ROOT owns SSH and execution. These packet scripts do not change the frozen
trainer or create a new training implementation. They call the existing
`run_owned_command`, process identity, shared profile/fit budget, GPU slot and
`observe_sequence.py` helpers. All eight cells progress independently; a missing
Mamba oracle or a missing approval cannot hold up admitted GRU cells.

Stage this directory outside `source-r1`. The sender-bound source inventory is
`R/manifests/l4-source-r1-manifest.json`,620files, SHA256
`7d3324230effb050ff083b939e26ad638e5df5c5b72ea70b0f8aed169997db4a`.
Every source file and additional importable Python are checked. The panel freezes
the original eight configs, all runtime-r3 derivatives and normalized config
digests, the physical GPU mapping, exact memberships, CPU coverage receipt,
first-profile clock and receiving script digests. Runtime-r3 changes only batch16,
diagnostics20, train telemetry10 and evalbatch32. All scientific policy remains
the same twelve-epoch fixed-final label-only comparison.

Use the existing fresh environment; do not modify shared environments. ROOT's two
initial GRU profiles on GPUs0/1 are adopted through their exact helper receipts
without replay. Their original owner is the ROOT process identity; it need not
contain cell_id. New profiles use GPUs2/3 GRU-local,4/5 genuine Mamba-2,6/7 EMA.

```bash
R=/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1
S="$R/source-r1"
PY=/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python
Q="$R/control/receiving-source-r1" # stage this reviewed directory here
export PYTHONPATH="$S/src"
"$PY" "$Q/selftest.py"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader > "$R/receipts/l4-gpu-inventory-r1.csv"
"$PY" "$Q/build_panel.py" --runtime "$R" \
  --split-manifest "$R/project/chronological_primary.json" \
  --gpu-inventory "$R/receipts/l4-gpu-inventory-r1.csv" --prior-profile-attempts 2
# Use the panel SHA printed by build_panel.py:
"$PY" "$Q/run_cohort.py" --panel "$R/control/l4-cohort-r1/panel.json" \
  --panel-sha256 '<printed SHA256>' --detach
```

The root-supplied `R/receipts/first-profile-r1.json` must contain the actual
numeric first_profile_epoch. The fixed wall is24hours, with fitting closed at
hour23 to retain a one-hour collection reserve. The existing budget helpers
count the two initial profiles plus new attempts, with twelve profiles and eight
scientific fits maximum. Profile failures remain terminal and consume their cap;
there is no automatic retry or fitting resume. `--phase profiles` is available
for a foreground engineering-only pass; default all waits for separate approvals.

The actual coverage receipt `R/receipts/data-coverage-r1.json` must say
status=passed, split_sha256=`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`,
fit_train=10325, fit_valid=0, evaluation_sequences=1873 and selected_rows=3528273.
The immutable receipt and actual manifest are both bound. Sequence IDs supply no
additional chronology; validation stays outside fitting ancestry and remains
development-exposed.

Each cell uses the following paths:

```text
R/runs/<cell>/profile/profile_receipt.json        unapproved native profile20
R/receipts/owned-profile-<cell>.json              exact owned engineering command
R/approvals/<cell>.json                          separate ROOT-reviewed copy
R/runs/<cell>/fit/                               twelve-epoch scientific artifacts
R/receipts/owned-fit-<cell>.json                  exact owned fitting command
R/runs/<cell>/collection.json                    raw artifact callback input
R/control/l4-cohort-r1/public_status.json         public per-cell summary
R/control/l4-cohort-r1/resources.json             ten-second resource observations
R/control/l4-cohort-r1/controller-owner.json      detached PID/birth/PGID receipt
```

ROOT approval is an exact native receipt copy with only approved changed to true
and additional reviewer="ROOT", reviewed_at and reason fields. Preserve the native
approval_note and all measured evidence. The runner never creates or modifies an
approval. It checks exact config/seed/source/data/membership/physical GPU identity
and then calls the canonical receipt validator; fitting independently repeats
canonical admission. Twenty successful updates reach the diagnostics20 cadence.

New profiles and fits explicitly unset CUBLAS_WORKSPACE_CONFIG and emulation to
match the initial ROOT profiles, preserve strict-FP32 TF32 controls with
NVIDIA_TF32_OVERRIDE=0 and TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0, and use eight CPU
threads per worker. Native Mamba kernel arithmetic still requires its own actual
source/device qualification; no CPU surrogate or IEEE claim is invented here.

Before each Mamba cell can profile, ROOT supplies
`R/receipts/mamba-oracle-<cell>.json` after the actual pinned CUDA test passes on
that physical GPU. Its fields are status="passed", exit_code=0, gpu_uuid matching
the cell, source_inventory_sha256 matching the trusted source inventory, test=
`tests/test_compact_sequence_models.py::test_mamba2_real_backend_oracle_causality_chunk_reset_and_gradients`,
and evidence={path:"<raw test log>",sha256:"<raw log digest>"}. Failed/missing
oracles never admit a stand-in architecture.

Disk/RAM checks run at ten-second cadence. New output is capped at8GB; physical
/data retains4GB free and /data2 retains3GB. The eight workers request64 CPU
threads total, owned worker RSS is capped at384GiB, and host MemAvailable retains
64GiB for page cache/headroom. Resource or deadline breaches signal only matching
receipt-owned PID/birth/PGID groups using the existing termination helper. The
small observed_worker relay retains the owned leader if observation fails, so
hard termination remains possible; foreign GPU4 services are never adopted.

Every full-validation journal event must report exactly1873sequences and
3528273selected rows with sequential completed epochs. Completion requires all
twelve events and the fixed-final terminal/checkpoint identity. A partial fit
cannot become a completed comparison. Public final WP/t0/t1 come from epoch12.
All original fit files and all-step journals remain in place, individually hashed
by collection.json. That create-only file is the later graph/bundle callback
boundary; no graphs, external publication or callback execution are fabricated.

Local tests:7synthetic gate/lifecycle/environment tests pass; Ruff format/check
and CLI smoke tests pass. Independent receiving review found the relay signal and
workspace identity issues; both are fixed and the reviewer re-ran7/7tests with
no remaining static blocker. Actual POSIX lifecycle/resource/profiles and all
scientific results remain ROOT's receiving evidence.
