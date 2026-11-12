# L4 next-70 packet, 10 November 2026

This first increment freezes six one-GPU supervised profiles: two shared GRU
controls, two target-0 specialists, and two target-1 specialists. The 6E fits
remain gated on a ROOT approval for each exact profile. The selected target uses
the canonical two-head model with only that channel's weighted loss numerator;
the global two-head denominator stays unchanged. Configs require the standalone
labels objective, so they have no teacher, reference, KD, hint, blend, or TL3
ancestry. All fitting normalizers use the 10,325 official train sequences.

The fixed primary endpoint is epoch 6. All epoch checkpoints are retained;
report-only monitoring scores all 1,873 official validation sequences and
3,528,273 selected rows each epoch. Validation is exposed development evidence,
not a pristine holdout. The official train/valid manifest SHA-256 is
`0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`.

The shared config changes the old floor-0 schedule to the newly declared
`warmup_cosine_v2` minimum-LR ratio 0.1 across every arm. This is a dated recipe
amendment, not an exact replay of the historical control. The model remains raw
112-input GRU256x2, with batch32, chunk1000, one accumulation chunk, FP32,
AdamW3e-4, decay0.01, clip1, and 5% warmup. Update diagnostics run every20
updates; native heartbeat sampling runs every10 seconds.

| GPU | Cell | Seed | Target numerator | Fit endpoint |
|---:|---|---:|---:|---|
| 0 | `shared_control_s203201` | 203201 | both channels | 6E |
| 1 | `shared_control_s203202` | 203202 | both channels | 6E |
| 2 | `t0_specialist_s203201` | 203201 | target0 | 6E |
| 3 | `t0_specialist_s203202` | 203202 | target0 | 6E |
| 4 | `t1_specialist_s203201` | 203201 | target1 | 6E |
| 5 | `t1_specialist_s203202` | 203202 | target1 | 6E |

The predeclared fixed specialists pairings are `(s0.t0,c.t1)`,
`(c.t0,s1.t1)`, and `(s0.t0,s1.t1)`. Use only the target-specific specialist
head and the corresponding shared control head. Do not learn blend weights from
validation. The two SSL lanes remain in the approved eight-lane allocation and
are pending the separate core extension: 2E self-supervised pretraining followed
by a same-backbone, optimizer-reset 4E supervised stage. They are not silently
removed or substituted by extra fits; the six controls/specialists can proceed
without waiting for them.

## Native paths and gates

ROOT owns native deployment, profile approval, launch, stop, and collection.
Use the new prepared project at
`/data/lc-epanov_s/tabdata-next70-20261110-r1/project`, the pinned Python
`/data2/lc-epanov_s/tabdata/.venv/bin/python`, and the B4021 source capsule that
ROOT froze at commit `9a5d2e7ca2549c24ff8217d5062df6de49cce9b5` with source
identity `b05a73fc8a8a7a4f3f46962e31fd81ee6b401db715109e36d4f0d6e32bf35140`.
Outputs belong under
`/data/lc-epanov_s/tabdata-next70-20261110-r1/runs`; shared source datasets stay
on `/data2` and are not copied. Preserve the unrelated GPU4 processes.

The existing L4 owned-command executor, observer, and wave resource/identity
checks are the implementation base. Freeze raw and normalized config hashes and
the source capsule in the native panel before profiles. Each cell needs its own
30-update actual L4 profile, finite-state and strict model/optimizer/scheduler/
host/CUDA-RNG reload receipt, exact source/config/split/device identity, then a
separate ROOT approval file before its fit claim is created. Do not automatically
retry a failed profile or fit. Release individually approved controls as soon as
they qualify; the remaining cells and pending SSL work do not gate them.

The first actual profile starts a fixed12-hour packet clock. Close new fit
admission at hour11 and reserve hour12 for full validation, checkpoints, graphs,
receipts, and collection. Output is capped at6GB on `/data`; protect `/data2`
free-space, host RAM, foreign processes, and exact birth-bound owned process
groups. No SSH/GPU work is performed by this local preparation.

ROOT reports all six actual L4 profiles passed30 updates and the strict
model/optimizer/scheduler/host/CUDA-RNG reload gate: 3.62–3.79 seconds and peak
reserved memory1,801,453,568 bytes (about1.68GiB) per the native receipts. ROOT
has admitted all six fresh6E fits. These profiles are engineering evidence only;
no fit-quality result is recorded here. The exact raw config hashes and cell
assignments are in [profile-plan-r1.json](native/profile-plan-r1.json); the
normalized hashes and approved profile hashes remain bound by ROOT's native fit
admission receipt.
