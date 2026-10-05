# E484 v2 independent source review

## Disposition

**GO for a newly admitted actual-shape CUDA probe using only the exact v2 hashes below.** The failed v1 CUDA probe remains an incident and is not repaired in place. This review does not authorize full fits; the root controller must first verify v2 probe parity, all 12 finite changed updates, and measured resource/teardown receipts.

## Exact identities checked

- `fit_context_mixture_v2.py`: `2dec963a966ba30e1dc3deed40472fd2d0129a2d24fa07585a3a7ed4e2e1dcbc`
- `tests_v2.py`: `0d7030f0197c949c3cd72ea0b1e338cfe8b86e66b7de69b9f00853a6f154c9d2`
- `gpu_probe_v2.py`: `06e4b056dd0c9ea3eeab3d7725918801a6796b68d7f82b734e191aa366a2018d`
- `preflight_cpu_v5/fit_manifest.json`: `26073971b2942ce0c70410b9dbb92b89ef1dbd8a992ed4b5a2fcb7167ae42a0f`

The v2 preflight manifest binds the three exact source hashes, records 12 one-update fit-only cells with finite parameters and nonzero changed-tensor counts, and sets `test_rows_loaded: false`.

## Review findings

The v1 probe failure was caused by comparing a CUDA parameter tensor with its CPU initialization using `torch.equal`. V2 adds a narrowly scoped canonical identity function: detach, transfer to CPU, make contiguous, then compare dtype, shape, and exact bytes. `check_common_parameter_bytes` uses this for every inherited parameter and buffer. This preserves exact initialization checking across devices; it does not loosen numeric tolerances or alter the science. V2 also changes its self-source inventory to the versioned v2 trainer/test/probe filenames. Diff review found no other change to the E484 forward equation or training recipe.

The previously reviewed cuDNN correction remains present: native parity uses train mode for backward with dropout fixed at zero, includes matching output/state/input/parameter gradients, and requires a clipped AdamW update to match the frozen reference exactly with a synthetic preclip norm above 1. The independent E484 implementation review and its six CPU tests are preserved in `INDEPENDENT_REVIEW.md`; this v2 check reran the seven v2 tests and confirmed Ruff and byte compilation.

The probe configures the worker before model allocation, captures active precision metadata inside the full IEEE context, runs native parity, then executes the full registered B4×256 one-update matrix for both seeds and all six arms. It checks finite loss/parameters and changed parameters, while the root controller remains responsible for observed VRAM, cgroup limits, deadlines, and teardown. The scorer in development is separate and must bind these v2 source identities plus the actual v2 probe attempt before any test loader call.

## Bounded CPU verification

Ruff, py_compile, and all seven `tests_v2.py` tests passed under a 1-CPU, 2-GiB, no-swap systemd scope. No CUDA call, full fit, checkpoint grade, or evaluation-row read was performed by this reviewer.
