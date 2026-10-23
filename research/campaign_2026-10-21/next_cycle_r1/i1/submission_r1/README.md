# Fixed i1 B/C mixture: CPU delivery

[Delivery status](DELIVERY_STATUS.json), [native build receipt](NATIVE_BUILD.json)
and [static ELF receipt](STATIC_ABI.json) record successful Linux compilation
for v2/avx2/v3, ELF64 x86-64, maximum GLIBC2.34 and no libpython dependency.
Ruff passed for all six changed Python files. Independent source review found
no remaining concrete contract blocker; numerical and runtime checks were not
run. `submissions/pr33_i1_cpu_runtime_20261023.zip` is a library handoff archive
with instructions and no model weights. It is **not** a competition submission.

User request, 23 October 2026: implement CPU inference and ZIP packaging in draft
PR #33; do not run local tests or training. The user then explicitly requested
continuing without the unavailable checkpoint payloads. This packet contains no
new fit, score, public forecast, or competition upload.

[FIXED_CANDIDATES.json](FIXED_CANDIDATES.json) pins two **separate** endpoints:
90% B + 10% C at zero-based epoch 2, F0/seed1001 and F2/seed1002. Each needs two
H512×1 checkpoints. We do not combine four models or select a cell from its
observed gain. Source: the frozen [six-fit grade](../runtime/GRADE_R1.json).

The reported local deltas are +0.00035953 and +0.00027145 versus the paired B
control. They are exposed-fold screening evidence. Neither the independent C
model nor this CPU package has demonstrated improvement over the public best
`tl3.zip`. The public score of the mixture is unknown.

## CPU implementation

The served layout is raw112 + event99 + causal i1 history156. The history uses
checkpoint-owned fitting scales, the preceding EMA residuals at half-lives32
and512, and current-transition change energy. First-row history is zero; state
advances on every callback, including rows that do not request a prediction,
and resets on sequence changes. The checkpoint's input mask is applied after
stored normalization: B masks the history while C retains it.

Only explicit supported metadata is accepted; a width of367 alone does not
identify the feature contract. The package uses FP32 weights and exact native
activations for these i1 members. It needs NumPy and the included native library
at inference, without Torch, model downloads, GPU or Internet access.

## Build on the receiving machine

Run from this PR checkout with Docker available. This first command only
compiles three Linux x86-64 variants and inspects ELF metadata; it does not run
model inference or tests. Compilation is serial, one CPU, 2GB RAM, no task swap.
The output directory must not already exist.

```sh
python3 scripts/build_i1_history_native.py \
  --output research_tmp/i1-native-20261023
```

The Docker toolchain comes from the official Rust1.94 image and the scorer's
Python3.11/bookworm base. Native artifacts are source-hash-bound; rebuilding is
required after Rust source or build-definition changes. The build produces
`NATIVE_BUILD.json`, `STATIC_ABI.json` and v2/avx2/v3 libraries. Static ELF checks
reject a version-specific libpython dependency and GLIBC requirements above2.36.

Required run directory layout, for one selected pair:

```text
RUNS/i1_f0_s1001_b_r1/model_ep2.pt
RUNS/i1_f0_s1001_b_r1/provenance.json
RUNS/i1_f0_s1001_c_r1/model_ep2.pt
RUNS/i1_f0_s1001_c_r1/provenance.json
```

The history prior is embedded in each checkpoint's `research_feature_prior`.
Do not replace the checkpoint with a similarly named `model.pt`, refit the
prior, use another epoch, or derive scales from validation/test rows.

With Python3.11+ and the existing research dependencies provisioned, run:

```sh
python scripts/package_i1_fixed_mix.py \
  --cell f0_s1001 --runs-dir /absolute/path/to/RUNS \
  --native-from research_tmp/i1-native-20261023 \
  --out-root submissions
```

Use `--cell f2_s1002` with the corresponding F2/seed1002 B/C directories to build
that separately graded pair. The wrapper checks the frozen checkpoint hashes,
provenance presence, grade identity, native source/library hashes, coefficients,
and compressed ZIP size. It delegates export to the existing packager, with
FP32 and provenance required. Existing output is never overwritten.

Output: `submissions/i1_90B10C_<cell>_ep2_fp32_20261023/submission.zip` and
`BUILD_RECEIPT.json`. The receipt binds source checkpoints, exported tensors,
solution, libraries and final ZIP. The ZIP has `solution.py` at its root.
No fixed member is dropped at runtime and no wall-clock policy changes weights.

## Remaining release checks

As requested, no local inference, tests, benchmarks or training were run.
The model ZIP cannot be assembled until the exact checkpoint payloads arrive.
Do not label a source/native handoff archive as a competition submission.

After weights arrive, a Linux owner must verify the **exact delivered ZIP**:
offline Python3.11 import; streaming parity against both frozen model outputs,
including warm-up and sequence reset; full paired-fold official scoring; and
one-CPU timing/peak memory under16GB with the complete test workload budget
of60minutes. ZIP size must be below20MB. Static compilation and ELF checks do
not establish numerical parity or timing. Two FP32 H512 models may be too slow;
quantization would be a separately measured serving variant, not a silent
replacement of this FP32 recipe.

## Portal observation, 23 October 2026

Read directly in authenticated background Chrome:
[team submissions](https://wundernn.io/connectome/submit),102total,0today.

| Submitted | Archive | Score | Duration | RAM |
|---|---|---:|---:|---:|
| 22 Oct 14:10 | `tl3_two_parents_20261022.zip` | 0.66771 | 28m31s | 643MB |
| 21 Oct 18:21 | `b300_h512_cp5_tl3_20261021.zip` | 0.66667 | 53m47s | 672MB |
| 15 Oct 23:52 | `tl3.zip` (team best) | 0.66832 | 39m12s | 656MB |

Team position shown:25/5282. The portal payloads were not downloaded or hashed;
filenames are not proof of exact checkpoint lineage. No i1 B/C upload is shown.
The team's private HF archive is still at
`167452c72a8e994b280446d2fca93f7c08e16f66` (12 October), predating the i1 fits.
