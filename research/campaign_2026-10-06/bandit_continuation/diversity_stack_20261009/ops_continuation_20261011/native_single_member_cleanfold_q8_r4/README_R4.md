# Native Q8 standalone controls, R4

R4 is a code-only successor to the frozen R3 packet. It evaluates the existing
student and HFT native members independently on the same exposed, source-qualified
F0/F1 memberships. Each prediction has exactly one Q8VNNI member. The runner
does not blend outputs, average parameters, fit models, or search weights.

R4 binds the R3 contract, the exact MAIN metadata-only input receipt
`INPUT_PREFLIGHT_MAIN_R1.json` (SHA-256
`e2b7bab342ee35dc1122f88909f06d9f980188d80a6471341a9ec9104352264c`), and all
producer receipts. That preflight records MAIN's frozen 5,615,509,128-byte
Parquet file and footer-plus-trailer SHA-256
`4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b`; it read
metadata only, not row-data pages. It does not replace the point-of-use check.
At execution, it opens the exact MAIN datasets/valid.parquet, checks size,
footer length and footer-plus-trailer SHA against R3, and compares inode/mtime
with the metadata-only preflight. It repeats that check on the same open file
descriptor after PyArrow opens it and immediately before row-group reads. The
R3 schema, complete 20,000-row groups, canonical membership hashes, model
checkpoints/provenance, weight archive, native ELF and official scorer are
checked against their pinned bytes. Outputs are create-only under the R3 output
root and must close the four-file shape/dtype/byte inventory.

The runner refuses to proceed without ROOT_ADMISSION_R3.json at the exact R3
path. That root-owned receipt must be admitted and authorized, carry the R3
decision, exactly match the R4 contract/runner/preflight/callback-fixture,
data, membership, checkpoint/provenance, weight, native-ELF and output-path
bindings, and carry the fixed CPU-only budget in CONTRACT_SINGLE_R4.json.
The executable also checks scripts/limited environment values and current
cgroup CPU, memory and swap limits; it requires one CPU, at most 1.5 GB RAM,
zero swap and no visible GPU. The receipt does not itself create authority.

The frozen launch form, only after the root admission exists and the Python
3.11 runtime gate is satisfied, is the following command from the MAIN root:

    env CUDA_VISIBLE_DEVICES= WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% \
      WNN_THREADS=1 WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=1800s \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
      ARROW_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
      scripts/limited /usr/bin/python3.11 \
      research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/native_single_member_cleanfold_q8_r4/run_single_controls_r4.py

At preparation time, no R3 root admission or actual R3 callback fixture is
present, and the existing Python 3.11 interpreter lacks NumPy, PyArrow and
PyTorch. The runner therefore remains blocked before loading model/data rows.
The stdlib gate tests and Python 3.13 static checks are not serving validation,
inference, or score evidence. F0/F1 are exposed folds, not fresh holdouts.

`DRAFT_SUPERSESSION_R1.json` preserves hashes of the prior unsealed draft, which
used the isolated R3 preflight. `CONTRACT_SINGLE_R4.json` and
`PREPARATION_FREEZE_R4.json` bind the MAIN-rooted receipt above.
`R4_STATIC_VALIDATION_FAILURE_R1.json` preserves the first post-rebind static
validation finding: declared NPY file sizes include a 128-byte header. The
corrected plan records payload and header bytes separately and requires the
declared total to match.
`DRAFT_SUPERSESSION_R3.json` records the later source-hash refresh after two
nonsemantic Ruff fixes; those prior contract/freeze/validation bytes are kept
as `*_DRAFT_PRE_LINT.json`.
