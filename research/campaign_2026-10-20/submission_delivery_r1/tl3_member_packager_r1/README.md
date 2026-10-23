# TL3 single-member H512 packager

**Status: an engineering fixture ZIP and native callback checks pass; no new trained candidate is promoted.** The [packager](../../../../scripts/package_tl3_h512_member.py) creates a new archive from the pinned TL3 ZIP and replaces only `weights_s5.npz`. It uses the existing checkpoint exporter, pinned archive verifier and canonical architecture checker. The original `solution.py`, three `wnn_cpp` libraries, two 263-input NPZ files and blend coefficients are hash-checked and preserved. The packager requires explicit checkpoint, run directory, output directory and artifact role; the output directory must be new and under physical `research_tmp`.

It accepts only an FP32 canonical 211-input H512×1 GRU with two outputs, requires its provenance sidecar, changes only the student `name` in `members.json`, and rejects an archive at or above 20,000,000 bytes. Its receipt records checkpoint and provenance hashes, source archive and entry hashes, exported tensor hashes, resulting ZIP and entry hashes, and code hashes. It does not load or score targets, run native inference, or claim quality or runtime.

## Engineering fixture command

The old `student_frontier_f0s3_h512_l1_r2@ep0` artifact exists for exercising this code path only. It has weak historical quality and must never be treated as a submission candidate or evidence for H512 promotion.

```sh
cd /home/vlad/development/wunder/wnn_connectome_starterpack
WNN_MEM_MAX=1200000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 CUDA_VISIBLE_DEVICES= \
  scripts/limited env PYTHONPATH="$PWD/src" \
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  scripts/package_tl3_h512_member.py \
  --run-reference student_frontier_f0s3_h512_l1_r2@ep0 \
  --runs-dir /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs \
  --output-dir /home/vlad/development/wunder/research_tmp/tl3-h512-engineering-fixture-20261021-r1 \
  --artifact-role engineering_fixture
```

The bounded callback probe below exercises two complete validation sequences in A–B–A order through the unchanged TL3 wrapper. It checks import, finite two-output rows, warmup behavior and reset/replay determinism. It reads features only and computes no score.

```sh
cd /home/vlad/development/wunder/wnn_connectome_starterpack
WNN_MEM_MAX=1200000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv-py311-profile/bin/python - \
  /home/vlad/development/wunder/research_tmp/tl3-h512-engineering-fixture-20261021-r1/submission.zip \
  /home/vlad/development/wunder/wnn_connectome_starterpack/datasets/valid.parquet <<'PY'
import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from utils import FEATURE_COLUMNS, DataPoint, SEQUENCE_LENGTH, validate_sequence

zip_path, data_path = map(Path, sys.argv[1:])
scratch = Path("/home/vlad/development/wunder/research_tmp")
with tempfile.TemporaryDirectory(prefix="tl3-h512-probe-", dir=scratch) as temporary:
    package = Path(temporary)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("candidate ZIP failed CRC validation")
        archive.extractall(package)
    spec = importlib.util.spec_from_file_location("tl3_probe_solution", package / "solution.py")
    if spec is None or spec.loader is None:
        raise ImportError("candidate ZIP has no loadable solution.py")
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)
    model = solution.PredictionModel()
    parquet = pq.ParquetFile(data_path)

    def replay(group: int) -> tuple[int, np.ndarray]:
        table = parquet.read_row_group(
            group, columns=["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS],
            use_threads=False,
        )
        sequence, need = validate_sequence(table)
        features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(
            np.float32, copy=False
        )
        if len(features) != SEQUENCE_LENGTH:
            raise ValueError("probe row group is not a complete sequence")
        outputs = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
        for step, row in enumerate(features):
            value = model.predict(DataPoint(sequence, step, bool(need[step]), row))
            if not need[step]:
                if value is not None:
                    raise ValueError("warmup callback must return None")
                continue
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("required callback rows must have two finite outputs")
            outputs[step] = value
        return sequence, outputs

    first = replay(0)
    second = replay(1)
    first_again = replay(0)
    if first[0] == second[0] or not np.array_equal(
        first[1].view(np.uint32), first_again[1].view(np.uint32)
    ):
        raise ValueError("sequence reset/replay check failed")
    print(json.dumps({"passed": True, "groups": [0, 1, 0], "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(), "scope": "engineering fixture only; no metric"}, sort_keys=True))
PY
```

## Production invocation

After root admits the all-data H512 checkpoint and completes the independent review, use its exact checkpoint reference with a new scratch destination:

```sh
cd /home/vlad/development/wunder/wnn_connectome_starterpack
WNN_MEM_MAX=1200000000 WNN_CPU_QUOTA=100% WNN_THREADS=1 CUDA_VISIBLE_DEVICES= \
  scripts/limited env PYTHONPATH="$PWD/src" \
  /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  scripts/package_tl3_h512_member.py \
  --run-reference r01_h512_all_data_s1003_cp5_r1@ep5 \
  --runs-dir /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs \
  --output-dir /home/vlad/development/wunder/research_tmp/tl3-h512-production-r1 \
  --artifact-role production_candidate
```

This packages an all-data model as a production candidate; it creates no clean-fold evidence. The separate callback probe, native comparison, runtime measurement, package acceptance and quality decision remain for root and an independent reviewer. The original builder handoff ran no model checks; the controller subsequently ran the actual checks recorded below.

## Integration review

Root independently inspected the packager and reused helpers; archive preservation, dimensions, dtype, finite/positive normalization, create-only output and provenance binding have no identified static blocker. Builder commit a3a4a0b769a916970368bcc349525ba0c165f7c8 adds the positive-std guard. Packaging uses the existing main Python environment because the native-only Python 3.11 environment has no Torch; the callback probe uses Python 3.11. Actual checks below must be bound to their emitted receipts.

## Actual engineering checks

The [fixture build](ENGINEERING_BUILD_RECEIPT_R1.json) produced a 17,091,324-byte
ZIP with SHA `0d660feb7419e01bcf613a5f906cc17d853509e282fb10495d19f41986dfb10c`.
Its payload remains in physical scratch; it uses the old weak H512 artifact
solely to test architecture compatibility. **Do not submit this fixture.**

The [Python 3.11 callback probe](ENGINEERING_CALLBACK_PROBE_R1.json) passes the
original v3 native wrapper on full feature-only sequences in A–B–A order:
finite two-target predictions, correct warmup behavior, and bitwise reset/replay.
No labels or metric were read. Peak process RSS was 334 MB.

The [alternating relative-cost check](ENGINEERING_RELATIVE_COST_R1.json) uses
the same two full groups, three repeats and the original TL3 ZIP as baseline.
Median callback-loop sequence times were 2.9103 s for TL3 and 3.7918 s for the
H512 fixture, a ratio of 1.303. Replays remained bitwise equal within each
endpoint. Peak process RSS was 349 MB. Shared-machine timing varies; this is
relative architecture cost, not measured official runtime or candidate quality.

An independent static review by `nll_followup_recipe` found no archive/tensor
construction blocker. The packager is a byte transformer: its operator-supplied
artifact role and separately recorded provenance hash do not establish fitting
membership, admission or ancestry. Root must validate the chosen checkpoint
against its accepted terminal receipt before production packaging. Final native
quality, precision behavior and target runtime remain unmeasured.
