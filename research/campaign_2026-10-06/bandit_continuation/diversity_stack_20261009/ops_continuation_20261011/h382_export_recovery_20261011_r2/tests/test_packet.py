"""Static contract checks for the H382 export-only recovery worker."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "build_export_recovery.py"
SPEC = importlib.util.spec_from_file_location("build_export_recovery", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import the frozen packet builder")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
EXPECTED_RUNS = BUILDER.EXPECTED_RUNS
HERE = BUILDER.HERE
patch_kernel = BUILDER.patch_kernel


def test_recovery_worker_is_export_only_and_collects_exact_run_names() -> None:
    manifest = json.loads((HERE / "checkpoint_asset_manifest.json").read_text())
    source = patch_kernel(manifest).decode()
    assert source == (HERE / "kernel/kernel.py").read_text()
    ast.parse(source)

    assert 'SOURCE_PACKET_ID = "vladkochetov07/wnn-kaggle-crossfold-h382-20261011-r3"' in source
    assert 'PACKET_ID = "vladkochetov07/wnn-h382-export-recovery-20261011-r2"' in source
    assert 'Path("/kaggle/input/datasets") / owner / CHECKPOINT_DATASET_SLUG' in source
    assert 'Path("/kaggle/input") / CHECKPOINT_DATASET_SLUG' in source
    assert 'if not input_identity.get("inputs_validated")' in source
    assert 'job["steps"] = export_steps' in source
    assert 'len(job["steps"]) != 1' in source
    assert 'ROOT / "cache" / f"pred_{name}_valid.{suffix}"' in source
    assert "prediction_dir / path.name" in source
    assert 'ROOT / "cache").glob("pred_kg_h382_*_valid.*")' not in source
    assert "train_rnn.py" in source  # Present only in the explicit no-fit guard.
    assert "subprocess.Popen(" in source
    assert 'RUNTIME_SECONDS = 3_600' in source
    assert '"prediction_finite": bool(finite_scan["finite"])' in source
    assert source.index('terminal["reference_prediction_hashes"] = {') < source.index(
        '(OUT / "terminal_inventory.json").write_text('
    )
    assert '"match": False' in source

    assert 'for job in jobs["jobs"]' in source
    assert 'name = str(job["name"])' in source
    input_validation = source.index("input_identity = link_inputs(jobs)")
    split_validation = source.index('if sha256_file(split_path) != jobs["split_manifest_sha256"]')
    checkpoint_stage = source.index("checkpoint_manifest = stage_rescued_checkpoints(jobs)")
    assert input_validation < split_validation < checkpoint_stage


def test_expected_export_receipts_are_three_distinct_r3_runs() -> None:
    runs = json.loads((HERE / "checkpoint_asset_manifest.json").read_text())["runs"]
    assert {item["name"] for item in runs} == {item[0] for item in EXPECTED_RUNS}
    assert all(item["reference_export_receipt"]["prediction_bytes"] == 299_680_128 for item in runs)


def test_chunked_prediction_scan_rejects_nonfinite_values(tmp_path: Path) -> None:
    import numpy as np

    source = (HERE / "kernel/kernel.py").read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "scan_prediction_finiteness"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {"Path": Path}
    exec(compile(isolated, "kernel.py", "exec"), namespace)
    scan = namespace["scan_prediction_finiteness"]

    finite_path = tmp_path / "finite.npy"
    np.save(finite_path, np.arange(24, dtype=np.float32).reshape(3, 4, 2))
    finite = scan(finite_path, chunk_rows=1)  # type: ignore[operator]
    assert finite == {"finite": True, "checked_rows": 3, "chunk_rows": 1}

    nonfinite_path = tmp_path / "nonfinite.npy"
    values = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    values[1, 2, 1] = np.inf
    np.save(nonfinite_path, values)
    nonfinite = scan(nonfinite_path, chunk_rows=1)  # type: ignore[operator]
    assert nonfinite == {"finite": False, "checked_rows": 2, "chunk_rows": 1}
