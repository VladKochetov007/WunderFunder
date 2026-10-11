"""Compare the R3 Parquet footer checker with its project producer."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRODUCER = ROOT / "scripts/build_campaign_splits.py"
WORKER_TEMPLATE = HERE / "kernel/template.py"
SPLIT = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
REGISTRY = ROOT / "research/campaign_2026-10-04/portable_inputs.json"
ROOT_RANGE_RECEIPT = HERE / "H382_R2_ACTUAL_INPUT_RANGE_PASS_R1.json"
EXPECTED_RAW = {
    "datasets/train.parquet": {
        "bytes": 29_240_219_011,
        "footer_bytes": 60_437_285,
        "footer_sha256": "071a60cc9e72a95b013fe6784318b92a6fb82b7c95b6371d1f21a0e5d04e3bfb",
    },
    "datasets/valid.parquet": {
        "bytes": 5_615_509_128,
        "footer_bytes": 10_686_926,
        "footer_sha256": "4e0a6ad91c87449cc74940e83195ec51c8900d31a25774537bc8283164a7d93b",
    },
}


def load_function(path: Path, name: str) -> Any:
    """Compile one unmodified function from a producer or worker source file."""
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {"Path": Path, "hashlib": hashlib, "os": os}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def validate_actual_registry() -> dict[str, dict[str, object]]:
    """Bind expected footer ranges to the actual fold-0 split and registry."""
    split: dict[str, Any] = json.loads(SPLIT.read_text())
    registry: dict[str, Any] = json.loads(REGISTRY.read_text())
    observed: dict[str, dict[str, object]] = {}
    for logical_path, expected in EXPECTED_RAW.items():
        identity = registry["sources"][logical_path]["original_identity"]
        if split["sources"][logical_path] != identity:
            raise ValueError(f"split and portable registry differ for {logical_path}")
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError(f"actual registry identity changed for {logical_path}: {identity}")
        observed[logical_path] = {
            "bytes": identity["bytes"],
            "footer_bytes": identity["footer_bytes"],
            "footer_sha256": identity["footer_sha256"],
            "split_registry_match": True,
        }
    root_receipt: dict[str, Any] = json.loads(ROOT_RANGE_RECEIPT.read_text())
    if root_receipt.get("result") != "pass" or root_receipt.get("actual_asset_count") != 11:
        raise ValueError("root actual-input footer-range receipt does not pass all eleven assets")
    root_observation = root_receipt["observations"]
    if (
        root_observation.get("split_manifest_sha256")
        != hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    ):
        raise ValueError("root range receipt binds a different split manifest")
    if (
        root_observation.get("portable_registry_sha256")
        != hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    ):
        raise ValueError("root range receipt binds a different portable registry")
    root_raw = {item["name"]: item for item in root_observation["raw_files"]}
    for logical_path, expected in EXPECTED_RAW.items():
        name = Path(logical_path).name
        item = root_raw[name]
        if (
            item.get("observed_bytes") != expected["bytes"]
            or item.get("expected_footer_sha256") != expected["footer_sha256"]
            or item.get("observed_footer_sha256") != expected["footer_sha256"]
        ):
            raise ValueError(f"root actual-input receipt does not verify {name}: {item}")
        observed[logical_path]["root_actual_footer_match"] = True
    return observed


def validate_fixture() -> dict[str, object]:
    """Compare exact producer and worker functions on a real small Parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    producer_identity = load_function(PRODUCER, "source_identity")
    worker_identity = load_function(WORKER_TEMPLATE, "parquet_footer_identity")
    with tempfile.TemporaryDirectory(prefix="h382-r3-footer-") as temporary:
        fixture = Path(temporary) / "tiny.parquet"
        table = pa.table(
            {
                "sequence": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array([0.25, -1.5, 2.0], type=pa.float32()),
            }
        )
        pq.write_table(table, fixture, row_group_size=2, compression="zstd")
        produced = producer_identity(fixture, is_parquet=True)
        observed = worker_identity(fixture, int(produced["footer_bytes"]))
        if observed["observed_footer_sha256"] != produced["footer_sha256"]:
            raise ValueError("worker and producer footer digests differ on the fixture")
        if observed["footer_range_bytes"] != int(produced["footer_bytes"]) + 8:
            raise ValueError("worker footer range omits Parquet framing bytes")
        with fixture.open("rb") as stream:
            stream.seek(-8, os.SEEK_END)
            trailer = stream.read(8)
        if trailer[4:] != b"PAR1":
            raise ValueError("PyArrow fixture has an unexpected Parquet trailer")
        return {
            "fixture_bytes": fixture.stat().st_size,
            "footer_bytes": produced["footer_bytes"],
            "hashed_range_bytes": observed["footer_range_bytes"],
            "footer_sha256": produced["footer_sha256"],
            "producer_worker_match": True,
        }


def main() -> None:
    """Run the bounded registry and fixture footer checks."""
    result = {
        "schema": "wnn.h382-r3-footer-contract-check.v1",
        "registry": validate_actual_registry(),
        "root_actual_input_review_sha256": hashlib.sha256(
            ROOT_RANGE_RECEIPT.read_bytes()
        ).hexdigest(),
        "small_parquet_fixture": validate_fixture(),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
