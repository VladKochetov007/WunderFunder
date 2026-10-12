from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

R8_ID = "vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r8"
R8_TITLE = "wnn-clean-t3-f0-h384-exp-20261012-r8"
R7_KERNEL_SHA256 = "f38ac7bd72c762b265e5951096e57b799cf632f43f7f96819a45d38772277cb6"
R7_BINDINGS_SHA256 = "39127244077e0937934fce1fedc131f7fa440ff18f7bc7fc3508f6f69e318d0f"
R7_PENDING_RELEASE_SHA256 = "7b76769bf34bf1f43b3ff23c9f5c59616cd80188c232b46df667083793b70940"
R7_PACKET = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    "/ops_continuation_20261011/clean_t3_f0_h390_r7_export_recovery_20261012_r1/kernel"
)

WRAPPER_TEMPLATE = '''from __future__ import annotations

import base64
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

R8_KERNEL_ID = "vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r8"
R8_TITLE = "wnn-clean-t3-f0-h384-exp-20261012-r8"
R7_BINDINGS_FILENAME = "R7_EXPORT_BINDINGS_R1.json"
R7_RELEASE_FILENAME = "ROOT_EXPORT_RELEASE_R7.json"
R7_BODY_B64 = (
@BODY_B64@
)
R7_BINDINGS_B64 = (
@BINDINGS_B64@
)
ROOT_RELEASE_B64 = (
@RELEASE_B64@
)
R7_BODY_SHA256 = "@BODY_SHA@"
R7_BINDINGS_SHA256 = "@BINDINGS_SHA@"
ROOT_RELEASE_SHA256 = "@RELEASE_SHA@"
CONTROL_DIR = Path("/kaggle/working/.wnn-r8-export-control")


def _decode_embedded() -> tuple[bytes, bytes, bytes]:
    body = base64.b64decode(R7_BODY_B64, validate=True)
    bindings = base64.b64decode(R7_BINDINGS_B64, validate=True)
    release = base64.b64decode(ROOT_RELEASE_B64, validate=True)
    if _sha256(body) != R7_BODY_SHA256:
        raise PermissionError("embedded R7 worker body changed")
    if _sha256(bindings) != R7_BINDINGS_SHA256:
        raise PermissionError("embedded R7 bindings changed")
    if _sha256(release) != ROOT_RELEASE_SHA256:
        raise PermissionError("embedded root release changed")
    return body, bindings, release


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_release(bindings_bytes: bytes, release_bytes: bytes) -> dict[str, Any]:
    bindings = json.loads(bindings_bytes)
    release = json.loads(release_bytes)
    if not isinstance(bindings, dict) or not isinstance(release, dict):
        raise ValueError("embedded binding/release must be JSON objects")
    if bindings.get("schema") != "wnn.kaggle.clean_t3_f0_export_only_r7_bindings.r1":
        raise PermissionError("unexpected frozen R7 binding schema")
    if release.get("schema") != "wnn.root.clean_t3_f0_export_only_release.r7.r1":
        raise PermissionError("unexpected ROOT R7 release schema")
    if release.get("status") != "admitted" or release.get("authorized") is not True:
        raise PermissionError("ROOT release is pending; exports remain blocked")
    if release.get("binding") != bindings.get("root_release_binding"):
        raise PermissionError("ROOT release does not bind the frozen R7 recipe")
    logical_id = bindings["root_release_binding"].get("target_kernel_id")
    if release.get("logical_recipe_target_kernel_id") != logical_id:
        raise PermissionError("ROOT release logical R7 target differs")
    if release.get("transport_target_kernel_id") != R8_KERNEL_ID:
        raise PermissionError("ROOT release does not authorize this R8 transport target")
    return bindings


def run_embedded(control_dir: Path, release_bytes: bytes | None = None) -> None:
    """Validate authority, materialize the three embedded controls, then run R7."""
    body, bindings, embedded_release = _decode_embedded()
    selected_release = embedded_release if release_bytes is None else release_bytes
    if release_bytes is not None and _sha256(selected_release) != ROOT_RELEASE_SHA256:
        raise PermissionError("provided ROOT receipt differs from the script seal")
    _validate_release(bindings, selected_release)
    if control_dir.exists() or control_dir.is_symlink():
        raise FileExistsError("R8 control directory already exists")
    control_dir.mkdir(parents=False, exist_ok=False)
    controls = {
        "kernel.py": body,
        R7_BINDINGS_FILENAME: bindings,
        R7_RELEASE_FILENAME: selected_release,
    }
    for name, content in controls.items():
        path = control_dir / name
        with path.open("xb") as output:
            output.write(content)
        if path.stat().st_size != len(content) or path.read_bytes() != content:
            raise OSError(f"R8 control file materialization failed: {name}")
    runpy.run_path(str(control_dir / "kernel.py"), run_name="__main__")


def main() -> None:
    run_embedded(CONTROL_DIR)


if __name__ == "__main__":
    main()
'''


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def b64_lines(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return "\n".join(f'    "{encoded[index : index + 76]}"' for index in range(0, len(encoded), 76))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build(
    packet: Path, output: Path, release_path: Path | None, receipt_path: Path
) -> dict[str, Any]:
    if receipt_path.resolve().is_relative_to(output.resolve()):
        raise ValueError("build receipt must be outside the Kaggle upload directory")
    kernel_dir = packet / R7_PACKET
    body = (kernel_dir / "kernel.py").read_bytes()
    bindings_bytes = (kernel_dir / "R7_EXPORT_BINDINGS_R1.json").read_bytes()
    pending_release = (kernel_dir / "ROOT_EXPORT_RELEASE_R7.json").read_bytes()
    metadata = read_json(kernel_dir / "kernel-metadata.json")
    if sha256(body) != R7_KERNEL_SHA256:
        raise ValueError("R7 worker body does not match the reviewed bytes")
    if sha256(bindings_bytes) != R7_BINDINGS_SHA256:
        raise ValueError("R7 bindings do not match the reviewed bytes")
    if sha256(pending_release) != R7_PENDING_RELEASE_SHA256:
        raise ValueError("R7 pending release template changed")
    if metadata.get("dataset_sources") != [
        "vladkochetov07/wnn-c20261004-fold0-assets",
        "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5",
        "vladkochetov07/wunder-funder",
    ]:
        raise ValueError("R7 data mount inventory changed")
    if metadata.get("kernel_sources") != [
        "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5-runner"
    ]:
        raise ValueError("R7 source mount inventory changed")
    release = pending_release if release_path is None else release_path.read_bytes()
    if release_path is not None:
        parsed = _validate_release_bytes(bindings_bytes, release)
        if parsed.get("transport_target_kernel_id") != R8_ID:
            raise PermissionError("ROOT receipt is not authorized for R8 transport")
    source = WRAPPER_TEMPLATE
    substitutions = {
        "@BODY_B64@": b64_lines(body),
        "@BINDINGS_B64@": b64_lines(bindings_bytes),
        "@RELEASE_B64@": b64_lines(release),
        "@BODY_SHA@": sha256(body),
        "@BINDINGS_SHA@": sha256(bindings_bytes),
        "@RELEASE_SHA@": sha256(release),
    }
    for marker, value in substitutions.items():
        source = source.replace(marker, value)
    if "@" in source:
        raise AssertionError("unexpanded R8 wrapper template marker")
    if len(source.encode("utf-8")) >= 1_000_000:
        raise ValueError("R8 single-file kernel exceeds the 1 MB source envelope")
    output.mkdir(parents=True, exist_ok=False)
    kernel_out = output / "kernel.py"
    with kernel_out.open("xb") as target:
        target.write(source.encode("utf-8"))
    metadata_out = dict(metadata)
    metadata_out["id"] = R8_ID
    metadata_out["title"] = R8_TITLE
    with (output / "kernel-metadata.json").open("x", encoding="utf-8") as target:
        json.dump(metadata_out, target, indent=2, sort_keys=True)
        target.write("\n")
    result = {
        "schema": "wnn.kaggle.r8_transport_build.r1",
        "status": "pending_root_release" if release_path is None else "root_release_embedded",
        "target_kernel_id": R8_ID,
        "target_title": R8_TITLE,
        "embedded_r7_body_sha256": sha256(body),
        "embedded_r7_bindings_sha256": sha256(bindings_bytes),
        "embedded_root_release_sha256": sha256(release),
        "r8_kernel_py_sha256": sha256(kernel_out.read_bytes()),
        "r8_kernel_py_bytes": kernel_out.stat().st_size,
        "r8_metadata_sha256": sha256((output / "kernel-metadata.json").read_bytes()),
        "declared_upload_files": ["kernel.py", "kernel-metadata.json"],
        "build_receipt_path": str(receipt_path),
        "r7_dataset_sources": metadata_out["dataset_sources"],
        "r7_kernel_sources": metadata_out["kernel_sources"],
        "transport_authorization": "required in ROOT release as transport_target_kernel_id",
        "science_recipe_binding_target": "vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r7",
    }
    with receipt_path.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    return result


def _validate_release_bytes(bindings_bytes: bytes, release_bytes: bytes) -> dict[str, Any]:
    bindings = json.loads(bindings_bytes)
    release = json.loads(release_bytes)
    if not isinstance(bindings, dict) or not isinstance(release, dict):
        raise ValueError("binding/release must be JSON objects")
    if release.get("schema") != "wnn.root.clean_t3_f0_export_only_release.r7.r1":
        raise PermissionError("unexpected ROOT export release schema")
    if release.get("status") != "admitted" or release.get("authorized") is not True:
        raise PermissionError("ROOT release is not admitted")
    if release.get("binding") != bindings.get("root_release_binding"):
        raise PermissionError("ROOT release recipe binding differs")
    if release.get("logical_recipe_target_kernel_id") != bindings["root_release_binding"].get(
        "target_kernel_id"
    ):
        raise PermissionError("ROOT logical target differs from the R7 recipe binding")
    return release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root-release", type=Path)
    args = parser.parse_args()
    result = build(args.repo, args.output, args.root_release, args.receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
