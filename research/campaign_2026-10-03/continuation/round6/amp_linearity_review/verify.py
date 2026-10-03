from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
AUTHOR = OWN.parent / "amp_linearity_audit"
ROOT = OWN.parents[4]


def digest(path: Path) -> str:
    """Read only code/document/provenance files using bounded streams."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Check author freeze and code structure without opening any actual tensors."""
    manifest = json.loads((AUTHOR / "HASHES.json").read_text())
    checks = {}
    for name, entry in manifest["artifacts"].items():
        actual = digest(AUTHOR / name)
        checks[name] = {
            "expected": entry["sha256"],
            "actual": actual,
            "matches": actual == entry["sha256"],
        }
    assert all(item["matches"] for item in checks.values())
    spec = json.loads((AUTHOR / "FROZEN_SPEC.json").read_text())
    assert all(digest(AUTHOR / name) == expected for name, expected in spec.items())
    preservation = json.loads((AUTHOR / "PRIOR_PRESERVATION.json").read_text())
    for name, identity in preservation.items():
        path = ROOT / "research/campaign_2026-10-03/continuation/round5" / name / "HASHES.json"
        assert digest(path) == identity["manifest_sha256"]
    adapter = ROOT / "research/campaign_2026-10-03/continuation/round4/live_capture_v4/fit_panel.py"
    accepted = json.loads(
        (
            ROOT
            / "research/campaign_2026-10-03/continuation/round4/capture_review_v4/candidate_runtime_hashes.json"
        ).read_text()
    )
    assert digest(adapter) == accepted["candidate_sha256"]["fit_panel.py"]
    tree = ast.parse((AUTHOR / "audit.py").read_text())
    main_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [node for node in ast.walk(main_function) if isinstance(node, ast.Call)]
    guarded = next(node.lineno for node in calls if ast.unparse(node.func) == "approval_guard")
    accessed = [
        node.lineno
        for node in calls
        if ast.unparse(node.func) in {"path.stat", "torch.load", "sha"}
    ]
    assert guarded < min(accessed)
    loads = [node for node in calls if ast.unparse(node.func) == "torch.load"]
    assert len(loads) == 1
    kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in loads[0].keywords}
    assert kwargs == {"map_location": "cpu", "mmap": True, "weights_only": False}
    chunk = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CHUNK" for target in node.targets)
    )
    workspace = 8 * chunk * 8
    assert workspace <= 32 * 1024**2
    forbidden_calls = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(
            word in ast.unparse(node.func) for word in ("autograd", "backward", "restore", "cuda")
        )
    ]
    assert not forbidden_calls
    result = {
        "all_author_files_match": True,
        "author_checks": checks,
        "author_manifest_sha256": digest(AUTHOR / "HASHES.json"),
        "frozen_spec_sha256": digest(AUTHOR / "FROZEN_SPEC.json"),
        "prior_manifests_only_checked": True,
        "capture_adapter_sha256": digest(adapter),
        "approval_precedes_capture_access": True,
        "Torch_load_CPU_mmap_only": True,
        "chunk_elements": chunk,
        "conservative_array_workspace_ceiling_bytes": workspace,
        "forbidden_model_or_GPU_calls": forbidden_calls,
        "actual_captured_tensor_reads": 0,
    }
    (OWN / "protected_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "all_author_files_match",
                    "approval_precedes_capture_access",
                    "Torch_load_CPU_mmap_only",
                    "conservative_array_workspace_ceiling_bytes",
                    "actual_captured_tensor_reads",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
