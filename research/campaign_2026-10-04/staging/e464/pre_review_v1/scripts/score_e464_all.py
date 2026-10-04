"""Verify E464 parent/head export lineage, then call the frozen paired scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
PARENTS = {
    "h093_learned": "c20261004_h093_learned_s3_final",
    "h093_hash": "c20261004_h093_hash_s3_final",
    "h093_universal": "c20261004_h093_universal_s3_final",
    "h093_gru_budget": "c20261004_h093_gru_budget_s3_final",
    "h094_conditional": "c20261004_h094_conditional_s3_final",
    "h094_unconditional": "c20261004_h094_unconditional_s3_final",
    "h094_diagonal": "c20261004_h094_diagonal_s3_final",
    "h094_gru_budget": "c20261004_h094_gru_budget_s3_final",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def closure_digest(files: list[dict[str, object]]) -> str:
    payload = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_all_valid(groups: object, canonical: list[object], label: str) -> None:
    if (not isinstance(groups, list) or len(groups) != 1873
            or any(type(group) is not int for group in groups)
            or len(set(groups)) != 1873 or groups != canonical):
        raise ValueError(f"{label} is not the unique canonical all-VALID sequence domain")


def verify_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    if (manifest.get("schema") != "e464-head-predictor-source-closure-v1"
            or not isinstance(files, list)
            or manifest.get("sha256") != closure_digest(files)):
        raise ValueError("E464 predictor source manifest is malformed")
    for item in files:
        path = ROOT / item["path"]
        if (not path.is_file() or path.stat().st_size != item["size"]
                or sha256_file(path) != item["sha256"]):
            raise ValueError(f"E464 predictor source changed: {item['path']}")
    return manifest


def verify_exports(screen_path: Path, manifest_path: Path) -> None:
    manifest = verify_manifest(manifest_path)
    valid_control_path = ROOT / "cache/pred_c20261004_h093_learned_s3_final_valid.json"
    valid_control = json.loads(valid_control_path.read_text())
    canonical = valid_control.get("row_groups")
    verify_all_valid(canonical, canonical, "fixed E424 all-VALID control")
    screen = json.loads(screen_path.read_text())
    if screen.get("folds") != ["holdout"] or screen.get("n_boot") != 2000:
        raise ValueError("E464 screen changed its fixed historical panel/bootstrap")
    arms = screen.get("arms", {})
    for arm, parent in PARENTS.items():
        child = f"c20261004_e464_{arm}_head_s3_final"
        label = {
            "h093_learned": "E464_H093_LEARNED", "h093_hash": "E464_H093_HASH",
            "h093_universal": "E464_H093_UNIVERSAL", "h093_gru_budget": "E464_H093_GRU",
            "h094_conditional": "E464_H094_CONDITIONAL",
            "h094_unconditional": "E464_H094_UNCONDITIONAL",
            "h094_diagonal": "E464_H094_DIAGONAL", "h094_gru_budget": "E464_H094_GRU",
        }[arm]
        if arms.get(label) != {parent: 0.5, child: 0.5}:
            raise ValueError(f"{label} is not the fixed equal parent/head mixture")
        pred_path = ROOT / f"cache/pred_{child}_valid.npy"
        sidecar_path = pred_path.with_suffix(".json")
        checkpoint_path = ROOT / f"artifacts/runs/{child}/model_ep0.pt"
        provenance_path = ROOT / f"artifacts/runs/{child}/provenance.json"
        parent_checkpoint = ROOT / f"artifacts/runs/{parent}/model_ep3.pt"
        parent_source = CAMPAIGN / f"specs/e424_{arm}_v2.sealed.json"
        audit_path = CAMPAIGN / f"staging/e464/audits/{child}.json"
        for path in (pred_path, sidecar_path, checkpoint_path, provenance_path,
                     parent_checkpoint, parent_source, audit_path):
            if not path.is_file():
                raise FileNotFoundError(f"E464 score input missing: {path}")
        sidecar = json.loads(sidecar_path.read_text())
        audit = json.loads(audit_path.read_text())
        provenance = json.loads(provenance_path.read_text())
        checkpoint_sha = sha256_file(checkpoint_path)
        parent_sha = sha256_file(parent_checkpoint)
        audit_sha = sha256_file(audit_path)
        parent_spec_sha = sha256_file(parent_source)
        if (
            sidecar.get("run") != child or sidecar.get("parent_run") != parent
            or sidecar.get("epoch") != 0 or sidecar.get("inference_precision") != "fp32"
            or sidecar.get("prediction_sha256") != sha256_file(pred_path)
            or sidecar.get("checkpoint_sha256") != checkpoint_sha
            or sidecar.get("parent_checkpoint_sha256") != parent_sha
            or sidecar.get("head_audit_sha256") != audit_sha
            or sidecar.get("parent_code_spec_sha256") != parent_spec_sha
            or sidecar.get("head_audit_status") != "pass"
            or audit.get("status") != "pass"
            or audit.get("child_checkpoint_sha256") != checkpoint_sha
            or audit.get("parent_checkpoint_sha256") != parent_sha
            or audit.get("parent_code_spec_sha256") != parent_spec_sha
            or audit.get("non_head_state_mismatches") != []
            or audit.get("head_layout_mismatches") != []
            or provenance.get("selected_epoch") != 0
            or provenance.get("checkpoint_sha256", {}).get("model_ep0.pt") != checkpoint_sha
        ):
            raise ValueError(f"{child}: export, checkpoint, provenance and audit lineage differ")
        verify_all_valid(sidecar.get("row_groups"), canonical, f"{child} prediction")
        closure = sidecar.get("code_source_closure", {})
        if (closure.get("schema") != manifest["schema"]
                or closure.get("files") != manifest["files"]
                or closure.get("sha256") != manifest["sha256"]
                or sidecar.get("export_source_manifest_sha256") != sha256_file(manifest_path)):
            raise ValueError(f"{child}: prediction code closure differs from the sealed manifest")
        for item in json.loads(parent_source.read_text()).get("code", []):
            path = ROOT / item["path"]
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"{child}: parent source closure changed at {item['path']}")
    if len(screen.get("comparisons", [])) != 16:
        raise ValueError("E464 screen must retain all 16 preregistered comparisons")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=(
        CAMPAIGN / "staging/e464/specs/e464_predictor_source_closure.draft.json"
    ))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite E464 report: {args.output}")
    verify_exports(args.screen, args.source_manifest)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/screen_score.py"), str(args.screen),
        "--output", str(args.output),
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
