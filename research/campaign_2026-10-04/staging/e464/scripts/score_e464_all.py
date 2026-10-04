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
ARCHES = {
    "h093_learned": "specialist",
    "h093_hash": "specialist",
    "h093_universal": "specialist",
    "h093_gru_budget": "gru",
    "h094_conditional": "prefix-adapter",
    "h094_unconditional": "prefix-adapter",
    "h094_diagonal": "prefix-adapter",
    "h094_gru_budget": "gru",
}
EXPECTED_COMPARISONS = [
    ["E464_H093_LEARNED", "E464_H093_GRU"],
    ["E464_H094_CONDITIONAL", "E464_H094_GRU"],
    ["E464_H093_LEARNED", "COMBO"],
    ["E464_H094_CONDITIONAL", "COMBO"],
    ["E464_H093_HASH", "E464_H093_GRU"],
    ["E464_H093_UNIVERSAL", "E464_H093_GRU"],
    ["E464_H094_UNCONDITIONAL", "E464_H094_GRU"],
    ["E464_H094_DIAGONAL", "E464_H094_GRU"],
    ["E464_H093_HASH", "COMBO"],
    ["E464_H093_UNIVERSAL", "COMBO"],
    ["E464_H094_UNCONDITIONAL", "COMBO"],
    ["E464_H094_DIAGONAL", "COMBO"],
    ["E464_H093_LEARNED", "E464_H093_UNIVERSAL"],
    ["E464_H093_LEARNED", "E464_H093_HASH"],
    ["E464_H094_CONDITIONAL", "E464_H094_DIAGONAL"],
    ["E464_H094_CONDITIONAL", "E464_H094_UNCONDITIONAL"],
]
COMBO_MIXTURE = {
    "c20261004_combo_s3_final_r1_fp32": 0.5,
    "c20261004_combo_head_s3_final_fp32": 0.5,
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
    if (
        not isinstance(groups, list)
        or len(groups) != 1873
        or any(type(group) is not int for group in groups)
        or len(set(groups)) != 1873
        or groups != canonical
    ):
        raise ValueError(f"{label} is not the unique canonical all-VALID sequence domain")


def verify_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    if (
        manifest.get("schema") != "e464-head-predictor-source-closure-v1"
        or not isinstance(files, list)
        or manifest.get("sha256") != closure_digest(files)
    ):
        raise ValueError("E464 predictor source manifest is malformed")
    for item in files:
        path = ROOT / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != item["size"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"E464 predictor source changed: {item['path']}")
    return manifest


def verify_exports(screen_path: Path, manifest_path: Path) -> None:
    import numpy as np

    manifest = verify_manifest(manifest_path)
    valid_control_path = ROOT / "cache/pred_c20261004_h093_learned_s3_final_valid.json"
    valid_control = json.loads(valid_control_path.read_text())
    canonical = valid_control.get("row_groups")
    verify_all_valid(canonical, canonical, "fixed E424 all-VALID control")
    screen = json.loads(screen_path.read_text())
    if screen.get("folds") != ["holdout"] or screen.get("n_boot") != 2000:
        raise ValueError("E464 screen changed its fixed historical panel/bootstrap")
    if screen.get("export_runs") != {
        "c20261004_combo_s3_final_r1_fp32": "c20261004_combo_s3_final_r1",
        "c20261004_combo_head_s3_final_fp32": "c20261004_combo_head_s3_final",
    }:
        raise ValueError("E464 screen combo export-to-checkpoint mapping changed")
    arms = screen.get("arms", {})
    expected_arm_names = {
        "COMBO",
        "E464_H093_LEARNED",
        "E464_H093_HASH",
        "E464_H093_UNIVERSAL",
        "E464_H093_GRU",
        "E464_H094_CONDITIONAL",
        "E464_H094_UNCONDITIONAL",
        "E464_H094_DIAGONAL",
        "E464_H094_GRU",
    }
    if set(arms) != expected_arm_names or arms.get("COMBO") != COMBO_MIXTURE:
        raise ValueError(
            "E464 screen arms or fixed combo control differ from the registered design"
        )
    if screen.get("comparisons") != EXPECTED_COMPARISONS:
        raise ValueError("E464 screen does not contain the exact 16 ordered registered contrasts")
    for arm, parent in PARENTS.items():
        child = f"c20261004_e464_{arm}_head_s3_final"
        label = {
            "h093_learned": "E464_H093_LEARNED",
            "h093_hash": "E464_H093_HASH",
            "h093_universal": "E464_H093_UNIVERSAL",
            "h093_gru_budget": "E464_H093_GRU",
            "h094_conditional": "E464_H094_CONDITIONAL",
            "h094_unconditional": "E464_H094_UNCONDITIONAL",
            "h094_diagonal": "E464_H094_DIAGONAL",
            "h094_gru_budget": "E464_H094_GRU",
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
        for path in (
            pred_path,
            sidecar_path,
            checkpoint_path,
            provenance_path,
            parent_checkpoint,
            parent_source,
            audit_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"E464 score input missing: {path}")
        sidecar = json.loads(sidecar_path.read_text())
        audit = json.loads(audit_path.read_text())
        provenance = json.loads(provenance_path.read_text())
        parent_provenance_path = ROOT / f"artifacts/runs/{parent}/provenance.json"
        parent_provenance = json.loads(parent_provenance_path.read_text())
        prediction = np.load(pred_path, mmap_mode="r", allow_pickle=False)
        if prediction.shape != (1873, 20_000, 2) or prediction.dtype != np.float32:
            raise ValueError(f"{child}: prediction cache must be FP32 [1873,20000,2]")
        del prediction
        checkpoint_sha = sha256_file(checkpoint_path)
        parent_sha = sha256_file(parent_checkpoint)
        child_provenance_sha = sha256_file(provenance_path)
        parent_provenance_sha = sha256_file(parent_provenance_path)
        audit_sha = sha256_file(audit_path)
        parent_spec_sha = sha256_file(parent_source)
        if (
            sidecar.get("run") != child
            or sidecar.get("parent_run") != parent
            or sidecar.get("epoch") != 0
            or sidecar.get("inference_precision") != "fp32"
            or sidecar.get("outputs_exported") != 2
            or sidecar.get("model_outputs") != 23
            or sidecar.get("prediction_sha256") != sha256_file(pred_path)
            or sidecar.get("checkpoint_sha256") != checkpoint_sha
            or sidecar.get("parent_checkpoint_sha256") != parent_sha
            or sidecar.get("child_provenance_sha256") != child_provenance_sha
            or sidecar.get("parent_provenance_sha256") != parent_provenance_sha
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
            or provenance.get("args", {}).get("arch") != ARCHES[arm]
            or provenance.get("args", {}).get("init") != f"{parent}@ep3"
            or provenance.get("args", {}).get("checkpoint_policy") != "fixed"
            or provenance.get("args", {}).get("checkpoint_epoch") != 0
            or provenance.get("args", {}).get("aux_struct") != 0.3
            or provenance.get("args", {}).get("aux_next") != 0.3
            or provenance.get("parent_checkpoint_sha256") not in (None, parent_sha)
            or len(provenance.get("fit_row_groups", {}).get("valid", [])) != 1505
            or set(provenance.get("fit_row_groups", {}).get("valid", []))
            != set(parent_provenance.get("fit_row_groups", {}).get("valid", []))
            or parent_provenance.get("selected_epoch") != 3
            or parent_provenance.get("checkpoint_sha256", {}).get("model_ep3.pt") != parent_sha
            or parent_provenance.get("args", {}).get("checkpoint_policy") != "fixed"
            or parent_provenance.get("args", {}).get("checkpoint_epoch") != 3
            or parent_provenance.get("args", {}).get("arch") != ARCHES[arm]
            or parent_provenance.get("args", {}).get("aux_struct") != 0.3
            or parent_provenance.get("args", {}).get("aux_next") != 0.3
            or audit.get("arm") != arm
            or audit.get("child_run") != child
            or audit.get("parent_run") != parent
            or audit.get("fixed_epoch") != 0
        ):
            raise ValueError(f"{child}: export, checkpoint, provenance and audit lineage differ")
        verify_all_valid(sidecar.get("row_groups"), canonical, f"{child} prediction")
        closure = sidecar.get("code_source_closure", {})
        if (
            closure.get("schema") != manifest["schema"]
            or closure.get("files") != manifest["files"]
            or closure.get("sha256") != manifest["sha256"]
            or sidecar.get("export_source_manifest_sha256") != sha256_file(manifest_path)
        ):
            raise ValueError(f"{child}: prediction code closure differs from the sealed manifest")
        for item in json.loads(parent_source.read_text()).get("code", []):
            path = ROOT / item["path"]
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"{child}: parent source closure changed at {item['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=(CAMPAIGN / "staging/e464/specs/e464_predictor_source_closure.draft.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite E464 report: {args.output}")
    verify_exports(args.screen, args.source_manifest)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/screen_score.py"),
            str(args.screen),
            "--output",
            str(args.output),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
