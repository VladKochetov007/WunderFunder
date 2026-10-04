"""Draft E464 fixed-epoch-0 exports and the full fixed parent/head report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
STAGING = CAMPAIGN / "staging/e464"
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
EXPORT_CODE = {
    "research/campaign_2026-10-04/staging/e464/scripts/predict_e464_head.py",
    "research/campaign_2026-10-04/checks/predict_e424_recurrent.py",
    "scripts/train_rnn.py",
    "scripts/predict_run.py",
}
SCORE_RUNS = {
    "c20261004_combo_s3_final_r1": ("ep3", "fp32"),
    "c20261004_combo_head_s3_final": ("ep0", "fp32"),
}
PARENT_SOURCE_SPECS = [f"research/campaign_2026-10-04/specs/e424_{arm}_v2.sealed.json"
                       for arm in PARENTS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def code_paths() -> list[str]:
    paths = set(EXPORT_CODE)
    paths.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "src/wnn").rglob("*.py"))
    for relative_spec in PARENT_SOURCE_SPECS:
        parent_spec = json.loads((ROOT / relative_spec).read_text())
        paths.update(item["path"] for item in parent_spec.get("code", []))
    if (ROOT / "utils.py").is_file():
        paths.add("utils.py")
    return sorted(paths)


def source_manifest(paths: list[str]) -> dict[str, object]:
    files = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative, "size": path.stat().st_size,
                      "sha256": sha256_file(path)})
    payload = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {"schema": "e464-head-predictor-source-closure-v1", "files": files,
            "sha256": hashlib.sha256(payload).hexdigest()}


def prediction_draft(arm: str, parent: str, child: str, code: list[str]) -> dict[str, object]:
    parent_spec = f"research/campaign_2026-10-04/specs/e424_{arm}_v2.sealed.json"
    audit = f"research/campaign_2026-10-04/staging/e464/audits/{child}.json"
    return {
        "version": 1,
        "name": f"e464_predict_{child}_fp32_v1",
        "kind": "gpu",
        "argv": [
            "{python}", "research/campaign_2026-10-04/staging/e464/scripts/predict_e464_head.py",
            arm, child, "valid", "--epoch", "0", "--inference-precision", "fp32",
            "--output-name", child, "--source-manifest",
            "research/campaign_2026-10-04/staging/e464/specs/e464_predictor_source_closure.draft.json",
        ],
        "code": code,
        "inputs": [
            "datasets/valid.parquet", "cache/valid_targets.npz",
            f"artifacts/runs/{parent}/model_ep3.pt",
            f"artifacts/runs/{parent}/provenance.json", parent_spec,
            f"artifacts/runs/{child}/model_ep0.pt",
            f"artifacts/runs/{child}/provenance.json", audit,
            *PARENT_SOURCE_SPECS,
            "research/campaign_2026-10-04/staging/e464/specs/e464_predictor_source_closure.draft.json",
        ],
        "artifacts": [f"cache/pred_{child}_valid.npy", f"cache/pred_{child}_valid.json"],
        "limits": {"cpus": 8, "ram_bytes": 24 * 1024**3,
                   "gpu_memory_bytes": 4_200_000_000, "gpu_stop_bytes": 4_800_000_000,
                   "timeout_seconds": 1800},
        "environment": {"WNN_LOADER_WORKERS": "2", "WNN_LOADER_RAM_GB": "3"},
    }


def paired_screen() -> dict[str, object]:
    labels = {
        "h093_learned": "E464_H093_LEARNED",
        "h093_hash": "E464_H093_HASH",
        "h093_universal": "E464_H093_UNIVERSAL",
        "h093_gru_budget": "E464_H093_GRU",
        "h094_conditional": "E464_H094_CONDITIONAL",
        "h094_unconditional": "E464_H094_UNCONDITIONAL",
        "h094_diagonal": "E464_H094_DIAGONAL",
        "h094_gru_budget": "E464_H094_GRU",
    }
    arms = {
        label: {PARENTS[arm]: 0.5, f"c20261004_e464_{arm}_head_s3_final": 0.5}
        for arm, label in labels.items()
    }
    arms["COMBO"] = {
        "c20261004_combo_s3_final_r1_fp32": 0.5,
        "c20261004_combo_head_s3_final_fp32": 0.5,
    }
    comparisons = [
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
    return {"folds": ["holdout"], "arms": arms, "comparisons": comparisons,
            "export_runs": {
                "c20261004_combo_s3_final_r1_fp32": "c20261004_combo_s3_final_r1",
                "c20261004_combo_head_s3_final_fp32": "c20261004_combo_head_s3_final",
            },
            "n_boot": 2000,
            "provenance": "fixed E424 epoch-3 parent plus E464 fixed epoch-0 head, each coefficient 0.5; no holdout selection"}


def score_draft(screen_path: Path, output_path: Path, child_runs: list[str]) -> dict[str, object]:
    template_path = CAMPAIGN / "specs/e424_score_all_v3.draft.json"
    template = json.loads(template_path.read_text())
    screen_rel, output_rel = rel(screen_path), rel(output_path)
    template["name"] = "e464_score_all_parent_head_fp32_v1"
    template["argv"] = [
        "{python}", "research/campaign_2026-10-04/staging/e464/scripts/score_e464_all.py",
        screen_rel, "--output", output_rel, "--source-manifest",
        "research/campaign_2026-10-04/staging/e464/specs/e464_predictor_source_closure.draft.json",
    ]
    template["code"] = sorted(set(template["code"]) | {
        "scripts/screen_score.py",
        "research/campaign_2026-10-04/staging/e464/scripts/score_e464_all.py",
        "research/campaign_2026-10-04/staging/e464/scripts/predict_e464_head.py",
    } | set(code_paths()))
    runs = dict(SCORE_RUNS)
    for arm, parent in PARENTS.items():
        runs[parent] = ("ep3", "fp32")
        runs[f"c20261004_e464_{arm}_head_s3_final"] = ("ep0", "fp32")
    inputs = [
        "research/campaign_2026-09-29/artifacts/splits/fold0.json",
        "cache/valid_targets.npz", "datasets/valid.parquet", "cache/chains.npz",
        screen_rel,
        "cache/pred_c20261004_h093_learned_s3_final_valid.json",
        "research/campaign_2026-10-04/staging/e464/specs/e464_predictor_source_closure.draft.json",
        *PARENT_SOURCE_SPECS,
    ]
    for run, (epoch, precision) in runs.items():
        export_run = run + "_fp32" if run in SCORE_RUNS else run
        inputs += [f"cache/pred_{export_run}_valid.npy", f"cache/pred_{export_run}_valid.json",
                   f"artifacts/runs/{run}/model_{epoch}.pt",
                   f"artifacts/runs/{run}/provenance.json"]
        if run.startswith("c20261004_e464_"):
            inputs.append(f"research/campaign_2026-10-04/staging/e464/audits/{run}.json")
    template["inputs"] = sorted(set(inputs))
    template["artifacts"] = [output_rel]
    template["limits"] = {"cpus": 1, "ram_bytes": 2 * 1024**3,
                           "gpu_memory_bytes": 0, "gpu_stop_bytes": 0,
                           "timeout_seconds": 900}
    template["environment"] = {}
    return template


def main() -> None:
    paths = code_paths()
    closure = source_manifest(paths)
    write_json(STAGING / "specs/e464_predictor_source_closure.draft.json", closure)
    children = []
    for arm, parent in PARENTS.items():
        child = f"c20261004_e464_{arm}_head_s3_final"
        children.append(child)
        write_json(STAGING / "specs" / f"e464_predict_{arm}_fp32_v1.draft.json",
                   prediction_draft(arm, parent, child, paths))
    screen_path = STAGING / "specs/e464_screen_all_fp32_v1.draft.json"
    write_json(screen_path, paired_screen())
    score_path = STAGING / "results/e464_report_all_fp32_v1.json"
    score = score_draft(screen_path, score_path, children)
    write_json(STAGING / "specs/e464_score_all_parent_head_fp32_v1.draft.json", score)
    print(json.dumps({"status": "draft_unsealed", "predictors": len(children),
                      "comparisons": len(paired_screen()["comparisons"]),
                      "source_files": len(paths),
                      "source_closure_sha256": closure["sha256"],
                      "e424_predictor_sha256": sha256_file(CAMPAIGN / "checks/predict_e424_recurrent.py")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
