"""Build fresh E427 Oct 5 draft identities without sealing or launching them."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "draft_specs"
ARMS = ("aligned", "stale", "noop")
OLD = ROOT / "research/campaign_2026-10-04/staging"
PIPE = OLD / "e427_pipeline_v1/execution_specs"
SMOKE = OLD / "e427_smoke_review/specs"
RUNS = {a: f"c20261004_e427_{a}_s3_1ep_v2" for a in ARMS}
FIT_SPEC_NAMES = {a: f"oct05_e427_{a}_s3_1ep_r1" for a in ARMS}
ID = {a: f"oct05-e427-{a}-s3-r1" for a in ARMS}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(name: str, value: dict) -> None:
    code = value.setdefault("code", [])
    for path in ("research/campaign_2026-10-05/volume_gate_recovery/controller.py",
                 "research/campaign_2026-10-05/volume_gate_recovery/build_drafts.py",
                 "research/campaign_2026-10-05/volume_gate_recovery/PREREG.md"):
        if path not in code:
            code.append(path)
    value["name"] = name
    dest = OUT / f"{name}.draft.json"
    dest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def swap(value: dict, replacements: dict[str, str]) -> dict:
    for field in ("argv", "artifacts", "inputs", "code"):
        if field in value:
            updated = []
            for item in value[field]:
                if isinstance(item, str):
                    for old, new in replacements.items():
                        item = item.replace(old, new)
                updated.append(item)
            value[field] = updated
    return value


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Remove only the obsolete draft identities emitted by an earlier version
    # of this new packet generator; never touch any Oct 4 source or runner file.
    for stale in OUT.glob("c20261005_e427_*_s3_1ep_r1.draft.json"):
        stale.unlink()
    for arm in ARMS:
        smoke = read(SMOKE / f"e427_smoke_{arm}_s3_b32x2_v1.draft.json")
        smoke_name = f"oct05_e427_smoke_{arm}_s3_b32x2_r1"
        smoke = swap(smoke, {
            f"e427_smoke_{arm}_s3_b32x2_v1": smoke_name,
            f"research/campaign_2026-10-04/checks/e427_smoke_{arm}_s3_b32x2_v1.json":
                f"research/campaign_2026-10-05/volume_gate_recovery/results/{smoke_name}.json",
        })
        write(smoke_name, smoke)

        fit = read(PIPE / f"c20261004_e427_{arm}_s3_1ep_v2.draft.json")
        old_run = f"c20261004_e427_{arm}_s3_1ep_v2"
        fit_name = RUNS[arm]
        fit_spec_name = FIT_SPEC_NAMES[arm]
        fit = swap(fit, {
            old_run: fit_name,
            f"research/campaign_2026-10-04/staging/e427_smoke_review":
                "research/campaign_2026-10-04/staging/e427_smoke_review",
        })
        write(fit_spec_name, fit)

        export = read(PIPE / f"e427_predict_{arm}_fp32_v2.draft.json")
        export_name = f"oct05_e427_export_{arm}_fp32_r1"
        export = swap(export, {
            old_run: fit_name,
            f"e427-predict-{arm}-fp32-v2": export_name,
            f"c20261004_e427_{arm}_s3_1ep_v2": fit_name,
            f"research/campaign_2026-10-04/staging/e427_pipeline_v1/scripts/predict_e427_fp32.py":
                f"research/campaign_2026-10-04/staging/e427_pipeline_v1/scripts/predict_e427_fp32.py",
            f"cache/pred_c20261004_e427_{arm}_s3_1ep_v2_valid.npy":
                f"cache/pred_{fit_name}_valid.npy",
            f"cache/pred_c20261004_e427_{arm}_s3_1ep_v2_valid.json":
                f"cache/pred_{fit_name}_valid.json",
        })
        write(export_name, export)

    score = read(PIPE / "e427_score_component_panels_v1.draft.json")
    score_name = "oct05_e427_score_component_panels_r1"
    score = swap(score, {
        "e427_report_v1.json": "e427_report_oct05_r1.json",
        "research/campaign_2026-10-04/staging/e427_pipeline_v1/results/e427_report_v1.json":
            "research/campaign_2026-10-05/volume_gate_recovery/results/e427_report_oct05_r1.json",
    })
    # Each fit output reference and prediction reference is replaced with the new identity.
    inputs = []
    for item in score["inputs"]:
        for arm in ARMS:
            old = f"c20261004_e427_{arm}_s3_1ep_v2"
            item = item.replace(old, RUNS[arm])
            item = item.replace(f"e427-{arm}-s3-1ep-v2", ID[arm])
        item = item.replace("research/campaign_2026-10-04/runner/attempts/",
                            "research/campaign_2026-10-05/volume_gate_recovery/runner/attempts/")
        inputs.append(item)
    score["inputs"] = inputs
    write(score_name, score)

    (OUT / "attempt_ids.json").write_text(json.dumps({
        "schema": "wnn.e427-oct05-identities-v1",
        "smoke": {a: f"oct05-e427-{a}-smoke-b32x2-r1" for a in ARMS},
        "fit": ID,
        "export": {a: f"oct05-e427-{a}-export-fp32-r1" for a in ARMS},
        "score": "oct05-e427-score-panels-r1",
        "runs": RUNS,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
