from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")


def main() -> None:
    for path in OUT.glob("f0_z3*_v1.json"):
        raw = path.read_text()
        for stage in ("student", "parent_hft", "student_hft"):
            raw = raw.replace(f"f0_z3_combo_s4_{stage}_v1", f"f0_z3_combo_s4_{stage}_v2")
        spec = json.loads(raw)
        if "hft" in spec["run_id"]:
            fit = next(step for step in spec["steps"] if step["name"] == "fit")
            fit["argv"][fit["argv"].index("--seed") + 1] = "3"
            spec["hypothesis"] = (
                "Exact Z3h fold0 analogue: seed4 backbone/student, seed3 GCorr finishing, fixed half endpoint"
            )
            spec["expected_outputs"].append(
                str(ROOT / "artifacts/runs" / spec["run_id"] / "provenance.json")
            )
            if "student_hft" in spec["run_id"]:
                core = "f0_z3_combo_s4_student_v2_core"
                spec["expected_outputs"].extend(
                    str(ROOT / "cache" / f"pred_{core}_valid{suffix}")
                    for suffix in (".npy", ".json")
                )
                spec["expected_outputs"].extend(
                    str(ROOT / "artifacts/runs" / core / name)
                    for name in ("model_ep1.pt", "provenance.json")
                )
        (OUT / path.name.replace("_v1", "_v2")).write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    main()
