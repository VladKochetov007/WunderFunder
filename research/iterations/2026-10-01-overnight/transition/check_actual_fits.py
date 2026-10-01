"""Controller's post-fit integrity check; does not read prediction targets or scores."""
import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / "actual-fit-checks.json"
    assert not output.exists()
    plan = json.loads((HERE / "execution-plan.json").read_text())
    records = []
    for job in plan["jobs"]:
        argv = job["train_argv"]
        init = argv[argv.index("--init") + 1].split("@ep")[0]
        parent_path = ROOT / "artifacts/runs" / init / "model_ep3.pt"
        path = ROOT / "artifacts/runs" / job["name"] / "model_ep1.pt"
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["epoch"] == 1
        assert ck["args"]["valid_fold"] == job["fold"]
        meta = ck["args"]["transition_features"]
        assert meta["source_fold"] == job["fold"] and meta["arm"] == job["arm"]
        assert meta["nuisance_sha256"] == plan["nuisance_sha256"][f'nuisance_f{job["fold"]}.json']
        frozen = [key for key in parent["model"] if not key.startswith("head.")]
        assert all(torch.equal(parent["model"][key], ck["model"][key]) for key in frozen)
        assert all(torch.isfinite(t).all() for t in ck["model"].values())
        assert any(not torch.equal(parent["model"][key], ck["model"][key])
                   for key in parent["model"] if key.startswith("head."))
        extra = ck["model"]["head_extra.weight"]
        assert extra.shape == (128, 68) and torch.count_nonzero(extra) > 0
        records.append({"name": job["name"], "checkpoint_sha256": sha(path),
                        "parent_sha256": sha(parent_path), "epoch": ck["epoch"],
                        "fold": job["fold"], "arm": job["arm"],
                        "frozen_tensors_bitexact": frozen, "all_finite": True,
                        "head_changed": True, "extra_weight_norm": float(extra.norm())})
    output.write_text(json.dumps({"status": "PASS", "checked": len(records),
                                  "script_sha256": sha(Path(__file__)), "records": records}, indent=2) + "\n")
    print(f'PASS {len(records)} fitted artifacts')


if __name__ == "__main__":
    main()
