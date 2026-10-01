"""Controller check of the trained/exported ENSHEAD artifacts, without score access."""
import hashlib
import json
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20)
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
        assert ck["epoch"] == 1 and ck["args"]["valid_fold"] == job["fold"]
        assert ck["args"]["ensemble_adaptation"] == job["adaptation"]
        assert set(ck["model"]) == set(parent["model"])
        assert all(torch.isfinite(t).all() for t in ck["model"].values())
        frozen = [key for key in parent["model"] if not key.startswith("head.")]
        assert all(torch.equal(parent["model"][key], ck["model"][key]) for key in frozen)
        assert any(not torch.equal(parent["model"][key], ck["model"][key])
                   for key in parent["model"] if key.startswith("head."))
        scale = ck["ensemble_scale_parameters"]
        if job["adaptation"] == "scale":
            assert scale["offset_fixed_zero"] == [0.0, 0.0]
            factors = torch.tensor(scale["log_scale"], device="cuda").exp().cpu()
            assert torch.isfinite(factors).all() and (factors > 0).all()
            for key, original in parent["model"].items():
                expected = original.clone()
                if key == "head.2.weight": expected[:2] *= factors[:, None]
                elif key == "head.2.bias": expected[:2] *= factors
                assert torch.equal(expected, ck["model"][key]), key
        else:
            assert scale is None
        records.append({"name": job["name"], "checkpoint_sha256": sha(path),
                        "parent_sha256": sha(parent_path), "epoch": 1, "fold": job["fold"],
                        "adaptation": job["adaptation"], "scale_parameters": scale,
                        "all_finite": True, "frozen_backbone_bitexact": True,
                        "scale_export_exact_when_applicable": True})
    output.write_text(json.dumps({"status": "PASS", "checked": len(records),
                                  "script_sha256": sha(Path(__file__)), "records": records}, indent=2) + "\n")
    print(f'PASS {len(records)} ENSHEAD fitted artifacts')


if __name__ == "__main__":
    main()
