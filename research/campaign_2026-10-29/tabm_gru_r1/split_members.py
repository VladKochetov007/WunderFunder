"""Split a TabM-GRU checkpoint into k plain-GRU run directories (exact member materialization).

Usage: split_members.py RUN EPOCH
Writes artifacts/runs/{RUN}_m{i}/model_ep{EPOCH}.pt with ensemble_k=1 args, so the ordinary
predict_run.py / native exporter can score each member. Provenance is copied and tagged.
"""

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wnn.training.models import make_model  # noqa: E402


def main() -> None:
    run, epoch = sys.argv[1], sys.argv[2]
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    ck = torch.load(ROOT / f"artifacts/runs/{run}/model_ep{epoch}.pt", map_location="cpu", weights_only=False)
    n_out = ck["model"]["head.2.weight"].shape[0]
    model = make_model(ck["args"], ck["n_in"], n_out)
    model.load_state_dict(ck["model"])
    for member in range(model.ensemble_k):
        out = copy.copy(ck)
        out["args"] = {**ck["args"], "ensemble_k": 1, "name": f"{run}_m{member}"}
        out["model"] = model.member_state_dict(member)
        out["provenance"] = {**ck["provenance"], "tabm_member_of": run, "tabm_member": member}
        plain = make_model(out["args"], ck["n_in"], n_out)
        plain.load_state_dict(out["model"])
        dest = ROOT / f"artifacts/runs/{run}_m{member}"
        dest.mkdir(parents=True, exist_ok=True)
        torch.save(out, dest / f"model_ep{epoch}.pt")
        print(dest)


if __name__ == "__main__":
    main()
