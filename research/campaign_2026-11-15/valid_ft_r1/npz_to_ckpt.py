"""Inverse of tlead_final_r1/export_npz.py: put original tl3.zip member NPZ tensors into a same-layout template
checkpoint so the PR49 trainer can fine-tune them (--init NAME@ep0). Aux head rows 2.. keep template values."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
RUNS = Path("/home/vlad/development/wunder/research_tmp/chrono_root/artifacts/runs")
def main(template: str, npz: str, name: str) -> None:
    ck = torch.load(RUNS / template / "model_ep3.pt", map_location="cpu", weights_only=False)
    z = np.load(npz); st = ck["model"]; assert set(z.files) == set(st)
    for k in z.files:
        v = torch.from_numpy(np.asarray(z[k], dtype=np.float32))
        if k.startswith("head.2."):
            new = st[k].clone(); new[:2] = v; v = new
        assert v.shape == st[k].shape, k
        st[k] = v.to(st[k].dtype)
    ck["args"] = dict(ck["args"], name=name); ck["epoch"] = 0
    ck["portable_import"] = {"kind": "npz_to_checkpoint", "npz": npz, "npz_sha256": hashlib.sha256(Path(npz).read_bytes()).hexdigest(),
                             "template": template, "note": "original team tl3.zip member; aux head rows from template"}
    (RUNS / name).mkdir(exist_ok=True); torch.save(ck, RUNS / name / "model_ep0.pt")
    for f in ("provenance.json",):
        if (RUNS / template / f).exists() and not (RUNS / name / f).exists(): (RUNS / name / f).write_bytes((RUNS / template / f).read_bytes())
    print(json.dumps({"name": name, "template": template, "npz": npz}))
if __name__ == "__main__":
    main(*sys.argv[1:4])
