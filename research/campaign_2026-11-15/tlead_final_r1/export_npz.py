"""Export an H256x2 tiedlead or linear-encoder GRU checkpoint to the TL3 wnn_cpp member NPZ layout (FP32, 2 main head rows)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch

RUNS = Path("/home/vlad/development/wunder/research_tmp/chrono_root/artifacts/runs")
REFS = {"tiedlead": Path("/home/vlad/development/wunder/research_tmp/q15_best_r1/tl3/weights_tl5.npz"),
        "linear": Path("/home/vlad/development/wunder/research_tmp/q15_best_r1/tl3/weights_s5.npz")}

def main(run: str, epoch: str, out: str) -> None:
    ck = torch.load(RUNS / run / f"model_ep{epoch}.pt", map_location="cpu", weights_only=False)
    a, state = ck["args"], ck["model"]
    assert a["arch"] == "gru" and a["enc"] in REFS and int(a["hidden"]) == 256 and int(a["layers"]) == 2
    ref = np.load(REFS[a["enc"]])
    assert set(state) == set(ref.files), sorted(set(state) ^ set(ref.files))
    arrays = {}
    for k in ref.files:
        v = state[k].float().numpy()
        if k.startswith("head.2."): v = v[:2]
        assert v.shape == ref[k].shape, (k, v.shape, ref[k].shape)
        arrays[k] = np.ascontiguousarray(v, dtype=np.float32)
    np.savez(out, **arrays)
    print(json.dumps({"run": run, "epoch": epoch, "npz": out,
        "sha256": hashlib.sha256(Path(out).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((RUNS / run / f"model_ep{epoch}.pt").read_bytes()).hexdigest()}))

if __name__ == "__main__":
    main(*sys.argv[1:4])
