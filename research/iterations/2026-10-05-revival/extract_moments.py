"""Frozen blend screen; row moments exported for chain bootstrap on DataSphere."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

R = Path.home() / "WunderFunder-creative"
OUT = Path(os.environ.get("WUNDER_REVIVAL_OUT", Path(__file__).resolve().parent))
base = {"cmp_fullft_s3": .375, "cmp_fullft_s4": .375, "orgbase": .0625}
def blend(names):
    return {"cmp_fullft_s3": .275, "cmp_fullft_s4": .275,
            "orgbase": .0625, **{n: .2 / len(names) for n in names}}
specs = {
    "base_pair_org": base,
    "raw_reference": blend(["cmp_rawwmse_s3"]),
    "lstm_pair": blend(["cmp_lstm_s3", "cmp_lstm_s4"]),
    "dual_seed3": blend(["cmp_dualc_s3"]),
    "twostream_seed3": blend(["cmp_tstream_s3"]),
}
if os.environ.get("WUNDER_REVIVAL_INCLUDE_NEW") == "1":
    specs.update({
        "raw_pair": blend(["cmp_rawwmse_s3", "cmp_rawwmse65r2_s4"]),
        "twostream_pair": blend(["cmp_tstream_s3", "cmp_tstream_s4"]),
        "onorm_seed3": blend(["cmp_onorm65r2_s3"]),
        "onorm_seed4": blend(["cmp_onorm65r2_s4"]),
        "onorm_pair": blend(["cmp_onorm65r2_s3", "cmp_onorm65r2_s4"]),
        "raw_seed4": blend(["cmp_rawwmse65r2_s4"]),
        "twostream_seed4": blend(["cmp_tstream_s4"]),
    })

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "moments.npz").exists():
        raise FileExistsError("Refusing to replace previous moments; choose a new output directory")
    cf = np.load(R / "cache/chain_folds.npz")
    trained = ((cf["split"] == "valid") & (cf["fold"] != 0)) | np.array([
        s == "train" and int(hashlib.sha256(f"half:{g}".encode()).hexdigest(), 16) % 2 == 0
        for s, g in zip(cf["split"], cf["row_group"])])
    seen_chains = set(cf["chain"][trained].tolist())
    payload = {"names": np.array(list(specs))}
    inventory = {}
    for split in ("valid", "train"):
        z = np.load(R / f"cache/{split}_targets.npz")
        y, groups = z["y"], z["row_group"]
        select = cf["split"] == split
        chains = dict(zip(cf["row_group"][select], cf["chain"][select]))
        folds = dict(zip(cf["row_group"][select], cf["fold"][select]))
        if split == "valid":
            ids = np.array([i for i, g in enumerate(groups) if folds[g] == 0])
            mask = z["mask"]
        else:
            half = np.array([int(hashlib.sha256(f"half:{g}".encode()).hexdigest(), 16) % 2 for g in groups])
            ids = np.random.default_rng(1).choice(np.where(half == 1)[0], 2000, replace=False)
            mask = np.load(R / "cache/propcache_prop_f0_train.npy", mmap_mode="r")
        arrays = {}
        for name in {n for spec in specs.values() for n in spec}:
            path = R / f"cache/pred_{name}_{split}.npy"
            arrays[name] = np.load(path, mmap_mode="r")
            st = path.stat()
            inventory[str(path)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
            assert arrays[name].shape == y.shape, (name, arrays[name].shape, y.shape)
        moments = np.zeros((len(specs), len(ids), 2, 6))
        for ii, i in enumerate(ids):
            yy = np.clip(y[i], -2, 2).astype(np.float64)
            q = np.array(mask[i if split == "valid" else groups[i]], dtype=np.float64)
            q[:99] = 0
            w = np.abs(yy) * q[:, None]
            row = {n: np.asarray(a[i], dtype=np.float64) for n, a in arrays.items()}
            for k, spec in enumerate(specs.values()):
                p = np.clip(sum(c * row[n] for n, c in spec.items()), -2, 2)
                assert np.isfinite(p).all()
                moments[k, ii] = np.stack([w.sum(0), (w*yy).sum(0), (w*p).sum(0),
                    (w*yy*yy).sum(0), (w*p*p).sum(0), (w*yy*p).sum(0)], axis=-1)
            if ii % 500 == 0:
                print(split, ii, "/", len(ids), flush=True)
        payload[split] = moments
        payload[split + "_chains"] = np.array([chains[g] for g in groups[ids]])
        payload[split + "_rowgroups"] = groups[ids]
        clean = np.array([chains[g] not in seen_chains for g in groups[ids]])
        payload[split + "_clean"] = moments[:, clean]
        payload[split + "_clean_chains"] = payload[split + "_chains"][clean]
        print(split, "clean chains sequences", int(clean.sum()), "/", len(ids), flush=True)
    np.savez_compressed(OUT / "moments.npz", **payload)
    (OUT / "inputs.json").write_text(json.dumps({"specs": specs, "inputs": inventory,
        "scope": "Exploratory reused composite proxy, not the full public incumbent; train mask is estimated. Chain resampling does not remove training overlap or adaptive selection."}, indent=2))
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
