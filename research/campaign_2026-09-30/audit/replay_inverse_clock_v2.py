"""Independent raw-input replay of frozen inverse maps; never refit a map."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pyarrow.parquet as pq
from scipy.special import ndtr

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[2]
DATA = AUDIT.parent / "data_reconstruction"
KNOTS = np.array([0,.01,.02,.05,.1,.2,.35,.5,.65,.8,.9,.95,.97,.99,.995,1.])


def increments(source: pq.ParquetFile, group: int) -> np.ndarray:
    """Select only declared quiet transitions using raw current/past inputs."""
    names = [f"i0_p{i}" for i in range(22)] + [f"i0_{k}{i}" for k in ("dp", "dv") for i in range(4)] + ["a5", "a6"]
    table = source.read_row_group(group, columns=names, use_threads=False)
    x = np.column_stack([table[n].to_numpy() for n in names]).astype(np.float64)
    assert np.isfinite(x).all()
    lo = float(np.float32(NormalDist().inv_cdf(1e-7-np.finfo(float).eps)))
    old, new = x[:-1, -2:], x[1:, -2:]
    quiet = np.all(x[1:, :30] == x[:-1, :30], axis=1)
    inside = np.all((old > lo) & (old < -lo) & (new > lo) & (new < -lo), axis=1)
    keep = quiet & inside & np.all(new >= old, axis=1)
    return ndtr(np.column_stack([old[:,0],new[:,0],old[:,1],new[:,1]])[keep])


def metrics(pair: np.ndarray) -> dict:
    """Recompute reported scale-free errors without epsilon or clipping."""
    energy = float(np.square(pair).mean())
    return {"rows": len(pair), "symmetric_relative_RMS": float(np.sqrt(np.square(pair[:,0]-pair[:,1]).mean()/energy)), "clock_RMS": np.sqrt(np.square(pair).mean(0)).tolist(), "clock_mean": pair.mean(0).tolist(), "paired_positive_fraction": float(np.all(pair > 0,axis=1).mean())}


def main() -> None:
    """Replay FIT sampling and heldout frozen maps, then independently block D3."""
    start = time.monotonic()
    panel = json.loads((DATA / "selection.json").read_text())
    record = json.loads((DATA / "inverse_clock/results.json").read_text())
    support = json.loads((DATA / "inverse_clock/fit_support.json").read_text())
    source = pq.ParquetFile(ROOT / "datasets/train.parquet")
    assert not set(panel["fit_components"]) & set(panel["test_components"])
    h = [np.array(record["solver"][f"clock{j}_heights"]) for j in (5,6)]
    reports = {}
    for split, groups in [("fit", panel["fit_train_groups"]), ("test",panel["test_train_groups"])]:
        collected = []
        for group in groups:
            u = increments(source,group)
            if split == "fit": u = u[np.linspace(0,len(u)-1,min(512,len(u)),dtype=int)]
            collected.append(u)
        u = np.concatenate(collected)
        candidate = np.column_stack([np.interp(u[:,2*i+1],KNOTS,h[i])-np.interp(u[:,2*i],KNOTS,h[i]) for i in range(2)])
        methods = {"candidate": candidate}
        for name, p in support["control_parameters"].items():
            z = u if name == "CDF" else -np.log1p(-np.minimum(u,1-1e-7))
            methods[name] = np.column_stack([z[:,1]-z[:,0],p["clock6_positive_scalar"]*(z[:,3]-z[:,2])])/p["observed_mean_scale"]
        bins = np.array(support["offset_bins"])
        ids = [np.clip(np.searchsorted(bins[i],u[:,2*i+1],side="right")-1,0,3) for i in range(2)]
        masses = np.bincount(ids[0]*4+ids[1],weights=np.square(candidate).mean(1),minlength=16)
        masses /= masses.sum()
        effective = float(1/np.square(masses).sum())
        reports[split] = {"methods": {k:metrics(v) for k,v in methods.items()}, "cell_squared_mass_fraction":masses.tolist(),"effective_cells":effective,"effective_cells_gate":effective>=4}
        original = record["solver"]["fit_analysis"] if split == "fit" else record["methods"]["candidate"]["all"]
        np.testing.assert_allclose(masses,original["cell_mass"]["squared_mass_fraction"],rtol=1e-10,atol=1e-12)
        expected = record["solver"]["fit_controls"] if split == "fit" else {k:v["all"]["metrics"] for k,v in record["methods"].items()}
        expected = {**expected,"candidate":original["metrics"]}
        for name, m in reports[split]["methods"].items():
            for k,v in m.items(): np.testing.assert_allclose(v,expected[name][k],rtol=1e-10,atol=1e-12)
    result = {"status":"pass_independent_replay", "model_clock_gate":"blocked_fit_effective_cells", "fit_is_exploratory":True,"fresh_components_reproduced":False,"refit":False,"target_reads":False,"raw_columns":"22price+8dp/dv+a5/a6","reports":reports,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"actor_source_sha256":record["source_sha256"],"wall_seconds":time.monotonic()-start,"max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with (AUDIT / "inverse_clock_independent_replay_v2.json").open("x") as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != "reports"}))


if __name__ == "__main__":
    main()
