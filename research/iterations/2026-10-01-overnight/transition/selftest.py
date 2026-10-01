"""Synthetic author checks only; no real data, predictions or outcome queries."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from features import WIDTH, batch_features, StreamingFeatures, event_probabilities, detect


def fixture():
    rng = np.random.default_rng(42717)
    pi = rng.uniform(.1, 1, 16); pi /= pi.sum()
    R = rng.uniform(.1, 1, (16,16)); R /= R.sum(1, keepdims=True)
    nuisance = {"centers": [-.4,.15,.2,.4], "pi": pi.tolist(), "R": R.tolist()}
    rows = rng.normal(size=(233,4)).astype(np.float32)
    rows[::4] = 0
    rows[8:13] = rows[7]
    rows[26, 0] = np.nan
    rows[47, 2] = np.inf
    rows[52, 3] = 5.19
    rows[58] = np.nextafter(np.float32(5.19), np.float32(0))
    rows[59] = [0,-0.0,0,0]
    rows[60] = [-0.0,0,0,0]
    rows[61] = [0,0,0,np.float32(1e-6)]
    rows[62] = [0,0,0,np.nextafter(np.float32(1e-6), np.float32(1))]
    rows[63] = nuisance["centers"]
    rows[64] = np.nextafter(rows[63], np.float32(5))
    return rows, nuisance


def numpy_checks():
    rows, nuisance = fixture()
    records = []
    for arm in ("F", "T"):
        batch = batch_features(rows, nuisance, arm)
        stream = StreamingFeatures(nuisance, arm)
        literal = np.array([stream.update(row) for row in rows])
        np.testing.assert_array_equal(literal, batch)
        for cut in (0,1,2,59,60,61,62,63,64,80,232,233):
            np.testing.assert_array_equal(batch_features(rows[:cut], nuisance, arm), batch[:cut])
        alternate = rows.copy()
        alternate[81:] = -alternate[81:]*3
        np.testing.assert_array_equal(batch_features(alternate, nuisance, arm)[:81], batch[:81])
        stream.reset()
        chunks = [rows[:13],rows[13:82],rows[82:]]
        chunked = np.concatenate([np.array([stream.update(row) for row in chunk]) for chunk in chunks])
        np.testing.assert_array_equal(chunked, batch)
        assert np.isfinite(batch).all() and batch.shape == (233,68)
        assert (batch[:, :51] >= 0).all() and (batch[:, :51] <= 1).all()
        assert np.abs(batch[:,51:67]).max() <= 1
        assert (batch[:,67] >= 0).all() and (batch[:,67] <= 1).all()
        records.append({"arm": arm, "rows":len(rows), "events":len(detect(rows)), "batch_stream_bitexact":True})
    f = batch_features(rows, nuisance, "F")
    t = batch_features(rows, nuisance, "T")
    np.testing.assert_array_equal(f[:,:35], t[:,:35])
    # Distinct raw vectors mapping to same category must still both emit.
    toy = np.array([[0,0,0,0],[1,2,3,4],[1.1,2,3,4]],dtype=np.float32)
    np.testing.assert_array_equal(detect(toy), [1,2])
    # ±0 is numerically unchanged; invalid current/previous suppress both edges.
    np.testing.assert_array_equal(detect(np.array([[0.,-0.,0.,0.],[-0.,0.,0.,0.]],np.float32)), [])
    x = np.array([[1,2,3,4],[np.nan,2,3,4],[2,1,4,3],[3,1,4,2]],np.float32)
    np.testing.assert_array_equal(detect(x), [3])
    long = np.tile(toy[1:], (10100,1))
    extra = batch_features(long,nuisance,"T")
    assert int(extra[:,34].sum()) == len(long)-1  # no hidden4096 cap
    assert extra[-1,32] == 1.0
    started=time.monotonic()
    for _ in range(10):
        batch_features(long[:20000],nuisance,"T")
    return {"arms":records,"causal_reset_invalid_threshold_samecategory":True,
            "synthetic_20000_rows_ms":100*(time.monotonic()-started),"no_event_cap":True}


def torch_checks():
    root=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(root/"src")); sys.path.insert(0,str(root/"scripts"))
    import torch
    from wnn.nn import RNNModel
    from model import FeatureHeadModel
    from train_adapter import skip_evaluation_preserving_rng
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(1)
    torch.manual_seed(77)
    baseline=RNNModel(211,hidden=12,layers=2,n_out=23)
    baseline_rng=torch.get_rng_state().clone()
    torch.manual_seed(77)
    candidate=FeatureHeadModel(211,hidden=12,layers=2,n_out=23)
    assert torch.equal(torch.get_rng_state(),baseline_rng)
    candidate.load_state_dict(baseline.state_dict())
    x=torch.randn(2,41,211)
    extra=torch.randn(2,41,68)
    with torch.no_grad():
        want,hw=baseline(x)
        got,hg=candidate(torch.cat([x,extra],-1))
    assert torch.equal(want,got) and torch.equal(hw,hg)
    # Once feature branch is active, recurrent state still matches exactly.
    candidate.head_extra.weight.data.normal_(0,.01)
    with torch.no_grad():
        allrows,state=candidate(torch.cat([x,extra],-1))
        h=None; chunked=[]
        for a,b in ((0,1),(1,13),(13,41)):
            y,h=candidate(torch.cat([x[:,a:b],extra[:,a:b]],-1),h)
            chunked.append(y)
        torch.testing.assert_close(torch.cat(chunked,1),allrows,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(h,state,rtol=1e-6,atol=1e-7)
        assert torch.equal(state,hw)
    for name,param in candidate.named_parameters():
        if name.startswith(("inp","gru")):
            param.requires_grad_(False)
    out,_=candidate(torch.cat([x,extra],-1)); out.square().mean().backward()
    assert all(p.grad is None for n,p in candidate.named_parameters() if n.startswith(("inp","gru")))
    assert candidate.head_extra.weight.grad.abs().sum()>0 and candidate.head[0].weight.grad.abs().sum()>0
    loader=DataLoader(TensorDataset(torch.zeros(17)),batch_size=4,shuffle=False,num_workers=0)
    torch.manual_seed(361)
    for _ in loader: pass
    expected=torch.get_rng_state().clone()
    torch.manual_seed(361)
    skip_evaluation_preserving_rng(None,loader,None,None)
    assert torch.equal(torch.get_rng_state(),expected)
    return {"initial_forecasts_bitexact":True,"extra_construction_rng_bitexact":True,
            "chunk_state_parity":True,"frozen_backbone_no_gradient":True,"eval_skip_rng_parity":True}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--torch",action="store_true"); args=parser.parse_args()
    result={"numpy":numpy_checks(),"empirical_data_read":False,"forecasting_fits":0,"WP_queries":0}
    if args.torch: result["torch"]=torch_checks()
    result["code_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__ == "__main__": main()
