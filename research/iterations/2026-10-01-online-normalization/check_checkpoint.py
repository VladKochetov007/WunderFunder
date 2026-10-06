"""CPU-only fixed checkpoint validator; no model construction, forward or data reads."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SHAPES = {
    "mean": (323,), "std": (323,), "inp.weight": (256, 323), "inp.bias": (256,),
    "gru.weight_ih_l0": (768, 256), "gru.weight_hh_l0": (768, 256),
    "gru.bias_ih_l0": (768,), "gru.bias_hh_l0": (768,),
    "head.0.weight": (256, 512), "head.0.bias": (256,),
    "head.2.weight": (23, 256), "head.2.bias": (23,),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job")
    ap.add_argument("--commands", type=Path, required=True)
    ap.add_argument("--parent", type=Path, help="Optional explicit own-parent endpoint; otherwise inferred from fixed plan")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists() and not a.output.is_symlink(), a.output
    assert a.output.resolve().is_relative_to(HERE), "write only the new experiment's receipts"
    plan = json.loads(a.commands.read_text())
    jobs = [j for j in plan["jobs"] if j["name"] == a.job]
    assert len(jobs) == 1
    job = jobs[0]
    for rel, expected in plan["source_sha256"].items():
        assert sha(ROOT / rel) == expected, rel
    assert len(job["resolved_args"]) == 103 and job["expected_n_in"] == 323
    args = job["resolved_args"]
    assert args["online_norm"] and args["hidden"] == 256 and args["layers"] == 1
    assert args["name"] == a.job and args["valid_fold"] == job["fold"]

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)

    def same_bytes(x, y):
        return x.dtype == y.dtype and x.shape == y.shape and torch.equal(
            x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))

    def load(path, expected_job, epoch):
        before = sha(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        expected_keys = {"model", "args", "n_in", "readout", "readout_tilted"}
        if epoch is not None:
            expected_keys.add("epoch")
        assert set(ck) == expected_keys, sorted(ck)
        assert ck["args"] == expected_job["resolved_args"] and ck["n_in"] == 323
        assert epoch is None or ck["epoch"] == epoch
        assert ck["readout"] is None and ck["readout_tilted"] is False
        state = ck["model"]
        assert set(state) == set(SHAPES)
        for k, shape in SHAPES.items():
            v = state[k]
            assert tuple(v.shape) == shape and v.dtype == torch.float32 and v.device.type == "cpu", k
            assert bool(torch.isfinite(v).all()), k
        assert bool((state["std"] > 0).all())
        assert sha(path) == before
        return state, {"path": str(path.relative_to(ROOT)), "sha256": before,
                       "bytes": path.stat().st_size, "epoch": epoch, "n_in": 323}

    endpoint = ROOT / job["endpoint"]
    epoch = 3 if job["stage"] == "parent" else 1
    assert args["epochs"] == epoch + 1
    state, record = load(endpoint, job, epoch)
    alias, alias_record = load(endpoint.parent / "model.pt", job, None)
    assert all(same_bytes(state[k], alias[k]) for k in state)
    history_path = endpoint.parent / "history.json"
    history = json.loads(history_path.read_text())
    assert len(history) == epoch + 1
    assert all(h["epoch"] == i and h["evaluation_skipped"] is True and
               h["metric_placeholder_not_a_readout"] is True for i, h in enumerate(history))

    frozen, changed, parent_record = [], [], None
    if job["stage"] == "parent":
        assert a.parent is None and args["eval_ema"] and args["ema_teacher"] == 1.0
        assert not args["init"] and not args["freeze"]
    else:
        assert job["stage"] == "H2" and args["freeze"] == "inp,gru"
        assert not args["eval_ema"] and not args["ema_teacher"] and not args["distill"] and not args["since_weight"]
        parent_name, separator, parent_epoch = args["init"].partition("@ep")
        assert separator and parent_epoch == "3"
        parent_job = next(j for j in plan["jobs"] if j["name"] == parent_name)
        assert parent_job["stage"] == "parent" and parent_job["fold"] == job["fold"]
        expected_parent = ROOT / parent_job["endpoint"]
        if a.parent is not None:
            given = a.parent if a.parent.is_absolute() else ROOT / a.parent
            assert given.resolve() == expected_parent.resolve()
        parent, parent_record = load(expected_parent, parent_job, 3)
        frozen = [k for k in state if k in ("mean", "std") or k.startswith(("inp.", "gru."))]
        assert all(same_bytes(state[k], parent[k]) for k in frozen)
        changed = [k for k in state if k.startswith("head.") and not same_bytes(state[k], parent[k])]
        assert changed, "head adaptation left every head tensor unchanged"

    result = {"status": "PASS_FIXED_ENDPOINT_FINITE_AND_LINEAGE", "utc": datetime.now(timezone.utc).isoformat(),
              "job": a.job, "stage": job["stage"], "fold": job["fold"],
              "commands_sha256": sha(a.commands), "validator_sha256": sha(Path(__file__)),
              "checkpoint": record, "model_pt": alias_record, "model_pt_state_byte_equals_endpoint": True,
              "history_sha256": sha(history_path), "all103_args_match": True,
              "state_shapes": {k: list(v) for k, v in SHAPES.items()}, "all_state_finite_float32": True,
              "parent": parent_record, "byte_identical_frozen_tensors": frozen, "changed_head_tensors": changed,
              "EMA_identity": "Parent checkpoint args eval_ema=true and bound frozen trainer saves ema_model.module; EMA trajectory is not reconstructed.",
              "data_or_targets_read": False, "model_forward_calls": 0, "metrics": 0}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"status": result["status"], "output": str(a.output), "sha256": sha(a.output), "checkpoint": record}))


if __name__ == "__main__":
    main()
