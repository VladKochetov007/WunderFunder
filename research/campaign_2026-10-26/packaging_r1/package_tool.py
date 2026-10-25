"""Build TL3-engine ZIPs with replaced members and replay them natively on the fixed quota panel.

Subcommands:
  export  RUN EPOCH OUT_NPZ            checkpoint -> member npz (FP32 state dict, main 2 head rows)
  build   OUT_ZIP NAME=NPZ=COEF ...    TL3 base ZIP with its members.json/weights replaced
  replay  ZIP OUT_DIR                  native callback on the Oct-24 64-sequence panel (A-B-A reset)
  score   PRED_NPY [REF_NAME]          official metric on the panel; optional torch-export agreement

The panel is the exposed source-qualified F0/F2 sample frozen on 2026-10-24
(quota REPLAY_PLAN_R1.json). All-data members were fitted on it: scores are descriptive.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE_ZIP = ROOT / "submissions/team_tl3_original_20261018.zip"
BASE_ZIP_SHA256 = "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
PLAN = ROOT / "research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/REPLAY_PLAN_R1.json"
REPLAY = ROOT / "research/campaign_2026-10-18/online_features_r1/tl3_replay"
KEEP = ("solution.py", "avx2/", "v2/", "v3/")
MEMBER_KEYS = (
    "mean", "std", "inp.weight", "inp.bias",
    *(f"gru.{w}_{k}_l{i}" for i in (0, 1) for w in ("weight", "bias") for k in ("ih", "hh")),
    "head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias",
)
STAMP = (2026, 10, 26, 0, 0, 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(run: str, epoch: str, out: Path) -> None:
    import torch

    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    ck = torch.load(ROOT / f"artifacts/runs/{run}/model_ep{epoch}.pt", map_location="cpu", weights_only=False)
    state = ck["model"]
    if ck["args"].get("arch") != "gru" or int(ck["args"].get("hidden")) != 256 or int(ck["args"].get("layers")) != 2:
        raise ValueError("member exporter supports the H256x2 GRU student layout only")
    arrays = {}
    for key in MEMBER_KEYS:
        value = state[key].float().numpy()
        arrays[key] = np.ascontiguousarray(value[:2] if key.startswith("head.2.") else value, dtype=np.float32)
    extra = sorted(set(state) - set(MEMBER_KEYS))
    if extra:
        raise ValueError(f"checkpoint has parameters the native engine ignores: {extra}")
    np.savez(out, **arrays)
    print(json.dumps({"run": run, "epoch": epoch, "npz": str(out), "sha256": sha256_file(out)}))


def build(out: Path, specs: list[str]) -> None:
    if sha256_file(BASE_ZIP) != BASE_ZIP_SHA256:
        raise ValueError("TL3 base ZIP identity changed")
    members, payload = [], {}
    for index, spec in enumerate(specs):
        name, npz, coef = spec.split("=")
        source = Path(npz)
        if source.name in ("weights_tl5.npz", "weights_tl6.npz") and source.parent == Path("base"):
            entry = source.name
        else:
            entry = f"weights_m{index}.npz"
            payload[entry] = source.read_bytes()
        members.append({"weights": entry, "name": name, "coef": float(coef)})
    if abs(sum(m["coef"] for m in members) - 1.0) > 1e-9:
        raise ValueError("member coefficients must sum to one")
    with zipfile.ZipFile(BASE_ZIP) as base, zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as dest:
        used = {m["weights"] for m in members}
        for info in sorted(base.infolist(), key=lambda item: item.filename):
            keep = info.filename.startswith(KEEP) or info.filename in used and info.filename.startswith("weights_tl")
            if keep and not info.is_dir():
                dest.writestr(zipfile.ZipInfo(info.filename, STAMP), base.read(info.filename), zipfile.ZIP_DEFLATED)
        for entry, data in sorted(payload.items()):
            dest.writestr(zipfile.ZipInfo(entry, STAMP), data, zipfile.ZIP_DEFLATED)
        dest.writestr(zipfile.ZipInfo("members.json", STAMP), json.dumps(members, indent=2) + "\n", zipfile.ZIP_DEFLATED)
    size = out.stat().st_size
    if size > 20_000_000:
        raise ValueError(f"ZIP exceeds 20 MB: {size}")
    print(json.dumps({"zip": str(out), "bytes": size, "sha256": sha256_file(out), "members": members}))


def replay(archive: Path, out: Path) -> None:
    import pyarrow.parquet as pq

    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(REPLAY)]
    from check_submission_abi import inspect_artifact
    from native_runtime import extract_safe, load_callback, run_sequence

    from utils import FEATURE_COLUMNS, DataPoint, validate_sequence

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("TL3 native extension requires Python 3.11")
    plan = json.loads(PLAN.read_text())
    abi = inspect_artifact(archive)
    if not abi["passed"]:
        raise ValueError("ZIP failed native ABI inspection")
    out.mkdir(parents=True, exist_ok=False)
    preds = np.zeros((len(plan["members"]), 20_000, 2), np.float32)
    parquet = pq.ParquetFile(ROOT / plan["raw_source"]["path"])
    started, wall = time.monotonic(), 0
    with tempfile.TemporaryDirectory(prefix="pkg-", dir=out) as tmp:
        with zipfile.ZipFile(archive) as stream:
            if stream.testzip() is not None:
                raise ValueError("ZIP CRC failure")
            extract_safe(stream, Path(tmp))
        _, model, native = load_callback(Path(tmp))
        first = None
        for index, member in enumerate(plan["members"]):
            table = parquet.read_row_group(
                member["row_group"], columns=["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS],
                use_threads=False,
            )
            sequence, need = validate_sequence(table)
            if sequence != member["seq_ix"]:
                raise ValueError("sequence identity changed")
            feats = np.column_stack([table[n].to_numpy() for n in FEATURE_COLUMNS]).astype(np.float32)
            values, _, _, elapsed = run_sequence(model, DataPoint, sequence, feats, need)
            preds[index], wall = values, wall + elapsed
            if index == 0:
                first = (sequence, feats.copy(), need.copy())
            if index == 1:
                again = run_sequence(model, DataPoint, *first)[0]
                if not np.array_equal(preds[0].view(np.uint32), again.view(np.uint32)):
                    raise ValueError("A-B-A bitwise reset failed")
        native_name = str(native.relative_to(Path(tmp)))
    if not np.isfinite(preds).all():
        raise ValueError("non-finite predictions")
    np.save(out / "predictions.npy", preds)
    receipt = {
        "zip": str(archive), "zip_sha256": sha256_file(archive), "zip_bytes": archive.stat().st_size,
        "abi_passed": True, "native_library": native_name, "reset_aba_bitwise": True,
        "predictions_sha256": sha256_file(out / "predictions.npy"), "sequences": len(plan["members"]),
        "callback_wall_seconds": wall / 1e9, "wall_seconds": time.monotonic() - started,
        "python": sys.version.split()[0],
    }
    (out / "REPLAY_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


def score(pred_path: Path, ref: str | None) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from wnn.gmetric import global_wp

    plan = json.loads(PLAN.read_text())
    preds = np.load(pred_path)
    with np.load(ROOT / "cache/valid_targets.npz") as z:
        pos = {int(g): i for i, g in enumerate(z["row_group"])}
        idx = [pos[m["row_group"]] for m in plan["members"]]
        y, mask = z["y"][idx].astype(np.float32), z["mask"][idx]
    folds = np.array([m["fold"] for m in plan["members"]])
    report = {}
    for fold in (0, 2):
        sel = folds == fold
        report[f"F{fold}"] = float(global_wp(y[sel][mask[sel]], preds[sel][mask[sel]], np.ones(int(mask[sel].sum()), bool))["wp"])
    if ref:
        side = json.loads((ROOT / f"cache/pred_{ref}_valid.json").read_text())
        arr = np.load(ROOT / f"cache/pred_{ref}_valid.npy", mmap_mode="r")
        rpos = {int(g): i for i, g in enumerate(side["row_groups"])}
        r = np.stack([np.asarray(arr[rpos[m["row_group"]]], np.float32) for m in plan["members"]])
        diff = (preds - r)[mask]
        report["torch_ref"] = ref
        report["max_abs_diff"] = float(np.abs(diff).max())
        report["rms_diff_over_rms_pred"] = float(np.sqrt((diff**2).mean() / (r[mask] ** 2).mean()))
        for fold in (0, 2):
            sel = folds == fold
            report[f"torch_F{fold}"] = float(global_wp(y[sel][mask[sel]], r[sel][mask[sel]], np.ones(int(mask[sel].sum()), bool))["wp"])
    print(json.dumps(report))


if __name__ == "__main__":
    command, *rest = sys.argv[1:]
    if command == "export":
        export(rest[0], rest[1], Path(rest[2]))
    elif command == "build":
        build(Path(rest[0]), rest[1:])
    elif command == "replay":
        replay(Path(rest[0]), Path(rest[1]))
    elif command == "score":
        score(Path(rest[0]), rest[1] if len(rest) > 1 else None)
    else:
        raise SystemExit(__doc__)
