"""Predict valid fold0 (467 sequences) for a run trained by train_rnn_plew19.py; full-size cache with NaN elsewhere."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(os.environ["WUNDER_RESEARCH_ROOT"])
HERE = Path(__file__).resolve().parent
os.environ["WUNDER_RESEARCH_ROOT"] = str(R)
sys.path[:0] = [str(HERE), str(R / "src")]
import train_rnn_plew19 as T  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def restore_checkpoint_settings(
    T: object, args: dict[str, object], ck: dict[str, object], root: Path, device: torch.device
) -> None:
    """Restore inference preprocessing and output settings from the checkpoint.

    Args:
        T: Frozen plew19 trainer module used by this exporter.
        args: Serialized training arguments from the checkpoint.
        ck: Checkpoint payload containing weights and optional readout tables.
        root: Creative repository root that holds referenced assets.
        device: Inference device for referenced models and readout tables.
    """
    T.LOOKAHEAD = tuple(int(v) for v in str(args.get("lookahead", "")).split(",") if v)
    T.STATE_RESET = int(args.get("state_reset", 0))
    T.SENTINEL_MODE = str(args.get("sentinel_mode", "none"))
    T.A_MODE = str(args.get("a_mode", "none"))
    T.DOMAIN_FLAG = bool(args.get("domain_flag", False))
    zero_groups = str(args.get("zero_groups", ""))
    T.ZERO_COLS = torch.tensor(T.zero_group_cols(zero_groups), dtype=torch.long) if zero_groups else None
    T.RAW_ONLY = bool(args.get("raw_only", False))
    T.HEURISTICS = bool(args.get("heuristics", False))
    T.QUANT = bool(args.get("quant_feats", False))
    T.QUANT_I1 = bool(args.get("quant_i1", False))
    T.ONLINE_NORM = bool(args.get("online_norm", False))
    T.INPUT_PROPENSITY = None
    prop_input = str(args.get("prop_input", ""))
    if prop_input:
        T.INPUT_PROPENSITY = T.load_propensity(root / "artifacts" / "runs" / prop_input / "model.pt", device)
    readout = ck.get("readout")
    T.READOUT = None if readout is None else tuple(value.to(device) for value in readout)
    T.READOUT_TILTED = bool(ck.get("readout_tilted", False))


def prediction_positions(
    row_groups: np.ndarray, split: str, fold_table: np.ndarray, full: bool
) -> np.ndarray:
    """Return canonical split positions for full or fold-0 research inference.

    Args:
        row_groups: Canonical row-group order for the selected split.
        split: Dataset split name.
        fold_table: Structured chain-fold table.
        full: Whether to predict every row group in the split.

    Returns:
        Positions into ``row_groups`` in its original order.
    """
    if full:
        return np.arange(len(row_groups), dtype=np.int64)
    if split != "valid":
        raise ValueError("non-valid splits require explicit full prediction")
    fold_split = fold_table["split"].astype(str)
    fold0 = set(fold_table["row_group"][(fold_split == "valid") & (fold_table["fold"] == 0)].tolist())
    return np.flatnonzero([int(group) in fold0 for group in row_groups])


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_settings(T: object, args: dict[str, object]) -> dict[str, object]:
    """Return serialized feature preprocessing settings restored from the checkpoint."""
    return {
        "lookahead": list(T.LOOKAHEAD),
        "state_reset": T.STATE_RESET,
        "sentinel_mode": T.SENTINEL_MODE,
        "a_mode": T.A_MODE,
        "domain_flag": T.DOMAIN_FLAG,
        "zero_groups": str(args.get("zero_groups", "")),
        "raw_only": T.RAW_ONLY,
        "heuristics": T.HEURISTICS,
        "quant_feats": T.QUANT,
        "quant_i1": T.QUANT_I1,
        "online_norm": T.ONLINE_NORM,
        "prop_input": str(args.get("prop_input", "")),
    }


def source_identity(path: Path, root: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": resolved.relative_to(root.resolve()).as_posix(), "sha256": sha256_file(resolved)}


def validate_export_causality(lookahead: tuple[int, ...], privileged: bool) -> bool:
    """Require an explicit opt-in before exporting a future-looking teacher."""
    if lookahead and not privileged:
        raise ValueError("checkpoint uses future lookahead; pass explicit privileged export flag")
    if privileged and not lookahead:
        raise ValueError("privileged export flag requires checkpoint lookahead features")
    return not bool(lookahead)


def validate_prediction_batch(predictions: np.ndarray) -> np.ndarray:
    """Reject malformed or nonfinite outputs before placing them in a cache."""
    array = np.asarray(predictions)
    if array.ndim != 3 or array.shape[1:] != (20000, 2):
        raise ValueError(f"prediction batch has malformed shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("prediction batch contains nonfinite values")
    return array


def prediction_metadata(
    out: Path,
    split: str,
    row_groups: np.ndarray,
    positions: np.ndarray,
    full: bool,
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    preprocessing: dict[str, object],
    source_sha256: dict[str, object],
    privileged: bool,
) -> dict[str, object]:
    """Describe cache coverage so training consumers can reject fold-only arrays.

    Args:
        out: Destination prediction array path.
        split: Dataset split name.
        row_groups: Canonical row-group order for the selected split.
        positions: Predicted positions in that order.
        full: Whether the caller requested full-split prediction.

    Returns:
        Sidecar values binding the array to its split and exact membership.
    """
    return {
        "schema": "wunder.full-prediction-cache.v1",
        "split": split,
        "prediction_name": out.name,
        "complete": bool(full and len(positions) == len(row_groups)),
        "row_groups": [int(group) for group in row_groups[positions]],
        "shape": [len(row_groups), 20000, 2],
        "dtype": "float32",
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing": preprocessing,
        "preprocessing_sha256": hashlib.sha256(
            json.dumps(preprocessing, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "source_sha256": source_sha256,
        "lookahead": preprocessing["lookahead"],
        "causal": not bool(preprocessing["lookahead"]),
        "privileged": privileged,
    }


def main(run, split="valid", full=False, privileged=False):
    out = R / f"cache/pred_{os.environ.get('OUT_TAG', run)}_{split}.npy"
    assert not out.exists()
    sidecar = out.with_suffix(".json")
    assert not sidecar.exists()
    checkpoint_path = R / f"artifacts/runs/{run}/model_ep{os.environ.get('EPOCH', '1')}.pt"
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    restore_checkpoint_settings(T, args, ck, R, torch.device("cuda"))
    causal = validate_export_causality(T.LOOKAHEAD, privileged)
    preprocessing = preprocessing_settings(T, args)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    source_sha256 = {
        "predictor": source_identity(Path(__file__), R),
        "trainer": source_identity(Path(T.__file__), R),
    }
    n_out = ck["model"]["head.2.weight"].shape[0]
    model = T.make_model(args, ck["n_in"], n_out)
    model.load_state_dict(ck["model"])
    model = model.cuda().eval()
    assert T.FAMILY_TEACHERS == ()
    rg = np.load(R / f"cache/{split}_targets.npz")["row_group"]
    cf = np.load(R / "cache/chain_folds.npz")
    if len(np.unique(rg)) != len(rg):
        raise ValueError(f"{split} target row groups contain duplicates")
    positions = prediction_positions(rg, split, cf, full)
    loader = DataLoader(T.Sequences(split, rg[positions]), batch_size=16, num_workers=2, prefetch_factor=1)
    tmp = out.with_suffix(".partial.npy")
    dest = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(rg), 20000, 2))
    if not full:
        dest[:] = np.nan
    off = 0
    for x, _, _ in loader:
        pred = validate_prediction_batch(T.predict(model, x, 4000, torch.device("cuda")))
        dest[positions[off:off + len(x)]] = pred
        off += len(x)
    assert off == len(positions)
    dest.flush()
    del dest
    tmp.replace(out)
    metadata = prediction_metadata(
        out,
        split,
        rg,
        positions,
        full,
        checkpoint_path=checkpoint_path.relative_to(R),
        checkpoint_sha256=checkpoint_sha256,
        preprocessing=preprocessing,
        source_sha256=source_sha256,
        privileged=privileged,
    )
    metadata["causal"] = causal
    sidecar_tmp = sidecar.with_suffix(".partial.json")
    sidecar_tmp.write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n")
    sidecar_tmp.replace(sidecar)
    print("wrote", out, off, "sequences")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: predict_any_env.py RUN [SPLIT] [full] [privileged]")
    tokens = sys.argv[2:]
    split = "valid"
    if tokens and tokens[0] not in {"full", "privileged"}:
        split = tokens.pop(0)
    if any(token not in {"full", "privileged"} for token in tokens):
        raise SystemExit("usage: predict_any_env.py RUN [SPLIT] [full] [privileged]")
    if len(tokens) != len(set(tokens)):
        raise SystemExit("full and privileged may each appear at most once")
    main(sys.argv[1], split, full="full" in tokens, privileged="privileged" in tokens)
