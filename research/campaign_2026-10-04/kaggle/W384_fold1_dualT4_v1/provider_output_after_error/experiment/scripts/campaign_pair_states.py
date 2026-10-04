"""Export paired frozen GRU states for a preregistered outer-fold head screen.

Only validation fitting rows with an official mask are used to train a head.
All evaluation rows are exported so exact and softened-mask scores use identical
predictions. The two checkpoints must share one clean outer-fold manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_rnn  # noqa: E402

from wnn.data import WARMUP  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402


def load_member(name: str, manifest_sha: str, device: torch.device) -> tuple[torch.nn.Module, dict, str]:
    """Load one fixed-epoch campaign member and validate its fold provenance."""
    path = ROOT / "artifacts" / "runs" / name / "model_ep0.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if args.get("split_manifest_sha256") != manifest_sha:
        raise ValueError(f"{name}: checkpoint belongs to another outer fold")
    if (
        args.get("checkpoint_policy") != "fixed"
        or checkpoint.get("epoch") != 0
        or args.get("checkpoint_epoch") != 0
    ):
        raise ValueError(f"{name}: checkpoint epoch was not fixed before evaluation")
    if args.get("no_eval_during_fit") is not True:
        raise ValueError(f"{name}: evaluation fold was available during backbone fitting")
    if args.get("arch") != "gru" or args.get("cell") != "gru" or args.get("enc") != "linear":
        raise ValueError(f"{name}: paired-state extractor only supports the incumbent GRU")
    model = train_rnn.make_model(args, checkpoint["n_in"], checkpoint["model"]["head.2.weight"].shape[0])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), args, file_sha256(path)


def check_shared_features(first: dict, second: dict) -> None:
    """Ensure both members see exactly the same causal input construction."""
    flags = ("raw_only", "online_norm", "heuristics", "prop_input", "enc")
    if any(first.get(key) != second.get(key) for key in flags):
        raise ValueError("paired members use different feature construction")
    if first.get("prop_input"):
        raise ValueError("propensity input requires a separately frozen input model")
    train_rnn.RAW_ONLY = bool(first.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(first.get("online_norm"))
    train_rnn.HEURISTICS = bool(first.get("heuristics"))


def create_arrays(prefix: Path, n_rows: int, hidden: int) -> dict[str, np.memmap]:
    """Preallocate bounded, resumable-on-failure memmaps for one fixed run."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    definitions = {
        "states": (np.float16, (n_rows, 2, hidden)),
        "predictions": (np.float32, (n_rows, 2, 2)),
        "targets": (np.float32, (n_rows, 2)),
        "mask": (np.bool_, (n_rows,)),
        "row_group": (np.int32, (n_rows,)),
    }
    return {
        key: np.lib.format.open_memmap(
            prefix.with_name(f"{prefix.name}_{key}.npy.partial"), mode="w+", dtype=dtype, shape=shape
        )
        for key, (dtype, shape) in definitions.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--member-a", required=True)
    parser.add_argument("--member-b", required=True)
    parser.add_argument("--mode", choices=("fit", "eval"), required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=4000)
    args = parser.parse_args()
    manifest = load_campaign_split(
        ROOT / "research" / "campaign_2026-09-29" / "artifacts" / "splits" / f"fold{args.fold}.json", ROOT
    )
    groups = np.asarray(manifest.valid_groups if args.mode == "fit" else manifest.evaluation_groups)
    if args.mode == "fit":
        with np.load(ROOT / "cache" / "valid_targets.npz") as cache:
            indices = {int(group): i for i, group in enumerate(cache["row_group"])}
            masks = cache["mask"]
            n_rows = sum(int(masks[indices[int(group)], WARMUP:].sum()) for group in groups)
    else:
        n_rows = len(groups) * 20_000

    configure_torch()
    device = torch.device("cuda")
    member_a, args_a, sha_a = load_member(args.member_a, manifest.sha256, device)
    member_b, args_b, sha_b = load_member(args.member_b, manifest.sha256, device)
    check_shared_features(args_a, args_b)
    hidden = int(args_a["hidden"])
    if hidden != int(args_b["hidden"]):
        raise ValueError("paired members have different hidden widths")
    prefix = ROOT / "research" / "campaign_2026-09-29" / "artifacts" / "pair_states" / args.name
    if any(prefix.parent.glob(f"{prefix.name}_*.npy")) or any(prefix.parent.glob(f"{prefix.name}_*.partial")):
        raise FileExistsError(f"state artifact already exists for {args.name}")
    arrays = create_arrays(prefix, n_rows, hidden)
    loader = DataLoader(
        train_rnn.Sequences("valid", groups),
        batch_size=args.batch,
        num_workers=2,
        prefetch_factor=1,
        pin_memory=True,
    )
    position = 0
    group_position = 0
    parity_maxabs = [0.0, 0.0]
    for x, y, mask in loader:
        batch_groups = groups[group_position : group_position + len(x)]
        group_position += len(x)
        inputs = train_rnn.build_inputs(x.to(device, non_blocking=True))
        states: list[torch.Tensor | None] = [None, None]
        parity_chunks = [[], []] if group_position == len(x) else None
        for step in range(0, 20_000, args.chunk):
            stop = min(step + args.chunk, 20_000)
            selected = mask[:, step:stop].numpy().copy()
            selected[:, : max(0, WARMUP - step)] = False
            keep = selected if args.mode == "fit" else np.ones_like(selected)
            rows = int(keep.sum())
            end = position + rows
            if args.mode == "fit":
                write_index = slice(position, end)
            else:
                sequence_offsets = np.arange(group_position - len(x), group_position, dtype=np.int64)
                write_index = (
                    sequence_offsets[:, None] * 20_000 + np.arange(step, stop, dtype=np.int64)[None, :]
                ).reshape(-1)
            arrays["targets"][write_index] = y[:, step:stop].numpy()[keep]
            arrays["mask"][write_index] = selected[keep]
            arrays["row_group"][write_index] = np.broadcast_to(
                batch_groups[:, None], keep.shape
            )[keep]
            for member_index, member in enumerate((member_a, member_b)):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    z = member.encode((inputs[:, step:stop] - member.mean) / member.std)
                    o, states[member_index] = member.gru(z, states[member_index])
                    prediction = member.head(torch.cat((o, z), dim=-1))[..., :2].float()
                if not torch.isfinite(prediction).all():
                    raise ValueError(f"member {member_index} emitted a nonfinite prediction")
                if parity_chunks is not None:
                    parity_chunks[member_index].append(prediction.cpu().numpy())
                arrays["states"][write_index, member_index] = o.cpu().numpy()[keep]
                arrays["predictions"][write_index, member_index] = prediction.cpu().numpy()[keep]
            position = end
        if parity_chunks is not None:
            for member_index, member in enumerate((member_a, member_b)):
                reference = train_rnn.predict(member, x, args.chunk, device)
                actual = np.concatenate(parity_chunks[member_index], axis=1)
                if not np.isfinite(actual).all() or not np.isfinite(reference).all():
                    raise ValueError("backbone emitted nonfinite predictions")
                parity_maxabs[member_index] = float(np.max(np.abs(actual - reference)))
                if parity_maxabs[member_index] > 1e-5:
                    raise ValueError(f"member {member_index} manual extraction differs from prediction")
        print(json.dumps({"sequences_done": group_position, "rows_written": position}), flush=True)
    if group_position != len(groups) or position != n_rows:
        raise RuntimeError(f"incomplete export: {group_position}/{len(groups)} groups, {position}/{n_rows} rows")
    for key, array in arrays.items():
        array.flush()
        del array
        partial = prefix.with_name(f"{prefix.name}_{key}.npy.partial")
        partial.replace(prefix.with_name(f"{prefix.name}_{key}.npy"))
    array_sha256 = {
        key: file_sha256(prefix.with_name(f"{prefix.name}_{key}.npy")) for key in arrays
    }
    metadata = {
        "name": args.name,
        "mode": args.mode,
        "fold": args.fold,
        "manifest_sha256": manifest.sha256,
        "members": [args.member_a, args.member_b],
        "member_checkpoint_sha256": [sha_a, sha_b],
        "groups": groups.tolist(),
        "n_rows": n_rows,
        "hidden": hidden,
        "batch": args.batch,
        "chunk": args.chunk,
        "float16_states": True,
        "predict_parity_maxabs": parity_maxabs,
        "arrays_sha256": array_sha256,
        "selection": "need_prediction AND is_scored" if args.mode == "fit" else "all rows",
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
