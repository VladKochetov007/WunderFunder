"""One fixed fit-only continuation pass over a causal parent GRU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from codebook import fit_codebooks
from e2e_model import JointTargetContinuation
from fit import ARMS, ROOT, file_sha256, tick_gpu_duty, verify_fit_prefix
from model import JointTargetHead, arm_loss, require_finite_training_state
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-prefix", type=Path, required=True)
    parser.add_argument("--member-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--batch-sequences", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--max-sequences", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    if args.batch_sequences != 1 or args.chunk < 1 or args.lr <= 0 or args.max_sequences < 1:
        parser.error("fixed continuation uses one sequence per batch and a positive chunk/lr")
    duty = None
    if args.device == "cuda":
        sys.path.insert(0, str(ROOT / "src"))
        from wnn.resources import configure_torch

        duty = configure_torch()
    else:
        torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for the end-to-end continuation")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    prefix = args.fit_prefix.resolve()
    metadata, arrays = verify_fit_prefix(prefix)

    member_name = metadata["members"][args.member_index]
    checkpoint_path = ROOT / "artifacts/runs" / member_name / "model_ep0.pt"
    checkpoint_sha = file_sha256(checkpoint_path)
    expected_sha = metadata["member_checkpoint_sha256"][args.member_index]
    if checkpoint_sha != expected_sha:
        raise ValueError("parent checkpoint differs from the checkpoint that produced fit states")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parent_args = checkpoint["args"]
    if parent_args.get("split_manifest_sha256") != metadata["manifest_sha256"]:
        raise ValueError("parent checkpoint belongs to another split")
    if parent_args.get("arch", "gru") != "gru" or parent_args.get("cell", "gru") != "gru":
        raise ValueError("E2E continuation currently supports the standard causal GRU only")
    if parent_args.get("enc", "linear") != "linear" or parent_args.get("prop_input"):
        raise ValueError("E2E continuation requires linear input encoding and no propensity input")
    if parent_args.get("checkpoint_policy") != "fixed" or parent_args.get("checkpoint_epoch") != 0:
        raise ValueError("parent must be a fixed epoch-0 checkpoint")
    if parent_args.get("no_eval_during_fit") is not True:
        raise ValueError("parent was not fit with evaluation hidden")
    if parent_args.get("sentinel_indicators") not in (None, "none"):
        raise ValueError("custom sentinel projection is outside this first continuation")

    sys.path.insert(0, str(ROOT / "scripts"))
    import train_rnn

    train_rnn.RAW_ONLY = bool(parent_args.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(parent_args.get("online_norm"))
    train_rnn.HEURISTICS = bool(parent_args.get("heuristics"))
    n_in = int(checkpoint["model"]["inp.weight"].shape[1])
    source_n_out = int(checkpoint["model"]["head.2.weight"].shape[0])
    if source_n_out < 2:
        raise ValueError("parent checkpoint does not contain both target readouts")
    n_out = 2
    model_args = dict(parent_args)
    backbone_state = dict(checkpoint["model"])
    backbone_state["head.2.weight"] = backbone_state["head.2.weight"][:2].contiguous()
    backbone_state["head.2.bias"] = backbone_state["head.2.bias"][:2].contiguous()
    hidden = int(parent_args["hidden"])

    split_path = (
        ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{metadata['fold']}.json"
    )
    from wnn.splits import load_campaign_split

    split = load_campaign_split(split_path, ROOT)
    fit_groups = list(split.valid_groups[: args.max_sequences])
    if not fit_groups:
        raise ValueError("fit manifest has no selected sequences")
    fit_row_mask = np.isin(arrays["row_group"], fit_groups)
    selected_targets = np.asarray(arrays["targets"][fit_row_mask], dtype=np.float32)
    if not len(selected_targets):
        raise ValueError("selected fit sequences contain no exported scored rows")
    stride = max(1, int(np.ceil(len(selected_targets) / 65_536)))
    paired, independent, shuffled, codebook_rows = fit_codebooks(
        selected_targets[::stride], seed=args.seed, max_fit_rows=65_536
    )
    dataset = train_rnn.Sequences("valid", fit_groups)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    codebook_tensor = torch.from_numpy(paired).to(device)
    independent_tensor = torch.from_numpy(independent).to(device)
    shuffled_tensor = torch.from_numpy(shuffled).to(device)
    if args.preflight_only:
        backbone = train_rnn.make_model(model_args, n_in, n_out)
        backbone.load_state_dict(backbone_state)
        backbone.to(device).eval()
        head = JointTargetHead(hidden, codebook_tensor, independent_tensor, shuffled_tensor).to(
            device
        )
        wrapper = JointTargetContinuation(backbone, head, "joint_tilted").to(device).eval()
        x, _, _ = dataset[0]
        inputs = train_rnn.build_inputs(x[:512][None].to(device))
        with torch.no_grad():
            prediction, _, _, _, parent = wrapper.forward_aux(inputs, None)
        if not torch.equal(prediction, parent.clamp(-2, 2)):
            raise AssertionError(
                "CPU preflight failed exact two-output parent initialization parity"
            )
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "device": str(device),
                    "fit_groups": fit_groups,
                    "fit_rows_for_codebook": len(selected_targets),
                    "codebook_rows": codebook_rows,
                    "parent_checkpoint_sha256": checkpoint_sha,
                    "callback_output_width": int(prediction.shape[-1]),
                    "evaluation_access": "none",
                }
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for a training run")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    arm_receipts: dict[str, dict[str, float | int]] = {}
    batches_per_epoch = len(loader)
    chunks_per_sequence = 20_000 // args.chunk + int(20_000 % args.chunk != 0)

    for arm in ARMS:
        torch.manual_seed(args.seed)
        backbone = train_rnn.make_model(model_args, n_in, n_out)
        backbone.load_state_dict(backbone_state)
        backbone.to(device).train()
        head = JointTargetHead(hidden, codebook_tensor, independent_tensor, shuffled_tensor).to(
            device
        )
        continuation = JointTargetContinuation(backbone, head, arm).to(device).train()
        optimizer = torch.optim.AdamW(continuation.parameters(), lr=args.lr, weight_decay=0.01)
        totals = np.zeros(3, dtype=np.float64)
        updates = 0
        parity_checked = False
        for x, y, scored in loader:
            x = x.to(device, non_blocking=True)
            targets_seq = y[..., :2].to(device, non_blocking=True).float()
            scored = scored.to(device, non_blocking=True)
            inputs = train_rnn.build_inputs(x)
            hidden_state = None
            for start in range(0, inputs.shape[1], args.chunk):
                stop = min(start + args.chunk, inputs.shape[1])
                pred, joint_logits, independent_logits, next_state, base_prediction = (
                    continuation.forward_aux(inputs[:, start:stop], hidden_state)
                )
                if not parity_checked:
                    if not torch.equal(pred, base_prediction.clamp(-2, 2)):
                        raise AssertionError(
                            "initial paired head does not equal its clipped parent output"
                        )
                    parity_checked = True
                valid = scored[:, start:stop].bool()
                valid[:, : max(0, 99 - start)] = False
                if valid.any():
                    target = targets_seq[:, start:stop].clamp(-2, 2)[valid]
                    prediction = pred[valid]
                    joint = joint_logits[valid]
                    independent_logits_rows = independent_logits[valid]
                    joint_y = torch.cdist(target, head.paired_codebook).argmin(dim=-1)
                    if arm == "shuffled_tilted":
                        joint_y = torch.cdist(target, head.shuffled_codebook).argmin(dim=-1)
                    marginal_y = torch.stack(
                        [
                            torch.cdist(
                                target[:, k : k + 1], head.independent_codebook[k, :, None]
                            ).argmin(dim=-1)
                            for k in range(2)
                        ],
                        dim=-1,
                    )
                    loss, ce, wmse = arm_loss(
                        prediction,
                        joint,
                        independent_logits_rows,
                        joint_y,
                        marginal_y,
                        arm,
                        target,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    require_finite_training_state(
                        loss, (ce, wmse), continuation.parameters(), "gradients"
                    )
                    optimizer.step()
                    require_finite_training_state(
                        loss, (ce, wmse), continuation.parameters(), "parameters"
                    )
                    tick_gpu_duty(duty, device)
                    totals += np.asarray([loss.item(), ce.item(), wmse.item()])
                    updates += 1
                hidden_state = None if next_state is None else next_state.detach()
        if not parity_checked or updates == 0:
            raise RuntimeError("fit pass produced no selected rows or parent-parity check")
        metrics = (totals / updates).tolist()
        arm_receipts[arm] = {
            "optimizer_updates": updates,
            "loss_per_update_mean": metrics[0],
            "quantized_ce_per_update_mean": metrics[1],
            "training_wmse_surrogate_per_update_mean": metrics[2],
        }
        torch.save(
            {
                "model": continuation.cpu().state_dict(),
                "arm": arm,
                "parent_checkpoint_sha256": checkpoint_sha,
                "fit_export_sha256": file_sha256(prefix.with_suffix(".json")),
                "paired_codebook": paired,
                "independent_codebook": independent,
                "shuffled_codebook": shuffled,
                "callback_output_width": 2,
                "epoch": 0,
                "training_args": {
                    "batch_sequences": 1,
                    "chunk": args.chunk,
                    "lr": args.lr,
                    "seed": args.seed,
                },
            },
            output_dir / f"{arm}.pt",
        )
        del optimizer, continuation, head, backbone
        torch.cuda.empty_cache()
    receipt = {
        "scope": "one-pass end-to-end causal-GRU continuation on frozen-fit groups only",
        "evaluation_access": "none",
        "fit_prefix": str(prefix),
        "fit_manifest_sha256": metadata["manifest_sha256"],
        "fit_groups": fit_groups,
        "parent_member": member_name,
        "parent_checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_output_channels": source_n_out,
        "continuation_parent_output_channels": 2,
        "unused_parent_auxiliary_channels_dropped": max(0, source_n_out - 2),
        "parent_initialization_parity": "checked before every arm's first optimizer update",
        "callback_output_width": 2,
        "component_provenance": "validated frozen fold manifest; fit groups only",
        "codebook_fit_rows": codebook_rows,
        "training_rows": len(selected_targets),
        "training_rows_source": "same-origin clipped targets in fit-only state export",
        "arms": list(ARMS),
        "optimizer_updates_per_arm": arm_receipts,
        "metrics_are_fit_diagnostics_only": True,
        "loader_batches": batches_per_epoch,
        "chunks_per_sequence_upper_bound": chunks_per_sequence,
    }
    (output_dir / "e2e_fit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "arms": list(ARMS), "fit_only": True}))


if __name__ == "__main__":
    main()
