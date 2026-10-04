"""Ensemble-aware head fine-tuning of fixed members on a clean fold.

Each member is trained alone, but the submission scores the raw average of member outputs, so how each member's
scale and offset sit inside that average is never optimised. ``--mode joint`` trains every member's head (trunks
frozen) on the pooled weighted correlation of the equal-weight average. ``--mode solo`` is the matched control:
same members, rows, order, steps, schedule and optimiser, but each head follows its own pooled correlation.

    joint_head_ft.py OUT --fold K --mode joint|solo --members RUN@EP RUN@EP ... [--lr 2e-4]

Writes artifacts/runs/OUT_m<k>/model_ep0.pt (loadable by predict_run.py) and exports cache predictions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import train_rnn  # noqa: E402
from train_rnn import Sequences, build_inputs, detach_state, forward_chunk, make_model  # noqa: E402
from wnn.data import WARMUP  # noqa: E402
from wnn.nn import weighted_pooled_corr_loss  # noqa: E402
from wnn.resources import configure_torch  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

MANIFEST = "research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
FEATURE_FLAGS = ("raw_only", "online_norm", "heuristics", "prop_input")


def load_member(ref: str, split_sha: str, device):
    run, _, epoch = ref.partition("@ep")
    path = ROOT / "artifacts" / "runs" / run / f"model_ep{epoch}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["args"].get("split_manifest_sha256") != split_sha:
        raise ValueError(f"{ref}: not fitted on this clean fold")
    model = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]).to(device)
    model.load_state_dict(ck["model"])
    for name, prm in model.named_parameters():
        prm.requires_grad_(name.startswith("head."))
    return model, ck, file_sha256(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--mode", choices=("joint", "solo"), required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    outputs = [ROOT / "artifacts" / "runs" / f"{args.out}_m{j}" for j in range(len(args.members))]
    if any(o.exists() for o in outputs):
        raise FileExistsError(f"{args.out}: output runs already exist")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    configure_torch()
    device = torch.device("cuda")
    split = load_campaign_split(ROOT / MANIFEST.format(fold=args.fold), ROOT)
    members = [load_member(ref, split.sha256, device) for ref in args.members]
    flags = {tuple(ck["args"].get(f) for f in FEATURE_FLAGS) for _, ck, _ in members}
    if len(flags) != 1 or any(flags.pop()):
        raise ValueError("members must share the plain feature path")

    groups = np.array(sorted(split.valid_groups))
    split.assert_fit_groups([], groups)
    loader = DataLoader(Sequences("valid", groups), batch_size=args.batch, shuffle=True, num_workers=4,
                        drop_last=True, generator=torch.Generator().manual_seed(args.seed))
    params = [p for model, _, _ in members for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total = len(loader) * math.ceil(20_000 / args.chunk) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05,
                                                final_div_factor=100)
    scaler = torch.amp.GradScaler()
    k = len(members)
    started, losses = time.time(), []
    for model, _, _ in members:
        model.train()
    for _epoch in range(args.epochs):
        for x, y, scored in loader:
            inputs = build_inputs(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            steps = torch.arange(inputs.shape[1], device=device)[None]
            selected = (steps >= WARMUP) & scored.to(device, non_blocking=True)
            states = [None] * k
            for s in range(0, inputs.shape[1], args.chunk):
                sl = slice(s, s + args.chunk)
                outs = []
                for j, (model, _, _) in enumerate(members):
                    with torch.autocast("cuda", dtype=torch.float16):
                        out, states[j] = forward_chunk(model, inputs, sl, states[j])
                    states[j] = detach_state(states[j])
                    outs.append(out[..., :2].float())
                if args.mode == "joint":
                    loss = weighted_pooled_corr_loss(torch.stack(outs).mean(0), y[:, sl], selected[:, sl])
                else:
                    loss = sum(weighted_pooled_corr_loss(o, y[:, sl], selected[:, sl]) for o in outs) / k
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(float(loss.detach()))
    seconds = time.time() - started
    record = {"mode": args.mode, "fold": args.fold, "members": args.members, "lr": args.lr, "epochs": args.epochs,
              "batch": args.batch, "chunk": args.chunk, "seed": args.seed, "steps": len(losses),
              "fit_valid_groups": len(groups), "seconds": round(seconds, 1),
              "loss_first_last": [float(np.mean(losses[:50])), float(np.mean(losses[-50:]))]}
    print("JOINT_FIT", json.dumps(record), flush=True)
    for j, (model, ck, parent_sha) in enumerate(members):
        run = f"{args.out}_m{j}"
        out_dir = ROOT / "artifacts" / "runs" / run
        out_dir.mkdir(parents=True, exist_ok=False)
        saved = copy.deepcopy(ck)
        saved["model"] = {name: v.detach().cpu() for name, v in model.state_dict().items()}
        saved["epoch"] = 0
        saved["args"] = dict(ck["args"], name=run, init=args.members[j], joint_head=record)
        saved["provenance"] = dict(ck.get("provenance", {}), joint_head=record, parent_checkpoint_sha256=parent_sha)
        torch.save(saved, out_dir / "model_ep0.pt")
        (out_dir / "provenance.json").write_text(json.dumps(
            {"joint_head": record, "parent_checkpoint_sha256": parent_sha,
             "checkpoint_sha256": {"model_ep0.pt": file_sha256(out_dir / "model_ep0.pt")},
             "split_manifest_sha256": split.sha256}, indent=2) + "\n")
    del members
    torch.cuda.empty_cache()
    for j in range(k):
        subprocess.run([sys.executable, str(ROOT / "scripts/predict_run.py"), f"{args.out}_m{j}", "valid",
                        "--epoch", "0"], check=True, cwd=ROOT)
    print("JOINT_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
