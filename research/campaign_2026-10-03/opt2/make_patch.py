from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace_once(source: str, old: str, new: str) -> str:
    """Replace one explicit source anchor or fail for controller review."""
    if source.count(old) != 1:
        raise ValueError(f"expected unique integration anchor: {old[:100]!r}")
    return source.replace(old, new)


def main() -> None:
    """Build a review-only diff from the captured baseline; never touch trainer."""
    source = (HERE / "baseline/train_rnn.py").read_text()
    patched = source

    patched = replace_once(
        patched,
        "from wnn.optim import GradStats, MuonAdamW, muon_param_groups, rnn_init_, wsd_lambda",
        "from wnn.optim import GradStats, rnn_init_\n\n"
        "# Experimental module remains isolated in research until controller review.\n"
        'sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/campaign_2026-10-03/opt2"))\n'
        "from training_dynamics import Dynamics, add_arguments, build_optimizer, build_scheduler\n"
        "from sampling import EpochOrderSampler, init_qmc_",
    )
    patched = replace_once(
        patched, 'choices=["onecycle", "wsd"]', 'choices=["onecycle", "wsd", "sgdr"]'
    )
    patched = replace_once(
        patched,
        "    args = ap.parse_args()\n",
        "    add_arguments(ap)\n"
        "    args = ap.parse_args()\n"
        "    if args.zoneout != 0:\n"
        '        ap.error("exact zoneout is unsupported for fused nn.GRU; no approximation is made")\n'
        '    if args.batch_order == "qmc":\n'
        '        ap.error("QMC sampler helper is tested; streamed training-input metadata integration is deferred")\n'
        '    if args.antithetic and args.batch_order != "seeded":\n'
        '        ap.error("--antithetic requires --batch-order seeded and a same-seed seeded control")\n'
        '    if args.init_qmc and (args.init or args.rnn_init != "default" or args.enc != "linear" or args.arch != "gru"):\n'
        '        ap.error("--init-qmc supports fresh plain GRU/LSTM RNNModel only; no checkpoint or rnn-init combination")\n',
    )
    patched = replace_once(
        patched,
        "    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=sampler is None,",
        '    if args.batch_order == "seeded":\n'
        "        if sampler is not None:\n"
        '            ap.error("seeded/antithetic order cannot replace weighted or replacement sampling")\n'
        "        sampler = EpochOrderSampler(len(train_ds), args.seed, args.antithetic)\n"
        "        sampler.set_epoch(-1)  # separate normalization pass; same for the paired control\n"
        "    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=sampler is None,",
    )
    patched = replace_once(
        patched,
        "    model = make_model(vars(args), n_in, n_out).to(device)\n",
        "    model = make_model(vars(args), n_in, n_out).to(device)\n"
        "    if args.init_qmc:\n"
        "        init_qmc_(model, args.seed)\n",
    )
    start = patched.index(
        '    if args.optim == "muon":\n', patched.index("    total = len(train_loader)")
    )
    end = patched.index("    amp_dtype = ", start)
    patched = (
        patched[:start]
        + (
            "    opt = build_optimizer(model, args)\n"
            "    sched = build_scheduler(opt, args, total)\n"
            "    dynamics = Dynamics(model, args, total)\n"
            '    provenance["opt2_switch_events"] = dynamics.events\n'
            '    provenance["opt2_sources"] = {\n'
            '        name: file_sha256(ROOT / "research/campaign_2026-10-03/opt2" / name)\n'
            '        for name in ("training_dynamics.py", "sampling.py")\n'
            "    }\n"
        )
        + patched[end:]
    )
    patched = replace_once(
        patched,
        "        epoch_order_identity = sample_order_identity(train_loader)\n",
        "        if isinstance(train_loader.sampler, EpochOrderSampler):\n"
        "            train_loader.sampler.set_epoch(epoch)\n"
        "        epoch_order_identity = sample_order_identity(train_loader)\n"
        "        if isinstance(train_loader.sampler, EpochOrderSampler):\n"
        '            epoch_order_identity["opt2_seeded_order"] = {"seed": args.seed, "epoch": epoch, "antithetic": args.antithetic}\n',
    )
    patched = replace_once(
        patched,
        "                opt.zero_grad(set_to_none=True)\n",
        "                opt = dynamics.maybe_switch(step_count, opt, sched)\n"
        "                opt.zero_grad(set_to_none=True)\n",
    )
    patched = replace_once(
        patched,
        "                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)\n",
        "                dynamics.perturb_and_clip(step_count)\n",
    )
    ast.parse(patched)
    (HERE / "trainer_candidate.py").write_text(patched)
    diff = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="a/scripts/train_rnn.py",
            tofile="b/scripts/train_rnn.py",
        )
    )
    (HERE / "trainer.patch").write_text(diff)
    (HERE / "PATCH_META.json").write_text(
        json.dumps(
            {
                "baseline_trainer_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "baseline_optimizer_sha256": hashlib.sha256(
                    (HERE / "baseline/optim.py").read_bytes()
                ).hexdigest(),
                "candidate_sha256": hashlib.sha256(patched.encode()).hexdigest(),
                "patch_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "status": "unapplied; rebase against live loader repair before integration",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
