"""Materialize the isolated trainer from exact declared source replacements."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "scripts/train_rnn.py"


def build():
    src = BASE.read_text()
    original = src
    changes = []

    def replace(before, after):
        nonlocal src
        if src.count(before) != 1:
            raise RuntimeError(f"source anchor count {src.count(before)}: {before[:100]}")
        src = src.replace(before, after)
        changes.append([before, after])

    replace('from wnn.splits import dev_halves, valid_row_groups\n',
            'from wnn.splits import dev_halves, valid_row_groups\nfrom sam_step import sam_backward, skip_evaluation_rng\n')
    replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[4]')
    replace('    ap.add_argument("--name", required=True)',
            '    ap.add_argument("--name", required=True)\n'
            '    ap.add_argument("--sam-stage", choices=["parent", "head"], required=True)\n'
            '    ap.add_argument("--sam-rho", type=float, choices=[0.0, 0.05], default=0.0)')
    replace('    args = ap.parse_args()\n', '    args = ap.parse_args()\n    validate_sam_recipe(args, ap)\n')
    replace('    out_dir.mkdir(parents=True, exist_ok=True)', '    out_dir.mkdir(parents=True, exist_ok=False)')
    # The first outgoing state stays on the original path; only capture its input.
    replace('                with torch.autocast("cuda", dtype=amp_dtype):\n                    out, h = forward_chunk(model, inputs, sl, h)',
            '                h_in = h\n                with torch.autocast("cuda", dtype=amp_dtype):\n                    out, h = forward_chunk(model, inputs, sl, h_in)')

    start = src.index('                if moments is not None:\n                    ce = F.cross_entropy',
                      src.index('            for s in range(0, inputs.shape[1], args.chunk):'))
    stop = src.index('                timer.mark("loss")', start)
    loss_block = src[start:stop]
    # The entire original loss code is preserved verbatim, one indent deeper.
    indented = ''.join('    ' + line if line.strip() else line for line in loss_block.splitlines(True))
    replace(loss_block,
            '                def chunk_loss(out):\n                    nonlocal norm_ema\n' + indented +
            '                    return loss\n\n                loss = chunk_loss(out)\n')
    replace('                scaler.scale(loss).backward()',
            '                def sam_closure():\n'
            '                    with torch.autocast("cuda", dtype=amp_dtype):\n'
            '                        perturbed_out, _ = forward_chunk(model, inputs, sl, h_in)\n'
            '                    return chunk_loss(perturbed_out)\n'
            '                sam_backward(loss, sam_closure, model.parameters(), scaler, args.sam_rho)')
    replace('        res = evaluate(eval_model, valid_loader, 4000, device)',
            '        skip_evaluation_rng(valid_loader)\n'
            '        res = {"evaluation_skipped": True, "policy": "fixed endpoint; no WP computed"}')
    before = '''        improved = res["weighted_pearson"] > best
        if improved:
            best = res["weighted_pearson"]
        if improved or (args.include_heldout_valid and epoch == args.epochs - 1):
            torch.save({"model": eval_model.state_dict(), "args": vars(args), "n_in": n_in,
                        "readout": None if READOUT is None else [t.cpu() for t in READOUT],
                        "readout_tilted": READOUT_TILTED}, out_dir / "model.pt")'''
    after = '''        if epoch == args.epochs - 1:
            torch.save({"model": eval_model.state_dict(), "args": vars(args), "n_in": n_in,
                        "epoch": epoch, "readout": None if READOUT is None else [t.cpu() for t in READOUT],
                        "readout_tilted": READOUT_TILTED}, out_dir / "model.pt")'''
    replace(before, after)
    replace('    print("BEST", best)\n    if args.include_heldout_valid:\n        print("SELECTED_FINAL", history[-1]["weighted_pearson"])',
            '    print("FIXED_ENDPOINT", args.epochs - 1, "validation metrics suppressed")')

    guard = '''
def validate_sam_recipe(args, parser):
    """Reject undeclared configurations before touching competition data."""
    common = dict(hidden=128, layers=2, dropout=0.0, chunk=2000, arch="gru", cell="gru", enc="linear",
                  aux_struct=0.3, aux_next=0.3, train_seqs=10607, n_valid_folds=4,
                  precision="fp16", wd=0.01, aux="", distill="", freeze="", aux_next_k="",
                  aux_tier=0.0, aux_mask=0.0, aux_sign=0.0, aux_quote=0.0, atom_head=0.0,
                  vrex=0.0, level_aug=0.0, level_drop=0.0, lookahead="", state_reset=0,
                  loss_norm="chunk", episode_weight=0.0, td_alpha=0.0, class_weight="",
                  class_balance=False, class_heads=False, raw_only=False, online_norm=False,
                  quant_feats=False, quant_i1=False, heuristics=False, prop_input="",
                  swa_frac=0.0, ema_metric_weight=False, valid_repeat=1, train_half=-1,
                  valid_train_fraction=1.0, epoch_samples=0, train_classes="", max_batches=0,
                  profile_steps=0, parity_dump="", deterministic=False)
    if args.sam_stage == "parent":
        common.update(train_source="train+valid", epochs=4, batch=32, loss="wmse", lr=1e-3,
                      mask_weight=False, prop_power=2.0, prop_eps=0.03, valid_true_mask=0.03,
                      ema_teacher=1.0, ema_decay=0.999, eval_ema=True, since_weight=32.0,
                      since_floor=0.25, init="", freeze="")
    else:
        common.update(train_source="valid", epochs=2, batch=16, loss="gcorr", lr=5e-4,
                      mask_weight=True, propensity="", ema_teacher=0.0, eval_ema=False,
                      since_weight=0.0, freeze="inp,gru", sam_rho=0.0)
        if not args.init.endswith("@ep3") or not args.init.startswith("on_sam_"):
            parser.error("head requires this study's exact fixed parent @ep3")
    for name, value in common.items():
        if getattr(args, name) != value:
            parser.error(f"undeclared {name}={getattr(args, name)!r}; required {value!r}")
    if not args.fast or not args.full_valid or args.seed not in (3, 4):
        parser.error("requires fast/full-valid and registered seed3 or4")
    if args.valid_fold not in (0, 1, -1):
        parser.error("requires fold0/1 or explicitly registered final all-valid fit")
    if args.valid_fold == -1:
        if not args.include_heldout_valid or args.eval_source != "holdout":
            parser.error("final fit requires include-heldout-valid and holdout eval metadata")
    elif args.include_heldout_valid or args.eval_source != "vfold":
        parser.error("fold fit requires vfold and excludes heldout validation")
    if args.sam_stage == "parent" and args.propensity != f"prop_f{max(0, args.valid_fold)}":
        parser.error("parent propensity must match registered fold")
    if not args.name.startswith("on_sam_") or "/" in args.name or ".." in args.name:
        parser.error("output name must stay inside the isolated on_sam_ namespace")
    args.evaluation_policy = "none; fixed parent ep3 or head ep1; preserve evaluation-loader RNG"

'''
    replace('def main():\n', guard + 'def main():\n')
    (HERE / "train_sam.py").write_text(src)
    receipt = {"base": str(BASE.relative_to(ROOT)),
               "base_sha256": hashlib.sha256(original.encode()).hexdigest(),
               "implementation_sha256": hashlib.sha256(src.encode()).hexdigest(),
               "changes": changes}
    (HERE / "trainer-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:v for k,v in receipt.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    build()
