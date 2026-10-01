"""Exact minimal transform of original H2 trainer; objective/update path unchanged."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]


def build():
    original=(ROOT/'scripts/train_rnn.py').read_text();source=original;changes=[]
    assert hashlib.sha256(original.encode()).hexdigest()=='f97047e6ef51f32097f2cbe2eefc7c57766771ebaed12ae8798a3c1b7efcd52f'
    def replace(before,after):
        nonlocal source
        assert source.count(before)==1,(before[:80],source.count(before))
        source=source.replace(before,after);changes.append([before,after])
    replace('from wnn.splits import dev_halves, valid_row_groups\n',
            'from wnn.splits import dev_halves, valid_row_groups\nfrom recipe import validate_recipe, verify_inputs, skip_evaluation_rng\nfrom model import ContextMemoryModel\n')
    replace('ROOT = Path(__file__).resolve().parents[1]','ROOT = Path(__file__).resolve().parents[4]')
    start=source.index('def make_model(args_dict, n_in, n_out):\n');end=source.index('\n\n@torch.no_grad()\ndef predict(',start)
    replace(source[start:end],'''def make_model(args_dict, n_in, n_out):
    return ContextMemoryModel(n_in, args_dict["hidden"], args_dict["layers"],
                              args_dict["dropout"], n_out, arm=args_dict["ctx_arm"])
''')
    replace('    ap.add_argument("--name", required=True)',
            '    ap.add_argument("--name", required=True)\n    ap.add_argument("--ctx-arm", choices=["current", "memory"], required=True)')
    replace('    args = ap.parse_args()\n','    args = ap.parse_args()\n    validate_recipe(args, ap)\n    verify_inputs(args)\n')
    replace('    out_dir.mkdir(parents=True, exist_ok=True)','    out_dir.mkdir(parents=True, exist_ok=False)')
    replace('        res = evaluate(eval_model, valid_loader, 4000, device)',
            '        skip_evaluation_rng(valid_loader)\n        res = {"evaluation_skipped": True, "policy": "fixed endpoint; no WP computed"}')
    start=source.index('        improved = res["weighted_pearson"] > best\n');end=source.index('\n        t0 = time.time()',start)
    replace(source[start:end],'''        if epoch == args.epochs - 1:
            torch.save({"model": eval_model.state_dict(), "args": vars(args), "n_in": n_in,
                        "epoch": epoch, "readout": None if READOUT is None else [t.cpu() for t in READOUT],
                        "readout_tilted": READOUT_TILTED}, out_dir / "model.pt")''')
    replace('    print("BEST", best)\n    if args.include_heldout_valid:\n        print("SELECTED_FINAL", history[-1]["weighted_pearson"])',
            '    print("FIXED_ENDPOINT", args.epochs - 1, "validation metrics suppressed")')
    (HERE/'train_ctx.py').write_text(source)
    receipt={'base':'scripts/train_rnn.py','base_sha256':hashlib.sha256(original.encode()).hexdigest(),
             'implementation_sha256':hashlib.sha256(source.encode()).hexdigest(),'changes':changes,
             'objective_optimizer_AMP_scheduler_clip_blocks_unchanged':True}
    (HERE/'trainer-source.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='changes'}))


if __name__=='__main__':build()
