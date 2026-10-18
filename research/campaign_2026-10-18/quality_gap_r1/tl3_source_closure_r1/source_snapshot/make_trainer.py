"""Build train_rnn_plew19.py from the frozen champion-recipe trainer (sha f922...).

Changes (and nothing else):
1. ARGS_FROM env: copy every argument of an archived checkpoint (the w19distilled recipe), then override
   name/seed/ple_knots from env. The recipe is identical by construction, not by a hand-written flag list.
2. --ple-knots: make_model builds FullPLEModel and installs the fixed knots (zero PLE weight), so training,
   checkpoint reload and prediction all use one code path.
3. Init load tolerates missing ple_* keys (the ancestor has none) besides family_head.*.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "frozen_train_rnn.py").read_text()
assert hashlib.sha256(src.encode()).hexdigest() == "f92260e5549107519048d0c94307e3ee3bbd6902910231144c0083af36d557b8"


def once(old, new):
    global src
    assert src.count(old) == 1, old[:100]
    src = src.replace(old, new, 1)


once('def make_model(args_dict, n_in, n_out):\n', '''def make_model(args_dict, n_in, n_out):
    if args_dict.get("ple_knots"):
        import numpy as _np
        from ple_model import FullPLEModel
        _m = FullPLEModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out)
        with _np.load(args_dict["ple_knots"], allow_pickle=False) as _f:
            _m.install_ple({k: _f[k] for k in ("feature", "left", "right", "support_count", "knot_count")})
        return _m
''')
once('    args = ap.parse_args()\n    global FAMILY_TEACHERS\n', '''    ap.add_argument("--ple-knots", default="")
    args = ap.parse_args()
    if os.environ.get("ARGS_FROM"):
        _ck = torch.load(os.environ["ARGS_FROM"], map_location="cpu", weights_only=False)["args"]
        _ck = vars(_ck) if not isinstance(_ck, dict) else _ck
        for _k, _v in _ck.items():
            setattr(args, _k, _v)
        args.name = os.environ["RUN_NAME"]
        args.seed = int(os.environ["RUN_SEED"])
        args.ple_knots = os.environ.get("PLE_KNOTS", "")
        print("ARGS_FROM", os.environ["ARGS_FROM"], "name", args.name, "seed", args.seed, "ple", args.ple_knots, flush=True)
    global FAMILY_TEACHERS
''')
once('''        if args.family_teachers:
            missing, unexpected = model.load_state_dict(init["model"], strict=False)
            assert not unexpected and all(k.startswith("family_head.") for k in missing), (missing, unexpected)
''', '''        if args.family_teachers or args.ple_knots:
            missing, unexpected = model.load_state_dict(init["model"], strict=False)
            assert not unexpected and all(k.startswith(("family_head.", "ple_")) for k in missing), (missing, unexpected)
            if args.ple_knots:
                import numpy as _np
                with _np.load(args.ple_knots, allow_pickle=False) as _f:
                    for _key in ("mean", "std"):
                        assert torch.equal(model.state_dict()[_key].cpu(), torch.from_numpy(_f[_key])), _key
                assert model.ple_weight.count_nonzero() == 0
''')
assert "import os" in src
(HERE / "train_rnn_plew19.py").write_text(src)
print("wrote", HERE / "train_rnn_plew19.py")
