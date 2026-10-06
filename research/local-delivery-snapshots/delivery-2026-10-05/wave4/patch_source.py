from pathlib import Path
D=Path(__file__).parent;p=D/'scripts/train_rnn.py';s=p.read_text()
s=s.replace('def make_model(args_dict, n_in, n_out):\n','''def make_model(args_dict, n_in, n_out):
    if args_dict.get("warm_init", "none") != "none":
        from warm_model import WarmModel
        return WarmModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out, args_dict["warm_init"])
''')
s=s.replace('    args = ap.parse_args()','''    ap.add_argument("--warm-init", choices=["none", "pool", "state"], default="none")
    args = ap.parse_args()''')
s=s.replace('        model.load_state_dict(init["model"])','''        if args.warm_init != "none":
            missing, unexpected = model.load_state_dict(init["model"], strict=False)
            assert not unexpected and all(k.startswith("warm_adapter.") for k in missing), (missing, unexpected)
        else:
            model.load_state_dict(init["model"])''')
s=s.replace('    frozen = tuple(v for v in args.freeze.split(",") if v)','''    if args.warm_init != "none":
        for pname, prm in model.named_parameters():
            prm.requires_grad_(pname.startswith("warm_adapter."))
    frozen = tuple(v for v in args.freeze.split(",") if v)''')
# Late chunks have no trainable path when only the initial state is trained.
# Keep the graph alive with a zero term; no late target gradient reaches warmup.
s=s.replace('                if args.atom_head > 0:','''                if args.warm_init != "none":
                    loss = loss + 0.0 * model.warm_adapter[2].weight.sum()
                if args.atom_head > 0:''')
p.write_text(s)
