from pathlib import Path
D=Path(__file__).parent
p=D/'scripts/train_rnn.py'
s=p.read_text()
needle='def make_model(args_dict, n_in, n_out):\n'
s=s.replace(needle,needle+'''    if args_dict.get("research_mode", "none") != "none":
        from research_models import AdapterModel
        return AdapterModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out, args_dict["research_mode"])
''')
needle='    args = ap.parse_args()'
assert needle in s
s=s.replace(needle,'''    ap.add_argument("--research-mode", choices=["none", "quiet", "shared", "ratio", "direct"], default="none")
'''+needle)
needle='        model.load_state_dict(init["model"])'
assert needle in s
s=s.replace(needle,'''        if args.research_mode != "none":
            missing, unexpected = model.load_state_dict(init["model"], strict=False)
            assert not unexpected and all(k.startswith("adapter.") for k in missing), (missing, unexpected)
        else:
            model.load_state_dict(init["model"])''')
needle='    frozen = tuple(v for v in args.freeze.split(",") if v)'
s=s.replace(needle,'''    if args.research_mode != "none":
        for pname, prm in model.named_parameters():
            prm.requires_grad_(pname.startswith("adapter."))
'''+needle)
needle='                if args.atom_head > 0:'
assert s.count(needle)==1
s=s.replace(needle,'''                if args.research_mode != "none":
                    from research_models import adapter_loss
                    loss = loss + adapter_loss(model, y[:, sl], scored_dev[:, sl], valid[:, sl])
'''+needle)
p.write_text(s)
