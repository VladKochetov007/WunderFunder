from pathlib import Path
D=Path(__file__).parent;p=D/'scripts/train_rnn.py';s=p.read_text()
s=s.replace('def make_model(args_dict, n_in, n_out):\n','''def make_model(args_dict, n_in, n_out):
    if args_dict.get("joint_mode", "none") != "none":
        from joint_model import JointModel
        return JointModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out, args_dict["joint_mode"])
''')
s=s.replace('    args = ap.parse_args()','''    ap.add_argument("--joint-mode", choices=["none", "independent", "coupled"], default="none")
    ap.add_argument("--init-other", default="")
    args = ap.parse_args()''')
s=s.replace('        model.load_state_dict(init["model"])','''        if args.joint_mode != "none":
            missing, unexpected = model.load_state_dict(init["model"], strict=False)
            assert not unexpected and all(k.startswith("other.") for k in missing), (missing, unexpected)
            other_run, _, other_ep = args.init_other.partition("@ep")
            other_file = f"model_ep{int(other_ep)}.pt" if other_ep else "model.pt"
            other = torch.load(ROOT / "artifacts" / "runs" / other_run / other_file, map_location="cpu", weights_only=False)
            model.other.load_state_dict(other["model"])
        else:
            model.load_state_dict(init["model"])''')
s=s.replace('                if args.atom_head > 0:','''                if args.joint_mode != "none":
                    from joint_model import joint_loss
                    loss = joint_loss(model, y[:, sl], main_valid[:, sl], objective)
                if args.atom_head > 0:''')
p.write_text(s)
