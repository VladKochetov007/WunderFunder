from pathlib import Path
D=Path(__file__).parent;p=D/'scripts/train_rnn.py';s=p.read_text()
s=s.replace('def make_model(args_dict, n_in, n_out):\n','''def make_model(args_dict, n_in, n_out):
    if args_dict.get("ssl_stage", "none") != "none":
        from ssl_model import SSLModel
        return SSLModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out, args_dict["ssl_stage"])
''')
s=s.replace('    args = ap.parse_args()','''    ap.add_argument("--ssl-stage", choices=["none", "control", "auxiliary", "pretrain", "ft"], default="none")
    args = ap.parse_args()''')
s=s.replace('        model.load_state_dict(init["model"])','''        if args.ssl_stage != "none":
            missing, unexpected = model.load_state_dict(init["model"], strict=False)
            assert not unexpected and all(k.startswith("ssl_head.") for k in missing), (missing, unexpected)
        else:
            model.load_state_dict(init["model"])''')
s=s.replace('                timer.mark("loss")','''                if args.ssl_stage != "none":
                    from ssl_model import ssl_loss
                    feature_loss = ssl_loss(model, inputs, s, out.shape[1])
                    loss = feature_loss if args.ssl_stage == "pretrain" else loss + .2 * feature_loss
                timer.mark("loss")''')
p.write_text(s)
