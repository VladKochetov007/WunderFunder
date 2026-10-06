from pathlib import Path
D=Path(__file__).parent;p=D/'scripts/train_rnn.py';s=p.read_text()
s=s.replace('def make_model(args_dict, n_in, n_out):\n','''def make_model(args_dict, n_in, n_out):
    if args_dict.get("adaptive_schedule", "none") != "none":
        from adaptive_model import AdaptiveModel
        return AdaptiveModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["event_hidden"], args_dict["dropout"], n_out, args_dict["adaptive_schedule"])
''')
s=s.replace('    args = ap.parse_args()','''    ap.add_argument("--adaptive-schedule", choices=["none", "all", "event8"], default="none")
    args = ap.parse_args()''')
p.write_text(s)
