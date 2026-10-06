from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='    model = make_model(vars(args), n_in, n_out).to(device)\n'
assert s.count(old)==1
s=s.replace(old,old+'''    if args.no_row_gru:  # event-clock only: the dual model's row GRU is zeroed and frozen (its output stays exactly 0)
        if args.arch != "dual":
            ap.error("--no-row-gru requires --arch dual")
        for prm in model.gru.parameters():
            prm.data.zero_()
            prm.requires_grad_(False)
''')
s=s.replace('    ap.add_argument("--i1swap-noswap",','    ap.add_argument("--no-row-gru", action="store_true")\n    ap.add_argument("--i1swap-noswap",',1)
p.write_text(s)
