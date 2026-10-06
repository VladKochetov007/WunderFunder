from pathlib import Path
D=Path(__file__).parent;p=D/'scripts/train_rnn.py';s=p.read_text()
s=s.replace('    args = ap.parse_args()','''    ap.add_argument("--event-cap-bootstrap", action="store_true")
    args = ap.parse_args()''')
old='''                        alive = (nxt_label[:, sl] > ahead) & (ahead < teach.shape[1])'''
new='''                        if args.event_cap_bootstrap:
                            ahead = torch.minimum(ahead, nxt_label[:, sl] - 1)
                            alive = (ahead > rows) & (ahead < teach.shape[1])
                        else:
                            alive = (nxt_label[:, sl] > ahead) & (ahead < teach.shape[1])'''
assert old in s;s=s.replace(old,new)
p.write_text(s)
