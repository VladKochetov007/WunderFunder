from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
a='RAW_ONLY = False  # zero the event-feature block'
assert s.count(a)==1
s=s.replace(a,'A_MODE = "none"  # additional features a0..a7 (raw cols 104-111): none | drop (zero) | demean (minus causal running mean)\n'+a)
b='''    if RAW_ONLY:
        ev = torch.zeros_like(ev)
'''
assert s.count(b)==1
s=s.replace(b,b+'''    if A_MODE != "none":
        a_cols = raw[..., 104:112]
        if A_MODE == "drop":
            a_new = torch.zeros_like(a_cols)
        else:
            n = torch.arange(1, raw.shape[1] + 1, device=raw.device, dtype=raw.dtype)[None, :, None]
            a_new = a_cols - torch.cumsum(a_cols, 1) / n
        raw = torch.cat([raw[..., :104], a_new, raw[..., 112:]], dim=-1)
''')
c='    ap.add_argument("--raw-only", action="store_true"'
assert s.count(c)==1
s=s.replace(c,'    ap.add_argument("--a-mode", choices=["none", "drop", "demean"], default="none")\n'+c)
d='''    if args.raw_only:
        global RAW_ONLY
'''
assert s.count(d)==1
s=s.replace(d,'''    global A_MODE
    A_MODE = args.a_mode
'''+d)
p.write_text(s)
q=Path("scripts/predict_run.py");t=q.read_text()
e='    train_rnn.RAW_ONLY = bool(ck["args"].get("raw_only"))\n'
assert t.count(e)==1
t=t.replace(e,e+'    train_rnn.A_MODE = ck["args"].get("a_mode") or "none"\n')
q.write_text(t)
