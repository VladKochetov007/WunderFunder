from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''        y = seq.targets
        if self.teachers:'''
assert s.count(old)==1
s=s.replace(old,'''        if DOMAIN_FLAG:  # real sequences: augmentation-domain flag 0 (appended raw column -> extra input)
            x = np.concatenate([x, np.zeros((len(x), 1), x.dtype)], axis=1)
'''+old)
old2='''        y, ok = np.array(ycache[pos[g]]), np.array(okcache[pos[g]])
'''
assert s.count(old2)==1
s=s.replace(old2,old2+'''        if DOMAIN_FLAG:
            x = np.concatenate([x, np.ones((len(x), 1), x.dtype)], axis=1)
''')
old3='A_MODE = "none"'
assert s.count(old3)==1
s=s.replace(old3,'DOMAIN_FLAG = False  # append a 0/1 column marking role-swapped (i1swap) sequences\n'+old3)
old4='''    global A_MODE
    A_MODE = args.a_mode
'''
assert s.count(old4)==1
s=s.replace(old4,old4+'''    global DOMAIN_FLAG
    DOMAIN_FLAG = args.domain_flag
''')
old5='    n_in = N_FEATURES + N_EVENT_FEATURES + N_FEATURES * len(LOOKAHEAD)\n'
assert s.count(old5)==1
s=s.replace(old5,old5+'    if args.domain_flag:\n        n_in += 1\n')
old6='    ap.add_argument("--i1swap",'
s=s.replace(old6,'    ap.add_argument("--domain-flag", action="store_true", help="extra input column: 1 on --i1swap sequences, 0 otherwise")\n'+old6)
p.write_text(s)
q=Path("scripts/predict_run.py");t=q.read_text()
e='    train_rnn.A_MODE = ck["args"].get("a_mode") or "none"\n'
assert t.count(e)==1
t=t.replace(e,e+'    train_rnn.DOMAIN_FLAG = bool(ck["args"].get("domain_flag"))\n');q.write_text(t)
