from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''        is_muon = lambda n, prm: prm.ndim == 2 and not n.startswith(("inp", "head.2", "mean", "std"))'''
assert s.count(old)==1
s=s.replace(old,'''        skip = ("head.2", "mean", "std") if args.muon_inp else ("inp", "head.2", "mean", "std")
        is_muon = lambda n, prm: prm.ndim == 2 and not n.startswith(skip)''')
s=s.replace('    ap.add_argument("--muon-momentum",','    ap.add_argument("--muon-inp", action="store_true", help="Muon also on the input projection")\n    ap.add_argument("--muon-momentum",',1)
p.write_text(s)
