from pathlib import Path
# cache builder variant: no swap (i0 rebuilt labels)
b=Path("research/iterations/2026-10-02-slow-fast/build_i1swap_cache.py").read_text()
b=b.replace("xs=swap_instruments(x);y,ok=labels_with_mask(xs[:,0:22],moves,spreads)","xs=x if NOSWAP else swap_instruments(x);y,ok=labels_with_mask(xs[:,0:22],moves,spreads)")
b=b.replace('moves,spreads=pickle.load','NOSWAP=len(sys.argv)>1 and sys.argv[1]=="noswap";TAG="i0rb" if NOSWAP else "i1swap"\nmoves,spreads=pickle.load')
b=b.replace('S+"cache/i1swap_train_y.npy"','S+f"cache/{TAG}_train_y.npy"').replace('S+"cache/i1swap_train_ok.npy"','S+f"cache/{TAG}_train_ok.npy"').replace('S+"cache/i1swap_train_rg.npy"','S+f"cache/{TAG}_train_rg.npy"')
Path("research/iterations/2026-10-02-slow-fast/build_i1swap_cache.py").write_text(b)
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''            rgs = np.load(ROOT / "cache" / "i1swap_train_rg.npy")
            I1SwapSequences._tables = ({int(g): j for j, g in enumerate(rgs)},
                                       np.load(ROOT / "cache" / "i1swap_train_y.npy", mmap_mode="r"),
                                       np.load(ROOT / "cache" / "i1swap_train_ok.npy", mmap_mode="r"))'''
new='''            tag = "i0rb" if I1SwapSequences.noswap else "i1swap"
            rgs = np.load(ROOT / "cache" / f"{tag}_train_rg.npy")
            I1SwapSequences._tables = ({int(g): j for j, g in enumerate(rgs)},
                                       np.load(ROOT / "cache" / f"{tag}_train_y.npy", mmap_mode="r"),
                                       np.load(ROOT / "cache" / f"{tag}_train_ok.npy", mmap_mode="r"))'''
assert s.count(old)==1;s=s.replace(old,new)
old2='        x = swap_instruments(self._file.read(g).features)\n'
assert s.count(old2)==1
s=s.replace(old2,'        x = self._file.read(g).features\n        if not I1SwapSequences.noswap:\n            x = swap_instruments(x)\n')
s=s.replace("    with_prop = False\n","    with_prop = False\n    noswap = False  # --i1swap-noswap: same class-3 sequences, i0 kept, labels replaced by rebuilt ones (label-noise control)\n",1)
old3='        I1SwapSequences.with_prop = bool(prop_cache)\n'
assert s.count(old3)==1
s=s.replace(old3,old3+'        I1SwapSequences.noswap = args.i1swap_noswap\n')
s=s.replace('    ap.add_argument("--domain-flag",','    ap.add_argument("--i1swap-noswap", action="store_true")\n    ap.add_argument("--domain-flag",',1)
p.write_text(s)
