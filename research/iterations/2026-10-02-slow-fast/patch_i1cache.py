from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''        if I1SwapSequences._tables is None:
            with open(ROOT / "cache" / "class3_moves_full.pkl", "rb") as f:
                I1SwapSequences._tables = pickle.load(f)
        moves, spreads = I1SwapSequences._tables
        seq = self._file.read(self.indices[i])
        x = swap_instruments(seq.features)
        y, ok = labels_with_mask(x[:, 0:22], moves, spreads)
'''
new='''        if I1SwapSequences._tables is None:  # labels/masks precomputed by build_i1swap_cache.py
            rgs = np.load(ROOT / "cache" / "i1swap_train_rg.npy")
            I1SwapSequences._tables = ({int(g): j for j, g in enumerate(rgs)},
                                       np.load(ROOT / "cache" / "i1swap_train_y.npy", mmap_mode="r"),
                                       np.load(ROOT / "cache" / "i1swap_train_ok.npy", mmap_mode="r"))
        pos, ycache, okcache = I1SwapSequences._tables
        g = int(self.indices[i])
        x = swap_instruments(self._file.read(g).features)
        y, ok = np.array(ycache[pos[g]]), np.array(okcache[pos[g]])
'''
assert s.count(old)==1
s=s.replace(old,new);p.write_text(s)
