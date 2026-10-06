from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''    train_ds = Sequences(train_split, train_idx, teachers, prop_cache.get(train_split),
                         quote_path=args.aux_quote > 0)
'''
assert s.count(old)==1
new='''    if args.seq_repeat:
        # weighting by repetition with plain shuffling (no sampling with replacement): each sequence appears
        # floor(w) or ceil(w) times (stochastic rounding, fixed rng); weights are normalised to mean 1 per source
        rw = np.load(args.seq_repeat)
        def repeat_idx(idx, w):
            w = np.asarray(w, dtype=np.float64)[idx]
            w = w / max(w.mean(), 1e-12)
            rng = np.random.default_rng(args.seed + 11)
            cnt = np.floor(w).astype(int) + (rng.random(len(w)) < (w - np.floor(w)))
            return np.repeat(idx, cnt)
        train_idx = repeat_idx(np.asarray(train_idx), rw[train_split])
        if extra_valid is not None:
            extra_valid = repeat_idx(np.asarray(extra_valid), rw["valid"])
        print(f"seq-repeat: train {len(train_idx)} valid {0 if extra_valid is None else len(extra_valid)}", flush=True)
'''+old
s=s.replace(old,new)
s=s.replace('    ap.add_argument("--seq-weight",','    ap.add_argument("--seq-repeat", default="", help="npz train/valid per-row_group weights applied by repetition (plain shuffle)")\n    ap.add_argument("--seq-weight",',1)
p.write_text(s)
