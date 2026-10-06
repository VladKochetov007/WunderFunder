from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
a='def true_mask_weights('
assert s.count(a)==1
s=s.replace(a,'''class I1SwapSequences(Sequences):
    """Class-3 train sequences with i0/i1 roles swapped; labels rebuilt from i1's future book events
    (wnn.i1swap). The reliable-label mask is returned in the scored slot, so --valid-true-mask weights it."""

    _tables = None

    def __getitem__(self, i):
        import pickle

        from wnn.i1swap import labels_with_mask, swap_instruments

        if self._file is None:
            self._file = SequenceFile(self.path)
        if I1SwapSequences._tables is None:
            with open(ROOT / "cache" / "class3_moves_full.pkl", "rb") as f:
                I1SwapSequences._tables = pickle.load(f)
        moves, spreads = I1SwapSequences._tables
        seq = self._file.read(self.indices[i])
        x = swap_instruments(seq.features)
        y, ok = labels_with_mask(x[:, 0:22], moves, spreads)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y), torch.from_numpy(ok)


'''+a)
b='    if args.epoch_samples:\n'
assert s.count(b)==1
s=s.replace(b,'''    if args.i1swap:
        if args.prop_cache or teachers:
            ap.error("--i1swap does not support --prop-cache or --distill")
        c3_cls = np.load(ROOT / "research" / "artifacts" / "market-classes-v1.npz")["train"]
        c3_train = np.array([g for g in range(10607) if g not in CORRUPT_TRAIN_ROW_GROUPS and c3_cls[g] == 3])
        c3_train = np.random.default_rng(args.seed + 7).permutation(c3_train)[: args.i1swap]
        train_ds = torch.utils.data.ConcatDataset([train_ds, I1SwapSequences("train", c3_train)])
        print(f"i1swap: +{len(c3_train)} role-swapped class-3 sequences", flush=True)
'''+b)
c='    ap.add_argument("--a-mode",'
assert s.count(c)==1
s=s.replace(c,'    ap.add_argument("--i1swap", type=int, default=0, help="add N class-3 train sequences with i0/i1 roles swapped (rebuilt labels)")\n'+c)
p.write_text(s)
