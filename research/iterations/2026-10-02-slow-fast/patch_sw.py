from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
a='    ap.add_argument("--class-weight", default=""'
assert s.count(a)==1
s=s.replace(a,'    ap.add_argument("--seq-weight", default="", help="npz with float arrays train/valid indexed by row_group: per-sequence sampling weights")\n'+a)
b='    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=sampler is None,'
assert s.count(b)==1
s=s.replace(b,'''    if args.seq_weight:
        # per-sequence sampling weights (same number of draws per epoch); a uniform file is the matched control
        sw = np.load(args.seq_weight)
        parts = [np.asarray(sw[train_split], dtype=np.float64)[train_idx]]
        if extra_valid is not None:
            parts += [np.asarray(sw["valid"], dtype=np.float64)[extra_valid]] * args.valid_repeat
        draws = args.epoch_samples or len(train_ds)
        sampler = WeightedRandomSampler(torch.from_numpy(np.concatenate(parts)), draws, replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
'''+b)
p.write_text(s)
