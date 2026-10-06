from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''    eligible_valid_count = 0 if valid_pool is None else len(valid_pool)
'''
assert s.count(old)==1
s=s.replace(old,'''    if args.extra_exclude_fold >= 0:  # nested simulation: a second validation fold is withheld from training too
        if fold_groups is None:
            ap.error("--extra-exclude-fold needs --valid-fold")
        excluded = all_valid[fold_of == args.extra_exclude_fold]
        if valid_pool is not None:
            valid_pool = np.setdiff1d(valid_pool, excluded)
        if extra_valid is not None:
            extra_valid = np.setdiff1d(extra_valid, excluded)
'''+old)
s=s.replace('    ap.add_argument("--no-row-gru",','    ap.add_argument("--extra-exclude-fold", type=int, default=-1)\n    ap.add_argument("--no-row-gru",',1)
p.write_text(s)
