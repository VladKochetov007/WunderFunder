from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y), torch.from_numpy(ok)
'''
new='''        out = (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y), torch.from_numpy(ok))
        if I1SwapSequences.with_prop:  # placeholder: rows of masked sequences are weighted by --valid-true-mask, not propensity
            out = out + (torch.ones(len(ok), dtype=torch.float32),)
        return out
'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("    _tables = None\n","    _tables = None\n    with_prop = False\n",1)
old2='''        if args.prop_cache or teachers:
            ap.error("--i1swap does not support --prop-cache or --distill")
'''
new2='''        if teachers or args.valid_true_mask < 0:
            ap.error("--i1swap needs --valid-true-mask (reliable-label weights) and no --distill")
        I1SwapSequences.with_prop = bool(prop_cache)
'''
assert s.count(old2)==1;s=s.replace(old2,new2);p.write_text(s)
