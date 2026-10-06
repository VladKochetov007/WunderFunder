from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''        y, ok = np.array(ycache[pos[g]]), np.array(okcache[pos[g]])
'''
assert s.count(old)==1
s=s.replace(old,old+'''        if I1SwapSequences.true_y:  # weighting control: true i0 labels, same reliable-row weighting as the augmentation
            assert I1SwapSequences.noswap
            y = np.ascontiguousarray(self._file.read(g).targets, dtype=np.float32)
''')
s=s.replace("    noswap = False  #","    true_y = False\n    noswap = False  #",1)
old2='        I1SwapSequences.noswap = args.i1swap_noswap\n'
assert s.count(old2)==1
s=s.replace(old2,old2+'        I1SwapSequences.true_y = args.i1swap_truey\n')
s=s.replace('    ap.add_argument("--i1swap-noswap", action="store_true")\n','    ap.add_argument("--i1swap-noswap", action="store_true")\n    ap.add_argument("--i1swap-truey", action="store_true")\n',1)
p.write_text(s)
