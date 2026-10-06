from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''        if I1SwapSequences.true_y:'''
assert s.count(old)==1
s=s.replace(old,'''        if I1SwapSequences.pseudo:  # noisy-student: teacher soft labels on the swapped stream, every row used
            if I1SwapSequences._pseudo is None:
                I1SwapSequences._pseudo = np.load(ROOT / "cache" / f"pred_{I1SwapSequences.pseudo}_i1swap.npy", mmap_mode="r")
            y = np.array(I1SwapSequences._pseudo[pos[g]], dtype=np.float32)
            ok = np.ones(len(y), bool)
'''+old)
s=s.replace("    true_y = False\n","    true_y = False\n    pseudo = \"\"\n    _pseudo = None\n",1)
old2='        I1SwapSequences.true_y = args.i1swap_truey\n'
assert s.count(old2)==1
s=s.replace(old2,old2+'        I1SwapSequences.pseudo = args.i1swap_pseudo\n')
s=s.replace('    ap.add_argument("--i1swap-truey", action="store_true")\n','    ap.add_argument("--i1swap-truey", action="store_true")\n    ap.add_argument("--i1swap-pseudo", default="", help="teacher name: labels from cache/pred_<name>_i1swap.npy")\n',1)
p.write_text(s)
