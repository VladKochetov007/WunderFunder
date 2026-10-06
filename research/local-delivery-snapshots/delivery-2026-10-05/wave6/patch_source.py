from pathlib import Path
D=Path(__file__).parent;p=D/'scripts/train_rnn.py';s=p.read_text()
s=s.replace('SEQ_BYTES =','FAMILY_TEACHERS = ()\nSEQ_BYTES =',1)
s=s.replace('            y = np.concatenate([y, t.astype(np.float32)], axis=1)','''            y = np.concatenate([y, t.astype(np.float32)], axis=1)
            if FAMILY_TEACHERS:
                if not hasattr(self, "_family_arrays"):
                    self._family_arrays = [np.load(ROOT / "cache" / f"pred_{n}_{self.split}.npy", mmap_mode="r") for n in FAMILY_TEACHERS]
                y = np.concatenate([y] + [np.asarray(a[j], dtype=np.float32) for a in self._family_arrays], axis=1)''')
s=s.replace('def make_model(args_dict, n_in, n_out):\n','''def make_model(args_dict, n_in, n_out):
    if args_dict.get("family_teachers"):
        from compress_model import CompressModel
        return CompressModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out)
''')
s=s.replace('    args = ap.parse_args()','''    ap.add_argument("--family-teachers", default="")
    args = ap.parse_args()
    global FAMILY_TEACHERS
    FAMILY_TEACHERS = tuple(v for v in args.family_teachers.split(",") if v)
    if FAMILY_TEACHERS:
        assert len(FAMILY_TEACHERS)==4 and args.distill''')
s=s.replace('        model.load_state_dict(init["model"])','''        if args.family_teachers:
            missing, unexpected = model.load_state_dict(init["model"], strict=False)
            assert not unexpected and all(k.startswith("family_head.") for k in missing), (missing, unexpected)
        else:
            model.load_state_dict(init["model"])''')
s=s.replace('            teach = None','            family_labels = y[..., 4:] if args.family_teachers else None\n            teach = None',1)
s=s.replace('                timer.mark("loss")','''                if args.family_teachers:
                    w = y[:, sl].float().clamp(-2, 2).abs().repeat(1, 1, 4) * main_valid[:, sl, None].float()
                    loss = loss + .1 * (w * (model.family_output.float() - family_labels[:, sl].float()).square()).sum() / w.sum().clamp_min(1e-8)
                timer.mark("loss")''')
p.write_text(s)
