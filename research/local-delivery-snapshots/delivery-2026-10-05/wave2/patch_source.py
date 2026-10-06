"""Create isolated experiment scripts, keeping shared trainer unchanged."""
from pathlib import Path
import hashlib
import json

D = Path(__file__).resolve().parent
out = D / "scripts"
out.mkdir(exist_ok=True)
p = (D / "source/train_rnn.py").read_text()
def replace(old, new):
    global p
    assert p.count(old) == 1, (old[:80], p.count(old))
    p = p.replace(old, new)

replace('ROOT = Path(__file__).resolve().parents[1]',
        'ROOT = Path(os.environ["WUNDER_RESEARCH_ROOT"])')
replace('RAW_ONLY = False  # zero the event-feature block (organizer-style raw view; keeps n_in for native export)',
        'RAW_ONLY = False\nSENTINEL_MODE = "none"')
replace('def build_inputs(x):', '''def sentinel_features(raw, mode):
    # Six columns in every experimental arm: indicator + causal missing-run age.
    a = raw[..., 109:112]
    missing = a <= -5.19
    rows = torch.arange(raw.shape[1], device=raw.device)[None, :, None].expand_as(a)
    last = torch.where(~missing, rows, torch.full_like(rows, -1)).cummax(dim=1).values
    age = torch.log1p((rows - last).clamp(max=10000).float())
    extra = torch.cat([missing.to(raw.dtype), age.to(raw.dtype)], -1)
    if mode == "zero":
        return raw, torch.zeros_like(extra)
    if mode == "indicator":
        extra[..., 3:] = 0
    changed = raw.clone()
    changed[..., 109:112] = torch.where(missing, torch.zeros_like(a), a)
    return changed, extra


def build_inputs(x):''')
replace('    inputs = torch.cat([raw, ev], dim=-1)\n',
        '    sentinel = None\n    if SENTINEL_MODE != "none":\n        raw, sentinel = sentinel_features(raw, SENTINEL_MODE)\n    inputs = torch.cat([raw, ev], dim=-1)\n    if sentinel is not None:\n        inputs = torch.cat([inputs, sentinel], dim=-1)\n')
replace('    ap.add_argument("--raw-only", action="store_true", help="zero the event-feature inputs (raw 112-column view)")',
        '''    ap.add_argument("--raw-only", action="store_true", help="zero the event-feature inputs")
    ap.add_argument("--sentinel-mode", choices=["none", "zero", "indicator", "age"], default="none")
    ap.add_argument("--train-index-file", default="")
    ap.add_argument("--complement-scale", type=float, default=0.0)''')
replace('    workers, prefetch = dataloader_budget(args.batch * SEQ_BYTES)',
        '''    if args.train_index_file:
        if train_split != "train" or extra_valid is not None:
            raise ValueError("explicit train indices require train-only source")
        allowed = np.load(args.train_index_file)
        train_idx = np.array([g for g in train_idx if g in set(allowed.tolist())])
        if len(train_idx) < args.batch:
            raise ValueError("too few approved training sequences")
    if args.complement_scale and not (args.distill and 0 < args.complement_scale <= 1):
        raise ValueError("complement loss requires a frozen teacher and 0 < scale <= 1")
    workers, prefetch = dataloader_budget(args.batch * SEQ_BYTES)''')
replace('    n_in = N_FEATURES + N_EVENT_FEATURES + N_FEATURES * len(LOOKAHEAD)',
        '''    n_in = N_FEATURES + N_EVENT_FEATURES + N_FEATURES * len(LOOKAHEAD)
    global SENTINEL_MODE
    SENTINEL_MODE = args.sentinel_mode
    if SENTINEL_MODE != "none":
        n_in += 6''')
replace('                elif teach is not None:\n', '''                elif teach is not None and args.complement_scale:
                    yc = y[:, sl].float().clamp(-2, 2)
                    c = args.complement_scale
                    pred = c * out[..., :2].float() + (1-c) * teach[:, sl].float()
                    w = yc.abs() * main_valid[:, sl][..., None]
                    loss = (w * ((pred - yc) / c) ** 2).sum() / w.sum().clamp_min(1e-8)
                elif teach is not None:
''')
(out / "train_rnn.py").write_text(p)
q = (D / "source/predict_run.py").read_text()
q = q.replace('import argparse', 'import argparse\nimport os')
q = q.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(os.environ["WUNDER_RESEARCH_ROOT"])')
q = q.replace('    train_rnn.RAW_ONLY =', '    train_rnn.SENTINEL_MODE = ck["args"].get("sentinel_mode", "none")\n    train_rnn.RAW_ONLY =')
(out / "predict_run.py").write_text(q)
manifest = {str(f.relative_to(D)): hashlib.sha256(f.read_bytes()).hexdigest()
            for folder in (D / "source", out) for f in folder.glob("*.py")}
(D / "source_hashes.json").write_text(json.dumps(manifest, indent=2))
