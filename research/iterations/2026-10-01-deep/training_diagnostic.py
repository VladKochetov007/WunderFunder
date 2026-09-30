"""Bounded training-side head-gradient audit; no optimizer updates or evaluation WP."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
from wnn.data import SequenceFile  # noqa: E402
from wnn.features import event_features_torch  # noqa: E402
from wnn.nn import RNNModel, weighted_pooled_corr_loss  # noqa: E402
from wnn.outcomes import N_SP_AUX, sequence_class, structure_labels  # noqa: E402
from train_rnn import next_change_labels  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino}


def dump_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    inventory = out.with_suffix('.inputs.json')
    if out.exists() or inventory.exists():
        raise FileExistsError('immutable output/inventory already exists')
    started = time.monotonic()
    result = {'status': 'started', 'optimizer_updates': 0, 'held_out_WP_queries': 0,
              'windows': [], 'precision': 'CPU float32, not AMP replay', 'selected': []}

    def stop(signum, frame):
        raise TimeoutError(f'diagnostic interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(450)
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        root = args.source_root
        checkpoint = root / 'artifacts/runs/ref_f0/model_ep3.pt'
        cfpath = root / 'cache/chain_folds.npz'
        clpath = root / 'research/artifacts/market-classes-v1.npz'
        datapath = root / 'datasets/valid.parquet'
        cf = np.load(cfpath)
        classes = np.load(clpath)['valid']
        eligible = (cf['split'] == 'valid') & (cf['fold'] != 0) & cf['pure']
        records = [(int(c), int(g), int(f)) for c, g, f in
                   zip(cf['chain'][eligible], cf['row_group'][eligible], cf['fold'][eligible])]
        chosen = []
        used = set()
        for k in range(4):
            rows = [r for r in records if int(classes[r[1]]) == k]
            rows.sort(key=lambda r: (hashlib.sha256(f'TDIAG1:{r[0]}'.encode()).hexdigest(), r[1]))
            count = 0
            for c, g, f in rows:
                if c in used:
                    continue
                chosen.append({'chain': c, 'row_group': g, 'fold': f, 'class': k})
                used.add(c)
                count += 1
                if count == 4:
                    break
            if count != 4:
                raise ValueError(f'insufficient distinct chains for class {k}')
        result['selected'] = chosen
        before = {str(p): fingerprint(p) for p in (checkpoint, cfpath, clpath, datapath)}
        ds = SequenceFile(datapath)
        xs, ys, masks, row_hashes = [], [], [], []
        for row in chosen:
            seq = ds.read(row['row_group'])
            if len(seq.features) != 20000 or seq.scored is None:
                raise ValueError('unexpected sequence length or missing scored mask')
            h = hashlib.sha256()
            for arr in (seq.features[:4000], seq.targets, seq.scored):
                a = np.ascontiguousarray(arr)
                h.update(str(a.dtype).encode())
                h.update(str(a.shape).encode())
                h.update(a.tobytes())
            row_hashes.append({'row_group': row['row_group'], 'sha256_used_arrays': h.hexdigest()})
            xs.append(seq.features[:4000].copy())
            ys.append(seq.targets.copy())
            masks.append(seq.scored.copy())
        inv = {'script_sha256': digest(__file__), 'plan_sha256': digest(Path(__file__).with_name('training-plan.json')),
               'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
               'chain_folds_sha256': digest(cfpath), 'classes_sha256': digest(clpath),
               'source_fingerprints': before, 'selected_used_array_hashes': row_hashes,
               'selected': chosen, 'data_role': 'ref_f0/head-FT training rows; none is fold0 evaluation',
               'code_sha256': {str(p.relative_to(REPO)): digest(p) for p in
                               [REPO/'src/wnn/nn.py', REPO/'src/wnn/features.py',
                                REPO/'src/wnn/outcomes.py', REPO/'scripts/train_rnn.py']}}
        dump_new(inventory, inv)
        result['inventory_sha256'] = digest(inventory)
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = ck['args']
        forbidden = ['online_norm', 'heuristics', 'raw_only', 'quant_feats', 'quant_i1', 'class_heads', 'lookahead']
        if any(cfg.get(k) for k in forbidden) or cfg.get('arch', 'gru') != 'gru' or cfg.get('enc', 'linear') != 'linear':
            raise ValueError('unsupported parent recipe; no substitute')
        if cfg.get('dropout', 0) != 0 or cfg.get('aux_struct') != .3 or cfg.get('aux_next') != .3:
            raise ValueError('parent dropout/auxiliary recipe differs')
        model = RNNModel(ck['n_in'], cfg['hidden'], cfg['layers'], 0,
                         n_out=ck['model']['head.2.weight'].shape[0])
        model.load_state_dict(ck['model'], strict=True)
        if model.head[2].out_features != 23:
            raise ValueError('expected 2+12+3+4+2 outputs')
        model.eval()
        result['model_mode'] = 'eval; identical to train for this dropout=0 linear-encoder GRU/head'
        result['parent_eval_ema'] = bool(cfg.get('eval_ema'))
        features = []
        with torch.no_grad():
            for start in range(0, 16, 4):
                x = torch.from_numpy(np.stack(xs[start:start+4]))
                inp = torch.cat([x, event_features_torch(x)], -1)
                z = model.encode((inp-model.mean)/model.std)
                recurrent, _ = model.gru(z)
                features.append(torch.cat([recurrent, z], -1))
        features = torch.cat(features)
        y = torch.from_numpy(np.stack(ys))
        valid = torch.arange(20000)[None].expand(16, -1) >= 99
        mask = valid & torch.from_numpy(np.stack(masks)).bool()
        sp, br, cl = structure_labels(y, sequence_class(y))
        nr, nu, nok = next_change_labels(y, valid)
        params = list(model.head.parameters())
        shared_size = params[0].numel() + params[1].numel()
        for a, b in ((0, 2000), (2000, 4000)):
            p = model.head(features[:, a:b])
            m = mask[:, a:b].float()
            nw = (mask & nok)[:, a:b].float()
            ce = sum(F.cross_entropy(p[..., lo:hi].reshape(-1, hi-lo), lab[:, a:b].reshape(-1),
                                     reduction='none').reshape(16, b-a)
                     for lo, hi, lab in ((2, 14, sp), (14, 17, br), (17, 21, cl)))
            ls = .3*(ce*m).sum()/m.sum().clamp_min(1e-8)
            ln = .3*((p[..., 21]-(nr[:, a:b]-3)/2).square()+
                      F.binary_cross_entropy_with_logits(p[..., 22], nu[:, a:b], reduction='none'))
            ln = (ln*nw).sum()/nw.sum().clamp_min(1e-8)
            lm = weighted_pooled_corr_loss(p[..., :2], y[:, a:b], m)
            grads = [torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, params, retain_graph=True)])
                     for loss in (lm, ls, ln)]
            gm, gs, gn = grads
            ga = gs+gn
            hidden = model.head[2].in_features
            output_bias_at = shared_size + 23*hidden
            def private_main(g):
                return torch.cat([g[shared_size:shared_size+2*hidden], g[output_bias_at:output_bias_at+2]])
            def private_aux(g):
                return torch.cat([g[shared_size+2*hidden:output_bias_at], g[output_bias_at+2:]])
            def norm(g): return float(g.norm())
            def cosine(g, h):
                den = norm(g)*norm(h)
                return None if den == 0 else float(torch.dot(g, h)/den)
            yc = y[:, a:b].clamp(-2, 2)
            w = yc.abs()*m[..., None]
            rec = {'rows': [a, b], 'scored_count': int(m.sum()), 'next_label_count': int(nw.sum()),
                   'shared_cosine_main_struct': cosine(gm[:shared_size], gs[:shared_size]),
                   'shared_cosine_main_next': cosine(gm[:shared_size], gn[:shared_size]),
                   'shared_cosine_main_aux': cosine(gm[:shared_size], ga[:shared_size]),
                   'norms_all_head': dict(zip(('main', 'struct_scaled', 'next_scaled'), map(norm, grads))),
                   'norms_shared_head': dict(zip(('main', 'struct_scaled', 'next_scaled'),
                                                [norm(g[:shared_size]) for g in grads])),
                   'norms_private_main_rows': dict(zip(('main', 'struct_scaled', 'next_scaled'),
                                                       [norm(private_main(g)) for g in grads])),
                   'norms_private_aux_rows': dict(zip(('main', 'struct_scaled', 'next_scaled'),
                                                      [norm(private_aux(g)) for g in grads])),
                   'clip_factor_control': min(1., 1/(norm(gm+ga)+1e-6)),
                   'clip_factor_main_only': min(1., 1/(norm(gm)+1e-6)),
                   'scored_fraction_abs_prediction_gt2': float(((p[..., :2].detach().abs()>2)*m[..., None]).sum()/
                                                              (2*m.sum()).clamp_min(1)),
                   'metric_weight_fraction_abs_prediction_gt2': float((w*(p[..., :2].detach().abs()>2)).sum()/w.sum().clamp_min(1e-8)),
                   'metric_mass_by_class': [float(w[j*4:(j+1)*4].sum()) for j in range(4)],
                   'aux_mask_mass_by_class': [float(m[j*4:(j+1)*4].sum()) for j in range(4)]}
            if not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('nonfinite gradient')
            result['windows'].append(rec)
        if any(fingerprint(p) != fp for p, fp in before.items()):
            raise RuntimeError('input fingerprint changed during audit')
        result['status'] = 'complete'
    except Exception as exc:
        result['status'] = 'timeout' if isinstance(exc, TimeoutError) else 'error'
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        signal.alarm(0)
        result['wall_seconds'] = time.monotonic()-started
        dump_new(out, result)
    if result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
