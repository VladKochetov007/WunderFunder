"""CPU-generated proofs only: actual loss source, payload, RNG and optimizer parity."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / 'training'))
sys.path.insert(0, str(ROOT / 'src'))
from ensemble_support import (PartnerPayload, PartnerSequences, main_prediction, scale_main,
                              state_sha, rng_sha, validate_recipe, sha)
from wnn.nn import RNNModel, weighted_pooled_corr_loss

OLD = ROOT / 'research/iterations/2026-10-01-overnight/training/train_aux.py'
NEW = HERE / 'training/train_ensemble.py'


def load_module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def extracted_loss(source, new):
    start = '                if moments is not None:\n                    ce = F.cross_entropy'
    block = source[source.index(start):source.index('                if args.atom_head > 0:', source.index(start))]
    prefix = 'def loss(out, y, teach, main_valid, partners, args):\n    moments = None\n    sl = slice(None)\n    objective = weighted_pooled_corr_loss\n'
    if new:
        prefix += '    supervised = main_prediction(out[..., :2], partners, args.ensemble_objective)\n'
    code = prefix + textwrap.indent(textwrap.dedent(block), '    ') + '\n    return loss\n'
    scope = dict(torch=torch, weighted_pooled_corr_loss=weighted_pooled_corr_loss,
                 main_prediction=main_prediction, scale_main=scale_main)
    exec(compile(code, '<actual-trainer-main-loss>', 'exec'), scope)
    return scope['loss']


class GeneratedRows(Dataset):
    split = 'train'
    indices = list(range(13))
    def __init__(self, prop):
        self.prop = prop
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        x = torch.full((7, 112), float(i)); y = torch.full((7, 4), float(i + 1))
        items = (x, y, torch.ones(7, dtype=torch.bool))
        return items + (torch.full((7,), 0.25),) if self.prop else items


class GeneratedPartner:
    def for_group(self, split, group):
        return np.full((7, 3, 2), group + 10, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); a = ap.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(1); torch.manual_seed(20261001)
    old, new = OLD.read_text(), NEW.read_text()
    original, candidate = extracted_loss(old, False), extracted_loss(new, True)
    # Every original function except main has identical AST, including feature building and prefetch.
    functions = lambda s: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(s).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    f0, f1 = functions(old), functions(new)
    assert {k: v for k, v in f0.items() if k != 'main'} == {k: v for k, v in f1.items() if k != 'main'}
    # The complete auxiliary + own EMA addition section is literally unchanged.
    aux = lambda s: s[s.index('                if args.atom_head > 0:', s.index('                if moments is not None:\n                    ce =')):s.index('                timer.mark("loss")')]
    assert aux(old) == aux(new)
    checks = []
    torch.manual_seed(31)
    y = torch.randn(3, 11, 2) * 2; teacher = torch.randn_like(y); valid = torch.rand(3, 11)
    partners = torch.randn(3, 11, 3, 2, requires_grad=True)
    for kind in ('distill', 'gcorr'):
        args = argparse.Namespace(ensemble_objective='control', distill_alpha=.5, td_alpha=0, loss_norm='batch', loss='gcorr')
        t = teacher if kind == 'distill' else None
        m0 = RNNModel(211, hidden=16, layers=1, n_out=23); m1 = copy.deepcopy(m0)
        x = torch.randn(3, 11, 211)
        o0 = torch.optim.AdamW(m0.parameters(), lr=.001, weight_decay=.01)
        o1 = torch.optim.AdamW(m1.parameters(), lr=.001, weight_decay=.01)
        for _ in range(2):
            a0 = m0(x)[0]; a1 = m1(x)[0]
            # Add own-output consistency and auxiliary stand-ins identically; source block equality above binds actual terms.
            extra0 = .3 * a0[..., 2:].square().mean() + (a0[..., :2] - .2).square().mean()
            extra1 = .3 * a1[..., 2:].square().mean() + (a1[..., :2] - .2).square().mean()
            l0 = original(a0, y, t, valid, partners, args) + extra0
            l1 = candidate(a1, y, t, valid, partners, args) + extra1
            assert torch.equal(l0, l1)
            o0.zero_grad(); o1.zero_grad(); l0.backward(); l1.backward()
            assert all(torch.equal(p.grad, q.grad) for p, q in zip(m0.parameters(), m1.parameters()))
            torch.nn.utils.clip_grad_norm_(m0.parameters(), 1.0); torch.nn.utils.clip_grad_norm_(m1.parameters(), 1.0)
            o0.step(); o1.step(); assert state_sha(m0) == state_sha(m1)
        assert partners.grad is None
        # Four times aggregate loss gives original main output gradient when anchors equal own detached p.
        p = (torch.randn(3, 11, 2) / 4).requires_grad_(); same = p.detach().unsqueeze(-2).expand(-1, -1, 3, -1).clone().requires_grad_()
        l0 = original(p, y, t, valid, same, args); g0, = torch.autograd.grad(l0, p)
        args.ensemble_objective = 'candidate'
        l1 = candidate(p, y, t, valid, same, args)
        g1, partner_grad = torch.autograd.grad(l1, (p, same), allow_unused=True)
        assert torch.allclose(g0, g1, atol=2e-7, rtol=2e-6), (kind, (g0-g1).abs().max().item())
        assert partner_grad is None
        # Independent explicit aggregate oracle with arbitrary anchors; no clipping or altered target-derived weights.
        e = (((p.float() + partners[..., 0, :].detach()) + partners[..., 1, :].detach()) + partners[..., 2, :].detach()) / 4
        expected = 4 * original(e, y, t, valid, partners, args)
        observed = candidate(p, y, t, valid, partners, args)
        assert torch.equal(expected, observed)
        checks.append(kind + '_source_control_loss_gradient_AdamW_and_candidate_gradient')
    p = torch.tensor([[[1e8, 20.0]]]); anchors = torch.tensor([[[[-1e8, 20.0], [1., 20.0], [1., 20.0]]]])
    blend = main_prediction(p, anchors, 'candidate')
    assert torch.equal(blend, torch.tensor([[[.5, 20.0]]]))
    assert not torch.equal(blend, (p + anchors.sum(-2)) / 4)
    checks.append('ordered_float32_nonassociation_and_no_training_clip')
    # Appending payload cannot change target/propensity indexing or sampler RNG/order.
    old_module = load_module(OLD, 'frozen_original_for_generated_normalization')
    for prop in (False, True):
        base = GeneratedRows(prop); wrapped = PartnerSequences(base, GeneratedPartner())
        def trace(ds):
            torch.manual_seed(123); m = RNNModel(211, hidden=16, layers=1, n_out=23)
            initial = state_sha(m); pre = rng_sha()
            loader = DataLoader(ds, batch_size=3, shuffle=True, drop_last=True, num_workers=0)
            old_module.fit_normalization(m, loader, torch.device('cpu'))
            normalized = state_sha(m)
            ids = []
            for batch in loader:
                ids.extend(batch[0][:, 0, 0].tolist())
                if ds is wrapped:
                    x, yy, mask, *rest = batch; pp = rest.pop()
                    assert yy.shape[-1] == 4 and pp.shape == (3, 7, 3, 2)
                    assert len(rest) == int(prop)
            return initial, pre, normalized, ids, rng_sha()
        assert trace(base) == trace(wrapped)
    checks.append('dedicated_payload_targets_propensity_normalization_sampler_rng_parity')
    # Exercise lazy mmap reads, arbitrary raw-group mapping, held fold rejection and NaN rejection with tiny files.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); reader = PartnerPayload.__new__(PartnerPayload)
        reader.root = root; reader.maps = {'train': {9: 1}, 'valid': {12: 0, 13: 1}}
        reader.allowed_valid = {0}; reader.records = {}; reader.arrays = {}
        for split in ('train', 'valid'):
            reader.records[split] = []
            for member in range(3):
                pth = root / f'{split}{member}.npy'; arr = np.full((2, 20000, 2), member, np.float32); arr[1] += 10
                np.save(pth, arr); st = pth.stat()
                reader.records[split].append(dict(path=pth.name, shape=list(arr.shape), dtype='float32', stat=dict(resolved=str(pth.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino, device=st.st_dev)))
        assert np.array_equal(reader.for_group('train', 9)[0, :, 0], [10, 11, 12])
        assert np.array_equal(reader.for_group('valid', 12)[0, :, 0], [0, 1, 2])
        try: reader.for_group('valid', 13)
        except AssertionError: pass
        else: raise AssertionError('held fold accepted')
        reader.arrays['train'] = [np.full((2, 20000, 2), np.nan, np.float32)] * 3
        try: reader.for_group('train', 9)
        except AssertionError: pass
        else: raise AssertionError('nonfinite payload accepted')
    checks.append('lazy_readonly_mmap_rowmap_fold_finite_guards')
    plan = json.loads((HERE / 'commands.json').read_text())
    for job in plan['jobs']:
        validate_recipe(argparse.Namespace(**job['resolved_args']), ROOT)
    checks.append('all_eight_exact_105_argument_recipes')
    # Real fixed model construction and all normalization buffers retain the ordinary RNG path.
    torch.manual_seed(3); ordinary = RNNModel(211, 256, 1, n_out=23); before = rng_sha(); digest = state_sha(ordinary)
    assert before == rng_sha() and ordinary.mean.numel() == 211
    result = dict(status='PASS_GENERATED_CPU_ONLY', checks=checks, model_initial_sha256=digest,
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), NEW, OLD, HERE / 'training/ensemble_support.py', HERE / 'commands.json']},
                  training_data_read=False, checkpoint_read=False, GPU_used=False, forecast_fits=0,
                  caveat='CPU synthetic parity does not establish CUDA determinism, production memory, or predictive benefit.')
    with a.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
