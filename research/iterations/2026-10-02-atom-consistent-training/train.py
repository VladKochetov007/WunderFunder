"""Isolated original trainer adapter; no losses, schedule or batches changed."""
from __future__ import annotations
import hashlib, json, sys, types
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(HERE))
from model import ConsistentAtomRNNModel
BASE = ROOT / 'scripts/train_rnn.py'
BASE_SHA = 'f97047e6ef51f32097f2cbe2eefc7c57766771ebaed12ae8798a3c1b7efcd52f'
assert hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA

def state_sha(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def generated_source():
    source = BASE.read_text()
    needle = '    args = ap.parse_args()'
    assert source.count(needle) == 1
    return source.replace(needle, '    ap.add_argument("--atom-decoder", choices=["original", "corrected"], required=True)\n' + needle)

base = types.ModuleType('atom_consistency_base')
base.__file__ = str(BASE)
exec(compile(generated_source(), str(BASE), 'exec'), base.__dict__)
MODELS = []
ORDER_RECORDS = []
NORMALIZED_INITIAL = []

def make_model(args, n_in, n_out):
    assert args['atom_head'] == .5 and args['hidden'] == 128 and args['layers'] == 2
    assert args['arch'] == 'gru' and args['aux_struct'] == .3 and not args['aux']
    binding = json.loads((HERE / 'original-metadata.json').read_text())
    atoms = np.asarray(binding['atoms'], dtype=np.float32)
    assert atoms.shape == (64, 2) and np.isfinite(atoms).all()
    assert hashlib.sha256(atoms.tobytes()).hexdigest() == binding['atom_tensor_sha256']
    start = 2 + base.N_SP_AUX + 3
    model = ConsistentAtomRNNModel(n_in, 128, 2, args['dropout'], n_out, (start, start + 4), atoms,
                                   atom_decoder=args['atom_decoder'])
    MODELS.append(dict(model=model, initial_sha256=state_sha(model.state_dict())))
    return model


def skip_evaluation(model, loader, chunk, device):
    # Nonshuffled DataLoader iteration consumes exactly this CPU base-seed draw.
    torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
    return dict(weighted_pearson=0., evaluation_skipped=True, metrics_computed=0,
                metric_placeholder_not_a_readout=True, reason='placeholder only: prospectively fixed parent ep3/head ep0')

class AuditedSampler:
    def __init__(self, sampler): self.sampler = sampler
    def __len__(self): return len(self.sampler)
    def __iter__(self):
        record = dict(indices=[], complete=False)
        ORDER_RECORDS.append(record)
        for index in self.sampler:
            record['indices'].append(int(index))
            yield index
        record['complete'] = True

original_loader = base.DataLoader
def audited_loader(*args, **kwargs):
    loader = original_loader(*args, **kwargs)
    if loader.drop_last:
        # Wrap existing sampler, preserving its generator and exact lazy iteration.
        loader.batch_sampler.sampler = AuditedSampler(loader.batch_sampler.sampler)
    return loader

original_normalize = base.fit_normalization
def audited_normalize(model, *args, **kwargs):
    result = original_normalize(model, *args, **kwargs)
    NORMALIZED_INITIAL.append(state_sha(model.state_dict()))
    return result

base.DataLoader = audited_loader
base.fit_normalization = audited_normalize
base.make_model = make_model
base.evaluate = skip_evaluation

def main():
    plan = json.loads((HERE / 'commands.json').read_text())
    argv = sys.argv[1:]
    name = argv[argv.index('--name') + 1]
    job = next(j for j in plan['jobs'] if j['name'] == name and j['stage'] != 'OOF')
    assert argv == job['argv'][2:], 'Only fixed registered commands allowed'
    out = ROOT / 'artifacts/runs' / name
    assert name.startswith('oct02_ac_') and not out.exists() and not out.is_symlink()
    base.main()
    ckpath = ROOT / job['endpoint']
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    assert ck['args'] == job['resolved_args'] and ck['epoch'] == job['resolved_args']['epochs'] - 1
    assert all(torch.isfinite(v).all() for v in ck['model'].values())
    # Original trainer's placeholder-selected alias would stay ep0; make alias fixed-final.
    torch.save({k:v for k,v in ck.items() if k != 'epoch'}, out / 'model.pt')
    frozen = None
    if job['stage'] == 'H2':
        parent_name = ck['args']['init'].split('@ep')[0]
        parent = torch.load(ROOT / 'artifacts/runs' / parent_name / 'model_ep3.pt', map_location='cpu', weights_only=False)
        frozen = all(torch.equal(v, parent['model'][k]) for k, v in ck['model'].items()
                     if k.startswith(('inp.', 'gru.')) or k in ('mean', 'std', 'atoms'))
        assert frozen
    receipt = dict(status='COMPLETE_FIXED_ENDPOINT', run=name, decoder=ck['args']['atom_decoder'],
                   epoch=ck['epoch'], checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),
                   initial_state_sha256=MODELS[0]['initial_sha256'], normalized_initial_sha256=NORMALIZED_INITIAL,
                   sampler_iterations=[dict(count=len(r['indices']), complete=r['complete'], sha256=hashlib.sha256(np.asarray(r['indices'], dtype='<i8').tobytes()).hexdigest()) for r in ORDER_RECORDS], frozen_trunk_exact=frozen,
                   source_base_sha256=BASE_SHA, metrics_computed=0,
                   atom_tensor_sha256=hashlib.sha256(ck['model']['atoms'].numpy().tobytes()).hexdigest())
    with (out / 'atom-receipt.json').open('x') as f: json.dump(receipt, f, indent=2)
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__': main()
