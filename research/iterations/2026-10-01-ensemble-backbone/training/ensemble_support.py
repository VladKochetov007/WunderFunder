"""Fixed study helpers: payload is distinct from labels; no random operations."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import torch
from torch.utils.data import Dataset

BASE = 'research/iterations/2026-10-01-continuation/student/commands.json'
BASE_SHA = 'f0804cda8b2dde75490496e2014898352ce205fe54837f8c7cbcdfa7823ce30a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def validate_recipe(args, root):
    """Refuse any settings outside the eight prospectively fixed jobs."""
    assert sha(root / BASE) == BASE_SHA
    m = re.fullmatch(r'eb_(control|candidate)_s3_(parent|h2)_f([01])', args.name)
    assert m is not None
    arm, stage, fold = m.group(1), m.group(2), int(m.group(3))
    assert args.ensemble_objective == arm and args.valid_fold == fold
    base = json.loads((root / BASE).read_text())
    job = next(j for j in base['jobs'] if j['fold'] == fold and j['stage'] == ('parent' if stage == 'parent' else 'H2'))
    expected = dict(job['resolved_args'], name=args.name)
    if stage == 'h2':
        expected['init'] = f'eb_{arm}_s3_parent_f{fold}@ep3'
    expected.update(ensemble_objective=arm, partner_bindings=f'research/iterations/2026-10-01-ensemble-backbone/partner-bindings_f{fold}.json')
    assert vars(args) == expected, {k: (vars(args).get(k), v) for k, v in expected.items() if vars(args).get(k) != v}


def path(root, rel):
    p = Path(rel)
    assert not p.is_absolute() and '..' not in p.parts
    return root / p  # Existing, declared read-only shared-input symlinks are permitted.


def check_stat(p, expected):
    s = p.stat()
    actual = dict(resolved=str(p.resolve()), size=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev)
    assert actual == expected, (str(p), actual, expected)


class PartnerPayload:
    """Small provenance reads now; prediction mmaps opened lazily in each worker."""
    def __init__(self, binding_path, root, fold):
        self.root, self.binding_path = Path(root), Path(binding_path)
        self.binding_bytes = self.binding_path.read_bytes()
        self.binding_sha256 = hashlib.sha256(self.binding_bytes).hexdigest()
        b = json.loads(self.binding_bytes)
        assert b['schema'] == 'ensemble-backbone-partners-v1' and b['fold'] == fold
        order = [f'{n}_f{fold}' for n in ('on_ctl128s3', 'on_ctl128s4', 'on_ctlc500')]
        assert b['partner_order'] == order and [m['run'] for m in b['members']] == order
        self.maps, self.records, self.arrays = {}, {}, {}
        for split, n in [('train', 10606), ('valid', 1873)]:
            rec = b['row_maps'][split]
            p = path(self.root, rec['path'])
            check_stat(p, rec['stat'])
            if rec.get('sha256') is not None:
                assert sha(p) == rec['sha256']
            with np.load(p, allow_pickle=False) as z:
                groups = z['row_group']
            assert groups.shape == (n,) and np.issubdtype(groups.dtype, np.integer)
            assert rec['rows'] == n and len(np.unique(groups)) == n
            assert hashlib.sha256(groups.tobytes()).hexdigest() == rec['row_group_sha256']
            if split == 'train':
                assert np.array_equal(np.sort(groups), np.delete(np.arange(10607), 528))
                assert rec['excluded'] == [528]
            self.maps[split] = {int(g): i for i, g in enumerate(groups)}
            self.records[split] = [m[split] for m in b['members']]
        cp = path(self.root, b['chain_folds']['path'])
        assert sha(cp) == b['chain_folds']['sha256']
        with np.load(cp, allow_pickle=False) as cf:
            is_valid = cf['split'] == 'valid'
            fold_map = dict(zip(cf['row_group'][is_valid].tolist(), cf['fold'][is_valid].tolist()))
        assert set(fold_map.values()) <= {0, 1, 2, 3}
        self.allowed_valid = set(b['training_valid_positions'])
        expected_valid = {p for g, p in self.maps['valid'].items() if int(fold_map[g]) % 4 != fold}
        assert self.allowed_valid == expected_valid
        for member in b['members']:
            ck = member['checkpoint']
            assert ck['epoch'] == 1 and sha(path(self.root, ck['path'])) == ck['sha256']
            for split, n in [('train', 10606), ('valid', 1873)]:
                rec = member[split]
                assert rec['shape'] == [n, 20000, 2] and rec['dtype'] == 'float32'
                assert len(rec['sha256']) == 64
                check_stat(path(self.root, rec['path']), rec['stat'])
                assert sha(path(self.root, rec['receipt_path'])) == rec['receipt_sha256']
        rec = b['source_input_receipt']
        assert sha(path(self.root, rec['path'])) == rec['sha256']

    def __getstate__(self):
        return dict(self.__dict__, arrays={})

    def for_group(self, split, raw_group):
        pos = self.maps[split][raw_group]
        assert split != 'valid' or pos in self.allowed_valid, 'held outer fold in partner payload'
        if split not in self.arrays:
            arrays = []
            for rec in self.records[split]:
                p = path(self.root, rec['path'])
                check_stat(p, rec['stat'])
                a = np.load(p, mmap_mode='r', allow_pickle=False)
                assert list(a.shape) == rec['shape'] and a.dtype == np.float32 and not a.flags.writeable
                arrays.append(a)
            self.arrays[split] = arrays
        result = np.stack([a[pos] for a in self.arrays[split]], axis=1)
        assert result.shape == (20000, 3, 2) and result.dtype == np.float32
        assert np.isfinite(result).all(), (split, raw_group, 'nonfinite partner')
        return result


class PartnerSequences(Dataset):
    def __init__(self, base, payload):
        self.base, self.payload = base, payload

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        original = self.base[i]
        partner = self.payload.for_group(self.base.split, int(self.base.indices[i]))
        return (*original, torch.from_numpy(partner))


def main_prediction(own, partner, mode):
    if mode == 'control':
        return own  # Preserve original dtype and exact standalone loss path.
    assert mode == 'candidate' and partner.shape == (*own.shape[:-1], 3, 2)
    assert partner.dtype == torch.float32
    a = partner.detach()
    return (((own.float() + a[..., 0, :]) + a[..., 1, :]) + a[..., 2, :]) / 4.0


def scale_main(loss, mode):
    return loss if mode == 'control' else 4.0 * loss


def state_sha(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_sha():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def row_order_sha(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def emit_proof(out_dir, record):
    with (out_dir / 'trajectory-proof.jsonl').open('a') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')
    print(json.dumps({'trajectory_proof': record}), flush=True)
