"""Isolated minGRU adapter for the existing parent/H2 recipe.

Source preparation only: no forecasting execution is allocated by this file.
The trainer file and its numerical losses/loaders are not edited. Checkpoints
record arch=mingru; H2 must freeze inp,cells, matching this model's names.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
ARCH_SHA = '16585c0fdcd6cf16b63234c64ea7a02d620a8f705f933fac4324c0a880a234a6'
TRAIN_SHA = 'f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f'


class MinimalParser(argparse.ArgumentParser):
    def add_argument(self, *names, **kwargs):
        if '--arch' in names:
            kwargs['choices'] = [*kwargs['choices'], 'mingru']
        return super().add_argument(*names, **kwargs)

    def parse_args(self, *args, **kwargs):
        value = super().parse_args(*args, **kwargs)
        if value.arch != 'mingru':
            self.error('This isolated adapter requires --arch mingru')
        if not value.name.startswith('oct02_mingru_'):
            self.error('Use a new oct02_mingru_ run name')
        if value.hidden != 256 or value.layers != 2 or value.seed != 3 or value.chunk != 2000:
            self.error('Fixed width256/layers2/seed3/chunk2000 required')
        if value.valid_fold not in (0,1) or value.n_valid_folds != 4:
            self.error('Only original four-way chain folds0/1 are allocated by this recipe')
        if value.enc != 'linear' or value.dropout or value.class_heads:
            self.error('Only the declared linear, dropout0, ordinary-head recipe is supported')
        frozen = set(filter(None, value.freeze.split(',')))
        if frozen and frozen != {'inp', 'cells'}:
            self.error('Head-only training must freeze exactly inp,cells')
        parent = f'oct02_mingru_s3_f{value.valid_fold}'
        expected = parent if not frozen else f'oct02_mingru_s3h2_f{value.valid_fold}'
        if value.name != expected or value.init != (parent+'@ep3' if frozen else ''):
            self.error('Run name and own-fold fixed epoch3 lineage differ')
        if value.epochs != (2 if frozen else 4) or value.batch != (16 if frozen else 32):
            self.error('Fixed parent/H2 epoch and batch counts differ')
        if (ROOT / 'artifacts' / 'runs' / value.name).exists():
            self.error('Refuse to overwrite an existing run')
        return value


def load_trainer():
    path = ROOT / 'research/iterations/2026-10-01-overnight/training/train_aux.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TRAIN_SHA
    assert hashlib.sha256((HERE/'minimal_rnn.py').read_bytes()).hexdigest() == ARCH_SHA
    spec = importlib.util.spec_from_file_location('oct02_minimal_trainer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Replace this module's argparse reference, not global argparse state.
    module.argparse = SimpleNamespace(**{**vars(argparse), 'ArgumentParser': MinimalParser})
    original_factory = module.make_model

    def factory(config, n_in, n_out):
        if config.get('arch') != 'mingru':
            return original_factory(config, n_in, n_out)
        from minimal_rnn import MinimalRNN
        assert config.get('enc', 'linear') == 'linear'
        assert config.get('dropout', 0.) == 0.
        assert not config.get('class_heads')
        if config.get('architecture_source_sha256', ARCH_SHA) != ARCH_SHA:
            raise ValueError('Checkpoint/model source hash mismatch')
        config['architecture_source_sha256'] = ARCH_SHA
        config['recurrent_state'] = 'positive FP32 tensor [layers,batch,hidden]'
        return MinimalRNN(n_in=n_in, hidden=config['hidden'], layers=config['layers'], n_out=n_out)

    module.make_model = factory
    return module


if __name__ == '__main__':
    load_trainer().main()
