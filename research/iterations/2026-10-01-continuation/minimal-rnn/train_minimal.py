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
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'src'))


class MinimalParser(argparse.ArgumentParser):
    def add_argument(self, *names, **kwargs):
        if '--arch' in names:
            kwargs['choices'] = [*kwargs['choices'], 'mingru']
        return super().add_argument(*names, **kwargs)

    def parse_args(self, *args, **kwargs):
        value = super().parse_args(*args, **kwargs)
        if value.arch != 'mingru':
            self.error('This isolated adapter requires --arch mingru')
        if not value.name.startswith('cont_mingru_'):
            self.error('Use a new cont_mingru_ run name')
        if value.enc != 'linear' or value.dropout or value.class_heads:
            self.error('Only the declared linear, dropout0, ordinary-head recipe is supported')
        frozen = set(filter(None, value.freeze.split(',')))
        if frozen and frozen != {'inp', 'cells'}:
            self.error('Head-only training must freeze exactly inp,cells')
        if (ROOT / 'artifacts' / 'runs' / value.name).exists():
            self.error('Refuse to overwrite an existing run')
        return value


def load_trainer():
    path = ROOT / 'research/iterations/2026-10-01-overnight/training/train_aux.py'
    spec = importlib.util.spec_from_file_location('continuation_minimal_trainer', path)
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
        config['architecture_source_sha256'] = hashlib.sha256((HERE / 'minimal_rnn.py').read_bytes()).hexdigest()
        config['recurrent_state'] = 'positive FP32 tensor [layers,batch,hidden]'
        return MinimalRNN(n_in=n_in, hidden=config['hidden'], layers=config['layers'], n_out=n_out)

    module.make_model = factory
    return module


if __name__ == '__main__':
    load_trainer().main()
