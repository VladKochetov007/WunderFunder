"""Generated-only factory/freeze/parser check; never calls the fitting main path."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from train_minimal import load_trainer

torch.set_num_threads(1)
torch.manual_seed(2042)
here = Path(__file__).resolve().parent
trainer = load_trainer()
config = {'arch': 'mingru', 'hidden': 256, 'layers': 2, 'enc': 'linear', 'dropout': 0., 'class_heads': False}
before = dict(config)
model = trainer.make_model(config, 211, 23)
assert set(config)-set(before) == {'architecture_source_sha256', 'recurrent_state'}
assert all(config[k] == v for k, v in before.items())
assert config['architecture_source_sha256'] == hashlib.sha256((here/'minimal_rnn.py').read_bytes()).hexdigest()
assert sum(p.numel() for p in model.parameters()) == 454679
output, state = model(torch.randn(2, 7, 211))
assert output.shape == (2, 7, 23) and state.shape == (2, 2, 256) and state.dtype == torch.float32
assert torch.isfinite(output).all() and torch.isfinite(state).all() and (state >= 0).all()
for name, param in model.named_parameters():
    if name.startswith(('inp', 'cells')):
        param.requires_grad_(False)
trainable = [name for name, param in model.named_parameters() if param.requires_grad]
assert trainable == ['head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias']
help_result = subprocess.run([sys.executable, str(here/'train_minimal.py'), '--help'], text=True, capture_output=True, timeout=20)
assert help_result.returncode == 0 and 'mingru' in help_result.stdout
print(json.dumps({'status': 'PASS_GENERATED_FACTORY_FREEZE_HELP_ONLY', 'source_sha256': hashlib.sha256((here/'train_minimal.py').read_bytes()).hexdigest(),
                  'architecture_source_sha256': config['architecture_source_sha256'], 'metadata_only_additions': sorted(set(config)-set(before)),
                  'training_parameters': 454679, 'output_shape': list(output.shape), 'state_shape': list(state.shape),
                  'trainable_after_H2_freeze': trainable, 'help_exit_code': help_result.returncode,
                  'data_checkpoint_reads': False, 'forecast_fit_or_metric': False, 'GPU_used': False}), flush=True)
