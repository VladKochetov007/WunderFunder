"""Run the PR49 canonical trainer (train-only chronological protocol) with strict IEEE FP32 kernels.

Usage: strict_launch.py PR49_SOURCE_ROOT [train_rnn args...]; data root comes from WNN_PROJECT_ROOT.
torch 2.14 defaults cuDNN RNNs to TF32 and the PR49 trainer has no --strict-fp32-kernels flag.
"""

import runpy
import sys
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.rnn.fp32_precision = "ieee"
source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source / "src"))
script = source / "scripts/train_rnn.py"
sys.argv = [str(script), *sys.argv[2:]]
print("STRICT_FP32", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
      torch.backends.cudnn.rnn.fp32_precision, "source", source, flush=True)
runpy.run_path(str(script), run_name="__main__")
