"""Run the frozen dynamics grader unchanged, allowlisting TorchVersion for our own checkpoints."""
import runpy
import sys

import torch

torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
