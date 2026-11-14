"""All-data final fit launcher (ADR0045 final stage): PR49 trainer, strict IEEE FP32, train + all valid.

The research trainer rejects any fit that includes validation sequences (ADR0044 guard). A final submission
fit is the one sanctioned exception, so this launcher disables that single guard in-process, without touching
the shared trainer source. Runs started here have NO held-out score: validation labels are in the fit.
Usage: final_launch.py PR49_SOURCE_ROOT [train_rnn args...]
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
import wnn.training.setup_lineage as setup_lineage  # noqa: E402

setup_lineage.require_chronological_primary_training = lambda *args, **kwargs: None
script = source / "scripts/train_rnn.py"
sys.argv = [str(script), *sys.argv[2:]]
print("FINAL_ALL_DATA_FIT guard=disabled STRICT_FP32", torch.backends.cuda.matmul.allow_tf32,
      torch.backends.cudnn.allow_tf32, torch.backends.cudnn.rnn.fp32_precision, flush=True)
runpy.run_path(str(script), run_name="__main__")
