"""Run the H390 child trainer copy with strict IEEE FP32 kernels and the h390m Muon contract.

SOURCE_ROOT is a copy of the frozen H390 R2 child source with two reviewed edits (FP32
Newton-Schulz; contract admits optim=muon/precision=fp32 under name vt_all_h390m_s{seed}_r1),
recorded in H390M_SOURCE_PATCH_R1.diff. The child's snapshot verifier is pointed at the pristine
child tree (PRISTINE_ROOT, inside its R2 packet) so the cache/packet identity checks still run
against the frozen bytes; every other check runs unchanged. torch 2.14 defaults cuDNN RNNs to
TF32, so TF32 is disabled before the trainer starts.
Usage: strict_fp32_launch.py SOURCE_ROOT PRISTINE_ROOT [train_rnn args...]
"""

import runpy
import sys
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.rnn.fp32_precision = "ieee"
source, pristine = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(source / "src"))
from wnn import final_distillation_cache  # noqa: E402

verify = final_distillation_cache.verify_h390_source_snapshot
final_distillation_cache.verify_h390_source_snapshot = lambda _root: verify(pristine)
script = source / "scripts/train_rnn.py"
sys.argv = [str(script), *sys.argv[3:]]
print("STRICT_FP32", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
      torch.backends.cudnn.rnn.fp32_precision, "pristine_verifier", pristine, flush=True)
runpy.run_path(str(script), run_name="__main__")
