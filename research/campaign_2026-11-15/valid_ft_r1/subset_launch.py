"""final_launch variant: keep only train groups listed in WNN_KEEP_TRAIN (npy) by marking the rest unusable."""
import os, runpy, sys
from pathlib import Path
import numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.rnn.fp32_precision = "ieee"
source = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(source / "src"))
import wnn.training.setup_lineage as setup_lineage  # noqa: E402
import wnn.training.runner as runner  # noqa: E402
setup_lineage.require_chronological_primary_training = lambda *a, **k: None
keep = set(np.load(os.environ["WNN_KEEP_TRAIN"]).tolist())
orig = runner.select_membership
def patched(run):
    env = run.environment
    object.__setattr__(env, "corrupt_train_groups", frozenset(set(env.corrupt_train_groups) | (set(range(10607)) - keep)))
    orig(run)
    print("SUBSET_TRAIN", len(run.data.train_idx), flush=True)
runner.select_membership = patched
script = source / "scripts/train_rnn.py"; sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
