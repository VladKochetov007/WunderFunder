"""OOF accuracy of a causal market-class detector: fold-k since parent's auxiliary class logits (head outputs
17:21), softmax averaged over rows 50..98 (warm-up only), argmax vs the retrospective class, on fold-k eval rows."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import train_rnn  # noqa: E402
from train_rnn import Sequences, build_inputs, make_model  # noqa: E402
from wnn.splits import load_campaign_split  # noqa: E402
torch.set_num_threads(3)
z = np.load(ROOT / "cache/valid_targets.npz"); cls = z["market_class"]; pos = {int(g): i for i, g in enumerate(z["row_group"])}
conf = np.zeros((4, 4), int); margins = []
for f in (0, 1):
    ck = torch.load(ROOT / f"artifacts/runs/c20260929_since_s3_f{f}/model_ep3.pt", map_location="cpu", weights_only=False)
    m = make_model(ck["args"], ck["n_in"], ck["model"]["head.2.weight"].shape[0]); m.load_state_dict(ck["model"]); m.eval()
    groups = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT).evaluation_groups
    ds = Sequences("valid", list(groups))
    with torch.no_grad():
        for j, g in enumerate(groups):
            x = ds[j][0][:99][None]
            out, _ = m(build_inputs(x))
            p = torch.softmax(out[0, 50:99, 17:21].float(), -1).mean(0).numpy()
            conf[cls[pos[g]], int(p.argmax())] += 1
            margins.append(float(np.sort(p)[-1] - np.sort(p)[-2]))
print("confusion (rows=true, cols=detected):\n", conf)
print("accuracy", conf.trace() / conf.sum(), "class3 recall", conf[3, 3] / conf[3].sum(), "class3 precision", conf[3, 3] / conf[:, 3].sum())
print("min margin", min(margins), "p05 margin", np.percentile(margins, 5))
