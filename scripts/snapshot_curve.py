"""Score every saved snapshot of a run on the evaluation fold: the learning curve without in-fit evaluation.

``train_rnn.py --intra-snapshots`` saves ``model_ep{E}p{B}.pt`` inside each epoch (plus the per-epoch ``model_ep{E}.pt``).
In-fit evaluation is refused under campaign manifests, so the curve is computed afterwards: each snapshot is linked as
``model_ep0.pt`` of a temporary run directory, scored with ``predict_run.py`` and the official ``global_wp_clip2_v1`` on the
fixed evaluation groups, then the temporary predictions are deleted.  Output: ``artifacts/runs/<run>/snapshot_curve.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from grade_prediction_bank import SCORED_FROM_ROW, Lazy, aligned, score, sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, default=ROOT / "scripts/train_rnn.py")
    args = parser.parse_args()
    run_dir = ROOT / "artifacts/runs" / args.run
    split = json.loads(args.split.read_text())
    groups = [int(g) for g in split["evaluation"]["valid"]]
    split_sha = sha256(args.split)
    with np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False) as cache:
        position = {int(g): i for i, g in enumerate(cache["row_group"])}
        index = [position[g] for g in groups]
        y = np.asarray(cache["y"][index], dtype=np.float32)
        mask = np.asarray(cache["mask"][index], dtype=bool)
    mask &= (np.arange(y.shape[1]) >= SCORED_FROM_ROW)[None, :]
    snapshots = []
    for path in run_dir.glob("model_ep*.pt"):
        match = re.fullmatch(r"model_ep(\d+)(?:p(\d+))?\.pt", path.name)
        if match:
            epoch, batch = int(match[1]), match[2]
            snapshots.append((epoch, int(batch) if batch is not None else 10**6, path))
    snapshots.sort()
    out_path = run_dir / "snapshot_curve.jsonl"
    done = {json.loads(line)["snapshot"] for line in out_path.read_text().splitlines()} if out_path.exists() else set()
    runner = ("import runpy,sys; from pathlib import Path; run_script=sys.argv[1]; trainer=sys.argv[2]; "
              "sys.path.insert(0,str(Path(trainer).resolve().parent)); sys.argv=[run_script,*sys.argv[3:]]; "
              "runpy.run_path(run_script,run_name='__main__')")
    for epoch, batch, path in snapshots:
        if path.name in done:
            continue
        temp = f"_snap_{args.run}_{path.stem}"
        temp_dir = ROOT / "artifacts/runs" / temp
        temp_dir.mkdir(parents=True, exist_ok=True)
        link = temp_dir / "model_ep0.pt"
        link.unlink(missing_ok=True)
        link.symlink_to(path.resolve())
        subprocess.run([sys.executable, "-c", runner, str(ROOT / "scripts/predict_run.py"), str(args.trainer), temp, "valid",
                        "--epoch", "0", "--output-name", temp], check=True, cwd=ROOT)
        prediction = ROOT / "cache" / f"pred_{temp}_valid.npy"
        result = aligned({"prediction": str(prediction.relative_to(ROOT))}, groups, split_sha)
        if isinstance(result, str):
            raise SystemExit(f"{path.name}: {result}")
        array, order = result
        scores = score(y, mask, Lazy(array, order))
        record = {"snapshot": path.name, "epoch": epoch, "batch": None if batch == 10**6 else batch,
                  "epoch_end": batch == 10**6, **scores}
        with out_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        for leftover in (prediction, prediction.with_suffix(".json")):
            leftover.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
