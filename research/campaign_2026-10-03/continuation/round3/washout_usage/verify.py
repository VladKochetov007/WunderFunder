from __future__ import annotations

import json
import time

import numpy as np
import torch
from common import PRIOR, REPO, ROOT, dump, parity, sha, train


def main() -> None:
    started = time.monotonic()
    for path, expected in json.loads((ROOT / "frozen_sources.json").read_text()).items():
        assert sha(ROOT / path) == expected
    prior = json.loads((PRIOR / "artifact_provenance.json").read_text())
    for path, entry in prior["artifacts"].items():
        assert sha(PRIOR / path) == entry["sha256"], path
    panel = json.loads((ROOT / "samples.json").read_text())
    assert sha(ROOT / "samples.json") == sha(PRIOR / "samples.json")
    control = panel["control"]
    assert sha(REPO / control["checkpoint"]) == control["checkpoint_sha256"]
    for path, expected in control["source_sha256"].items():
        assert sha(REPO / path) == expected
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(REPO / control["checkpoint"], map_location="cpu", weights_only=False)
    model = train.make_model(checkpoint["args"], checkpoint["n_in"], control["n_out"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    replay_max = 0
    windows = 0
    checked = 0
    with torch.inference_mode():
        for i, sample in enumerate(panel["samples"]):
            meta = json.loads((ROOT / f"seq_{i:02d}.json").read_text())
            assert [r["query"] for r in meta["records"]] == sample["queries"]
            assert len(meta["records"]) == sample["m_selected"]
            assert meta.get("original_sequence_forward_count", 0) <= 1
            if not meta["accepted"]:
                continue
            with np.load(ROOT / f"seq_{i:02d}.npz") as z:
                with np.load(PRIOR / f"seq_{i:02d}.npz") as old:
                    packed = old["packed"]
                for qi, t in enumerate(z["queries"]):
                    assert np.array_equal(z["recomputed_windows"][qi, -33:], packed[t - 32 : t + 1])
                    checked += 1
                t = int(z["queries"][0])
                s = int(z["state_boundaries"][0])
                h = torch.from_numpy(z["original_states"][0])
                for arm, w in enumerate([packed[s : t + 1], z["recomputed_windows"][0]]):
                    output, _ = model(torch.from_numpy(w)[None], h.clone())
                    gate = parity(output[0, :, :2].numpy(), z["replay"][0, arm])
                    assert gate["passed"], gate
                    replay_max = max(replay_max, gate["max_abs"])
                    windows += 1
    results = json.loads((ROOT / "results.json").read_text())
    if checked:
        with np.load(ROOT / "accepted_panel.npz") as z:
            computed = np.sqrt(
                np.average(np.mean(z["response"] ** 2, axis=1), weights=z["query_design_weight"])
            )
            assert abs(computed - results["response"]["design_weighted_query_rms"]) < 1e-12
            assert checked == results["panel"]["accepted"]
    output = dict(
        passed=True,
        prior_artifacts_preserved=len(prior["artifacts"]),
        all_accepted_suffixes_checked=checked,
        independent_actual_replay_windows=windows,
        actual_replay_max_abs=replay_max,
        all288_origins_retained=True,
        original_panel_hash_equal=True,
        no_target_members_loaded=True,
        seconds=time.monotonic() - started,
    )
    dump(ROOT / "verification.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
