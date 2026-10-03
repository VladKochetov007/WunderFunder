from __future__ import annotations

import argparse
import importlib
import json
import logging
import resource
import time
import traceback

import numpy as np
import torch
from common import PRIOR, REPO, ROOT, dump, sha, signal_features

train = importlib.import_module("train_rnn")


def main(index: int) -> None:
    started = time.monotonic()
    for path, expected in json.loads((ROOT / "frozen_sources.json").read_text()).items():
        assert sha(ROOT / path) == expected, path
    panel = json.loads((ROOT / "samples.json").read_text())
    sample = panel["samples"][index]
    control = panel["control"]
    assert not (ROOT / f"seq_{index:02d}.json").exists(), "Preserve completed extraction"
    manifest = json.loads((PRIOR / "artifact_provenance.json").read_text())
    source = PRIOR / f"seq_{index:02d}.npz"
    assert sha(source) == manifest["artifacts"][source.name]["sha256"]
    for path, expected in control["source_sha256"].items():
        assert sha(REPO / path) == expected
    assert sha(REPO / control["checkpoint"]) == control["checkpoint_sha256"]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    ckpt = torch.load(REPO / control["checkpoint"], map_location="cpu", weights_only=False)
    model = train.make_model(ckpt["args"], ckpt["n_in"], control["n_out"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    with np.load(source) as z:
        raw, packed, std = z["raw"], z["packed"], z["std"]
        boundaries, states = z["state_boundaries"], z["original_states"]
        original_replay = z["replay"][:, 0]
    own, peer, age = signal_features(raw, std)
    ledger = []
    features = []
    peers = []
    shifts = []
    outcomes = []
    queries = []
    ages = []
    hs = []
    max_error = 0.0
    canary_count = 0
    with torch.inference_mode():
        for qi, t in enumerate(sample["queries"]):
            entry = dict(
                component=sample["component"],
                seq_ix=sample["seq_ix"],
                row_group=sample["row_group"],
                query=t,
                query_pi=sample["query_pi"],
                original_inverse_query_probability=1 / sample["query_pi"],
                split="fit" if sample["component"] in panel["fit_components"] else "test",
                future_available=t + 32 < len(raw),
                shift_history_available=t >= 256,
            )
            if not entry["future_available"]:
                entry.update(eligible=False, reason="missing_future_end")
                ledger.append(entry)
                continue
            s = t - 288
            boundary = int(np.flatnonzero(boundaries == s)[0])
            output, h = model(
                torch.from_numpy(packed[s : t + 1])[None],
                torch.from_numpy(states[boundary]).clone(),
            )
            error = float(np.abs(output[0, :, :2].numpy() - original_replay[qi]).max())
            max_error = max(max_error, error)
            assert np.allclose(output[0, :, :2].numpy(), original_replay[qi], atol=2e-5, rtol=2e-5)
            assert torch.isfinite(h).all()
            # Features only use <=t. This future observable is read only as proxy supervision.
            x = np.r_[packed[t], own[t], h.numpy().reshape(-1)]
            y = float((float(raw[t + 32, 0]) - float(raw[t, 0])) / std[0])
            finite = bool(np.isfinite(x).all() and np.isfinite([peer[t], peer[t - 256], y]).all())
            entry.update(eligible=finite, reason="eligible" if finite else "nonfinite")
            ledger.append(entry)
            if not finite:
                continue
            features.append(x)
            peers.append(peer[t])
            shifts.append(peer[t - 256])
            outcomes.append(y)
            queries.append(t)
            ages.append(age[t])
            hs.append(h.numpy())
            if canary_count == 0:
                future = raw.copy()
                future[t + 1 :] = 0
                own_future, peer_future, age_future = signal_features(future, std)
                assert np.array_equal(own[: t + 1], own_future[: t + 1])
                assert np.array_equal(peer[: t + 1], peer_future[: t + 1])
                assert np.array_equal(age[: t + 1], age_future[: t + 1])
                rebuilt = train.build_inputs(torch.from_numpy(future)[None])[0].numpy()
                assert np.array_equal(rebuilt[s : t + 1], packed[s : t + 1])
                again, again_h = model(
                    torch.from_numpy(rebuilt[s : t + 1])[None],
                    torch.from_numpy(states[boundary]).clone(),
                )
                assert np.array_equal(again_h.numpy(), h.numpy())
                assert np.array_equal(again.numpy(), output.numpy())
                canary_count += 1
    np.savez(
        ROOT / f"seq_{index:02d}.npz",
        base=np.array(features),
        peer=np.array(peers),
        shift=np.array(shifts),
        target=np.array(outcomes),
        queries=np.array(queries),
        peer_age=np.array(ages),
        gru_state=np.array(hs),
    )
    result = dict(
        sample=sample,
        ledger=ledger,
        eligible=len(queries),
        future_end_missing=sum(not r["future_available"] for r in ledger),
        nonfinite=sum(r["reason"] == "nonfinite" for r in ledger),
        shift_history_missing=sum(not r["shift_history_available"] for r in ledger),
        state_replay_queries=len(queries),
        model_prefix_forwards=0,
        identity_max_abs=max_error,
        future_canaries=canary_count,
        causal_feature_equality=True,
        no_competition_target_members_loaded=True,
        source_array_sha256=sha(source),
        seconds=time.monotonic() - started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    dump(f"seq_{index:02d}.json", result)
    state = json.loads((ROOT / "STATE.json").read_text())
    state.update(
        phase="extracting", completed_sequences=sorted(set(state["completed_sequences"] + [index]))
    )
    dump("STATE.json", state)
    logging.info(
        "sequence%s eligible%s end_missing%s seconds%.2f identity%.3g",
        index,
        result["eligible"],
        result["future_end_missing"],
        result["seconds"],
        max_error,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, choices=range(24), required=True)
    arg = parser.parse_args()
    try:
        main(arg.sequence)
    except Exception:
        error = traceback.format_exc()
        (ROOT / f"failure_{arg.sequence:02d}.txt").write_text(error)
        state = json.loads((ROOT / "STATE.json").read_text())
        state.update(phase="failure", failure=error)
        dump("STATE.json", state)
        raise
