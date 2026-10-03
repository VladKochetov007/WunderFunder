from __future__ import annotations

import argparse
import json
import logging
import resource
import time
import traceback

import numpy as np
import torch
from common import PRIOR, REPO, ROOT, donor, dump, parity, raw_window, rms, sha, train


def original_states(
    model: torch.nn.Module, packed: np.ndarray, starts: list[int], stop: int
) -> tuple[np.ndarray, dict[int, torch.Tensor]]:
    cuts = sorted({0, stop, *range(0, stop, 1000), *starts})
    state = torch.zeros(2, 1, 256)
    pred = np.empty((stop, 2), dtype=np.float32)
    states = {}
    inputs = torch.from_numpy(packed)[None]
    wanted = set(starts)
    for s, e in zip(cuts[:-1], cuts[1:], strict=True):
        if s in wanted:
            states[s] = state.clone()
        output, state = model(inputs[:, s:e], state)
        assert torch.isfinite(output).all() and torch.isfinite(state).all()
        pred[s:e] = output[0, :, :2].numpy()
    return pred, states


def main(index: int) -> None:
    started = time.monotonic()
    state = json.loads((ROOT / "STATE.json").read_text())
    for path, expected in json.loads((ROOT / "frozen_sources.json").read_text()).items():
        assert sha(ROOT / path) == expected, path
    panel = json.loads((ROOT / "samples.json").read_text())
    sample = panel["samples"][index]
    prior_provenance = json.loads((PRIOR / "artifact_provenance.json").read_text())
    source = PRIOR / f"seq_{index:02d}.npz"
    assert sha(source) == prior_provenance["artifacts"][source.name]["sha256"]
    assert not (ROOT / f"seq_{index:02d}.json").exists(), "Preserve completed chunk"
    control = panel["control"]
    for path, expected in control["source_sha256"].items():
        assert sha(REPO / path) == expected
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with np.load(source) as z:
        raw, packed, std = z["raw"], z["packed"], z["std"]
    records = []
    for t in sample["queries"]:
        d = donor(raw, t, std)
        if d["lag"] is None:
            records.append(dict(**d, accepted=False, reason="unmatched"))
            continue
        _, _, gate = raw_window(raw, packed, d)
        records.append(dict(**d, **gate))
    # Complete original-origin gate table freezes before prediction values are loaded.
    dump(
        ROOT / f"gates_{index:02d}.json",
        dict(
            sample=sample,
            prior_array_sha256=sha(source),
            records=records,
            feature_atol=0.0,
            feature_rtol=0.0,
        ),
    )
    accepted = [r for r in records if r["accepted"]]
    if not accepted:
        result = dict(
            sample=sample,
            accepted=0,
            rejected=len(records),
            records=records,
            model_forward_executed=False,
            seconds=time.monotonic() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
    else:
        assert json.loads((ROOT / "controls.json").read_text())["passed"]
        assert sha(REPO / control["checkpoint"]) == control["checkpoint_sha256"]
        checkpoint = torch.load(
            REPO / control["checkpoint"], map_location="cpu", weights_only=False
        )
        model = train.make_model(checkpoint["args"], checkpoint["n_in"], control["n_out"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        with np.load(source) as z:
            old_boundaries = z["state_boundaries"]
            old_states = z["original_states"]
            reference = z["reference"]
        lookup = {
            int(s): torch.from_numpy(old_states[j]).clone() for j, s in enumerate(old_boundaries)
        }
        starts = [r["start"] for r in accepted]
        reused = [s for s in starts if s in lookup]
        missing = [s for s in starts if s not in lookup]
        prefix_gate = None
        with torch.inference_mode():
            if missing:
                base, fresh = original_states(
                    model, packed, missing, max(r["query"] for r in accepted) + 1
                )
                prefix_gate = parity(base, reference[: len(base)])
                assert prefix_gate["passed"], prefix_gate
                lookup.update(fresh)
            predictions = []
            identities = []
            hidden = []
            replay = []
            windows = []
            rows = []
            canary = None
            for ai, d in enumerate(accepted):
                t, s = d["query"], d["start"]
                altered, changed, gate = raw_window(raw, packed, d)
                assert gate["accepted"]
                assert np.array_equal(changed[-33:], packed[t - 32 : t + 1])
                original = packed[s : t + 1]
                original_output, original_h = model(
                    torch.from_numpy(original)[None], lookup[s].clone()
                )
                changed_output, changed_h = model(
                    torch.from_numpy(changed)[None], lookup[s].clone()
                )
                assert (
                    torch.isfinite(original_output).all() and torch.isfinite(changed_output).all()
                )
                assert torch.isfinite(original_h).all() and torch.isfinite(changed_h).all()
                identity = parity(original_output[0, :, :2].numpy(), reference[s : t + 1])
                assert identity["passed"], identity
                predictions.append(changed_output[0, -1, :2].numpy())
                identities.append(original_output[0, -1, :2].numpy())
                hidden.append(
                    torch.sqrt(torch.mean((changed_h - original_h) ** 2, dim=(1, 2))).numpy()
                )
                replay.append(
                    np.stack([original_output[0, :, :2].numpy(), changed_output[0, :, :2].numpy()])
                )
                windows.append(changed.copy())
                changed_px = np.r_[False, np.any(raw[1:, :22] != raw[:-1, :22], axis=1)]
                last = np.maximum.accumulate(np.where(changed_px, np.arange(len(raw)), 0))
                rows.append(
                    dict(
                        query=t,
                        identity=identity,
                        raw_displacement=d["amplitude"],
                        query_response_rms=rms(predictions[-1] - identities[-1]),
                        original_quiet_age=int(t - last[t]),
                        seam_original=rms((raw[s, :52] - raw[s - 1, :52]) / std[:52]),
                        seam_altered=rms((altered[s, :52] - altered[s - 1, :52]) / std[:52]),
                        seam_return=rms(
                            (altered[d["end"], :52] - altered[d["end"] - 1, :52]) / std[:52]
                        ),
                    )
                )
                if ai == 0:
                    future = raw.copy()
                    future[s : d["end"], :52] = altered[s : d["end"], :52]
                    future[t + 1 :] = 0
                    future_inputs = train.build_inputs(torch.from_numpy(future)[None])[0].numpy()
                    assert np.array_equal(future_inputs[s : t + 1], changed)
                    actual, _ = model(
                        torch.from_numpy(future_inputs[s : t + 1])[None], lookup[s].clone()
                    )
                    canary = parity(actual[0, :, :2].numpy(), changed_output[0, :, :2].numpy())
                    assert canary["passed"]
        np.savez(
            ROOT / f"seq_{index:02d}.npz",
            queries=np.array([r["query"] for r in accepted]),
            predictions=np.array(predictions),
            identity_predictions=np.array(identities),
            original_reference_query=reference[[r["query"] for r in accepted]],
            hidden_rms=np.array(hidden),
            replay=np.array(replay),
            recomputed_windows=np.array(windows),
            state_boundaries=np.array(starts),
            original_states=np.stack([lookup[s].numpy() for s in starts]),
        )
        result = dict(
            sample=sample,
            accepted=len(accepted),
            rejected=len(records) - len(accepted),
            records=records,
            accepted_results=rows,
            future_canary=canary,
            original_prefix_cpu_reference=prefix_gate,
            model_forward_executed=True,
            original_sequence_forward_count=int(bool(missing)),
            exact_prior_boundary_states_reused=len(reused),
            new_boundary_states=len(missing),
            seconds=time.monotonic() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
    dump(ROOT / f"seq_{index:02d}.json", result)
    state["phase"] = "replaying"
    state["completed_sequences"] = sorted(set(state["completed_sequences"] + [index]))
    dump(ROOT / "STATE.json", state)
    logging.info(
        "sequence%s accepted%s rejected%s seconds%.2f RSS%.1fMiB",
        index,
        result["accepted"],
        result["rejected"],
        result["seconds"],
        result["peak_rss_mib"],
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
        dump(ROOT / "STATE.json", state)
        raise
