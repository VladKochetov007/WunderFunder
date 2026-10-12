"""Export a trained RNN run as a one-row stateful ONNX graph and verify it.

    scripts/limited .venv/bin/python scripts/export_onnx.py RUN [--out submissions/RUN/model.onnx]
Checks ORT streaming output against batch torch predictions on a validation
sequence and times per-row inference with the baseline's single-thread session.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

if __package__:
    from scripts import train_rnn
else:
    import train_rnn

from wnn.data import SequenceFile
from wnn.research_features import (
    MOMENT_FEATURE_MODES,
    research_feature_control_width,
    validate_causal_moment_prior,
)
from wnn.streaming import StreamingGraph, initial_state

ROOT = Path(__file__).resolve().parents[1]
NAMES_IN = ["row", "prev", "step", "last", "held", "h", "first"]
NAMES_OUT = ["pred", "prev_out", "step_out", "last_out", "held_out", "h_out", "first_out"]


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def validate_parity_membership(
    checkpoint: dict[str, object], parity_split: str, parity_group: int
) -> None:
    """Keep numeric-map parity probes inside the checkpoint's fitting membership."""
    args = checkpoint.get("args", {})
    if not isinstance(args, dict) or args.get("numeric_map", "none") == "none":
        return
    if parity_split != "train":
        raise ValueError("numeric-map parity must use a declared fitting train group")
    provenance = checkpoint.get("provenance", {})
    fit_row_groups = provenance.get("fit_row_groups", {}) if isinstance(provenance, dict) else {}
    train_groups = fit_row_groups.get("train") if isinstance(fit_row_groups, dict) else None
    if (
        not isinstance(train_groups, list)
        or not train_groups
        or any(type(group) is not int or group < 0 for group in train_groups)
        or parity_group not in train_groups
    ):
        raise ValueError("numeric-map parity group is outside checkpoint fitting membership")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="")
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--parity-split", choices=("train", "valid"), default="valid")
    ap.add_argument("--parity-group", type=int, default=0)
    args = ap.parse_args()
    ck = torch.load(
        ROOT / "artifacts" / "runs" / args.run / "model.pt", map_location="cpu", weights_only=False
    )
    validate_parity_membership(ck, args.parity_split, args.parity_group)
    a = ck["args"]
    if a.get("readout_activation", "gelu") != "gelu":
        raise SystemExit("non-GELU readout requires an explicitly supported streaming exporter")
    if (
        a.get("arch", "gru") != "gru"
        or a.get("cell", "gru") != "gru"
        or a.get("prop_input")
        or a.get("raw_only")
        or a.get("online_norm")
        or a.get("heuristics")
        or a.get("privileged_mode", "none") != "none"
        or a.get("sentinel_base_width")
        or a.get("sentinel_indicators", "none") != "none"
        or a.get("numeric_map", "none") != "none"
    ):
        raise SystemExit("only plain causal-input GRU runs are exportable here")
    research_features = a.get("research_features", "none")
    research_feature_control = a.get("research_feature_control", "none")
    research_feature_control_width(research_features, research_feature_control)
    if research_features in MOMENT_FEATURE_MODES:
        prior = ck.get("research_feature_prior")
        if not isinstance(prior, dict):
            raise SystemExit("causal moment checkpoint lacks its fitting-only feature prior")
        prior = validate_causal_moment_prior(prior)
    elif research_features == "none":
        prior = None
    else:
        raise SystemExit(f"streaming export does not support {research_features!r}")
    n_out = 2 + 2 * len([v for v in a.get("aux", "").split(",") if v])
    model = train_rnn.make_model(a, ck["n_in"], n_out).eval()
    model.load_state_dict(ck["model"])
    graph = StreamingGraph(model, research_features, prior, research_feature_control).eval()
    out = Path(args.out) if args.out else ROOT / "submissions" / args.run / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    state = initial_state(model, research_features, prior, research_feature_control)
    input_names = list(NAMES_IN)
    output_names = list(NAMES_OUT)
    if research_features in MOMENT_FEATURE_MODES:
        input_names.append("moment_state")
        output_names.append("moment_state_out")
    torch.onnx.export(
        graph,
        (torch.zeros(1, 112), *state),
        str(out),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamo=False,
    )

    if args.parity_group < 0:
        ap.error("--parity-group must be nonnegative")
    seq = SequenceFile(ROOT / "datasets" / f"{args.parity_split}.parquet").read(
        args.parity_group, targets=False, mask=False
    )
    x = seq.features[: args.rows]
    train_rnn.RESEARCH_FEATURES = research_features
    train_rnn.RESEARCH_FEATURE_PRIOR = prior
    train_rnn.RESEARCH_FEATURE_CONTROL = research_feature_control
    with torch.no_grad():
        want, _ = model(train_rnn.build_inputs(torch.from_numpy(x)[None]))
    want = want[0, :, :2].numpy()

    sess = session(out)
    feeds = {k: v.numpy() for k, v in zip(input_names[1:], state, strict=True)}
    got = np.empty((len(x), 2), np.float32)
    t0 = time.perf_counter()
    for i, row in enumerate(x):
        res = sess.run(None, {"row": row[None], **feeds})
        got[i] = res[0][0]
        feeds = dict(zip(input_names[1:], res[1:], strict=True))
    us = (time.perf_counter() - t0) / len(x) * 1e6
    err = float(np.abs(got - want).max())
    print(
        json.dumps(
            {
                "run": args.run,
                "hidden": a["hidden"],
                "layers": a["layers"],
                "onnx": str(out),
                "parity_split": args.parity_split,
                "parity_group": args.parity_group,
                "bytes": out.stat().st_size,
                "max_abs_err_vs_batch": err,
                "us_per_row": round(us, 2),
                "est_minutes_valid_size": round(us * 37.46e6 / 60e6, 1),
            }
        )
    )
    if err > 1e-3:
        raise SystemExit("streaming ONNX disagrees with batch model")


if __name__ == "__main__":
    main()
