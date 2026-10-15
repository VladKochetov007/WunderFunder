"""Compare exact FP32 and BF16-rounded-dot surrogates on the pinned V8 prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wnn.data import SequenceFile
from wnn.features import event_features
from wnn.nn import RNNModel

PREFIX_ROWS = 137
SOURCE_GROUP = ("valid", 254)
EXPECTED_SEQ_IX = 267_134_787
EXPECTED_JAX_RNN_SHA256 = "294eff11d906fe434dba2c4d0ed4741ad2191f756c4a35204474cb229ff939dc"
EXPECTED_INPUT_SHA256 = "68af663f9a28e26142d3dc20b6780550c55c741c6fdc2a1db8fdd12afe1d3f08"
EXPECTED_EMA_SHA256 = "fdc038206756202cdb9a1831be0ea79da49c27e66842b97259725eca43aa77e2"
EXPECTED_TPU_PREDICTION_SHA256 = "9c3a2218a030fa7a4bb06bed82305d825926132c57ae7414dff73563834986c7"
EXPECTED_PACKET_CODE_SHA256 = "ce334ceb2ddb67bb0b61d34f99586fc327dfa29ae89d77ba77a21891594e8247"


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks without materializing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    """Hash a contiguous ndarray's raw bytes."""
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    """Load the no-pickle exported EMA state as CPU FP32 tensors."""
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: torch.from_numpy(np.array(archive[name], dtype=np.float32, copy=True))
            for name in archive.files
        }


def linear(
    values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    bf16_operands: bool,
) -> torch.Tensor:
    """Apply one FP32-accumulation matmul, optionally rounding both dot operands."""
    if bf16_operands:
        lhs = values.to(torch.bfloat16).to(torch.float32)
        rhs = weight.to(torch.bfloat16).to(torch.float32)
    else:
        lhs = values
        rhs = weight
    return torch.matmul(lhs, rhs.transpose(-1, -2)) + bias


def exact_gelu(values: torch.Tensor) -> torch.Tensor:
    """Use the exact-erf GELU selected by the frozen JAX forward."""
    return F.gelu(values, approximate="none")


def manual_forward(
    inputs: torch.Tensor,
    state: dict[str, torch.Tensor],
    *,
    bf16_operands: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the frozen two-layer PyTorch GRU equations with FP32 state/nonlinearities."""
    normalized = (inputs - state["mean"]) / state["std"]
    projected = exact_gelu(
        linear(
            normalized,
            state["inp.weight"],
            state["inp.bias"],
            bf16_operands=bf16_operands,
        )
    )
    recurrent = projected
    final_states: list[torch.Tensor] = []
    for layer in range(2):
        weight_ih = state[f"gru.weight_ih_l{layer}"]
        weight_hh = state[f"gru.weight_hh_l{layer}"]
        bias_ih = state[f"gru.bias_ih_l{layer}"]
        bias_hh = state[f"gru.bias_hh_l{layer}"]
        hidden = inputs.new_zeros((inputs.shape[0], 256))
        outputs: list[torch.Tensor] = []
        for row_ix in range(inputs.shape[1]):
            row = recurrent[:, row_ix, :]
            input_gates = linear(row, weight_ih, bias_ih, bf16_operands=bf16_operands)
            hidden_gates = linear(hidden, weight_hh, bias_hh, bf16_operands=bf16_operands)
            input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            candidate = torch.tanh(input_new + reset * hidden_new)
            hidden = (1.0 - update) * candidate + update * hidden
            outputs.append(hidden)
        recurrent = torch.stack(outputs, dim=1)
        final_states.append(hidden)

    features = torch.cat((recurrent, projected), dim=-1)
    head_hidden = exact_gelu(
        linear(
            features,
            state["head.0.weight"],
            state["head.0.bias"],
            bf16_operands=bf16_operands,
        )
    )
    predictions = linear(
        head_hidden,
        state["head.2.weight"],
        state["head.2.bias"],
        bf16_operands=bf16_operands,
    )
    return predictions, torch.stack(final_states, dim=0)


def error_summary(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    """Return comparable maximum, mean, and RMS absolute error summaries."""
    difference = (first - second).abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "rms": float(torch.sqrt(torch.mean(difference.square())).item()),
    }


def load_reference_inputs(root: Path, r4_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the producer-identical causal 211-wide inputs without labels or masks."""
    sequence = SequenceFile(root / "datasets/valid.parquet").read(
        SOURCE_GROUP[1], targets=False, mask=False
    )
    if sequence.seq_ix != EXPECTED_SEQ_IX:
        raise ValueError(
            f"source-qualified group {SOURCE_GROUP} has seq_ix={sequence.seq_ix}, "
            f"expected {EXPECTED_SEQ_IX}"
        )
    inputs = np.ascontiguousarray(
        np.concatenate((sequence.features, event_features(sequence.features)), axis=-1)[None],
        dtype=np.float32,
    )
    input_sha256 = sha256_array(inputs)
    with r4_path.open("r", encoding="utf-8") as stream:
        r4 = json.load(stream)
    if r4.get("numpy_input_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError("R4 NumPy-input receipt no longer has the pinned producer hash")
    if not r4.get("numpy_input_matches_tpu_export"):
        raise ValueError("R4 does not confirm exact NumPy-input identity with TPU export")
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError(f"local engineered input hash changed: {input_sha256}")
    return inputs, {
        "seq_ix": int(sequence.seq_ix),
        "rows": int(sequence.features.shape[0]),
        "raw_features_sha256": sha256_array(sequence.features),
        "engineered_input_sha256": input_sha256,
        "r4_input_sha256": str(r4["numpy_input_sha256"]),
        "r4_source_sha256": str(r4["source_sha256"]),
        "targets_loaded": False,
        "scoring_mask_loaded": False,
    }


def main() -> None:
    """Run the bounded 137-row exact-reference and BF16-dot surrogate comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="source repository root")
    parser.add_argument(
        "--jax-rnn-source",
        type=Path,
        required=True,
        help="exact staged JAX RNN source used by the V8 worker",
    )
    parser.add_argument("--output", type=Path, required=True, help="new JSON receipt path")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    jax_rnn_source = args.jax_rnn_source.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    packet = root / "research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8"
    artifacts = root / "artifacts/kaggle_tpu_v8_terminal_20261015_r1/cloud_preflight_tpu_v8"
    receipt_path = artifacts / "realdata_reload/realdata_update_receipt.json"
    r4_path = packet / "FEATURE_FORWARD_DIAGNOSIS_R4.json"
    r4_provenance_path = packet / "R4_EXECUTED_SOURCE_PROVENANCE_R1.json"
    model_path = artifacts / "realdata_reload/ema_after_one_chunk_fp32.npz"
    prediction_path = artifacts / "realdata_reload/updated_ema_valid_fit_group_prediction.npz"
    materialization_path = artifacts / "source_materialization.json"
    manifest_path = packet / "packet_manifest.json"

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    packet_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    r4_provenance = json.loads(r4_provenance_path.read_text(encoding="utf-8"))
    packet_sha256 = sha256_file(packet / "packet/probe.py")
    rnn_sha256 = sha256_file(jax_rnn_source)
    model_sha256 = sha256_file(model_path)
    prediction_sha256 = sha256_file(prediction_path)
    if (
        packet_sha256 != EXPECTED_PACKET_CODE_SHA256
        or packet_manifest["code_sha256"] != packet_sha256
    ):
        raise ValueError("executed V8 packet code differs from its manifest pin")
    if rnn_sha256 != EXPECTED_JAX_RNN_SHA256:
        raise ValueError("staged JAX RNN source differs from the V8 materialization pin")
    if materialization["files"]["src/wnn_jax/rnn.py"]["sha256"] != rnn_sha256:
        raise ValueError("JAX source hash differs from the V8 materialization receipt")
    if r4_provenance["sha256"] != sha256_file(packet / r4_provenance["executed_source"]):
        raise ValueError("R4 executed-source provenance does not match its source file")
    if model_sha256 != EXPECTED_EMA_SHA256 or model_sha256 != receipt["checkpoint"]["ema_sha256"]:
        raise ValueError("updated EMA bytes differ from the V8 producer receipt")
    if (
        prediction_sha256 != EXPECTED_TPU_PREDICTION_SHA256
        or prediction_sha256 != receipt["prediction_export"]["sha256"]
    ):
        raise ValueError("TPU prediction bytes differ from the V8 producer receipt")
    if receipt["prediction_export"]["checkpoint_sha256"] != model_sha256:
        raise ValueError("prediction export is not bound to the pinned updated EMA")

    with np.load(prediction_path, allow_pickle=False) as archive:
        tpu_prediction = torch.from_numpy(
            np.array(archive["predictions"], dtype=np.float32, copy=True)
        )
        embedded_input_sha256 = str(archive["input_sha256"].item())
        embedded_checkpoint_sha256 = str(archive["checkpoint_sha256"].item())
    if embedded_input_sha256 != EXPECTED_INPUT_SHA256:
        raise ValueError("prediction NPZ input identity differs from the producer receipt")
    if embedded_checkpoint_sha256 != model_sha256:
        raise ValueError("prediction NPZ checkpoint identity differs from the EMA file")
    if receipt["prediction_export"]["source_qualified_group"] != list(SOURCE_GROUP):
        raise ValueError("prediction receipt has a different source-qualified sequence")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    inputs_np, input_identity = load_reference_inputs(root, r4_path)
    inputs = torch.from_numpy(inputs_np[:, :PREFIX_ROWS].copy())
    ema_state = load_tensor_state(model_path)
    model = RNNModel(211, hidden=256, layers=2, n_out=23).eval()
    model.load_state_dict(ema_state, strict=True)

    started = time.perf_counter()
    with torch.inference_mode():
        torch_prediction, torch_state = model(inputs)
        fp32_prediction, fp32_state = manual_forward(inputs, ema_state, bf16_operands=False)
    fp32_prediction_matches = bool(
        torch.allclose(fp32_prediction, torch_prediction, atol=2e-5, rtol=2e-5)
    )
    fp32_state_matches = bool(torch.allclose(fp32_state, torch_state, atol=2e-5, rtol=2e-5))
    tpu_prefix = tpu_prediction[:, :PREFIX_ROWS, :]
    results: dict[str, Any] = {
        "fp32_manual_vs_pytorch_all23": error_summary(fp32_prediction, torch_prediction),
        "fp32_manual_vs_pytorch_main2": error_summary(
            fp32_prediction[..., :2], torch_prediction[..., :2]
        ),
        "fp32_manual_state_vs_pytorch_state": error_summary(fp32_state, torch_state),
        "fp32_pytorch_vs_tpu_all23": error_summary(torch_prediction, tpu_prefix),
        "fp32_pytorch_vs_tpu_main2": error_summary(torch_prediction[..., :2], tpu_prefix[..., :2]),
        "bf16_surrogate_run": False,
    }
    if fp32_prediction_matches and fp32_state_matches:
        with torch.inference_mode():
            bf16_prediction, bf16_state = manual_forward(inputs, ema_state, bf16_operands=True)
        results.update(
            {
                "bf16_surrogate_run": True,
                "bf16_dot_surrogate_vs_tpu_all23": error_summary(bf16_prediction, tpu_prefix),
                "bf16_dot_surrogate_vs_tpu_main2": error_summary(
                    bf16_prediction[..., :2], tpu_prefix[..., :2]
                ),
                "bf16_dot_surrogate_vs_fp32_pytorch_all23": error_summary(
                    bf16_prediction, torch_prediction
                ),
                "bf16_dot_surrogate_state_vs_pytorch_state": error_summary(bf16_state, torch_state),
                "bf16_dot_surrogate_improves_main2_max_error_to_tpu": bool(
                    (bf16_prediction[..., :2] - tpu_prefix[..., :2]).abs().max()
                    < (torch_prediction[..., :2] - tpu_prefix[..., :2]).abs().max()
                ),
            }
        )
    elapsed_seconds = time.perf_counter() - started
    finite_values = [torch_prediction, torch_state, fp32_prediction, fp32_state]
    if results["bf16_surrogate_run"]:
        finite_values.extend((bf16_prediction, bf16_state))
    if not all(bool(torch.isfinite(value).all()) for value in finite_values):
        raise FloatingPointError("nonfinite output in the bounded prefix diagnostic")

    receipt_out: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "hypothesis_id": "H-TPU-V8-MATMUL-PRECISION-20261015",
        "experiment_id": "E-TPU-V8-BF16-OPERAND-PREFIX-20261015-R1",
        "status": (
            "completed"
            if fp32_prediction_matches and fp32_state_matches
            else "fp32_manual_control_failed_bf16_skipped"
        ),
        "classification": "137-row fitting-sequence forward-arithmetic diagnostic; no score or quality claim",
        "source_qualified_group": list(SOURCE_GROUP),
        "prefix_rows": PREFIX_ROWS,
        "input_identity": input_identity,
        "parameter_identity": {
            "parameter_source": "reloaded updated EMA, not student weights",
            "ema_sha256": model_sha256,
            "prediction_npz_sha256": prediction_sha256,
            "prediction_embedded_ema_sha256": embedded_checkpoint_sha256,
            "ema_reload_bitwise_identity_from_v8_receipt": receipt["checkpoint"][
                "save_reload_ema_identity"
            ],
            "prediction_export_ema_identity": receipt["prediction_export"]["checkpoint_sha256"],
        },
        "source_pins": {
            "v8_packet_sha256": packet_sha256,
            "jax_rnn_source_sha256": rnn_sha256,
            "r4_result_sha256": sha256_file(r4_path),
            "r4_source_sha256": str(r4_provenance["sha256"]),
            "producer_update_receipt_sha256": sha256_file(receipt_path),
            "producer_input_sha256": EXPECTED_INPUT_SHA256,
            "producer_prediction_sha256": EXPECTED_TPU_PREDICTION_SHA256,
            "jax_version_on_tpu": json.loads(
                (artifacts / "tpu_inventory.json").read_text(encoding="utf-8")
            ).get("jax_version"),
        },
        "global_jax_matmul_precision": {
            "worker_source_sets_global_config": False,
            "worker_launch_receipt_captured_inherited_precision_environment": False,
            "actual_runtime_value": None,
            "interpretation": "backend default is possible but not confirmed for the executed Kaggle worker",
        },
        "surrogate_definition": {
            "fp32_control": "frozen JAX equations replayed in Torch with FP32 dot inputs, FP32 bias, FP32 activations and FP32 recurrent state",
            "bf16_operand_variant": "round only both operands at every matmul through BF16 then back to FP32; Torch FP32 accumulation, bias additions, normalization, GELU, sigmoid, tanh, gates and recurrent state remain FP32",
            "not_claimed": "not a JAX/XLA execution or exact TPU DotAlgorithmPreset implementation",
        },
        "errors": results,
        "tolerance": {
            "manual_fp32_matches_pytorch_all23_atol_2e-5_rtol_2e-5": fp32_prediction_matches,
            "manual_fp32_state_matches_pytorch_atol_2e-5_rtol_2e-5": fp32_state_matches,
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "measured_compute_seconds": elapsed_seconds,
            "process_peak_rss_bytes_linux": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
        "limitations": [
            "The sequence is fitting data and was used by the producer's training update; this is not held-out evidence.",
            "Only the first 137 causal rows were evaluated locally; no full-20k CPU replay was run.",
            "The inherited JAX matmul precision setting was not captured by the V8 worker receipts.",
            "A CPU BF16-rounding surrogate can support or weaken the mechanism hypothesis but cannot prove the executed TPU compiler used that algorithm.",
            "No fit, GPU, TPU, cloud action, or model-quality score is part of this experiment.",
        ],
        "quality_claim": False,
        "score": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt_out, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt_out, sort_keys=True))
    if receipt_out["status"] != "completed":
        raise SystemExit("manual FP32 control failed; BF16 surrogate was correctly skipped")


if __name__ == "__main__":
    main()
