"""Export one audited E464 fixed-epoch-0 head as a common FP32 predictor.

The model construction, sequence loader, preprocessing and recurrent prediction
call mirror the frozen E424 epoch-3 exporter. This entry point is isolated so
the E424 file remains byte-for-byte unchanged while E464 accepts epoch 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
ARM_PARENT = {
    "h093_learned": "c20261004_h093_learned_s3_final",
    "h093_hash": "c20261004_h093_hash_s3_final",
    "h093_universal": "c20261004_h093_universal_s3_final",
    "h093_gru_budget": "c20261004_h093_gru_budget_s3_final",
    "h094_conditional": "c20261004_h094_conditional_s3_final",
    "h094_unconditional": "c20261004_h094_unconditional_s3_final",
    "h094_diagonal": "c20261004_h094_diagonal_s3_final",
    "h094_gru_budget": "c20261004_h094_gru_budget_s3_final",
}
PARENT_SOURCE_SPEC = {
    "h093_learned": "e424_h093_learned_v2.sealed.json",
    "h093_hash": "e424_h093_hash_v2.sealed.json",
    "h093_universal": "e424_h093_universal_v2.sealed.json",
    "h093_gru_budget": "e424_h093_gru_budget_v2.sealed.json",
    "h094_conditional": "e424_h094_conditional_v2.sealed.json",
    "h094_unconditional": "e424_h094_unconditional_v2.sealed.json",
    "h094_diagonal": "e424_h094_diagonal_v2.sealed.json",
    "h094_gru_budget": "e424_h094_gru_budget_v2.sealed.json",
}
FROZEN_E424_PREDICTOR_SHA256 = "fc81eb788035944c5f2e885ec562a1c6d98e2031c6cb82d81864b1fb5dfe4dd5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_source_closure() -> list[dict[str, object]]:
    paths = {
        Path(__file__).resolve(),
        CAMPAIGN / "checks/predict_e424_recurrent.py",
        ROOT / "scripts/train_rnn.py",
        ROOT / "scripts/predict_run.py",
        *((ROOT / "src/wnn").rglob("*.py")),
    }
    for arm in ARM_PARENT:
        spec_path = CAMPAIGN / "specs" / PARENT_SOURCE_SPEC[arm]
        parent_spec = json.loads(spec_path.read_text())
        paths.update(ROOT / item["path"] for item in parent_spec.get("code", []))
    if (ROOT / "utils.py").is_file():
        paths.add(ROOT / "utils.py")
    return [
        {"path": path.relative_to(ROOT).as_posix(), "size": path.stat().st_size,
         "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]


def closure_digest(files: list[dict[str, object]]) -> str:
    payload = json.dumps(files, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _weight_width(state: Mapping[str, object], key: str) -> int:
    weight = state.get(key)
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[0]) <= 0:
        raise ValueError(f"checkpoint is missing a matrix head tensor {key!r}")
    return int(shape[0])


def infer_output_width(arch: str, state: Mapping[str, object]) -> int:
    if arch == "specialist":
        width = _weight_width(state, "universal_head.2.weight")
        expert_weights = sorted(
            key for key in state if key.startswith("expert_heads.") and key.endswith(".weight")
        )
        if not expert_weights:
            raise ValueError("specialist checkpoint has no expert output heads")
        for key in expert_weights:
            if _weight_width(state, key) != width:
                raise ValueError("specialist expert head widths disagree")
        return width
    if arch in {"prefix-adapter", "gru"}:
        return _weight_width(state, "head.2.weight")
    raise ValueError(f"unsupported E464 recurrent architecture {arch!r}")


def _validate_audit(arm: str, run_name: str, checkpoint_sha: str, audit_path: Path,
                    parent_sha: str, parent_spec_path: Path) -> dict[str, object]:
    audit = json.loads(audit_path.read_text())
    parent_spec_sha = sha256_file(parent_spec_path)
    if (
        audit.get("status") != "pass"
        or audit.get("arm") != arm
        or audit.get("child_run") != run_name
        or audit.get("parent_run") != ARM_PARENT[arm]
        or audit.get("parent_checkpoint_sha256") != parent_sha
        or audit.get("parent_code_spec_path") != str(parent_spec_path.relative_to(ROOT))
        or audit.get("parent_code_spec_sha256") != parent_spec_sha
        or audit.get("child_checkpoint_sha256") != checkpoint_sha
        or audit.get("fixed_epoch") != 0
        or audit.get("trainer_returncode") != 0
        or audit.get("non_head_state_mismatches") != []
        or audit.get("head_layout_mismatches") != []
        or audit.get("unchanged_non_head_tensors", 0) <= 0
        or not audit.get("changed_head_parameter_tensors")
    ):
        raise ValueError("E464 audit sidecar does not prove the fixed head-only transfer")
    parent_code = json.loads(parent_spec_path.read_text())
    if audit.get("parent_code_spec_aggregate_sha256") != parent_code.get("code_sha256"):
        raise ValueError("E464 audit refers to a different parent source closure")
    for item in parent_code.get("code", []):
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"E464 parent training source closure changed: {item['path']}")
    return audit


def _predict(args: argparse.Namespace) -> None:
    """Run complete all-VALID FP32 inference after validating fit identity."""
    if args.epoch != 0 or args.inference_precision != "fp32":
        raise ValueError("E464 exports are fixed epoch-0 FP32 predictions")
    if args.arm not in ARM_PARENT:
        raise ValueError(f"unknown E464 arm: {args.arm}")
    expected_run = f"c20261004_e464_{args.arm}_head_s3_final"
    if args.name != expected_run:
        raise ValueError("E464 child run does not match the fixed arm identity")
    if not args.name.replace("_", "").isalnum():
        raise ValueError("run name must contain only letters, digits, and underscores")
    output_name = args.output_name or args.name
    if not output_name.replace("_", "").isalnum():
        raise ValueError("output name must contain only letters, digits, and underscores")

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    import train_rnn
    from torch.utils.data import DataLoader
    from train_rnn import Sequences, make_model, predict
    from wnn.data import SEQUENCE_LENGTH as DATA_SEQUENCE_LENGTH
    from wnn.propensity import load_propensity
    from wnn.resources import configure_torch, dataloader_budget
    from wnn.sentinel import SentinelPreprocessor
    from wnn.splits import file_sha256

    if args.split != "valid":
        raise ValueError("E464 scorer expects complete VALID predictions")
    parent = ARM_PARENT[args.arm]
    child_dir = ROOT / "artifacts/runs" / args.name
    parent_dir = ROOT / "artifacts/runs" / parent
    checkpoint_path = child_dir / "model_ep0.pt"
    child_provenance_path = child_dir / "provenance.json"
    parent_checkpoint_path = parent_dir / "model_ep3.pt"
    parent_provenance_path = parent_dir / "provenance.json"
    audit_path = CAMPAIGN / "staging/e464/audits" / f"{args.name}.json"
    parent_spec_path = CAMPAIGN / "specs" / PARENT_SOURCE_SPEC[args.arm]
    e424_predictor_path = CAMPAIGN / "checks/predict_e424_recurrent.py"
    if sha256_file(e424_predictor_path) != FROZEN_E424_PREDICTOR_SHA256:
        raise ValueError("frozen E424 epoch-3 predictor source has changed")
    for path in (checkpoint_path, child_provenance_path, parent_checkpoint_path,
                 parent_provenance_path, audit_path, parent_spec_path):
        if not path.is_file():
            raise FileNotFoundError(f"required E464 prediction input is absent: {path}")
    checkpoint_sha = file_sha256(checkpoint_path)
    parent_sha = file_sha256(parent_checkpoint_path)
    child_provenance_sha = file_sha256(child_provenance_path)
    parent_provenance_sha = file_sha256(parent_provenance_path)
    audit_sha = sha256_file(audit_path)
    source_manifest_sha = sha256_file(args.source_manifest)
    audit = _validate_audit(args.arm, args.name, checkpoint_sha, audit_path, parent_sha,
                            parent_spec_path)
    code_files_before = code_source_closure()
    code_closure_sha = closure_digest(code_files_before)
    source_manifest = json.loads(args.source_manifest.read_text())
    if (source_manifest.get("schema") != "e464-head-predictor-source-closure-v1"
            or source_manifest.get("files") != code_files_before
            or source_manifest.get("sha256") != code_closure_sha):
        raise ValueError("E464 predictor code closure differs from the draft-bound manifest")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args")
    state = checkpoint.get("model")
    if not isinstance(checkpoint_args, dict) or not isinstance(state, Mapping):
        raise ValueError("E464 epoch-0 checkpoint lacks args or model state")
    if checkpoint.get("epoch") != 0:
        raise ValueError("model_ep0.pt does not contain fixed epoch 0")
    if checkpoint_args.get("checkpoint_policy") != "fixed" or checkpoint_args.get("checkpoint_epoch") != 0:
        raise ValueError("E464 checkpoint metadata does not fix epoch 0")
    parent_checkpoint = torch.load(parent_checkpoint_path, map_location="cpu", weights_only=False)
    parent_state = parent_checkpoint.get("model")
    if not isinstance(parent_state, Mapping):
        raise ValueError("fixed E424 parent checkpoint lacks model state")
    parent_args = parent_checkpoint.get("args", {})
    if checkpoint_args.get("arch") != parent_args.get("arch"):
        raise ValueError("E464 child architecture differs from its fixed parent")
    if checkpoint_args.get("init") != f"{parent}@ep3":
        raise ValueError("E464 child was not initialized from its fixed epoch-3 parent")
    if checkpoint_args.get("privileged_mode", "none") != "none":
        raise ValueError("privileged teacher inputs cannot be exported as causal predictions")
    if (checkpoint_args.get("quote_cache_root")
            and checkpoint.get("provenance", {}).get("quote_cache") is None):
        raise ValueError("E464 checkpoint lacks quote-cache provenance")
    if checkpoint.get("quantization") is not None or checkpoint.get("factorization") is not None:
        raise ValueError("E464 FP32 exporter does not accept quantized/factorized checkpoints")
    child_provenance = json.loads(child_provenance_path.read_text())
    if (
        child_provenance.get("selected_epoch") != 0
        or child_provenance.get("checkpoint_sha256", {}).get("model_ep0.pt") != checkpoint_sha
        or child_provenance.get("parent_checkpoint_sha256") not in (None, parent_sha)
    ):
        raise ValueError("E464 child provenance does not bind the fixed checkpoint/parent")
    if checkpoint_args.get("aux_struct") != parent_args.get("aux_struct") or checkpoint_args.get("aux_next") != parent_args.get("aux_next"):
        raise ValueError("E464 child did not preserve incumbent parent auxiliary targets")
    width = infer_output_width(str(checkpoint_args.get("arch", "")), state)
    if width != 23:
        raise ValueError(f"E464 architecture requires the 23-output parent head, got {width}")

    configure_torch()
    device = torch.device("cuda")
    model = make_model(checkpoint_args, checkpoint["n_in"], width).to(device)
    model.load_state_dict(state, strict=True)
    model._inference_fp32 = True
    train_rnn.SENTINEL_PREPROCESSOR = (
        SentinelPreprocessor(checkpoint["sentinel_preprocessing"])
        if checkpoint.get("sentinel_preprocessing") else None
    )
    train_rnn.RAW_ONLY = bool(checkpoint_args.get("raw_only"))
    train_rnn.ONLINE_NORM = bool(checkpoint_args.get("online_norm"))
    train_rnn.HEURISTICS = bool(checkpoint_args.get("heuristics"))
    train_rnn.RESEARCH_FEATURES = checkpoint_args.get("research_features", "none")
    input_propensity = checkpoint_args.get("prop_input")
    train_rnn.INPUT_PROPENSITY = (
        load_propensity(ROOT / "artifacts/runs" / input_propensity / "model.pt", device)
        if input_propensity else None
    )
    if checkpoint.get("readout") is not None:
        train_rnn.READOUT = [tensor.to(device) for tensor in checkpoint["readout"]]
        train_rnn.READOUT_TILTED = checkpoint.get("readout_tilted", False)

    groups = np.load(ROOT / "cache/valid_targets.npz", allow_pickle=False)["row_group"]
    dataset = Sequences("valid", groups)
    workers, prefetch = dataloader_budget(dataset.batch_tensor_bytes(16), max_workers=2)
    loader = DataLoader(dataset, batch_size=16, num_workers=workers,
                        prefetch_factor=prefetch if workers else None)
    final = ROOT / "cache" / f"pred_{output_name}_valid.npy"
    sidecar_path = final.with_suffix(".json")
    if final.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite E464 prediction artifact: {final}")
    temporary = final.with_name(final.name + ".partial")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                      shape=(len(groups), DATA_SEQUENCE_LENGTH, 2))
    written = 0
    for features, _, _ in loader:
        prediction = predict(model, features, 4000, device)
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"non-finite prediction values near VALID sequence {written}")
        output[written:written + len(prediction)] = prediction[..., :2]
        written += len(prediction)
    if written != len(groups):
        raise RuntimeError(f"wrote {written} of {len(groups)} VALID sequences")
    if (file_sha256(checkpoint_path) != checkpoint_sha
            or file_sha256(parent_checkpoint_path) != parent_sha
            or file_sha256(child_provenance_path) != child_provenance_sha
            or file_sha256(parent_provenance_path) != parent_provenance_sha
            or sha256_file(audit_path) != audit_sha
            or sha256_file(args.source_manifest) != source_manifest_sha
            or code_source_closure() != code_files_before):
        raise RuntimeError("a bound E464 checkpoint, provenance or audit changed during export")
    output.flush()
    del output
    temporary.replace(final)
    sidecar = {
        "run": args.name, "parent_run": parent, "arm": args.arm, "split": "valid",
        "epoch": 0, "fixed_checkpoint_policy": "fixed", "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": parent_sha,
        "child_provenance_sha256": child_provenance_sha,
        "parent_provenance_sha256": parent_provenance_sha,
        "head_audit_sha256": audit_sha,
        "export_source_manifest_sha256": source_manifest_sha,
        "parent_code_spec_sha256": sha256_file(parent_spec_path),
        "parent_code_spec_aggregate_sha256": audit["parent_code_spec_aggregate_sha256"],
        "frozen_e424_epoch3_predictor_sha256": FROZEN_E424_PREDICTOR_SHA256,
        "code_source_closure": {
            "schema": "e464-head-predictor-source-closure-v1",
            "files": code_files_before,
            "sha256": code_closure_sha,
        },
        "head_audit_status": audit["status"],
        "unchanged_non_head_tensors": audit["unchanged_non_head_tensors"],
        "prediction_sha256": file_sha256(final),
        "split_manifest_sha256": checkpoint_args.get("split_manifest_sha256"),
        "gate_mode": checkpoint_args.get("specialist_route", "ordinary"),
        "inference_precision": "fp32", "outputs_exported": 2, "model_outputs": width,
        "row_groups": groups.tolist(),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=sorted(ARM_PARENT))
    parser.add_argument("name")
    parser.add_argument("split", nargs="?", default="valid", choices=("valid",))
    parser.add_argument("--epoch", type=int, choices=(0,), default=0)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--inference-precision", choices=("fp32",), default="fp32")
    parser.add_argument("--source-manifest", type=Path, default=(
        CAMPAIGN / "staging/e464/specs/e464_predictor_source_closure.draft.json"
    ))
    _predict(parser.parse_args())


if __name__ == "__main__":
    main()
