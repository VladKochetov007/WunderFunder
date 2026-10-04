"""Run one registered E464 head fit and prove the parent representation stayed frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
PARENT_NAMES = {
    "h093_learned": "c20261004_h093_learned_s3_final",
    "h093_hash": "c20261004_h093_hash_s3_final",
    "h093_universal": "c20261004_h093_universal_s3_final",
    "h093_gru_budget": "c20261004_h093_gru_budget_s3_final",
    "h094_conditional": "c20261004_h094_conditional_s3_final",
    "h094_unconditional": "c20261004_h094_unconditional_s3_final",
    "h094_diagonal": "c20261004_h094_diagonal_s3_final",
    "h094_gru_budget": "c20261004_h094_gru_budget_s3_final",
}


def architecture_scope(arch: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return frozen parameter prefixes and the exact trainable head prefixes."""
    if arch == "specialist":
        return (
            ("inp", "gru", "experts", "prefix_proj", "router"),
            ("universal_head.", "expert_heads."),
        )
    if arch == "prefix-adapter":
        return (("inp", "gru", "adapter_", "prefix_prior"), ("head.",))
    if arch == "gru":
        return (("inp", "gru"), ("head.",))
    raise ValueError(f"unsupported E464 parent architecture: {arch!r}")


def _prefix_match(name: str, prefix: str) -> bool:
    """Mirror trainer prefix freezing while rejecting accidental token extensions."""
    if prefix.endswith((".", "_")):
        return name.startswith(prefix)
    return name == prefix or name.startswith(prefix + ".")


def classify_parameters(
    parameter_names: Sequence[str], arch: str
) -> tuple[list[str], list[str]]:
    """Fail unless every named parameter is frozen backbone or explicit head."""
    frozen_prefixes, head_prefixes = architecture_scope(arch)
    frozen: list[str] = []
    head: list[str] = []
    invalid: list[str] = []
    for name in parameter_names:
        is_head = any(_prefix_match(name, prefix) for prefix in head_prefixes)
        is_frozen = any(_prefix_match(name, prefix) for prefix in frozen_prefixes)
        if is_head and is_frozen:
            invalid.append(name)
        elif is_head:
            head.append(name)
        elif is_frozen:
            frozen.append(name)
        else:
            invalid.append(name)
    if invalid:
        raise ValueError(f"parameter names fall outside the E464 freeze contract: {invalid}")
    if not head:
        raise ValueError("architecture has no parameters in the registered trainable head")
    return frozen, head


def tensor_fingerprint(tensor: Any) -> dict[str, object]:
    """Hash exact contiguous CPU storage together with dtype and shape."""
    value = tensor.detach().cpu().contiguous()
    raw = value.numpy().tobytes(order="C")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def compare_head_metadata(
    before: Mapping[str, Mapping[str, object]],
    after: Mapping[str, Mapping[str, object]],
    trainable_head_names: Sequence[str],
) -> list[dict[str, object]]:
    """Require trainable head tensors to retain the parent shape and dtype."""
    mismatches = []
    for name in trainable_head_names:
        if name not in before or name not in after:
            mismatches.append({"kind": "head_tensor_missing", "name": name})
        elif before[name].get("shape") != after[name].get("shape") or before[name].get("dtype") != after[name].get("dtype"):
            mismatches.append({"kind": "head_tensor_layout_changed", "name": name,
                               "before": {"shape": before[name].get("shape"), "dtype": before[name].get("dtype")},
                               "after": {"shape": after[name].get("shape"), "dtype": after[name].get("dtype")}})
    return mismatches


def compare_non_head_state(
    before: Mapping[str, Mapping[str, object]],
    after: Mapping[str, Mapping[str, object]],
    trainable_head_names: Sequence[str],
) -> list[dict[str, object]]:
    """Return all missing/changed non-head state entries; buffers are never exempt."""
    allowed = set(trainable_head_names)
    mismatches: list[dict[str, object]] = []
    if set(before) != set(after):
        mismatches.append({
            "kind": "state_key_set",
            "missing_after": sorted(set(before) - set(after)),
            "unexpected_after": sorted(set(after) - set(before)),
        })
    for name in sorted(set(before) | set(after)):
        if name in allowed or name not in before or name not in after:
            continue
        if before[name] != after[name]:
            mismatches.append({"kind": "non_head_state_changed", "name": name})
    return mismatches


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _trainer_argv(parent: str, run_name: str, freeze: Sequence[str]) -> list[str]:
    """Derive the incumbent FT recipe, suppress evaluation, and specialize freeze scope."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import continuation_recipes

    argv = continuation_recipes.head_args(run_name, f"{parent}@ep3", None)
    result: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--eval-source":
            index += 2
            continue
        if argv[index] == "--freeze":
            result.extend(("--freeze", ",".join(freeze)))
            index += 2
            continue
        result.append(argv[index])
        index += 1
    if "--freeze" not in result:
        result.extend(("--freeze", ",".join(freeze)))
    if "--no-eval-during-fit" not in result:
        result.append("--no-eval-during-fit")
    return result


def _load_parent_model(checkpoint_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load a parent checkpoint using the frozen root model factory."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    import train_rnn

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("args"), dict):
        raise ValueError("parent checkpoint lacks its architecture arguments")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("parent checkpoint lacks its model state mapping")
    arch = str(checkpoint["args"].get("arch", "gru"))
    if arch == "specialist":
        weight_key = "universal_head.2.weight"
    elif arch == "prefix-adapter":
        weight_key = "head.2.weight"
    elif arch == "gru":
        weight_key = "head.2.weight"
    else:
        raise ValueError(f"unsupported E464 architecture: {arch!r}")
    weight = state.get(weight_key)
    if weight is None or len(weight.shape) != 2:
        raise ValueError(f"parent checkpoint missing compatible output matrix {weight_key}")
    output_width = int(weight.shape[0])
    if arch == "specialist":
        widths = [int(value.shape[0]) for key, value in state.items()
                  if key.startswith("expert_heads.") and key.endswith(".weight")]
        if not widths or any(width != output_width for width in widths):
            raise ValueError("specialist output head widths disagree")
    model = train_rnn.make_model(checkpoint["args"], int(checkpoint["n_in"]), output_width)
    model.load_state_dict(state, strict=True)
    if int(output_width) != 23:
        raise ValueError(f"E464 requires the incumbent 23-output auxiliary architecture, got {output_width}")
    return model, checkpoint, state


def run(args: argparse.Namespace) -> dict[str, object]:
    """Validate parent scope, run the existing trainer, then audit epoch-0 state."""
    if args.arm not in PARENT_NAMES or PARENT_NAMES[args.arm] != args.parent:
        raise ValueError("arm and parent run do not match the registered E464 set")
    expected_run_name = f"c20261004_e464_{args.arm}_head_s3_final"
    if args.run_name != expected_run_name:
        raise ValueError("child run name does not match the fixed E464 identity")
    parent_dir = ROOT / "artifacts/runs" / args.parent
    parent_checkpoint = parent_dir / "model_ep3.pt"
    parent_provenance_path = parent_dir / "provenance.json"
    run_dir = ROOT / "artifacts/runs" / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite E464 child run: {run_dir}")
    audit_path = ROOT / args.audit
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite audit sidecar: {audit_path}")
    if not parent_checkpoint.is_file() or not parent_provenance_path.is_file():
        raise FileNotFoundError("E464 parent checkpoint/provenance is incomplete")
    parent_checkpoint_sha = file_sha256(parent_checkpoint)
    parent_provenance_sha = file_sha256(parent_provenance_path)
    parent_code_spec = ROOT / args.parent_code_spec
    if not parent_code_spec.is_file():
        raise FileNotFoundError("parent E424 sealed source specification is missing")
    parent_code_spec_sha = file_sha256(parent_code_spec)
    if parent_checkpoint_sha != args.expected_parent_sha256:
        raise ValueError("parent checkpoint hash differs from the draft-bound identity")
    if parent_provenance_sha != args.expected_parent_provenance_sha256:
        raise ValueError("parent provenance hash differs from the draft-bound identity")
    if parent_code_spec_sha != args.expected_parent_code_spec_sha256:
        raise ValueError("parent E424 sealed source spec differs from the draft-bound identity")
    code_spec = json.loads(parent_code_spec.read_text())
    for item in code_spec.get("code", []):
        if file_sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"source closure differs from parent E424 code identity: {item['path']}")

    parent_provenance = json.loads(parent_provenance_path.read_text())
    if parent_provenance.get("checkpoint_sha256", {}).get("model_ep3.pt") != parent_checkpoint_sha:
        raise ValueError("parent provenance does not bind model_ep3.pt bytes")
    if parent_provenance.get("selected_epoch") != 3:
        raise ValueError("E464 parent did not select fixed epoch 3")
    parent_args = parent_provenance.get("args", {})
    if parent_args.get("checkpoint_policy") != "fixed" or parent_args.get("checkpoint_epoch") != 3:
        raise ValueError("E464 parent checkpoint policy is not fixed epoch 3")
    code_hashes = {item["path"]: item["sha256"] for item in code_spec.get("code", [])}
    model_source = "src/wnn/specialist.py" if parent_args.get("arch") in ("specialist", "prefix-adapter") else "src/wnn/nn.py"
    if parent_provenance.get("model_source_sha256") is not None and parent_provenance["model_source_sha256"] != code_hashes.get(model_source):
        raise ValueError("parent provenance model source hash differs from the sealed E424 source closure")
    if parent_provenance.get("run_identity_source_sha256") != code_hashes.get("src/wnn/run_identity.py"):
        raise ValueError("parent provenance run-identity source differs from the sealed E424 source closure")
    parent_fit_groups = parent_provenance.get("fit_row_groups", {}).get("valid", [])
    if len(parent_fit_groups) != 1505 or len(set(parent_fit_groups)) != 1505:
        raise ValueError("parent fit provenance does not identify 1,505 unique valid fitting sequences")
    parent_fit_identity = parent_provenance.get("fit_membership_identity", {}).get("valid", {})
    if parent_fit_identity.get("count") != 1505:
        raise ValueError("parent fit-membership summary does not identify 1,505 valid sequences")

    model, parent_checkpoint_data, parent_state = _load_parent_model(parent_checkpoint)
    arch = str(parent_checkpoint_data["args"].get("arch", "gru"))
    frozen_prefixes, head_prefixes = architecture_scope(arch)
    frozen_names, head_names = classify_parameters(
        [name for name, _ in model.named_parameters()], arch
    )
    # Verify the classifier is exhaustive for model state keys, and snapshots include buffers.
    state_names = set(parent_state)
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    if parameter_names | buffer_names != state_names:
        raise ValueError("parent state_dict keys do not equal named parameters plus buffers")
    if set(head_names) - parameter_names or set(frozen_names) - parameter_names:
        raise ValueError("parameter classification references absent model parameters")
    before = {name: tensor_fingerprint(value) for name, value in parent_state.items()}
    actual_freeze = _trainer_argv(args.parent, args.run_name, frozen_prefixes)
    if actual_freeze[actual_freeze.index("--freeze") + 1] != ",".join(frozen_prefixes):
        raise RuntimeError("trainer argv freeze prefixes differ from the checked architecture scope")
    if "--no-eval-during-fit" not in actual_freeze:
        raise RuntimeError("E464 must suppress in-fit evaluation")

    if args.preflight_only:
        evidence = {
            "status": "preflight_pass",
            "trainer_started": False,
            "protocol": "E464 actual-parent architecture/name classification only",
            "arm": args.arm,
            "parent_run": args.parent,
            "child_run": args.run_name,
            "parent_epoch": 3,
            "parent_checkpoint_sha256": parent_checkpoint_sha,
            "parent_provenance_sha256": parent_provenance_sha,
            "parent_code_spec_path": args.parent_code_spec,
            "parent_code_spec_sha256": parent_code_spec_sha,
            "parent_code_spec_aggregate_sha256": code_spec.get("code_sha256"),
            "architecture": arch,
            "freeze_prefixes": list(frozen_prefixes),
            "trainable_head_prefixes": list(head_prefixes),
            "frozen_parameter_names": frozen_names,
            "trainable_head_parameter_names": head_names,
            "state_tensor_fingerprints": before,
            "parameter_count": len(parameter_names),
            "buffer_count": len(buffer_names),
            "expected_fit_valid_group_count": len(parent_fit_groups),
            "trainer_argv_contract": actual_freeze,
        }
        write_json(audit_path, evidence)
        return evidence

    env = os.environ.copy()
    source_path = str(ROOT / "src")
    env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, *actual_freeze[1:]], cwd=ROOT, env=env, check=False
    )
    evidence: dict[str, object] = {
        "protocol": "E464 fixed-parent architecture-aware head-only fine-tune",
        "arm": args.arm,
        "parent_run": args.parent,
        "child_run": args.run_name,
        "parent_epoch": 3,
        "parent_checkpoint_sha256": parent_checkpoint_sha,
        "parent_provenance_sha256": parent_provenance_sha,
        "parent_code_spec_path": args.parent_code_spec,
        "parent_code_spec_sha256": parent_code_spec_sha,
        "parent_code_spec_aggregate_sha256": code_spec.get("code_sha256"),
        "parent_model_source_path": model_source,
        "parent_model_source_sha256": code_hashes.get(model_source),
        "parent_model_source_explicit_in_provenance": parent_provenance.get("model_source_sha256") is not None,
        "architecture": arch,
        "freeze_prefixes": list(frozen_prefixes),
        "trainable_head_prefixes": list(head_prefixes),
        "frozen_parameter_names": frozen_names,
        "trainable_head_parameter_names": head_names,
        "pre_fit_state_tensors": len(before),
        "pre_fit_parameter_count": len(parameter_names),
        "pre_fit_buffer_count": len(buffer_names),
        "trainer_argv": actual_freeze,
        "trainer_returncode": completed.returncode,
    }
    try:
        if file_sha256(parent_checkpoint) != parent_checkpoint_sha or file_sha256(parent_provenance_path) != parent_provenance_sha:
            raise RuntimeError("parent checkpoint or provenance changed during head fitting")
        if completed.returncode:
            raise RuntimeError(f"root trainer exited with status {completed.returncode}")
        child_checkpoint_path = run_dir / "model_ep0.pt"
        child_provenance_path = run_dir / "provenance.json"
        history_path = run_dir / "history.json"
        if not all(path.is_file() for path in (child_checkpoint_path, run_dir / "model.pt",
                                                child_provenance_path, history_path,
                                                run_dir / "training_state.pt")):
            raise FileNotFoundError("trainer did not export every required E464 epoch-0 artifact")
        child_checkpoint_sha = file_sha256(child_checkpoint_path)
        import torch

        child = torch.load(child_checkpoint_path, map_location="cpu", weights_only=False)
        after_state = child.get("model")
        if not isinstance(after_state, Mapping):
            raise ValueError("child epoch-0 checkpoint lacks model state")
        after = {name: tensor_fingerprint(value) for name, value in after_state.items()}
        head_layout_mismatches = compare_head_metadata(before, after, head_names)
        if head_layout_mismatches:
            raise RuntimeError(f"head tensor layout changed: {head_layout_mismatches[:12]}")
        mismatches = compare_non_head_state(before, after, head_names)
        if mismatches:
            raise RuntimeError(f"non-head state changed: {mismatches[:12]}")
        child_provenance = json.loads(child_provenance_path.read_text())
        child_args = child.get("args", {})
        child_fit_groups = child_provenance.get("fit_row_groups", {}).get("valid", [])
        child_fit_identity = child_provenance.get("fit_membership_identity", {}).get("valid", {})
        expected_recipe = {
            "seed": 3,
            "epochs": 1,
            "batch": 16,
            "chunk": 2000,
            "lr": 0.0005,
            "wd": 0.01,
            "loss": "gcorr",
            "optim": "adamw",
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 0,
            "mask_weight": True,
            "no_eval_during_fit": True,
            "train_source": "valid",
            "inherit_init_preprocessing": True,
            "freeze": ",".join(frozen_prefixes),
            "eval_ema": False,
            "full_valid": True,
        }
        mismatched_args = {
            key: {"expected": value, "actual": child_args.get(key)}
            for key, value in expected_recipe.items()
            if child_args.get(key) != value
        }
        if child.get("epoch") != 0 or child_provenance.get("selected_epoch") != 0:
            raise ValueError("child checkpoint is not the fixed epoch-0 output")
        child_checkpoint_hashes = child_provenance.get("checkpoint_sha256", {})
        if child_checkpoint_sha != child_checkpoint_hashes.get("model_ep0.pt"):
            raise ValueError("child provenance does not bind model_ep0.pt bytes")
        if file_sha256(run_dir / "model.pt") != child_checkpoint_hashes.get("model.pt"):
            raise ValueError("child provenance does not bind its fixed model.pt alias")
        if mismatched_args:
            raise ValueError(f"head fine-tune recipe changed: {mismatched_args}")
        if len(child_fit_groups) != 1505 or sorted(child_fit_groups) != sorted(parent_fit_groups):
            raise ValueError("child fit membership differs from the parent 1,505-sequence valid membership")
        inherited_aux_fields = ("aux", "aux_struct", "aux_next", "aux_next_k", "aux_tier",
                                "aux_mask", "aux_sign", "aux_quote", "quote_cache_root")
        aux_mismatches = {field: {"parent": parent_args.get(field), "child": child_args.get(field)}
                          for field in inherited_aux_fields
                          if child_args.get(field) != parent_args.get(field)}
        if aux_mismatches:
            raise ValueError(f"parent auxiliary settings were not inherited exactly: {aux_mismatches}")
        if file_sha256(parent_checkpoint) != parent_checkpoint_sha or file_sha256(parent_provenance_path) != parent_provenance_sha:
            raise RuntimeError("parent checkpoint or provenance changed during head fitting")
        if child_args.get("arch", "gru") != arch:
            raise ValueError("child architecture differs from its bound parent")
        history = json.loads(history_path.read_text())
        if len(history) != 1 or history[0].get("evaluation") != "withheld":
            raise ValueError("E464 must produce one epoch and withhold evaluation")
        updates = child_provenance.get("update_diagnostics", [])
        if len(updates) != 1:
            raise ValueError("E464 provenance must contain exactly one epoch update record")
        epoch_updates = updates[0].get("epoch", {})
        attempts = epoch_updates.get("optimizer_attempts", 0)
        completed_updates = epoch_updates.get("optimizer_completed_updates", 0)
        skipped_updates = epoch_updates.get("amp_skipped_updates", 0)
        if epoch_updates.get("completed_batches") != 93 or not attempts or completed_updates <= 0:
            raise ValueError("E464 head fit lacks expected batches or a successful optimizer update")
        if completed_updates + skipped_updates != attempts:
            raise ValueError("E464 AMP attempt/success/skip counters do not reconcile")
        changed_heads = [name for name in head_names if before[name] != after.get(name)]
        if not changed_heads:
            raise ValueError("E464 head fit completed without changing any trainable head tensor")
        evidence.update({
            "status": "pass",
            "child_checkpoint_sha256": child_checkpoint_sha,
            "child_fit_membership_identity": child_fit_identity,
            "fit_row_group_set_sha256": hashlib.sha256(json.dumps(sorted(parent_fit_groups), separators=(",", ":")).encode()).hexdigest(),
            "fixed_epoch": 0,
            "non_head_state_mismatches": [],
            "head_layout_mismatches": [],
            "unchanged_non_head_tensors": len(set(before) - set(head_names)),
            "changed_head_parameter_tensors": changed_heads,
            "aux_struct": child_args.get("aux_struct"),
            "aux_next": child_args.get("aux_next"),
            "evaluation": "withheld",
            "update_counters": epoch_updates,
        })
    except BaseException as exc:
        evidence.update({
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        write_json(audit_path, evidence)
        raise
    write_json(audit_path, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(PARENT_NAMES), required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--expected-parent-provenance-sha256", required=True)
    parser.add_argument("--parent-code-spec", required=True)
    parser.add_argument("--expected-parent-code-spec-sha256", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="inspect and classify the actual parent model, but do not launch training")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in (
        "status", "arm", "parent_run", "child_run", "parent_checkpoint_sha256",
        "child_checkpoint_sha256", "unchanged_non_head_tensors",
        "changed_head_parameter_tensors",
    )}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
