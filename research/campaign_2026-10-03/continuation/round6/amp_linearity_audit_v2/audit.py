from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from metadata import PROJECTION_VERSION, canonical_metadata

# Trusted capture containers contain heterogeneous tensors and metadata.
Payload = dict[str, Any]
ROOT = Path(__file__).resolve().parent
TERMS = ("main", "ema", "struct", "next")
CASES = {
    "synthetic_actual_architecture_normal": "synthetic_cuda_actual_architecture",
    "real_panel_normal": "real_fit_capture",
}
CHUNK = 131072


def sha(path: Path) -> str:
    """Hash bounded blocks; never called on pre.pt to avoid input-byte scans."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value: object) -> bool:
    """Validate declared SHA256 syntax, not approval authenticity."""
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def approval_guard(approval: Payload) -> None:
    """Fail before tensor loads unless both independently authorized cases bind."""
    if (
        approval.get("schema_version") != 4
        or approval.get("independent_measurement_review_passed") is not True
        or approval.get("controller_analysis_authorized") is not True
    ):
        raise ValueError("UNAVAILABLE_INDEPENDENT_V4_REVIEW_AND_CONTROLLER_AUTHORIZATION")
    if not is_sha(approval.get("independent_measurement_review_receipt_sha256")):
        raise ValueError("Independent review receipt required")
    if set(approval.get("cases", {})) != set(CASES):
        raise ValueError("Exactly both frozen normal cases required; no replacement")
    for name, case in approval["cases"].items():
        projection = case.get("metadata_identity")
        if (
            not isinstance(projection, dict)
            or projection.get("projection_version") != PROJECTION_VERSION
            or "metadata" in case
        ):
            raise ValueError(
                "Explicit canonical metadata identity required; raw metadata forbidden"
            )
        if (
            case.get("default_ordinary_passed") is not True
            or case.get("forced_skip") is not False
            or case.get("evidence_kind") != CASES[name]
        ):
            raise ValueError("Default ordinary/normal evidence prerequisite failed")
        if set(case.get("sha256", {})) != {"pre", "post", "components"} or not all(
            is_sha(v) for v in case["sha256"].values()
        ):
            raise ValueError("Independently approved file hashes required")


def counter(value: object) -> int:
    """Read scalar integral clocks without touching any model/input arrays."""
    if isinstance(value, torch.Tensor):
        if (
            value.device.type != "cpu"
            or value.numel() != 1
            or value.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Invalid clock tensor")
        value = int(value)
    if type(value) is not int or value < 0:
        raise ValueError("Invalid integral clock")
    return value


def tensor_metadata(value: object, identity: Payload) -> None:
    """Check tensor metadata only; data finiteness is checked in tiny chunks."""
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or not value.is_floating_point()
        or not value.is_contiguous()
        or list(value.shape) != identity["shape"]
        or str(value.dtype) != identity["dtype"]
    ):
        raise ValueError("Tensor shape/dtype/contiguity identity mismatch")


def parameter_map(
    pre: Payload, components: Payload, post: Payload, case: Payload
) -> list[tuple[str, str, str]]:
    """Access pre metadata and tensor shapes only, never values/batch/moments."""
    if (
        pre["version"] != 4
        or pre["phase"] != "pre_forward_pre_zero_grad"
        or pre["force_skip"] is not False
        or post.get("skipped", False) is not False
    ):
        raise ValueError("Unsupported or forced/skipped capture")
    if (
        pre["source"] != case["source"]
        or pre["execution"] != case["execution"]
        or canonical_metadata(pre["metadata"]) != case["metadata_identity"]
    ):
        raise ValueError("Approved source/backend/case metadata differs")
    if (
        counter(post["clock"]["skips"]) != counter(pre["clock"]["skips"])
        or counter(post["clock"]["attempts"]) != counter(pre["clock"]["attempts"]) + 1
    ):
        raise ValueError("Nonordinary post clocks")
    canonical = case["optimizer_id_names"]
    if {str(k): v for k, v in pre["optimizer_id_names"].items()} != canonical:
        raise ValueError("Canonical serialized ID/name map differs")
    groups = pre["optimizer"]["param_groups"]
    order = pre["optimizer_names"]
    if len(groups) != len(order) or len(groups) != len(case["optimizer_groups"]):
        raise ValueError("Optimizer group count differs")
    entries = []
    for i, (group, names, approved) in enumerate(
        zip(groups, order, case["optimizer_groups"], strict=True)
    ):
        observed = dict(ids=group["params"], names=names, muon=group.get("muon", False))
        if observed != approved or len(group["params"]) != len(names):
            raise ValueError("Approved optimizer group order/flags differs")
        for index, name in zip(group["params"], names, strict=True):
            if canonical.get(str(index)) != name:
                raise ValueError("Optimizer ID/name association differs")
            tensor_metadata(pre["student"][name], case["tensor_identity"][name])
            entries.append((name, "muon" if approved["muon"] else "adam", f"optimizer_group_{i}"))
    names = {e[0] for e in entries}
    if (
        len(names) != len(entries)
        or names != set(canonical.values())
        or names != set(case["tensor_identity"])
    ):
        raise ValueError("Incomplete/duplicate canonical names")
    if set(components["component_gradients"]) != set(TERMS) or set(post["preclip"]) != names:
        raise ValueError("Four component terms or whole-gradient names differ")
    for term in TERMS:
        if set(components["component_gradients"][term]) != names:
            raise ValueError("Missing/extra component parameter names")
    return entries


def empty() -> Payload:
    """Initialize scalar accumulators only."""
    return dict(
        elements=0,
        active_parameters=0,
        unused_parameters=0,
        error_sq=0.0,
        reference_sq=0.0,
        sum_sq=0.0,
        max_abs=0.0,
    )


def finished(row: Payload) -> Payload:
    """Use undefined relative norm for zero reference, without fitted epsilon."""
    error = math.sqrt(row["error_sq"])
    reference = math.sqrt(row["reference_sq"])
    return dict(
        elements=row["elements"],
        active_parameters=row["active_parameters"],
        unused_parameters=row["unused_parameters"],
        error_l2=error,
        error_max_abs=row["max_abs"],
        error_rms=error / math.sqrt(row["elements"]) if row["elements"] else 0.0,
        reference_l2=reference,
        component_fp64_sum_l2=math.sqrt(row["sum_sq"]),
        relative_error_l2=error / reference if reference else None,
        zero_reference=reference == 0,
    )


def discrepancy(pre: Payload, components: Payload, post: Payload, case: Payload) -> Payload:
    """Measure arithmetic differences only; no autograd, coefficient or bound."""
    entries = parameter_map(pre, components, post, case)
    scopes = {"total": empty()}
    parameters = []
    max_workspace = 0
    for name, family, group in entries:
        identity = case["tensor_identity"][name]
        size = math.prod(identity["shape"])
        gradients = [components["component_gradients"][t][name] for t in TERMS]
        reference = post["preclip"][name]
        active = reference is not None
        if active != any(g is not None for g in gradients):
            raise ValueError("Whole/component unused participation mismatch")
        for tensor in [reference, *gradients]:
            if tensor is not None:
                tensor_metadata(tensor, identity)
        row = empty()
        row.update(elements=size, active_parameters=int(active), unused_parameters=int(not active))
        if active:
            for start in range(0, size, CHUNK):
                stop = min(start + CHUNK, size)
                summed = np.zeros(stop - start, dtype=np.float64)
                for gradient in gradients:
                    if gradient is not None:
                        values = gradient.view(-1)[start:stop].numpy().astype(np.float64)
                        if not np.isfinite(values).all():
                            raise ValueError("Nonfinite component gradient")
                        summed += values
                        del values
                actual = reference.view(-1)[start:stop].numpy().astype(np.float64)
                error = summed - actual
                if not np.isfinite(actual).all() or not np.isfinite(error).all():
                    raise ValueError("Nonfinite reference or arithmetic error")
                row["error_sq"] += float(error @ error)
                row["reference_sq"] += float(actual @ actual)
                row["sum_sq"] += float(summed @ summed)
                row["max_abs"] = max(row["max_abs"], float(np.abs(error).max()))
                # Conservative allowance for live FP64 arrays and boolean temporaries.
                max_workspace = max(max_workspace, 8 * (stop - start) * 8)
                del actual, error, summed
        if not all(
            math.isfinite(row[k]) for k in ["error_sq", "reference_sq", "sum_sq", "max_abs"]
        ):
            raise ValueError("Nonfinite accumulated diagnostic")
        scope = (
            "shared"
            if name.startswith(("inp.", "gru."))
            else "readout"
            if name.startswith("head.")
            else "other"
        )
        for label in ["total", family, group, scope]:
            output = scopes.setdefault(label, empty())
            for key in [
                "elements",
                "active_parameters",
                "unused_parameters",
                "error_sq",
                "reference_sq",
                "sum_sq",
            ]:
                output[key] += row[key]
            output["max_abs"] = max(output["max_abs"], row["max_abs"])
        parameters.append(
            dict(
                name=name,
                optimizer=family,
                serialized_group=group,
                name_scope=scope,
                metrics=finished(row),
            )
        )
    if max_workspace > 32 * 1024**2:
        raise ValueError("Workspace exceeded")
    return dict(
        status="DESCRIPTIVE_DIFFERENT_OPERATION_DISCREPANCY_ONLY",
        scopes={k: finished(v) for k, v in scopes.items()},
        parameters=parameters,
        max_converted_workspace_bytes=max_workspace,
        pre_values_or_inputs_accessed=False,
        coefficient_or_scale_reapplied=False,
        acceptance_bound_inferred=False,
        calibrated_allowance=None,
        update_attribution=False,
        scientific_conflict_or_score_claim=False,
    )


def main() -> None:
    """Load only after external review/controller approval; retain fixed exclusions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Output must remain in own directory")
    approval = json.loads(args.approval.read_text())
    approval_guard(approval)  # Must precede ALL capture path access/load.
    torch.set_num_threads(1)
    results = {}
    for name in CASES:
        case = approval["cases"][name]
        try:
            payloads = {}
            for kind in ["pre", "post", "components"]:
                path = Path(case["paths"][kind])
                if path.stat().st_size > (700_000_000 if kind == "pre" else 64_000_000):
                    raise ValueError("Capture file exceeds bounded size")
                if kind != "pre" and sha(path) != case["sha256"][kind]:
                    raise ValueError("Post/components approved content hash differs")
                # Approved trusted local PyTorch payloads, never trainer restoration.
                payloads[kind] = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            results[name] = discrepancy(
                payloads["pre"], payloads["components"], payloads["post"], case
            )
            results[name]["approved_file_sha256"] = case["sha256"]
            results[name]["pre_sha_local_verification"] = (
                "NOT REHASHED: approved declaration; metadata-only access prevents serialized input scan"
            )
            del payloads
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            results[name] = dict(
                status="REJECTED_FIXED_CASE_NO_REPLACEMENT",
                reason=f"{type(error).__name__}: {error}",
            )
    args.output.write_text(
        json.dumps(
            dict(
                cases=results,
                approval_sha256=sha(args.approval),
                prereg_sha256=sha(ROOT / "PREREG.md"),
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
