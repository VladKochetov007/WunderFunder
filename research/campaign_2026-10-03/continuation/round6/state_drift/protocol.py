"""One isolated h intervention using the frozen v4 source adapter.

Any is confined to heterogeneous capture dictionaries and source module interfaces.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
V4 = HERE.parents[1] / "round4/live_capture_v4"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(V4))
import baselines as floors  # noqa: E402
import live_capture as cap  # noqa: E402


def tensor_identity(value: torch.Tensor) -> str:
    """Hash exact tensor shape/dtype/bytes without changing scientific dtype."""
    tensor = value.detach().cpu().contiguous()
    meta = json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    return hashlib.sha256(meta.encode() + tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def require_runtime(payload: dict[str, Any], receipt_path: Path | None) -> None:
    """Block actual CUDA private-RNG validation/restores outside later approved runtime."""
    if payload["execution"]["device"] != "cuda":
        if payload["metadata"].get("evidence_kind") != "synthetic_cpu_actual_code":
            raise ValueError("CPU accepts tiny synthetic protocol controls only")
        return
    if receipt_path is None:
        raise ValueError("UNAPPROVED CUDA private-RNG validation/restore")
    import runner as v4_runner
    from run_real import own_hashes

    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "approved"
        or receipt.get("default_v4_gate") != "PASS"
        or receipt.get("independent_state_drift_review") != "PASS"
        or receipt.get("candidate_sha256") != own_hashes()
        or receipt.get("reviewer") in (None, "", "state_drift", "round6-state-drift")
    ):
        raise ValueError("UNAPPROVED state-drift receipt")
    v4_runner.verify_review(Path(receipt["v4_review_receipt"]))
    v4_runner.runtime_guard()


def prefix_state(payload: dict[str, Any], receipt_path: Path | None = None) -> torch.Tensor:
    """Recompute only student prefix under current pre weights, starting None."""
    require_runtime(payload, receipt_path)
    env = cap.restore(payload, cap.trainer_module())
    chunk = payload["args"]["chunk"]
    if payload["metadata"]["s"] != 4 * chunk:
        raise ValueError("requires exactly four prefix chunks")
    state = None
    device = env["device"].type
    # Preserve the student's training forward context, discard each graph after
    # detaching h. No prefix backward or optimizer update is performed.
    with torch.enable_grad():
        for start in range(0, 4 * chunk, chunk):
            with torch.autocast(
                device,
                dtype=env["amp_dtype"],
                enabled=device == "cuda",
                cache_enabled=payload["execution"]["autocast_cache"],
            ):
                out, state = env["forward_chunk"](
                    env["model"], env["inputs"], slice(start, start + chunk), state
                )
            state = env["detach_state"](state)
            del out
    if state is None or state.dtype != payload["h"].dtype or not torch.isfinite(state).all():
        raise ValueError("prefix hidden dtype/finite contract violated")
    # No clocks or moments advanced during prefix, and input/normalization untouched.
    cap.compare(cap.cloned(env["model"].state_dict()), payload["student"])
    cap.compare(cap.cloned(env["opt"].state_dict()), payload["optimizer"])
    cap.compare(env["sched"].state_dict(), payload["scheduler"])
    cap.compare(env["scaler"].state_dict(), payload["scaler"])
    cap.compare(cap.cloned(env["ema_model"].module.state_dict()), payload["ema"])
    if (
        env["ema_model"].n != payload["ema_n"]
        or env["step_count"] != payload["clock"]["step_count"]
    ):
        raise ValueError("prefix advanced update clock")
    return cap.cloned(state)


class LocalCapture(cap.Capture):
    """Observe mutable state without recopying full immutable sequence inputs on host."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(True, force_skip=payload["force_skip"])
        self.payload = payload

    def before(self, env: dict[str, Any]) -> None:
        """Snapshot actual restored state; reuse source-bound immutable batch handles.

        v4.restore clones every batch field into the execution device verbatim;
        original chunk loss reads these fields without updating them. Source
        identities gate both functions. Avoid a second ~600MB full-host copy.
        """
        clock = env["epoch_counter"]
        self.terms = {}
        self.pre = {
            **self.payload,
            "student": cap.cloned(env["model"].state_dict()),
            "ema": cap.cloned(env["ema_model"].module.state_dict()),
            "optimizer": cap.cloned(env["opt"].state_dict()),
            "scheduler": cap.cloned(env["sched"].state_dict()),
            "scaler": cap.cloned(env["scaler"].state_dict()),
            "h": cap.cloned(env["h"]),
            "h_ema": cap.cloned(env["h_ema"]),
            "ema_n": env["ema_model"].n,
            "ema_decay": env["ema_model"].decay,
            "rng": cap.get_rng(env["device"].type),
            "clock": {
                "step_count": env["step_count"],
                "attempts": clock.attempts,
                "skips": cap.cloned(clock.skips),
                "ema_updates": clock.ema_updates,
                "completed_batches": clock.completed_batches,
            },
        }


def local_update(
    payload: dict[str, Any], hidden: torch.Tensor, receipt_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore ALL original clocks/RNG; change only detached incoming student h."""
    require_runtime(payload, receipt_path)
    env = cap.restore(payload, cap.trainer_module())
    env["h"] = cap.cloned(hidden, env["device"].type)
    observer = LocalCapture(payload)
    env["_capture"] = observer
    exec(cap.compiled_chunk(env["device"].type == "cpu"), env)
    for key in payload:
        if key != "h":
            cap.compare(payload[key], observer.pre[key], path="pre." + key)
    return observer.post, observer.pre


def vector(value: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten the WHOLE named parameter vector; no component decomposition."""
    if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in value.values()):
        raise ValueError("whole vector contains absent/nonfinite gradients or updates")
    return torch.cat([value[k].double().reshape(-1) for k in sorted(value)])


def observable(post: dict[str, Any], key: str) -> torch.Tensor:
    """Select preregistered main predictions or whole gradient/update."""
    if key == "prediction":
        return post["out"][..., :2].double().reshape(-1)
    return vector(post["preclip" if key == "gradient" else "parameter_delta"])


def rms(value: torch.Tensor) -> float:
    """Compute full-vector RMS in float64 for diagnostic reporting."""
    return float(value.double().square().mean().sqrt())


def pairwise_floor(samples: list[torch.Tensor]) -> float:
    """Max pairwise repeat RMS, never from scientific comparison."""
    return max((rms(a - b) for i, a in enumerate(samples) for b in samples[:i]), default=0.0)


def execute(
    payload: dict[str, Any],
    observed_post: dict[str, Any],
    output: Path,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze independent floors, gate default parity, then do ONE binary drift test."""
    require_runtime(payload, receipt_path)
    cap.validate(payload)  # includes private installed RNG validation before global setters
    if payload["force_skip"]:
        raise ValueError("scientific test requires a normal, non-planted-skip capture")
    output.mkdir(parents=True, exist_ok=False)
    baseline, policy = floors.calibrate(payload, output / "default_baselines")
    floors.compare_policy(observed_post, baseline, policy)
    prefix_samples = [prefix_state(payload, receipt_path) for _ in range(6)]
    prefix_floor = pairwise_floor(prefix_samples)
    ceiling = 1e-5 + 1e-4 * max(float(h.abs().max()) for h in prefix_samples)
    if 2 * prefix_floor > ceiling or (
        payload["execution"]["device"] == "cpu" and prefix_floor != 0
    ):
        raise ValueError("prefix-only repeated numerical floor is unstable")
    defaults = [
        torch.load(output / "default_baselines" / f"original_default_{i}.pt", weights_only=False)
        for i in range(6)
    ]
    independent = {
        key: pairwise_floor([observable(post, key) for post in defaults])
        for key in ("prediction", "gradient", "update")
    }
    independent["hidden"] = prefix_floor
    frozen = {
        "floors_rms": independent,
        "source": "six default-only and six prefix-only independent repeats; before counterfactual",
        "multiplier": 5,
        "policy_sha256": cap.sha(output / "default_baselines/frozen_floor.json"),
    }
    (output / "independent_floors.json").write_text(json.dumps(frozen, indent=2) + "\n")
    frozen_hash = cap.sha(output / "independent_floors.json")
    del defaults
    reference = prefix_samples[0]
    alternative, intervention_pre = local_update(payload, reference, receipt_path)
    cap.compare(baseline["target"], alternative["target"])
    floors.compare_policy(baseline["out_ema"], alternative["out_ema"], policy, path="root.out_ema")
    for key in ("rng", "clock", "scheduler", "scaler", "h_ema"):
        cap.compare(baseline[key], alternative[key], path="post." + key)
    hidden_difference = rms(reference.double() - payload["h"].double())
    effects = {
        "hidden": {
            "rms_difference": hidden_difference,
            "relative_l2": hidden_difference / max(rms(payload["h"]), 1e-12),
        }
    }
    for key in ("prediction", "gradient", "update"):
        a, b = observable(baseline, key), observable(alternative, key)
        diff = rms(a - b)
        effects[key] = {
            "rms_difference": diff,
            "relative_l2": diff / max(rms(a), 1e-12),
            "reference_rms": rms(a),
        }
    detectable = {
        key: effect["rms_difference"] > 5 * independent[key] for key, effect in effects.items()
    }
    material = (
        detectable["hidden"]
        and effects["hidden"]["relative_l2"] >= 0.01
        and detectable["prediction"]
        and effects["prediction"]["rms_difference"] >= 1e-4
        and any(detectable[k] and effects[k]["relative_l2"] >= 0.01 for k in ("gradient", "update"))
    )
    outcome = (
        "MATERIAL_LOCAL_DRIFT"
        if material
        else "DETECTABLE_SUBTHRESHOLD"
        if any(detectable.values())
        else "UNRESOLVED_AT_NUMERIC_FLOOR"
    )
    if frozen_hash != cap.sha(output / "independent_floors.json"):
        raise ValueError("independent floor changed")
    torch.save(
        {
            "reference_h": reference,
            "incoming_h_override": intervention_pre["h"],
            "unchanged_pre_source": "original externally hash-bound pre.pt; all pre fields except h verified",
            "counterfactual_post": alternative,
        },
        output / "binary.pt",
    )
    result = {
        "outcome": outcome,
        "effects": effects,
        "detectable": detectable,
        "independent_floors": frozen,
        "unchanged": [
            "all pre state except detached student h",
            "EMA out",
            "teacher/targets",
            "weights/denominators",
            "post RNG and update clocks",
        ],
        "scientific_scope": "local early fresh-panel trajectory only; no scoring, fullfit or component attribution",
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
