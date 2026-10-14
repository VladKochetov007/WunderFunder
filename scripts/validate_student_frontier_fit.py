"""Check the fixed F0 seed-3 student fit receipt before prediction export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MAX_CONTRACT_BYTES = 256_000
MAX_RUN_JSON_BYTES = 4_000_000
MAX_CHECKPOINT_BYTES = 64_000_000
RECORDED_SOURCE_PATHS = {
    "scripts/train_rnn.py",
    "src/wnn/student_frontier.py",
    "src/wnn/student_frontier_order.py",
    "src/wnn/training_sampling.py",
    "src/wnn/nn.py",
    "src/wnn/features.py",
    "src/wnn/distillation_cache.py",
    "src/wnn/splits.py",
    "src/wnn/propensity.py",
    "src/wnn/training_dynamics.py",
    "src/wnn/optim.py",
}


def sha256_file(path: Path) -> str:
    """Hash a bounded-size immutable fit artifact without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, maximum_bytes: int) -> dict[str, object]:
    """Read a small JSON receipt only after enforcing its size bound."""
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise ValueError(f"{path} exceeds its {maximum_bytes}-byte read bound")
    with path.open("rb") as source:
        raw = source.read(maximum_bytes + 1)
    if len(raw) != size:
        raise ValueError(f"{path} changed while being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_equal(actual: object, expected: object, description: str) -> None:
    """Reject any fit receipt field that differs from the frozen recipe."""
    if actual != expected:
        raise ValueError(f"{description} differs: expected {expected!r}, got {actual!r}")


def ordered_group_sha256(groups: list[object]) -> str:
    """Hash an ordered row-group list in the trainer's int64 little-endian form."""
    digest = hashlib.sha256()
    for group in groups:
        if not isinstance(group, int):
            raise ValueError("fit row-group identities must be integers")
        digest.update(int(group).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def require_membership(actual: object, expected: list[object], description: str) -> list[object]:
    """Check exact group membership without confusing order with split identity."""
    if not isinstance(actual, list) or len(actual) != len(set(actual)):
        raise ValueError(f"{description} must be an ordered list without duplicates")
    if set(actual) != set(expected):
        raise ValueError(f"{description} has different source-qualified membership")
    return actual


def validate_fit_receipt(
    root: Path,
    run: str,
    contract_path: Path,
    expected_contract_sha256: str,
    hidden: int,
    layers: int,
    output_path: Path,
) -> dict[str, object]:
    """Verify fixed epoch, exact exposure/order, finite updates and checkpoint."""
    contract = read_json(contract_path, MAX_CONTRACT_BYTES)
    contract_sha256 = sha256_file(contract_path)
    require_equal(contract_sha256, expected_contract_sha256, "consumer contract SHA256")
    require_equal(contract.get("id"), "cpu-student-frontier-f0-seed3-b8-r2-consumer", "contract id")

    if not run or Path(run).name != run:
        raise ValueError("run must be a single safe artifact-directory name")
    run_root = root / "artifacts" / "runs" / run
    history_path = run_root / "history.json"
    provenance_path = run_root / "provenance.json"
    checkpoint_path = run_root / "model_ep0.pt"
    history_size = history_path.stat().st_size
    checkpoint_size = checkpoint_path.stat().st_size
    if history_size <= 0 or history_size > MAX_RUN_JSON_BYTES:
        raise ValueError("history.json exceeds its read bound")
    if checkpoint_size <= 0 or checkpoint_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("epoch-0 checkpoint exceeds its read/hash bound")
    with history_path.open("rb") as source:
        history_raw = source.read(MAX_RUN_JSON_BYTES + 1)
    if len(history_raw) != history_size:
        raise ValueError("history.json changed while being read")
    history = json.loads(history_raw)
    if not isinstance(history, list) or len(history) != 1 or not isinstance(history[0], dict):
        raise ValueError("expected exactly one completed epoch-0 history entry")
    provenance = read_json(provenance_path, MAX_RUN_JSON_BYTES)

    args = provenance.get("args")
    if not isinstance(args, dict):
        raise ValueError("provenance lacks serialized trainer arguments")
    recipe = contract["recipe"]
    stage = contract["stages"]["f0_seed3_fit"]
    optimizer = recipe["optimizer"]
    for name, expected in {
        "name": run,
        "student_frontier_contract": str(contract_path.resolve()),
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": hidden,
        "layers": layers,
        "dropout": 0.0,
        "seed": 3,
        "precision": recipe["precision"],
        "batch": recipe["batch_size"],
        "chunk": recipe["chunk_rows"],
        "epochs": stage["epochs"],
        "checkpoint_policy": recipe["checkpoint_policy"],
        "checkpoint_epoch": stage["checkpoint_epoch"],
        "no_eval_during_fit": recipe["no_eval_during_fit"],
        "profile_steps": stage["profile_steps"],
        "max_batches": stage["max_batches"],
        "batch_order": "seeded",
        "train_source": recipe["train_source"],
        "train_seqs": recipe["train_seqs_argument"],
        "valid_train_fraction": recipe["valid_train_fraction"],
        "valid_repeat": recipe["valid_repeat"],
        "valid_fold": recipe["valid_fold"],
        "n_valid_folds": recipe["n_valid_folds"],
        "full_valid": recipe["full_valid"],
        "loss": "wmse",
        "loss_norm": "chunk",
        "distill_cache_variant": "homogeneous",
        "distill_cache_scale": 1.0,
        "distill_alpha": 0.9,
        "propensity": contract["propensity"]["run"],
        "prop_cache": True,
        "prop_power": contract["propensity"]["power"],
        "prop_eps": contract["propensity"]["epsilon"],
        "valid_true_mask": recipe["propensity"]["valid_true_mask_epsilon"],
        "optim": str(optimizer["name"]).lower(),
        "sched": str(recipe["schedule"]["name"]).replace("LR", "").lower(),
        "lr": optimizer["learning_rate"],
        "beta2": optimizer["beta2"],
        "wd": optimizer["weight_decay"],
        "clip": optimizer["gradient_clip_norm"],
        "fused_adam": optimizer["fused"],
        "sync_free": optimizer["sync_free"],
        "warmup_frac": recipe["schedule"]["warmup_fraction"],
        "head_lr_mult": 1.0,
        "grad_stats": optimizer["gradient_stats"],
        "deterministic": True,
    }.items():
        require_equal(args.get(name), expected, f"trainer argument {name}")
    require_equal(optimizer.get("beta1"), 0.9, "AdamW constructor beta1")
    if "beta1" in args:
        raise ValueError("beta1 must remain the trainer's source-bound default, not a CLI override")
    require_equal(recipe["schedule"].get("cycle_momentum"), True, "OneCycle momentum cycling")
    require_equal(recipe["schedule"].get("total_updates"), 14920, "OneCycle update horizon")
    if (
        Path(str(args.get("split_manifest"))).resolve()
        != (root / contract["split"]["path"]).resolve()
    ):
        raise ValueError("trainer split-manifest path differs from the frozen fold")
    if (
        Path(str(args.get("distill_cache_root"))).resolve()
        != (root / contract["cache"]["root"]).resolve()
    ):
        raise ValueError("trainer distillation-cache path differs from the frozen producer")

    require_equal(
        provenance.get("split_manifest_sha256"), contract["split"]["sha256"], "split hash"
    )
    expected_groups = contract["split"]["source_qualified_group_ids"]
    fit_groups = provenance.get("fit_row_groups")
    if not isinstance(fit_groups, dict):
        raise ValueError("provenance lacks source-qualified fit groups")
    fit_train = require_membership(
        fit_groups.get("train"), expected_groups["train"], "train fit groups"
    )
    fit_valid = require_membership(
        fit_groups.get("valid"), expected_groups["valid_fit"], "valid fit groups"
    )
    require_membership(
        provenance.get("evaluation_row_groups"),
        expected_groups["valid_evaluation"],
        "evaluation groups",
    )
    fit_identity = provenance.get("fit_membership_identity")
    if not isinstance(fit_identity, dict):
        raise ValueError("provenance lacks fit-membership identity")
    for split_name, groups in (("train", fit_train), ("valid", fit_valid)):
        require_equal(
            fit_identity.get(split_name),
            {"count": len(groups), "ordered_int64_le_sha256": ordered_group_sha256(groups)},
            f"{split_name} fit-membership identity",
        )
    require_equal(fit_identity.get("valid_repeat"), 1, "valid fit repetition")
    require_equal(
        fit_identity.get("dataset_length"),
        len(fit_train) + len(fit_valid),
        "fit dataset length",
    )

    frontier = provenance.get("student_frontier")
    if not isinstance(frontier, dict):
        raise ValueError("provenance lacks student-frontier identity")
    require_equal(frontier.get("contract_id"), contract["id"], "frontier contract id")
    require_equal(frontier.get("contract_sha256"), contract_sha256, "frontier contract SHA256")
    require_equal(frontier.get("stage"), "f0_seed3_fit", "frontier stage")
    require_equal(frontier.get("stage_settings"), stage, "frontier stage settings")
    source_root = Path(contract["source_root"]).resolve(strict=True)
    source_hashes = contract.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("contract lacks pinned source hashes")
    for relative_path, expected_hash in source_hashes.items():
        require_equal(
            sha256_file(source_root / relative_path),
            expected_hash,
            f"pinned source {relative_path}",
        )
    recorded_sources = frontier.get("source_sha256")
    if not isinstance(recorded_sources, dict):
        raise ValueError("provenance lacks recorded trainer source hashes")
    require_equal(
        set(recorded_sources), RECORDED_SOURCE_PATHS, "recorded trainer source hash file set"
    )
    for relative_path, actual_hash in recorded_sources.items():
        if relative_path not in source_hashes:
            raise ValueError(f"provenance records an unpinned source: {relative_path}")
        require_equal(actual_hash, source_hashes[relative_path], f"recorded source {relative_path}")
    training_dynamics_sha256 = recorded_sources["src/wnn/training_dynamics.py"]
    order = frontier.get("sample_order")
    if not isinstance(order, dict):
        raise ValueError("provenance lacks source-qualified sampler order")
    expected_order = contract["sample_order"]
    expected_observed_keys = set(expected_order) - {"sampler", "source_qualified_row_groups"}
    require_equal(set(order), expected_observed_keys, "sampler-order fields")
    for key, value in order.items():
        require_equal(value, expected_order[key], f"sampler order {key}")
    require_equal(
        order.get("train_group_order_sha256"),
        ordered_group_sha256(fit_train),
        "ordered train fit groups",
    )
    require_equal(
        order.get("valid_group_order_sha256"),
        ordered_group_sha256(fit_valid),
        "ordered valid fit groups",
    )
    require_equal(order.get("exposed_sequence_count"), 11936, "exposed sequence count")
    require_equal(order.get("dropped_sequence_count"), 4, "drop-last count")
    require_equal(order.get("optimizer_updates_per_epoch"), 14920, "planned update count")

    cache_binding = frontier.get("cache_binding")
    if not isinstance(cache_binding, dict):
        raise ValueError("provenance lacks strict producer-cache binding")
    require_equal(
        cache_binding.get("cache_metadata_sha256"),
        contract["cache"]["metadata_sha256"],
        "cache metadata hash",
    )
    require_equal(cache_binding.get("stage"), "f0_seed3_fit", "cache-binding stage")
    require_equal(
        cache_binding.get("verified_target_files"),
        contract["cache"]["verified_targets"],
        "verified target hashes",
    )
    require_equal(
        cache_binding.get("source_qualified_group_identities"),
        contract["split"]["groups"],
        "cache-binding source-qualified group identities",
    )
    require_equal(
        cache_binding.get("teacher_sources"), contract["cache"]["teachers"], "teacher ancestry"
    )
    require_equal(
        cache_binding.get("target_formula"),
        contract["cache"]["target_contract"]["formula"],
        "teacher-target formula",
    )
    require_equal(
        cache_binding.get("target_semantics"),
        contract["target_semantics"],
        "teacher-target semantics",
    )
    recorded_propensity = cache_binding.get("propensity")
    expected_propensity = contract["propensity"]
    if not isinstance(recorded_propensity, dict):
        raise ValueError("cache binding lacks propensity ancestry")
    for recorded_name, contract_name in (
        ("run", "run"),
        ("checkpoint_sha256", "checkpoint_sha256"),
        ("provenance_sha256", "provenance_sha256"),
        ("cache_files", "caches"),
        ("power", "power"),
        ("valid_true_mask_eps", "valid_true_mask_eps"),
    ):
        require_equal(
            recorded_propensity.get(recorded_name),
            expected_propensity[contract_name],
            f"propensity ancestry {recorded_name}",
        )

    expected_normalization = contract["normalization_source"]["normalization_identity"]
    normalization = provenance.get("normalization_identity")
    require_equal(normalization, expected_normalization, "common normalization identity")
    normalization_source = frontier.get("normalization_source")
    if not isinstance(normalization_source, dict):
        raise ValueError("provenance lacks normalization-source identity")
    expected_normalization_source = contract["normalization_source"]
    for name in (
        "run",
        "epoch",
        "checkpoint_file",
        "checkpoint_sha256",
        "provenance_sha256",
        "normalization_identity",
    ):
        require_equal(
            normalization_source.get(name),
            expected_normalization_source[name],
            f"normalization source {name}",
        )
    require_equal(
        normalization_source.get("normalization_identity"), normalization, "installed normalization"
    )
    require_equal(
        normalization_source.get("loaded_fields_only"),
        ["mean", "std"],
        "normalization source loaded fields",
    )
    require_equal(
        normalization_source.get("learned_weights_loaded"), False, "normalization-only source use"
    )

    epoch = history[0]
    require_equal(epoch.get("epoch"), 0, "history epoch")
    require_equal(epoch.get("evaluation"), "withheld", "fit evaluation policy")
    diagnostics = epoch.get("update_diagnostics")
    if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get("epoch"), dict):
        raise ValueError("history lacks epoch update diagnostics")
    counts = diagnostics["epoch"]
    history_order_identity = diagnostics.get("sample_order_identity")
    if not isinstance(history_order_identity, dict):
        raise ValueError("history lacks the epoch sampler identity")
    require_equal(
        history_order_identity.get("student_frontier_order"),
        order,
        "history/provenance sampler identity",
    )
    for name, expected in {
        "completed_batches": 1492,
        "optimizer_attempts": 14920,
        "optimizer_completed_updates": 14920,
        "amp_skipped_updates": 0,
    }.items():
        require_equal(counts.get(name), expected, f"update count {name}")
    grad_stats = diagnostics.get("grad_stats")
    if not isinstance(grad_stats, dict):
        raise ValueError("history lacks finite-gradient diagnostics")
    require_equal(grad_stats.get("steps"), 14920, "gradient-stat steps")
    require_equal(grad_stats.get("nonfinite_steps"), 0, "nonfinite gradient steps")
    require_equal(provenance.get("selected_epoch"), 0, "selected fixed epoch")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    hashes = provenance.get("checkpoint_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("provenance lacks checkpoint hash map")
    require_equal(hashes.get("model_ep0.pt"), checkpoint_sha256, "epoch-0 checkpoint hash")

    import inspect

    import torch

    update_counter = provenance.get("update_counter_implementation")
    if not isinstance(update_counter, dict):
        raise ValueError("provenance lacks the trainer's recorded PyTorch version")
    require_equal(update_counter.get("torch_version"), torch.__version__, "PyTorch version")
    onecycle_signature = inspect.signature(torch.optim.lr_scheduler.OneCycleLR).parameters
    scheduler_defaults = {
        "anneal_strategy": "cos",
        "base_momentum": 0.85,
        "div_factor": 25.0,
        "max_momentum": 0.95,
        "three_phase": False,
    }
    for name, expected in scheduler_defaults.items():
        require_equal(onecycle_signature[name].default, expected, f"OneCycleLR default {name}")
    if args["optim"] == "sgd":
        raise ValueError("the frozen AdamW recipe must enable OneCycle momentum cycling")
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    schedule_optimizer = torch.optim.AdamW(
        [parameter],
        lr=float(optimizer["learning_rate"]),
        betas=(float(optimizer["beta1"]), float(optimizer["beta2"])),
    )
    schedule = recipe["schedule"]
    schedule_steps = int(schedule["total_updates"])
    schedule_lr = torch.optim.lr_scheduler.OneCycleLR(
        schedule_optimizer,
        max_lr=float(optimizer["learning_rate"]),
        total_steps=schedule_steps,
        pct_start=float(schedule["warmup_fraction"]),
        final_div_factor=float(schedule["final_div_factor"]),
        cycle_momentum=bool(schedule["cycle_momentum"]),
    )
    momentum_values = [float(schedule_optimizer.param_groups[0]["betas"][0])]
    for _ in range(schedule_steps - 1):
        parameter.grad = torch.zeros_like(parameter)
        schedule_optimizer.step()
        schedule_lr.step()
        momentum_values.append(float(schedule_optimizer.param_groups[0]["betas"][0]))
    for value, expected, description in (
        (momentum_values[0], 0.95, "OneCycle initial beta1"),
        (min(momentum_values), 0.85, "OneCycle minimum beta1"),
        (momentum_values[-1], 0.95, "OneCycle terminal beta1"),
    ):
        if abs(value - expected) > 1e-12:
            raise ValueError(f"{description} differs: expected {expected!r}, got {value!r}")
    del schedule_lr, schedule_optimizer, parameter

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require_equal(checkpoint.get("epoch"), 0, "embedded checkpoint epoch")
    require_equal(checkpoint.get("args"), args, "checkpoint/trainer arguments")
    embedded = checkpoint.get("provenance")
    if not isinstance(embedded, dict):
        raise ValueError("epoch-0 checkpoint lacks embedded provenance")
    for field in (
        "split_manifest_sha256",
        "fit_row_groups",
        "evaluation_row_groups",
        "fit_membership_identity",
        "normalization_identity",
        "student_frontier",
        "update_diagnostics",
    ):
        require_equal(embedded.get(field), provenance.get(field), f"checkpoint/provenance {field}")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("epoch-0 checkpoint lacks its model tensor mapping")
    finite_model_tensors = 0
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"checkpoint model entry {name!r} is not a tensor")
        if torch.is_floating_point(tensor):
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"checkpoint model tensor {name!r} contains nonfinite values")
            finite_model_tensors += 1
    del checkpoint

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {output_path}")
    receipt = {
        "format": "wnn.student-frontier-fit-receipt-validation",
        "version": 1,
        "run": run,
        "contract_sha256": contract_sha256,
        "shape": [hidden, layers],
        "epoch": 0,
        "exposed_sequences": 11936,
        "dropped_tail_sequences": 4,
        "completed_batches": 1492,
        "optimizer_attempts": 14920,
        "optimizer_completed_updates": 14920,
        "amp_skipped_updates": 0,
        "normalization_identity": normalization,
        "normalization_source": normalization_source,
        "propensity_source": recorded_propensity,
        "optimizer_semantics": {
            "name": str(optimizer["name"]).lower(),
            "constructor_beta1": optimizer["beta1"],
            "beta1_cli_override": None,
            "beta2": optimizer["beta2"],
            "schedule": "OneCycleLR",
            "cycle_momentum": True,
            "beta1_path": [momentum_values[0], min(momentum_values), momentum_values[-1]],
            "schedule_updates": schedule_steps,
            "training_dynamics_sha256": training_dynamics_sha256,
            "torch_version": torch.__version__,
            "validation": "source-bound default plus OneCycleLR runtime reconstruction",
        },
        "batch_membership_sha256": order["batch_membership_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "finite_model_tensors_checked": finite_model_tensors,
        "gradient_stats_steps": grad_stats["steps"],
        "nonfinite_gradient_steps": grad_stats["nonfinite_steps"],
        "validation": (
            "passed; counts, recorded gradients and terminal model tensors checked; "
            "per-step loss finiteness is not separately serialized; no quality evaluation performed"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    """Validate one terminal fit before the sequence admits prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate_fit_receipt(
        args.root.resolve(strict=True),
        args.run,
        args.contract.resolve(strict=True),
        args.contract_sha256,
        args.hidden,
        args.layers,
        args.out.resolve(),
    )
    print(json.dumps({"run": receipt["run"], "validation": receipt["validation"]}))


if __name__ == "__main__":
    main()
