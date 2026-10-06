"""Resolve a fixed-denominator template only from its bound fit-only report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
TOKEN = "${calibration.reference_mass}"
RUN_BLOCK = re.compile(r"_s(?P<seed>[34])_f(?P<fold>[01])\Z")
TEACHER_HASHES = (
    "c20260929_since_s3_f{fold}",
    "c20260929_since_s4_f{fold}",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_hashes(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"calibration contract is missing {label}")
    result: dict[str, str] = {}
    for name, identity in value.items():
        if not isinstance(name, str) or not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError(f"invalid {label} identity")
        result[name] = identity
    return result


def _relative_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ValueError(f"invalid repository-relative path: {relative}")
    result = root / path
    if not result.is_file():
        raise ValueError(f"required calibration dependency is missing: {relative}")
    return result


def _membership_identity(groups: object) -> dict[str, object]:
    if not isinstance(groups, list) or any(type(value) is not int for value in groups):
        raise ValueError("fold fitting groups must be an integer list")
    values = np.asarray(groups, dtype="<i8")
    return {
        "count": int(len(values)),
        "ordered_int64_le_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _verify_source_closure(root: Path, report_sources: object) -> None:
    declared = _required_hashes(report_sources, "trainer sources")
    package = root / "src/wnn"
    expected = {
        str(path.relative_to(package)): _sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    expected["trainer"] = _sha256(root / "scripts/train_rnn.py")
    if declared != expected:
        raise ValueError("calibration trainer source closure differs from current source tree")


def _verify_teacher_and_cache_inputs(
    root: Path,
    *,
    contract: dict[str, object],
    options: dict[str, object],
    fold: int,
    manifest: dict[str, object],
) -> None:
    """Match current teacher, propensity, target and component cache contents."""
    split_sha = str(options["split_manifest_sha256"])
    fold_fit = manifest["fit"]
    assert isinstance(fold_fit, dict)
    target_row_order: dict[str, list[int]] = {}
    for split in ("train", "valid"):
        with np.load(_relative_file(root, f"cache/{split}_targets.npz"), allow_pickle=False) as target_cache:
            target_row_order[split] = np.asarray(target_cache["row_group"], dtype=np.int64).tolist()
    expected_prediction_hashes: dict[str, str] = {}
    for template in TEACHER_HASHES:
        run = template.format(fold=fold)
        provenance_path = _relative_file(root, f"artifacts/runs/{run}/provenance.json")
        model_path = _relative_file(root, f"artifacts/runs/{run}/model.pt")
        epoch3_path = _relative_file(root, f"artifacts/runs/{run}/model_ep3.pt")
        provenance = json.loads(provenance_path.read_text())
        model_sha = _sha256(model_path)
        epoch3_sha = _sha256(epoch3_path)
        args = provenance.get("args", {})
        match = re.search(r"_s([34])_f", run)
        if (
            match is None
            or not isinstance(args, dict)
            or args.get("seed") != int(match.group(1))
            or args.get("valid_fold") != fold
            or args.get("split_manifest_sha256") != split_sha
            or args.get("checkpoint_policy") != "fixed"
            or args.get("checkpoint_epoch") != 3
            or provenance.get("split_manifest_sha256") != split_sha
        ):
            raise ValueError(f"teacher provenance is not bound to fold {fold}: {run}")
        checkpoints = provenance.get("checkpoint_sha256")
        if (
            not isinstance(checkpoints, dict)
            or checkpoints.get("model.pt") != model_sha
            or checkpoints.get("model_ep3.pt") != epoch3_sha
        ):
            raise ValueError(f"teacher model/epoch-3 checkpoint hash differs: {run}")
        fit_groups = provenance.get("fit_row_groups")
        if not isinstance(fit_groups, dict):
            raise ValueError(f"teacher provenance lacks fitting membership: {run}")
        expected_train = np.random.default_rng(args["seed"]).permutation(
            np.asarray(fold_fit["train"], dtype=np.int64)
        ).tolist()
        if fit_groups.get("train") != expected_train or fit_groups.get("valid") != fold_fit["valid"]:
            raise ValueError(f"teacher fitting membership differs from fold/seed: {run}")
        for split in ("train", "valid"):
            relative = f"cache/pred_{run}_{split}.npy"
            sidecar_relative = f"cache/pred_{run}_{split}.json"
            prediction_sha = _sha256(_relative_file(root, relative))
            sidecar = json.loads(_relative_file(root, sidecar_relative).read_text())
            if (
                sidecar.get("run") != run
                or sidecar.get("split") != split
                or sidecar.get("epoch") != 3
                or sidecar.get("split_manifest_sha256") != split_sha
                or sidecar.get("checkpoint_sha256") != epoch3_sha
                or sidecar.get("prediction_sha256") != prediction_sha
                or sidecar.get("row_groups") != target_row_order[split]
            ):
                raise ValueError(f"teacher sidecar does not bind current prediction bytes: {relative}")
            expected_prediction_hashes[relative.removeprefix("cache/")] = prediction_sha

    reported_predictions = _required_hashes(contract.get("teacher_predictions"), "teacher predictions")
    if reported_predictions != expected_prediction_hashes:
        raise ValueError("calibration teacher prediction hashes differ from current fold caches")

    caches = _required_hashes(contract.get("consumed_caches"), "consumed caches")
    propensity_name = f"c20260929_prop_f{fold}"
    prop_model_path = _relative_file(root, f"artifacts/runs/{propensity_name}/model.pt")
    prop_model_sha = _sha256(prop_model_path)
    ancestors = contract.get("ancestors")
    if not isinstance(ancestors, dict) or ancestors.get("propensity_checkpoint_sha256") != prop_model_sha:
        raise ValueError("calibration propensity checkpoint differs from current model")
    prop_provenance = json.loads(
        _relative_file(root, f"artifacts/runs/{propensity_name}/provenance.json").read_text()
    )
    if prop_provenance.get("checkpoint_sha256", {}).get("model.pt") != prop_model_sha:
        raise ValueError("propensity provenance does not bind the current model")

    expected_caches: dict[str, str] = {}
    for split in ("train", "valid"):
        target_relative = f"cache/{split}_targets.npz"
        expected_caches[target_relative.removeprefix("cache/")] = _sha256(
            _relative_file(root, target_relative)
        )
        prop_array = f"cache/propcache_{propensity_name}_{split}.npy"
        prop_json = f"cache/propcache_{propensity_name}_{split}.json"
        prop_sha = _sha256(_relative_file(root, prop_array))
        metadata = json.loads(_relative_file(root, prop_json).read_text())
        if (
            metadata.get("model") != propensity_name
            or metadata.get("model_sha256") != prop_model_sha
            or metadata.get("split") != split
        ):
            raise ValueError(f"propensity sidecar does not bind current cache: {split}")
        expected_caches[prop_array.removeprefix("cache/")] = prop_sha
        expected_caches[prop_json.removeprefix("cache/")] = _sha256(_relative_file(root, prop_json))
    if caches != expected_caches:
        raise ValueError("calibration consumed-cache hashes differ from current target/propensity inputs")

    portable_sources = manifest.get("sources")
    if not isinstance(portable_sources, dict):
        raise ValueError("fold manifest is missing source identities")
    for relative in ("cache/chains.npz", "cache/chain_folds.npz"):
        record = portable_sources.get(relative)
        if not isinstance(record, dict) or _sha256(_relative_file(root, relative)) != record.get("sha256"):
            raise ValueError(f"current chain source identity differs: {relative}")


def validate_report(
    report: object,
    *,
    seed: int,
    fold: int,
    calibration_path: str,
    root: Path = ROOT,
) -> tuple[float, str]:
    """Check mass schema, fixed recipe/fold, source and consumed-input identity."""
    if not isinstance(report, dict):
        raise ValueError("calibration report must be an object")
    if report.get("format") != "wnn.distill-reference-mass" or report.get("version") != 1:
        raise ValueError("unsupported calibration report format")
    mass = report.get("reference_mass")
    if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(float(mass)) or float(mass) <= 0:
        raise ValueError("calibration reference_mass must be finite and positive")
    if type(report.get("calibration_epochs")) is not int or report["calibration_epochs"] != 1:
        raise ValueError("calibration must contain exactly one fit-only pass")
    if report.get("estimator") != "mean over actual sampled fit batches and update windows":
        raise ValueError("calibration estimator does not match the preregistered fit-batch mean")
    updates = report.get("updates")
    if type(updates) is not int or updates <= 0:
        raise ValueError("calibration update count must be a positive integer")
    for name in ("minimum", "maximum", "standard_deviation"):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"calibration {name} must be finite")
    if (
        report["minimum"] < 0
        or report["maximum"] < report["minimum"]
        or not report["minimum"] <= float(mass) <= report["maximum"]
        or report["standard_deviation"] < 0
        or type(report.get("zero_mass_updates")) is not int
        or not 0 <= report["zero_mass_updates"] <= updates
    ):
        raise ValueError("calibration mass summary is internally inconsistent")
    observations = report.get("update_masses")
    if (
        not isinstance(observations, dict)
        or observations.get("format") != "wnn.float64-le-update-masses"
        or observations.get("version") != 1
        or observations.get("dtype") != "<f8"
    ):
        raise ValueError("calibration lacks replayable per-update mass observations")
    values = observations.get("values")
    if (
        not isinstance(values, list)
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)
        or type(observations.get("count")) is not int
        or observations["count"] != updates
        or len(values) != updates
    ):
        raise ValueError("calibration per-update mass count disagrees with update summary")
    mass_values = np.asarray(values, dtype="<f8")
    if not np.isfinite(mass_values).all() or (mass_values < 0).any():
        raise ValueError("calibration per-update masses must be finite and nonnegative")
    observation_sha = observations.get("sha256")
    if (
        not isinstance(observation_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", observation_sha)
        or hashlib.sha256(mass_values.tobytes()).hexdigest() != observation_sha
    ):
        raise ValueError("calibration per-update mass identity is invalid")
    recomputed = {
        "reference_mass": float(mass_values.mean()),
        "minimum": float(mass_values.min()),
        "maximum": float(mass_values.max()),
        "standard_deviation": float(mass_values.std()),
        "zero_mass_updates": int((mass_values == 0).sum()),
    }
    for field in ("reference_mass", "minimum", "maximum", "standard_deviation"):
        if not math.isclose(float(report[field]), recomputed[field], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"calibration {field} differs from recorded per-update masses")
    if report["zero_mass_updates"] != recomputed["zero_mass_updates"]:
        raise ValueError("calibration zero-mass count differs from recorded per-update masses")

    contract = report.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("calibration report is missing its trainer contract")
    options = contract.get("options")
    if not isinstance(options, dict):
        raise ValueError("calibration report is missing bound trainer options")
    if (
        options.get("seed") != seed
        or options.get("valid_fold") != fold
        or options.get("train_source") != "train+valid"
        or options.get("device") != "cpu"
        or options.get("distill") != f"c20260929_since_s3_f{fold},c20260929_since_s4_f{fold}"
        or options.get("no_eval_during_fit") is not True
        or options.get("batch_order") != "seeded"
        or options.get("loss") != "wmse"
        or options.get("hidden") != 256
        or options.get("layers") != 2
        or options.get("batch") != 32
        or options.get("chunk") != 1000
        or options.get("update_rows") != 0
        or options.get("origin_view") != "off"
        or options.get("epochs") != 4
        or options.get("checkpoint_policy") != "fixed"
        or options.get("checkpoint_epoch") != 3
        or options.get("optim") != "muon"
        or options.get("muon_scope") != "rnn"
        or options.get("ema_teacher") != 1.0
        or options.get("ema_decay") != 0.999
        or options.get("eval_ema") is not True
        or options.get("aux_struct") != 0.3
        or options.get("aux_next") != 0.3
        or options.get("distill_denominator") != "chunk"
        or options.get("distill_reference_mass") != 0.0
        or options.get("distill_alpha") != 0.7
        or options.get("distill_boot_h") != 8
        or options.get("aux_sign") != 0.0
        or options.get("distill_reanchor") != 0.0
        or options.get("calibration_epochs") != 1
        or options.get("calibrate_distill_reference") != calibration_path
        or options.get("propensity") != f"c20260929_prop_f{fold}"
        or options.get("prop_cache") is not True
        or options.get("fast") is not False
        or options.get("pin_memory") is not False
        or options.get("fused_adam") is not False
        or options.get("since_weight") != 32.0
        or options.get("since_floor") != 0.25
        or options.get("valid_true_mask") != 0.03
        or options.get("prop_power") != 2.0
        or options.get("prop_eps") != 0.03
        or options.get("valid_repeat") != 1
        or options.get("valid_train_fraction") != 1.0
        or options.get("include_heldout_valid") is not False
    ):
        raise ValueError("calibration recipe, seed/fold, or fitting-only identity disagrees")
    manifest = root / f"research/campaign_2026-09-29/artifacts/splits/fold{fold}.json"
    if not manifest.is_file() or options.get("split_manifest_sha256") != _sha256(manifest):
        raise ValueError("calibration report is not bound to the current fold manifest")
    manifest_value = json.loads(manifest.read_text())
    portable_manifest_path = root / "research/campaign_2026-10-04/portable_inputs.json"
    portable_manifest = json.loads(_relative_file(root, str(portable_manifest_path.relative_to(root))).read_text())
    if portable_manifest.get("split_manifest_sha256") != _sha256(manifest):
        raise ValueError("portable input registry is not bound to the current fold manifest")
    manifest_sources = manifest_value.get("sources")
    portable_sources = portable_manifest.get("sources")
    if not isinstance(manifest_sources, dict) or not isinstance(portable_sources, dict):
        raise ValueError("fold and portable manifests must record source identities")
    datasets = contract.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != {"train", "valid"}:
        raise ValueError("calibration report must bind train and valid source data")
    for split in ("train", "valid"):
        record = datasets[split]
        logical = f"datasets/{split}.parquet"
        expected_source = portable_sources.get(logical)
        frozen_source = manifest_sources.get(logical)
        if not isinstance(expected_source, dict) or not isinstance(frozen_source, dict):
            raise ValueError(f"portable dataset identity is missing: {logical}")
        original = expected_source.get("original_identity")
        if frozen_source != original or not isinstance(original, dict):
            raise ValueError(f"portable and fold dataset identities differ: {logical}")
        source_path = _relative_file(root, logical)
        if source_path.stat().st_size != original.get("bytes"):
            raise ValueError(f"current dataset byte length differs: {logical}")
        logical_sha = expected_source.get("sha256")
        if not isinstance(logical_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", logical_sha):
            raise ValueError(f"portable dataset logical identity is invalid: {logical}")
        if _sha256(source_path) != logical_sha:
            raise ValueError(f"current dataset content differs from portable identity: {logical}")
        expected_dataset = {"bytes": source_path.stat().st_size, "sha256": logical_sha}
        if record != expected_dataset:
            raise ValueError(f"calibration dataset logical identity differs: {logical}")

    fit = manifest_value.get("fit")
    if not isinstance(fit, dict):
        raise ValueError("fold manifest is missing fitting groups")
    expected_train = np.random.default_rng(seed).permutation(np.asarray(fit["train"], dtype=np.int64)).tolist()
    expected_membership = {
        "train": _membership_identity(expected_train),
        "valid": _membership_identity(fit["valid"]),
        "valid_repeat": 1,
        "dataset_length": len(fit["train"]) + len(fit["valid"]),
    }
    if contract.get("fit_membership") != expected_membership:
        raise ValueError("calibration fitting membership count/order does not match seed and fold")
    _verify_source_closure(root, contract.get("sources"))
    _verify_teacher_and_cache_inputs(
        root,
        contract=contract,
        options=options,
        fold=fold,
        manifest=manifest_value,
    )
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("devices") != []:
        raise ValueError("calibration report was not produced with CUDA hidden")
    order = report.get("sample_order")
    if not isinstance(order, list) or len(order) != 1 or not isinstance(order[0], dict):
        raise ValueError("calibration report does not record one pass of sample order")
    if not re.fullmatch(r"[0-9a-f]{64}", str(order[0].get("retained_dataset_indices_sha256", ""))):
        raise ValueError("calibration sample order lacks an identity hash")
    retained = expected_membership["dataset_length"] // 32 * 32
    expected_updates = retained // 32 * math.ceil(20_000 / 1000)
    if (
        order[0].get("retained_count") != retained
        or order[0].get("unique_retained_count") != retained
        or order[0].get("sampler_class") != "EpochOrderSampler"
        or order[0].get("batch_size") != 32
        or order[0].get("drop_last") is not True
        or order[0].get("num_workers") != 0
        or order[0].get("persistent_workers") is not False
        or len(order[0].get("dropped_dataset_indices", [])) != expected_membership["dataset_length"] - retained
        or any(type(value) is not int for value in order[0].get("dropped_dataset_indices", []))
        or len(set(order[0].get("dropped_dataset_indices", [])))
        != len(order[0].get("dropped_dataset_indices", []))
        or any(
            value < 0 or value >= expected_membership["dataset_length"]
            for value in order[0].get("dropped_dataset_indices", [])
        )
        or updates != expected_updates
    ):
        raise ValueError("calibration sample order or update count does not cover the full fit pool exactly once")
    return float(mass), _sha256_bytes(json.dumps(report, sort_keys=True, separators=(",", ":")).encode())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resolve_template(
    template_path: Path,
    calibration_file: Path,
    *,
    output_path: Path | None = None,
    root: Path = ROOT,
) -> Path:
    """Write a `.draft.json` only after matching its calibration dependency."""
    template = json.loads(template_path.read_text())
    if not isinstance(template, dict) or not isinstance(template.get("argv"), list):
        raise ValueError("template is not a job specification")
    if not template_path.name.endswith(".template.json"):
        raise ValueError("input must use the explicit .template.json suffix")
    match = RUN_BLOCK.search(str(template.get("name", "")))
    if match is None:
        raise ValueError("template name must bind a registered seed/fold")
    seed, fold = int(match.group("seed")), int(match.group("fold"))
    if seed not in (3, 4) or fold not in (0, 1):
        raise ValueError("template is outside the preregistered seed/fold blocks")
    rel_calibration = str(calibration_file.resolve().relative_to(root.resolve()))
    if rel_calibration not in template.get("inputs", []):
        raise ValueError("calibration report is not a declared template input")
    report = json.loads(calibration_file.read_text())
    mass, report_identity = validate_report(
        report,
        seed=seed,
        fold=fold,
        calibration_path=rel_calibration,
        root=root,
    )
    argv = list(template["argv"])
    plan_path: Path | None = None
    resolved_plan_path: Path | None = None
    resolved_plan_identity: str | None = None
    token_count = argv.count(TOKEN)
    if token_count == 1:
        argv[argv.index(TOKEN)] = repr(mass)
    elif token_count == 0 and "--plan" in argv:
        plan_rel = argv[argv.index("--plan") + 1]
        plan_rel_path = Path(plan_rel)
        if plan_rel_path.is_absolute() or ".." in plan_rel_path.parts or plan_rel_path.as_posix() != plan_rel:
            raise ValueError("probe plan path must stay relative to the experiment root")
        if plan_rel not in template.get("inputs", []):
            raise ValueError("probe plan is not a declared template input")
        plan_path = root / plan_rel
        plan = json.loads(plan_path.read_text())
        total_tokens = 0
        for step in plan.get("steps", []):
            step_argv = step.get("argv")
            if not isinstance(step_argv, list):
                raise ValueError("probe plan steps require argv lists")
            total_tokens += step_argv.count(TOKEN)
            if TOKEN in step_argv:
                step_argv[step_argv.index(TOKEN)] = repr(mass)
        if total_tokens != 1:
            raise ValueError("probe plan must contain one unresolved reference-mass token")
        resolved_plan_path = plan_path.with_name(plan_path.stem + ".resolved.json")
        resolved_plan_rel = str(resolved_plan_path.resolve().relative_to(root.resolve()))
        resolved_plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        argv[argv.index("--plan") + 1] = resolved_plan_rel
        resolved_plan_identity = _sha256(resolved_plan_path)
    else:
        raise ValueError("template must contain one unresolved reference-mass token")

    resolved = dict(template)
    resolved["argv"] = argv
    binding_path = template_path.with_name(template_path.name.replace(".template.json", ".calibration_binding.json"))
    binding = {
        "format": "wnn.e496-calibration-binding",
        "version": 1,
        "template_path": str(template_path.resolve().relative_to(root.resolve())),
        "template_sha256": _sha256(template_path),
        "path": rel_calibration,
        "file_sha256": _sha256(calibration_file),
        "canonical_report_sha256": report_identity,
        "reference_mass": mass,
        "seed": seed,
        "fold": fold,
        "validated_fit_only": True,
    }
    if plan_path is not None and resolved_plan_path is not None:
        binding["plan_path"] = str(plan_path.resolve().relative_to(root.resolve()))
        binding["plan_sha256"] = _sha256(plan_path)
        binding["resolved_plan_path"] = str(resolved_plan_path.resolve().relative_to(root.resolve()))
        binding["resolved_plan_sha256"] = resolved_plan_identity
    destination = output_path or template_path.with_name(template_path.name.replace(".template.json", ".draft.json"))
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.write_text(json.dumps(binding, indent=2, allow_nan=False) + "\n")
    new_inputs = [str(binding_path.resolve().relative_to(root.resolve()))]
    if resolved_plan_path is not None:
        new_inputs.append(str(resolved_plan_path.resolve().relative_to(root.resolve())))
    resolved["inputs"] = sorted(set(resolved["inputs"] + new_inputs))
    destination.write_text(json.dumps(resolved, indent=2, allow_nan=False) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(resolve_template(args.template, args.calibration, output_path=args.output))


if __name__ == "__main__":
    main()
