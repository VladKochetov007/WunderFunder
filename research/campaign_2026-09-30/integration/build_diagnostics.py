"""Build separate instrumentation patch and combined previews without live edits."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(f"diagnostic insertion context changed: {before[:80]}")
    return source.replace(before, after)


def instrument(source: str) -> str:
    source = replace(source, "from wnn.resources import configure_torch, dataloader_budget\n", "from wnn.resources import configure_torch, dataloader_budget\nfrom wnn.run_identity import (EpochUpdateCounter, group_identity, normalization_identity,\n                              sample_order_identity, scaler_source_identity)\n")
    source = replace(source, "    frozen = tuple(v for v in args.freeze.split(\",\") if v)\n", "    provenance[\"normalization_identity\"] = normalization_identity(model)\n    provenance[\"update_counter_implementation\"] = scaler_source_identity()\n    provenance[\"run_identity_source_sha256\"] = file_sha256(ROOT / \"src/wnn/run_identity.py\")\n    provenance[\"fit_membership_identity\"] = {\n        \"train\": group_identity(provenance[\"fit_row_groups\"][\"train\"]),\n        \"valid\": group_identity(provenance[\"fit_row_groups\"][\"valid\"]),\n        \"valid_repeat\": args.valid_repeat, \"dataset_length\": len(train_ds),\n    }\n    frozen = tuple(v for v in args.freeze.split(\",\") if v)\n")
    source = replace(source, "    for epoch in range(args.epochs):\n        model.train()\n", "    cumulative_updates = {}\n    provenance[\"update_diagnostics\"] = []\n    for epoch in range(args.epochs):\n        model.train()\n        epoch_counter = EpochUpdateCounter(device)\n        epoch_order_identity = sample_order_identity(train_loader)\n")
    source = replace(source, "                scaler.unscale_(opt)\n", "                scaler.unscale_(opt)\n                epoch_counter.observe_unscaled(scaler, opt)\n")
    source = replace(source, "                if ema_model is not None:\n                    ema_model.update_parameters(model)\n", "                if ema_model is not None:\n                    ema_model.update_parameters(model)\n                    epoch_counter.ema_updates += 1\n")
    source = replace(source, "            timer.stop()\n", "            timer.stop()\n            epoch_counter.completed_batches += 1\n")
    source = replace(source, "        train_seconds = round(time.time() - t0, 1)\n", "        update_counts = epoch_counter.finish(cumulative_updates)\n        update_counts[\"sample_order_identity\"] = epoch_order_identity\n        update_counts[\"normalization_identity\"] = provenance[\"normalization_identity\"]\n        provenance[\"update_diagnostics\"].append({\"epoch_index\": epoch, **update_counts})\n        provenance[\"cumulative_updates\"] = cumulative_updates.copy()\n        train_seconds = round(time.time() - t0, 1)\n")
    source = replace(source, "        history.append(res)\n", "        res[\"update_diagnostics\"] = update_counts\n        history.append(res)\n")
    ast.parse(source)
    return source


def main() -> None:
    helper = (HERE / "run_identity.py").read_text()
    outputs = {}
    for label, path in [("diagnostics_only", HERE / "base/scripts/train_rnn.py"), ("ssm_spread_diagnostics", HERE / "ssm_spread/scripts/train_rnn.py"), ("all_three_diagnostics", HERE / "all_three/scripts/train_rnn.py")]:
        original, proposed = path.read_text(), instrument(path.read_text())
        own = HERE / label
        (own / "scripts").mkdir(parents=True, exist_ok=True)
        (own / "src/wnn").mkdir(parents=True, exist_ok=True)
        (own / "scripts/train_rnn.py").write_text(proposed)
        (own / "src/wnn/run_identity.py").write_text(helper)
        # Separate diagnostic patch always describes only these extra changes.
        patch = "".join(difflib.unified_diff([], helper.splitlines(keepends=True), fromfile="/dev/null", tofile="b/src/wnn/run_identity.py"))
        patch += "".join(difflib.unified_diff(original.splitlines(keepends=True), proposed.splitlines(keepends=True), fromfile="a/scripts/train_rnn.py", tofile="b/scripts/train_rnn.py"))
        patch_path = HERE / (label + ".patch")
        patch_path.write_text(patch)
        outputs[label] = {"base_sha256": hashlib.sha256(original.encode()).hexdigest(), "proposed_trainer_sha256": hashlib.sha256(proposed.encode()).hexdigest(), "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}
        if label != "diagnostics_only":
            baseline = (HERE / "base/scripts/train_rnn.py").read_text()
            parts = []
            modules = {}
            for module in sorted((path.parent.parent / "src/wnn").glob("*.py")):
                text = module.read_text()
                (own / "src/wnn" / module.name).write_text(text)
                modules[module.name] = hashlib.sha256(text.encode()).hexdigest()
                parts.extend(difflib.unified_diff([], text.splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/src/wnn/{module.name}"))
            parts.extend(difflib.unified_diff([], helper.splitlines(keepends=True), fromfile="/dev/null", tofile="b/src/wnn/run_identity.py"))
            parts.extend(difflib.unified_diff(baseline.splitlines(keepends=True), proposed.splitlines(keepends=True), fromfile="a/scripts/train_rnn.py", tofile="b/scripts/train_rnn.py"))
            full = "".join(parts)
            (HERE / (label + "_full.patch")).write_text(full)
            outputs[label]["combined_from_original_dirty_trainer_patch_sha256"] = hashlib.sha256(full.encode()).hexdigest()
            outputs[label]["frozen_module_sha256"] = modules
    (HERE / "diagnostics_manifest.json").write_text(json.dumps({"applied": False, "helper_sha256": hashlib.sha256(helper.encode()).hexdigest(), "outputs": outputs, "training_arithmetic_changes": False, "synchronization": "One skip-counter scalar item per completed epoch; buffer hashes computed once before fitting; CPU RNG fingerprints read before each existing loader iteration"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
