"""Synthetic known-answer and provenance rejection checks for screen binding."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import subprocess
import tempfile
from pathlib import Path

import build_screen_specs as binder

HERE = Path(__file__).resolve().parent


def main() -> None:
    limits = dict(line.split("=", 1) for line in subprocess.check_output(
        ["systemctl", "--user", "show", os.environ["WNN_SCOPE_UNIT"] + ".scope",
         "-p", "MemoryMax", "-p", "MemorySwapMax", "-p", "CPUQuotaPerSecUSec", "-p", "Slice"], text=True).splitlines())
    assert limits == {"MemoryMax": "2147483648", "MemorySwapMax": "0", "CPUQuotaPerSecUSec": "1s", "Slice": "wnn-campaign.slice"}
    rejects = []

    def reject(name: str, action: object) -> None:
        try:
            action()
        except (ValueError, KeyError):
            rejects.append(name)
        else:
            raise AssertionError(f"accepted invalid fixture: {name}")

    with tempfile.TemporaryDirectory(prefix="wnn_screen_binder_") as temp:
        root = Path(temp)
        binder.ROOT = root
        binder.CAMPAIGN = root / "campaign"
        binder.CAMPAIGN.mkdir()

        def identity(path: Path) -> dict:
            return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        def artifact(value: dict) -> Path:
            path = root / value["path"]
            if identity(path)["sha256"] != value["sha256"]:
                raise ValueError("fixture commitment changed")
            return path

        binder.identity = identity
        binder.artifact = artifact
        binder.preset = lambda name, family: {"name": name, "checkpoint_epoch": 3, "epochs": 4}
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"sequences": {"valid": [{"row_group": i} for i in range(1873)]}}))
        reference = {"fold": 0, "manifest": identity(manifest), "roles": {"reference": [{"run": "seed3", "weight": .5}, {"run": "seed4", "weight": .5}]}}
        binder.reference_design = lambda: {"folds": [reference]}
        sources = {}
        for path in ["scripts/train_rnn.py", "scripts/predict_run.py", "scripts/campaign_screen_pipeline.py", "src/wnn/nn.py", "src/wnn/features.py", "src/wnn/outcomes.py", "src/wnn/run_identity.py", "src/wnn/structured_state.py"]:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("synthetic source\n")
            sources[path] = identity(target)["sha256"]
        names = binder.FAMILIES["structured"]
        recipe = {"status": "frozen-before-fit", "source_sha256": sources,
                  "stage_expected_args": {name: {"name": name, "checkpoint_epoch": 3, "epochs": 4} for name in names},
                  "arms": {arm: {"log_directory": f"logs/{arm}", "stages": [
                      {"stage": "fit", "argv": [".venv/bin/python", "scripts/train_rnn.py", "--name", name]},
                      {"stage": "export", "argv": [".venv/bin/python", "scripts/predict_run.py", "--name", name]}]}
                           for arm, name in zip(["multi", "single"], names, strict=True)}}
        recipe_path = root / "recipe.json"
        recipe_path.write_text(json.dumps(recipe))
        plan = root / "plan.json"
        plan.write_text("{}")
        frozen = binder.freeze("structured", recipe_path, plan)
        assert len(frozen["job_commands"]) == 2
        for name, command in frozen["job_commands"].items():
            assert command[command.index("--recipe-sha256") + 1] == identity(recipe_path)["sha256"]
            assert command[command.index("--recipe") + 1] == "recipe.json"
            assert frozen["stage_expected_args"][name]["epochs"] == 4
        assert "job_commands" not in recipe
        assert frozen["design"]["derived_predictions"]["candidate_blend"] == {"reference": .8, "candidate": .2}
        command = frozen["job_commands"][names[0]]
        bad = command.copy()
        bad[bad.index("--recipe-sha256") + 1] = "wrong"
        reject("wrong_outer_recipe_hash", lambda: binder.pipeline_arm(bad, recipe_path, identity(recipe_path)["sha256"]))
        reject("duplicated_outer_option", lambda: binder.pipeline_arm(command + ["--arm", "extra"], recipe_path, identity(recipe_path)["sha256"]))
        reject("changed_full_arg", lambda: binder.require_args({"seed": 4}, {"seed": 3}, "synthetic"))
        reservation = root / "logs/multi"
        reservation.mkdir(parents=True)
        reject("reserved_arm_before_freeze", lambda: binder.freeze("structured", recipe_path, plan))
        reservation.rmdir()
        prior = {"stages_by_job": {"a": ["a"], "b": ["b"]}, "stage_expected_args": {name: {"epochs": 1, "valid_repeat": 2, "batch": 2, "chunk": 2000, "ema_teacher": 1} for name in ["a", "b"]}}
        counts = {"optimizer_attempts": 20, "amp_skipped_updates": 0, "optimizer_completed_updates": 20, "ema_updates": 20, "completed_batches": 2}
        group_hash = lambda groups: hashlib.sha256(b"".join(int(g).to_bytes(8, "little", signed=True) for g in groups)).hexdigest()
        provenance = {"fit_row_groups": {"train": [7, 8], "valid": [9]},
                      "fit_membership_identity": {"train": {"count": 2, "ordered_int64_le_sha256": group_hash([7, 8])}, "valid": {"count": 1, "ordered_int64_le_sha256": group_hash([9])}, "dataset_length": 4, "valid_repeat": 2},
                      "normalization_identity": {"mean": "exact"},
                      "update_diagnostics": [{"epoch_index": 0, "epoch": counts, "cumulative": counts, "sample_order_identity": {"rng": "exact"}, "normalization_identity": {"mean": "exact"}}],
                      "cumulative_updates": counts, "update_counter_implementation": {"source": "fixed"}, "run_identity_source_sha256": "fixed"}
        lineage = {}
        for name in ["a", "b"]:
            path = root / f"{name}.json"
            path.write_text(json.dumps(provenance))
            lineage[name] = {"stages": {name: {"checkpoint": {"sha256": name}, "provenance": identity(path)}}}
        assert binder.check_parity(prior, lineage)["status"] == "pass"
        def changed(name: str, mutate: object) -> None:
            variant = copy.deepcopy(provenance)
            mutate(variant)
            path = root / "b.json"
            path.write_text(json.dumps(variant))
            lineage["b"]["stages"]["b"]["provenance"] = identity(path)
            reject(name, lambda: binder.check_parity(prior, lineage))
        changed("missing_epoch", lambda d: d["update_diagnostics"].clear())
        changed("incomplete_batches", lambda d: d["update_diagnostics"][0]["epoch"].update(completed_batches=1))
        changed("changed_sampler_rng", lambda d: d["update_diagnostics"][0]["sample_order_identity"].update(rng="changed"))
        changed("changed_normalization", lambda d: d["normalization_identity"].update(mean="changed"))
        changed("changed_membership_bytes", lambda d: d["fit_membership_identity"]["train"].update(ordered_int64_le_sha256="changed"))
        changed("consistent_unequal_successful_updates", lambda d: d["update_diagnostics"][0]["epoch"].update(amp_skipped_updates=1, optimizer_completed_updates=19))
        changed("missing_ema_call", lambda d: d["update_diagnostics"][0]["epoch"].update(ema_updates=19))
    result = {"passed": True, "rejections": rejects, "known_answer_two_arm_freeze": True,
              "known_answer_epoch_parity": True, "no_real_target_prediction_or_fit_artifact_read": True,
              "binder_sha256": hashlib.sha256((HERE / "build_screen_specs.py").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope_properties": limits, "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with (HERE / "screen_binder_checks_v1.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
