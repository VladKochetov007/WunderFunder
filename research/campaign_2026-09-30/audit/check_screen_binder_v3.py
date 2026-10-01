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

import build_screen_specs_v3 as binder

HERE = Path(__file__).resolve().parent


def main() -> None:
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    limits = {"rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS)), "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": os.environ.get("OPENBLAS_NUM_THREADS")}
    assert limits["rlimit_as"] == [2147483648, 2147483648] and len(limits["cpu_affinity"]) == 1 and limits["threads"] == "1"
    rejects = []

    def reject(name: str, action: object) -> None:
        try:
            action()
        except (ValueError, KeyError, FileNotFoundError):
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
        prior = {"stages_by_job": {"a": ["a"], "b": ["b"]}, "stage_expected_args": {name: {"epochs": 1, "valid_repeat": 2, "batch": 2, "chunk": 2000, "ema_teacher": 1, "epoch_samples": 0, "class_balance": False} for name in ["a", "b"]}}
        counts = {"optimizer_attempts": 20, "amp_skipped_updates": 0, "optimizer_completed_updates": 20, "ema_updates": 20, "completed_batches": 2}
        group_hash = lambda groups: hashlib.sha256(b"".join(int(g).to_bytes(8, "little", signed=True) for g in groups)).hexdigest()
        provenance = {"fit_row_groups": {"train": [7, 8, 10], "valid": [9]},
                      "fit_membership_identity": {"train": {"count": 3, "ordered_int64_le_sha256": group_hash([7, 8, 10])}, "valid": {"count": 1, "ordered_int64_le_sha256": group_hash([9])}, "dataset_length": 5, "valid_repeat": 2},
                      "normalization_identity": {"mean": "exact"},
                      "update_diagnostics": [{"epoch_index": 0, "epoch": counts, "cumulative": counts, "sample_order_identity": {"rng": "exact", "drop_last": True, "batch_size": 2, "sampler_class": "RandomSampler", "sampler_replacement": False, "sampler_num_samples": 5}, "normalization_identity": {"mean": "exact"}}],
                      "cumulative_updates": counts, "update_counter_implementation": {"source": "fixed"}, "run_identity_source_sha256": "fixed"}
        lineage = {}
        for name in ["a", "b"]:
            path = root / f"{name}.json"
            path.write_text(json.dumps(provenance))
            lineage[name] = {"stages": {name: {"checkpoint": {"sha256": name}, "provenance": identity(path)}}}
        assert binder.check_parity(prior, lineage)["status"] == "pass"
        order = provenance["update_diagnostics"][0]["sample_order_identity"]
        assert binder.loader_counts(order, prior["stage_expected_args"]["a"], 5) == (2, 20)
        assert binder.loader_counts({**order, "drop_last": False}, prior["stage_expected_args"]["a"], 5) == (3, 30)
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
        changed("drop_last_false_disagrees_with_frozen_source", lambda d: d["update_diagnostics"][0]["sample_order_identity"].update(drop_last=False))
        changed("drop_last_nonboolean", lambda d: d["update_diagnostics"][0]["sample_order_identity"].update(drop_last=1))
        changed("wrong_batch_size", lambda d: d["update_diagnostics"][0]["sample_order_identity"].update(batch_size=3))
        changed("wrong_sampler_membership_count", lambda d: d["update_diagnostics"][0]["sample_order_identity"].update(sampler_num_samples=4))
        changed("wrong_sampler_class", lambda d: d["update_diagnostics"][0]["sample_order_identity"].update(sampler_class="SequentialSampler"))
        changed("wrong_sampler_replacement", lambda d: d["update_diagnostics"][0]["sample_order_identity"].update(sampler_replacement=True))
        def ceil_counts(d: dict) -> None:
            counts = {"optimizer_attempts": 30, "amp_skipped_updates": 0, "optimizer_completed_updates": 30, "ema_updates": 30, "completed_batches": 3}
            d["update_diagnostics"][0]["epoch"] = counts
            d["update_diagnostics"][0]["cumulative"] = counts
            d["cumulative_updates"] = counts
        changed("ceil_count_with_true_drop_last", ceil_counts)
        # Separate synthetic interrupted prefix and exactly prescribed suffix.
        logical = "c20260930_ft_spread_proxy_f0"
        original_pipeline = {"arm": "proxy_known", "log_directory": "logs/proxy_original", "stages": [
            {"stage": stage, "argv": ["python", "scripts/train_rnn.py" if stage != "prediction" else "scripts/predict_run.py", stage]}
            for stage in ["continuation", "FT_head", "FT_full", "prediction"]]}
        original_recipe_path = root / "original_recipe.json"
        original_recipe_path.write_text(json.dumps({"source_sha256": sources, "arms": {"proxy_known": original_pipeline}, "cap_gpu_seconds_per_arm": 720}))
        closure_path = root / "fragment_closure.json"
        closure_path.write_text(json.dumps({"source_sha256": sources}))
        original_design_path = root / "original_design.json"
        original_wrapper = {"design": {"folds": [{"roles": {"matched_control": [{"run": logical, "weight": .5}, {"run": "seed4", "weight": .5}]}}]}, "builder_source_sha256": "unchanged"}
        original_design_path.write_text(json.dumps(original_wrapper))
        origlogs = root / original_pipeline["log_directory"]
        origlogs.mkdir(parents=True)
        prefix_path = origlogs / "00_continuation.json"
        prefix_path.write_text(json.dumps({"stage": "continuation", "command": original_pipeline["stages"][0]["argv"], "exit_code": 0, "duration_seconds": 199}))
        (origlogs / "00_continuation.log").write_text("synthetic prefix")
        (origlogs / "recipe_identity.json").write_text(json.dumps({"sha256": identity(original_recipe_path)["sha256"], "arm": "proxy_known"}))
        jobs = binder.CAMPAIGN / "jobs"
        jobs.mkdir()
        original_job_path = jobs / f"{logical}.json"
        original_command = ["python", "original_pipeline"]
        original_job_path.write_text(json.dumps({"kind": "gpu", "status": "failed", "exit_code": 1, "command": original_command, "elapsed_seconds": 200, "source_sha256": sources, "timeout_seconds": 720}))
        checkpoint = root / "artifacts/runs/c20260930_spread_proxy_f0/model_ep0.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint")
        completed_provenance = checkpoint.parent / "provenance.json"
        completed_provenance.write_text("{}")
        archive = root / "stop_archive.txt"
        archive.write_text("automatic queued expiration")
        resume_path = root / "resume_recipe.json"
        resumed_job = logical + "_resume_v1"
        resume_recipe = {"source_sha256": sources, "arms": {"proxy_resume": {"log_directory": "logs/resumed", "stages": original_pipeline["stages"][1:]}},
                         "resume": {"original_recipe": identity(original_recipe_path), "original_job": identity(original_job_path), "completed_stage": identity(prefix_path), "initial_checkpoint": identity(checkpoint)}}
        commitment_path = root / "resume_commitment.json"
        def recommit(recipe: dict) -> dict:
            resume_path.write_text(json.dumps(recipe))
            command = ["python", "scripts/campaign_screen_pipeline.py", "--recipe", str(resume_path.relative_to(root)), "--recipe-sha256", identity(resume_path)["sha256"], "--arm", "proxy_resume"]
            commitment = {"status": "frozen-before-missing-stages", "numeric_or_candidate_quality_seen": False, "new_job_name": resumed_job, "original_design": identity(original_design_path), "resume_recipe": identity(resume_path), "command": command, "timeout_seconds": 467, "stop_archive": identity(archive)}
            commitment_path.write_text(json.dumps(commitment))
            return commitment
        commitment = recommit(resume_recipe)
        amendment = {"resume_commitment": identity(commitment_path), "original_preregistration": identity(original_design_path), "completed_checkpoint": identity(checkpoint), "completed_provenance": identity(completed_provenance)}
        fragment_prior = {"job_commands": {logical: original_command}, "pipeline_recipe": identity(original_recipe_path), "pipeline_arms": {logical: original_pipeline}, "job_source_closure": identity(closure_path)}
        derived = binder.operational_design(original_wrapper, commitment)
        assert derived["design"]["folds"][0]["roles"]["matched_control"][0]["job_name"] == resumed_job
        stripped = copy.deepcopy(derived)
        del stripped["design"]["folds"][0]["roles"]["matched_control"][0]["job_name"]
        assert stripped == original_wrapper
        fragments = binder.fragment_plan(fragment_prior, logical, amendment)
        assert len(fragments) == 2 and len(fragments[0]["pipeline"]["stages"]) == 1 and len(fragments[1]["pipeline"]["stages"]) == 3
        assert binder.bind_fragment(fragments[0], sources)["record"]["status"] == "failed"
        reject("missing_resume_job_cannot_bind", lambda: binder.bind_fragment(fragments[1], sources))
        resumedlogs = root / fragments[1]["pipeline"]["log_directory"]
        resumedlogs.mkdir(parents=True)
        (resumedlogs / "recipe_identity.json").write_text(json.dumps({"sha256": identity(resume_path)["sha256"], "arm": "proxy_resume"}))
        for index, stage in enumerate(fragments[1]["pipeline"]["stages"]):
            (resumedlogs / f"{index:02d}_{stage['stage']}.json").write_text(json.dumps({"stage": stage["stage"], "command": stage["argv"], "exit_code": 0, "duration_seconds": 20}))
            (resumedlogs / f"{index:02d}_{stage['stage']}.log").write_text("synthetic suffix")
        resume_job_path = jobs / f"{resumed_job}.json"
        resume_job_path.write_text(json.dumps({"kind": "gpu", "status": "finished", "exit_code": 0, "command": commitment["command"], "elapsed_seconds": 100, "source_sha256": sources, "timeout_seconds": 467}))
        assert binder.bind_fragment(fragments[1], sources)["record"]["status"] == "finished"
        assert sum(binder.bind_fragment(f, sources)["record"]["elapsed_seconds"] for f in fragments) == 300
        assert binder.fragment_elapsed([{"record": {"elapsed_seconds": 200}}, {"record": {"elapsed_seconds": 100}}], 720) == 300
        reject("aggregate_allowance_exceeded", lambda: binder.fragment_elapsed([{"record": {"elapsed_seconds": 253}}, {"record": {"elapsed_seconds": 468}}], 720))
        reject("quality_seen_resume_refused", lambda: binder.operational_design(original_wrapper, {**commitment, "numeric_or_candidate_quality_seen": True}))
        changed_recipe = copy.deepcopy(resume_recipe)
        changed_recipe["arms"]["proxy_resume"]["stages"] = original_pipeline["stages"]
        recommit(changed_recipe)
        amendment["resume_commitment"] = identity(commitment_path)
        reject("resume_repeats_continuation", lambda: binder.fragment_plan(fragment_prior, logical, amendment))
        changed_recipe = copy.deepcopy(resume_recipe)
        changed_recipe["source_sha256"] = {**sources, "scripts/train_rnn.py": "changed"}
        recommit(changed_recipe)
        amendment["resume_commitment"] = identity(commitment_path)
        reject("resume_changes_source", lambda: binder.fragment_plan(fragment_prior, logical, amendment))
        recommit(resume_recipe)
        amendment["resume_commitment"] = identity(commitment_path)
        forbidden = origlogs / "01_FT_head.log"
        forbidden.write_text("unexpected old stage")
        reject("old_prefix_executed_extra_stage", lambda: binder.bind_fragment(fragments[0], sources))
        forbidden.unlink()
        resumed_record = json.loads(resume_job_path.read_text())
        resume_job_path.write_text(json.dumps({**resumed_record, "status": "failed", "exit_code": 1}))
        reject("failed_resume_not_successful", lambda: binder.bind_fragment(fragments[1], sources))
        resume_job_path.write_text(json.dumps({**resumed_record, "timeout_seconds": 720}))
        reject("resume_allowance_changed", lambda: binder.bind_fragment(fragments[1], sources))
    result = {"passed": True, "rejections": rejects, "known_answer_two_arm_freeze": True,
              "known_answer_epoch_parity": True, "floor_5div2_true_20attempts": True, "ceil_5div2_false_30attempts": True,
              "no_real_target_prediction_or_fit_artifact_read": True,
              "fragment_prefix_suffix_known_answer": True, "operational_projection_exact": True,
              "binder_sha256": hashlib.sha256((HERE / "build_screen_specs_v3.py").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "resource_fallback": limits, "hard_cgroup_2GiB_claim": False,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    with (HERE / "screen_binder_checks_v3b.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
