"""Build commands from source/metadata only. Never imports Torch or starts a job."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / "research/iterations/2026-10-01-overnight/training"
TRAINER = OLD / "train_aux.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser_from_source():
    """Interpret literal argparse declarations, not any trainer statements."""
    parser = argparse.ArgumentParser(add_help=False)
    for node in ast.walk(ast.parse(TRAINER.read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        args = [ast.literal_eval(x) for x in node.args]
        kw = {}
        for item in node.keywords:
            if item.arg == "type":
                assert isinstance(item.value, ast.Name)
                kw[item.arg] = {"int": int, "float": float, "str": str}[item.value.id]
            else:
                kw[item.arg] = ast.literal_eval(item.value)
        parser.add_argument(*args, **kw)
    return parser


def resolve(parser, argv):
    values = vars(parser.parse_args(argv))
    if values["fast"]:
        for key in ("sync_free", "fused_adam", "pin_memory", "prop_cache"):
            values[key] = True
    return values


def main():
    assert sha(TRAINER) == "f13e61b4d43f011c132f6b7f88df3f62775d0a43189776437c42cc484ac0f39f"
    parser = parser_from_source()
    defaults = resolve(parser, ["--name", "unused"])
    inventory = json.loads((OLD / "checkpoint-inventory.json").read_text())["files"]
    head_plan = json.loads((OLD / "plan.json").read_text())
    jobs = []
    for fold in (0, 1):
        parent_name = f"cont_sh256_s3_f{fold}"
        head_name = f"cont_sh256_s3h2_f{fold}"
        original = f"/home/grinchenkoeg/WunderFunder/artifacts/runs/d256_f{fold}/model_ep3.pt"
        old = inventory[original]
        assert set(old["args"]) <= set(defaults)
        desired = defaults | old["args"] | {"name": parent_name, "layers": 1}
        argv = []
        for key, value in desired.items():
            if key != "name" and value == defaults[key]:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                assert value
                argv.append(flag)
            else:
                argv.extend([flag, str(value)])
        assert resolve(parser, argv) == desired
        inherited_diff = {k: [v, desired[k]] for k, v in old["args"].items() if v != desired[k]}
        assert set(inherited_diff) == {"name", "layers"}
        jobs.append({"name": parent_name, "stage": "parent", "fold": fold,
                     "argv": [".venv/bin/python", str(TRAINER.relative_to(ROOT)), *argv],
                     "resolved_args": desired, "metadata_reference": original,
                     "historical_checkpoint_sha256": old["sha256"],
                     "inherited_argument_changes": inherited_diff,
                     "new_parser_defaults": {k: v for k, v in desired.items() if k not in old["args"]},
                     "endpoint": f"artifacts/runs/{parent_name}/model_ep3.pt",
                     "dependency": "required external inventory inputs below", "wall_limit_seconds": 3600})
        head = next(x for x in head_plan["jobs"] if x["name"] == f"on_ctld256_f{fold}")
        head_argv = list(head["train_argv"])
        head_argv[head_argv.index("--name") + 1] = head_name
        head_argv[head_argv.index("--init") + 1] = parent_name + "@ep3"
        head_argv.extend(["--layers", "1"])
        head_args = resolve(parser, head_argv[2:])
        old_head = resolve(parser, head["train_argv"][2:])
        diff = {k: [v, head_args[k]] for k, v in old_head.items() if v != head_args[k]}
        assert set(diff) == {"name", "init", "layers"}
        jobs.append({"name": head_name, "stage": "H2", "fold": fold, "argv": head_argv,
                     "resolved_args": head_args, "reference_job": head["name"],
                     "inherited_argument_changes": diff,
                     "endpoint": f"artifacts/runs/{head_name}/model_ep1.pt",
                     "dependency": f"{parent_name}: verified fixed epoch3", "wall_limit_seconds": 1200,
                     "OOF_argv": [".venv/bin/python", str((HERE / "predict_fixed.py").relative_to(ROOT)),
                                  head_name, "--fold", str(fold), "--epoch", "1", "--batch", "16",
                                  "--output-dir", "research/iterations/2026-10-01-continuation/cache"],
                     "OOF_wall_limit_seconds": 1200})
    source_paths = [TRAINER, OLD / "checkpoint-inventory.json", OLD / "plan.json",
                    HERE / "predict_fixed.py", Path(__file__), ROOT / "scripts/train_rnn.py",
                    *[ROOT / "src/wnn" / (x + ".py") for x in
                      ("nn", "resources", "data", "features", "splits", "outcomes", "propensity")]]
    result = {
        "status": "SOURCE_PREPARATION_ONLY_NO_ALLOCATION", "execution_authorized": False,
        "working_directory": "<verified creative checkout root>",
        "trainer_ROOT_rule": "frozen train_aux.py parents[4]; direct invocation in its current location resolves creative root",
        "source_sha256": {str(x.relative_to(ROOT)): sha(x) for x in source_paths},
        "predictor_derivation": {"original": str((OLD / "predict_fixed.py").relative_to(ROOT)),
                                 "original_sha256": sha(OLD / "predict_fixed.py"),
                                 "change": "Only reject existing .npy.partial and .inputs.json as well as .npy; unchanged relative ROOT/HERE arithmetic scopes writes to continuation."},
        "budget": {"fits": 4, "H2_OOF_jobs": 2, "parent_OOF_jobs": 0, "WP_queries_authorized": 0,
                   "maximum_concurrent_GPU_jobs": 1, "retries": 0, "automatic_followups": False},
        "execution_order": "single explicit GPU: fold0 parent,H2,OOF; then fold1 parent,H2,OOF",
        "proposed_resource_limits": {"GPU_process_NVML_MiB_max": 5632, "host_scope_MemoryMax": "20G",
                                     "MemorySwapMax": 0, "CPUQuota_per_lane": "600%",
                                     "whole_stage_seconds_max": 6000,
                                     "GPU_and_CPU_affinity": "unassigned; root inventory/allocation required",
                                     "absolute_deadline_UTC": None,
                                     "WNN_GPU_FRACTION": "root must set from actual GPU total with allocator ceiling below5632MiB with prospectively declared non-PyTorch headroom; not itself the NVML cap"},
        "environment": {"PYTHONPATH": "src", "WNN_THREADS": "4", "WNN_LOADER_WORKERS": "2",
                        "WNN_LOADER_RAM_GB": "4", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                        "OPENBLAS_NUM_THREADS": "4", "NUMBA_NUM_THREADS": "4", "ARROW_NUM_THREADS": "4",
                        "WNN_GPU_DUTY": "1.0"},
        "required_inventory_inputs": {
            "data": ["datasets/train.parquet", "datasets/valid.parquet", "cache/chain_folds.npz", "cache/train_targets.npz", "cache/valid_targets.npz"],
            "teachers": {str(f): [f"h256_f{f}", f"h256s4_f{f}"] for f in (0, 1)},
            "teacher_cache_contract": "Required, NOT verified: cache/pred_<teacher>_<train|valid>.npy with row order from respective *_targets.npz; prove same-fold causal teacher provenance and finite training-side coverage. No automatic regeneration.",
            "propensity": "For each fold: artifacts/runs/prop_f{fold}/model.pt plus cache/propcache_prop_f{fold}_{train,valid}.npy and .json; exact model hash and row order required.",
            "existing_comparison_H2": [f"artifacts/runs/{x}_f{f}/model_ep1.pt" for f in (0, 1) for x in ("on_ctld256", "on_ctl128s3", "on_ctl128s4", "on_ctlc500")],
            "identity_status": "Logical names inferred from preserved source/metadata only; remote paths, current hashes and resources are unverified."},
        "comparisons": {
            "candidate": "cont_sh256_s3h2 + on_ctl128s3 + on_ctl128s4 + on_ctlc500, each .25",
            "primary": "on_ctld256 + on_ctl128s3 + on_ctl128s4 + on_ctlc500, each .25",
            "secondary": "Same candidate fixed4 versus on_ctl128s3 + on_ctl128s4 + on_ctlc500, each 1/3",
            "endpoint": "model_ep1.pt; folds0/1; no epoch/member/weight selection",
            "gates": "Separate prospective root allocation must bind quality/runtime criteria; no metric command included."},
        "launch_requirements": [
            "Root allocation must bind inventory, source hashes, GPUs/CPU affinity, actual memory-derived allocator fraction, new absolute deadline and limits before any command is used.",
            "All four run directories must be absent, including empty directories/symlinks; all new logs and prediction .npy/.partial/.inputs.json paths absent. Trainer itself only refuses nonempty directories.",
            "Do not use overnight/run_queue.py: expired hard-coded deadline, writes old ledger, forces old prediction stage. Use only separately allocated continuation/student/run_plan.py, sequential on one explicit GPU.",
            "Stop on any fit/OOF failure, timeout or resource violation and retain partial files. No smaller batch, disabled fast path, cache changes, retries or partial checkpoint selection automatically.",
            "Parent saves fixed epoch3 EMA; H2 own-fold epoch3 lineage and frozen input/GRU must be verified before H2; H2 endpoint is epoch1.",
            "No new full211-feature cache or copy/hash of large existing caches during this source preparation.",
            "Same seed does not preserve RNG consumption, normalization sample or shuffled data order when layer count changes. No silent reseed/initialization wrapper."],
        "jobs": jobs}
    (HERE / "commands.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "SOURCE_ONLY_COMMANDS_BUILT", "jobs": len(jobs),
                      "resolved_arguments_each": [len(j["resolved_args"]) for j in jobs],
                      "commands_sha256": sha(HERE / "commands.json")}))


if __name__ == "__main__":
    main()
