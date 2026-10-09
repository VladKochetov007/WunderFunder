"""Drive rolling Kaggle 2xT4 sampled-recipe cohorts, one provider step per invocation.

Run from a systemd timer.  Rounds live under ``rounds/<tag>/`` and advance
through phases: sealed -> dataset_created -> ready -> pushed -> collected ->
banked.  When no round is active the driver samples the next round from the
recipe grid (excluding recipes already run), seals it with a root admission
bound to this standing user authorization, and starts again.  It never
creates a second dataset or kernel for a round, checks the live weekly GPU
quota before each push, and stops admitting after two consecutive failed
rounds or when ``STOP`` exists.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import seal_sampled_cohort as seal  # noqa: E402

KAGGLE = str(ROOT / ".venv/bin/kaggle")
ROUNDS = HERE / "rounds"
STATE = ROUNDS / "STATE.json"
TEMPLATE_SPEC = HERE.parent / "experiment_sampler_r3/cohort_spec_r1.json"
BANK = ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank/BANK.jsonl"
ARMS_PER_ROUND = 12
MIN_GPU_SECONDS_LEFT = 2400
AUTHORIZATION = (
    "User 2026-10-09: keep running random combination experiments on Kaggle GPUs and local compute, "
    "save predictions/weights and continue automatically; root claude-code rolling driver"
)

LOSSES = [("wmse", None), ("huber", 1.0), ("huber", 2.0), ("huber", 4.0)]
ACTIVATIONS = ["gelu", "fixed_mix"]
SCHEDULES = (
    [("onecycle", {"warmup_frac": 0.05, "peak_lr": 0.001, "final_div_factor": 100.0})]
    + [
        ("wsd", {"warmup_frac": 0.05, "decay_share": d, "final_lr_fraction": f})
        for d in (0.05, 0.1, 0.2, 0.3, 0.4)
        for f in (0.01, 0.1)
    ]
    + [("sgdr", {"warmup_frac": 0.05, "restarts": r, "final_lr_fraction": 0.01}) for r in (1, 2, 3)]
)


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(canonical(value))
    tmp.replace(path)


def log(event: str, **fields) -> None:
    ROUNDS.mkdir(parents=True, exist_ok=True)
    with (ROUNDS / "driver_log.jsonl").open("a") as handle:
        handle.write(json.dumps({"utc": now(), "event": event, **fields}, sort_keys=True) + "\n")


def run(*args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout, check=False)


def recipe_key(loss, delta, activation, scheduler, parameters) -> str:
    parts = [loss if delta is None else f"huber{delta:g}", activation, scheduler]
    if scheduler == "wsd":
        parts.append(f"d{parameters['decay_share']:g}f{parameters['final_lr_fraction']:g}")
    elif scheduler == "sgdr":
        parts.append(f"r{parameters['restarts']}")
    return "_".join(parts).replace(".", "p")


GRID = {
    recipe_key(loss, delta, act, sched, params): (loss, delta, act, sched, params)
    for (loss, delta), act, (sched, params) in itertools.product(LOSSES, ACTIVATIONS, SCHEDULES)
}
ALREADY_RUN = {
    "wmse_gelu_onecycle", "huber1_gelu_onecycle", "huber2_gelu_onecycle", "huber4_gelu_onecycle",
    "wmse_fixed_mix_onecycle", "huber1_fixed_mix_onecycle", "wmse_gelu_wsd_d0p2f0p01", "wmse_gelu_sgdr_r1",
}


def completed_keys(fold: int = 0) -> set[str]:
    keys = set(ALREADY_RUN) if fold == 0 else set()
    for spec_path in ROUNDS.glob("r*/cohort_spec.json"):
        spec = read(spec_path)
        if int(spec.get("fold", 0)) == fold:
            keys.update(arm["recipe_key"] for arm in spec["arms"])
    return keys


ALIASES = {
    "s_adamw_wmse": "wmse_gelu_onecycle", "s_adamw_huber1": "huber1_gelu_onecycle",
    "s_adamw_huber2": "huber2_gelu_onecycle", "s_adamw_huber4": "huber4_gelu_onecycle",
    "s_fixmix_wmse": "wmse_fixed_mix_onecycle", "s_fixmix_huber1": "huber1_fixed_mix_onecycle",
    "s_wsd_wmse": "wmse_gelu_wsd_d0p2f0p01", "s_sgdr_wmse": "wmse_gelu_sgdr_r1",
}


def fold0_ranking() -> list[str]:
    """Recipe keys ranked by their latest fold-0 small-screen score (best first)."""
    grade = ROOT / "research/campaign_2026-10-06/bandit_continuation/prediction_bank/LATEST_GRADE_FOLD0.json"
    if not grade.exists():
        return []
    singles = read(grade)["singles"]
    keys = {}
    for name in singles:
        key = ALIASES.get(name, name[2:] if name.startswith("s_") else None)
        if key in GRID and (key not in keys or singles[name]["wp"] > singles[keys[key]]["wp"]):
            keys[key] = name
    return sorted(keys, key=lambda key: -singles[keys[key]]["wp"])


def next_tag() -> str:
    numbers = [int(p.name[1:]) for p in ROUNDS.glob("r*") if p.is_dir() and p.name[1:].isdigit()]
    return f"r{max(numbers, default=3) + 1}"


FOLD_PINS = HERE / "fold_pins"


def available_folds() -> list[int]:
    """Fold 0 always; other folds once their collected cache pins exist."""
    return [0] + sorted(int(p.stem.split("_")[1]) for p in FOLD_PINS.glob("fold_*.json"))


def build_spec(tag: str, keys: list[str], fold: int = 0) -> dict:
    seal.configure_round(tag)
    spec = read(TEMPLATE_SPEC)
    if fold:
        record = read(FOLD_PINS / f"fold_{fold}.json")
        pins = record["pins"]
        spec["fold"], spec["fold_pins"] = fold, pins
        spec["parsed_cache"].update(
            kernel=pins["SOURCE_KERNEL"], cache_manifest_sha256=pins["EXPECTED_CACHE_MANIFEST_SHA256"],
            cache_key=pins["EXPECTED_CACHE_KEY"], validation_sha256=pins["EXPECTED_CPU_VALIDATION_SHA256"],
            source_stage_receipt_sha256=pins["EXPECTED_SOURCE_STAGE_RECEIPT_SHA256"],
            membership_sha256=pins["EXPECTED_MEMBERSHIP_SHA256"], split_sha256=pins["EXPECTED_SPLIT_SHA256"],
            registry_sha256=pins["EXPECTED_REGISTRY_SHA256"], fit_train_count=record["fit_train_count"],
            evaluation_count=record["evaluation_count"],
        )
        spec["common_recipe"]["fold"] = fold
    spec["kernel_id"] = seal.KERNEL_ID
    spec["runtime_source"]["runner_overlay_dataset"] = seal.OVERLAY_DATASET
    spec["runtime_source"]["runner_overlay_title"] = seal.OVERLAY_TITLE
    spec.pop("deferred_candidate_ids", None)
    arms = []
    for key in keys:
        loss, delta, act, sched, params = GRID[key]
        recipe = {"loss": loss, "huber_delta": delta, "activation": act, "scheduler": sched,
                  "scheduler_parameters": params}
        arms.append({
            "queue_id": f"SAMPLED_{tag.upper()}_{key.upper()}",
            "candidate_id": f"{tag}_{key}",
            "candidate_recipe_sha256": hashlib.sha256(canonical(recipe)).hexdigest(),
            "recipe_key": key,
            "run_name": f"sampled_{tag}_{key}",
            "parent_candidates": ["D01_ADAMW_WMSE"],
            "axis": "rolling random combination of loss x activation x schedule",
            **recipe,
        })
    spec["arms"] = arms
    spec["execution"]["waves"] = [[arms[i]["candidate_id"], arms[i + 1]["candidate_id"]]
                                  for i in range(0, len(arms), 2)]
    return spec


def sample_round(tag: str, fold: int = 0) -> list[str]:
    done = completed_keys(fold)
    if fold == 0:
        remaining = sorted(set(GRID) - done)
        rng = random.Random(f"wnn-rolling-{tag}")
        rng.shuffle(remaining)
    else:
        # Cross-fold confirmation: matched controls first, then the best fold-0 recipes.
        priority = ["wmse_gelu_onecycle", "wmse_gelu_wsd_d0p2f0p01", *fold0_ranking()]
        remaining = [key for key in dict.fromkeys(priority) if key not in done]
    picked = remaining[:ARMS_PER_ROUND]
    return picked[: len(picked) // 2 * 2]


def choose_fold(tag: str) -> int:
    folds = available_folds()
    return folds[int(tag[1:]) % len(folds)]


def gpu_seconds_left() -> float:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()
        )
    return (quota.gpu_quota.total_time_allowed - quota.gpu_quota.time_used - quota.gpu_quota.time_reserved).total_seconds()


def start_round() -> dict | None:
    tag = next_tag()
    fold = choose_fold(tag)
    keys = sample_round(tag, fold)
    if len(keys) < 2 and fold != 0:
        fold, keys = 0, sample_round(tag, 0)
    if len(keys) < 2:
        log("grid_exhausted")
        return None
    directory = ROUNDS / tag
    spec = build_spec(tag, keys, fold)
    seal.configure_fold(spec, ROOT)
    write(directory / "cohort_spec.json", spec)
    spec_path = directory / "cohort_spec.json"
    worker_sha = seal.sha_file(HERE / "sampled_recipe_worker.py")
    # Seal once unadmitted to learn the overlay manifest hash, then seal the admitted packet.
    probe = seal.seal_candidate(directory / "probe_unadmitted", spec_path)
    admission = seal._admission_payload(
        seal.sha_file(spec_path), probe["runner_overlay_manifest_sha256"], worker_sha,
        seal.SOURCE_CODE_MANIFEST_SHA256, push_allowed=True,
    )
    admission |= {"authority": AUTHORIZATION, "created_utc": now(), "gpu_count": 2,
                  "gpu_model": "NvidiaTeslaT4", "provider_seconds": 3600,
                  "maximum_output_bytes": 12_000_000_000}
    write(directory / "root_admission.json", admission)
    receipt = seal.seal_candidate(directory / "sealed", spec_path, directory / "root_admission.json")
    shutil.rmtree(directory / "probe_unadmitted")
    state = {"tag": tag, "fold": fold, "phase": "sealed", "kernel": seal.KERNEL_ID, "dataset": seal.OVERLAY_DATASET,
             "arms": keys, "seal_receipt": receipt, "created_utc": now()}
    write(directory / "round_state.json", state)
    log("sealed", tag=tag, fold=fold, arms=keys)
    return state


def stage_and_create(state: dict) -> None:
    directory = ROUNDS / state["tag"]
    source = directory / "sealed/runner_overlay_asset"
    stage = Path(f"/tmp/wsr_{state['tag']}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    for path in source.iterdir():
        shutil.copyfile(path, stage / path.name)
    for path in stage.glob("*.wnn"):
        if seal.sha_file(path) != path.stem:
            raise RuntimeError(f"staged payload hash differs: {path.name}")
    listing = run("datasets", "list", "--mine", "-s", state["dataset"].split("/")[1])
    if state["dataset"] in listing.stdout:
        log("dataset_already_exists", tag=state["tag"])
    else:
        result = run("datasets", "create", "-p", str(stage), timeout=900)
        log("dataset_create", tag=state["tag"], rc=result.returncode, tail=(result.stdout + result.stderr)[-400:])
        if result.returncode != 0:
            state["create_failures"] = state.get("create_failures", 0) + 1
            write(directory / "round_state.json", state)
            return
    state["phase"] = "dataset_created"
    write(directory / "round_state.json", state)


def check_ready(state: dict) -> None:
    result = run("datasets", "status", state["dataset"])
    if "ready" in result.stdout.lower():
        state["phase"] = "ready"
        write(ROUNDS / state["tag"] / "round_state.json", state)
        log("dataset_ready", tag=state["tag"])


def push(state: dict) -> None:
    left = gpu_seconds_left()
    if left < MIN_GPU_SECONDS_LEFT:
        log("quota_hold", tag=state["tag"], gpu_seconds_left=left)
        return
    kernel_dir = ROUNDS / state["tag"] / "sealed/kernel"
    result = run("kernels", "push", "-p", str(kernel_dir), "-t", "3600", "--accelerator", "NvidiaTeslaT4", timeout=600)
    text = (result.stdout + result.stderr)[-600:]
    log("push", tag=state["tag"], rc=result.returncode, tail=text, gpu_seconds_left=left)
    state["push"] = {"utc": now(), "rc": result.returncode, "tail": text, "gpu_seconds_left": left}
    if result.returncode == 0 and "successfully pushed" in text:
        state["phase"] = "pushed"
    else:
        state["push_failures"] = state.get("push_failures", 0) + 1
    write(ROUNDS / state["tag"] / "round_state.json", state)


def collect(state: dict) -> None:
    dest = ROUNDS / state["tag"] / "terminal_collect"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/kaggle_collect_kernel_output.py"), "--kernel", state["kernel"],
         "--dest", str(dest), "--kaggle", KAGGLE],
        timeout=3300, check=False,
    )
    marker = dest / "COLLECTED.json"
    if marker.exists():
        state["phase"] = "collected"
        state["terminal"] = read(marker)
        write(ROUNDS / state["tag"] / "round_state.json", state)
        log("collected", tag=state["tag"], terminal=state["terminal"]["terminal_status"][-40:])


def bank(state: dict) -> None:
    arms_root = ROUNDS / state["tag"] / "terminal_collect/outputs/out/arms"
    spec = read(ROUNDS / state["tag"] / "cohort_spec.json")
    existing = {json.loads(line)["name"] for line in BANK.read_text().splitlines() if line.strip()}
    added = []
    with BANK.open("a") as handle:
        for arm in spec["arms"]:
            prediction = arms_root / arm["queue_id"] / "predictions_valid.npy"
            fold = int(spec.get("fold", 0))
            name = f"s_{arm['recipe_key']}" if fold == 0 else f"s_f{fold}_{arm['recipe_key']}"
            if not prediction.is_file() or name in existing:
                continue
            tags = {"scale": "small_screen_350seq_220upd", "arch": "gru256x2", "seed": 20261009, "optim": "adamw",
                    "loss": arm["recipe_key"].split("_")[0], "act": arm["activation"], "sched": arm["scheduler"],
                    "sched_params": arm["scheduler_parameters"], "round": state["tag"], "prec": "fp32",
                    "recipe_key": arm["recipe_key"]}
            handle.write(json.dumps({"name": name, "fold": fold, "prediction": str(prediction.relative_to(ROOT)),
                                     "tags": tags, "source": "kaggle_t4"}) + "\n")
            added.append(name)
    complete = "COMPLETE" in state["terminal"]["terminal_status"] and added
    state["phase"] = "banked"
    state["banked"] = added
    state["outcome"] = "success" if complete else "failure"
    write(ROUNDS / state["tag"] / "round_state.json", state)
    driver = read(STATE) if STATE.exists() else {}
    record_outcome(driver, state, bool(complete))
    driver["active"] = None
    write(STATE, driver)
    log("banked", tag=state["tag"], added=added, outcome=state["outcome"])


def record_outcome(driver: dict, state: dict, success: bool) -> None:
    """A failed fold-k round disables only that fold; fold-0 failures count toward the stop."""
    fold = int(state.get("fold", 0))
    if success:
        driver["consecutive_failures"] = 0
    elif fold:
        pins = FOLD_PINS / f"fold_{fold}.json"
        if pins.exists():
            pins.rename(FOLD_PINS / f"fold_{fold}.disabled_{state['tag']}")
        log("fold_disabled", fold=fold, tag=state["tag"])
    else:
        driver["consecutive_failures"] = driver.get("consecutive_failures", 0) + 1


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    ROUNDS.mkdir(parents=True, exist_ok=True)
    driver = read(STATE) if STATE.exists() else {"active": None, "consecutive_failures": 0}
    active = driver.get("active")
    if active is None:
        if (HERE / "STOP").exists() or driver.get("consecutive_failures", 0) >= 2:
            return 0
        state = start_round()
        if state is None:
            return 0
        driver["active"] = state["tag"]
        write(STATE, driver)
        return 0
    state = read(ROUNDS / active / "round_state.json")
    if state.get("create_failures", 0) >= 3 or state.get("push_failures", 0) >= 3:
        log("round_abandoned", tag=active)
        state["phase"], state["outcome"] = "abandoned", "failure"
        write(ROUNDS / active / "round_state.json", state)
        record_outcome(driver, state, False)
        driver["active"] = None
        write(STATE, driver)
        return 0
    step = {"sealed": stage_and_create, "dataset_created": check_ready, "ready": push,
            "pushed": collect, "collected": bank}.get(state["phase"])
    if step is not None:
        step(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
