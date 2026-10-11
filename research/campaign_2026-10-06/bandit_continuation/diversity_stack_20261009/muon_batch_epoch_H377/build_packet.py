from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = Path(__file__).resolve().parent
STAGING = CAMPAIGN / "staging_refill_jobs_r1"
SOURCE_MANIFEST = CAMPAIGN / "LR377_SOURCE_MANIFEST_R1.json"
PROJECT_ALIASES = CAMPAIGN / "LR377_PROJECT_ALIASES_R1.json"
STUDENT_ENDPOINT = CAMPAIGN / "LR_FIXED_STUDENT_ENDPOINT_R1.json"
SOURCE_JOB = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1"
    / "refill_jobs_b/bl_muonall_prop0_e4_s3.json"
)
SPLIT_PATH = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold1.json"
INPUT_REGISTRY = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009"
    / "multifold_r1/portable_inputs_fold1.json"
)
BUDGETS_PATH = CAMPAIGN.parent / "ops_continuation_20261011/LR_SCHEDULE_BUDGETS_R1.json"
REMOTE_ROOT = "/root/wnn"
REMOTE_SOURCE_ROOT = "/root/wnn/lr377_source_r1"
REMOTE_SOURCE_MANIFEST = f"{REMOTE_SOURCE_ROOT}/SOURCE_MANIFEST.json"
REMOTE_SPLIT = f"{REMOTE_ROOT}/research/campaign_2026-09-29/artifacts/splits/fold1.json"
REMOTE_REGISTRY = (
    f"{REMOTE_ROOT}/research/campaign_2026-10-06/bandit_continuation"
    "/diversity_stack_20261009/multifold_r1/portable_inputs_fold1.json"
)
SOURCE_MANIFEST_SHA256 = "382beb68d556dd63215a9f7467bda09a23626ec3e1a2195739cffcab6819d9cc"
SOURCE_FILE_COUNT = 58
SOURCE_BYTE_COUNT = 638_829
PROJECT_ALIASES_SHA256 = "93c8f948fc091fbd2565eaa3c4bc3e7db7560631a61855159d7f178461d42042"
STUDENT_ENDPOINT_SHA256 = "ab92b3b417e11fce4aefd9fc99701199f0f293fed096f6a739c0b172fc562a2b"
PROJECT_ALIASES_EXPECTED = {
    "cache": "/root/wnn/cache",
    "datasets": "/root/wnn/datasets",
    "artifacts": "/root/wnn/artifacts",
    "research": "/root/wnn/research",
}
SPLIT_SHA256 = "31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae"
SEQUENCE_LENGTH = 20_000
EVALUATION_SEQUENCE_COUNT = 1_873
FIT_EPOCHS = 3
FIT_CHECKPOINT_EPOCH = 2
UPDATE_ROWS = 1_000
MAX_JOB_SECONDS = 5_400


def sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 fingerprint for exact packet or source bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a small source or manifest file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, object]:
    """Read a JSON object and reject other top-level structures."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def as_string_list(value: object, label: str) -> list[str]:
    """Validate a JSON string array used as a command argv."""
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string array")
    return value


def option_value(argv: list[str], option: str) -> str:
    """Return one unique option value from a flat command argv."""
    matches = [index for index, value in enumerate(argv) if value == option]
    if len(matches) != 1 or matches[0] + 1 >= len(argv):
        raise ValueError(f"expected one value for {option}")
    return argv[matches[0] + 1]


def replace_option(argv: list[str], option: str, value: str) -> list[str]:
    """Copy argv while replacing a single option value."""
    result = list(argv)
    matches = [index for index, item in enumerate(result) if item == option]
    if len(matches) != 1 or matches[0] + 1 >= len(result):
        raise ValueError(f"cannot safely replace {option}")
    result[matches[0] + 1] = value
    return result


def remove_switch(argv: list[str], switch: str) -> list[str]:
    """Copy argv while removing one boolean switch."""
    if argv.count(switch) != 1:
        raise ValueError(f"expected one {switch} switch")
    result = list(argv)
    result.remove(switch)
    return result


def source_environment(argv: list[str]) -> list[str]:
    """Pin trainer imports and data provenance to the frozen source/project roots."""
    return [
        "env",
        f"WNN_PROJECT_ROOT={REMOTE_ROOT}",
        f"PYTHONPATH={REMOTE_SOURCE_ROOT}/src",
        f"WNN_PORTABLE_DATA_MANIFEST={REMOTE_REGISTRY}",
        *argv,
    ]


def timeout_argv(argv: list[str], seconds: int) -> list[str]:
    """Enforce a host-side timeout even though the refill daemon ignores metadata."""
    if seconds < 1 or seconds > MAX_JOB_SECONDS:
        raise ValueError("command timeout must be within the 5,400-second bound")
    return ["timeout", "--signal=TERM", "--kill-after=10s", f"{seconds}s", *argv]


def load_source_recipe() -> tuple[dict[str, object], list[str], list[str]]:
    """Load and verify the completed current-lead recipe used as the parent."""
    job = read_object(SOURCE_JOB)
    steps = job.get("steps")
    if not isinstance(steps, list) or len(steps) != 2:
        raise ValueError("completed seed3 prop0 producer recipe must have fit and export steps")
    fit = steps[0]
    export = steps[1]
    if not isinstance(fit, dict) or not isinstance(export, dict):
        raise ValueError("producer recipe steps must be objects")
    if fit.get("label") != "fit" or export.get("label") != "export":
        raise ValueError("producer recipe step labels differ from the audited fit/export path")
    fit_argv = as_string_list(fit.get("argv"), "fit argv")
    export_argv = as_string_list(export.get("argv"), "export argv")
    expected = {
        "--device": "cuda",
        "--arch": "gru",
        "--cell": "gru",
        "--enc": "linear",
        "--hidden": "256",
        "--layers": "2",
        "--dropout": "0",
        "--seed": "3",
        "--valid-fold": "1",
        "--eval-source": "vfold",
        "--train-source": "train+valid",
        "--train-seqs": "10607",
        "--epochs": "4",
        "--checkpoint-policy": "fixed",
        "--checkpoint-epoch": "3",
        "--batch": "32",
        "--chunk": "1000",
        "--loss": "wmse",
        "--loss-norm": "chunk",
        "--lr": "0.001",
        "--wd": "0.01",
        "--optim": "muon",
        "--muon-scope": "all",
        "--sched": "onecycle",
        "--precision": "fp16",
        "--propensity": "c20260929_prop_f1",
        "--prop-power": "0",
        "--prop-eps": "0.03",
        "--valid-true-mask": "0.03",
        "--aux-struct": "0.3",
        "--aux-next": "0.3",
        "--since-weight": "32",
    }
    for option, value in expected.items():
        if option_value(fit_argv, option) != value:
            raise ValueError(f"completed producer recipe has unexpected {option}")
    for switch in ("--no-eval-during-fit", "--fast", "--save-training-state"):
        if fit_argv.count(switch) != 1:
            raise ValueError(f"completed producer recipe must have one {switch}")
    if "--label-weight-power" in fit_argv or "--class-weight" in fit_argv:
        raise ValueError("producer recipe is not the default label/class weighting treatment")
    if job.get("name") != "vt_f1_muonall_prop0_e4_s3":
        raise ValueError("producer recipe is not the completed seed3 prop0 run")
    if fit_argv[1] != f"{REMOTE_ROOT}/scripts/train_rnn.py":
        raise ValueError("producer fit does not use the live remote trainer path")
    if "scripts/predict_run.py" not in export_argv[3]:
        raise ValueError("producer export does not use predict_run.py")
    fit_argv[1] = f"{REMOTE_SOURCE_ROOT}/scripts/train_rnn.py"
    export_argv[3] = f"{REMOTE_SOURCE_ROOT}/scripts/predict_run.py"
    export_argv[4] = f"{REMOTE_SOURCE_ROOT}/scripts/train_rnn.py"
    return job, source_environment(fit_argv), source_environment(export_argv)


def load_membership() -> dict[str, object]:
    """Capture the exact source-qualified fitting and held-out group identities."""
    split = read_object(SPLIT_PATH)
    counts = split.get("counts")
    fit = split.get("fit")
    evaluation = split.get("evaluation")
    if (
        not isinstance(counts, dict)
        or not isinstance(fit, dict)
        or not isinstance(evaluation, dict)
    ):
        raise ValueError("fold1 split manifest is missing counts or membership")
    train_groups = fit.get("train")
    valid_fit_groups = fit.get("valid")
    eval_valid_groups = evaluation.get("valid")
    if any(
        not isinstance(value, list) for value in (train_groups, valid_fit_groups, eval_valid_groups)
    ):
        raise ValueError("fold1 split membership fields must be lists")
    train_ids = [int(value) for value in train_groups]
    valid_fit_ids = [int(value) for value in valid_fit_groups]
    eval_ids = [int(value) for value in eval_valid_groups]
    if len(train_ids) != int(counts["fit_train"]):
        raise ValueError("fold1 training group count differs from the split summary")
    if len(valid_fit_ids) != int(counts["fit_valid"]):
        raise ValueError("fold1 fit-valid group count differs from the split summary")
    if len(eval_ids) != int(counts["evaluation_valid"]):
        raise ValueError("fold1 evaluation group count differs from the split summary")
    if set(valid_fit_ids) & set(eval_ids):
        raise ValueError("fold1 fitting and evaluation valid groups overlap")
    qualified = [
        {"source": source, "row_group": group}
        for source, groups in (("train.parquet", train_ids), ("valid.parquet", valid_fit_ids))
        for group in groups
    ]
    return {
        "schema": "h377-source-qualified-membership-v1",
        "split_manifest": str(SPLIT_PATH.relative_to(ROOT)),
        "split_manifest_sha256": sha256_file(SPLIT_PATH),
        "fit_train_count": len(train_ids),
        "fit_valid_count": len(valid_fit_ids),
        "fit_total_count": len(qualified),
        "fit_source_qualified_groups": qualified,
        "fit_source_qualified_sha256": sha256_bytes(
            json.dumps(qualified, sort_keys=True, separators=(",", ":")).encode()
        ),
        "evaluation_valid_count": len(eval_ids),
        "evaluation_valid_row_groups": eval_ids,
        "evaluation_valid_sha256": sha256_bytes(
            json.dumps(
                [{"source": "valid.parquet", "row_group": group} for group in eval_ids],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
        "counts": counts,
        "evaluation_exposure": "fold1 has been repeatedly used; this is exposed local evidence",
    }


def load_schedule_budgets() -> tuple[dict[str, object], dict[str, object]]:
    """Return the actual-membership B32 and B64 schedule rows from the audit receipt."""
    budget = read_object(BUDGETS_PATH)
    rows = budget.get("rows")
    if not isinstance(rows, list):
        raise ValueError("schedule budget receipt is missing rows")
    named = {row.get("label"): row for row in rows if isinstance(row, dict)}
    base = named.get("base")
    b64 = named.get("b64_sqrt")
    if not isinstance(base, dict) or not isinstance(b64, dict):
        raise ValueError("schedule budget receipt lacks base or B64 sqrt rows")
    if (
        budget.get("dataset_length") != 11_949
        or base.get("attempts") != 22_380
        or b64.get("attempts") != 11_160
    ):
        raise ValueError("schedule receipt no longer matches the observed drop-last update counts")
    return base, b64


def make_source_preflight(input_registry_sha256: str) -> list[str]:
    """Create a bounded full-source, alias, split and portable-input identity check."""
    source_manifest = read_object(SOURCE_MANIFEST)
    aliases = read_object(PROJECT_ALIASES)
    sources = source_manifest.get("sources")
    if (
        source_manifest.get("schema") != "wnn.frozen-source-snapshot.v1"
        or source_manifest.get("remote_root") != REMOTE_SOURCE_ROOT
        or not isinstance(sources, dict)
        or len(sources) != SOURCE_FILE_COUNT
        or not isinstance(aliases, dict)
        or aliases != PROJECT_ALIASES_EXPECTED
    ):
        raise ValueError("frozen source or project-alias receipt differs from the root snapshot")
    normalized_sources = {str(path): str(digest) for path, digest in sources.items()}
    if any(len(digest) != 64 for digest in normalized_sources.values()):
        raise ValueError("source manifest contains a malformed file digest")
    required = {"scripts/train_rnn.py", "scripts/predict_run.py"}
    if not required.issubset(normalized_sources):
        raise ValueError("source manifest does not bind both trainer and predictor")
    if (
        len(
            [
                path
                for path in normalized_sources
                if path.startswith("src/wnn/") and path.endswith(".py")
            ]
        )
        != SOURCE_FILE_COUNT - 2
    ):
        raise ValueError("source manifest does not cover the complete src/wnn Python tree")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("local source manifest hash differs from root's immutable receipt")
    if sha256_file(PROJECT_ALIASES) != PROJECT_ALIASES_SHA256:
        raise ValueError("local project-alias receipt hash differs from root's receipt")
    code = (
        "import hashlib,json,os,pathlib,sys; "
        f"root=pathlib.Path({json.dumps(REMOTE_ROOT)}); "
        f"snapshot=pathlib.Path({json.dumps(REMOTE_SOURCE_ROOT)}); "
        f"manifest_path=pathlib.Path({json.dumps(REMOTE_SOURCE_MANIFEST)}); "
        f"manifest_sha={json.dumps(SOURCE_MANIFEST_SHA256)}; "
        f"expected_sources={json.dumps(normalized_sources, sort_keys=True)}; "
        f"expected_aliases={json.dumps(PROJECT_ALIASES_EXPECTED, sort_keys=True)}; "
        f"registry_path={json.dumps(REMOTE_REGISTRY)}; "
        f"registry_sha={json.dumps(input_registry_sha256)}; "
        f"split_path={json.dumps(REMOTE_SPLIT)}; "
        f"split_sha={json.dumps(SPLIT_SHA256)}; "
        "sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(); bad=[]; "
        "bad.append('WNN_PROJECT_ROOT') if os.environ.get('WNN_PROJECT_ROOT')!=str(root) else None; "
        f"bad.append('PYTHONPATH') if os.environ.get('PYTHONPATH')!={json.dumps(REMOTE_SOURCE_ROOT + '/src')} else None; "
        "bad.append('WNN_PORTABLE_DATA_MANIFEST') if os.environ.get('WNN_PORTABLE_DATA_MANIFEST')!=registry_path else None; "
        "bad.append('source_manifest') if sha(manifest_path)!=manifest_sha else None; "
        "snapshot_manifest=json.loads(manifest_path.read_text()); "
        "bad.append('manifest_schema') if snapshot_manifest.get('schema')!='wnn.frozen-source-snapshot.v1' or snapshot_manifest.get('remote_root')!=str(snapshot) or snapshot_manifest.get('sources')!=expected_sources else None; "
        "observed={str(p.relative_to(snapshot)) for base in (snapshot/'scripts',snapshot/'src/wnn') for p in base.rglob('*.py')}; "
        "bad.append('source_file_set') if observed!=set(expected_sources) else None; "
        "[(bad.append('source:'+rel) if not (snapshot/rel).is_file() or sha(snapshot/rel)!=digest else None) for rel,digest in expected_sources.items()]; "
        "[(bad.append('alias:'+name) if not (snapshot/name).is_symlink() or str((snapshot/name).resolve())!=target else None) for name,target in expected_aliases.items()]; "
        "bad.append('split_manifest') if sha(split_path)!=split_sha else None; "
        "bad.append('portable_registry') if sha(registry_path)!=registry_sha else None; "
        "registry=json.loads(pathlib.Path(registry_path).read_text()); "
        "sys.exit('source/input preflight mismatch: '+','.join(bad)) if bad else print('H377_SOURCE_INPUT_PREFLIGHT_OK',json.dumps({'source_files':len(expected_sources),'manifest_sha256':manifest_sha,'registry_sources':sorted(registry.get('sources',{}))}))"
    )
    return timeout_argv(source_environment(["python", "-c", code]), 15)


def build_profile_job(fit_template: list[str], registry_sha256: str) -> dict[str, object]:
    """Build the no-checkpoint actual-B64 profile that gates any full B64 fit."""
    name = "vt_f1_h377_muonall_prop0_b64sqrt_profile_s3"
    argv = replace_option(fit_template, "--name", name)
    argv = replace_option(argv, "--epochs", str(FIT_EPOCHS))
    argv = replace_option(argv, "--checkpoint-epoch", str(FIT_CHECKPOINT_EPOCH))
    argv = replace_option(argv, "--batch", "64")
    argv = replace_option(argv, "--lr", "0.0014142135623730952")
    argv = remove_switch(argv, "--save-training-state")
    argv.extend(["--profile-warmup", "5", "--profile-steps", "3"])
    return {
        "name": name,
        "steps": [
            {"label": "preflight", "timeout": 15, "argv": make_source_preflight(registry_sha256)},
            {
                "label": "profile",
                "timeout": 600,
                "argv": timeout_argv(argv, 600),
            },
        ],
        "packet_role": "profile_only_no_fit_checkpoint",
        "expected_full_fit_artifact": False,
        "profile_shape": {
            "sequence_batch": 64,
            "rows_per_sequence": SEQUENCE_LENGTH,
            "update_rows": UPDATE_ROWS,
        },
    }


def make_prediction_validator(name: str, registry_sha256: str) -> list[str]:
    """Validate export header, row order, checkpoint binding and sidecar hashes."""
    code = (
        "import hashlib,json,numpy as np,pathlib,sys; "
        f"root=pathlib.Path({json.dumps(REMOTE_ROOT)}); name={json.dumps(name)}; "
        f"split_sha={json.dumps(SPLIT_SHA256)}; registry_sha={json.dumps(registry_sha256)}; "
        f"registry=root/{json.dumps(str(INPUT_REGISTRY.relative_to(ROOT)))}; "
        "pred=root/'cache'/f'pred_{name}_valid.npy'; side=pred.with_suffix('.json'); "
        "ckpt=root/'artifacts'/'runs'/name/'model_ep2.pt'; targets=root/'cache'/'valid_targets.npz'; "
        "sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(); "
        "bad=[]; meta=json.loads(side.read_text()); arr=np.load(pred,mmap_mode='r',allow_pickle=False); "
        f"bad.append('shape') if list(arr.shape)!=[{EVALUATION_SEQUENCE_COUNT},{SEQUENCE_LENGTH},2] else None; "
        "bad.append('dtype') if arr.dtype!=np.dtype('float32') else None; "
        "bad.append('run_split_epoch') if (meta.get('run')!=name or meta.get('split')!='valid' or meta.get('epoch')!=2) else None; "
        "bad.append('checkpoint_sha256') if meta.get('checkpoint_sha256')!=sha(ckpt) else None; "
        "bad.append('prediction_sha256') if meta.get('prediction_sha256')!=sha(pred) else None; "
        "bad.append('split_manifest_sha256') if meta.get('split_manifest_sha256')!=split_sha else None; "
        "bad.append('portable_input_manifest_sha256') if sha(registry)!=registry_sha else None; "
        "raw_groups=meta.get('row_groups'); groups=np.asarray(raw_groups,dtype=np.int64) if isinstance(raw_groups,list) else np.asarray([],dtype=np.int64); "
        f"bad.append('row_group_count') if groups.shape!=({EVALUATION_SEQUENCE_COUNT},) else None; "
        "expected=np.asarray(np.load(targets,allow_pickle=False)['row_group'],dtype=np.int64); "
        "bad.append('row_order') if not np.array_equal(groups,expected) else None; "
        "bad.append('target_group_count') if expected.shape!=groups.shape else None; "
        "sys.exit('H377 export validation failed: '+','.join(bad)) if bad else print('H377_EXPORT_OK',json.dumps({'run':name,'shape':list(arr.shape),'dtype':str(arr.dtype),'prediction_sha256':meta['prediction_sha256'],'sidecar_sha256':sha(side),'checkpoint_sha256':meta['checkpoint_sha256'],'row_groups':len(groups),'portable_input_manifest_sha256':registry_sha}))"
    )
    return timeout_argv(source_environment(["python", "-c", code]), 60)


def build_fit_job(
    fit_template: list[str],
    export_template: list[str],
    registry_sha256: str,
    arm: dict[str, str | float | int],
) -> dict[str, object]:
    """Build one exact, single-seed batch/LR fit and full-validation export."""
    name = str(arm["name"])
    argv = replace_option(fit_template, "--name", name)
    argv = replace_option(argv, "--epochs", str(FIT_EPOCHS))
    argv = replace_option(argv, "--checkpoint-epoch", str(FIT_CHECKPOINT_EPOCH))
    argv = replace_option(argv, "--batch", str(arm["batch"]))
    argv = replace_option(argv, "--lr", str(arm["lr"]))
    argv = replace_option(argv, "--wd", str(arm["wd"]))
    export = replace_option(export_template, "vt_f1_muonall_prop0_e4_s3", name)
    export = replace_option(export, "--epoch", str(FIT_CHECKPOINT_EPOCH))
    return {
        "name": name,
        "steps": [
            {"label": "preflight", "timeout": 15, "argv": make_source_preflight(registry_sha256)},
            {"label": "fit", "timeout": 4_300, "argv": timeout_argv(argv, 4_300)},
            {"label": "export", "timeout": 900, "argv": timeout_argv(export, 900)},
            {
                "label": "validate_export",
                "timeout": 60,
                "argv": make_prediction_validator(name, registry_sha256),
            },
        ],
        "packet_role": "fresh_fold1_fit_and_all_valid_export",
        "arm": {key: value for key, value in arm.items() if key != "name"},
        "expected_export": {
            "path": f"cache/pred_{name}_valid.npy",
            "shape": [EVALUATION_SEQUENCE_COUNT, SEQUENCE_LENGTH, 2],
            "dtype": "float32",
            "sidecar": f"cache/pred_{name}_valid.json",
            "checkpoint_epoch": FIT_CHECKPOINT_EPOCH,
            "row_order": "cache/valid_targets.npz row_group order; grader applies fixed fold1 membership",
        },
    }


def canonical_bytes(value: object) -> bytes:
    """Serialize packet objects deterministically with a final newline."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_immutable(path: Path, content: bytes) -> None:
    """Write deterministic packet bytes once; refuse mutation of frozen staging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite changed frozen packet: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def expected_packet() -> tuple[dict[str, bytes], dict[str, object]]:
    """Build every staged job and manifest from frozen producer and split receipts."""
    source_job, fit_template, export_template = load_source_recipe()
    membership = load_membership()
    base_budget, b64_budget = load_schedule_budgets()
    registry_sha256 = sha256_file(INPUT_REGISTRY)
    base_sum = float(base_budget["sum_lr"])
    b64_sum_approx = float(b64_budget["sum_lr"])
    b64_lr_reference = float(b64_budget["peak_lr"])
    exact_sqrt_lr = 0.001 * math.sqrt(2.0)
    b64_sum = b64_sum_approx * exact_sqrt_lr / b64_lr_reference
    dose_matched_wd = 0.01 * base_sum / b64_sum
    arms: list[dict[str, str | float | int]] = [
        {
            "id": "h377_b32_control",
            "name": "vt_f1_h377_muonall_prop0_b32_lr001_wd001_s3_e3",
            "batch": 32,
            "lr": "0.001",
            "wd": "0.01",
        },
        {
            "id": "h377_b64_sqrt_lr_only",
            "name": "vt_f1_h377_muonall_prop0_b64_sqrt_wd001_s3_e3",
            "batch": 64,
            "lr": f"{exact_sqrt_lr:.17g}",
            "wd": "0.01",
        },
        {
            "id": "h377_b64_sqrt_decay_dose_matched",
            "name": "vt_f1_h377_muonall_prop0_b64_sqrt_wdmatch_s3_e3",
            "batch": 64,
            "lr": f"{exact_sqrt_lr:.17g}",
            "wd": f"{dose_matched_wd:.17g}",
        },
        {
            "id": "h377_b64_linear_lr",
            "name": "vt_f1_h377_muonall_prop0_b64_linear_wd001_s3_e3",
            "batch": 64,
            "lr": "0.002",
            "wd": "0.01",
        },
    ]
    jobs: list[tuple[str, dict[str, object]]] = [
        (
            "p00_h377_b64_profile_s3.json",
            build_profile_job(fit_template, registry_sha256),
        )
    ]
    for index, arm in enumerate(arms, start=10):
        jobs.append(
            (
                f"p{index:02d}_{arm['id']}_s3.json",
                build_fit_job(fit_template, export_template, registry_sha256, arm),
            )
        )
    artifacts = {f"staging_refill_jobs_r1/{name}": canonical_bytes(job) for name, job in jobs}
    source_manifest_bytes = SOURCE_MANIFEST.read_bytes()
    aliases_bytes = PROJECT_ALIASES.read_bytes()
    student_endpoint_bytes = STUDENT_ENDPOINT.read_bytes()
    student_endpoint = read_object(STUDENT_ENDPOINT)
    if (
        sha256_bytes(student_endpoint_bytes) != STUDENT_ENDPOINT_SHA256
        or student_endpoint.get("schema") != "wnn.fixed-comparison-endpoint.v1"
        or student_endpoint.get("frozen_before_h377_results") is not True
        or student_endpoint.get("run") != "vt_f1_dsq_dm_t3_a9_e6"
        or student_endpoint.get("epoch") != 5
        or student_endpoint.get("checkpoint_sha256")
        != "c8e08b33a316f3539d4381429233b90403b76de83bda6dbecc84d7952f8212ad"
        or student_endpoint.get("prediction_sha256")
        != "bf1d4a52a67ccd530b0fff80976a6759d6a6265b7c699f77a73cfca9e044d0ad"
        or student_endpoint.get("sidecar_sha256")
        != "2323e59cde003aec195b6216a6367c1c9e9c3e893605b9575165f930ea48cefc"
        or student_endpoint.get("local_official_score") != 0.671031473
    ):
        raise ValueError(
            "conditional student endpoint differs from its immutable pre-grade receipt"
        )
    artifacts["LR377_SOURCE_MANIFEST_R1.json"] = source_manifest_bytes
    artifacts["LR377_PROJECT_ALIASES_R1.json"] = aliases_bytes
    artifacts["LR_FIXED_STUDENT_ENDPOINT_R1.json"] = student_endpoint_bytes
    membership_bytes = canonical_bytes(membership)
    artifacts["MEMBERSHIP_R1.json"] = membership_bytes
    manifest = {
        "schema": "h377-batch-packet-manifest-v1",
        "experiment_id": "H377",
        "packet_revision": "R1",
        "staging_only": True,
        "provider_mutation_performed": False,
        "source_recipe": {
            "job": str(SOURCE_JOB.relative_to(ROOT)),
            "job_name": source_job["name"],
            "job_sha256": sha256_file(SOURCE_JOB),
            "source_role": "completed seed3 prop0 Muon-all current-lead recipe; fresh fit ancestry",
        },
        "source_bindings": {
            "remote_root": REMOTE_ROOT,
            "remote_source_root": REMOTE_SOURCE_ROOT,
            "remote_source_manifest": REMOTE_SOURCE_MANIFEST,
            "source_manifest_record": "LR377_SOURCE_MANIFEST_R1.json",
            "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
            "source_file_count": SOURCE_FILE_COUNT,
            "source_byte_count": SOURCE_BYTE_COUNT,
            "source_files_sha256": read_object(SOURCE_MANIFEST)["sources"],
            "project_alias_record": "LR377_PROJECT_ALIASES_R1.json",
            "project_aliases_sha256": sha256_bytes(aliases_bytes),
            "project_aliases": PROJECT_ALIASES_EXPECTED,
            "fixed_student_endpoint_record": "LR_FIXED_STUDENT_ENDPOINT_R1.json",
            "fixed_student_endpoint_sha256": sha256_bytes(student_endpoint_bytes),
            "note": "Current Vast source frozen byte-for-byte in lr377_source_r1. The older v6 snapshot has different bytes and is not substituted.",
            "portable_input_manifest": str(INPUT_REGISTRY.relative_to(ROOT)),
            "portable_input_manifest_sha256": registry_sha256,
            "split_manifest_sha256": sha256_file(SPLIT_PATH),
        },
        "membership": {
            "record": "MEMBERSHIP_R1.json",
            "sha256": sha256_bytes(membership_bytes),
            "fit_sequences": membership["fit_total_count"],
            "evaluation_valid_sequences": membership["evaluation_valid_count"],
            "actual_membership_source": "fold1 split manifest; train=10533, fit-valid=1416, actual=11949",
        },
        "fixed_recipe": {
            "seed": 3,
            "fold": 1,
            "epochs": FIT_EPOCHS,
            "checkpoint_epoch": FIT_CHECKPOINT_EPOCH,
            "architecture": "GRU-256, 2 layers, linear encoder, zero dropout",
            "optimizer": "Muon on qualifying 2-D groups via --muon-scope all; AdamW fallback for vectors/biases",
            "schedule": "OneCycle, cycling beta1, fp16, weighted-MSE chunk normalization",
            "propensity": "prop-power=0 with existing propensity object retained; default label and class weights",
            "chunk_rows": UPDATE_ROWS,
            "fit_membership_sha256": membership["fit_source_qualified_sha256"],
        },
        "schedule_arithmetic": {
            "receipt": str(BUDGETS_PATH.relative_to(ROOT)),
            "receipt_sha256": sha256_file(BUDGETS_PATH),
            "base_b32": {
                key: base_budget[key]
                for key in ("batch", "epochs", "attempts", "peak_lr", "sum_lr")
            },
            "b64_exact_sqrt_lr": {
                "batch": 64,
                "epochs": FIT_EPOCHS,
                "attempts": b64_budget["attempts"],
                "lr": exact_sqrt_lr,
                "sum_lr": b64_sum,
            },
            "dose_matched_weight_decay": dose_matched_wd,
            "matching_definition": "exact match of nominal integral wd * sum(attempted LR), using actual 11949 membership, drop_last, 20 chunk updates per batch and OneCycle schedule sums; not an exact multiplicative retention match",
            "interpretation": "LR scaling remains a hybrid treatment; all matrices and AdamW fallback share the LR schedule. No SDE prescription or pure-Muon claim.",
        },
        "arms": [{key: value for key, value in arm.items()} for arm in arms],
        "comparison": {
            "primary": "global_wp_clip2_v1 on fixed exposed fold1 evaluation groups",
            "endpoint": "each arm versus fresh B32 control with the same seed, source, membership and model recipe",
            "conditional_value": {
                "receipt": "LR_FIXED_STUDENT_ENDPOINT_R1.json",
                "receipt_sha256": sha256_bytes(student_endpoint_bytes),
                "student_endpoint": f"{student_endpoint['run']}@ep{student_endpoint['epoch']}",
                "checkpoint_sha256": student_endpoint["checkpoint_sha256"],
                "prediction_path": student_endpoint["prediction_path"],
                "prediction_sha256": student_endpoint["prediction_sha256"],
                "prediction_bytes": student_endpoint["prediction_bytes"],
                "prediction_shape": student_endpoint["prediction_shape"],
                "prediction_dtype": student_endpoint["prediction_dtype"],
                "sidecar_sha256": student_endpoint["sidecar_sha256"],
                "standalone_local_score": student_endpoint["local_official_score"],
                "blend": "0.5 * student_prediction + 0.5 * H377_prediction",
                "matched_local_endpoint": "same frozen student's standalone local score from its pre-grade receipt",
            },
            "mix_weights": [0.5, 0.5],
            "no_weight_tuning": True,
            "seed_replicates_in_packet": 1,
            "uncertainty": "evaluation-component resampling and training-seed variation reported separately; no public-score forecast",
        },
        "delivery": {
            "profile_job": "p00_h377_b64_profile_s3.json must finish and be inspected before any H377 fit file is queued",
            "fit_job_upper_bound_seconds": MAX_JOB_SECONDS,
            "timeout_enforced_in_argv": True,
            "daemon_timeout_metadata_is_not_enforced": True,
            "all_valid_output": [EVALUATION_SEQUENCE_COUNT, SEQUENCE_LENGTH, 2],
            "fit_and_prediction_sidecars_required": True,
            "source_upload": False,
            "daemon_replacement": False,
        },
        "known_pending_older_recipe": {
            "snapshot_utc": "2026-10-11 10:26 UTC",
            "b64_linear": "p03_bl_scale_b64_lrlin running; older Muon-rnn/no-propensity recipe",
            "b64_sqrt": "p04_bl_scale_b64_lrsqrt pending; older Muon-rnn/no-propensity recipe",
            "action": "preserve these attempts; use their terminal evidence when available, without treating them as duplicates of current Muon-all x prop0 H377 arms",
        },
        "staged_jobs": [name for name, _ in jobs],
    }
    artifacts["PACKET_MANIFEST_R1.json"] = canonical_bytes(manifest)
    return artifacts, manifest


def validate_expected(artifacts: dict[str, bytes]) -> dict[str, object]:
    """Run static, bounded checks without importing the trainer or starting a fit."""
    if len(artifacts) != 10:
        raise ValueError(
            "H377 R1 must contain five jobs, three source receipts, membership and a manifest"
        )
    jobs: list[dict[str, object]] = []
    for relative, data in artifacts.items():
        if not relative.startswith("staging_refill_jobs_r1/"):
            continue
        job = json.loads(data)
        if not isinstance(job, dict):
            raise ValueError(f"staged job is not a JSON object: {relative}")
        jobs.append(job)
    names = [str(job["name"]) for job in jobs]
    if len(names) != len(set(names)) or len(jobs) != 5:
        raise ValueError("staged job names must be five unique IDs")
    profile_jobs = [
        job for job in jobs if job.get("packet_role") == "profile_only_no_fit_checkpoint"
    ]
    if len(profile_jobs) != 1:
        raise ValueError("packet needs exactly one profile-only job")
    profile_steps = profile_jobs[0]["steps"]
    if not isinstance(profile_steps, list) or [step.get("label") for step in profile_steps] != [
        "preflight",
        "profile",
    ]:
        raise ValueError("profile must run as a standalone first job after identity preflight")
    profile_argv = profile_steps[1].get("argv")
    if not isinstance(profile_argv, list) or "--save-training-state" in profile_argv:
        raise ValueError("profile must not save or claim a full-fit checkpoint")
    if (
        f"{REMOTE_SOURCE_ROOT}/scripts/train_rnn.py" not in profile_argv
        or f"{REMOTE_ROOT}/scripts/train_rnn.py" in profile_argv
    ):
        raise ValueError("profile must invoke only the immutable R1 trainer snapshot")
    for flag, value in (
        ("--batch", "64"),
        ("--epochs", "3"),
        ("--profile-warmup", "5"),
        ("--profile-steps", "3"),
    ):
        if option_value(profile_argv, flag) != value:
            raise ValueError(f"profile shape or bound has unexpected {flag}")
    if profile_steps[1].get("timeout") != 600 or profile_argv[:4] != [
        "timeout",
        "--signal=TERM",
        "--kill-after=10s",
        "600s",
    ]:
        raise ValueError("actual-B64 profile must be explicitly bounded to 600 seconds")
    full_fits = [
        job for job in jobs if job.get("packet_role") == "fresh_fold1_fit_and_all_valid_export"
    ]
    if len(full_fits) != 4:
        raise ValueError("packet must contain exactly four preregistered fit comparisons")
    for job in full_fits:
        steps = job.get("steps")
        if not isinstance(steps, list) or [step.get("label") for step in steps] != [
            "preflight",
            "fit",
            "export",
            "validate_export",
        ]:
            raise ValueError("fit jobs must preserve preflight, fit, export and validation order")
        total_bound = 0
        for step in steps:
            timeout = step.get("timeout")
            argv = step.get("argv")
            if not isinstance(timeout, int) or not isinstance(argv, list):
                raise ValueError("every step needs bounded timeout metadata and argv")
            if timeout > MAX_JOB_SECONDS or argv[:1] != ["timeout"]:
                raise ValueError("every staged command must enforce an explicit <=5400s timeout")
            timeout_value = argv[3] if len(argv) > 3 else ""
            if timeout_value != f"{timeout}s":
                raise ValueError("GNU timeout argv and daemon step metadata disagree")
            if "-c" in argv:
                try:
                    compile(argv[argv.index("-c") + 1], f"{job['name']}:{step['label']}", "exec")
                except (IndexError, SyntaxError) as error:
                    raise ValueError(
                        f"inline static command is invalid: {job['name']} {step['label']}"
                    ) from error
            total_bound += timeout + 10
        if total_bound > MAX_JOB_SECONDS:
            raise ValueError("cumulative step timeouts and kill grace exceed 5400 seconds")
        fit_argv = steps[1].get("argv")
        export_argv = steps[2].get("argv")
        validation_argv = steps[3].get("argv")
        if not all(isinstance(value, list) for value in (fit_argv, export_argv, validation_argv)):
            raise ValueError("fit, export and validation argv must be lists")
        if f"{REMOTE_SOURCE_ROOT}/scripts/train_rnn.py" not in fit_argv:
            raise ValueError("fit must invoke only the immutable R1 trainer snapshot")
        if f"{REMOTE_SOURCE_ROOT}/scripts/predict_run.py" not in export_argv:
            raise ValueError("export must invoke only the immutable R1 predictor snapshot")
        if (
            option_value(fit_argv, "--epochs") != "3"
            or option_value(fit_argv, "--checkpoint-epoch") != "2"
        ):
            raise ValueError("fit checkpoint policy must select epoch 2 from a three-epoch cycle")
        if option_value(export_argv, "--epoch") != "2":
            raise ValueError("export must select the frozen epoch-2 checkpoint")
        if job.get("expected_export", {}).get("shape") != [1_873, 20_000, 2]:
            raise ValueError("export artifact shape must match all supplied validation sequences")
        if validation_argv[:4] != ["timeout", "--signal=TERM", "--kill-after=10s", "60s"]:
            raise ValueError(
                "all-valid artifact validation must be explicitly bounded to 60 seconds"
            )
    budget_base, budget_b64 = load_schedule_budgets()
    exact_lr = 0.001 * math.sqrt(2.0)
    s1_b64 = float(budget_b64["sum_lr"]) * exact_lr / float(budget_b64["peak_lr"])
    expected_wd = 0.01 * float(budget_base["sum_lr"]) / s1_b64
    if abs(expected_wd - 0.014180787422138722) > 1e-15:
        raise ValueError("nominal integrated weight-decay match changed")
    membership = json.loads(artifacts["MEMBERSHIP_R1.json"])
    if (
        membership.get("fit_total_count") != 11_949
        or membership.get("evaluation_valid_count") != 457
    ):
        raise ValueError("source-qualified membership counts differ from the fold1 receipt")
    if sha256_bytes(artifacts["LR377_SOURCE_MANIFEST_R1.json"]) != SOURCE_MANIFEST_SHA256:
        raise ValueError("packet source manifest is not the root-frozen snapshot receipt")
    if sha256_bytes(artifacts["LR377_PROJECT_ALIASES_R1.json"]) != PROJECT_ALIASES_SHA256:
        raise ValueError("packet project-alias receipt differs from the root-frozen receipt")
    endpoint = json.loads(artifacts["LR_FIXED_STUDENT_ENDPOINT_R1.json"])
    if (
        sha256_bytes(artifacts["LR_FIXED_STUDENT_ENDPOINT_R1.json"]) != STUDENT_ENDPOINT_SHA256
        or endpoint.get("frozen_before_h377_results") is not True
        or endpoint.get("prediction_sha256")
        != "bf1d4a52a67ccd530b0fff80976a6759d6a6265b7c699f77a73cfca9e044d0ad"
        or endpoint.get("sidecar_sha256")
        != "2323e59cde003aec195b6216a6367c1c9e9c3e893605b9575165f930ea48cefc"
    ):
        raise ValueError("conditional 50/50 comparison lacks its exact frozen student artifacts")
    return {
        "schema": "h377-static-validation-v1",
        "status": "PASS_STATIC_ONLY",
        "job_count": len(jobs),
        "full_fit_count": len(full_fits),
        "profile_count": len(profile_jobs),
        "fit_membership_count": membership["fit_total_count"],
        "evaluation_membership_count": membership["evaluation_valid_count"],
        "b32_attempts": budget_base["attempts"],
        "b64_attempts": budget_b64["attempts"],
        "exact_sqrt_lr": exact_lr,
        "dose_matched_wd": expected_wd,
        "official_full_valid_shape": [1_873, 20_000, 2],
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_file_count": SOURCE_FILE_COUNT,
        "conditional_student_prediction_sha256": endpoint["prediction_sha256"],
        "conditional_student_sidecar_sha256": endpoint["sidecar_sha256"],
        "conditional_student_score": endpoint["local_official_score"],
        "gpu_or_model_fit_started": False,
        "provider_mutation_performed": False,
        "limitations": [
            "actual remote profile receipt remains a root/operator gate",
            "single seed and exposed fold1 do not establish recipe rank or public transfer",
        ],
    }


def build_packet() -> None:
    """Write immutable H377 staging artifacts and a static validation receipt."""
    artifacts, _manifest = expected_packet()
    validation = validate_expected(artifacts)
    for relative, content in artifacts.items():
        write_immutable(CAMPAIGN / relative, content)
    builder_hash = sha256_file(Path(__file__))
    validation["builder_sha256"] = builder_hash
    validation["packet_manifest_sha256"] = sha256_bytes(artifacts["PACKET_MANIFEST_R1.json"])
    validation["staged_jobs"] = [
        relative for relative in artifacts if relative.startswith("staging_refill_jobs_r1/")
    ]
    write_immutable(CAMPAIGN / "STATIC_VALIDATION_R1.json", canonical_bytes(validation))


def check_packet() -> None:
    """Verify every frozen staged byte against deterministic regeneration."""
    artifacts, _manifest = expected_packet()
    validation = validate_expected(artifacts)
    for relative, content in artifacts.items():
        path = CAMPAIGN / relative
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"frozen packet differs from deterministic regeneration: {path}")
    receipt = read_object(CAMPAIGN / "STATIC_VALIDATION_R1.json")
    if receipt.get("status") != "PASS_STATIC_ONLY" or receipt.get("builder_sha256") != sha256_file(
        Path(__file__)
    ):
        raise ValueError("static validation receipt is missing or bound to another builder")
    if receipt.get("packet_manifest_sha256") != sha256_bytes(artifacts["PACKET_MANIFEST_R1.json"]):
        raise ValueError("static validation receipt does not bind the packet manifest")
    if validation.get("status") != receipt.get("status"):
        raise ValueError("regenerated validation result differs from the saved receipt")
    print(
        json.dumps(
            {
                "status": "PASS_STATIC_ONLY",
                "jobs": 5,
                "manifest_sha256": receipt["packet_manifest_sha256"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    """Build once or check the immutable H377 packet using stdlib only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    args = parser.parse_args()
    if args.action == "build":
        build_packet()
    else:
        check_packet()


if __name__ == "__main__":
    main()
