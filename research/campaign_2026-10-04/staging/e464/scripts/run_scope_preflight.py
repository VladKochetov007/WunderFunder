"""Run E464 scope tests and eight bounded, actual-parent preflight checks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
STAGING = ROOT / "research/campaign_2026-10-04/staging/e464"
EXECUTION_SPECS = STAGING / "execution_specs"
OUTPUT = ROOT / "research/campaign_2026-10-04/checks/e464_scope_preflight_v1"
TESTS = STAGING / "tests"
EXPECTED_ARMS = (
    "h093_learned",
    "h093_hash",
    "h093_universal",
    "h093_gru_budget",
    "h094_conditional",
    "h094_unconditional",
    "h094_diagonal",
    "h094_gru_budget",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def preflight_argv(spec: dict[str, Any], audit_path: Path) -> list[str]:
    raw = spec.get("argv")
    if not isinstance(raw, list) or not raw or raw[0] != "{python}":
        raise ValueError("fit draft argv must begin with the {python} placeholder")
    argv = [sys.executable, *(str(item) for item in raw[1:])]
    if len(argv) < 3 or argv[1] != "research/campaign_2026-10-04/staging/e464/scripts/run_head_fit.py":
        raise ValueError("fit draft does not call the registered E464 wrapper")
    if "--preflight-only" in argv:
        raise ValueError("fit draft unexpectedly already contains --preflight-only")
    if argv.count("--audit") != 1:
        raise ValueError("fit draft must contain exactly one --audit argument")
    audit_index = argv.index("--audit")
    if audit_index + 1 >= len(argv):
        raise ValueError("fit draft --audit is missing its value")
    argv[audit_index + 1] = audit_path.relative_to(ROOT).as_posix()
    argv.append("--preflight-only")
    if any(token in argv for token in ("--epochs", "train_rnn.py")):
        raise ValueError("preflight argv must not include a trainer command")
    return argv


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite preflight directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    # Lock the intended one-epoch shape arithmetic independently of the runner.
    fit_sequences, batch_size = 1505, 16
    sequence_rows, update_rows = 20_000, 2_000
    batches = fit_sequences // batch_size
    attempts_per_batch = sequence_rows // update_rows
    attempts = batches * attempts_per_batch
    if (batches, attempts_per_batch, attempts) != (94, 10, 940):
        raise RuntimeError("E464 fixed recipe arithmetic changed unexpectedly")

    tests_log = OUTPUT / "stdlib_tests.log"
    tests_argv = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(TESTS),
        "-p",
        "test_*.py",
        "-v",
    ]
    tests = subprocess.run(
        tests_argv,
        cwd=ROOT,
        env=child_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
        check=False,
    )
    tests_log.write_text(tests.stdout, encoding="utf-8")
    if tests.returncode != 0:
        raise RuntimeError(f"E464 stdlib tests failed; see {tests_log}")
    if "Ran 8 tests" not in tests.stdout or "OK" not in tests.stdout:
        raise RuntimeError("expected exactly eight passing E464 stdlib tests")

    records: list[dict[str, Any]] = []
    for arm in EXPECTED_ARMS:
        spec_path = EXECUTION_SPECS / f"c20261004_e464_{arm}_head_s3_final.draft.json"
        spec = load_json(spec_path)
        if spec.get("kind") != "gpu" or spec.get("name") != f"c20261004_e464_{arm}_head_s3_final":
            raise ValueError(f"unexpected registered fit identity in {spec_path}")
        argv = preflight_argv(spec, OUTPUT / f"{arm}.json")
        log_path = OUTPUT / f"{arm}.log"
        result = subprocess.run(
            argv,
            cwd=ROOT,
            env=child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=75,
            check=False,
        )
        log_path.write_text(result.stdout, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"{arm} actual-parent preflight failed; see {log_path}")
        audit = load_json(OUTPUT / f"{arm}.json")
        if audit.get("status") != "preflight_pass" or audit.get("trainer_started") is not False:
            raise RuntimeError(f"{arm} did not return a non-training preflight pass")
        if audit.get("arm") != arm or audit.get("expected_fit_valid_group_count") != fit_sequences:
            raise RuntimeError(f"{arm} parent identity or 1,505-group count mismatch")
        if audit.get("parent_checkpoint_sha256") != _argv_value(argv, "--expected-parent-sha256"):
            raise RuntimeError(f"{arm} audit parent hash differs from its registered argv")
        records.append(
            {
                "arm": arm,
                "returncode": result.returncode,
                "audit_sha256": sha256_file(OUTPUT / f"{arm}.json"),
                "log_sha256": sha256_file(log_path),
                "parent_checkpoint_sha256": audit["parent_checkpoint_sha256"],
                "architecture": audit["architecture"],
                "parameters": audit["parameter_count"],
                "buffers": audit["buffer_count"],
                "fit_groups": audit["expected_fit_valid_group_count"],
                "frozen_parameter_count": len(audit["frozen_parameter_names"]),
                "trainable_head_parameter_count": len(audit["trainable_head_parameter_names"]),
                "trainer_started": False,
            }
        )

    report = {
        "status": "pass",
        "protocol": "E464 eight-parent CPU-only scope preflight; no training",
        "tests": {"expected": 8, "passed": 8, "log_sha256": sha256_file(tests_log)},
        "training_shape_arithmetic": {
            "fit_sequences": fit_sequences,
            "batch_size": batch_size,
            "drop_last_batches": batches,
            "rows_per_sequence": sequence_rows,
            "rows_per_optimizer_update": update_rows,
            "updates_per_batch": attempts_per_batch,
            "optimizer_attempts_per_epoch": attempts,
        },
        "arms": records,
    }
    (OUTPUT / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "arms": len(records), "trainer_started": False}))
    return 0


def _argv_value(argv: list[str], option: str) -> str:
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise ValueError(f"missing argv value for {option}")
    return argv[index + 1]


if __name__ == "__main__":
    raise SystemExit(main())
