from __future__ import annotations

import importlib.util
import sys
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
P = Path(__file__).parent / "runtime/scripts/kaggle_gpu_runner.py"
spec = importlib.util.spec_from_file_location("w384_v6_runner", P)
runner = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(runner)

G = 1024**3
GPU = [{"index": 0, "uuid": "T4-0", "name": "Tesla T4", "total_bytes": 16*G, "used_bytes": 0},
       {"index": 1, "uuid": "T4-1", "name": "Tesla T4", "total_bytes": 16*G, "used_bytes": 0}]
SPECS = [{"name": f"w{w}", "limits": {"ram_bytes": 8*G, "cpus": 2, "gpu_stop_bytes": 13*G}} for w in (256,384)]

def host(cap, available, affinity=4):
    return {"capacity_bytes": int(cap*G), "available_bytes": int(available*G), "affinity": list(range(affinity))}

def check_reject(label, specs, h, devices=GPU):
    try:
        runner.admit_specs(specs, h, devices)
    except ValueError:
        return
    raise AssertionError(f"unsafe case admitted: {label}")

# Science, masks, data and schedule retain the staged v5 values; only RAM reservation and
# the two resource-observability wrappers differ in source code identity.
V4 = ROOT / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4_transport/provider_output_v1/experiment/kaggle_specs"
HERE = Path(__file__).parent
for name in (
    "c20261004_w256_adamw_s3_f1_t4v4_probe",
    "c20261004_w256_adamw_s3_f1_t4v4_fit",
    "c20261004_w384_adamw_s3_f1_t4v4_probe",
    "c20261004_w384_adamw_s3_f1_t4v4_fit",
):
    old = json.loads((V4 / f"{name}.json").read_text())
    new = json.loads((HERE / f"specs/{name}.json").read_text())
    for field in ("argv", "inputs", "artifacts", "environment", "kind", "name", "version"):
        assert new[field] == old[field], (name, field)
    old_limits = dict(old["limits"])
    new_limits = dict(new["limits"])
    old_limits.pop("ram_bytes")
    new_limits.pop("ram_bytes")
    assert new_limits == old_limits and new["limits"]["ram_bytes"] == 8*G, name
    old_code = {x["path"]: (x["size"],x["sha256"]) for x in old["code"]}
    new_code = {x["path"]: (x["size"],x["sha256"]) for x in new["code"]}
    for path, identity in old_code.items():
        if path.endswith("/phase_runner.py") or path == "scripts/kaggle_gpu_runner.py":
            continue
        assert new_code[path] == identity, (name, path)
    assert len(new_code) == len(old_code)

# Exact operational-budget boundaries, including admission error that caused v5 failure.
assert runner.admit_specs(SPECS, host(32, 24), GPU) == 2      # budget=min(25.6,17.6)=17.6
assert runner.admit_specs(SPECS, host(32, 18), GPU) == 1      # budget=11.6, one 8 GiB worker
assert runner.admit_specs(SPECS, host(32, 16), GPU) == 1      # budget=9.6, one worker
check_reject("budget below one reservation", SPECS, host(32, 14.3))
check_reject("80 percent capacity cap", SPECS, host(9, 9))
check_reject("less than two CPU slots", SPECS, host(32, 24, affinity=1))
assert runner.admit_specs(SPECS, host(32, 24, affinity=2), GPU) == 1
check_reject("preexisting GPU memory", SPECS, host(32,24), [dict(GPU[0], used_bytes=128*1024**2+1), GPU[1]])
check_reject("stop threshold reaches physical VRAM", [{"limits": {"ram_bytes": 8*G,"cpus":1,"gpu_stop_bytes":16*G}}], host(32,24), GPU[:1])
# Different job reservations: worst-pair ordering controls concurrency, not input order.
hetero = [{"limits":{"ram_bytes":12*G,"cpus":2,"gpu_stop_bytes":13*G}}, {"limits":{"ram_bytes":8*G,"cpus":2,"gpu_stop_bytes":13*G}}]
assert runner.admit_specs(hetero, host(32,24), GPU) == 1

# A failed cache hint is observable and never overrides the independently sampled host budget.
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "declared.dat").write_bytes(b"bounded-fixture")
    original = runner.os.posix_fadvise
    def fail_fadvise(fd, offset, length, advice):
        raise OSError(38, "fixture fadvise failure")
    runner.os.posix_fadvise = fail_fadvise
    try:
        advice = runner.release_verified_file_cache(tmp, [{"path": "declared.dat", "size": 15}])
    finally:
        runner.os.posix_fadvise = original
    assert advice["status"] == "partial_failure" and advice["failures"][0]["errno"] == 38
    low_host = host(32, 18)
    assert runner.admit_specs(SPECS, low_host, GPU) == 1
    assert low_host["available_bytes"] == int(18*G)

# Exercise the real run-path paired-capacity failure: receipt is written and no
# subprocess is spawned when the host can fit one reservation but not the pair.
from kaggle_common import canonical_sha256
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    spec_file = tmp / "spec.json"
    spec_obj = {
        "version": 1, "name": "memory-gate-test-256", "kind": "gpu",
        "argv": ["{python}", "must_not_run.py"], "code": [],
        "code_sha256": canonical_sha256([]), "inputs": [], "artifacts": [],
        "limits": {"cpus": 2, "ram_bytes": 8*G, "timeout_seconds": 60,
                   "gpu_memory_bytes": 11*G, "gpu_stop_bytes": 13*G},
        "environment": {},
    }
    spec_file.write_text(json.dumps(spec_obj))
    second_spec = dict(spec_obj, name="memory-gate-test-384")
    second_spec_file = tmp / "spec2.json"
    second_spec_file.write_text(json.dumps(second_spec))
    runner.host_inventory = lambda: host(32, 18)
    runner.gpu_inventory = lambda: GPU
    out = tmp / "out"
    spawned = []
    old_popen = runner.subprocess.Popen
    runner.subprocess.Popen = lambda *a, **k: spawned.append((a, k))
    try:
        try:
            runner.run(tmp, [spec_file, second_spec_file], out)
        except ValueError as exc:
            assert "concurrency 2" in str(exc)
        else:
            raise AssertionError("paired phase must reject concurrency one")
    except ValueError as exc:
        raise
    finally:
        runner.subprocess.Popen = old_popen
    inventory = json.loads((out / "inventory.json").read_text())
    admission = json.loads((out / "admission.json").read_text())
    assert inventory["after_validation_and_cache_advice"]["host"]["available_bytes"] == int(18*G)
    assert admission["status"] == "rejected" and admission["error_type"] == "InsufficientPairedCapacity"
    assert admission["devices"] == GPU
    assert spawned == []
print("passed 12 resource, cache-advice, and fail-before-worker cases; guards unchanged")
