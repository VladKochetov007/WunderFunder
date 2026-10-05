"""Create the review packet's immutable source/draft identity manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import controller

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SPECS = HERE / "draft_specs"


def main() -> None:
    drafts = sorted(SPECS.glob("*.draft.json"))
    draft_values = [(p, json.loads(p.read_text())) for p in drafts]
    code_paths = sorted({x for _, spec in draft_values for x in spec["code"]})
    code = []
    for rel in code_paths:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"draft code input missing: {rel}")
        code.append({"path": rel, "size": path.stat().st_size, "sha256": controller.sha256_file(path)})
    draft_ids = []
    fresh_outputs = []
    for p, spec in draft_values:
        for rel in spec["artifacts"]:
            output_path = ROOT / rel
            if output_path.exists():
                raise FileExistsError(f"fresh E427 output collision: {rel}")
            fresh_outputs.append(rel)
        draft_ids.append({"path": str(p.relative_to(ROOT)), "name": spec["name"],
                          "sha256": controller.sha256_file(p), "kind": spec["kind"],
                          "limits": spec["limits"], "inputs": spec["inputs"],
                          "artifacts": spec["artifacts"]})
    e464 = json.loads((HERE / "E464_ADMISSION_RECEIPT.json").read_text())
    attempt_ids_path = SPECS / "attempt_ids.json"
    test_path = HERE / "test_controller.py"
    verifier_path = HERE / "build_manifest.py"
    readme_path = HERE / "README.md"
    ram_report_path = HERE / "RAM_FEASIBILITY.md"
    stats_path = ROOT / "research/campaign_2026-10-04/checks/e427_context_stats_s3_v1.json"
    stats = json.loads(stats_path.read_text())
    parent = ROOT / stats["parent_checkpoint_path"]
    prov = parent.parent / "provenance.json"
    portable_path = ROOT / "research/campaign_2026-10-04/portable_inputs.json"
    portable = json.loads(portable_path.read_text())
    manifest = {
        "schema": "wnn.e427-oct05-recovery-manifest-v1",
        "status": "draft_only_blocked_by_unchanged_resource_reservation",
        "code": code,
        "draft_specs": draft_ids,
        "fresh_outputs_absent_at_packet_build": sorted(fresh_outputs),
        "attempt_ids": {"path": str(attempt_ids_path.relative_to(ROOT)),
                        "sha256": controller.sha256_file(attempt_ids_path)},
        "verification": {"test_path": str(test_path.relative_to(ROOT)),
                         "test_sha256": controller.sha256_file(test_path),
                         "manifest_builder_path": str(verifier_path.relative_to(ROOT)),
                         "manifest_builder_sha256": controller.sha256_file(verifier_path),
                         "readme_path": str(readme_path.relative_to(ROOT)),
                         "readme_sha256": controller.sha256_file(readme_path),
                         "ram_feasibility_path": str(ram_report_path.relative_to(ROOT)),
                         "ram_feasibility_sha256": controller.sha256_file(ram_report_path),
                         "result": "6 bounded CPU unit tests pass; py_compile pass"},
        "fixed_design_path": "research/campaign_2026-10-04/staging/e427_pipeline_v1/specs/e427_screen_v1.json",
        "fixed_design_sha256": controller.sha256_file(ROOT / "research/campaign_2026-10-04/staging/e427_pipeline_v1/specs/e427_screen_v1.json"),
        "e464_receipt": {"path": str((HERE / "E464_ADMISSION_RECEIPT.json").relative_to(ROOT)),
                         "sha256": controller.sha256_file(HERE / "E464_ADMISSION_RECEIPT.json"),
                         "attempt_count": len(e464["attempts"]), "report_sha256": e464["report_sha256"]},
        "e427_prerequisites": {
            "stats": {"path": str(stats_path.relative_to(ROOT)), "size": stats_path.stat().st_size,
                      "sha256": controller.sha256_file(stats_path)},
            "parent_checkpoint": {"path": str(parent.relative_to(ROOT)), "size": parent.stat().st_size,
                                  "sha256": controller.sha256_file(parent)},
            "parent_provenance": {"path": str(prov.relative_to(ROOT)), "size": prov.stat().st_size,
                                  "sha256": controller.sha256_file(prov)},
            "portable_input_manifest": {"path": str(portable_path.relative_to(ROOT)), "size": portable_path.stat().st_size,
                                         "sha256": controller.sha256_file(portable_path)},
            "pinned_data": {p: {"size": x["original_identity"]["bytes"], "sha256": x["sha256"]}
                            for p, x in portable["sources"].items() if p in stats["input_sha256"] or p in
                            ("datasets/train.parquet", "datasets/valid.parquet", "cache/train_targets.npz", "cache/valid_targets.npz")},
            "packet_time_validation": "stats/parent/provenance bytes were rehashed; 29.24GB train and 5.62GB valid files retain pinned portable-manifest hashes and must be rehashed by controller before any admission; not streamed in packet build",
        },
        "resource_policy": {
            "parent_hard_ram_bytes": 30_000_000_000,
            "worker_ram_bytes": controller.WORKER_RAM_BYTES,
            "support_ram_reserve_bytes": controller.SUPPORT_RESERVE_BYTES,
            "combined_bytes": controller.WORKER_RAM_BYTES + controller.SUPPORT_RESERVE_BYTES,
            "shortfall_before_usage_bytes": controller.WORKER_RAM_BYTES + controller.SUPPORT_RESERVE_BYTES - 30_000_000_000,
            "gpu_allocator_bytes": controller.GPU_ALLOCATOR_BYTES,
            "gpu_stop_bytes": controller.GPU_STOP_BYTES,
            "aggregate_gpu_timeout_seconds": 18_900,
            "stop_new_stage_utc": "2026-10-05T07:00:00Z",
        },
        "runner_state": "research/campaign_2026-10-05/volume_gate_recovery/runner",
        "system_observation": {
            "parent_cgroup_snapshot": {"utc": "2026-10-05T01:03:27Z", "memory_current_bytes": 25_484_705_792,
                                       "memory_soft_bytes": 27_000_000_000, "memory_hard_bytes": 30_000_000_000,
                                       "source": "root-provided parent-slice observation; stale by packet build"},
            "packet_host_observation_utc": "2026-10-05T01:14:03Z",
            "packet_host_gpu": "RTX 3060 Laptop, 6144 MiB total, 359 MiB used, 5499 MiB free, 0% utilization",
            "fresh_admission": "required; packet host cgroup memory.current unavailable at observation path; do not infer fit admission",
        },
        "launch_authority": "root must review and seal; this packet has no launch path and started no worker",
    }
    dest = HERE / "IMMUTABLE_MANIFEST.json"
    dest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    (HERE / "IMMUTABLE_MANIFEST.sha256").write_text(f"{digest}  {dest.name}\n")


if __name__ == "__main__":
    main()
