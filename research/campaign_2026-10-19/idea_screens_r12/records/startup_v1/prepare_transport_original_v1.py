from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
CAMPAIGN = REPO / "research/campaign_2026-10-19"
R10_PACKET = CAMPAIGN / "idea_screens_r10/records/startup_v1"
R11_PACKET = CAMPAIGN / "idea_screens_r11/records/startup_v1"
R12 = CAMPAIGN / "idea_screens_r12"
PACKET = R12 / "records/startup_v1"
PRIVATE = Path("/Users/nb/Documents/wunder/runtime-20261018/eda19-r12-v1")
PYTHON = "/Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python"

PINS = {
    "assembler": "bae60cedb3d0f3316db784fc0cdde52d1b3fdb638fe21a7a83f6eefdc2ab40df",
    "inline": "b9b44d7a50fcfe7b926bb4aa6fa1c781e9d202b65996d61915deae188a5a3734",
    "base_spec": "715803eb2269c1714218c714af7ed59e6077d21a4adcfe81b742200e4b22aa68",
    "manifest": "20d26c8adfddef62feb76edebba254751bd2f7c3cbecddfd2487606c7690ec86",
    "review": "706ac94949b722a6bbc3813700242ab62f1dd8f2a85efa026404ec89703b1b18",
    "tests": "dc7623a70722f1a19d9c8bbfe5695e9afac19ea8982e9f36e5596b0edbc69b60",
    "smoke": "c9a8d9e5893c62e1d67e779cbdae84c4dd243ce07a28b8d30ca499a22250df5b",
}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def replace_exact(
    source: str, old: str, new: str, count: int, label: str, log: list[dict[str, Any]]
) -> str:
    found = source.count(old)
    if found != count:
        raise RuntimeError(f"{label}: expected {count} literal occurrences, found {found}")
    log.append(
        {
            "id": label,
            "occurrences": count,
            "source_literal_sha256": sha(old.encode()),
            "replacement_literal_sha256": sha(new.encode()),
        }
    )
    return source.replace(old, new)


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)


def main() -> None:
    manifest_bytes = (R12 / "PREPARATION_MANIFEST.json").read_bytes()
    review_bytes = (R12 / "IMPLEMENTATION_REVIEW.json").read_bytes()
    tests_bytes = (R12 / "TEST_RECEIPT.json").read_bytes()
    readme_bytes = (R12 / "README.md").read_bytes()
    smoke_bytes = (PACKET / "LOCAL_IMPORT_SMOKE.json").read_bytes()
    for name, payload in [
        ("manifest", manifest_bytes),
        ("review", review_bytes),
        ("tests", tests_bytes),
        ("smoke", smoke_bytes),
    ]:
        if sha(payload) != PINS[name]:
            raise RuntimeError(f"R12 {name} bytes differ from the reviewed freeze")

    manifest = json.loads(manifest_bytes)
    review = json.loads(review_bytes)
    smoke = json.loads(smoke_bytes)
    if manifest.get("schema") != "nf19.r12_preparation.v1":
        raise RuntimeError("unexpected R12 manifest schema")
    if len(manifest.get("files", {})) != 32 or len(manifest.get("runtime_py", {})) != 26:
        raise RuntimeError("R12 manifest is not the reviewed 32-pin/26-module closure")
    if not str(review.get("status", "")).startswith("PASS_SOURCE_AND_SYNTHETIC_REVIEW"):
        raise RuntimeError("R12 implementation review is not PASS")
    if review.get("bindings", {}).get("preparation_manifest_sha256") != PINS["manifest"]:
        raise RuntimeError("R12 review does not bind the frozen manifest")
    cold_smoke = smoke.get("cold_smoke", {})
    if (
        smoke.get("status") != "PASS"
        or smoke.get("manifest_sha256") != PINS["manifest"]
        or cold_smoke.get("status") != "PASS"
        or any(cold_smoke.get("sentinel_calls", {}).values())
    ):
        raise RuntimeError("R12 cold-import smoke does not pass against the frozen manifest")
    if smoke.get("array_model_or_training_payloads_read") is not False:
        raise RuntimeError("R12 cold-import smoke reports a forbidden payload read")

    base_spec_path = R11_PACKET / "ASSEMBLY_SPEC.json"
    base_spec_bytes = base_spec_path.read_bytes()
    assembler_path = R11_PACKET / "assemble_transport_v1.py"
    assembler_bytes = assembler_path.read_bytes()
    inline_path = R11_PACKET / "root_inline_v1.py"
    inline_bytes = inline_path.read_bytes()
    for name, payload in [
        ("base_spec", base_spec_bytes),
        ("assembler", assembler_bytes),
        ("inline", inline_bytes),
    ]:
        if sha(payload) != PINS[name]:
            raise RuntimeError(f"pinned original R11 {name} changed")
    base_spec = json.loads(base_spec_bytes)
    for name, expected in base_spec["r10_base_pins"].items():
        if sha((R10_PACKET / name).read_bytes()) != expected:
            raise RuntimeError(f"pinned R10 helper changed: {name}")

    spec_path = PACKET / "ASSEMBLY_SPEC.json"
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    expected_spec_values = {
        "schema": "nf19.r12_startup_assembly_spec.v1",
        "unit": "computer-vision-homework-eda19-r12-v1.service",
        "destination": "support-source/eda19-r12-v1",
        "output": "research_tmp/nf19-r12-v1",
        "sample": base_spec["sample"],
        "sample_sha256": base_spec["sample_sha256"],
        "membership_sha256": base_spec["membership_sha256"],
        "source_snapshot": base_spec["source_snapshot"],
        "runtime_py": manifest["runtime_py"],
        "r10_base_pins": base_spec["r10_base_pins"],
    }
    if any(spec.get(key) != value for key, value in expected_spec_values.items()):
        raise RuntimeError("R12 assembly spec differs from frozen inputs or fixed identity")
    canonical_names = sorted(
        name for name in manifest["files"] if name.startswith("canonical:")
    )
    if spec.get("canonical_pin_names") != canonical_names or len(canonical_names) != 6:
        raise RuntimeError("R12 assembly spec does not bind all six canonical pins")

    inline = inline_bytes.decode("utf-8")
    inline_log: list[dict[str, Any]] = []
    for old, new, count, label in [
        ("idea_screens_r11", "idea_screens_r12", 3, "inline-package-namespace"),
        ("r11", "r12", 4, "inline-schema-and-path-identities"),
        ("R11", "R12", 14, "inline-labels"),
    ]:
        inline = replace_exact(inline, old, new, count, label, inline_log)
    if "idea_screens_r11" in inline or "r11" in inline or "R11" in inline:
        raise RuntimeError("R12 original-inline replay retained an R11 identity")
    compile(inline, "root_inline_r12_original_v1.py", "exec")
    inline_output = PACKET / "root_inline_r12_original_v1.py"
    write_new(inline_output, inline.encode("utf-8"))

    assembler = assembler_bytes.decode("utf-8")
    assembler_log: list[dict[str, Any]] = []
    for old, new, count, label in [
        ("idea_screens_r11", "idea_screens_r12", 25, "assembler-package-namespace"),
        ("r11", "r12", 17, "assembler-schema-and-path-identities"),
        ("R11", "R12", 29, "assembler-labels"),
        ("three-file set", "six-file set", 2, "canonical-pin-count-description"),
        (
            "root_inline_v1.py",
            "root_inline_r12_original_v1.py",
            2,
            "original-inline-template-path",
        ),
    ]:
        assembler = replace_exact(assembler, old, new, count, label, assembler_log)
    if "idea_screens_r11" in assembler or "r11" in assembler or "R11" in assembler:
        raise RuntimeError("R12 original-assembler replay retained an R11 identity")
    compile(assembler, "assemble_transport_r12_original_v1.py", "exec")

    PRIVATE.mkdir(parents=True, exist_ok=False)
    private_assembler = PRIVATE / "assemble_transport_r12.py"
    write_new(private_assembler, assembler.encode("utf-8"))
    result = subprocess.run(
        [PYTHON, str(private_assembler)],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("local assembly failed before any remote action: " + result.stderr[-4000:])

    receipt_path = PACKET / "ASSEMBLY_RECEIPT.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    for name, entry in receipt["generated_scripts"].items():
        generated_path = PRIVATE / name
        generated_bytes = generated_path.read_bytes()
        if len(generated_bytes) != entry["bytes"] or sha(generated_bytes) != entry["sha256"]:
            raise RuntimeError(f"private generated script failed receipt check: {name}")

    replay = {
        "schema": "nf19.r12_transport_replay.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_LOCAL_ASSEMBLY_ONLY",
        "remote_actions_performed": False,
        "source_bindings": {
            "r11_original_assembler_sha256": sha(assembler_bytes),
            "r11_original_inline_sha256": sha(inline_bytes),
            "r11_base_spec_sha256": sha(base_spec_bytes),
            "r10_helper_pins": base_spec["r10_base_pins"],
            "r12_manifest_sha256": sha(manifest_bytes),
            "r12_review_sha256": sha(review_bytes),
            "r12_test_receipt_sha256": sha(tests_bytes),
            "r12_readme_sha256": sha(readme_bytes),
            "r12_import_smoke_sha256": sha(smoke_bytes),
        },
        "identity": {
            "unit": spec["unit"],
            "source": spec["destination"],
            "output": spec["output"],
            "sample_sha256": spec["sample_sha256"],
            "membership_sha256": spec["membership_sha256"],
        },
        "spec_path": str(spec_path.relative_to(REPO)),
        "spec_sha256": sha(spec_bytes),
        "inline_template_path": str(inline_output.relative_to(REPO)),
        "inline_template_sha256": sha(inline_output.read_bytes()),
        "private_assembler": {
            "path": str(private_assembler),
            "bytes": private_assembler.stat().st_size,
            "sha256": sha(private_assembler.read_bytes()),
        },
        "assembler_identity_patches": assembler_log,
        "inline_identity_patches": inline_log,
        "assembly_receipt_path": str(receipt_path.relative_to(REPO)),
        "assembly_receipt_sha256": sha(receipt_bytes),
        "generated_scripts_private": {
            name: {**entry, "path": str(PRIVATE / name)}
            for name, entry in receipt["generated_scripts"].items()
        },
        "assembly_literal_patch_log": receipt["literal_patch_log"],
    }
    replay_bytes = (json.dumps(replay, indent=2, sort_keys=True) + "\n").encode()
    write_new(PACKET / "REPLAY.json", replay_bytes)
    navigation = (
        "# R12 startup transport record\n\n"
        "The first local adapter based on R11 attempt-2 failed before producing an assembly receipt; its bytes remain in `replay_transport.py`, `root_inline_r12.py`, and the original `ASSEMBLY_SPEC.json`. This successful replay instead starts from the pinned R11 original v1 assembler and inline template, applying only occurrence-counted R11-to-R12 identity substitutions. The scientific source README is transported byte-for-byte.\n\n"
        "`REPLAY.json` records the pinned input hashes, literal substitutions, assembly receipt, and private generated script hashes. `LOCAL_IMPORT_SMOKE.json` records the bounded 26-module import and forbidden-read sentinel check. This local assembly performed no remote actions.\n"
    ).encode()
    write_new(PACKET / "README_ORIGINAL_V1.md", navigation)
    print(
        json.dumps(
            {
                "status": "PASS_LOCAL_ASSEMBLY_ONLY",
                "replay_sha256": sha(replay_bytes),
                "assembly_receipt_sha256": sha(receipt_bytes),
                "private_directory": str(PRIVATE),
                "generated_scripts_private": replay["generated_scripts_private"],
                "remote_actions_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
