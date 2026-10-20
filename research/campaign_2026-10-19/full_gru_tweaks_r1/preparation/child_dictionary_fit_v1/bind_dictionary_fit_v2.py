"""Guard and conditionally render the one dictionary-only child admission body."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import pprint
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
REPOSITORY = HERE.parents[4]
SOURCE_CHECKOUT = Path("/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019")
EMA_DIRECTORY = CAMPAIGN / "preparation/child_ema_fit_v1"
EMA_BODY = EMA_DIRECTORY / "REMOTE_SUBMISSION.py"
EMA_RECIPE = EMA_DIRECTORY / "EMA_RECIPE_BINDING_V1.json"
EMA_FINAL_BINDING = EMA_DIRECTORY / "EMA_FINAL_BINDING_RECEIPT_V1.json"
RECIPE = HERE / "DICTIONARY_RECIPE_BINDING_V1.json"
TEMPLATE = HERE / "DICTIONARY_ADMISSION_TEMPLATE_V1.json"
OUTPUT_BODY = HERE / "REMOTE_SUBMISSION.py"
EMA_ACCEPTANCE = CAMPAIGN / "records/child_ema_fit_v1/TERMINAL_ACCEPTANCE.json"
EMA_EXACT_STOP = CAMPAIGN / "records/child_ema_fit_v1/exact_stop/EXACT_STOP.json"

EXPECTED_RECIPE_BYTES = 28_944
EXPECTED_RECIPE_SHA256 = "fedd5a6fb651acabd081335422648d3549daade4aefeb5113bb5280eb34569a5"
EXPECTED_TEMPLATE_BODY_BYTES = 163_040
EXPECTED_TEMPLATE_BODY_SHA256 = "f83ac3120a2e445f1752fcd026123a0e8a21ba53594d27b398462a5236ffebd6"
EXPECTED_EMA_RECIPE_SHA256 = "7650cc81e432e76153c1f291defc9931e8d958e02e9d22acb6ed4d80c1273f72"
EXPECTED_EMA_FINAL_BINDING_SHA256 = (
    "4c9d5ca5e569660a62c91af53780af2e9c3e8c1aa8b2a9c808bfeda6fe47fc80"
)
EXPECTED_EMA_BINDER_SHA256 = "17be7d163966c354b961162b758ee7978d703196ed859006f8d96e64bb11c302"
EXPECTED_EMA_ARGV_SHA256 = "a20b4ede3b501daa6cd39e847c212ca3d6215dde81e130a8ab09a48004beb449"
EXPECTED_DICTIONARY_ARGV_SHA256 = "096b036c8af6220512e51b2002f3598729274b482d5db4b1df8099b26f91ab6d"
EXPECTED_SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
EXPECTED_SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
EXPECTED_EMA_RUN = "fg19_child_ema_f0_s1001_fit_v1"
EXPECTED_EMA_UNIT = "computer-vision-homework-fg19-child-ema-fit-v1.service"
EXPECTED_EMA_INVOCATION = "da5baf82d22b4f02890e184193b8caf7"
EXPECTED_DICTIONARY_RUN = "fg19_child_dictionary_f0_s1001_fit_v1"
EXPECTED_DICTIONARY_UNIT = "computer-vision-homework-fg19-child-dictionary-fit-v1.service"
EXPECTED_DICTIONARY_ACCEPTANCE_SHA256 = (
    "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f"
)
EXPECTED_DICTIONARY_STOP_SHA256 = "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158"
EXPECTED_DICTIONARY_SHA256 = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
EXPECTED_DICTIONARY_BYTES = 22_557
EXPECTED_CONTROL_ACCEPTANCE_SHA256 = (
    "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2"
)
EXPECTED_CONTROL_STOP_SHA256 = "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012"
EXPECTED_PARENT_ACCEPTANCE_SHA256 = (
    "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8"
)
EXPECTED_PARENT_STOP_SHA256 = "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8"
EXPECTED_PARENT_CHECKPOINT_SHA256 = (
    "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
)
EXPECTED_PROFILE_ACCEPTANCE_SHA256 = (
    "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045"
)
EXPECTED_PROFILE_STOP_SHA256 = "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce"
EXPECTED_PROFILE_INITIAL_STATE_SHA256 = (
    "254b29510a93a689ef3f259e65890fb6be76cfe112b483b15fd91763ea74c2cd"
)
EXPECTED_CACHE_ACCEPTANCE_SHA256 = (
    "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
)
EXPECTED_CACHE_STOP_SHA256 = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"
EXPECTED_EMA_ACCEPTANCE_SCHEMA = "fg19-child-ema-fit-terminal-acceptance.v1"
EXPECTED_EMA_FILES = frozenset(
    {
        "history.json",
        "model.pt",
        "model_ep0.pt",
        "model_ep1.pt",
        "model_ep2.pt",
        "model_ep3.pt",
        "model_ep4.pt",
        "model_ep5.pt",
        "provenance.json",
        "training_state.pt",
    }
)
EXPECTED_NORMALIZATION_IDENTITY = {
    "mean": {
        "bytes_sha256": "72def7c8dee31e2f469e49cb1c6059440669896d4bbded28d61f0cf35e4a41f2",
        "dtype": "torch.float32",
        "shape": [467],
    },
    "std": {
        "bytes_sha256": "3bf84643202ca8995f8797634d76e34292745914e902c234ee022f94f9258798",
        "dtype": "torch.float32",
        "shape": [467],
    },
}


def _sha256(raw: bytes) -> str:
    """Return the lowercase SHA-256 digest for exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def _read_pinned(path: Path, byte_count: int, digest: str, label: str) -> bytes:
    """Read one bounded local preparation artifact and require its frozen pin."""
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path:
        raise ValueError(f"frozen {label} is not a physical regular file: {path}")
    raw = path.read_bytes()
    if info.st_size != byte_count or len(raw) != byte_count or _sha256(raw) != digest:
        raise ValueError(f"frozen {label} changed: {path}")
    return raw


def _load_object(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must have an object root")
    return value


def _literal_assignment(source: str, name: str) -> tuple[ast.Assign, Any]:
    """Read one module-level literal assignment without executing the module."""
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return node, ast.literal_eval(node.value)
    raise ValueError(f"missing literal assignment {name}")


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def _replace_predecessor_links(source: str) -> str:
    """Patch admission and launch links through their distinct surrounding contexts."""
    source = _replace_once(
        source,
        '        "child_profile": profile,\n        "control_predecessor": control_predecessor,\n',
        '        "child_profile": profile,\n'
        '        "control_predecessor": control_predecessor,\n'
        '        "ema_predecessor": ema_predecessor,\n',
        "admission EMA predecessor link",
    )
    return _replace_once(
        source,
        '        "submission": submission,\n        "control_predecessor": control_predecessor,\n',
        '        "submission": submission,\n'
        '        "control_predecessor": control_predecessor,\n'
        '        "ema_predecessor": ema_predecessor,\n',
        "launch EMA predecessor link",
    )


def _check_synthetic_predecessor_link_patch() -> dict[str, object]:
    """Regression-check the two context-specific insertions without receipt data."""
    fixture = (
        "    admission = {\n"
        '        "child_profile": profile,\n'
        '        "control_predecessor": control_predecessor,\n'
        "    }\n"
        "    launch = {\n"
        '        "submission": submission,\n'
        '        "control_predecessor": control_predecessor,\n'
        "    }\n"
    )
    expected = (
        "    admission = {\n"
        '        "child_profile": profile,\n'
        '        "control_predecessor": control_predecessor,\n'
        '        "ema_predecessor": ema_predecessor,\n'
        "    }\n"
        "    launch = {\n"
        '        "submission": submission,\n'
        '        "control_predecessor": control_predecessor,\n'
        '        "ema_predecessor": ema_predecessor,\n'
        "    }\n"
    )
    observed = _replace_predecessor_links(fixture)
    if observed != expected:
        raise ValueError("synthetic admission/launch predecessor link patch differs")
    ast.parse("def _fixture():\n" + observed)
    return {
        "status": "PASS_SYNTHETIC_CONTEXT_PATCH_ONLY",
        "receipt_data_used": False,
        "executable_body_rendered": False,
        "admission_context_replaced": True,
        "launch_context_replaced": True,
    }


def _replace_scratch_contract(source: str) -> str:
    """Add the exact physical scratch path and project device/filesystem gate."""
    old_gate = (
        '    scratch_root = _physical(B / "research_tmp")\n'
        '    scratch = _run(["findmnt", "-T", str(scratch_root), "-n", "-o", "FSTYPE,SOURCE,TARGET"]).strip()\n'
        "    statvfs = os.statvfs(scratch_root)\n"
        "    free = statvfs.f_bavail * statvfs.f_frsize\n"
        '    if scratch.split()[0] in {"tmpfs", "ramfs"} or free < 25 * (1 << 30):\n'
        '        raise RuntimeError("physical scratch free space below 25 GiB")\n'
    )
    new_gate = (
        '    expected_scratch = B / "research_tmp"\n'
        '    if str(expected_scratch) != "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp":\n'
        '        raise RuntimeError("scratch path differs from the exact campaign research_tmp")\n'
        "    scratch_root = _physical(expected_scratch)\n"
        "    if scratch_root != expected_scratch or str(scratch_root) != str(expected_scratch) or scratch_root.is_symlink():\n"
        '        raise RuntimeError("scratch path is not the exact physical campaign research_tmp")\n'
        "    project_device = os.stat(B, follow_symlinks=False).st_dev\n"
        "    scratch_device = os.stat(scratch_root, follow_symlinks=False).st_dev\n"
        '    scratch = _run(["findmnt", "-T", str(scratch_root), "-n", "-o", "FSTYPE,SOURCE,TARGET"]).strip()\n'
        "    scratch_parts = scratch.split()\n"
        "    if (\n"
        "        project_device != 64512\n"
        "        or scratch_device != 64512\n"
        "        or scratch_device != project_device\n"
        "        or len(scratch_parts) != 3\n"
        '        or scratch_parts[0] != "ext4"\n'
        "    ):\n"
        '        raise RuntimeError("scratch filesystem/device differs from the accepted physical project volume")\n'
        "    statvfs = os.statvfs(scratch_root)\n"
        "    free = statvfs.f_bavail * statvfs.f_frsize\n"
        "    if free < 25 * (1 << 30):\n"
        '        raise RuntimeError("physical scratch free space below 25 GiB")\n'
    )
    source = _replace_once(source, old_gate, new_gate, "physical scratch path/device gate")
    return _replace_once(
        source,
        '        "scratch": scratch,\n        "scratch_free_bytes": free,\n',
        '        "scratch": scratch,\n'
        '        "scratch_identity": {\n'
        '            "path": str(scratch_root),\n'
        '            "filesystem": scratch_parts[0],\n'
        '            "source": scratch_parts[1],\n'
        '            "mount_target": scratch_parts[2],\n'
        '            "device": scratch_device,\n'
        '            "project_device": project_device,\n'
        "        },\n"
        '        "scratch_free_bytes": free,\n',
        "scratch identity admission receipt",
    )


def _check_synthetic_scratch_patch() -> dict[str, object]:
    """Regression-check the scratch gate rewrite without receipts or a body."""
    fixture = (
        "def _fixture(B, _physical, _run, os):\n"
        '    scratch_root = _physical(B / "research_tmp")\n'
        '    scratch = _run(["findmnt", "-T", str(scratch_root), "-n", "-o", "FSTYPE,SOURCE,TARGET"]).strip()\n'
        "    statvfs = os.statvfs(scratch_root)\n"
        "    free = statvfs.f_bavail * statvfs.f_frsize\n"
        '    if scratch.split()[0] in {"tmpfs", "ramfs"} or free < 25 * (1 << 30):\n'
        '        raise RuntimeError("physical scratch free space below 25 GiB")\n'
        "    admission = {\n"
        '        "scratch": scratch,\n'
        '        "scratch_free_bytes": free,\n'
        "    }\n"
    )
    observed = _replace_scratch_contract(fixture)
    ast.parse(observed)
    required = (
        'str(expected_scratch) != "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp"',
        "project_device != 64512",
        "scratch_device != 64512",
        'scratch_parts[0] != "ext4"',
        '"scratch_identity": {',
    )
    if any(text not in observed for text in required):
        raise ValueError("synthetic physical scratch gate patch is incomplete")
    return {
        "status": "PASS_SYNTHETIC_SCRATCH_GATE_PATCH_ONLY",
        "receipt_data_used": False,
        "executable_body_rendered": False,
        "exact_campaign_path_checked": True,
        "expected_filesystem": "ext4",
        "expected_project_device": 64512,
    }


def _replace_argv_assignment(source: str, argv: list[str]) -> str:
    node, _ = _literal_assignment(source, "TRAINER_ARGV")
    if node.end_lineno is None:
        raise ValueError("trainer argv assignment has no source span")
    lines = source.splitlines(keepends=True)
    replacement = "TRAINER_ARGV = " + pprint.pformat(argv, width=88, sort_dicts=False) + "\n"
    return "".join(lines[: node.lineno - 1]) + replacement + "".join(lines[node.end_lineno :])


def _assert_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _verify_mode_source(recipe: dict[str, Any]) -> dict[str, object]:
    """Verify pinned source text and recipe claims for the sparse dictionary mode."""
    feature = recipe.get("source_feature_semantics")
    if not isinstance(feature, dict):
        raise ValueError("dictionary source-feature semantics are missing")
    expected_semantics = {
        "representation_mode": "sparse",
        "accepted_cli_value": "sparse",
        "raw_width": 112,
        "code_width": 16,
        "top_k": 4,
        "appended_width": 256,
        "legacy_prefix_width": 211,
        "common_input_width": 467,
        "no_parameter_ema_flag": True,
        "no_ema_innovation_features": True,
    }
    if any(feature.get(key) != value for key, value in expected_semantics.items()):
        raise ValueError(
            "dictionary feature semantics differ from the frozen source-qualified recipe"
        )
    active_slot = feature.get("active_slot")
    inactive_slot = feature.get("inactive_slot")
    if not isinstance(active_slot, dict) or not isinstance(inactive_slot, dict):
        raise ValueError("sparse active/inactive feature slots are missing")
    if (
        active_slot.get("columns") != "appended suffix 0:128"
        or active_slot.get("code_columns") != "0:16"
        or active_slot.get("residual_columns") != "16:128"
        or inactive_slot.get("columns") != "appended suffix 128:256"
        or inactive_slot.get("value") != "exact zeros in sparse mode"
    ):
        raise ValueError("sparse code/residual/zero-slot layout differs from the frozen recipe")

    pins = feature.get("source_component_bindings")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("source component pins are missing")
    verified: dict[str, dict[str, object]] = {}
    for relative, pin in pins.items():
        if not isinstance(relative, str) or not isinstance(pin, dict):
            raise ValueError("malformed source component pin")
        path = SOURCE_CHECKOUT / relative
        raw = _read_pinned(path, pin.get("bytes"), pin.get("sha256"), relative)
        verified[relative] = {"bytes": len(raw), "sha256": _sha256(raw)}

    representation = (SOURCE_CHECKOUT / "src/wnn/training/input_representation.py").read_text(
        encoding="utf-8"
    )
    dictionary = (
        SOURCE_CHECKOUT / "src/wnn/training/input_representation_dictionary.py"
    ).read_text(encoding="utf-8")
    cli_features = (SOURCE_CHECKOUT / "src/wnn/training/cli_features.py").read_text(
        encoding="utf-8"
    )
    required = (
        (representation, 'if mode in ("sparse", "ema_sparse")'),
        (
            representation,
            "torch.argsort(code.abs(), dim=-1, descending=True, stable=True)[..., :TOP_K]",
        ),
        (representation, "residual = normalized - F.linear(code, fitted.decoder_weight)"),
        (representation, "output[..., :CODE_WIDTH].copy_(code)"),
        (representation, "output[..., CODE_WIDTH:SLOT_WIDTH].copy_(residual)"),
        (representation, "output[..., SLOT_WIDTH:].zero_()"),
        (dictionary, '"top_k": 4'),
        (dictionary, "stable signed top-four codes"),
        (cli_features, '"sparse"'),
    )
    if any(snippet not in text for text, snippet in required):
        raise ValueError("pinned source no longer spells the frozen sparse mode semantics")

    source = recipe.get("canonical_source")
    if not isinstance(source, dict) or (
        source.get("commit") != EXPECTED_SOURCE_COMMIT
        or source.get("source_file_count") != 167
        or source.get("source_manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("canonical source manifest identity differs from the frozen recipe")
    if SOURCE_CHECKOUT.is_symlink() or SOURCE_CHECKOUT.resolve(strict=True) != SOURCE_CHECKOUT:
        raise ValueError("canonical source checkout path is a symlink or path alias")
    revision = subprocess.run(
        ["git", "-C", str(SOURCE_CHECKOUT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if revision != EXPECTED_SOURCE_COMMIT:
        raise ValueError(f"canonical source checkout revision changed: {revision}")
    return {
        "commit": revision,
        "mode": "sparse",
        "code_width": 16,
        "stable_signed_top_k": 4,
        "residual_width": 112,
        "unused_suffix_columns": "128:256 exact zeros",
        "common_input_width": 467,
        "source_components": verified,
        "source_suite_or_profile_rerun": False,
    }


def _expected_ema_argv(ema_argv: list[str], dictionary_argv: list[str]) -> list[dict[str, object]]:
    """Require that only the run identity and representation mode change."""
    changed = [
        {"index": index, "from": old, "to": new}
        for index, (old, new) in enumerate(zip(ema_argv, dictionary_argv, strict=True))
        if old != new
    ]
    expected = [
        {"index": 1, "from": EXPECTED_EMA_RUN, "to": EXPECTED_DICTIONARY_RUN},
        {"index": 48, "from": "ema", "to": "sparse"},
    ]
    if changed != expected:
        raise ValueError(f"dictionary argv changed outside the two authorized values: {changed!r}")
    for flag, value in (
        ("--name", EXPECTED_DICTIONARY_RUN),
        ("--representation-mode", "sparse"),
        (
            "--representation-cache",
            "/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt",
        ),
        ("--init", "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3"),
        ("--mean-variance-objective", "wmse"),
        ("--mean-variance-head", "1"),
    ):
        try:
            index = dictionary_argv.index(flag)
        except ValueError as error:
            raise ValueError(f"dictionary argv is missing {flag}") from error
        if index + 1 >= len(dictionary_argv) or dictionary_argv[index + 1] != value:
            raise ValueError(f"dictionary argv has an unexpected value for {flag}")
    if (
        "--inherit-init-preprocessing" not in dictionary_argv
        or "--no-eval-during-fit" not in dictionary_argv
    ):
        raise ValueError("dictionary argv lost inherited normalization or no-evaluation gates")
    return changed


def _check_unique_patch_anchors(source: str) -> dict[str, int]:
    anchors = {
        "unit_assignment": 'UNIT = "computer-vision-homework-fg19-child-ema-fit-v1.service"',
        "main_receipt_gate": (
            "    profile_acceptance_raw, profile_stop_raw = _profile_receipt_bytes()\n"
            "    control_acceptance_raw, control_stop_raw = _control_receipt_bytes()\n"
            "    control_predecessor = _control_identity(control_acceptance_raw, control_stop_raw)\n"
            "    receipt_bytes, staged_pins = _decode_staged_receipts()\n"
            '    receipt_bytes.update({"CONTROL_TERMINAL_ACCEPTANCE.json": control_acceptance_raw, '
            '"CONTROL_EXACT_STOP.json": control_stop_raw})\n'
            '    staged_pins.update({"CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA256, '
            '"CONTROL_EXACT_STOP.json": CONTROL_EXACT_STOP_SHA256})\n'
        ),
        "admission_predecessor_link": (
            '        "child_profile": profile,\n'
            '        "control_predecessor": control_predecessor,\n'
        ),
        "launch_predecessor_link": (
            '        "submission": submission,\n'
            '        "control_predecessor": control_predecessor,\n'
        ),
        "physical_scratch_gate": (
            '    scratch_root = _physical(B / "research_tmp")\n'
            '    scratch = _run(["findmnt", "-T", str(scratch_root), "-n", "-o", "FSTYPE,SOURCE,TARGET"]).strip()\n'
            "    statvfs = os.statvfs(scratch_root)\n"
            "    free = statvfs.f_bavail * statvfs.f_frsize\n"
            '    if scratch.split()[0] in {"tmpfs", "ramfs"} or free < 25 * (1 << 30):\n'
            '        raise RuntimeError("physical scratch free space below 25 GiB")\n'
        ),
        "scratch_admission_identity": (
            '        "scratch": scratch,\n        "scratch_free_bytes": free,\n'
        ),
        "trainer_argv_digest": 'TRAINER_ARGV_SHA256 = "a20b4ede3b501daa6cd39e847c212ca3d6215dde81e130a8ab09a48004beb449"',
        "admission_mode": '            "representation_mode": "ema",\n',
        "admission_schema": '"schema": "fg19-child-ema-fit-admission.v1"',
        "launch_schema": '"schema": "fg19-child-ema-fit-launch.v1"',
        "ema_output_run_error": 'FileExistsError("EMA output/run path exists")',
        "unknown_launch_outcome_policy": '"never retry; inspect this exact unit and durable EMA output"',
    }
    counts = {label: source.count(anchor) for label, anchor in anchors.items()}
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"reviewed EMA body patch anchors are not unique: {counts!r}")
    _literal_assignment(source, "TRAINER_ARGV")
    _literal_assignment(source, "TRAINER_ARGV_SHA256")
    return counts


def _static_check() -> dict[str, object]:
    """Verify the frozen recipe/template and the conditional patch without B access."""
    recipe_raw = _read_pinned(
        RECIPE, EXPECTED_RECIPE_BYTES, EXPECTED_RECIPE_SHA256, "dictionary recipe"
    )
    recipe = _load_object(recipe_raw, "dictionary recipe")
    template_raw = TEMPLATE.read_bytes()
    template = _load_object(template_raw, "dictionary admission template")
    if (
        template.get("schema") != "fg19-child-dictionary-fit-admission-template.v1"
        or template.get("status") != "LOCAL_TEMPLATE_FROZEN__EMA_TERMINAL_RECEIPTS_UNBOUND__NO_BODY"
        or template.get("current_artifacts", {}).get("actual_ema_acceptance_sha256")
        != "UNBOUND_PENDING_ROOT"
        or template.get("current_artifacts", {}).get("actual_ema_exact_stop_sha256")
        != "UNBOUND_PENDING_ROOT"
        or template.get("current_artifacts", {}).get("body_rendered") is not False
    ):
        raise ValueError("dictionary template is no longer the frozen unbound local template")
    identity = recipe.get("identity")
    successor = template.get("successor")
    expected_identity = {
        "run_name": EXPECTED_DICTIONARY_RUN,
        "unit": EXPECTED_DICTIONARY_UNIT,
        "representation_mode": "sparse",
        "objective": "wmse",
        "private_evidence_path": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_child_dictionary_fit_v1",
        "run_artifact_path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/runs/fg19_child_dictionary_f0_s1001_fit_v1",
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError("frozen dictionary recipe identity changed")
    if not isinstance(successor, dict) or any(
        successor.get(key) != template_value
        for key, template_value in {
            "run_name": identity["run_name"],
            "unit": identity["unit"],
            "representation_mode": identity["representation_mode"],
            "objective": identity["objective"],
        }.items()
    ):
        raise ValueError("conditional template identity differs from the frozen recipe")
    expected_template_lineage = {
        "control_acceptance_sha256": EXPECTED_CONTROL_ACCEPTANCE_SHA256,
        "control_exact_stop_sha256": EXPECTED_CONTROL_STOP_SHA256,
        "parent_acceptance_sha256": EXPECTED_PARENT_ACCEPTANCE_SHA256,
        "parent_exact_stop_sha256": EXPECTED_PARENT_STOP_SHA256,
        "parent_checkpoint_sha256": EXPECTED_PARENT_CHECKPOINT_SHA256,
        "dictionary_acceptance_sha256": EXPECTED_DICTIONARY_ACCEPTANCE_SHA256,
        "dictionary_exact_stop_sha256": EXPECTED_DICTIONARY_STOP_SHA256,
        "dictionary_registered_sha256": EXPECTED_DICTIONARY_SHA256,
        "profile_acceptance_sha256": EXPECTED_PROFILE_ACCEPTANCE_SHA256,
        "profile_exact_stop_sha256": EXPECTED_PROFILE_STOP_SHA256,
        "t3_cache_acceptance_sha256": EXPECTED_CACHE_ACCEPTANCE_SHA256,
        "t3_cache_exact_stop_sha256": EXPECTED_CACHE_STOP_SHA256,
    }
    template_lineage = template.get("lineage_and_shared_gates")
    if not isinstance(template_lineage, dict) or any(
        template_lineage.get(key) != value for key, value in expected_template_lineage.items()
    ):
        raise ValueError("template changed frozen CONTROL/parent/dictionary/profile/cache lineage")
    predecessor = template.get("sequential_predecessor")
    if not isinstance(predecessor, dict) or any(
        predecessor.get(key) != value
        for key, value in {
            "acceptance_path": "records/child_ema_fit_v1/TERMINAL_ACCEPTANCE.json",
            "acceptance_sha256": "UNBOUND_PENDING_ROOT_ACTUAL_ACCEPTANCE",
            "exact_stop_path": "records/child_ema_fit_v1/exact_stop/STDOUT.raw.json",
            "exact_stop_sha256": "UNBOUND_PENDING_ROOT_ACTUAL_EXACT_STOP",
            "invocation_id": EXPECTED_EMA_INVOCATION,
            "required_status": "ACCEPTED_AND_EXACT_STOPPED",
            "run_name": EXPECTED_EMA_RUN,
            "unit": EXPECTED_EMA_UNIT,
        }.items()
    ):
        raise ValueError("EMA predecessor template is not explicitly unbound and fail closed")
    fit = recipe.get("fit")
    if not isinstance(fit, dict) or any(
        fit.get(key) != value
        for key, value in {
            "seed": 1001,
            "batch_size": 16,
            "chunk_rows": 2000,
            "epochs": 6,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch_zero_based": 5,
            "batch_order": "seeded",
        }.items()
    ):
        raise ValueError("dictionary fit fields differ from the fixed shared recipe")
    resource_contract = recipe.get("resource_contract")
    resource_template = template.get("resource_and_scope_contract")
    if not isinstance(resource_contract, dict) or not isinstance(resource_template, dict):
        raise ValueError("fixed resource contract or template resource gates are missing")
    expected_resource_values = {
        "gpu_uuid": "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944",
        "owned_gpu_workers": 1,
        "aggregate_cpu_cores": 2,
        "worker_cpu_quota": "100%",
        "worker_memory_high_bytes": 7_500_000_000,
        "worker_memory_max_bytes": 8_000_000_000,
        "worker_swap_max_bytes": 0,
        "cuda_allocator_bytes": 4_294_967_296,
        "cuda_free_reserve_bytes": 1_000_000_000,
        "parent_memory_target_bytes": 17_000_000_000,
        "parent_memory_high_bytes": 18_000_000_000,
        "parent_memory_max_bytes": 20_000_000_000,
        "parent_swap_max_bytes": 0,
        "physical_scratch_min_free_bytes": 26_843_545_600,
        "runtime_max_seconds": 86_400,
        "loader_workers": 0,
    }
    for key, expected in expected_resource_values.items():
        actual = (
            resource_template.get(key) if key in resource_template else resource_contract.get(key)
        )
        if actual != expected:
            raise ValueError(f"resource contract changed at {key}: {actual!r}")
    parent_envelope = resource_contract.get("parent_resource_envelope")
    if not isinstance(parent_envelope, dict) or any(
        parent_envelope.get(key) != value
        for key, value in {
            "cpu_quota": "200%",
            "memory_target_bytes": 17_000_000_000,
            "memory_high_bytes": 18_000_000_000,
            "memory_max_bytes": 20_000_000_000,
        }.items()
    ):
        raise ValueError("parent cgroup envelope differs from the frozen fit contract")
    thread_caps = resource_contract.get("thread_caps")
    if not isinstance(thread_caps, dict) or any(
        thread_caps.get(key) != value
        for key, value in {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "WNN_LOADER_RAM_GB": "1",
            "WNN_LOADER_WORKERS": "0",
            "WNN_THREADS": "1",
        }.items()
    ):
        raise ValueError("worker thread/loader caps differ from the fixed resource contract")
    if any(
        resource_template.get(key) is not expected
        for key, expected in {
            "all_populated_cgroups_must_be_empty": True,
            "foreign_processes_untouched": True,
            "fresh_output_and_run_dir_absence_required": True,
            "fresh_unit_not_found_required": True,
            "new_geometry_profile_required": False,
        }.items()
    ):
        raise ValueError("fresh ownership/output/unit/profile gates changed")
    dictionary_preflight = recipe.get("dictionary_cache_preflight")
    if (
        not isinstance(dictionary_preflight, dict)
        or any(
            dictionary_preflight.get(key) != value
            for key, value in {
                "path": "/home/grinchenkoeg/computer-vision-homework-20261018/data/artifacts/research/fg19_dictionary_f0_v1/input_representation.pt",
                "registered_bytes": EXPECTED_DICTIONARY_BYTES,
                "registered_sha256": EXPECTED_DICTIONARY_SHA256,
                "acceptance_sha256": EXPECTED_DICTIONARY_ACCEPTANCE_SHA256,
                "exact_stop_sha256": EXPECTED_DICTIONARY_STOP_SHA256,
            }.items()
        )
        or dictionary_preflight.get("missing_or_mismatch_action", "").lower().find("fail closed")
        < 0
    ):
        raise ValueError("fresh exact dictionary cache gate or fail-closed fallback policy changed")
    if _sha256(EMA_RECIPE.read_bytes()) != EXPECTED_EMA_RECIPE_SHA256:
        raise ValueError("frozen EMA recipe binding changed")
    _read_pinned(
        EMA_BODY,
        EXPECTED_TEMPLATE_BODY_BYTES,
        EXPECTED_TEMPLATE_BODY_SHA256,
        "reviewed EMA admission template body",
    )
    final_binding_raw = EMA_FINAL_BINDING.read_bytes()
    if _sha256(final_binding_raw) != EXPECTED_EMA_FINAL_BINDING_SHA256:
        raise ValueError("frozen EMA final binding receipt changed")
    final_binding = _load_object(final_binding_raw, "EMA final binding receipt")
    if (
        final_binding.get("schema") != "fg19-child-ema-fit-final-binding.v1"
        or final_binding.get("status")
        != "PASS_LOCAL_ACTUAL_CONTROL_RECEIPT_BINDING_AND_BODY_IDENTITY_ONLY"
        or final_binding.get("execution_boundary", {}).get("successor_launched") is not False
    ):
        raise ValueError("EMA final binding receipt does not match the reviewed local-only state")
    ema_successor = final_binding.get("successor")
    if not isinstance(ema_successor, dict) or any(
        ema_successor.get(key) != value
        for key, value in {
            "run_name": EXPECTED_EMA_RUN,
            "unit": EXPECTED_EMA_UNIT,
            "representation_mode": "ema",
            "trainer_argv_sha256": EXPECTED_EMA_ARGV_SHA256,
        }.items()
    ):
        raise ValueError("reviewed EMA final binding receipt identifies another successor")
    ema_binder = EMA_DIRECTORY / "bind_successor.py"
    binder_size = ema_binder.stat(follow_symlinks=False).st_size
    _read_pinned(ema_binder, binder_size, EXPECTED_EMA_BINDER_SHA256, "EMA local binder")

    ema_recipe = _load_object(EMA_RECIPE.read_bytes(), "EMA recipe")
    ema_argv = ema_recipe.get("trainer_argv")
    dictionary_argv = recipe.get("trainer_argv")
    if not isinstance(ema_argv, list) or not isinstance(dictionary_argv, list):
        raise ValueError("frozen EMA/dictionary canonical argv is missing")
    if (
        hashlib.sha256(json.dumps(ema_argv, separators=(",", ":")).encode()).hexdigest()
        != EXPECTED_EMA_ARGV_SHA256
    ):
        raise ValueError("EMA canonical argv digest changed")
    if (
        dictionary_argv != recipe.get("trainer_argv")
        or recipe.get("trainer_argv_sha256") != EXPECTED_DICTIONARY_ARGV_SHA256
        or hashlib.sha256(json.dumps(dictionary_argv, separators=(",", ":")).encode()).hexdigest()
        != EXPECTED_DICTIONARY_ARGV_SHA256
    ):
        raise ValueError("dictionary canonical argv digest differs from frozen recipe")
    delta = _expected_ema_argv(ema_argv, dictionary_argv)
    source_semantics = _verify_mode_source(recipe)

    original = EMA_BODY.read_text(encoding="utf-8")
    anchors = _check_unique_patch_anchors(original)
    predecessor_link_fixture = _check_synthetic_predecessor_link_patch()
    scratch_gate_fixture = _check_synthetic_scratch_patch()
    output_exists = os.path.lexists(OUTPUT_BODY)
    if output_exists:
        raise FileExistsError(
            f"conditional output already exists; refusing overwrite: {OUTPUT_BODY}"
        )
    if RECIPE.resolve(strict=True) != RECIPE or TEMPLATE.resolve(strict=True) != TEMPLATE:
        raise ValueError("template or recipe path resolves through an alias")
    return {
        "schema": "fg19-child-dictionary-local-template-check.v2",
        "status": "PASS_LOCAL_TEMPLATE_AND_FAIL_CLOSED_RENDERER_PREREQUISITES_V2",
        "renderer_repair": {
            "supersedes_path": str(HERE / "bind_dictionary_fit.py"),
            "supersedes_sha256": "67f1693bca3bab00de779c020cba66473b3343818b6e1ad3ef60593ff5e2ae10",
            "reason": "The prior renderer patched two distinct links through one non-unique line anchor.",
            "admission_and_launch_contexts_now_patched_separately": True,
            "synthetic_context_regression": "PASS_SYNTHETIC_CONTEXT_PATCH_ONLY",
            "scratch_gate": "exact campaign path, ext4, project device 64512, same device",
            "synthetic_scratch_regression": "PASS_SYNTHETIC_SCRATCH_GATE_PATCH_ONLY",
        },
        "recipe": {"path": str(RECIPE), "bytes": len(recipe_raw), "sha256": EXPECTED_RECIPE_SHA256},
        "template": {
            "path": str(TEMPLATE),
            "bytes": len(template_raw),
            "sha256": _sha256(template_raw),
        },
        "ema_base_body": {
            "path": str(EMA_BODY),
            "bytes": EXPECTED_TEMPLATE_BODY_BYTES,
            "sha256": EXPECTED_TEMPLATE_BODY_SHA256,
        },
        "ema_final_binding_receipt_sha256": EXPECTED_EMA_FINAL_BINDING_SHA256,
        "renderer": {
            "path": str(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size,
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
        "source_qualified_mode": source_semantics,
        "trainer_argv_sha256": EXPECTED_DICTIONARY_ARGV_SHA256,
        "trainer_argv_delta_from_ema": delta,
        "unique_patch_anchors": anchors,
        "predecessor_link_fixture": predecessor_link_fixture,
        "scratch_gate_fixture": scratch_gate_fixture,
        "actual_ema_acceptance_sha256": "UNBOUND_PENDING_ROOT_ACTUAL_ACCEPTANCE",
        "actual_ema_exact_stop_sha256": "UNBOUND_PENDING_ROOT_ACTUAL_EXACT_STOP",
        "actual_ema_exact_stop_path": str(EMA_EXACT_STOP),
        "body_rendered": False,
        "output_exists": output_exists,
        "b_staging_or_execution_performed": False,
    }


def _read_actual_receipt(
    path: Path, expected_parent: Path, expected_name: str, limit: int
) -> bytes:
    """Read one fixed local receipt as a physical bounded file, rejecting aliases."""
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError(f"receipt path is a symlink or alias: {path}")
    if (
        not path.is_relative_to(expected_parent)
        or path.parent != expected_parent
        or path.name != expected_name
    ):
        raise ValueError(f"receipt is outside its exact fixed location: {path}")
    info = path.stat(follow_symlinks=False)
    if not path.is_file() or info.st_size <= 0 or info.st_size > limit:
        raise ValueError(f"receipt is not a bounded regular file: {path}")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError(f"receipt changed while reading: {path}")
    return raw


def _find_counter(receipt: dict[str, Any], names: tuple[str, ...]) -> object:
    for candidate in (receipt, receipt.get("fit_counters"), receipt.get("fit")):
        if isinstance(candidate, dict):
            for name in names:
                if name in candidate:
                    return candidate[name]
    return None


def _validate_ema_receipts(
    acceptance_raw: bytes,
    stop_raw: bytes,
    expected_acceptance_sha256: str,
    expected_stop_sha256: str,
) -> dict[str, object]:
    """Require accepted EMA model/state/source/ancestry and its exact successful stop."""
    if _sha256(acceptance_raw) != expected_acceptance_sha256:
        raise ValueError("EMA acceptance bytes differ from the root-supplied digest")
    if _sha256(stop_raw) != expected_stop_sha256:
        raise ValueError("EMA exact-stop bytes differ from the root-supplied digest")
    if (
        not acceptance_raw
        or len(acceptance_raw) > 256_000
        or not stop_raw
        or len(stop_raw) > 64_000
    ):
        raise ValueError("EMA acceptance/stop is empty or exceeds the bounded metadata size")
    acceptance = _load_object(acceptance_raw, "EMA acceptance")
    stop = _load_object(stop_raw, "EMA exact stop")
    if (
        acceptance.get("schema") != EXPECTED_EMA_ACCEPTANCE_SCHEMA
        or acceptance.get("status") != "ACCEPTED"
        or acceptance.get("run_name") != EXPECTED_EMA_RUN
        or acceptance.get("unit") != EXPECTED_EMA_UNIT
        or acceptance.get("invocation_id") != EXPECTED_EMA_INVOCATION
    ):
        raise ValueError(
            "EMA acceptance schema/status/run/unit/invocation is not the frozen predecessor"
        )
    source = acceptance.get("source")
    nested_source = source if isinstance(source, dict) else {}
    source_identity = (
        acceptance.get("source_commit", nested_source.get("commit")),
        acceptance.get("source_file_count", nested_source.get("file_count")),
        acceptance.get("source_manifest_sha256", nested_source.get("manifest_sha256")),
    )
    if source_identity != (EXPECTED_SOURCE_COMMIT, 167, EXPECTED_SOURCE_MANIFEST_SHA256):
        raise ValueError("EMA acceptance source identity differs from the frozen canonical source")
    counts = {
        "successful_updates": _find_counter(
            acceptance,
            ("successful_updates", "optimizer_completed_updates", "observed_optimizer_updates"),
        ),
        "completed_batches": _find_counter(acceptance, ("completed_batches", "observed_batches")),
        "amp_skipped_updates": _find_counter(acceptance, ("amp_skipped_updates",)),
        "nonfinite_updates": _find_counter(
            acceptance, ("nonfinite_updates", "nonfinite_optimizer_updates")
        ),
        "next_epoch": _find_counter(acceptance, ("next_epoch",)),
        "selected_epoch": _find_counter(acceptance, ("selected_epoch",)),
    }
    if counts != {
        "successful_updates": 44_760,
        "completed_batches": 4_476,
        "amp_skipped_updates": 0,
        "nonfinite_updates": 0,
        "next_epoch": 6,
        "selected_epoch": 5,
    }:
        raise ValueError(
            f"EMA acceptance counters do not match the fixed six-epoch fit: {counts!r}"
        )
    if acceptance.get("epochs") != 6:
        raise ValueError("EMA acceptance does not identify exactly six completed fixed epochs")
    if (
        acceptance.get("strict_fp32_finite_model_optimizer_verified") is not True
        or acceptance.get("selected_model_epoch5_and_training_state_tensor_identity_verified")
        is not True
    ):
        raise ValueError("EMA acceptance lacks strict finite model/optimizer and saved-state proof")
    feature = acceptance.get("input_representation")
    if (
        not isinstance(feature, dict)
        or feature.get("mode") != "ema"
        or acceptance.get("representation_mode") != "ema"
    ):
        raise ValueError("EMA acceptance does not identify the EMA input representation")
    optimizer = acceptance.get("optimizer")
    normalization = acceptance.get("normalization_identity")
    rng_state = acceptance.get("rng_observed_after_restricted_load")
    sample_order = acceptance.get("sample_order_observations")
    if (
        not isinstance(optimizer, dict)
        or optimizer.get("step") != 44_760
        or not isinstance(normalization, dict)
        or normalization != EXPECTED_NORMALIZATION_IDENTITY
        or not isinstance(rng_state, dict)
        or not isinstance(rng_state.get("torch"), dict)
        or not isinstance(rng_state.get("cuda"), dict)
        or not isinstance(sample_order, list)
        or len(sample_order) != 6
        or [item.get("epoch") for item in sample_order if isinstance(item, dict)] != list(range(6))
        or any(
            not isinstance(item, dict)
            or item.get("rng_dtype") != "torch.uint8"
            or item.get("rng_shape") != [5056]
            or not _is_sha256_value(item.get("rng_sha256"))
            for item in sample_order
        )
    ):
        raise ValueError(
            "EMA optimizer, inherited normalization, restricted-load RNG, or sample-order evidence is incomplete"
        )
    model_fingerprint = _assert_sha(
        acceptance.get("model_tensor_fingerprint"), "EMA model fingerprint"
    )
    files = acceptance.get("files")
    if not isinstance(files, dict) or set(files) != EXPECTED_EMA_FILES:
        raise ValueError("EMA acceptance lacks the terminal file identity map")
    for filename, file_pin in files.items():
        if (
            not isinstance(filename, str)
            or not isinstance(file_pin, dict)
            or type(file_pin.get("bytes")) is not int
            or file_pin["bytes"] <= 0
            or not _is_sha256_value(file_pin.get("sha256"))
        ):
            raise ValueError("EMA acceptance contains a malformed terminal file pin")
    model_ep5 = files.get("model_ep5.pt")
    final_model = files.get("model.pt")
    state_file = files.get("training_state.pt")
    if (
        not isinstance(model_ep5, dict)
        or not isinstance(final_model, dict)
        or not isinstance(state_file, dict)
    ):
        raise ValueError("EMA acceptance lacks selected model or training-state file identities")
    model_sha = _assert_sha(model_ep5.get("sha256"), "EMA model_ep5 SHA-256")
    if (
        type(model_ep5.get("bytes")) is not int
        or model_ep5["bytes"] <= 0
        or not _is_sha256_value(final_model.get("sha256"))
        or type(final_model.get("bytes")) is not int
        or final_model["bytes"] <= 0
        or not _is_sha256_value(state_file.get("sha256"))
        or type(state_file.get("bytes")) is not int
        or state_file["bytes"] <= 0
    ):
        raise ValueError("EMA selected model and training-state file pins are inconsistent")
    if (
        acceptance.get("main_pid") != 0
        or acceptance.get("exec_main_code") != 1
        or acceptance.get("exec_main_status") != 0
    ):
        raise ValueError("EMA acceptance does not identify the successful exited native fit")
    if acceptance.get("quality_score") is not None:
        raise ValueError("EMA terminal acceptance must not introduce a quality score")
    ancestry = acceptance.get("predecessor_ancestry")
    if not isinstance(ancestry, dict):
        raise ValueError("EMA acceptance lacks source and fixed-predecessor ancestry")
    expected_ancestry = {
        "parent_acceptance_sha256": EXPECTED_PARENT_ACCEPTANCE_SHA256,
        "parent_exact_stop_sha256": EXPECTED_PARENT_STOP_SHA256,
        "parent_checkpoint_sha256": EXPECTED_PARENT_CHECKPOINT_SHA256,
        "dictionary_acceptance_sha256": EXPECTED_DICTIONARY_ACCEPTANCE_SHA256,
        "dictionary_exact_stop_sha256": EXPECTED_DICTIONARY_STOP_SHA256,
        "dictionary_sha256": EXPECTED_DICTIONARY_SHA256,
        "profile_acceptance_sha256": EXPECTED_PROFILE_ACCEPTANCE_SHA256,
        "profile_exact_stop_sha256": EXPECTED_PROFILE_STOP_SHA256,
        "profile_initial_state_sha256": EXPECTED_PROFILE_INITIAL_STATE_SHA256,
    }
    if any(ancestry.get(key) != value for key, value in expected_ancestry.items()):
        raise ValueError(
            "EMA acceptance ancestry differs from the frozen parent/dictionary/profile"
        )
    control = ancestry.get("control")
    if not isinstance(control, dict) or any(
        control.get(key) != value
        for key, value in {
            "acceptance_sha256": EXPECTED_CONTROL_ACCEPTANCE_SHA256,
            "exact_stop_sha256": EXPECTED_CONTROL_STOP_SHA256,
            "invocation_id": "e66e0d0703314e4fb5d47dcdbfcaf202",
            "selected_epoch": 5,
            "successful_updates": 44_760,
        }.items()
    ):
        raise ValueError("EMA acceptance does not preserve the accepted CONTROL ancestry")
    cache = acceptance.get("cache")
    if not isinstance(cache, dict) or any(
        cache.get(key) != value
        for key, value in {
            "acceptance_sha256": EXPECTED_CACHE_ACCEPTANCE_SHA256,
            "exact_stop_sha256": EXPECTED_CACHE_STOP_SHA256,
        }.items()
    ):
        raise ValueError(
            "EMA acceptance cache ancestry differs from the accepted homogeneous T3 cache"
        )

    invocation = acceptance.get("invocation_id")
    if (
        stop.get("unit") != EXPECTED_EMA_UNIT
        or stop.get("invocation_id") != invocation
        or stop.get("acceptance_sha256") != expected_acceptance_sha256
    ):
        raise ValueError("EMA exact stop does not bind the exact accepted bytes/unit/invocation")
    before, after = stop.get("before"), stop.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("EMA exact-stop before/after states are missing")
    if (
        before.get("InvocationID") != invocation
        or str(before.get("MainPID")) != "0"
        or str(before.get("ExecMainCode")) != "1"
        or str(before.get("ExecMainStatus")) != "0"
        or before.get("ActiveState") != "active"
        or before.get("SubState") != "exited"
        or after.get("LoadState") != "not-found"
        or after.get("ActiveState") != "inactive"
        or after.get("SubState") != "dead"
        or str(after.get("MainPID")) != "0"
        or after.get("InvocationID") != ""
        or stop.get("populated_before") != {}
        or stop.get("populated_after") != {}
    ):
        raise ValueError(
            "EMA unit was not the exact successful exited invocation and stopped afterward"
        )
    stop_result = stop.get("stop")
    if isinstance(stop_result, dict):
        if (
            stop_result.get("returncode") != 0
            or stop_result.get("stdout") != ""
            or stop_result.get("stderr") != ""
        ):
            raise ValueError("EMA exact-stop command did not return cleanly")
    elif stop.get("stop_returncode") != 0:
        raise ValueError("EMA exact-stop receipt lacks a successful stop command result")
    return {
        "acceptance_sha256": expected_acceptance_sha256,
        "exact_stop_sha256": expected_stop_sha256,
        "run_name": EXPECTED_EMA_RUN,
        "unit": EXPECTED_EMA_UNIT,
        "invocation_id": invocation,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "selected_epoch": 5,
        "successful_updates": 44_760,
        "completed_batches": 4_476,
        "model_ep5_sha256": model_sha,
        "model_tensor_fingerprint": model_fingerprint,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "source_file_count": 167,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "initialization_parent_checkpoint_sha256": EXPECTED_PARENT_CHECKPOINT_SHA256,
        "representation_mode": "ema",
    }


def _is_sha256_value(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _ema_gate_source(identity: dict[str, object], acceptance_raw: bytes, stop_raw: bytes) -> str:
    """Build literal EMA predecessor pins and runtime validation functions."""
    return f"""
EMA_PREDECESSOR_RUN = {EXPECTED_EMA_RUN!r}
EMA_PREDECESSOR_UNIT = {EXPECTED_EMA_UNIT!r}
EMA_PREDECESSOR_ACCEPTANCE_SHA256 = {identity["acceptance_sha256"]!r}
EMA_PREDECESSOR_EXACT_STOP_SHA256 = {identity["exact_stop_sha256"]!r}
EMA_PREDECESSOR_ACCEPTANCE_B64 = {base64.b64encode(acceptance_raw).decode("ascii")!r}
EMA_PREDECESSOR_EXACT_STOP_B64 = {base64.b64encode(stop_raw).decode("ascii")!r}


def _ema_predecessor_receipt_bytes() -> tuple[bytes, bytes]:
    acceptance_raw = base64.b64decode(EMA_PREDECESSOR_ACCEPTANCE_B64, validate=True)
    stop_raw = base64.b64decode(EMA_PREDECESSOR_EXACT_STOP_B64, validate=True)
    if (
        hashlib.sha256(acceptance_raw).hexdigest() != EMA_PREDECESSOR_ACCEPTANCE_SHA256
        or hashlib.sha256(stop_raw).hexdigest() != EMA_PREDECESSOR_EXACT_STOP_SHA256
    ):
        raise RuntimeError("bound EMA predecessor acceptance/stop bytes changed")
    return acceptance_raw, stop_raw


def _ema_predecessor_identity(acceptance_raw: bytes, stop_raw: bytes) -> dict[str, object]:
    receipt = json.loads(acceptance_raw)
    stop = json.loads(stop_raw)
    if (
        not isinstance(receipt, dict)
        or not isinstance(stop, dict)
        or receipt.get("schema") != {EXPECTED_EMA_ACCEPTANCE_SCHEMA!r}
        or receipt.get("status") != "ACCEPTED"
        or receipt.get("run_name") != EMA_PREDECESSOR_RUN
        or receipt.get("unit") != EMA_PREDECESSOR_UNIT
        or receipt.get("invocation_id") != {EXPECTED_EMA_INVOCATION!r}
        or receipt.get("successful_updates") != 44760
        or receipt.get("completed_batches") != 4476
        or receipt.get("amp_skipped_updates") != 0
        or receipt.get("nonfinite_updates") != 0
        or receipt.get("next_epoch") != 6
        or receipt.get("selected_epoch") != 5
        or receipt.get("epochs") != 6
        or receipt.get("strict_fp32_finite_model_optimizer_verified") is not True
        or receipt.get("selected_model_epoch5_and_training_state_tensor_identity_verified") is not True
        or receipt.get("main_pid") != 0
        or receipt.get("exec_main_code") != 1
        or receipt.get("exec_main_status") != 0
        or receipt.get("quality_score") is not None
        or receipt.get("source_commit") != {EXPECTED_SOURCE_COMMIT!r}
        or receipt.get("source_file_count") != 167
        or receipt.get("source_manifest_sha256") != {EXPECTED_SOURCE_MANIFEST_SHA256!r}
        or receipt.get("representation_mode") != "ema"
    ):
        raise RuntimeError("EMA predecessor lacks accepted frozen source/model/state identity")
    optimizer = receipt.get("optimizer")
    normalization = receipt.get("normalization_identity")
    rng_state = receipt.get("rng_observed_after_restricted_load")
    sample_order = receipt.get("sample_order_observations")
    if (
        not isinstance(optimizer, dict)
        or optimizer.get("step") != 44760
        or normalization != {EXPECTED_NORMALIZATION_IDENTITY!r}
        or not isinstance(rng_state, dict)
        or not isinstance(rng_state.get("torch"), dict)
        or not isinstance(rng_state.get("cuda"), dict)
        or not isinstance(sample_order, list)
        or len(sample_order) != 6
        or [item.get("epoch") for item in sample_order if isinstance(item, dict)] != list(range(6))
        or any(
            not isinstance(item, dict)
            or item.get("rng_dtype") != "torch.uint8"
            or item.get("rng_shape") != [5056]
            or not _is_sha256(item.get("rng_sha256"))
            for item in sample_order
        )
    ):
        raise RuntimeError("EMA predecessor optimizer, normalization, RNG, or sample-order evidence is incomplete")
    invocation = receipt.get("invocation_id")
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != {EXPECTED_EMA_FILES!r}:
        raise RuntimeError("EMA predecessor selected model/state files are missing")
    if any(
        not isinstance(name, str)
        or not isinstance(pin, dict)
        or type(pin.get("bytes")) is not int
        or pin["bytes"] <= 0
        or not _is_sha256(pin.get("sha256"))
        for name, pin in files.items()
    ):
        raise RuntimeError("EMA predecessor terminal file map is malformed")
    model = files.get("model_ep5.pt")
    final_model = files.get("model.pt")
    state_file = files.get("training_state.pt")
    if (
        not isinstance(model, dict)
        or not _is_sha256(model.get("sha256"))
        or type(model.get("bytes")) is not int
        or model["bytes"] <= 0
        or not isinstance(final_model, dict)
        or not _is_sha256(final_model.get("sha256"))
        or type(final_model.get("bytes")) is not int
        or final_model["bytes"] <= 0
        or not isinstance(state_file, dict)
        or not _is_sha256(state_file.get("sha256"))
        or type(state_file.get("bytes")) is not int
        or state_file["bytes"] <= 0
    ):
        raise RuntimeError("EMA predecessor model_ep5 SHA-256 is missing")
    ancestry = receipt.get("predecessor_ancestry")
    expected_ancestry = {dict(identity["expected_ancestry"])}
    if not isinstance(ancestry, dict) or any(ancestry.get(k) != v for k, v in expected_ancestry.items()):
        raise RuntimeError("EMA predecessor ancestry differs from the fixed CONTROL/parent/dictionary/profile")
    control = ancestry.get("control")
    expected_control = {dict(identity["expected_control_ancestry"])}
    if not isinstance(control, dict) or any(control.get(k) != v for k, v in expected_control.items()):
        raise RuntimeError("EMA predecessor lost the accepted CONTROL lineage")
    cache = receipt.get("cache")
    expected_cache = {dict(identity["expected_cache"])}
    if not isinstance(cache, dict) or any(cache.get(k) != v for k, v in expected_cache.items()):
        raise RuntimeError("EMA predecessor no longer binds the accepted homogeneous cache")
    feature = receipt.get("input_representation")
    if not isinstance(feature, dict) or feature.get("mode") != "ema":
        raise RuntimeError("EMA predecessor acceptance does not identify the EMA input representation")
    if not _is_sha256(receipt.get("model_tensor_fingerprint")):
        raise RuntimeError("EMA predecessor model tensor fingerprint is missing")
    before, after = stop.get("before"), stop.get("after")
    if (
        stop.get("unit") != EMA_PREDECESSOR_UNIT
        or stop.get("invocation_id") != invocation
        or stop.get("acceptance_sha256") != EMA_PREDECESSOR_ACCEPTANCE_SHA256
        or not isinstance(before, dict)
        or not isinstance(after, dict)
        or before.get("InvocationID") != invocation
        or str(before.get("MainPID")) != "0"
        or str(before.get("ExecMainCode")) != "1"
        or str(before.get("ExecMainStatus")) != "0"
        or before.get("ActiveState") != "active"
        or before.get("SubState") != "exited"
        or after.get("LoadState") != "not-found"
        or after.get("ActiveState") != "inactive"
        or after.get("SubState") != "dead"
        or str(after.get("MainPID")) != "0"
        or after.get("InvocationID") != ""
        or stop.get("populated_before") != {{}}
        or stop.get("populated_after") != {{}}
    ):
        raise RuntimeError("EMA predecessor exact-stop receipt does not match the accepted invocation")
    stop_result = stop.get("stop")
    if isinstance(stop_result, dict):
        if (
            stop_result.get("returncode") != 0
            or stop_result.get("stdout") != ""
            or stop_result.get("stderr") != ""
        ):
            raise RuntimeError("EMA predecessor exact-stop command did not return cleanly")
    elif stop.get("stop_returncode") != 0:
        raise RuntimeError("EMA predecessor exact-stop receipt lacks successful command status")
    return {{
        "acceptance_sha256": EMA_PREDECESSOR_ACCEPTANCE_SHA256,
        "exact_stop_sha256": EMA_PREDECESSOR_EXACT_STOP_SHA256,
        "run_name": EMA_PREDECESSOR_RUN,
        "unit": EMA_PREDECESSOR_UNIT,
        "invocation_id": invocation,
        "status": "ACCEPTED_AND_EXACT_STOPPED",
        "selected_epoch": 5,
        "successful_updates": 44760,
        "model_ep5_sha256": model["sha256"],
    }}
"""


def _render_body(acceptance_raw: bytes, stop_raw: bytes, acceptance_sha: str, stop_sha: str) -> str:
    """Render the reviewed wrapper only after exact accepted EMA receipts validate."""
    recipe = _load_object(RECIPE.read_bytes(), "dictionary recipe")
    identity = _validate_ema_receipts(acceptance_raw, stop_raw, acceptance_sha, stop_sha)
    inputs = recipe["accepted_inputs"]
    parent = inputs["parent"]
    dictionary = inputs["shared_dictionary"]
    profile = inputs["child_shape_profile"]
    cache = inputs["homogeneous_t3_cache"]
    control = recipe["control_predecessor_gate"]
    expected_ancestry = {
        "parent_acceptance_sha256": parent["acceptance_sha256"],
        "parent_exact_stop_sha256": parent["exact_stop_sha256"],
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "dictionary_acceptance_sha256": dictionary["acceptance_sha256"],
        "dictionary_exact_stop_sha256": dictionary["exact_stop_sha256"],
        "dictionary_sha256": dictionary["sha256"],
        "profile_acceptance_sha256": profile["acceptance_sha256"],
        "profile_exact_stop_sha256": profile["exact_stop_sha256"],
        "profile_initial_state_sha256": EXPECTED_PROFILE_INITIAL_STATE_SHA256,
    }
    expected_control = {
        "acceptance_sha256": control["acceptance_sha256"],
        "exact_stop_sha256": control["exact_stop_sha256"],
        "invocation_id": control["invocation_id"],
        "selected_epoch": 5,
        "successful_updates": 44_760,
    }
    expected_cache = {
        "acceptance_sha256": cache["acceptance_sha256"],
        "exact_stop_sha256": cache["exact_stop_sha256"],
    }
    source = EMA_BODY.read_text(encoding="utf-8")
    body = _replace_argv_assignment(source, recipe["trainer_argv"])
    body = _replace_once(
        body,
        '"""Fail-closed admission template for the F0 seed-1001 EMA fit."""',
        '"""Fail-closed admission template for the F0 seed-1001 dictionary fit."""',
        "module description",
    )
    body = _replace_once(
        body,
        'OUTPUT = B / "research_tmp/fg19_child_ema_fit_v1"',
        'OUTPUT = B / "research_tmp/fg19_child_dictionary_fit_v1"',
        "private evidence path",
    )
    body = _replace_once(
        body, f'RUN = "{EXPECTED_EMA_RUN}"', f'RUN = "{EXPECTED_DICTIONARY_RUN}"', "run identity"
    )
    body = _replace_once(
        body,
        f'UNIT = "{EXPECTED_EMA_UNIT}"',
        f'UNIT = "{EXPECTED_DICTIONARY_UNIT}"',
        "unit identity",
    )
    body = _replace_once(
        body,
        f'TRAINER_ARGV_SHA256 = "{EXPECTED_EMA_ARGV_SHA256}"',
        f'TRAINER_ARGV_SHA256 = "{EXPECTED_DICTIONARY_ARGV_SHA256}"',
        "trainer argv hash",
    )
    body = _replace_once(
        body,
        '"schema": "fg19-child-ema-fit-admission.v1"',
        '"schema": "fg19-child-dictionary-fit-admission.v1"',
        "admission schema",
    )
    body = _replace_once(
        body,
        '"schema": "fg19-child-ema-fit-launch.v1"',
        '"schema": "fg19-child-dictionary-fit-launch.v1"',
        "launch schema",
    )
    body = _replace_once(
        body,
        '"representation_mode": "ema"',
        '"representation_mode": "sparse"',
        "admission representation mode",
    )
    body = body.replace("EMA fit", "dictionary fit")
    body = body.replace("EMA unit", "dictionary unit").replace(
        "EMA destination", "dictionary destination"
    )
    body = body.replace("EMA canonical trainer", "dictionary canonical trainer")
    body = body.replace("EMA staged receipt", "child staged receipt")
    body = _replace_once(
        body,
        'FileExistsError("EMA output/run path exists")',
        'FileExistsError("dictionary output/run path exists")',
        "dictionary output/run error label",
    )
    body = _replace_once(
        body,
        f'UNIT = "{EXPECTED_DICTIONARY_UNIT}"\n',
        f'UNIT = "{EXPECTED_DICTIONARY_UNIT}"\n'
        + _ema_gate_source(
            {
                **identity,
                "expected_ancestry": expected_ancestry,
                "expected_control_ancestry": expected_control,
                "expected_cache": expected_cache,
            },
            acceptance_raw,
            stop_raw,
        ),
        "EMA predecessor constants and runtime validator",
    )

    old_gate = (
        "    profile_acceptance_raw, profile_stop_raw = _profile_receipt_bytes()\n"
        "    control_acceptance_raw, control_stop_raw = _control_receipt_bytes()\n"
        "    control_predecessor = _control_identity(control_acceptance_raw, control_stop_raw)\n"
        "    receipt_bytes, staged_pins = _decode_staged_receipts()\n"
        '    receipt_bytes.update({"CONTROL_TERMINAL_ACCEPTANCE.json": control_acceptance_raw, "CONTROL_EXACT_STOP.json": control_stop_raw})\n'
        '    staged_pins.update({"CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA256, "CONTROL_EXACT_STOP.json": CONTROL_EXACT_STOP_SHA256})\n'
    )
    new_gate = (
        "    profile_acceptance_raw, profile_stop_raw = _profile_receipt_bytes()\n"
        "    control_acceptance_raw, control_stop_raw = _control_receipt_bytes()\n"
        "    control_predecessor = _control_identity(control_acceptance_raw, control_stop_raw)\n"
        "    ema_acceptance_raw, ema_stop_raw = _ema_predecessor_receipt_bytes()\n"
        "    ema_predecessor = _ema_predecessor_identity(ema_acceptance_raw, ema_stop_raw)\n"
        "    receipt_bytes, staged_pins = _decode_staged_receipts()\n"
        '    receipt_bytes.update({"CONTROL_TERMINAL_ACCEPTANCE.json": control_acceptance_raw, "CONTROL_EXACT_STOP.json": control_stop_raw, "EMA_TERMINAL_ACCEPTANCE.json": ema_acceptance_raw, "EMA_EXACT_STOP.json": ema_stop_raw})\n'
        '    staged_pins.update({"CONTROL_TERMINAL_ACCEPTANCE.json": CONTROL_ACCEPTANCE_SHA256, "CONTROL_EXACT_STOP.json": CONTROL_EXACT_STOP_SHA256, "EMA_TERMINAL_ACCEPTANCE.json": EMA_PREDECESSOR_ACCEPTANCE_SHA256, "EMA_EXACT_STOP.json": EMA_PREDECESSOR_EXACT_STOP_SHA256})\n'
    )
    body = _replace_once(body, old_gate, new_gate, "successor predecessor receipt gate")
    body = _replace_predecessor_links(body)
    body = _replace_scratch_contract(body)
    body = _replace_once(
        body,
        '"never retry; inspect this exact unit and durable EMA output"',
        '"never retry; inspect this exact unit and durable dictionary output"',
        "unknown launch outcome policy",
    )
    ast.parse(body)
    return body


def render(acceptance_sha256: str, exact_stop_sha256: str) -> dict[str, object]:
    """Validate actual local EMA receipts, then create the body once with O_EXCL."""
    if re.fullmatch(r"[0-9a-f]{64}", acceptance_sha256) is None:
        raise ValueError("EMA acceptance SHA is unbound or malformed")
    if re.fullmatch(r"[0-9a-f]{64}", exact_stop_sha256) is None:
        raise ValueError("EMA exact-stop SHA is unbound or malformed")
    _static_check()
    if os.path.lexists(OUTPUT_BODY):
        raise FileExistsError(f"refusing to overwrite conditional output: {OUTPUT_BODY}")
    records = CAMPAIGN / "records/child_ema_fit_v1"
    acceptance_raw = _read_actual_receipt(
        EMA_ACCEPTANCE, records, "TERMINAL_ACCEPTANCE.json", 256_000
    )
    stop_raw = _read_actual_receipt(
        EMA_EXACT_STOP, records / "exact_stop", "EXACT_STOP.json", 64_000
    )
    identity = _validate_ema_receipts(
        acceptance_raw, stop_raw, acceptance_sha256, exact_stop_sha256
    )
    body = _render_body(acceptance_raw, stop_raw, acceptance_sha256, exact_stop_sha256)
    body_raw = body.encode("utf-8")
    ast.parse(body)
    descriptor = os.open(OUTPUT_BODY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body_raw)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(HERE, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        OUTPUT_BODY.unlink(missing_ok=True)
        raise
    return {
        "status": "PASS_LOCAL_BODY_RENDERED_AFTER_EMA_RECEIPT_VALIDATION",
        "body_path": str(OUTPUT_BODY),
        "body_bytes": len(body_raw),
        "body_sha256": _sha256(body_raw),
        "ema_predecessor": identity,
        "b_staging_or_execution_performed": False,
    }


def main() -> int:
    """Run local checks by default; render only with explicit receipt SHA arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify local frozen template and no-body state"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="conditionally render after actual EMA acceptance and stop validate",
    )
    parser.add_argument("--ema-acceptance-sha256")
    parser.add_argument("--ema-exact-stop-sha256")
    args = parser.parse_args()
    if args.render:
        if not args.ema_acceptance_sha256 or not args.ema_exact_stop_sha256:
            parser.error("--render requires both actual EMA receipt SHA-256 values")
        result = render(args.ema_acceptance_sha256, args.ema_exact_stop_sha256)
    else:
        if args.ema_acceptance_sha256 or args.ema_exact_stop_sha256:
            parser.error("receipt hashes are accepted only with explicit --render")
        result = _static_check()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
