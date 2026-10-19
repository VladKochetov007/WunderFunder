from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

REPO = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
CAMPAIGN = REPO / "research/campaign_2026-10-19"
R10 = CAMPAIGN / "idea_screens_r10"
R11 = CAMPAIGN / "idea_screens_r11"
PACKET = R11 / "records/startup_v1"
PRIVATE = Path("/Users/nb/Documents/wunder/runtime-20261018/eda19-r11-attempt2")
SPEC_PATH = PACKET / "ASSEMBLY_SPEC_ATTEMPT2.json"

R10_BASE_PINS = {
    "remote_body_v1.py": "e3b3b13217bda561fc8e638ce693226ae9ed944d791b443480fc6b074afa1fb0",
    "capture_remote_v2.py": "7f7e7576380d9796ace256d70cbc6333036f4758fb8ce2bd0507395707c18f00",
    "stop_remote_v1.py": "0664e2544ad0dc4bbaa5d2b701bdb31706dd3831752e955e8a844ec9d75828ca",
}
EXTRAS = {
    "idea_screens_r11/PREPARATION_MANIFEST.json",
    "idea_screens_r11/IMPLEMENTATION_REVIEW.json",
    "idea_screens_r11/TEST_RECEIPT.json",
    "idea_screens_r11/README.md",
}
MAX_FILE_BYTES = 16_000_000
MAX_BUNDLE_BYTES = 64_000_000
EXPECTED_RUNTIME_ROOTS = {
    "idea_screens_r1",
    "idea_screens_r2",
    "idea_screens_r3",
    "idea_screens_r7",
    "idea_screens_r11",
}
EXPECTED_R7_SOURCES = {
    "idea_screens_r7/__init__.py",
    "idea_screens_r7/model_math.py",
    "idea_screens_r7/panel.py",
    "idea_screens_r7/runtime_support.py",
}
PLACEHOLDERS = (
    "__MANIFEST_SHA__",
    "__CANONICAL_PINS_REPR__",
    "__CANONICAL_PIN_REPR__",
    "__MODULES_REPR__",
    "__EXPECTED_RUNTIME_REPR__",
    "__ALLOWED_ROOTS_REPR__",
    "__R7_ALLOWLIST_REPR__",
    "__INLINE_REPR__",
)


def digest(payload: bytes) -> str:
    """Return the SHA-256 identity of exact transport bytes."""
    return hashlib.sha256(payload).hexdigest()


def read_pinned(path: Path, expected_sha256: str) -> bytes:
    """Read one local template only after checking its frozen byte identity."""
    payload = path.read_bytes()
    if digest(payload) != expected_sha256:
        raise RuntimeError(f"frozen R10 helper changed: {path}")
    return payload


def replace_exact(
    text: str, old: str, new: str, count: int, patch_id: str, log: list[dict[str, Any]]
) -> str:
    """Apply one literal patch only when its source occurrence count is exact."""
    observed = text.count(old)
    if observed != count:
        raise RuntimeError(
            f"patch {patch_id!r} expected {count} source occurrences, found {observed}"
        )
    log.append(
        {
            "id": patch_id,
            "occurrences": count,
            "source_sha256": digest(old.encode()),
            "replacement_sha256": digest(new.encode()),
        }
    )
    return text.replace(old, new)


def checked_relpath(name: str) -> PurePosixPath:
    """Reject absolute, noncanonical, or parent-traversing bundle paths."""
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or path.as_posix() != name:
        raise RuntimeError(f"unsafe sealed bundle path: {name}")
    return path


def validate_spec(spec: dict[str, Any]) -> None:
    """Require the recorded R11 identity and resource envelope to match the reviewed assignment."""
    identity = {
        "unit": "computer-vision-homework-eda19-r11-v2.service",
        "destination": "support-source/eda19-r11-v2",
        "output": "research_tmp/nf19-r11-v2",
        "sample": "research_tmp/nf19-sample-r3-v2",
        "sample_sha256": "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64",
        "membership_sha256": "e809d4810eeea7d2065ea8f2153b53ece3bde285f7900ce3d49e468e18045217",
        "source_snapshot": "a266b036",
    }
    if any(spec.get(key) != value for key, value in identity.items()):
        raise RuntimeError("R11 startup identity differs from the reviewed fixed assignment")
    if set(spec.get("allowed_bundle_prefixes", [])) != EXPECTED_RUNTIME_ROOTS:
        raise RuntimeError("R11 allowed bundle roots differ from the reviewed source closure")
    if set(spec.get("r7_source_allowlist", [])) != EXPECTED_R7_SOURCES:
        raise RuntimeError("R11 R7 source allowlist differs from the four reviewed primitives")
    if set(spec.get("extra_transport_paths", [])) != EXTRAS:
        raise RuntimeError("R11 extras differ from the four reviewed metadata paths")
    limits = spec.get("limits", {})
    expected_limits = {
        "per_file_bytes": MAX_FILE_BYTES,
        "bundle_bytes": MAX_BUNDLE_BYTES,
        "parent_cpu_cores": 2,
        "parent_memory_high_bytes": 18_000_000_000,
        "parent_memory_max_bytes": 20_000_000_000,
        "parent_memory_swap_bytes": 0,
        "parent_memory_reservation_target_bytes": 17_000_000_000,
        "job_cpu_cores": 0.5,
        "job_memory_high_bytes": 1_400_000_000,
        "job_memory_max_bytes": 1_500_000_000,
        "job_memory_swap_bytes": 0,
        "job_runtime_seconds": 600,
    }
    if any(limits.get(key) != value for key, value in expected_limits.items()):
        raise RuntimeError("R11 resource envelope differs from the reviewed R10-derived caps")


def manifest_and_transport(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], bytes, dict[str, dict[str, Any]], dict[str, str]]:
    """Validate frozen closure and build a byte-exact, bounded source bundle."""
    manifest_path = R11 / "PREPARATION_MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("schema") != "nf19.r11_preparation.v1":
        raise RuntimeError("R11 preparation manifest is missing or has the wrong schema")
    file_pins = manifest.get("files")
    runtime_py = manifest.get("runtime_py")
    if not isinstance(file_pins, dict) or not isinstance(runtime_py, dict):
        raise RuntimeError("R11 manifest files/runtime_py must be objects")
    if runtime_py != spec["runtime_py"]:
        raise RuntimeError("R11 runtime_py differs from the frozen startup closure")

    canonical_names = {
        name for name in file_pins if isinstance(name, str) and name.startswith("canonical:")
    }
    if canonical_names != set(spec["canonical_pin_names"]):
        raise RuntimeError("R11 canonical semantic pin names differ from the frozen three-file set")

    allowed_roots = set(spec["allowed_bundle_prefixes"])
    r7_allowlist = set(spec["r7_source_allowlist"])
    noncanonical: set[str] = set()
    canonical_pins: dict[str, str] = {}
    for name, expected_sha256 in file_pins.items():
        if not isinstance(name, str) or not isinstance(expected_sha256, str):
            raise RuntimeError("R11 manifest pins must map string paths to SHA-256 strings")
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise RuntimeError(f"invalid R11 manifest SHA-256: {name}")
        path_name = name.removeprefix("canonical:") if name.startswith("canonical:") else name
        path = checked_relpath(path_name)
        if name.startswith("canonical:"):
            if path.parts[0] != "wnn":
                raise RuntimeError(f"canonical pin escapes the WNN namespace: {name}")
            canonical_pins[name] = expected_sha256
            continue
        if path.parts[0] not in allowed_roots:
            raise RuntimeError(f"manifest path escapes frozen bundle roots: {name}")
        if path.parts[0] == "idea_screens_r7" and name not in r7_allowlist:
            raise RuntimeError(f"R7 transport exceeds the four-file runtime allowlist: {name}")
        noncanonical.add(name)
    if set(canonical_pins) != set(spec["canonical_pin_names"]):
        raise RuntimeError("R11 canonical pins differ from the exact startup pin set")
    for module_name, relative in spec["runtime_py"].items():
        if relative.startswith("canonical:"):
            if relative not in canonical_pins:
                raise RuntimeError(f"runtime canonical module lacks a frozen pin: {module_name}")
        elif relative not in file_pins:
            raise RuntimeError(f"runtime source is absent from the manifest pins: {relative}")

    expected_files = noncanonical | EXTRAS
    files: dict[str, dict[str, Any]] = {}
    total = 0
    for name in sorted(expected_files):
        path = checked_relpath(name)
        source = CAMPAIGN / Path(*path.parts)
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"required R11 source is missing or is a symlink: {name}")
        payload = source.read_bytes()
        expected_sha256 = file_pins.get(name)
        if expected_sha256 is not None and digest(payload) != expected_sha256:
            raise RuntimeError(f"local R11 source differs from its manifest pin: {name}")
        if len(payload) > MAX_FILE_BYTES:
            raise RuntimeError(f"R11 source exceeds the fixed per-file limit: {name}")
        total += len(payload)
        if total > MAX_BUNDLE_BYTES:
            raise RuntimeError("R11 source bundle exceeds the fixed 64 MB staging bound")
        files[name] = {
            "bytes": len(payload),
            "sha256": digest(payload),
            "base64": base64.b64encode(payload).decode("ascii"),
        }
    return manifest, manifest_bytes, files, canonical_pins


def replace_block(
    text: str,
    start: str,
    end: str,
    replacement: str,
    patch_id: str,
    log: list[dict[str, Any]],
) -> str:
    """Replace one uniquely delimited source block and record its identities."""
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"patch {patch_id!r} has missing or repeated block markers")
    first = text.index(start)
    last = text.index(end, first)
    old = text[first:last]
    log.append(
        {
            "id": patch_id,
            "occurrences": 1,
            "source_sha256": digest(old.encode()),
            "replacement_sha256": digest(replacement.encode()),
        }
    )
    return text[:first] + replacement + text[last:]


def patch_remote_body(
    source: str,
    spec: dict[str, Any],
    canonical_pins: dict[str, str],
    manifest_sha256: str,
    inline: str,
    log: list[dict[str, Any]],
) -> str:
    """Bind R11 identity, exact closure and canonical pins into the R10 body."""
    source = replace_exact(source, "r10-v1", "r10-v2", 3, "attempt2-body-identities", log)
    for old, new, count, label in [
        ("idea_screens_r10", "idea_screens_r11", 15, "bundle-namespace"),
        ("R10", "R11", 20, "messages"),
        ("r10", "r11", 6, "schemas-and-identities"),
    ]:
        source = replace_exact(source, old, new, count, label, log)

    old_constants = """CANONICAL_DATA_SHA256 = __CANONICAL_PIN_REPR__
RUNTIME_PY = __MODULES_REPR__
EXPECTED_RUNTIME_PY = {
    "idea_screens_r11": "idea_screens_r11/__init__.py",
    "idea_screens_r11.geometry": "idea_screens_r11/geometry.py",
    "idea_screens_r11.panel_io": "idea_screens_r11/panel_io.py",
    "idea_screens_r11.run_probe": "idea_screens_r11/run_probe.py",
}
MANIFEST_REL = "idea_screens_r11/PREPARATION_MANIFEST.json"
EXTRA_REL = {
    MANIFEST_REL,
    "idea_screens_r11/IMPLEMENTATION_REVIEW.json",
    "idea_screens_r11/TEST_RECEIPT.json",
    "idea_screens_r11/README.md",
}
"""
    new_constants = """CANONICAL_PINS = __CANONICAL_PINS_REPR__
RUNTIME_PY = __MODULES_REPR__
EXPECTED_RUNTIME_PY = __EXPECTED_RUNTIME_REPR__
ALLOWED_BUNDLE_ROOTS = __ALLOWED_ROOTS_REPR__
R7_SOURCE_ALLOWLIST = __R7_ALLOWLIST_REPR__
CANONICAL_ROOT = B / "source-a266b036/src"
MANIFEST_REL = "idea_screens_r11/PREPARATION_MANIFEST.json"
EXTRA_REL = {
    MANIFEST_REL,
    "idea_screens_r11/IMPLEMENTATION_REVIEW.json",
    "idea_screens_r11/TEST_RECEIPT.json",
    "idea_screens_r11/README.md",
}
"""
    source = replace_exact(
        source, old_constants, new_constants, 1, "runtime-and-pin-constants", log
    )

    helper_marker = "\ndef write_create_only(path, value):\n"
    helper = """
def verify_canonical_source_pins():
    for pin_name, expected_sha256 in CANONICAL_PINS.items():
        relative = pathlib.PurePosixPath(pin_name.removeprefix("canonical:"))
        if (
            relative.is_absolute()
            or not relative.parts
            or relative.parts[0] != "wnn"
            or ".." in relative.parts
        ):
            raise RuntimeError("canonical semantic pin escapes the WNN source root: " + pin_name)
        path = CANONICAL_ROOT / relative
        if not path.is_file() or path.is_symlink() or sha_file(path) != expected_sha256:
            raise RuntimeError("canonical WNN source pin differs: " + pin_name)

"""
    source = replace_exact(
        source, helper_marker, helper + helper_marker, 1, "canonical-pin-verifier", log
    )

    old_pin_validation = """    file_pins = manifest.get("files")
    if (
        not isinstance(file_pins, dict)
        or file_pins.get("canonical:wnn/data.py") != CANONICAL_DATA_SHA256
    ):
        raise RuntimeError("manifest canonical semantic pin differs from the separately frozen pin")
    if any(not isinstance(name, str) for name in file_pins):
        raise RuntimeError("manifest file pin names must be strings")
    canonical_names = {name for name in file_pins if name.startswith("canonical:")}
    if canonical_names != {"canonical:wnn/data.py"}:
        raise RuntimeError("R11 may carry only the canonical wnn/data.py semantic pin")
    noncanonical = {name for name in file_pins if not name.startswith("canonical:")}
    if any(
        not name.startswith("idea_screens_r11/")
        or ".." in pathlib.PurePosixPath(name).parts
        or pathlib.PurePosixPath(name).is_absolute()
        or pathlib.PurePosixPath(name).as_posix() != name
        or not isinstance(file_pins[name], str)
        or len(file_pins[name]) != 64
        or any(char not in "0123456789abcdef" for char in file_pins[name])
        for name in noncanonical
    ):
        raise RuntimeError("manifest noncanonical source path escapes the R11 allowlist")
"""
    new_pin_validation = """    file_pins = manifest.get("files")
    if not isinstance(file_pins, dict) or any(not isinstance(name, str) for name in file_pins):
        raise RuntimeError("manifest file pins must be a string-keyed object")
    canonical_pins = {
        name: digest for name, digest in file_pins.items() if name.startswith("canonical:")
    }
    if canonical_pins != CANONICAL_PINS:
        raise RuntimeError("manifest canonical semantic pins differ from the frozen three-file set")
    noncanonical = {name for name in file_pins if not name.startswith("canonical:")}
    for name in noncanonical:
        rel = pathlib.PurePosixPath(name)
        if (
            rel.is_absolute()
            or not rel.parts
            or ".." in rel.parts
            or rel.as_posix() != name
            or rel.parts[0] not in ALLOWED_BUNDLE_ROOTS
            or not isinstance(file_pins[name], str)
            or len(file_pins[name]) != 64
            or any(char not in "0123456789abcdef" for char in file_pins[name])
        ):
            raise RuntimeError("manifest noncanonical source path escapes the frozen bundle roots")
        if rel.parts[0] == "idea_screens_r7" and name not in R7_SOURCE_ALLOWLIST:
            raise RuntimeError("manifest R7 source exceeds the four-file runtime allowlist")
    if set(file_pins) != noncanonical | set(CANONICAL_PINS):
        raise RuntimeError("manifest contains an unexpected source pin")
"""
    source = replace_exact(
        source, old_pin_validation, new_pin_validation, 1, "manifest-pin-validation", log
    )

    source = replace_exact(
        source,
        """            or rel.parts[0] != "idea_screens_r11"
""",
        """            or rel.parts[0] not in ALLOWED_BUNDLE_ROOTS
            or (rel.parts[0] == "idea_screens_r7" and name not in R7_SOURCE_ALLOWLIST)
""",
        1,
        "bundle-prefix-guard",
        log,
    )

    source = replace_exact(
        source,
        """    canonical_path = B / "source-a266b036/src/wnn/data.py"
    if not canonical_path.is_file() or sha_file(canonical_path) != CANONICAL_DATA_SHA256:
        raise RuntimeError("canonical data.py semantic source pin differs")
""",
        "    verify_canonical_source_pins()\n",
        1,
        "initial-canonical-pin-check",
        log,
    )
    source = replace_exact(
        source,
        """    if sha_file(B / "source-a266b036/src/wnn/data.py") != CANONICAL_DATA_SHA256:
        raise RuntimeError("canonical data.py semantic pin changed before launch")
""",
        "    verify_canonical_source_pins()\n",
        1,
        "prelaunch-canonical-pin-check",
        log,
    )
    source = replace_exact(
        source,
        '        "PYTHONPATH=" + str(DEST),\n',
        '        "PYTHONPATH=" + str(CANONICAL_ROOT) + ":" + str(DEST),\n',
        1,
        "canonical-pythonpath",
        log,
    )
    source = replace_exact(
        source,
        '        "canonical_data_py_sha256": CANONICAL_DATA_SHA256,\n',
        '        "canonical_source_sha256": CANONICAL_PINS,\n',
        1,
        "admission-canonical-bindings",
        log,
    )

    replacements = {
        "__MANIFEST_SHA__": manifest_sha256,
        "__CANONICAL_PINS_REPR__": repr(canonical_pins),
        "__MODULES_REPR__": repr(spec["runtime_py"]),
        "__EXPECTED_RUNTIME_REPR__": repr(spec["runtime_py"]),
        "__ALLOWED_ROOTS_REPR__": repr(sorted(spec["allowed_bundle_prefixes"])),
        "__R7_ALLOWLIST_REPR__": repr(sorted(spec["r7_source_allowlist"])),
        "__INLINE_REPR__": repr(inline),
    }
    for placeholder, value in replacements.items():
        source = replace_exact(source, placeholder, value, 1, "bind-" + placeholder.strip("_"), log)
    if any(token in source for token in PLACEHOLDERS) or "idea_screens_r10" in source:
        raise RuntimeError("assembled R11 source retains an unbound or stale R10 token")
    compile(source, "r11_remote.py", "exec")
    return source


def patch_identity_helper(
    source: str, identity: list[tuple[str, str, int, str]], log: list[dict[str, Any]]
) -> str:
    """Apply only the recorded R11 identity substitutions to an R10 helper."""
    for old, new, count, label in identity:
        source = replace_exact(source, old, new, count, label, log)
    if "r10" in source or "R10" in source:
        raise RuntimeError("identity-substituted helper retains an R10 marker")
    compile(source, "r11_helper.py", "exec")
    return source


def main() -> None:
    """Assemble pinned R11 scripts locally without performing remote actions."""
    spec = json.loads(SPEC_PATH.read_bytes())
    if spec.get("schema") != "nf19.r11_startup_assembly_spec.v1":
        raise RuntimeError("unexpected R11 startup assembly specification")
    if spec.get("r10_base_pins") != R10_BASE_PINS:
        raise RuntimeError("assembly specification does not pin the reviewed R10 helpers")
    validate_spec(spec)
    _manifest, manifest_bytes, files, canonical_pins = manifest_and_transport(spec)
    manifest_sha256 = digest(manifest_bytes)

    helper_dir = R10 / "records/startup_v1"
    remote_body = read_pinned(
        helper_dir / "remote_body_v1.py", R10_BASE_PINS["remote_body_v1.py"]
    ).decode()
    capture = read_pinned(
        helper_dir / "capture_remote_v2.py", R10_BASE_PINS["capture_remote_v2.py"]
    ).decode()
    stop = read_pinned(
        helper_dir / "stop_remote_v1.py", R10_BASE_PINS["stop_remote_v1.py"]
    ).decode()
    inline = (PACKET / "root_inline_attempt2.py").read_text()
    inline_log: list[dict[str, Any]] = []
    for placeholder, value in [
        ("__MANIFEST_SHA__", manifest_sha256),
        ("__CANONICAL_PIN_REPR__", repr(canonical_pins)),
        ("__MODULES_REPR__", repr(spec["runtime_py"])),
    ]:
        inline = replace_exact(
            inline, placeholder, value, 1, "inline-bind-" + placeholder.strip("_"), inline_log
        )
    if any(token in inline for token in PLACEHOLDERS) or "idea_screens_r10" in inline:
        raise RuntimeError("R11 inline import smoke retains an unbound or stale token")
    compile(inline, "r11_inline.py", "exec")

    body_log: list[dict[str, Any]] = []
    remote_body = patch_remote_body(
        remote_body, spec, canonical_pins, manifest_sha256, inline, body_log
    )
    capture_log: list[dict[str, Any]] = []
    capture = patch_identity_helper(
        capture,
        [
            ("r10-v1", "r10-v2", 3, "capture-attempt2-identities"),
            ("r10", "r11", 4, "capture-identities"),
        ],
        capture_log,
    )
    stop_log: list[dict[str, Any]] = []
    stop = patch_identity_helper(
        stop,
        [
            ("r10-v1", "r10-v2", 2, "stop-attempt2-identities"),
            ("r10", "r11", 3, "stop-identities"),
        ],
        stop_log,
    )
    cfg = base64.b64encode(json.dumps({"files": files}, sort_keys=True).encode()).decode()
    script = f"import base64,json\nCFG=json.loads(base64.b64decode({cfg!r}))\n" + remote_body
    compile(script, "r11_remote.py", "exec")

    PRIVATE.mkdir(parents=True, exist_ok=True)
    outputs = {
        PRIVATE / "r11_remote.py": script,
        PRIVATE / "r11_capture.py": capture,
        PRIVATE / "r11_stop.py": stop,
    }
    for path, text in outputs.items():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)

    receipt = {
        "schema": "nf19.r11_transport_assembly.attempt2.v1",
        "prepared_utc": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "preparation_manifest_sha256": manifest_sha256,
        "assembly_spec_sha256": digest(SPEC_PATH.read_bytes()),
        "assembler_sha256": digest(Path(__file__).read_bytes()),
        "root_inline_template_sha256": digest((PACKET / "root_inline_attempt2.py").read_bytes()),
        "r10_helper_pins": R10_BASE_PINS,
        "runtime_py": spec["runtime_py"],
        "canonical_source_sha256": canonical_pins,
        "transport_files": {
            name: {key: value for key, value in entry.items() if key != "base64"}
            for name, entry in files.items()
        },
        "generated_scripts": {
            path.name: {"bytes": len(text.encode()), "sha256": digest(text.encode())}
            for path, text in outputs.items()
        },
        "literal_patch_log": {
            "root_inline": inline_log,
            "remote_body": body_log,
            "capture_remote_v2": capture_log,
            "stop_remote_v1": stop_log,
        },
        "remote_actions_performed": False,
    }
    with (PACKET / "ASSEMBLY_RECEIPT_ATTEMPT2.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in receipt.items()
                if key not in {"transport_files", "literal_patch_log"}
            }
        )
    )


if __name__ == "__main__":
    main()
