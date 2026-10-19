from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

MEMORY_HIGH_BYTES = 1_400_000_000
MEMORY_EXTRA_BUDGET_BYTES = 256_000_000


def _check_physical_output(output: Path, scratch: Path) -> None:
    """Require a new direct child of physical scratch, never tmpfs or RAM storage."""
    if output.exists():
        raise FileExistsError(output)
    if output.parent.resolve() != scratch.resolve():
        raise ValueError("R6 output must be a direct child of existing research_tmp")
    if not scratch.is_dir():
        raise FileNotFoundError(scratch)
    mount_types: list[tuple[int, str]] = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3 and scratch.resolve().is_relative_to(Path(fields[1])):
            mount_types.append((len(fields[1]), fields[2]))
    if not mount_types or max(mount_types)[1] in {"tmpfs", "ramfs"}:
        raise ValueError("R6 scratch is RAM-backed or cannot be verified")


def _cgroup_current_bytes() -> int:
    """Read this process's cgroup-v2 memory.current or fail closed."""
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, _, relative = line.partition(":")
        controllers, _, path = relative.partition(":")
        if hierarchy == "0" and not controllers:
            candidate = Path("/sys/fs/cgroup") / path.lstrip("/") / "memory.current"
            if candidate.is_file():
                return int(candidate.read_text().strip())
    raise RuntimeError("cannot read this process's cgroup-v2 memory.current")


def _process_rss_bytes() -> int | None:
    """Read process RSS separately from hierarchical cgroup memory usage."""
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _guard_memory(stage: str) -> dict[str, int | str | None]:
    """Fail before graph construction if the conservative numerical margin is missing."""
    current = _cgroup_current_bytes()
    if current + MEMORY_EXTRA_BUDGET_BYTES > MEMORY_HIGH_BYTES:
        raise MemoryError(
            f"{stage} memory margin missing: current={current}, "
            f"extra={MEMORY_EXTRA_BUDGET_BYTES}, high={MEMORY_HIGH_BYTES}"
        )
    return {
        "stage": stage,
        "process_rss_bytes": _process_rss_bytes(),
        "own_cgroup_current_bytes": current,
        "conservative_extra_budget_bytes": MEMORY_EXTRA_BUDGET_BYTES,
        "required_high_bytes": MEMORY_HIGH_BYTES,
    }


def _write_json(path: Path, value: object) -> str:
    """Create a compact finite JSON receipt and return its content SHA-256."""
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _append_json_line(path: Path, value: object) -> None:
    """Append a finite JSON progress receipt to the create-only stream."""
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("ab") as stream:
        stream.write(payload)


def _verify_runtime_inventory(source_root: Path, pins: dict[str, str]) -> dict[str, str]:
    """Require every imported Python ``wnn`` module to match its canonical source pin."""
    expected = {key: value for key, value in pins.items() if key.startswith("canonical:")}
    observed: dict[str, str] = {}
    package_root = (source_root / "src").resolve()
    for name, module in sorted(sys.modules.items()):
        if name != "wnn" and not name.startswith("wnn."):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            continue
        path = Path(module_path).resolve()
        if path.suffix != ".py":
            raise ValueError(f"un-pinned non-Python wnn module imported: {name}")
        try:
            relative = path.relative_to(package_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"wnn module imported outside canonical source: {name}") from exc
        key = f"canonical:{relative}"
        if key not in expected:
            raise ValueError(f"runtime imported unpinned wnn module: {key}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected[key]:
            raise ValueError(f"runtime wnn module differs from frozen pin: {key}")
        observed[key] = actual
    if set(observed) != set(expected):
        raise ValueError(
            f"canonical import closure differs: missing={sorted(set(expected) - set(observed))}"
        )
    return observed


def _runtime_modules(
    source_root: Path, pins: dict[str, str]
) -> tuple[ModuleType, dict[str, object]]:
    """Import the pinned canonical components and validate their loaded source inventory."""
    if "wnn" in sys.modules:
        raise RuntimeError("wnn was imported before R6 source validation")
    sys.path.insert(0, str(source_root / "src"))
    torch = importlib.import_module("torch")
    for module in (
        "wnn.data",
        "wnn.features",
        "wnn.nn",
        "wnn.training.environment",
        "wnn.training.forward",
        "wnn.training.inputs",
        "wnn.training.models",
    ):
        importlib.import_module(module)
    imported = _verify_runtime_inventory(source_root, pins)
    return torch, {"wnn_source_inventory": imported, "torch_version": str(torch.__version__)}
