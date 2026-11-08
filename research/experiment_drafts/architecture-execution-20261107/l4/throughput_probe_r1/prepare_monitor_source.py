"""Prepare a new source capsule with bounded GRU gate telemetry; never edit source-r1."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

EXPECTED_PARENT = "7d3324230effb050ff083b939e26ad638e5df5c5b72ea70b0f8aed169997db4a"
GATES = "src/wnn/training/sequence_monitor_gates.py"
DIAGNOSTICS = "src/wnn/training/sequence_monitor_diagnostics.py"
HELPER = '''


def _sample_gru_gate_prefix(
    module: nn.LSTM | nn.GRU, inputs: Tensor, state: object
) -> tuple[Tensor, object]:
    """Replay only the first two streams and64 prefix rows with their actual initial state.

    These are contiguous prefix rows from the current TBPTT chunk. Skipping
    intervening rows or resetting a carried state would describe different gates.
    LSTM diagnostic behavior is unchanged by this GRU-specific optimization.
    """
    if not isinstance(module, nn.GRU):
        return inputs, state
    batch = min(inputs.shape[0], MAX_GRU_GATE_SEQUENCES)
    sampled = inputs[:batch, :MAX_GRU_GATE_PREFIX_ROWS].detach()
    if isinstance(state, Tensor):
        state = state[:, :batch].detach()
    return sampled, state
'''


def patched_sources(gates: str, diagnostics: str) -> tuple[str, str]:
    """Apply only the two explicit diagnostic changes, rejecting drifted source anchors."""
    anchor = "    hidden_rows, cell_rows = _initial_state(module, inputs, state)"
    import_anchor = "from wnn.training.sequence_monitor_gates import collect_recurrent_gates"
    call_anchor = "                collect_recurrent_gates(\n"
    if (
        gates.count(anchor) != 1
        or diagnostics.count(import_anchor) != 1
        or diagnostics.count(call_anchor) != 1
    ):
        raise ValueError("frozen diagnostic source anchors differ; do not patch implicitly")
    gates = gates.replace(
        "from torch.nn import functional as F\n",
        "from torch.nn import functional as F\n\nMAX_GRU_GATE_SEQUENCES = 2\nMAX_GRU_GATE_PREFIX_ROWS = 64\n",
        1,
    )
    gates = (
        gates.replace(
            anchor,
            "    inputs, state = _sample_gru_gate_prefix(module, inputs, state)\n" + anchor,
            1,
        ).rstrip()
        + HELPER
    )
    scope = """                if isinstance(module, nn.GRU):
                    x = inputs[0]
                    self._maximum["local/gate/replay_sequence_count"] = x.new_tensor(min(x.shape[0], 2))
                    self._maximum["local/gate/replay_prefix_rows"] = x.new_tensor(min(x.shape[1], 64))
"""
    diagnostics = diagnostics.replace(call_anchor, scope + call_anchor, 1)
    ast.parse(gates)
    ast.parse(diagnostics)
    return gates, diagnostics


def prepare(source: Path, inventory: Path, destination: Path) -> dict[str, object]:
    """Create a distinct620-file capsule and whitelist receipt; perform no model execution."""
    raw = inventory.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_PARENT:
        raise ValueError("source inventory does not match frozen source-r1")
    files = json.loads(raw)["files"]
    if (
        len(files) != 620
        or destination.exists()
        or destination.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("requires the exact parent inventory and a fresh outside destination")
    changed = dict(
        zip(
            (GATES, DIAGNOSTICS),
            patched_sources((source / GATES).read_text(), (source / DIAGNOSTICS).read_text()),
            strict=True,
        )
    )
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, expected in files.items():
        original = source / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("parent inventory path is not a safe relative source path")
        if (
            original.is_symlink()
            or not original.resolve().is_relative_to(source.resolve())
            or hashlib.sha256(original.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"parent source changed: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in changed:
            target.write_text(changed[name], encoding="utf-8", newline="\n")
        else:
            shutil.copyfile(original, target)
        outputs[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    differences = {
        name: {"parent_sha256": files[name], "new_sha256": digest}
        for name, digest in outputs.items()
        if files[name] != digest
    }
    if set(differences) != {GATES, DIAGNOSTICS}:
        raise ValueError("source change escaped the exact two-file diagnostic whitelist")
    manifest = {
        "schema": "wnn.diagnostic-source-capsule.v1",
        "parent_inventory_sha256": EXPECTED_PARENT,
        "files": outputs,
    }
    manifest_path = destination.parent / f"{destination.name}-manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    receipt = {
        "schema": "wnn.gru-gate-prefix-source.v1",
        "parent_inventory_sha256": EXPECTED_PARENT,
        "new_inventory_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source": str(destination),
        "manifest": str(manifest_path),
        "changed_files": differences,
        "scope": "GRU diagnostic gate replay only: first2sequences × contiguous first64rows of current chunk, actual provided chunk-start state; no RNG calls; no model, loss, backward, optimizer or scheduler change",
        "checkpoint_migration_authorized_by_this_receipt": False,
        "models_executed": 0,
    }
    with (destination.parent / f"{destination.name}-patch-receipt.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main() -> None:
    """Prepare a separate immutable source revision for ROOT-owned native profiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.inventory, args.output), indent=2))


if __name__ == "__main__":
    main()
