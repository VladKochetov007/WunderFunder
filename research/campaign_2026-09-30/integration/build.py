"""Merge frozen proposals against the exact dirty trainer without live edits."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMPAIGN = HERE.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(base: str, proposals: list[str]) -> str:
    """Combine independent edits; the shared architecture option is explicit."""
    lines = base.splitlines(keepends=True)
    inserts: dict[int, list[str]] = defaultdict(list)
    replaces: dict[tuple[int, int], list[str]] = {}
    arch_options = ["gru", "dual", "gated"]
    for proposal in proposals:
        other = proposal.splitlines(keepends=True)
        for tag, a, b, c, d in difflib.SequenceMatcher(a=lines, b=other, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            change = other[c:d]
            if tag == "insert":
                inserts[a].extend(change)
            elif b == a + 1 and 'ap.add_argument("--arch"' in lines[a]:
                choice_line = change[0]
                choices = ast.literal_eval(choice_line.split("choices=", 1)[1].split(", default=", 1)[0])
                arch_options.extend(option for option in choices if option not in arch_options)
                replaces[(a, b)] = [f'    ap.add_argument("--arch", choices={json.dumps(arch_options)}, default="gru")\n']
                inserts[b].extend(change[1:])
            else:
                if (a, b) in replaces and replaces[(a, b)] != change:
                    raise ValueError(f"conflicting replacement at {a}:{b}")
                replaces[(a, b)] = change
    intervals = sorted(replaces)
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise ValueError("overlapping replacements")
    for a, b in intervals:
        if any(a < i < b for i in inserts):
            raise ValueError("insertion inside replacement")
    starts = {a: (b, replaces[(a, b)]) for a, b in intervals}
    result, index = [], 0
    while index <= len(lines):
        result.extend(inserts.get(index, []))
        if index == len(lines):
            break
        if index in starts:
            index, change = starts[index]
            result.extend(change)
        else:
            result.append(lines[index])
            index += 1
    source = "".join(result)
    ast.parse(source)
    return source


def main() -> None:
    base_path = ROOT / "scripts/train_rnn.py"
    base = base_path.read_text()
    (HERE / "base/scripts").mkdir(parents=True, exist_ok=True)
    (HERE / "base/scripts/train_rnn.py").write_text(base)
    variants, module_sources, inputs = [], {}, {}
    for family, filename, destination in [
        ("structured_state", "model.py", "structured_state.py"),
        ("spread_aux", "trainer_support.py", "spread_aux.py"),
        ("event_content", "model.py", "event_content.py"),
    ]:
        own = CAMPAIGN / family
        target = HERE / f"individual/{family}"
        if target.exists():
            shutil.rmtree(target)
        (target / "scripts").mkdir(parents=True, exist_ok=True)
        (target / "src/wnn").mkdir(parents=True, exist_ok=True)
        (target / "scripts/train_rnn.py").write_text(base)
        patch = own / "integration.patch"
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)], cwd=target, check=True, capture_output=True)
        variants.append((target / "scripts/train_rnn.py").read_text())
        frozen = own / filename
        assert (target / "src/wnn" / destination).read_bytes() == frozen.read_bytes()
        module_sources[destination] = frozen
        inputs[str(patch.relative_to(ROOT))] = sha(patch)
        inputs[str(frozen.relative_to(ROOT))] = sha(frozen)
    outputs = {}
    for label, count in [("ssm_spread", 2), ("all_three", 3)]:
        target = HERE / label
        (target / "scripts").mkdir(parents=True, exist_ok=True)
        (target / "src/wnn").mkdir(parents=True, exist_ok=True)
        proposed = merge(base, variants[:count])
        (target / "scripts/train_rnn.py").write_text(proposed)
        patch_parts = []
        selected = list(module_sources.items())[:count]
        for name, source in selected:
            shutil.copyfile(source, target / "src/wnn" / name)
            patch_parts.extend(difflib.unified_diff([], source.read_text().splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/src/wnn/{name}"))
        patch_parts.extend(difflib.unified_diff(base.splitlines(keepends=True), proposed.splitlines(keepends=True), fromfile="a/scripts/train_rnn.py", tofile="b/scripts/train_rnn.py"))
        patch_path = HERE / f"{label}.patch"
        patch_path.write_text("".join(patch_parts))
        outputs[label] = {"patch_sha256": sha(patch_path), "trainer_sha256": sha(target / "scripts/train_rnn.py"), "modules": {name: sha(source) for name, source in selected}}
    closure = {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "src/wnn").glob("*.py"))}
    for relative in ["scripts/train_rnn.py", "scripts/predict_run.py", "scripts/ft_gclp.sh", "scripts/limited", "research/campaign_2026-09-29/artifacts/splits/fold0.json", "artifacts/runs/c20260929_since_s3_f0/provenance.json", "artifacts/runs/c20260929_since_s3_f0/model_ep3.pt", "artifacts/runs/c20260929_ft_since_s3_f0/provenance.json", "artifacts/runs/c20260929_ft_since_s3_f0/model_ep0.pt", "artifacts/runs/c20260929_ft_since_s4_f0/model_ep0.pt", "artifacts/runs/c20260929_prop_f0/model.pt", "research/campaign_2026-09-30/spread_aux/source_freeze.json", "research/campaign_2026-09-30/spread_aux/metadata.json"]:
        path = ROOT / relative
        closure[relative] = sha(path)
    assert base_path.read_text() == base
    (HERE / "source_closure.json").write_text(json.dumps({"applied": False, "base_trainer_sha256": sha(base_path), "inputs": inputs, "outputs": outputs, "local_runtime_and_artifact_closure": closure, "gate": "all clean parents/FT/prediction jobs finished; all-three also needs independent D1 grade; root recipe freeze before fit"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
