"""Freeze matched since-scale followups and stage one persistent-RPC transfer."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RPC = ROOT.parent / "cluster-all-experiments-20261112/l4-rpc/requests"


def write_json(path: Path, value: object) -> None:
    """Write new immutable JSON, refusing to overwrite existing evidence."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main() -> None:
    """Preserve failed KD plans and allocate their four unused slots to valid labels."""
    packet = ROOT / "since-scale-followup-r1"
    (packet / "configs").mkdir(parents=True, exist_ok=False)
    cells = []
    for seed in (203401, 203402):
        baseline = ROOT / f"pr49-distillation-r1/configs/label_since32_k16_seed{seed}.json"
        original = json.loads(baseline.read_bytes())
        for scale in (8, 128):
            cell = f"age{scale}_k16_seed{seed}"
            config = dict(original, since_weight=float(scale))
            differences = [key for key in original if original[key] != config[key]]
            assert differences == ["since_weight"]
            path = packet / "configs" / f"{cell}.json"
            write_json(path, config)
            cells.append(
                {
                    "cell": cell,
                    "config": f"configs/{cell}.json",
                    "seed": seed,
                    "since_weight": float(scale),
                    "kd_weight": 0.0,
                    "teacher_cache_required": False,
                    "role": "targeted_followup",
                    "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "matched_since32_config_sha256": hashlib.sha256(
                        baseline.read_bytes()
                    ).hexdigest(),
                    "changed_fields": differences,
                }
            )
    manifest = {
        "schema": "wnn.f3-since-scale-followup.r1",
        "source_commit": "390c846cbdacd8e0e06e6079dd3b04312fa6bcc5",
        "cells": cells,
        "fixed_endpoint": "6 epochs; final epoch index5; all checkpoints retained",
        "allocation": "four new labels-only fits, GPU4-7, replaces rejected KD slots",
        "primary_validation": "official split 10325 fitting /1873 valid;282 linked train excluded",
        "reason": "test the causal age-weight scale around since32 using the same paired seeds",
        "limitations": "exposed validation; two seeds; no independent significance claim",
        "failed_kd_plans": ["../pr49-distillation-r1", "../pr49-distillation-r2"],
        "budget": "unchanged CLOCK_R1; no extra time and no changes to active fits",
    }
    manifest_path = packet / "PANEL_MANIFEST_R1.json"
    write_json(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    controller = (ROOT / "l4_wave_r1.py").read_text(encoding="utf-8")
    changes = {
        '"intake/pr49-distillation-r1"': '"age-intake-r1/since-scale-followup-r1"',
        'choices=("profile-labels", "profile-kd", "fit-labels", "fit-kd")': 'choices=("profile-age", "fit-age")',
        "44fdf3c1d5017c744b3070a5f8dac35fdb95b0ebb8721792e2d19f943267c33d": manifest_sha,
        "gpu = index + (4 if kd else 0)": "gpu = index + 4",
    }
    for before, after in changes.items():
        assert controller.count(before) == 1, before
        controller = controller.replace(before, after)
    compile(controller, "l4_age_wave_r1.py", "exec")
    with (packet / "l4_age_wave_r1.py").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(controller)
    (packet / "README.md").write_text(
        "# Matched causal since-scale followups\n\nFour labels-only fits compare tau8 and tau128 "
        "with the active tau32 and zero controls at seeds203401/203402. Only since_weight "
        "changes. Source, chronological split, optimizer, six-epoch endpoint and online "
        "full validation remain fixed. KD plans were rejected before scientific fitting "
        "and are preserved. Profiles and actual config validation precede admission. "
        "Existing validation is exposed. No submission-quality or significance claim.\n",
        encoding="utf-8",
    )
    archive = ROOT / "l4-since-scale-followup-r1.tar.gz"
    with tarfile.open(archive, "x:gz") as handle:
        handle.add(packet, arcname=packet.name)
    request = {
        "transfer": {
            "local_archive": str(archive),
            "remote_extract_dir": "/data/lc-epanov_s/tabdata-final-quality-20261112-r1/age-intake-r1",
        }
    }
    temporary = RPC / "166-age-panel-transfer-r1.tmp"
    write_json(temporary, request)
    temporary.rename(temporary.with_suffix(".json"))
    print(json.dumps({"manifest_sha256": manifest_sha, "cells": [cell["cell"] for cell in cells]}))


if __name__ == "__main__":
    main()
