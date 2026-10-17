"""Install the reviewed four-file packet with hash preconditions and rollback.

Never edits historical evidence or data. This is a manual engineering action,
not an experiment launcher. Running it again refuses an existing backup.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    """Stage exact reviewed bytes, verify all preconditions, then activate them."""
    here = Path(__file__).resolve().parent
    root = here.parents[3]
    manifest = json.loads((here / "PATCH_MANIFEST.json").read_text())
    staging = "/home/grinchenkoeg/wunder_research_reset_20261017_r1/patch_staging"
    subprocess.run(["ssh", "beleriand", "mkdir", "-p", staging], check=True)
    for row in manifest["files"]:
        local = root / row["patched_copy"]
        assert hashlib.sha256(local.read_bytes()).hexdigest() == row["after_sha256"]
        subprocess.run(["scp", str(local), f"beleriand:{staging}/{row['name']}.py"], check=True)
    # repr serializes data as a Python literal; no manifest values become shell code.
    remote = "import json\nmanifest = " + repr(manifest) + "\n"
    remote += "staging = " + repr(staging) + "\n"
    remote += '''
import datetime, hashlib, os, pathlib, py_compile, shutil
def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
suffix = ".before-bugfix-20261017"
for row in manifest["files"]:
    target = pathlib.Path(row["remote_path"])
    stage = pathlib.Path(staging) / (row["name"] + ".py")
    assert digest(target) == row["before_sha256"], str(target)
    assert digest(stage) == row["after_sha256"], str(stage)
    assert not pathlib.Path(str(target) + suffix).exists(), str(target)
    py_compile.compile(str(stage), doraise=True)
for row in manifest["files"]:
    target = row["remote_path"]
    shutil.copy2(target, target + suffix)
activated = []
try:
    for row in manifest["files"]:
        target = row["remote_path"]
        os.replace(pathlib.Path(staging) / (row["name"] + ".py"), target)
        activated.append(target)
        assert digest(target) == row["after_sha256"]
except BaseException:
    for target in activated:
        shutil.copy2(target + suffix, target)
    raise
print(json.dumps({"status": "deployed", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": "beleriand", "files": [{"path": r["remote_path"], "sha256": digest(r["remote_path"]),
    "backup": r["remote_path"] + suffix, "backup_sha256": digest(r["remote_path"] + suffix)}
    for r in manifest["files"]], "training_steps": 0}, indent=2))
'''
    completed = subprocess.run(["ssh", "beleriand", "python3", "-"], input=remote,
                               text=True, capture_output=True, check=True)
    receipt = json.loads(completed.stdout)
    (here / "DEPLOYMENT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
