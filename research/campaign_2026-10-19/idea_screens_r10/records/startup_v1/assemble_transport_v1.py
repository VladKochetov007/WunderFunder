from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

REPO = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
CAMPAIGN = REPO / "research/campaign_2026-10-19"
R10 = CAMPAIGN / "idea_screens_r10"
PACKET = R10 / "records/startup_v1"
PRIVATE = Path("/Users/nb/Documents/wunder/runtime-20261018/eda19-r10-v1")


def digest(payload: bytes) -> str:
    """Return an exact transport-byte identity."""
    return hashlib.sha256(payload).hexdigest()


manifest_bytes = (R10 / "PREPARATION_MANIFEST.json").read_bytes()
manifest = json.loads(manifest_bytes)
manifest_sha = digest(manifest_bytes)
paths = {name for name in manifest["files"] if not name.startswith("canonical:")}
paths.update(
    "idea_screens_r10/" + name
    for name in [
        "PREPARATION_MANIFEST.json",
        "IMPLEMENTATION_REVIEW.json",
        "TEST_RECEIPT.json",
        "README.md",
    ]
)
files = {}
for name in sorted(paths):
    payload = (CAMPAIGN / name).read_bytes()
    if name in manifest["files"]:
        assert digest(payload) == manifest["files"][name], name
    files[name] = {
        "bytes": len(payload),
        "sha256": digest(payload),
        "base64": base64.b64encode(payload).decode(),
    }
replacements = {
    "__MANIFEST_SHA__": manifest_sha,
    "__MODULES_REPR__": repr(manifest["runtime_py"]),
    "__CANONICAL_PIN_REPR__": repr(manifest["files"]["canonical:wnn/data.py"]),
}
inline = (PACKET / "root_inline_v1.py").read_text()
body = (PACKET / "remote_body_v1.py").read_text()
for key, value in replacements.items():
    inline = inline.replace(key, value)
    body = body.replace(key, value)
body = body.replace("__INLINE_REPR__", repr(inline))
compile(inline, "r10_inline.py", "exec")
config = base64.b64encode(json.dumps({"files": files}).encode()).decode()
script = f"import base64,json\nCFG=json.loads(base64.b64decode({config!r}))\n" + body
assert "__INLINE_REPR__" not in script and "__MANIFEST_SHA__" not in script
compile(script, "r10_remote.py", "exec")
PRIVATE.mkdir(parents=True, exist_ok=True)
destination = PRIVATE / "r10_remote.py"
with destination.open("x") as stream:
    stream.write(script)
summary = {
    "schema": "nf19.r10_transport.v1",
    "script_bytes": len(script.encode()),
    "script_sha256": digest(script.encode()),
    "preparation_manifest_sha256": manifest_sha,
    "files": {
        name: {key: value for key, value in entry.items() if key != "base64"}
        for name, entry in files.items()
    },
}
with (PACKET / "TRANSPORT_BINDINGS.json").open("x") as stream:
    json.dump(summary, stream, indent=2)
    stream.write("\n")
print(json.dumps({key: value for key, value in summary.items() if key != "files"}))
