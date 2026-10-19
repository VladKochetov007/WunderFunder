from pathlib import Path
import base64, hashlib, json

repo = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
campaign = repo / "research/campaign_2026-10-19"
r = campaign / "idea_screens_r9"
p = Path("/Users/nb/Documents/wunder/runtime-20261018/eda19-r9-v1")
packet = r / "records/startup_v1"
m = json.loads((r / "PREPARATION_MANIFEST.json").read_bytes())
files = {}
names = [n for n in m["files"] if not n.startswith("canonical:")]
names += [
    "idea_screens_r9/" + n
    for n in [
        "PREPARATION_MANIFEST.json",
        "DESIGN_REVIEW.json",
        "IMPLEMENTATION_REVIEW.json",
        "TEST_RECEIPT.json",
        "README.md",
    ]
]
for n in names:
    b = (campaign / n).read_bytes()
    h = hashlib.sha256(b).hexdigest()
    if n in m["files"]:
        assert h == m["files"][n], n
    files[n] = dict(base64=base64.b64encode(b).decode(), bytes=len(b), sha256=h)
canonical = {
    n.removeprefix("canonical:wnn/"): v
    for n, v in m["files"].items()
    if n.startswith("canonical:wnn/")
}
s = (packet / "remote_body_v1.py").read_text().replace("__CANONICAL_REPR__", repr(canonical))
helper_dir=repo/'research/campaign_2026-10-18/priority25_execution_r4/f3_terminal_execution_r1'
helper_pins={n:hashlib.sha256((helper_dir/n).read_bytes()).hexdigest() for n in ['terminal_gate.py','terminal_remote_v1.py','README.md','PREPARATION_MANIFEST.json','INDEPENDENT_REVIEW_V1.json']}
s=s.replace('__GATE_HELPER_PINS_REPR__',repr(helper_pins))
s = s.replace(
    "__INLINE_REPR__",
    repr(
        (packet / "root_inline_v1.py")
        .read_text()
        .replace(
            "__R9_MANIFEST_SHA__",
            hashlib.sha256((r / "PREPARATION_MANIFEST.json").read_bytes()).hexdigest(),
        )
    ),
).replace(
    "__R9_MANIFEST_SHA__",
    hashlib.sha256((r / "PREPARATION_MANIFEST.json").read_bytes()).hexdigest(),
)
config = base64.b64encode(json.dumps(dict(files=files)).encode()).decode()
script = (
    "import base64,datetime,hashlib,json,os,pathlib,shutil,subprocess,time\nCFG=json.loads(base64.b64decode("
    + repr(config)
    + "))\n"
    + s
)
compile(script, "r9_remote.py", "exec")
with (p / "r9_remote.py").open("x") as f:
    f.write(script)
print(
    json.dumps(
        dict(
            script_bytes=len(script.encode()),
            script_sha256=hashlib.sha256(script.encode()).hexdigest(),
            files=len(files),
            source_bytes=sum(e["bytes"] for e in files.values()),
            canonical_count=len(canonical),
        )
    )
)
