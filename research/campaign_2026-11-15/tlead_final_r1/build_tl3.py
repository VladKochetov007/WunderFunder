"""Build a TL3-runtime ZIP: runtime bytes from the original team tl3.zip, members from NPZ paths.

Usage: build_tl3.py OUT_ZIP ENTRY=NPZ_PATH=NAME=COEF ...  (ENTRY like weights_tl7.npz)
"""
import hashlib, json, sys, zipfile
from pathlib import Path

BASE = Path("/home/vlad/development/wunder/wnn_connectome_starterpack/submissions/team_tl3_original_20261018.zip")
BASE_SHA = "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
RUNTIME = ["solution.py", "avx2/wnn_cpp.cpython-311-x86_64-linux-gnu.so",
           "v2/wnn_cpp.cpython-311-x86_64-linux-gnu.so", "v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so"]
sha = lambda b: hashlib.sha256(b).hexdigest()

def main(out: str, specs: list[str]) -> None:
    raw = BASE.read_bytes(); assert sha(raw) == BASE_SHA
    with zipfile.ZipFile(BASE) as z: entries = {k: z.read(k) for k in RUNTIME}
    members = []
    for s in specs:
        entry, path, name, coef = s.split("=")
        entries[entry] = Path(path).read_bytes() if path != "BASE" else zipfile.ZipFile(BASE).read(entry)
        members.append({"weights": entry, "name": name, "coef": float(coef)})
    assert abs(sum(m["coef"] for m in members) - 1) < 1e-9
    entries["members.json"] = json.dumps(members, indent=2).encode()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for k in sorted(entries):
            info = zipfile.ZipInfo(k, date_time=(2026, 11, 15, 0, 0, 0)); info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED; z.writestr(info, entries[k])
    rec = {"zip": out, "bytes": Path(out).stat().st_size, "sha256": sha(Path(out).read_bytes()),
           "base_zip_sha256": BASE_SHA, "members": members,
           "entry_sha256": {k: sha(v) for k, v in sorted(entries.items())}}
    Path(out).with_suffix(".receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("zip", "bytes", "sha256")}))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
