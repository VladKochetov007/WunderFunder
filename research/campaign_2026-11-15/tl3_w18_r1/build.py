"""Build fixed TL3-family ZIPs adding/replacing the student slot with team W18-final.

Runtime files and TL5/TL6/S5 weights are copied byte-for-byte from the original
team tl3.zip; W18 weights byte-for-byte from tabdata-w18-final.zip. Only members.json changes.
"""
import hashlib, json, sys, zipfile
from pathlib import Path

BASE = Path("/home/vlad/development/wunder/wnn_connectome_starterpack/submissions/team_tl3_original_20261018.zip")
W18 = Path("/home/vlad/development/wunder/research_tmp/q15_best_r1/w18/submission/tabdata-w18-final.zip")
OUT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack/submissions/final_20261115")
BASE_SHA = "d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569"
CANDIDATES = {
    # parity probes (not submissions)
    "probe_w18_only": [("weights_w18.npz", "tl3final_w18final_distilled_s5@ep1", 1.0)],
    "probe_s5_only": [("weights_s5.npz", "distilled_s5@ep1", 1.0)],
    # submission candidates
    "tl3_w18swap": [("weights_w18.npz", "tl3final_w18final_distilled_s5@ep1", 0.34),
                    ("weights_tl5.npz", "fttall_tlead_s5@ep3", 0.33),
                    ("weights_tl6.npz", "fttall_tlead_s6@ep3", 0.33)],
    "tl3_s5w18": [("weights_s5.npz", "distilled_s5@ep1", 0.17),
                  ("weights_w18.npz", "tl3final_w18final_distilled_s5@ep1", 0.17),
                  ("weights_tl5.npz", "fttall_tlead_s5@ep3", 0.33),
                  ("weights_tl6.npz", "fttall_tlead_s6@ep3", 0.33)],
}
RUNTIME = ["solution.py", "avx2/wnn_cpp.cpython-311-x86_64-linux-gnu.so",
           "v2/wnn_cpp.cpython-311-x86_64-linux-gnu.so", "v3/wnn_cpp.cpython-311-x86_64-linux-gnu.so"]

def sha(b): return hashlib.sha256(b).hexdigest()

def main():
    assert sha(BASE.read_bytes()) == BASE_SHA
    with zipfile.ZipFile(BASE) as z:
        files = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    with zipfile.ZipFile(W18) as z:
        files["weights_w18.npz"] = z.read("weights_w18.npz")
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"base_zip_sha256": BASE_SHA, "w18_zip_sha256": sha(W18.read_bytes()), "candidates": {}}
    for name, members in CANDIDATES.items():
        assert abs(sum(c for _, _, c in members) - 1.0) < 1e-12
        mj = json.dumps([{"weights": w, "name": n, "coef": c} for w, n, c in members], indent=2).encode()
        entries = {k: files[k] for k in RUNTIME}
        entries.update({w: files[w] for w, _, _ in members})
        entries["members.json"] = mj
        dest = OUT / f"{name}.zip"
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for k in sorted(entries):
                info = zipfile.ZipInfo(k, date_time=(2026, 11, 15, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, entries[k])
        receipt["candidates"][name] = {"zip": str(dest), "bytes": dest.stat().st_size,
            "sha256": sha(dest.read_bytes()), "members": json.loads(mj),
            "entry_sha256": {k: sha(v) for k, v in sorted(entries.items())}}
    Path("BUILD_RECEIPT_R1.json").write_text(json.dumps(receipt, indent=1))
    for k, v in receipt["candidates"].items(): print(k, v["bytes"], v["sha256"])

if __name__ == "__main__":
    main()
