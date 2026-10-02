"""Token-overlap check of our shipped submission code against teammate branch submission code.

Reads teammate files via `git show` only to measure overlap (nothing is copied). Python is tokenized with `tokenize`
(comments dropped, identifiers kept); C++/Rust with a regex lexer. Reports difflib ratio, the share of our tokens
inside common runs of >= 12 tokens, the longest common run, and shared identifiers (>= 5 chars).
"""
import difflib, io, json, re, subprocess, sys, tokenize, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|->|::|[^\sA-Za-z0-9_]")


def py_tokens(src):
    out = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        out.append("STR" if t.type == tokenize.STRING else t.string)
    return out


def c_tokens(src):
    src = re.sub(r"//[^\n]*|/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r'"(?:\\.|[^"\\])*"', "STR", src)
    return LEX.findall(src)


def compare(ours, theirs):
    sm = difflib.SequenceMatcher(None, ours, theirs, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 12]
    idents = lambda ts: {t for t in ts if re.fullmatch(r"[A-Za-z_]\w{4,}", t)}
    shared = sorted(idents(ours) & idents(theirs))
    return {"ratio": round(sm.ratio(), 3), "share_in_runs12": round(sum(b.size for b in blocks) / max(1, len(ours)), 3),
            "longest_run": max([b.size for b in sm.get_matching_blocks()] or [0]), "shared_identifiers": len(shared),
            "examples": shared[:25]}


def git_show(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout


zip_path = sys.argv[1]
with zipfile.ZipFile(ROOT / zip_path) as z:
    ours_py = z.read("solution.py").decode()
ours_rs = "\n".join((ROOT / "native/src" / f).read_text() for f in ("rnn_fast.rs", "half.rs", "kernels.rs", "python.rs", "approx.rs", "stream.rs", "rnn.rs"))
ref = "origin/cpp-engine"
report = {"zip": zip_path, "ours_py_tokens": len(py_tokens(ours_py))}
report["solution.py_vs_cpp-engine:submission/solution_cpp.py"] = compare(py_tokens(ours_py), py_tokens(git_show(ref, "submission/solution_cpp.py")))
cpp = "\n".join(git_show(ref, p) for p in ("native_cpp/include/wnn/kernels.hpp", "native_cpp/include/wnn/model.hpp",
                                           "native_cpp/include/wnn/activations.hpp", "native_cpp/include/wnn/online_norm.hpp",
                                           "native_cpp/src/bindings.cpp"))
report["native_rust_vs_cpp-engine:native_cpp"] = compare(c_tokens(ours_rs), c_tokens(cpp))
# calibration: same metric between two unrelated-by-design files of our own repo history
report["control_ours:solution.py_vs_baseline/solution.py"] = compare(py_tokens(ours_py), py_tokens((ROOT / "baseline/solution.py").read_text()))
print(json.dumps(report, indent=1))


def run_mask(ours, theirs, min_run=12):
    sm = difflib.SequenceMatcher(None, ours, theirs, autojunk=False)
    m = [False] * len(ours)
    for b in sm.get_matching_blocks():
        if b.size >= min_run:
            for i in range(b.a, b.a + b.size):
                m[i] = True
    return m


# Attribution: overlap with nb's file that is NOT explained by Vlad's own pre-split files (merge-base bfe5e619).
ours = py_tokens(ours_py)
nb = run_mask(ours, py_tokens(git_show(ref, "submission/solution_cpp.py")))
anc = [False] * len(ours)
for f in ("solution.py", "solution_multi.py", "solution_routed.py", "solution_ensemble.py", "solution_blend.py"):
    anc = [a or b for a, b in zip(anc, run_mask(ours, py_tokens(git_show("bfe5e619", f"submission/{f}"))))]
only_nb = sum(n and not a for n, a in zip(nb, anc))
attr = {"tokens": len(ours), "in_runs_with_nb": sum(nb), "in_runs_with_own_pre_split_code": sum(anc),
        "in_runs_with_nb_but_not_own_ancestor": only_nb, "share_unexplained": round(only_nb / len(ours), 3)}
print(json.dumps({"attribution": attr}, indent=1))
