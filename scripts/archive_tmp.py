"""Archive useful project debris from /tmp: small text to git-trackable research/, everything else to an ignored folder.

Per file: skip junk (caches, site-packages, shared objects); drop files already identical to a git-tracked blob;
send secret-looking files to a private (ignored, 0600) folder; send small text/log/trace/plot files to
``research/tmp_archive_<tag>/`` (committable); send everything else (weights, arrays, archives, large files) to the ignored
``artifacts/tmp_archive_<tag>/``.  Identical contents (sha256) are stored once.  Nothing is deleted unless
``--delete-originals`` is given, and then only for a top-level entry whose every file is archived-and-verified, a
git duplicate, a content duplicate or junk, and which no process has open and nothing touched recently.
Dry run by default; a MANIFEST.json (source, sha256, bytes, action, destination) is written for audits.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path("/tmp")

SYSTEM = [
    "claude-1000", "codex-daemon-*", "codex-bwrap-*", ".X*", "*.sock", "systemd-*", "tmux-*", ".ICE-unix", ".font-unix",
    "pytest-of-*", "node-compile-cache", "torchinductor_*", "hsperfdata_*", "vscode*", ".vscode*", "pyright*", "scoped_dir*",
    "mcp-*", "proc-macro-*", "pymp-*", "snap-private-tmp", "gitkraken", "opencode", "sessions", "uv-*.lock", "vercel-plugin-*",
    "exthost-*.cpuprofile", "python-languageserver-cancellation", "*.so", ".org.chromium.*", "h230_ruff_cache", "e528-mplconfig",
    "tmpnhb_l61s", "Cache", "Notification", "w2v", ".kaggle", "wnn-codex-support.lock", "wnn-sol-jax-deps",
    "wnn-independent-py311-*", "wnn-terminal-collector-py311-cache", "wnn-autopilot-review-pycache", ".Test-unix",
]
JUNK_PARTS = {"__pycache__", ".git", "node_modules", ".venv", "site-packages", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
JUNK_SUFFIX = (".pyc", ".pyo", ".so", ".o", ".tmp", ".partial", ".lock", ".swp")
TEXT_SUFFIX = {".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".log", ".yaml", ".yml", ".toml", ".cfg", ".ini",
               ".png", ".svg", ".html", ".rs", ".cpp", ".h", ".hpp", ".cu", ".ipynb", ".stdout", ".stderr", ".out", ".err", ".diff", ".patch"}
SECRET_NAME = re.compile(r"(private|secret|token|credential|signed|apikey|api_key|\.env|id_rsa|known_hosts)", re.I)
SECRET_BODY = re.compile(
    rb"(X-Goog-Signature|X-Amz-Signature|AWSAccessKeyId|AKIA[0-9A-Z]{12,}|BEGIN [A-Z ]*PRIVATE KEY|vastai\s*=\s*\w|"
    rb"KGAT_[0-9a-f]{8,}|\"key\"\s*:\s*\"[0-9a-f]{32,}\"|https?://[^\s\"']*[?&]Signature=)", re.I)
TEXT_MAX = 5 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_blobs() -> set[str]:
    out = subprocess.run(["git", "ls-tree", "-r", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return {line.split()[2] for line in out.splitlines() if line.strip()}


def git_blob_ids(paths: list[Path]) -> dict[Path, str]:
    if not paths:
        return {}
    result = subprocess.run(["git", "hash-object", "--stdin-paths"], cwd=ROOT, input="\n".join(map(str, paths)),
                            capture_output=True, text=True, check=True)
    return dict(zip(paths, result.stdout.split(), strict=True))


def repo_big_files() -> dict[int, list[Path]]:
    """Index untracked bulk files of the repo (caches, run directories) by size for duplicate detection."""
    index: dict[int, list[Path]] = {}
    for base in (ROOT / "cache", ROOT / "artifacts", ROOT / "datasets"):
        for path in base.rglob("*") if base.exists() else []:
            if path.is_file() and not path.is_symlink() and "tmp_archive_" not in str(path):
                size = path.stat().st_size
                if size >= 1_000_000:
                    index.setdefault(size, []).append(path)
    return index


def in_use(entry: Path) -> bool:
    out = subprocess.run(["lsof", "-w", "+D", str(entry)], capture_output=True, text=True, check=False, timeout=60)
    if len([line for line in out.stdout.splitlines() if line.strip()]) > 1:
        return True
    needle = str(entry)
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if needle in cmdline or cwd.startswith(needle):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="20261009")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--delete-originals", action="store_true")
    parser.add_argument("--recent-minutes", type=int, default=20)
    args = parser.parse_args()
    git_root = ROOT / f"research/tmp_archive_{args.tag}"
    ignored_root = ROOT / f"artifacts/tmp_archive_{args.tag}"
    private_root = ignored_root / "_private"
    entries = sorted(p for p in TMP.iterdir() if not any(fnmatch.fnmatch(p.name, pat) for pat in SYSTEM))
    blobs = tracked_blobs()
    repo_index = repo_big_files()
    repo_hash_cache: dict[Path, str] = {}
    seen: dict[str, str] = {}
    manifest, per_entry = [], {}
    for entry in entries:
        files = [entry] if entry.is_file() or entry.is_symlink() else [p for p in entry.rglob("*") if p.is_file() or p.is_symlink()]
        if entry.is_dir() and not files:
            per_entry[entry.name] = {"files": 0, "clean": True}
            continue
        candidates = [p for p in files if not p.is_symlink() and p.suffix.lower() in TEXT_SUFFIX and p.stat().st_size <= TEXT_MAX
                      and not any(part in JUNK_PARTS for part in p.parts)]
        blob_ids = git_blob_ids(candidates)
        clean = True
        for path in files:
            rel = path.relative_to(TMP)
            record = {"source": str(path), "bytes": 0}
            try:
                size = path.stat().st_size
            except OSError:
                record["action"] = "unreadable"
                clean = False
                manifest.append(record)
                continue
            record["bytes"] = size
            if path.is_symlink() or any(part in JUNK_PARTS for part in path.parts) or path.name.endswith(JUNK_SUFFIX):
                record["action"] = "junk"
            elif path in blob_ids and blob_ids[path] in blobs:
                record["action"] = "git_duplicate"
            else:
                try:
                    digest = sha256(path)
                except OSError:
                    record["action"] = "unreadable"
                    clean = False
                    manifest.append(record)
                    continue
                record["sha256"] = digest
                text = path.suffix.lower() in TEXT_SUFFIX and size <= TEXT_MAX
                secret = bool(SECRET_NAME.search(path.name))
                if text and not secret and path.suffix.lower() not in {".png"}:
                    secret = bool(SECRET_BODY.search(path.read_bytes()))
                twin = None
                if size >= 1_000_000:
                    for other in repo_index.get(size, []):
                        if repo_hash_cache.setdefault(other, sha256(other)) == digest:
                            twin = other
                            break
                if twin is not None:
                    record["action"], record["duplicate_of"] = "repo_duplicate", str(twin)
                elif "/nested/vladkochetov07/" in str(path) or (path.suffix.lower() == ".zip" and size > 1_000_000_000):
                    record["action"] = "regenerable_download"
                elif digest in seen:
                    record["action"] = "content_duplicate"
                    record["duplicate_of"] = seen[digest]
                else:
                    if secret:
                        record["action"], dest = "private_ignored", private_root / rel
                    elif text:
                        record["action"], dest = "git", git_root / rel
                    else:
                        record["action"], dest = "ignored", ignored_root / rel
                    record["dest"] = str(dest)
                    seen[digest] = str(dest)
                    if args.execute:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, dest)
                        if sha256(dest) != digest:
                            record["action"], clean = "copy_mismatch", False
                        elif record["action"] == "private_ignored":
                            os.chmod(dest, 0o600)
            manifest.append(record)
        per_entry[entry.name] = {"files": len(files), "clean": clean}
    summary = Counter()
    for r in manifest:
        summary[r["action"]] += 1
        summary[f"bytes_{r['action']}"] += r["bytes"]
    print(json.dumps({"entries": len(entries), "files": len(manifest), "summary": dict(summary)}, indent=1))
    if not args.execute:
        biggest = sorted((r for r in manifest if r["action"] in ("ignored", "git")), key=lambda r: -r["bytes"])[:12]
        for r in biggest:
            print(f"  {r['bytes'] / 1e6:9.1f} MB {r['action']:8s} {r['source']}")
        by_entry = Counter()
        for r in manifest:
            if r["action"] == "ignored":
                by_entry[r["source"].split("/")[2]] += r["bytes"]
        print("ignored bytes by top-level entry:", {k: round(v / 1e6) for k, v in by_entry.most_common(10)})
        print("unreadable:", [r["source"] for r in manifest if r["action"] == "unreadable"][:8])
        return 0
    git_root.mkdir(parents=True, exist_ok=True)
    (git_root / "MANIFEST.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "dest" or r["action"] in ("git", "ignored")} for r in manifest], indent=1))
    removed, kept = [], []
    if args.delete_originals:
        cutoff = time.time() - 60 * args.recent_minutes
        for entry in entries:
            info = per_entry[entry.name]
            if not info["clean"]:
                kept.append((entry.name, "incomplete"))
                continue
            newest = max([p.stat().st_mtime for p in ([entry] if entry.is_file() else entry.rglob("*")) if p.exists()] or [0])
            if newest > cutoff:
                kept.append((entry.name, "recent"))
            elif entry.is_dir() and in_use(entry):
                kept.append((entry.name, "in_use"))
            else:
                shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
                removed.append(entry.name)
    (git_root / "DELETION_RECEIPT.json").write_text(json.dumps({"removed": removed, "kept": kept}, indent=1))
    print(json.dumps({"removed": len(removed), "kept": kept[:20], "kept_count": len(kept)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
