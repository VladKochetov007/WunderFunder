#!/usr/bin/env python3
"""Redact credential-like HTTP query values in a publication campaign copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote_plus, urlsplit, urlunsplit

DEFAULT_ROOT = Path(
    "/home/vlad/development/wunder/research_tmp/resume4h-publication-20261024"
    "/research/campaign_2026-10-24/resume4h_r1"
)
SAFE_PARENT = Path("/home/vlad/development/wunder/research_tmp").resolve()
SOURCE_CAMPAIGN = Path(
    "/home/vlad/development/wunder/wnn_connectome_starterpack/research"
    "/campaign_2026-10-24/resume4h_r1"
).resolve()
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".patch", ".py", ".sh"}
OMITTED_DUMP = re.compile(r"PR\d+_FILES_GH\.json\Z")
URL_PATTERN = re.compile(r"https?://[^\s<>`\"']+")
TRAILING_PUNCTUATION = ".,;:!?)]}"
REDACTED_VALUES = {"redacted", "%5bredacted%5d"}


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def query_class(raw_key: str) -> str | None:
    """Classify a sensitive parameter name without inspecting its value."""
    decoded = unquote_plus(raw_key).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", decoded).strip("_")
    if "api_key" in normalized or "apikey" in normalized:
        return "api_key"
    for term in ("token", "signature", "credential", "secret"):
        if term in normalized:
            return term
    return None


def redact_url(url_text: str) -> tuple[str, Counter[str]]:
    """Replace only values for credential-like query parameters."""
    suffix = ""
    while url_text and url_text[-1] in TRAILING_PUNCTUATION:
        suffix = url_text[-1] + suffix
        url_text = url_text[:-1]
    parts = urlsplit(url_text)
    if not parts.query:
        return url_text + suffix, Counter()

    counts: Counter[str] = Counter()
    query_parts: list[str] = []
    for item in parts.query.split("&"):
        raw_key, separator, raw_value = item.partition("=")
        category = query_class(raw_key)
        if category is None or not separator or raw_value.lower() in REDACTED_VALUES:
            query_parts.append(item)
            continue
        counts[category] += 1
        query_parts.append(f"{raw_key}=REDACTED")

    redacted = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, "&".join(query_parts), parts.fragment)
    )
    return redacted + suffix, counts


def transform_text(text: str) -> tuple[str, Counter[str]]:
    """Redact credential query values from every HTTP(S) URL in text."""
    counts: Counter[str] = Counter()

    def replace(match: re.Match[str]) -> str:
        redacted, found = redact_url(match.group(0))
        counts.update(found)
        return redacted

    return URL_PATTERN.sub(replace, text), counts


def eligible_files(root: Path) -> tuple[list[Path], list[str]]:
    """List publication text files and explicitly omitted raw PR dumps."""
    selected: list[Path] = []
    omitted: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if OMITTED_DUMP.fullmatch(path.name):
            omitted.append(path.relative_to(root).as_posix())
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            selected.append(path)
    return sorted(selected), sorted(omitted)


def create_receipt_path(receipt_dir: Path) -> Path:
    """Choose a create-only R-numbered receipt path."""
    for revision in range(1, 1000):
        path = receipt_dir / f"URL_QUERY_REDACTION_R{revision}.json"
        if not path.exists():
            return path
    raise RuntimeError("no free URL redaction receipt revision remains")


def main() -> int:
    """Sanitize publication-copy metadata and create a count-only receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    if root == SOURCE_CAMPAIGN or SOURCE_CAMPAIGN in root.parents:
        raise SystemExit("refusing to modify the frozen source campaign")
    if SAFE_PARENT not in root.parents:
        raise SystemExit("publication root must remain under research_tmp")
    if root.name != "resume4h_r1" or root.parent.name != "campaign_2026-10-24":
        raise SystemExit("publication root does not match the expected campaign layout")

    files, omitted = eligible_files(root)
    changed: list[dict[str, object]] = []
    total_counts: Counter[str] = Counter()
    for path in files:
        original = path.read_bytes()
        try:
            text = original.decode("utf-8")
        except UnicodeDecodeError:
            continue
        sanitized_text, counts = transform_text(text)
        sanitized = sanitized_text.encode("utf-8")
        if not counts:
            continue
        # Atomic replacement keeps the publication copy readable if interrupted.
        mode = stat.S_IMODE(path.stat().st_mode)
        temporary = path.with_name(path.name + ".redaction-tmp")
        temporary.write_bytes(sanitized)
        temporary.chmod(mode)
        temporary.replace(path)
        total_counts.update(counts)
        changed.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "original_sha256": sha256_bytes(original),
                "sanitized_sha256": sha256_bytes(sanitized),
                "redacted_query_value_count": int(sum(counts.values())),
                "redacted_parameter_name_counts": dict(sorted(counts.items())),
            }
        )

    # Rescan the same text scope; report only names/counts, never values.
    remaining_by_name: Counter[str] = Counter()
    for path in files:
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        for match in URL_PATTERN.finditer(text):
            raw_url = match.group(0)
            while raw_url and raw_url[-1] in TRAILING_PUNCTUATION:
                raw_url = raw_url[:-1]
            query = urlsplit(raw_url).query
            for item in query.split("&") if query else ():
                raw_key, separator, raw_value = item.partition("=")
                category = query_class(raw_key)
                if category is not None and separator and raw_value.lower() not in REDACTED_VALUES:
                    remaining_by_name[category] += 1

    script_path = Path(__file__).resolve()
    receipt = {
        "schema": "wnn.publication-url-query-redaction.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "publication_root": str(root),
        "source_campaign_root_untouched": str(SOURCE_CAMPAIGN),
        "script_sha256": sha256_bytes(script_path.read_bytes()),
        "text_suffixes_scanned": sorted(TEXT_SUFFIXES),
        "excluded_filename_pattern": OMITTED_DUMP.pattern,
        "excluded_omitted_raw_dumps": omitted,
        "scanned_text_file_count": len(files),
        "changed_file_count": len(changed),
        "redacted_query_value_count": int(sum(total_counts.values())),
        "redacted_parameter_name_counts": dict(sorted(total_counts.items())),
        "changed_files": changed,
        "postscan_remaining_credential_query_values": int(sum(remaining_by_name.values())),
        "postscan_remaining_parameter_name_counts": dict(sorted(remaining_by_name.items())),
        "redaction_value": "REDACTED",
        "preservation_note": "Only credential-like HTTP(S) query values were replaced. URL host/path, source refs, noncredential query segments, fragments, prose, and decoded source artifacts were left unchanged.",
    }
    receipt_dir = Path(__file__).resolve().parent
    receipt_path = create_receipt_path(receipt_dir)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(
        json.dumps(
            {
                "receipt": receipt_path.name,
                "scanned_text_file_count": len(files),
                "changed_file_count": len(changed),
                "redacted_query_value_count": int(sum(total_counts.values())),
                "parameter_names": dict(sorted(total_counts.items())),
                "excluded_dump_count": len(omitted),
                "remaining_sensitive_query_values": int(sum(remaining_by_name.values())),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if not remaining_by_name else 1


if __name__ == "__main__":
    sys.exit(main())
