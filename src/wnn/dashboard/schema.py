from __future__ import annotations

import json
import math
import re
from datetime import datetime

EVENT_SCHEMA = "wnn.monitor.v1"
EVENT_KINDS = {"status", "train_step", "evaluation", "diagnostics", "resources"}
METRIC_KEY = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\Z")
SECRET_TEXT = re.compile(
    r"(?i)\b(api[_-]?key|client[_-]?secret|access[_-]?key|ssh[_-]?key|token|secret|"
    r"password|passwd|authorization|cookie)\b"
    r"\s*[:=]\s*([^\s,;]+)"
)
BEARER_TEXT = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*")
URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@")
SENSITIVE_KEY_PARTS = (
    "secret",
    "token",
    "credential",
    "password",
    "passwd",
    "api_key",
    "api-key",
    "private_key",
    "private-key",
    "ssh_key",
    "ssh-key",
    "client_secret",
    "client-secret",
    "access_key",
    "access-key",
    "authorization",
    "cookie",
    "signed_url",
    "signed-url",
)
MAX_METRICS_PER_EVENT = 512
MAX_METADATA_DEPTH = 8
MAX_METADATA_ITEMS = 300
MAX_METADATA_STRING = 4_000
MAX_CONFIG_DEPTH = 32
MAX_METRIC_KEYS_PER_RUN = 2_048


def _is_finite_number(value: object) -> bool:
    """Check finite numeric values without overflowing on large JSON integers."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _parse_time(value: object) -> str | None:
    """Return a canonical ISO timestamp for a timezone-aware event time."""
    if not isinstance(value, str) or len(value) > 80:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.isoformat()


def parse_event(line: bytes) -> dict[str, object] | None:
    """Validate one monitor event and return its bounded normalized form."""
    if not line or len(line) > 131_072:
        return None
    try:
        decoded: object = json.loads(line)
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None
    schema = decoded.get("schema")
    kind = decoded.get("kind")
    step = decoded.get("step")
    epoch = decoded.get("epoch")
    event_time = _parse_time(decoded.get("wall_time"))
    metrics = decoded.get("metrics")
    context = decoded.get("context")
    if schema != EVENT_SCHEMA or kind not in EVENT_KINDS:
        return None
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        return None
    if not _is_finite_number(epoch):
        return None
    if event_time is None or not isinstance(metrics, dict) or not isinstance(context, dict):
        return None
    if len(metrics) > MAX_METRICS_PER_EVENT:
        return None
    normalized_metrics: dict[str, int | float] = {}
    for key, value in metrics.items():
        if (
            not isinstance(key, str)
            or len(key) > 128
            or METRIC_KEY.fullmatch(key) is None
            or not _is_finite_number(value)
        ):
            return None
        normalized_metrics[key] = value
    event: dict[str, object] = {
        "schema": EVENT_SCHEMA,
        "kind": kind,
        "step": step,
        "epoch": float(epoch),
        "wall_time": event_time,
        "metrics": normalized_metrics,
        "context": sanitize_json(context),
    }
    rank = decoded.get("rank")
    if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 0:
        event["rank"] = rank
    return event


def sanitize_json(value: object, depth: int = 0) -> object:
    """Copy display metadata with explicit bounds and common secret redaction."""
    if depth > MAX_METADATA_DEPTH:
        return "[display truncated: maximum metadata depth reached]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return (
            value if value.bit_length() <= 1_024 else "[display omitted: integer exceeds 1024 bits]"
        )
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        clipped = value[:MAX_METADATA_STRING]
        if len(value) > MAX_METADATA_STRING:
            clipped += (
                f" [display truncated: {len(value) - MAX_METADATA_STRING} characters omitted]"
            )
        clipped = URL_CREDENTIALS.sub(r"\1[redacted]@", clipped)
        clipped = SECRET_TEXT.sub(r"\1=[redacted]", clipped)
        return BEARER_TEXT.sub("Bearer [redacted]", clipped)
    if isinstance(value, list):
        clean = [sanitize_json(item, depth + 1) for item in value[:MAX_METADATA_ITEMS]]
        if len(value) > MAX_METADATA_ITEMS:
            clean.append(
                f"[display truncated: {len(value) - MAX_METADATA_ITEMS} of {len(value)} items omitted]"
            )
        return clean
    if isinstance(value, dict):
        clean: dict[str, object] = {}
        for key, item in list(value.items())[:MAX_METADATA_ITEMS]:
            if not isinstance(key, str):
                continue
            lowered = key.casefold().replace(" ", "_")
            if any(part in lowered for part in SENSITIVE_KEY_PARTS):
                clean[key] = "[redacted]"
            else:
                shown_key = key[:200]
                if len(key) > 200:
                    shown_key += " [display key truncated]"
                clean[shown_key] = sanitize_json(item, depth + 1)
        if len(value) > MAX_METADATA_ITEMS:
            clean["__display_truncation__"] = {
                "omitted_items": len(value) - MAX_METADATA_ITEMS,
                "total_items": len(value),
            }
        return clean
    return "[unsupported value]"


def sanitize_config(value: object, depth: int = 0) -> object:
    """Redact sensitive config fields without truncating any config values."""
    if depth > MAX_CONFIG_DEPTH:
        raise ValueError("config exceeds the dashboard display depth bound")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return "[non-finite number]"
        return value
    if isinstance(value, str):
        redacted = URL_CREDENTIALS.sub(r"\1[redacted]@", value)
        redacted = SECRET_TEXT.sub(r"\1=[redacted]", redacted)
        return BEARER_TEXT.sub("Bearer [redacted]", redacted)
    if isinstance(value, list):
        return [sanitize_config(item, depth + 1) for item in value]
    if isinstance(value, dict):
        clean: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("config contains a non-string key")
            lowered = key.casefold().replace(" ", "_")
            clean[key] = (
                "[redacted]"
                if any(part in lowered for part in SENSITIVE_KEY_PARTS)
                else sanitize_config(item, depth + 1)
            )
        return clean
    raise ValueError("config contains a non-JSON value")
