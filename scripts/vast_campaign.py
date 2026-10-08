#!/usr/bin/env python3
"""Staged Vast.ai lifecycle adapter for one receipt-backed campaign attempt.

The direct create path is disabled. The systemd-only run command requires explicit
root admission and owns one receipt-bound instance through collection and teardown.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib
import json
import logging
import os
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from http.client import HTTPMessage
from pathlib import Path, PurePosixPath
from typing import IO, Never, Protocol, cast

if __package__:
    from . import kaggle_vast_prefetch
else:
    import kaggle_vast_prefetch

LOG = logging.getLogger("vast_campaign")
API_ROOT = "https://console.vast.ai"
API_BASE = f"{API_ROOT}/api/v0"
API_V1_BASE = f"{API_ROOT}/api/v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / (
    "wnn-campaign/vast/ledger.json"
)
MAX_CAMPAIGN_USD = Decimal("20.00")
FIRST_COHORT_MAX_USD = Decimal("4.00")
LATER_RELEASE_MAX_USD = Decimal("16.00")
WORKER_STARTUP_TIMEOUT_SECONDS = 1800
MAX_UPLOAD_TIMEOUT_SECONDS = 5400
MAX_WORKER_SETUP_TIMEOUT_SECONDS = 3600
MAX_COLLECTION_ATTEMPTS = 2
FATAL_PREFETCH_STATUSES = frozenset(
    {"invalid_plan", "transport_failed", "invalid_remote_receipt", "unsafe_destination"}
)
DEFAULT_SSH_PUBLIC_KEY = Path.home() / ".ssh/id_ed25519.pub"
TERMINAL_STARTUP_STATUSES = {"deleted", "exited", "failed", "offline", "stopped"}
MIN_GPU_RAM_GIB = Decimal("24")
MIN_HOST_RAM_GIB = Decimal("24")
MIB_PER_GIB = Decimal(1024)
BYTES_PER_GB = Decimal(1_000_000_000)
BYTES_PER_MIB = Decimal(1024 * 1024)
MONEY_KEYS = ("compute_storage_usd", "upload_usd", "download_usd", "cleanup_usd")
ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
LABEL_RE = re.compile(r"wnn-e529-[a-z0-9][a-z0-9-]{6,63}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
INSTANCE_ID_RE = re.compile(r"[1-9][0-9]{0,19}\Z")
SAFE_REMOTE_PART_RE = re.compile(r"[A-Za-z0-9._/-]+\Z")
OFFER_FIELDS = (
    "id",
    "machine_id",
    "gpu_name",
    "gpu_ram",
    "num_gpus",
    "cpu_cores",
    "cpu_cores_effective",
    "cpu_ram",
    "cuda_max_good",
    "disk_space",
    "dph_total",
    "dph_gpu",
    "dph_cpu",
    "dph_ram",
    "dph_disk",
    "reliability",
    "reliability2",
    "verified",
    "verification",
    "is_vm_deverified",
    "rentable",
    "rented",
    "external",
    "type",
    "geolocation",
    "inet_down",
    "inet_down_cost",
    "inet_up",
    "inet_up_cost",
    "storage_cost",
)
INSTANCE_FIELDS = (
    "id",
    "label",
    "actual_status",
    "cur_state",
    "machine_id",
    "gpu_name",
    "num_gpus",
    "disk_space",
    "dph_total",
    "start_date",
    "end_date",
    "total_cost",
    "cost",
)


class CampaignError(RuntimeError):
    """A provider, packet, admission, or durable-ledger invariant failed."""


class WorkerStagingIntegrityError(CampaignError):
    """A staged worker file failed its receipt-bound remote checksum check."""


class WorkerStartupPending(CampaignError):
    """The receipt-owned worker is still booting or its SSH endpoint is not ready."""


class WorkerStartupFailure(CampaignError):
    """The receipt-owned worker cannot become ready before its startup deadline."""


class ApiError(CampaignError):
    """A bounded HTTPS request failed without exposing response bodies or secrets."""

    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        if http_status is not None and (
            isinstance(http_status, bool)
            or not isinstance(http_status, int)
            or not 100 <= http_status <= 599
        ):
            raise ValueError("HTTP status must be an integer from 100 through 599")
        super().__init__(message)
        self.http_status = http_status


class Provider(Protocol):
    """Minimal API surface used by the lifecycle state machine and its tests."""

    def current_user(self) -> dict[str, object]: ...

    def list_instances(self) -> list[dict[str, object]]: ...

    def search_offers(self, criteria: dict[str, object]) -> list[dict[str, object]]: ...

    def search_offer_for_revalidation(
        self, offer: Mapping[str, object], allocated_storage: int
    ) -> list[dict[str, object]]: ...

    def create_instance(self, offer_id: int, payload: dict[str, object]) -> dict[str, object]: ...

    def get_instance(self, instance_id: str) -> dict[str, object]: ...

    def destroy_instance(self, instance_id: str) -> dict[str, object]: ...

    def attach_ssh_key(self, instance_id: str, public_key: str) -> dict[str, object]: ...


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so a bearer credential is never forwarded elsewhere."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


class VastApi:
    """Small HTTPS client for read-only inspection and exact-instance lifecycle calls."""

    def __init__(self, api_key: str, timeout_seconds: float = 15.0) -> None:
        if not api_key or "\n" in api_key or "\r" in api_key:
            raise ValueError("Vast API key must be a non-empty single-line secret")
        if timeout_seconds <= 0 or timeout_seconds > 60:
            raise ValueError("API timeout must be in (0, 60] seconds")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._opener = urllib.request.build_opener(NoRedirectHandler())

    def _request(
        self,
        method: str,
        route: str,
        body: dict[str, object] | None = None,
        *,
        api_version: int = 0,
        query: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        if not route.startswith("/") or "?" in route or "#" in route:
            raise ValueError("API route must be a relative path without query parameters")
        if api_version not in {0, 1}:
            raise ValueError("only Vast API versions 0 and 1 are supported")
        if api_version == 1 and (method != "GET" or route != "/instances/"):
            raise ValueError("the v1 transport is restricted to read-only instance inventory")
        if query is not None and any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in query.items()
        ):
            raise ValueError("API query parameters must contain string keys and values")
        base = API_BASE if api_version == 0 else API_V1_BASE
        url = base + route
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None if body is None else json.dumps(body, sort_keys=True).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                if response.geturl() != url:
                    raise ApiError("Vast API changed the request URL; redirects are disabled")
                raw = response.read(5_000_001)
        except urllib.error.HTTPError as error:
            # Preserve only a validated numeric status before closing the response.
            http_status = error.code
            if (
                isinstance(http_status, bool)
                or not isinstance(http_status, int)
                or not 100 <= http_status <= 599
            ):
                http_status = None
            error.close()
            raise ApiError(
                f"Vast API returned HTTP {http_status} for {method} {route}",
                http_status=http_status,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ApiError(
                f"Vast API {method} {route} failed: {type(error).__name__}; reconcile before retry"
            ) from None
        if len(raw) > 5_000_000:
            raise ApiError("Vast API response exceeded the 5 MB safety limit")
        if not raw:
            return {}
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError("Vast API returned non-JSON response") from None
        if not isinstance(result, dict):
            raise ApiError("Vast API response must be a JSON object")
        return result

    def current_user(self) -> dict[str, object]:
        """Fetch the current account identity and credit balance."""
        return self._request("GET", "/users/current/")

    def list_instances(self) -> list[dict[str, object]]:
        """List every current-account instance through the supported v1 API."""
        query = {
            "select_filters": json.dumps({}),
            "order_by": json.dumps([{"col": "id", "dir": "asc"}]),
            "limit": "25",
        }
        rows: list[dict[str, object]] = []
        seen_tokens: set[str] = set()
        while True:
            page = self._request("GET", "/instances/", api_version=1, query=query)
            page_rows = page.get("instances")
            if not isinstance(page_rows, list) or any(
                not isinstance(row, dict) for row in page_rows
            ):
                raise ApiError("Vast v1 instance response did not contain an instances list")
            rows.extend(cast(list[dict[str, object]], page_rows))
            token = page.get("next_token")
            if token is None or token == "":
                return rows
            if not isinstance(token, str) or token in seen_tokens:
                raise ApiError("Vast v1 instance pagination returned an invalid or repeated token")
            seen_tokens.add(token)
            query["after_token"] = token

    def search_offers(self, criteria: dict[str, object]) -> list[dict[str, object]]:
        """Search verified, rentable, on-demand offers through the documented bundles API."""
        query = dict(criteria)
        for key, required in (
            ("verified", {"eq": True}),
            ("rentable", {"eq": True}),
            ("external", {"eq": False}),
            ("rented", {"eq": False}),
            ("type", "on-demand"),
        ):
            if key in query and query[key] != required:
                raise ValueError(f"offer search cannot relax the required {key}={required!r}")
            query[key] = required
        raw_limit = query.get("limit", 10)
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, str)):
            raise ValueError("offer search limit must be an integer")
        query["limit"] = min(20, max(1, int(raw_limit)))
        result = self._request("POST", "/bundles/", query)
        offers = result.get("offers")
        if not isinstance(offers, list) or any(not isinstance(row, dict) for row in offers):
            raise ApiError("Vast offer response did not contain an offers list")
        return offers

    def search_offer_for_revalidation(
        self, offer: Mapping[str, object], allocated_storage: int
    ) -> list[dict[str, object]]:
        """Recheck a frozen offer by stable machine/GPU attributes and filter its ID locally."""
        if isinstance(allocated_storage, bool) or allocated_storage <= 0:
            raise ValueError("allocated_storage must be a positive integer GB")
        machine_id = offer.get("machine_id")
        gpu_name = offer.get("gpu_name")
        num_gpus = offer.get("num_gpus")
        if (
            isinstance(machine_id, bool)
            or not isinstance(machine_id, (int, str))
            or not isinstance(gpu_name, str)
            or not gpu_name
            or isinstance(num_gpus, bool)
            or not isinstance(num_gpus, (int, float))
        ):
            raise ValueError("frozen machine ID, GPU name, and GPU count are required")
        criteria: dict[str, object] = {
            "machine_id": {"eq": machine_id},
            "gpu_name": {"eq": gpu_name},
            "num_gpus": {"eq": num_gpus},
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "external": {"eq": False},
            "rented": {"eq": False},
            "type": "on-demand",
            "allocated_storage": allocated_storage,
            "limit": 20,
        }
        result = self._request("POST", "/bundles/", criteria)
        offers = result.get("offers")
        if not isinstance(offers, list) or any(not isinstance(row, dict) for row in offers):
            raise ApiError("Vast offer response did not contain an offers list")
        return offers

    def create_instance(self, offer_id: int, payload: dict[str, object]) -> dict[str, object]:
        """Create one instance from one exact on-demand offer."""
        return self._request("PUT", f"/asks/{offer_id}/", payload)

    def get_instance(self, instance_id: str) -> dict[str, object]:
        """Read one instance record by its receipt-bound numeric ID."""
        return self._request("GET", f"/instances/{instance_id}/")

    def destroy_instance(self, instance_id: str) -> dict[str, object]:
        """Destroy one exact instance; callers must prove ownership first."""
        return self._request("DELETE", f"/instances/{instance_id}/")

    def attach_ssh_key(self, instance_id: str, public_key: str) -> dict[str, object]:
        """Attach the controller's public key to one receipt-owned instance."""
        return self._request("POST", f"/instances/{instance_id}/ssh/", {"ssh_key": public_key})


def load_api_key(credential_file: Path | None = None) -> str:
    """Parse the project .env `vastai` value without executing or printing it."""
    path = credential_file or ROOT / ".env"
    if not path.is_file():
        raise CampaignError("Vast API key is unavailable in environment or dotenv file")
    try:
        dotenv = importlib.import_module("dotenv")
    except ImportError as error:
        raise CampaignError(
            "python-dotenv is required to parse the Vast credential file"
        ) from error
    dotenv_values = getattr(dotenv, "dotenv_values", None)
    if not callable(dotenv_values):
        raise CampaignError("python-dotenv does not expose dotenv_values")
    values = dotenv_values(path, encoding="utf-8")
    if not isinstance(values, Mapping):
        raise CampaignError("python-dotenv returned an invalid credential mapping")
    value = values.get("vastai")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise CampaignError("Vast dotenv file contains no `vastai` key")


def load_ssh_public_key(path: Path | None = None) -> tuple[str, str]:
    """Load the existing controller public key without reading or printing its private key."""
    public_path = path or DEFAULT_SSH_PUBLIC_KEY
    private_path = public_path.with_suffix("")
    if public_path.is_symlink() or private_path.is_symlink():
        raise CampaignError("controller SSH key paths must not be symbolic links")
    try:
        public_stat = public_path.stat()
        private_stat = private_path.stat()
        public_key = public_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise CampaignError(f"controller SSH key is unavailable: {type(error).__name__}") from None
    if not stat.S_ISREG(public_stat.st_mode) or not stat.S_ISREG(private_stat.st_mode):
        raise CampaignError("controller SSH key and public key must be regular files")
    if private_stat.st_mode & 0o077:
        raise CampaignError("controller private SSH key permissions must be owner-only")
    if len(public_key) > 16_384 or "\n" in public_key or "\r" in public_key:
        raise CampaignError("controller SSH public key must be one bounded line")
    fields = public_key.split()
    if len(fields) < 2 or fields[0] != "ssh-ed25519":
        raise CampaignError("controller public key must be one OpenSSH ed25519 key")
    try:
        key_blob = base64.b64decode(fields[1], validate=True)
    except ValueError:
        raise CampaignError("controller SSH public key encoding is malformed") from None
    if not key_blob:
        raise CampaignError("controller SSH public key encoding is empty")
    return public_key, hashlib.sha256(public_key.encode("utf-8")).hexdigest()


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp for durable campaign receipts."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON so a root approval binds the exact request semantics."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file incrementally without loading a large packet into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _known_host_fingerprints(contents: str, expected_host: str) -> list[str]:
    """Validate pinned OpenSSH known-host lines and return their sorted SHA-256 IDs."""
    fingerprints: set[str] = set()
    for line in contents.splitlines():
        fields = line.split()
        if len(fields) < 3 or fields[0] != expected_host:
            raise CampaignError("campaign known-hosts file contains an unexpected host entry")
        try:
            blob = base64.b64decode(fields[2], validate=True)
            key_type = blob[4 : 4 + int.from_bytes(blob[:4], "big")].decode("ascii")
        except (ValueError, UnicodeDecodeError, IndexError):
            raise CampaignError("campaign known-hosts file contains a malformed key") from None
        if len(blob) < 5 or key_type != fields[1]:
            raise CampaignError("campaign known-hosts file key type does not match its key blob")
        fingerprint = base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii").rstrip("=")
        fingerprints.add(f"SHA256:{fingerprint}")
    return sorted(fingerprints)


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read JSON object {path}: {type(error).__name__}") from None
    if not isinstance(value, dict):
        raise CampaignError(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _empty_ledger() -> dict[str, object]:
    return {"schema_version": 1, "account_scope_sha256": None, "attempts": {}}


@contextmanager
def ledger_transaction(path: Path) -> Iterator[dict[str, object]]:
    """Serialize one durable-ledger read/modify/write across controller processes."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_suffix(path.suffix + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "r+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            state = _empty_ledger() if not path.exists() else _read_object(path)
            attempts = state.get("attempts")
            if state.get("schema_version") != 1 or not isinstance(attempts, dict):
                raise CampaignError("Vast ledger schema is invalid; preserve it for review")
            yield state
            _atomic_json(path, state)
    finally:
        if os.path.exists(lock_path):
            os.chmod(lock_path, 0o600)


def _attempts(state: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = state.get("attempts")
    if not isinstance(rows, dict):
        raise CampaignError("Vast ledger attempts must be an object")
    if any(not isinstance(value, dict) for value in rows.values()):
        raise CampaignError("Vast ledger contains a malformed attempt; admission is closed")
    return cast(dict[str, dict[str, object]], rows)


def _decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise CampaignError(f"{field} must be a nonnegative USD decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise CampaignError(f"{field} is not a valid USD amount") from None
    if not result.is_finite() or result < 0:
        raise CampaignError(f"{field} must be finite and nonnegative")
    return result


def _money(value: object, field: str) -> Decimal:
    result = _decimal(value, field)
    if result.quantize(Decimal("0.01")) != result:
        raise CampaignError(f"{field} must be finite, nonnegative, and cent-exact")
    return result


def _money_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise CampaignError(f"{field} must be a lowercase SHA-256")
    return value


def _account_scope(user: Mapping[str, object]) -> str:
    user_id = user.get("id")
    if isinstance(user_id, bool) or not isinstance(user_id, (str, int)) or not str(user_id):
        raise CampaignError("Vast current-user response lacks a stable account ID")
    return hashlib.sha256(f"vast-account:{user_id}".encode()).hexdigest()


def _balance(user: Mapping[str, object]) -> Decimal:
    # The live API includes `credit` alongside the documented `balance` field.
    # Vast's own CLI labels `credit` as account credit. Prefer it when present;
    # malformed credit must not silently fall back to another value.
    if "credit" in user:
        return _decimal(user["credit"], "Vast available credit")
    return _decimal(user.get("balance"), "Vast account balance")


def _wrapped_instance(response: Mapping[str, object]) -> dict[str, object]:
    value = response.get("instances", response)
    if not isinstance(value, dict):
        raise CampaignError("Vast instance response did not contain an instance object")
    return value


def _instance_id(instance: Mapping[str, object]) -> str:
    value = instance.get("id")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise CampaignError("Vast instance record lacks a numeric ID")
    result = str(value)
    if INSTANCE_ID_RE.fullmatch(result) is None:
        raise CampaignError("Vast instance record contains an invalid ID")
    return result


def _sanitized_instance(instance: Mapping[str, object]) -> dict[str, object]:
    sanitized = {
        key: instance.get(key) for key in INSTANCE_FIELDS if key in instance and key != "id"
    }
    label = instance.get("label")
    if isinstance(label, str) and LABEL_RE.fullmatch(label) is None:
        sanitized.pop("label", None)
        sanitized["label_sha256"] = hashlib.sha256(label.encode()).hexdigest()
    if "id" in instance:
        sanitized["instance_id_sha256"] = hashlib.sha256(str(instance["id"]).encode()).hexdigest()
    return sanitized


def _offer_snapshot(
    offer: Mapping[str, object], *, query_type: str | None = None
) -> dict[str, object]:
    query_type = query_type or cast(str | None, offer.get("query_type"))
    if query_type not in {"on-demand", "ondemand"}:
        raise CampaignError("offer must be bound to a verified on-demand query")
    snapshot = {key: offer.get(key) for key in OFFER_FIELDS}
    offer_id = snapshot.get("id")
    if isinstance(offer_id, bool) or not isinstance(offer_id, (str, int)):
        raise CampaignError("offer is missing its numeric ID")
    snapshot["id"] = int(offer_id)
    verification = snapshot.get("verification")
    deverified = snapshot.get("is_vm_deverified")
    if deverified is not None and not isinstance(deverified, bool):
        raise CampaignError("offer is_vm_deverified status must be boolean when present")
    if verification is not None:
        verified = verification == "verified" and deverified is not True
    else:
        verified = snapshot.get("verified") is True and deverified is not True
    snapshot["verified_for_admission"] = verified
    if not verified or snapshot["rentable"] is not True:
        raise CampaignError("Vast offer must be verified and rentable")
    if snapshot.get("rented") is True or snapshot.get("external") is True:
        raise CampaignError("Vast offer must be unrented and internally hosted")
    rental_type = snapshot.get("type")
    if rental_type not in (None, "ondemand", "on-demand"):
        raise CampaignError("only on-demand offers are admitted for the first campaign")
    snapshot["query_type"] = "on-demand"
    if snapshot.get("machine_id") is None:
        raise CampaignError("offer must expose a machine_id for ownership reconciliation")
    gpu_count = _decimal(snapshot.get("num_gpus"), "offer num_gpus")
    gpu_ram_mib = _decimal(snapshot.get("gpu_ram"), "offer gpu_ram_mib")
    host_ram_mib = _decimal(snapshot.get("cpu_ram"), "offer cpu_ram_mib")
    if gpu_count != 1:
        raise CampaignError("initial paid lane requires exactly one GPU worker")
    if gpu_ram_mib / MIB_PER_GIB < MIN_GPU_RAM_GIB:
        raise CampaignError("verified offer must provide at least 24 GiB GPU memory")
    if host_ram_mib / MIB_PER_GIB < MIN_HOST_RAM_GIB:
        raise CampaignError("verified offer must provide at least 24 GiB host memory")
    _decimal(snapshot.get("dph_total"), "offer dph_total")
    for field in ("inet_down_cost", "inet_up_cost", "storage_cost"):
        _decimal(snapshot.get(field), f"offer {field}")
    snapshot["gpu_ram_gib"] = format((gpu_ram_mib / MIB_PER_GIB).quantize(Decimal("0.001")), "f")
    snapshot["cpu_ram_gib"] = format((host_ram_mib / MIB_PER_GIB).quantize(Decimal("0.001")), "f")
    return snapshot


def _offer_id(snapshot: Mapping[str, object]) -> int:
    value = snapshot.get("id")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CampaignError("offer ID must be a positive integer")
    return value


def _valid_remote_relative(value: object) -> str:
    if not isinstance(value, str) or SAFE_REMOTE_PART_RE.fullmatch(value) is None:
        raise CampaignError("artifact path contains unsupported characters")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise CampaignError("artifact path must be a safe relative POSIX path")
    return path.as_posix()


def _valid_remote_root(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise CampaignError("remote output directory must be absolute")
    normalized = PurePosixPath(value)
    if ".." in normalized.parts or SAFE_REMOTE_PART_RE.fullmatch(value.lstrip("/")) is None:
        raise CampaignError("remote output directory contains unsafe path components")
    return normalized.as_posix().rstrip("/")


def _validate_intent(intent: Mapping[str, object], path: Path) -> dict[str, object]:
    if intent.get("schema_version") != 1:
        raise CampaignError("unsupported frozen intent schema")
    attempt = intent.get("attempt_id")
    label = intent.get("campaign_label")
    if not isinstance(attempt, str) or ATTEMPT_RE.fullmatch(attempt) is None:
        raise CampaignError("intent attempt_id is invalid")
    if not isinstance(label, str) or LABEL_RE.fullmatch(label) is None:
        raise CampaignError("campaign_label must be a unique wnn-e529-* label")
    _sha256(intent.get("account_scope_sha256"), "intent account scope")
    packet_sha = _sha256(intent.get("packet_sha256"), "intent packet_sha256")
    packet_path_value = intent.get("packet_path")
    if not isinstance(packet_path_value, str):
        raise CampaignError("intent must identify the exact frozen packet file")
    packet_path = Path(packet_path_value).resolve(strict=True)
    if not packet_path.is_relative_to(ROOT):
        raise CampaignError("frozen packet must remain inside the project root")
    if not packet_path.is_file() or file_sha256(packet_path) != packet_sha:
        raise CampaignError("frozen packet file is missing or its SHA-256 changed")
    inputs_value = intent.get("worker_inputs", [])
    if not isinstance(inputs_value, list) or any(not isinstance(row, dict) for row in inputs_value):
        raise CampaignError("worker_inputs must be a list of frozen input file records")
    worker_inputs: list[dict[str, object]] = []
    remote_input_paths: set[str] = set()
    total_upload_bytes = packet_path.stat().st_size
    for row in inputs_value:
        if set(row) != {"source_path", "source_sha256", "size_bytes", "remote_path"}:
            raise CampaignError(
                "worker input record must contain only its path, hash, size and target"
            )
        source_value = row.get("source_path")
        if not isinstance(source_value, str) or Path(source_value).is_absolute():
            raise CampaignError("worker input source_path must be project-relative")
        if ".." in Path(source_value).parts:
            raise CampaignError("worker input source_path cannot traverse parent directories")
        source_candidate = ROOT / source_value
        if any(
            part.lower().startswith(".env")
            or "api_key" in part.lower()
            or "credential" in part.lower()
            or part.lower() in {"id_rsa", "id_ed25519", "secret", "secrets"}
            for part in Path(source_value).parts
        ):
            raise CampaignError("worker input path is credential-like and cannot be uploaded")
        if source_candidate.is_symlink():
            raise CampaignError("worker input source cannot be a symbolic link")
        source_path = source_candidate.resolve(strict=True)
        if not source_path.is_relative_to(ROOT) or not source_path.is_file():
            raise CampaignError("worker input source must be a regular project file")
        size_bytes = row.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            raise CampaignError("worker input size_bytes must be a positive integer")
        if source_path.stat().st_size != size_bytes:
            raise CampaignError("worker input size differs from its frozen identity receipt")
        source_sha256 = _sha256(row.get("source_sha256"), "worker input source_sha256")
        remote_path = _valid_remote_relative(row.get("remote_path"))
        if remote_path in remote_input_paths:
            raise CampaignError("worker inputs contain duplicate remote paths")
        remote_input_paths.add(remote_path)
        total_upload_bytes += size_bytes
        worker_inputs.append(
            {
                "source_path": source_value,
                "source_sha256": source_sha256,
                "size_bytes": size_bytes,
                "remote_path": remote_path,
            }
        )
    max_upload_bytes = intent.get("max_upload_bytes")
    if (
        isinstance(max_upload_bytes, bool)
        or not isinstance(max_upload_bytes, int)
        or max_upload_bytes <= 0
    ):
        raise CampaignError("intent must set a positive total upload byte ceiling")
    if total_upload_bytes > max_upload_bytes:
        raise CampaignError("frozen packet and inputs exceed the total upload byte ceiling")
    upload_timeout = intent.get("upload_timeout_seconds")
    if isinstance(upload_timeout, bool) or not isinstance(upload_timeout, int):
        raise CampaignError("intent upload_timeout_seconds must be an integer")
    if upload_timeout <= 0 or upload_timeout > MAX_UPLOAD_TIMEOUT_SECONDS:
        raise CampaignError(
            f"intent upload_timeout_seconds must be in (0, {MAX_UPLOAD_TIMEOUT_SECONDS}]"
        )
    offer = intent.get("offer")
    if not isinstance(offer, dict):
        raise CampaignError("intent must contain the exact selected offer snapshot")
    snapshot = _offer_snapshot(offer, query_type=cast(str | None, offer.get("query_type")))
    reliability_floor: Decimal | None = None
    if "min_offer_reliability" in intent:
        reliability_floor = _decimal(intent.get("min_offer_reliability"), "min_offer_reliability")
        if reliability_floor <= 0 or reliability_floor > 1:
            raise CampaignError("min_offer_reliability must be in (0, 1]")
        for field in ("reliability", "reliability2"):
            frozen_reliability = _decimal(snapshot.get(field), f"frozen offer {field}")
            if frozen_reliability < reliability_floor:
                raise CampaignError(f"frozen offer {field} is below the admitted reliability floor")
    payload = intent.get("create_payload")
    if not isinstance(payload, dict):
        raise CampaignError("intent must contain one frozen instance request payload")
    allowed = {
        "client_id",
        "image",
        "disk",
        "runtype",
        "label",
        "onstart",
        "cancel_unavail",
        "env",
        "use_jupyter_lab",
        "python_utf8",
        "lang_utf8",
        "jupyter_dir",
    }
    if set(payload) - allowed:
        raise CampaignError("instance payload contains unsupported or persistent-resource fields")
    if payload.get("client_id", "me") != "me":
        raise CampaignError("instance client_id must remain in the authenticated account scope")
    if payload.get("label") != label or payload.get("runtype") != "ssh_direc ssh_proxy":
        raise CampaignError("instance label and SSH direct runtype must match the frozen intent")
    if payload.get("cancel_unavail") is not True:
        raise CampaignError("instance request must fail rather than wait for an unavailable offer")
    image = payload.get("image")
    if not isinstance(image, str) or re.search(r"@sha256:[0-9a-f]{64}\Z", image) is None:
        raise CampaignError("container image must be pinned by immutable SHA-256 digest")
    disk = payload.get("disk")
    if isinstance(disk, bool) or not isinstance(disk, int) or disk <= 0:
        raise CampaignError("instance disk size must be a positive integer GB")
    offer_disk = _decimal(snapshot.get("disk_space"), "offer disk_space")
    if Decimal(disk) > offer_disk:
        raise CampaignError("frozen disk request exceeds selected offer capacity")
    if not isinstance(payload.get("onstart"), str) or not payload["onstart"].strip():
        raise CampaignError("frozen instance payload must contain its bounded onstart command")
    if payload["onstart"] != "sleep infinity":
        raise CampaignError("Vast worker image must idle until the receipt-bound packet is staged")
    if "env" in payload:
        worker_env = payload["env"]
        if not isinstance(worker_env, dict):
            raise CampaignError("instance env must be a JSON object")
        if any(
            any(secret_word in str(key).lower() for secret_word in ("vast", "api_key", "secret"))
            for key in worker_env
        ):
            raise CampaignError("instance env cannot contain Vast credentials or secret fields")
    request_hash = canonical_sha256({"offer_id": _offer_id(snapshot), "payload": payload})
    expected_files = intent.get("expected_artifacts")
    if not isinstance(expected_files, list) or not expected_files:
        raise CampaignError("intent must list expected output files")
    normalized_files = [_valid_remote_relative(value) for value in expected_files]
    if len(set(normalized_files)) != len(normalized_files):
        raise CampaignError("intent expected_artifacts contains duplicate paths")
    runtime = intent.get("max_runtime_seconds")
    if isinstance(runtime, bool) or not isinstance(runtime, int) or runtime <= 0:
        raise CampaignError("intent must set a positive max_runtime_seconds")
    setup_timeout = intent.get("worker_setup_timeout_seconds", 0)
    if (
        isinstance(setup_timeout, bool)
        or not isinstance(setup_timeout, int)
        or setup_timeout < 0
        or setup_timeout > MAX_WORKER_SETUP_TIMEOUT_SECONDS
    ):
        raise CampaignError(
            "intent worker_setup_timeout_seconds must be in "
            f"[0, {MAX_WORKER_SETUP_TIMEOUT_SECONDS}]"
        )
    max_bytes = intent.get("max_artifact_bytes")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise CampaignError("intent must set a positive artifact collection byte ceiling")
    worker_ram = _decimal(intent.get("max_worker_host_ram_gb"), "max_worker_host_ram_gb")
    host_ram_mib = _decimal(snapshot.get("cpu_ram"), "offer cpu_ram_mib")
    worker_ram_bytes = worker_ram * BYTES_PER_GB
    host_ram_bytes = host_ram_mib * BYTES_PER_MIB
    if worker_ram <= 0 or worker_ram_bytes > host_ram_bytes * Decimal("0.8"):
        raise CampaignError(
            "worker host RAM ceiling must preserve at least 20% selected-host headroom"
        )
    return {
        "attempt_id": attempt,
        "campaign_label": label,
        "account_scope_sha256": intent["account_scope_sha256"],
        "packet_sha256": packet_sha,
        "packet_path": str(packet_path),
        "offer": snapshot,
        "offer_id": _offer_id(snapshot),
        "offer_snapshot_sha256": canonical_sha256(snapshot),
        "min_offer_reliability": (
            None if reliability_floor is None else format(reliability_floor.normalize(), "f")
        ),
        "create_payload": payload,
        "request_sha256": request_hash,
        "intent_sha256": file_sha256(path),
        "worker_inputs": worker_inputs,
        "worker_inputs_sha256": canonical_sha256(worker_inputs),
        "max_upload_bytes": max_upload_bytes,
        "upload_timeout_seconds": upload_timeout,
        "expected_artifacts": normalized_files,
        "remote_output_dir": _valid_remote_root(intent.get("remote_output_dir")),
        "max_runtime_seconds": runtime,
        "worker_setup_timeout_seconds": setup_timeout,
        "max_artifact_bytes": max_bytes,
        "max_worker_host_ram_gb": worker_ram,
    }


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise CampaignError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise CampaignError(f"{field} must be an ISO-8601 UTC timestamp") from None
    if parsed.tzinfo is None:
        raise CampaignError(f"{field} must include its timezone")
    return parsed.astimezone(UTC)


def _parse_provider_time(value: object) -> datetime | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return _parse_time(value, "provider timestamp")
        except CampaignError:
            return None
    return None


def _compare_revalidated_offer(
    frozen: Mapping[str, object],
    live: Mapping[str, object],
    *,
    reliability_floor: object,
) -> dict[str, dict[str, object]]:
    """Require exact offer identity/resources, with only floor-bounded reliability drift."""
    if reliability_floor is None:
        if canonical_sha256(dict(live)) != canonical_sha256(dict(frozen)):
            raise CampaignError("fresh offer differs from the frozen, root-approved offer snapshot")
        return {}

    floor = _decimal(reliability_floor, "min_offer_reliability")
    if floor <= 0 or floor > 1:
        raise CampaignError("min_offer_reliability must be in (0, 1]")
    telemetry_fields = {"reliability", "reliability2"}
    if set(live) != set(frozen):
        raise CampaignError("fresh offer fields differ from the frozen offer snapshot")
    for key in frozen:
        if key not in telemetry_fields and live[key] != frozen[key]:
            raise CampaignError("fresh offer differs from the frozen, root-approved offer snapshot")

    allowed_diffs: dict[str, dict[str, object]] = {}
    for field in sorted(telemetry_fields):
        frozen_value = _decimal(frozen.get(field), f"frozen offer {field}")
        live_value = _decimal(live.get(field), f"fresh offer {field}")
        if live_value < floor:
            raise CampaignError(f"fresh offer {field} is below the admitted reliability floor")
        if live_value != frozen_value:
            allowed_diffs[field] = {"frozen": frozen[field], "live": live[field]}
    return allowed_diffs


def _require_fresh_root_window(
    approval: Mapping[str, object], dispatch_receipt: Mapping[str, object]
) -> None:
    """Recheck short-lived launch authority after read-only provider preflight."""
    now = datetime.now(UTC)
    created = _parse_time(approval.get("created_utc"), "root admission created_utc")
    expires = _parse_time(approval.get("expires_utc"), "root admission expires_utc")
    dispatch_created = _parse_time(
        dispatch_receipt.get("created_utc"), "dispatch receipt created_utc"
    )
    if created > now or expires <= now or expires <= created:
        raise CampaignError("root admission expired during provider preflight")
    if expires - created > timedelta(seconds=120):
        raise CampaignError("root admission validity exceeds 120 seconds")
    if dispatch_created > now or now - dispatch_created > timedelta(seconds=120):
        raise CampaignError("dispatcher pre-submit intent expired during provider preflight")
    vast_value = approval.get("vast_admission")
    if not isinstance(vast_value, dict):
        raise CampaignError("root admission lost its Vast-specific deadline")
    hard_stop = _parse_time(vast_value.get("hard_stop_utc"), "hard_stop_utc")
    if hard_stop <= now:
        raise CampaignError("Vast paid hard-stop elapsed during provider preflight")


def _validate_admission(
    approval: Mapping[str, object],
    intent: Mapping[str, object],
    normalized: Mapping[str, object],
    dispatch_receipt: Mapping[str, object],
    admission_path: Path,
) -> tuple[Decimal, Decimal, str, dict[str, str], Decimal | None, str | None]:
    required_text = (
        "owner",
        "authorized",
        "attempt_id",
        "job_sha256",
        "created_utc",
        "expires_utc",
        "source_hashes",
        "vast_admission",
    )
    missing = [key for key in required_text if key not in approval]
    if missing:
        raise CampaignError(f"root admission is missing required fields: {', '.join(missing)}")
    if approval.get("owner") != "root" or approval.get("authorized") is not True:
        raise CampaignError("an explicit root admission is required")
    if approval.get("attempt_id") != normalized["attempt_id"]:
        raise CampaignError("root admission attempt_id differs from frozen intent")
    if dispatch_receipt.get("attempt_id") != normalized["attempt_id"]:
        raise CampaignError("dispatcher pre-submit receipt attempt_id differs")
    dispatch_status = dispatch_receipt.get("status")
    if dispatch_receipt.get("lane") != "vast" or dispatch_status not in {
        "launch_uncertain",
        "submitted",
    }:
        raise CampaignError("Vast mutation requires the root dispatcher's persisted launch intent")
    if dispatch_receipt.get("job_sha256") != approval.get("job_sha256"):
        raise CampaignError("dispatcher pre-submit receipt is not bound to root admission job hash")
    if dispatch_status == "launch_uncertain" and "returncode" in dispatch_receipt:
        raise CampaignError("uncertain dispatcher receipt cannot contain a completed return code")
    if dispatch_status == "submitted":
        returncode = dispatch_receipt.get("returncode")
        if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode != 0:
            raise CampaignError("submitted dispatcher receipt must record a successful return code")
        observed = _parse_time(dispatch_receipt.get("observed_utc"), "dispatch observed_utc")
        observed_now = datetime.now(UTC)
        if observed > observed_now or observed_now - observed > timedelta(seconds=120):
            raise CampaignError(
                "submitted dispatcher receipt is outside the fresh 120-second window"
            )
    payload_value = normalized.get("create_payload")
    offer_value = normalized.get("offer")
    runtime_value = normalized.get("max_runtime_seconds")
    if not isinstance(payload_value, dict) or not isinstance(offer_value, dict):
        raise CampaignError("validated frozen intent lost its payload or offer snapshot")
    if isinstance(runtime_value, bool) or not isinstance(runtime_value, int):
        raise CampaignError("validated frozen intent lost its bounded runtime")
    payload = cast(dict[str, object], payload_value)
    offer = cast(dict[str, object], offer_value)
    vast = approval.get("vast_admission")
    if not isinstance(vast, dict):
        raise CampaignError("root admission lacks its Vast-specific exact-action authorization")
    required_vast = (
        "decision",
        "authorized_actions",
        "attempt_id",
        "campaign_label",
        "intent_sha256",
        "packet_sha256",
        "request_sha256",
        "account_scope_sha256",
        "offer_snapshot_sha256",
        "offer_id",
        "payment_policy_receipt_sha256",
        "cohort_stage",
        "max_all_in_usd",
        "minimum_live_balance_usd",
        "total_campaign_ceiling_usd",
        "max_runtime_seconds",
        "max_upload_bytes",
        "upload_timeout_seconds",
        "worker_inputs_sha256",
        "max_disk_gb",
        "max_worker_host_ram_gb",
        "hard_stop_utc",
        "cleanup_deadline_utc",
    )
    if normalized.get("min_offer_reliability") is not None:
        required_vast += ("min_offer_reliability",)
    missing_vast = [key for key in required_vast if key not in vast]
    if missing_vast:
        raise CampaignError(f"Vast-specific root admission is missing: {', '.join(missing_vast)}")
    if vast.get("decision") != "admit_one_vast_instance":
        raise CampaignError("Vast-specific root decision must admit one instance")
    if vast.get("authorized_actions") != ["create_one_instance"]:
        raise CampaignError("Vast admission must authorize only create_one_instance")
    if vast.get("attempt_id") != normalized["attempt_id"]:
        raise CampaignError("Vast admission attempt_id differs from frozen intent")
    exact_pairs = (
        ("campaign_label", normalized["campaign_label"]),
        ("intent_sha256", normalized["intent_sha256"]),
        ("packet_sha256", normalized["packet_sha256"]),
        ("request_sha256", normalized["request_sha256"]),
        ("account_scope_sha256", normalized["account_scope_sha256"]),
        ("offer_snapshot_sha256", normalized["offer_snapshot_sha256"]),
        ("offer_id", _offer_id(offer)),
        ("max_runtime_seconds", normalized["max_runtime_seconds"]),
        ("max_upload_bytes", normalized["max_upload_bytes"]),
        ("upload_timeout_seconds", normalized["upload_timeout_seconds"]),
        ("worker_inputs_sha256", normalized["worker_inputs_sha256"]),
        ("max_disk_gb", payload["disk"]),
    )
    if normalized.get("min_offer_reliability") is not None:
        exact_pairs += (("min_offer_reliability", normalized["min_offer_reliability"]),)
    if normalized["worker_setup_timeout_seconds"] > 0:
        exact_pairs += (
            (
                "worker_setup_timeout_seconds",
                normalized["worker_setup_timeout_seconds"],
            ),
        )
    for key, expected in exact_pairs:
        if vast.get(key) != expected:
            raise CampaignError(f"Vast admission {key} does not match frozen intent")
    if vast.get("specific_unverified_offer_exception") is not None:
        raise CampaignError("this controller admits verified on-demand offers only")
    admitted_worker_ram = _decimal(vast.get("max_worker_host_ram_gb"), "max_worker_host_ram_gb")
    if admitted_worker_ram != normalized["max_worker_host_ram_gb"]:
        raise CampaignError("Vast admission max_worker_host_ram_gb differs from frozen intent")
    _sha256(vast["payment_policy_receipt_sha256"], "payment policy receipt")
    root_receipt = file_sha256(admission_path)
    created = _parse_time(approval["created_utc"], "dispatcher admission created_utc")
    expires = _parse_time(approval["expires_utc"], "dispatcher admission expires_utc")
    now = datetime.now(UTC)
    if created > now or expires <= now or expires <= created:
        raise CampaignError("root dispatcher admission is not currently valid")
    if expires - created > timedelta(seconds=120):
        raise CampaignError("root dispatcher admission validity exceeds 120 seconds")
    if dispatch_receipt.get("created_utc") is None:
        raise CampaignError("dispatcher pre-submit receipt has no creation time")
    dispatch_created = _parse_time(dispatch_receipt["created_utc"], "dispatch created_utc")
    if dispatch_created > now or now - dispatch_created > timedelta(seconds=120):
        raise CampaignError("dispatcher pre-submit intent is older than 120 seconds")
    hard_stop = _parse_time(vast["hard_stop_utc"], "hard_stop_utc")
    cleanup_deadline = _parse_time(vast["cleanup_deadline_utc"], "cleanup_deadline_utc")
    if hard_stop <= now or cleanup_deadline < hard_stop:
        raise CampaignError("Vast admission hard-stop and cleanup deadlines are invalid")
    if hard_stop - now > timedelta(seconds=runtime_value + 1800):
        raise CampaignError("Vast paid-idle deadline exceeds runtime plus bounded startup margin")
    if cleanup_deadline - hard_stop > timedelta(seconds=1800):
        raise CampaignError("Vast teardown retry window cannot exceed 30 minutes after hard stop")
    source_hashes = approval.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise CampaignError("root admission must bind exact source hashes")
    for relative, digest in source_hashes.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise CampaignError("root admission source hash map is malformed")
        source_path = (ROOT / relative).resolve(strict=True)
        if not source_path.is_relative_to(ROOT) or file_sha256(source_path) != digest:
            raise CampaignError(f"root-admitted source changed: {relative}")
    if vast.get("intent_sha256") != normalized["intent_sha256"]:
        raise CampaignError("root admission does not bind the frozen packet intent")
    stage = vast.get("cohort_stage")
    ceiling = _money(vast["max_all_in_usd"], "root max_all_in_usd")
    campaign_ceiling = _money(vast["total_campaign_ceiling_usd"], "root total_campaign_ceiling_usd")
    if campaign_ceiling > MAX_CAMPAIGN_USD or campaign_ceiling <= 0:
        raise CampaignError("root campaign ceiling must be positive and no greater than $20")
    if ceiling <= 0 or ceiling > campaign_ceiling:
        raise CampaignError("attempt all-in ceiling exceeds the campaign ceiling")
    if stage == "first_cohort":
        if ceiling > FIRST_COHORT_MAX_USD:
            raise CampaignError("first Vast cohort is limited to $4 all-in")
        if vast.get("evidence_release_receipt_sha256") is not None:
            raise CampaignError("first cohort must not use a later evidence release")
        release = None
        release_receipt = None
    elif stage == "parallel_authorized":
        authority_relative = "specs/adr/0015-flexible-credit-bounded-parallel-compute.md"
        authority_hash = file_sha256(ROOT / authority_relative)
        if vast.get("spending_authority_sha256") != authority_hash:
            raise CampaignError("parallel cohort lacks the exact spending authority")
        if source_hashes.get(authority_relative) != authority_hash:
            raise CampaignError("parallel spending authority is not source-bound")
        if vast.get("evidence_release_receipt_sha256") is not None:
            raise CampaignError("parallel cohort must not consume an evidence release")
        release = None
        release_receipt = None
    elif stage == "evidence_released":
        if ceiling > LATER_RELEASE_MAX_USD:
            raise CampaignError("later Vast evidence releases are limited to the remaining $16")
        release_receipt = _sha256(
            vast.get("evidence_release_receipt_sha256"), "evidence release receipt"
        )
        release = _money(vast.get("evidence_released_usd"), "evidence_released_usd")
        if release < ceiling or release > LATER_RELEASE_MAX_USD:
            raise CampaignError("later-cohort ceiling exceeds its explicit evidence release")
    else:
        raise CampaignError(
            "cohort_stage must be first_cohort, parallel_authorized or evidence_released"
        )
    balance_floor = _money(vast["minimum_live_balance_usd"], "minimum_live_balance_usd")
    costs = vast.get("cost_breakdown_usd")
    if not isinstance(costs, dict) or set(costs) != set(MONEY_KEYS):
        raise CampaignError(f"root admission cost breakdown must contain exactly {MONEY_KEYS}")
    parsed_costs = {key: _money(costs[key], key) for key in MONEY_KEYS}
    if sum(parsed_costs.values(), Decimal("0.00")) != ceiling:
        raise CampaignError(
            "all-in ceiling must equal the complete compute/storage/transfer/cleanup sum"
        )
    rate = _decimal(offer.get("dph_total"), "offer dph_total")
    storage_rate = _decimal(offer.get("storage_cost"), "offer storage_cost")
    upload_rate = _decimal(offer.get("inet_down_cost"), "offer inet_down_cost")
    download_rate = _decimal(offer.get("inet_up_cost"), "offer inet_up_cost")
    disk_gb = _decimal(payload.get("disk"), "create_payload disk")
    bounded_lifetime_hours = Decimal(str((cleanup_deadline - created).total_seconds())) / Decimal(
        3600
    )
    minimum_compute_storage = (
        rate * bounded_lifetime_hours
        + disk_gb * storage_rate * bounded_lifetime_hours / Decimal(30 * 24)
    ).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if parsed_costs["compute_storage_usd"] < minimum_compute_storage:
        raise CampaignError(
            "cost envelope under-reserves the full bounded compute and storage lifetime"
        )
    minimum_upload = (
        Decimal(normalized["max_upload_bytes"]) * upload_rate / BYTES_PER_GB * Decimal(2)
    ).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    minimum_download = (
        Decimal(normalized["max_artifact_bytes"]) * download_rate / BYTES_PER_GB
    ).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if parsed_costs["upload_usd"] < minimum_upload:
        raise CampaignError("cost envelope under-reserves the admitted maximum upload bytes")
    if parsed_costs["download_usd"] < minimum_download:
        raise CampaignError("cost envelope under-reserves the admitted maximum download bytes")
    if balance_floor < ceiling:
        raise CampaignError("minimum verified account balance is below the attempt ceiling")
    return (
        ceiling,
        balance_floor,
        root_receipt,
        {key: _money_text(value) for key, value in parsed_costs.items()},
        release,
        release_receipt,
    )


def _ledger_account(state: dict[str, object], account_scope: str) -> None:
    current = state.get("account_scope_sha256")
    if current is None:
        state["account_scope_sha256"] = account_scope
    elif current != account_scope:
        raise CampaignError("Vast ledger belongs to a different account scope; do not adopt it")


def _committed_usd(rows: Mapping[str, Mapping[str, object]]) -> Decimal:
    total = Decimal("0.00")
    for row in rows.values():
        if row.get("status") == "settled":
            total += _money(row.get("actual_all_in_usd"), "settled actual_all_in_usd")
        else:
            total += _money(row.get("reserved_all_in_usd"), "reserved_all_in_usd")
    return total


def _validate_offer_criteria(criteria: Mapping[str, object]) -> dict[str, object]:
    protected = {"verified", "rentable", "external", "rented", "type"}
    if protected & criteria.keys():
        raise CampaignError("criteria may not override verified, rentable, or on-demand filters")
    query = dict(criteria)
    minimum_mib = int(MIN_GPU_RAM_GIB * MIB_PER_GIB)
    for field in ("gpu_ram", "cpu_ram"):
        requested = query.get(field)
        if requested is None:
            query[field] = {"gte": minimum_mib}
            continue
        if not isinstance(requested, dict):
            raise CampaignError(f"{field} must use a Vast numeric filter object")
        conditions = dict(requested)
        equality = conditions.get("eq")
        if equality is not None and _decimal(equality, f"{field} equality") < minimum_mib:
            raise CampaignError(f"{field} must require at least {minimum_mib} MiB (24 GiB)")
        lower_bound = conditions.get("gte")
        if lower_bound is None or _decimal(lower_bound, f"{field} minimum") < minimum_mib:
            conditions["gte"] = minimum_mib
        query[field] = conditions
    protected = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "external": {"eq": False},
        "rented": {"eq": False},
        "type": "on-demand",
    }
    for field, required in protected.items():
        if field in query and query[field] != required:
            raise CampaignError(f"offer search cannot relax the required {field} filter")
        query[field] = required
    allocated_storage = query.get("allocated_storage")
    if _decimal(allocated_storage, "allocated_storage") <= 0:
        raise CampaignError("offer search must bind allocated_storage in GB to the worker disk")
    raw_limit = criteria.get("limit", 10)
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, str)):
        raise CampaignError("offer search limit must be an integer") from None
    try:
        query["limit"] = min(20, max(1, int(raw_limit)))
    except ValueError:
        raise CampaignError("offer search limit must be an integer") from None
    return query


def inspect_account(api: Provider) -> dict[str, object]:
    """Return a private-data-minimized account balance and instance inventory."""
    user = api.current_user()
    scope = _account_scope(user)
    balance = _balance(user)
    instances = api.list_instances()
    sanitized = [_sanitized_instance(instance) for instance in instances]
    return {
        "observed_at_utc": utc_now(),
        "account_scope_sha256": scope,
        "available_credit_usd": _money_text(balance),
        "available_credit_field": "credit" if "credit" in user else "balance",
        "instance_count": len(sanitized),
        "instances": sanitized,
        "source": "Vast read-only current-user and instance-list endpoints",
    }


def search_verified_offers(api: Provider, criteria: dict[str, object]) -> list[dict[str, object]]:
    """Search current verified rentable on-demand offers and return sanitized snapshots."""
    query = _validate_offer_criteria(criteria)
    rows = api.search_offers(query)
    result: list[dict[str, object]] = []
    for row in rows:
        verification = row.get("verification")
        if (
            row.get("is_vm_deverified") is True
            or (verification is not None and verification != "verified")
            or (verification is None and row.get("verified") is not True)
            or row.get("rentable") is not True
            or row.get("rented") is True
            or row.get("external") is True
            or row.get("type") not in (None, "ondemand", "on-demand")
        ):
            # The bundles endpoint can return rows that do not satisfy every
            # requested filter; omit explicit non-candidates and validate all
            # remaining snapshot fields strictly below.
            continue
        snapshot = _offer_snapshot(row, query_type="on-demand")
        result.append({"offer": snapshot, "offer_snapshot_sha256": canonical_sha256(snapshot)})
    return result


class VastCampaign:
    """Receipt-backed controller for one account-scoped, explicitly admitted attempt."""

    def __init__(self, api: Provider, ledger_path: Path) -> None:
        self.api = api
        self.ledger_path = ledger_path

    def create_one(
        self, intent_path: Path, admission_path: Path, dispatch_receipt_path: Path
    ) -> dict[str, object]:
        """Keep direct creation disabled; admitted work must use the supervised run path."""
        del intent_path, admission_path, dispatch_receipt_path
        raise CampaignError(
            "direct Vast create is disabled; use the systemd-supervised run command"
        )

    def _create_one_unarmed(
        self, intent_path: Path, admission_path: Path, dispatch_receipt_path: Path
    ) -> dict[str, object]:
        """Mock-test seam for the disabled lifecycle state machine; never use operationally."""
        intent_path = intent_path.resolve(strict=True)
        admission_path = admission_path.resolve(strict=True)
        if not intent_path.is_relative_to(ROOT) or not admission_path.is_relative_to(ROOT):
            raise CampaignError("intent and root admission must be stored inside the project root")
        intent = _read_object(intent_path)
        normalized = _validate_intent(intent, intent_path)
        _, ssh_key_sha256 = load_ssh_public_key()
        root_receipt_before = file_sha256(admission_path)
        admission = _read_object(admission_path)
        dispatch_receipt_path = dispatch_receipt_path.resolve(strict=True)
        if not dispatch_receipt_path.is_relative_to(ROOT):
            raise CampaignError("dispatcher receipt must be stored inside the project runtime")
        if (
            dispatch_receipt_path.parent.name != "attempts"
            or dispatch_receipt_path.name != f"{normalized['attempt_id']}.json"
        ):
            raise CampaignError(
                "dispatcher receipt must be the exact runtime/attempts/<attempt>.json file"
            )
        dispatch_receipt_hash = file_sha256(dispatch_receipt_path)
        dispatch_receipt = _read_object(dispatch_receipt_path)
        ceiling, balance_floor, root_receipt, costs, release_capacity, release_hash = (
            _validate_admission(admission, intent, normalized, dispatch_receipt, admission_path)
        )
        attempt_id = str(normalized["attempt_id"])
        with ledger_transaction(self.ledger_path) as state:
            rows = _attempts(state)
            if attempt_id in rows:
                raise CampaignError(
                    "attempt already exists in ledger; never POST the same attempt twice"
                )

        instances = self.api.list_instances()
        label = str(normalized["campaign_label"])
        if any(row.get("label") == label for row in instances):
            raise CampaignError(
                "campaign label already exists; existing instances are never adopted"
            )
        offer_snapshot = cast(dict[str, object], normalized["offer"])
        create_payload = cast(dict[str, object], normalized["create_payload"])
        offer_id = _offer_id(offer_snapshot)
        offers = self.api.search_offer_for_revalidation(offer_snapshot, int(create_payload["disk"]))
        matching = [
            _offer_snapshot(row, query_type="on-demand")
            for row in offers
            if str(row.get("id")) == str(offer_id)
        ]
        if len(matching) != 1:
            raise CampaignError("fresh offer differs from the frozen, root-approved offer snapshot")
        live_offer_snapshot = matching[0]
        reliability_diffs = _compare_revalidated_offer(
            offer_snapshot,
            live_offer_snapshot,
            reliability_floor=normalized.get("min_offer_reliability"),
        )
        user = self.api.current_user()
        account_scope = _account_scope(user)
        if account_scope != normalized["account_scope_sha256"]:
            raise CampaignError(
                "current Vast account does not match frozen intent and root admission"
            )
        balance = _balance(user)
        if balance < balance_floor or balance < ceiling:
            raise CampaignError(
                "fresh Vast balance is below the root-admitted floor or attempt ceiling"
            )
        if not isinstance(admission.get("vast_admission"), dict):
            raise CampaignError("Vast-specific root admission is malformed")
        vast_admission = cast(dict[str, object], admission["vast_admission"])
        disk_cap = vast_admission.get("max_disk_gb")
        if isinstance(disk_cap, bool) or not isinstance(disk_cap, int) or disk_cap < 1:
            raise CampaignError("root max_disk_gb must be a positive integer")
        disk_size = create_payload.get("disk")
        if isinstance(disk_size, bool) or not isinstance(disk_size, int):
            raise CampaignError("frozen disk size is malformed")
        if disk_size > disk_cap:
            raise CampaignError("frozen disk size exceeds root admission")
        submitted_at = utc_now()
        startup_deadline = min(
            _parse_time(submitted_at, "submitted_at_utc")
            + timedelta(seconds=WORKER_STARTUP_TIMEOUT_SECONDS),
            _parse_time(vast_admission["hard_stop_utc"], "hard_stop_utc"),
        ).isoformat()
        baseline_ids = sorted(_instance_id(row) for row in instances)
        if (
            file_sha256(admission_path) != root_receipt_before
            or root_receipt != root_receipt_before
        ):
            raise CampaignError("root admission changed during provider preflight")
        if file_sha256(dispatch_receipt_path) != dispatch_receipt_hash:
            raise CampaignError("dispatcher launch intent changed during provider preflight")
        _require_fresh_root_window(admission, dispatch_receipt)
        with ledger_transaction(self.ledger_path) as state:
            rows = _attempts(state)
            _ledger_account(state, account_scope)
            if attempt_id in rows:
                raise CampaignError("attempt was concurrently admitted; do not submit again")
            stage = vast_admission.get("cohort_stage")
            if stage == "first_cohort":
                prior_first_cohort = sum(
                    (
                        _money(row.get("actual_all_in_usd"), "first-cohort actual spend")
                        if row.get("status") == "settled"
                        else _money(row.get("reserved_all_in_usd"), "first-cohort reservation")
                        for row in rows.values()
                        if row.get("cohort_stage") == "first_cohort"
                    ),
                    Decimal("0.00"),
                )
                if prior_first_cohort + ceiling > FIRST_COHORT_MAX_USD:
                    raise CampaignError(
                        "first-cohort reserved and settled spend is limited to $4 total"
                    )
            if stage == "evidence_released":
                if release_hash is None or release_capacity is None:
                    raise CampaignError("later cohort lacks its root evidence-release receipt")
                if any(
                    row.get("evidence_release_receipt_sha256") == release_hash
                    for row in rows.values()
                ):
                    raise CampaignError("evidence-release receipt was already used")
                prior_releases = sum(
                    (
                        _money(row.get("evidence_released_usd", "0.00"), "evidence release")
                        for row in rows.values()
                        if row.get("evidence_released_usd") is not None
                    ),
                    Decimal("0.00"),
                )
                if prior_releases + release_capacity > LATER_RELEASE_MAX_USD:
                    raise CampaignError("cumulative evidence releases exceed the remaining $16")
            total = _committed_usd(rows) + ceiling
            campaign_ceiling = _money(
                vast_admission["total_campaign_ceiling_usd"], "total_campaign_ceiling_usd"
            )
            if total > campaign_ceiling or total > MAX_CAMPAIGN_USD:
                raise CampaignError(
                    "new reservation exceeds the $20 cumulative Vast credit ceiling"
                )
            rows[attempt_id] = {
                "attempt_id": attempt_id,
                "campaign_label": label,
                "account_scope_sha256": account_scope,
                "packet_sha256": normalized["packet_sha256"],
                "packet_path": normalized["packet_path"],
                "worker_inputs": normalized["worker_inputs"],
                "worker_inputs_sha256": normalized["worker_inputs_sha256"],
                "max_upload_bytes": normalized["max_upload_bytes"],
                "upload_timeout_seconds": normalized["upload_timeout_seconds"],
                "intent_sha256": normalized["intent_sha256"],
                "request_sha256": normalized["request_sha256"],
                "offer_id": offer_id,
                "job_sha256": dispatch_receipt["job_sha256"],
                "machine_id": offer_snapshot["machine_id"],
                "offer_snapshot_sha256": normalized["offer_snapshot_sha256"],
                "offer_revalidation_snapshot": live_offer_snapshot,
                "offer_revalidation_snapshot_sha256": canonical_sha256(live_offer_snapshot),
                "offer_revalidation_allowed_diffs": reliability_diffs,
                "min_offer_reliability": normalized.get("min_offer_reliability"),
                "max_runtime_seconds": normalized["max_runtime_seconds"],
                "worker_setup_timeout_seconds": normalized["worker_setup_timeout_seconds"],
                "max_worker_host_ram_gb": str(normalized["max_worker_host_ram_gb"]),
                "hard_deadline_seconds_after_create": normalized["max_runtime_seconds"],
                "remote_output_dir": normalized["remote_output_dir"],
                "expected_artifacts": normalized["expected_artifacts"],
                "max_artifact_bytes": normalized["max_artifact_bytes"],
                "create_payload": create_payload,
                "ssh_key_sha256": ssh_key_sha256,
                "ssh_key_attach_status": "pending",
                "pre_submit_intent": {
                    "submitted_at_utc": submitted_at,
                    "dispatcher_receipt_path": str(dispatch_receipt_path),
                    "dispatcher_receipt_sha256": dispatch_receipt_hash,
                    "baseline_instance_ids": baseline_ids,
                    "baseline_instance_records": [_sanitized_instance(row) for row in instances],
                    "root_controller_receipt_sha256": root_receipt,
                    "cost_breakdown_usd": costs,
                    "offer_revalidation_snapshot": live_offer_snapshot,
                    "offer_revalidation_snapshot_sha256": canonical_sha256(live_offer_snapshot),
                    "offer_revalidation_allowed_diffs": reliability_diffs,
                    "min_offer_reliability": normalized.get("min_offer_reliability"),
                },
                "cohort_stage": str(stage),
                "evidence_release_receipt_sha256": release_hash,
                "evidence_released_usd": (
                    None if release_capacity is None else _money_text(release_capacity)
                ),
                "reserved_all_in_usd": _money_text(ceiling),
                "actual_all_in_usd": None,
                "instance_id": None,
                "worker_stage_attempts": 0,
                "collection_attempts": 0,
                "collection_terminal_failure": False,
                # A crash after this durable write is treated as an ambiguous submit.
                "status": "launch_uncertain",
                "hard_stop_utc": vast_admission["hard_stop_utc"],
                "worker_startup_deadline_utc": startup_deadline,
                "cleanup_deadline_utc": vast_admission["cleanup_deadline_utc"],
                "updated_at_utc": submitted_at,
            }
        try:
            response = self.api.create_instance(offer_id, create_payload)
        except Exception as error:
            # Any exception after the single request begins can follow a remote success.
            # Persist only the bounded exception type and validated numeric HTTP status.
            http_status = error.http_status if isinstance(error, ApiError) else None
            if (
                isinstance(http_status, bool)
                or not isinstance(http_status, int)
                or not 100 <= http_status <= 599
            ):
                http_status = None
            self._update_attempt(
                attempt_id,
                status="launch_uncertain",
                last_error_type=type(error).__name__,
                last_error_http_status=http_status,
                updated_at_utc=utc_now(),
            )
            raise CampaignError(
                "create response is ambiguous; reservation remains held; reconcile this attempt, never resubmit"
            ) from None
        instance_id_raw = response.get("new_contract")
        if isinstance(instance_id_raw, bool) or not isinstance(instance_id_raw, (str, int)):
            self._update_attempt(
                attempt_id,
                status="launch_uncertain",
                last_error_type="CreateResponseMissingInstanceId",
                updated_at_utc=utc_now(),
            )
            raise CampaignError(
                "Vast create response lacks new_contract; reconcile, never resubmit"
            )
        instance_id = str(instance_id_raw)
        if INSTANCE_ID_RE.fullmatch(instance_id) is None:
            self._update_attempt(
                attempt_id,
                status="launch_uncertain",
                last_error_type="InvalidCreateResponseInstanceId",
                updated_at_utc=utc_now(),
            )
            raise CampaignError("Vast returned an invalid instance ID; reconcile, never resubmit")
        pre_submit = self._get_attempt(attempt_id).get("pre_submit_intent")
        if not isinstance(pre_submit, dict):
            raise CampaignError("persisted pre-submit baseline disappeared after submit")
        baseline_ids = pre_submit.get("baseline_instance_ids")
        if not isinstance(baseline_ids, list) or any(
            not isinstance(value, str) for value in baseline_ids
        ):
            raise CampaignError("persisted pre-submit baseline is malformed after submit")
        if instance_id in baseline_ids:
            self._update_attempt(
                attempt_id,
                reported_create_instance_id=instance_id,
                last_error_type="CreateResponseReusedBaselineInstanceId",
                status="launch_uncertain",
                updated_at_utc=utc_now(),
            )
            raise CampaignError(
                "Vast create response reused a pre-submit instance ID; no instance adopted; "
                "reconcile, never resubmit"
            )
        try:
            provider_row = _wrapped_instance(self.api.get_instance(instance_id))
        except Exception as error:
            self._update_attempt(
                attempt_id,
                reported_create_instance_id=instance_id,
                last_error_type=type(error).__name__,
                status="launch_uncertain",
                updated_at_utc=utc_now(),
            )
            raise CampaignError(
                "Vast create response could not be verified against an instance record; "
                "no instance adopted; reconcile, never resubmit"
            ) from None
        created_value = provider_row.get("start_date") or provider_row.get("created_at")
        provider_created_at = _parse_provider_time(created_value)
        submitted_at = _parse_time(pre_submit.get("submitted_at_utc"), "submitted_at_utc")
        if (
            _instance_id(provider_row) != instance_id
            or provider_row.get("label") != label
            or provider_row.get("machine_id") != offer_snapshot.get("machine_id")
            or provider_created_at is None
            or provider_created_at < submitted_at
        ):
            self._update_attempt(
                attempt_id,
                reported_create_instance_id=instance_id,
                last_error_type="CreateResponseInstanceNotVerified",
                status="launch_uncertain",
                updated_at_utc=utc_now(),
            )
            raise CampaignError(
                "Vast create response does not prove a new matching instance; no instance "
                "adopted; reconcile, never resubmit"
            )
        self._update_attempt(
            attempt_id,
            instance_id=instance_id,
            status="created",
            creation_response_received_at_utc=utc_now(),
            creation_receipt_source="Vast PUT response new_contract",
            updated_at_utc=utc_now(),
        )
        return _status_view(self._get_attempt(attempt_id))

    def reconcile(self, attempt_id: str) -> dict[str, object]:
        """Reconcile an ambiguous create from its pre-submit baseline and exact offer identity."""
        record = self._get_attempt(attempt_id)
        if record.get("instance_id") is not None:
            raise CampaignError("an attempt with a persisted instance ID is no longer ambiguous")
        if record.get("status") not in {"launch_uncertain", "needs_review"}:
            raise CampaignError("only an unresolved ambiguous create can be reconciled")
        user = self.api.current_user()
        scope = _account_scope(user)
        self._require_owned_scope(record, scope)
        instances = self.api.list_instances()
        pre_submit = record.get("pre_submit_intent")
        if not isinstance(pre_submit, dict):
            raise CampaignError("ambiguous attempt has no persisted pre-submit intent")
        before = pre_submit.get("baseline_instance_ids")
        if not isinstance(before, list) or any(not isinstance(value, str) for value in before):
            raise CampaignError(
                "ambiguous attempt baseline is malformed; do not adopt any instance"
            )
        label = record.get("campaign_label")
        machine_id = record.get("machine_id")
        submitted_at = _parse_time(pre_submit.get("submitted_at_utc"), "submitted_at_utc")
        candidates: list[dict[str, object]] = []
        label_matches: list[dict[str, object]] = []
        for instance in instances:
            if instance.get("label") != label:
                continue
            label_matches.append(instance)
            instance_id = _instance_id(instance)
            if instance_id in before or instance.get("machine_id") != machine_id:
                continue
            created_value = instance.get("start_date") or instance.get("created_at")
            created_at = _parse_provider_time(created_value)
            if created_at is None:
                continue
            if created_at >= submitted_at:
                candidates.append(instance)
        if len(candidates) != 1 or len(label_matches) != 1:
            self._update_attempt(
                attempt_id,
                status="needs_review",
                reconcile_result=(
                    "no_unambiguous_instance_match"
                    if not candidates
                    else "multiple_instance_matches"
                ),
                updated_at_utc=utc_now(),
            )
            return self.attempt_status(attempt_id)
        instance_id = _instance_id(candidates[0])
        startup_deadline = min(
            submitted_at + timedelta(seconds=WORKER_STARTUP_TIMEOUT_SECONDS),
            _parse_time(record.get("hard_stop_utc"), "hard_stop_utc"),
        ).isoformat()
        updated = self._update_attempt(
            attempt_id,
            instance_id=instance_id,
            status="created",
            worker_startup_deadline_utc=record.get("worker_startup_deadline_utc")
            or startup_deadline,
            creation_receipt_source="reconciled from persisted intent, new ID, offer and timestamp",
            reconciled_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        return _status_view(updated)

    def attempt_status(self, attempt_id: str) -> dict[str, object]:
        """Read an owned attempt's status and persist its latest provider observation."""
        record = self._get_attempt(attempt_id)
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or INSTANCE_ID_RE.fullmatch(instance_id) is None:
            return _status_view(record)
        user = self.api.current_user()
        self._require_owned_scope(record, _account_scope(user))
        provider_record = _wrapped_instance(self.api.get_instance(instance_id))
        self._assert_provider_ownership(record, provider_record)
        updated = self._update_attempt(
            attempt_id,
            last_provider_status=provider_record.get("actual_status"),
            last_provider_observed_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        return _status_view(updated)

    def _startup_deadline(self, record: Mapping[str, object]) -> datetime:
        """Return the persisted startup deadline, bounded by the admitted hard stop."""
        raw_deadline = record.get("worker_startup_deadline_utc")
        if isinstance(raw_deadline, str):
            deadline = _parse_time(raw_deadline, "worker_startup_deadline_utc")
        else:
            pre_submit = record.get("pre_submit_intent")
            if not isinstance(pre_submit, Mapping):
                raise CampaignError("owned attempt lacks its startup deadline receipt")
            submitted = _parse_time(pre_submit.get("submitted_at_utc"), "submitted_at_utc")
            deadline = submitted + timedelta(seconds=WORKER_STARTUP_TIMEOUT_SECONDS)
        return min(deadline, _parse_time(record.get("hard_stop_utc"), "hard_stop_utc"))

    def _attach_owned_ssh_key(
        self, attempt_id: str, instance: Mapping[str, object]
    ) -> dict[str, object]:
        """Attach only the controller public key to the proven campaign-owned instance."""
        record = self._get_attempt(attempt_id)
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str):
            raise CampaignError("SSH key attachment requires a persisted owned instance ID")
        if record.get("ssh_key_attach_status") == "attached":
            return record
        if record.get("ssh_key_attach_status") != "pending":
            raise WorkerStartupFailure(
                "prior owned-instance SSH key attachment outcome is unresolved; fail closed"
            )
        self._assert_provider_ownership(record, instance)
        user = self.api.current_user()
        self._require_owned_scope(record, _account_scope(user))
        public_key, public_hash = load_ssh_public_key()
        if public_hash != record.get("ssh_key_sha256"):
            raise WorkerStartupFailure("controller SSH public key changed after create admission")
        self._update_attempt(
            attempt_id,
            ssh_key_attach_status="attaching",
            ssh_key_attach_started_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        try:
            # The API key and public-key text are used only in this controller-side request.
            self.api.attach_ssh_key(instance_id, public_key)
        except Exception as error:
            self._update_attempt(
                attempt_id,
                ssh_key_attach_status="failed",
                ssh_key_attach_error_type=type(error).__name__,
                updated_at_utc=utc_now(),
            )
            raise WorkerStartupFailure(
                f"controller SSH key attachment failed: {type(error).__name__}"
            ) from None
        return self._update_attempt(
            attempt_id,
            ssh_key_attach_status="attached",
            ssh_key_attach_completed_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )

    def _campaign_known_hosts(
        self,
        attempt_id: str,
        instance: Mapping[str, object],
        *,
        allow_first_use: bool,
    ) -> Path:
        """Pin the first owned endpoint key in a per-attempt file and verify it thereafter."""
        record = self._get_attempt(attempt_id)
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str):
            raise CampaignError("host-key pinning requires a persisted owned instance ID")
        self._assert_provider_ownership(record, instance)
        user = self.api.current_user()
        self._require_owned_scope(record, _account_scope(user))
        if str(instance.get("actual_status", "")).lower() != "running":
            raise WorkerStartupPending("owned instance is not running; defer host-key discovery")
        if record.get("ssh_key_attach_status") != "attached":
            raise WorkerStartupFailure("owned-instance SSH key has not been attached")
        host, port = _ssh_endpoint(instance)
        host_token = host if port == 22 else f"[{host}]:{port}"
        known_hosts_path = ROOT / ".runtime" / "vast" / "known_hosts" / f"{attempt_id}.known_hosts"
        stored_path = record.get("known_hosts_path")
        if stored_path is not None:
            if (
                stored_path != str(known_hosts_path)
                or record.get("known_hosts_host") != host
                or record.get("known_hosts_port") != port
            ):
                raise CampaignError("campaign endpoint differs from its persisted host-key pin")
            try:
                known_hosts_stat = known_hosts_path.lstat()
                contents = known_hosts_path.read_text(encoding="ascii")
            except (OSError, UnicodeDecodeError):
                raise CampaignError("persisted campaign known-hosts file is unavailable") from None
            if not stat.S_ISREG(known_hosts_stat.st_mode):
                raise CampaignError("persisted campaign known-hosts path is not a regular file")
            fingerprints = _known_host_fingerprints(contents, host_token)
            if file_sha256(known_hosts_path) != record.get(
                "known_hosts_sha256"
            ) or fingerprints != record.get("known_hosts_fingerprints"):
                raise CampaignError("persisted campaign known-hosts pin changed")
            return known_hosts_path
        if not allow_first_use:
            raise CampaignError("owned endpoint has no persisted campaign host-key pin")
        if known_hosts_path.exists() or known_hosts_path.is_symlink():
            raise CampaignError("unreceipted campaign known-hosts file exists; refusing overwrite")
        try:
            scan = subprocess.run(
                ["ssh-keyscan", "-T", "5", "-p", str(port), host],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise WorkerStartupPending(
                f"owned endpoint host-key scan is not ready: {type(error).__name__}"
            ) from None
        if scan.returncode != 0 or not scan.stdout.strip():
            raise WorkerStartupPending("owned endpoint has not returned an SSH host key yet")
        accepted: list[str] = []
        for line in scan.stdout.splitlines():
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 3 or fields[0] != host_token:
                continue
            try:
                blob = base64.b64decode(fields[2], validate=True)
                key_type_bytes = blob[4 : 4 + int.from_bytes(blob[:4], "big")]
                key_type = key_type_bytes.decode("ascii")
            except (ValueError, UnicodeDecodeError, IndexError):
                raise WorkerStartupFailure(
                    "owned endpoint returned a malformed SSH host key"
                ) from None
            if len(blob) < 5 or key_type != fields[1]:
                raise WorkerStartupFailure("owned endpoint returned an inconsistent SSH host key")
            accepted.append(" ".join(fields[:3]))
        contents = "\n".join(dict.fromkeys(accepted)) + "\n" if accepted else ""
        if not contents:
            raise WorkerStartupPending("owned endpoint host-key scan returned no matching key")
        known_hosts_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        descriptor = os.open(known_hosts_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as output:
                output.write(contents)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            known_hosts_path.unlink(missing_ok=True)
            raise
        fingerprints = _known_host_fingerprints(contents, host_token)
        if not fingerprints:
            known_hosts_path.unlink(missing_ok=True)
            raise WorkerStartupFailure("owned endpoint host-key scan produced no valid key")
        self._update_attempt(
            attempt_id,
            known_hosts_path=str(known_hosts_path),
            known_hosts_host=host,
            known_hosts_port=port,
            known_hosts_sha256=file_sha256(known_hosts_path),
            known_hosts_fingerprints=fingerprints,
            known_hosts_pinned_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        return known_hosts_path

    def _stage_and_launch(
        self,
        attempt_id: str,
        *,
        collection_timeout_seconds: int = 300,
        prefetch_plan_path: Path | None = None,
        prefetch_admission_path: Path | None = None,
    ) -> dict[str, object]:
        """Stage hash-bound files and launch once on the receipt-owned worker."""
        if collection_timeout_seconds <= 0 or collection_timeout_seconds > 3600:
            raise ValueError("collection timeout must be in (0, 3600] seconds")
        if (prefetch_plan_path is None) != (prefetch_admission_path is None):
            raise CampaignError(
                "private prefetch plan and public admission must be supplied together"
            )
        record = self._get_attempt(attempt_id)
        if record.get("prefetch_status") in FATAL_PREFETCH_STATUSES:
            raise WorkerStagingIntegrityError("persisted fatal prefetch prevents stage retry")
        if record.get("worker_launch_status") in {"launch_uncertain", "worker_started"}:
            return _status_view(record)
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str):
            return _status_view(record)
        user = self.api.current_user()
        self._require_owned_scope(record, _account_scope(user))
        instance = _wrapped_instance(self.api.get_instance(instance_id))
        self._assert_provider_ownership(record, instance)
        provider_status = str(instance.get("actual_status", "")).lower()
        startup_deadline = self._startup_deadline(record)
        if datetime.now(UTC) >= startup_deadline:
            self._update_attempt(
                attempt_id,
                status="failed_before_worker",
                worker_startup_status=provider_status or "unknown",
                worker_terminal_failure=True,
                worker_failure_error_type="startup_deadline",
                supervisor_last_phase="worker_startup_failed",
                updated_at_utc=utc_now(),
            )
            raise WorkerStartupFailure("owned instance exceeded its bounded startup deadline")
        if provider_status != "running":
            if provider_status in TERMINAL_STARTUP_STATUSES:
                failure = "provider_terminal_status" if provider_status else "startup_deadline"
                self._update_attempt(
                    attempt_id,
                    status="failed_before_worker",
                    worker_startup_status=provider_status or "unknown",
                    worker_terminal_failure=True,
                    worker_failure_error_type=failure,
                    supervisor_last_phase="worker_startup_failed",
                    updated_at_utc=utc_now(),
                )
                raise WorkerStartupFailure(
                    f"owned instance failed to reach running before startup deadline: {failure}"
                )
            return self._update_attempt(
                attempt_id,
                worker_startup_status=provider_status or "unknown",
                supervisor_last_phase="waiting_for_worker_startup",
                updated_at_utc=utc_now(),
            )
        if record.get("ssh_key_attach_status") == "pending":
            record = self._attach_owned_ssh_key(attempt_id, instance)
        known_hosts_path = self._campaign_known_hosts(attempt_id, instance, allow_first_use=True)
        record = self._get_attempt(attempt_id)
        stage_attempts = record.get("worker_stage_attempts", 0)
        if (
            isinstance(stage_attempts, bool)
            or not isinstance(stage_attempts, int)
            or stage_attempts < 0
        ):
            raise CampaignError("owned attempt worker stage attempt count is malformed")
        if stage_attempts >= 2:
            raise CampaignError("worker staging retry limit reached")
        self._update_attempt(
            attempt_id,
            worker_stage_attempts=stage_attempts + 1,
            worker_stage_started_at_utc=record.get("worker_stage_started_at_utc") or utc_now(),
            updated_at_utc=utc_now(),
        )
        host, port = _ssh_endpoint(instance)

        packet_path_value = record.get("packet_path")
        if not isinstance(packet_path_value, str):
            raise CampaignError("owned attempt lacks its frozen packet path")
        packet_path = Path(packet_path_value).resolve(strict=True)
        if not packet_path.is_relative_to(ROOT) or not packet_path.is_file():
            raise CampaignError("frozen packet is no longer a regular project file")
        packet_sha256 = _sha256(record.get("packet_sha256"), "persisted packet SHA-256")
        if file_sha256(packet_path) != packet_sha256:
            raise CampaignError("frozen worker packet changed after admission")
        max_upload_bytes = record.get("max_upload_bytes")
        upload_timeout = record.get("upload_timeout_seconds")
        if (
            isinstance(max_upload_bytes, bool)
            or not isinstance(max_upload_bytes, int)
            or isinstance(upload_timeout, bool)
            or not isinstance(upload_timeout, int)
        ):
            raise CampaignError("owned attempt upload ceilings are malformed")
        max_artifact_bytes = record.get("max_artifact_bytes")
        if (
            isinstance(max_artifact_bytes, bool)
            or not isinstance(max_artifact_bytes, int)
            or max_artifact_bytes <= 0
        ):
            raise CampaignError("owned attempt artifact byte ceiling is malformed")
        inputs_value = record.get("worker_inputs")
        if not isinstance(inputs_value, list) or any(
            not isinstance(row, dict) for row in inputs_value
        ):
            raise CampaignError("owned attempt worker input list is malformed")
        inputs = cast(list[dict[str, object]], inputs_value)
        input_bytes = 0
        resolved_inputs: list[tuple[dict[str, object], Path]] = []
        for row in inputs:
            source_value = row.get("source_path")
            size_bytes = row.get("size_bytes")
            if not isinstance(source_value, str) or Path(source_value).is_absolute():
                raise CampaignError("persisted worker input source path is malformed")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
                raise CampaignError("persisted worker input size is malformed")
            source_candidate = ROOT / source_value
            if source_candidate.is_symlink():
                raise CampaignError("worker input source cannot be a symbolic link")
            source_path = source_candidate.resolve(strict=True)
            if not source_path.is_relative_to(ROOT) or not source_path.is_file():
                raise CampaignError("worker input source is no longer a project file")
            if source_path.stat().st_size != size_bytes:
                raise CampaignError("worker input size changed after its identity receipt")
            _sha256(row.get("source_sha256"), "worker input source SHA-256")
            remote_path = _valid_remote_relative(row.get("remote_path"))
            input_bytes += size_bytes
            resolved_inputs.append((row, source_path))
        total_upload_bytes = packet_path.stat().st_size + input_bytes
        if total_upload_bytes > max_upload_bytes:
            raise CampaignError("packet and worker inputs exceed the admitted upload ceiling")
        payload = record.get("create_payload")
        if not isinstance(payload, dict):
            raise CampaignError("owned attempt has no frozen instance payload")
        disk_gb = payload.get("disk")
        if isinstance(disk_gb, bool) or not isinstance(disk_gb, int) or disk_gb <= 0:
            raise CampaignError("owned attempt has no valid worker disk ceiling")
        unpacked_packet_bytes = _validate_worker_bundle(
            packet_path,
            disk_gb * int(BYTES_PER_GB)
            - input_bytes
            - packet_path.stat().st_size
            - max_artifact_bytes,
        )
        required_disk_bytes = (
            input_bytes + packet_path.stat().st_size + unpacked_packet_bytes + max_artifact_bytes
        )
        if required_disk_bytes > disk_gb * int(BYTES_PER_GB):
            raise CampaignError("staged inputs and unpacked packet exceed worker disk capacity")

        output_root = _valid_remote_root(record.get("remote_output_dir"))
        label = record.get("campaign_label")
        if not isinstance(label, str) or LABEL_RE.fullmatch(label) is None:
            raise CampaignError("owned attempt has no valid campaign label")
        remote_work_dir = f"{output_root}/.worker/{label}"
        self._update_attempt(
            attempt_id,
            status="staging_inputs",
            upload_started_at_utc=record.get("upload_started_at_utc") or utc_now(),
            upload_total_bytes=total_upload_bytes,
            updated_at_utc=utc_now(),
        )
        upload_started_at = self._get_attempt(attempt_id).get("upload_started_at_utc")
        upload_started = _parse_time(upload_started_at, "upload_started_at_utc")
        elapsed_upload = max(0.0, (datetime.now(UTC) - upload_started).total_seconds())
        hard_stop_value = record.get("hard_stop_utc")
        hard_stop = _parse_time(hard_stop_value, "hard_stop_utc")
        remaining_hard_stop = max(0, int((hard_stop - datetime.now(UTC)).total_seconds()))
        remaining_upload = min(upload_timeout - elapsed_upload, remaining_hard_stop)
        if remaining_upload <= 0:
            raise CampaignError(
                "frozen packet and input upload deadline expired or cohort hard stop reached"
            )
        deadline = time.monotonic() + remaining_upload

        def remaining_seconds() -> int:
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise CampaignError("frozen packet and input upload deadline expired")
            return remaining

        _ssh_command(
            host,
            port,
            f"mkdir -p -- {shlex.quote(remote_work_dir)} {shlex.quote(output_root)}",
            min(30, remaining_seconds()),
            known_hosts_path=known_hosts_path,
        )
        remote_files: list[tuple[Path, str, str, int]] = [
            (
                packet_path,
                packet_sha256,
                f"{remote_work_dir}/packet.tar",
                packet_path.stat().st_size,
            )
        ]
        for row, source_path in resolved_inputs:
            sha256 = _sha256(row.get("source_sha256"), "worker input source SHA-256")
            remote_relative = _valid_remote_relative(row.get("remote_path"))
            remote_files.append(
                (
                    source_path,
                    sha256,
                    f"{remote_work_dir}/inputs/{remote_relative}",
                    cast(int, row["size_bytes"]),
                )
            )
        if prefetch_plan_path is not None and prefetch_admission_path is not None:
            stage_record = self._get_attempt(attempt_id)
            if stage_record.get("prefetch_status") is None:
                try:
                    prepared = kaggle_vast_prefetch.prepare_prefetch(
                        prefetch_plan_path,
                        prefetch_admission_path,
                        attempt_id=attempt_id,
                        campaign_label=label,
                        frozen_inputs=inputs,
                        runtime_root=ROOT / ".runtime",
                    )
                except kaggle_vast_prefetch.PrefetchPlanError:
                    self._update_attempt(
                        attempt_id,
                        prefetch_status="invalid_plan",
                        worker_terminal_failure=True,
                        prefetch_error_type="PrefetchPlanError",
                        updated_at_utc=utc_now(),
                    )
                    raise WorkerStagingIntegrityError(
                        "private prefetch plan does not match its public admission and frozen inputs"
                    ) from None
                prefetch_paths = {item.remote_path for item in prepared.items}
                non_prefetch_bytes = sum(
                    size_bytes
                    for _, _, remote_path, size_bytes in remote_files
                    if not any(remote_path.endswith(f"/inputs/{path}") for path in prefetch_paths)
                )
                # Preserve upload runway for the packet and launch. One MiB/s is a
                # conservative staging reserve; prefetch itself is capped at 20 minutes.
                post_fetch_reserve = max(300, (non_prefetch_bytes + 1_048_575) // 1_048_576 + 120)
                prefetch_timeout = min(
                    kaggle_vast_prefetch.MAX_PREFETCH_SECONDS,
                    remaining_seconds() - post_fetch_reserve,
                )
                if prefetch_timeout < 90:
                    self._update_attempt(
                        attempt_id,
                        prefetch_status="skipped_insufficient_runway",
                        prefetch_admission_sha256=prepared.admission_sha256,
                        prefetch_private_plan_sha256=prepared.private_plan_sha256,
                        prefetch_remote_program_sha256=kaggle_vast_prefetch.REMOTE_PROGRAM_SHA256,
                        prefetch_elapsed_ms=0,
                        prefetch_receipt=[],
                        updated_at_utc=utc_now(),
                    )
                else:
                    private_payload = kaggle_vast_prefetch.make_remote_payload(
                        prepared,
                        attempt_id=attempt_id,
                        campaign_label=label,
                        output_root=output_root,
                        budget_seconds=prefetch_timeout - 20,
                    )
                    remote_program = "python3 -c " + shlex.quote(
                        kaggle_vast_prefetch.REMOTE_PROGRAM
                    )
                    prefetch_started = time.monotonic()
                    try:
                        raw_result = _ssh_stdin(
                            host,
                            port,
                            remote_program,
                            prefetch_timeout,
                            known_hosts_path=known_hosts_path,
                            input_bytes=private_payload,
                            allowed_exit_codes=(1,),
                        )
                    except (CampaignError, OSError, subprocess.SubprocessError):
                        self._update_attempt(
                            attempt_id,
                            prefetch_status="transport_failed",
                            worker_terminal_failure=True,
                            prefetch_admission_sha256=prepared.admission_sha256,
                            prefetch_private_plan_sha256=prepared.private_plan_sha256,
                            prefetch_remote_program_sha256=kaggle_vast_prefetch.REMOTE_PROGRAM_SHA256,
                            prefetch_elapsed_ms=int((time.monotonic() - prefetch_started) * 1000),
                            prefetch_receipt=[],
                            prefetch_error_type="SSHTransportError",
                            updated_at_utc=utc_now(),
                        )
                        raise WorkerStagingIntegrityError(
                            "signed prefetch transport failed; refusing concurrent rsync fallback"
                        ) from None
                    try:
                        prefetch_result = kaggle_vast_prefetch.validate_remote_result(
                            raw_result, prepared
                        )
                    except kaggle_vast_prefetch.PrefetchPlanError:
                        self._update_attempt(
                            attempt_id,
                            prefetch_status="invalid_remote_receipt",
                            worker_terminal_failure=True,
                            prefetch_admission_sha256=prepared.admission_sha256,
                            prefetch_private_plan_sha256=prepared.private_plan_sha256,
                            prefetch_remote_program_sha256=kaggle_vast_prefetch.REMOTE_PROGRAM_SHA256,
                            prefetch_elapsed_ms=int((time.monotonic() - prefetch_started) * 1000),
                            prefetch_receipt=[],
                            prefetch_error_type="PrefetchReceiptError",
                            updated_at_utc=utc_now(),
                        )
                        raise WorkerStagingIntegrityError(
                            "signed prefetch returned an invalid nonsecret receipt"
                        ) from None
                    receipt = cast(list[dict[str, object]], prefetch_result["completed"])
                    if prefetch_result["error_code"] == "unsafe_destination":
                        self._update_attempt(
                            attempt_id,
                            prefetch_status="unsafe_destination",
                            worker_terminal_failure=True,
                            prefetch_admission_sha256=prepared.admission_sha256,
                            prefetch_private_plan_sha256=prepared.private_plan_sha256,
                            prefetch_remote_program_sha256=kaggle_vast_prefetch.REMOTE_PROGRAM_SHA256,
                            prefetch_elapsed_ms=int((time.monotonic() - prefetch_started) * 1000),
                            prefetch_receipt=receipt,
                            prefetch_error_type="UnsafeRemoteDestination",
                            updated_at_utc=utc_now(),
                        )
                        raise WorkerStagingIntegrityError(
                            "signed prefetch refused an unsafe remote destination"
                        )
                    self._update_attempt(
                        attempt_id,
                        prefetch_status=(
                            "complete" if prefetch_result["status"] == "complete" else "failed"
                        ),
                        prefetch_admission_sha256=prepared.admission_sha256,
                        prefetch_private_plan_sha256=prepared.private_plan_sha256,
                        prefetch_remote_program_sha256=kaggle_vast_prefetch.REMOTE_PROGRAM_SHA256,
                        prefetch_elapsed_ms=int((time.monotonic() - prefetch_started) * 1000),
                        prefetch_bytes=sum(int(row["size_bytes"]) for row in receipt),
                        prefetch_receipt=receipt,
                        prefetch_error_type=prefetch_result["error_code"],
                        updated_at_utc=utc_now(),
                    )
        verified_paths: list[str] = []
        verified_bytes = 0
        for source_path, expected_sha256, remote_path, size_bytes in remote_files:
            remote_parent = str(PurePosixPath(remote_path).parent)
            _ssh_command(
                host,
                port,
                f"mkdir -p -- {shlex.quote(remote_parent)}",
                min(30, remaining_seconds()),
                known_hosts_path=known_hosts_path,
            )
            check_existing = (
                f"printf '%s  %s\\n' {shlex.quote(expected_sha256)} "
                f"{shlex.quote(remote_path)} | sha256sum --check --status"
            )
            existing_status = _ssh_command(
                host,
                port,
                check_existing,
                remaining_seconds(),
                allowed_exit_codes=(1,),
                known_hosts_path=known_hosts_path,
            )
            if existing_status != 0:
                temporary_path = f"{remote_path}.part"
                _rsync_push(
                    host,
                    port,
                    source_path,
                    temporary_path,
                    remaining_seconds(),
                    size_bytes,
                    known_hosts_path,
                )
                verify_and_move = (
                    f"printf '%s  %s\\n' {shlex.quote(expected_sha256)} "
                    f"{shlex.quote(temporary_path)} | sha256sum --check --status && "
                    f"mv -f -- {shlex.quote(temporary_path)} {shlex.quote(remote_path)}"
                )
                verify_status = _ssh_command(
                    host,
                    port,
                    verify_and_move,
                    remaining_seconds(),
                    allowed_exit_codes=(1,),
                    known_hosts_path=known_hosts_path,
                )
                if verify_status != 0:
                    raise WorkerStagingIntegrityError(
                        "uploaded worker file failed its frozen remote SHA-256 check"
                    )
            verified_paths.append(remote_path)
            verified_bytes += size_bytes
            self._update_attempt(
                attempt_id,
                upload_verified_files=verified_paths,
                upload_verified_bytes=verified_bytes,
                updated_at_utc=utc_now(),
            )

        expected_value = record.get("expected_artifacts")
        if not isinstance(expected_value, list) or any(
            not isinstance(value, str) for value in expected_value
        ):
            raise CampaignError("owned attempt expected artifact list is malformed")
        expected_json = json.dumps(expected_value, sort_keys=True, separators=(",", ":"))
        input_json = json.dumps(
            [
                {
                    "path": str(row["remote_path"]),
                    "sha256": str(row["source_sha256"]),
                    "size_bytes": int(row["size_bytes"]),
                }
                for row in inputs
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        work_dir = f"{remote_work_dir}/work"
        hard_stop = _parse_time(record.get("hard_stop_utc"), "hard_stop_utc")
        remaining_hard_stop = int((hard_stop - datetime.now(UTC)).total_seconds())
        maximum_runtime = record.get("max_runtime_seconds")
        worker_setup_timeout = record.get("worker_setup_timeout_seconds", 0)
        if (
            isinstance(maximum_runtime, bool)
            or not isinstance(maximum_runtime, int)
            or maximum_runtime <= 0
        ):
            raise CampaignError("owned attempt worker runtime ceiling is malformed")
        if (
            isinstance(worker_setup_timeout, bool)
            or not isinstance(worker_setup_timeout, int)
            or worker_setup_timeout < 0
            or worker_setup_timeout > MAX_WORKER_SETUP_TIMEOUT_SECONDS
        ):
            raise CampaignError("owned attempt worker setup timeout is malformed")
        worker_runtime_seconds = min(
            maximum_runtime,
            remaining_hard_stop - collection_timeout_seconds - worker_setup_timeout,
        )
        # Reserve bounded packet setup and collection before handing the remaining
        # worker deadline to the packet. The H217 worker then reserves 90 seconds
        # inside WNN_MAX_RUNTIME_SECONDS for its own finalization.
        if worker_runtime_seconds <= 90:
            raise CampaignError("insufficient hard-stop runway for worker and artifact collection")
        environment = {
            "WNN_ATTEMPT_ID": attempt_id,
            "WNN_PACKET_SHA256": packet_sha256,
            "WNN_OUTPUT_DIR": output_root,
            "WNN_INPUT_DIR": f"{remote_work_dir}/inputs",
            "WNN_EXPECTED_ARTIFACTS": expected_json,
            "WNN_INPUT_MANIFEST": input_json,
            "WNN_MAX_RUNTIME_SECONDS": str(worker_runtime_seconds),
        }
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items())
        remote_packet = f"{remote_work_dir}/packet.tar"
        remote_run = (
            "set -eu; "
            f"mkdir -p -- {shlex.quote(work_dir)}; "
            f"tar --no-same-owner -xf {shlex.quote(remote_packet)} -C {shlex.quote(work_dir)}; "
            f"test -f {shlex.quote(work_dir + '/run.sh')}; "
            f"cd {shlex.quote(work_dir)}; "
            f"nohup env {exports} bash ./run.sh "
            f"> {shlex.quote(output_root + '/worker.log')} 2>&1 < /dev/null &"
        )
        self._update_attempt(
            attempt_id,
            status="worker_launch_uncertain",
            worker_launch_status="launch_uncertain",
            worker_launch_requested_at_utc=utc_now(),
            worker_setup_timeout_seconds=worker_setup_timeout,
            worker_runtime_seconds=worker_runtime_seconds,
            remote_work_dir=remote_work_dir,
            upload_completed_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        _ssh_command(
            host,
            port,
            remote_run,
            min(30, remaining_seconds()),
            known_hosts_path=known_hosts_path,
        )
        result = self._update_attempt(
            attempt_id,
            status="worker_running",
            worker_launch_status="worker_started",
            worker_started_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        return _status_view(result)

    def _worker_manifest_is_terminal(
        self,
        attempt_id: str,
        destination: Path,
        downloader: Callable[[str, int, str, Path, int, int, Path], None] | None,
    ) -> bool:
        """Read only the owned worker's small manifest to detect terminal output."""
        record = self._get_attempt(attempt_id)
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str):
            return False
        user = self.api.current_user()
        self._require_owned_scope(record, _account_scope(user))
        instance = _wrapped_instance(self.api.get_instance(instance_id))
        self._assert_provider_ownership(record, instance)
        host, port = _ssh_endpoint(instance)
        known_hosts_path = self._campaign_known_hosts(attempt_id, instance, allow_first_use=False)
        output_root = _valid_remote_root(record.get("remote_output_dir"))
        manifest_path = destination / "artifact_manifest.json"
        manifest_path.unlink(missing_ok=True)
        fetch = downloader or _rsync_fetch
        fetch(
            host,
            port,
            f"{output_root}/artifact_manifest.json",
            manifest_path,
            60,
            5_000_000,
            known_hosts_path,
        )
        file_stat = manifest_path.lstat()
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > 5_000_000:
            raise CampaignError("worker manifest probe is not a bounded regular file")
        manifest = _read_object(manifest_path)
        if manifest.get("attempt_id") != attempt_id or manifest.get("packet_sha256") != record.get(
            "packet_sha256"
        ):
            raise CampaignError("worker manifest probe is not bound to this frozen attempt")
        return manifest.get("worker_terminal") is True

    def run_attempt(
        self,
        intent_path: Path,
        admission_path: Path,
        dispatch_receipt_path: Path,
        destination: Path,
        *,
        poll_interval_seconds: int = 30,
        collection_timeout_seconds: int = 300,
        prefetch_plan_path: Path | None = None,
        prefetch_admission_path: Path | None = None,
        downloader: Callable[[str, int, str, Path, int, int, Path], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> dict[str, object]:
        """Create once, resume an existing receipt and supervise it to verified teardown."""
        intent_path = intent_path.resolve(strict=True)
        normalized = _validate_intent(_read_object(intent_path), intent_path)
        attempt_id = str(normalized["attempt_id"])
        with ledger_transaction(self.ledger_path) as state:
            existing = _attempts(state).get(attempt_id)
            existing_record = None if existing is None else dict(existing)
        if existing_record is None:
            try:
                self._create_one_unarmed(intent_path, admission_path, dispatch_receipt_path)
            except CampaignError:
                with ledger_transaction(self.ledger_path) as state:
                    persisted = _attempts(state).get(attempt_id)
                if persisted is None:
                    raise
        elif existing_record.get("intent_sha256") != normalized["intent_sha256"]:
            raise CampaignError(
                "existing attempt intent differs; never create or resume changed work"
            )
        return self.supervise(
            attempt_id,
            destination,
            poll_interval_seconds=poll_interval_seconds,
            collection_timeout_seconds=collection_timeout_seconds,
            prefetch_plan_path=prefetch_plan_path,
            prefetch_admission_path=prefetch_admission_path,
            downloader=downloader,
            sleep_fn=sleep_fn,
        )

    def supervise(
        self,
        attempt_id: str,
        destination: Path,
        *,
        poll_interval_seconds: int = 30,
        collection_timeout_seconds: int = 300,
        prefetch_plan_path: Path | None = None,
        prefetch_admission_path: Path | None = None,
        downloader: Callable[[str, int, str, Path, int, int, Path], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] | None = None,
    ) -> dict[str, object]:
        """Poll one ledger-owned instance, collect terminal artifacts, and destroy it."""
        if poll_interval_seconds <= 0 or poll_interval_seconds > 300:
            raise ValueError("supervisor poll interval must be in (0, 300] seconds")
        if collection_timeout_seconds <= 0 or collection_timeout_seconds > 3600:
            raise ValueError("collection timeout must be in (0, 3600] seconds")
        now = now_fn or (lambda: datetime.now(UTC))
        record = self._get_attempt(attempt_id)
        hard_stop = _parse_time(record.get("hard_stop_utc"), "hard_stop_utc")
        cleanup_deadline = _parse_time(record.get("cleanup_deadline_utc"), "cleanup_deadline_utc")
        if cleanup_deadline < hard_stop:
            raise CampaignError("owned attempt cleanup deadline precedes its hard stop")
        output = destination.resolve()
        if not output.is_relative_to(ROOT):
            raise CampaignError("supervisor destination must remain inside the project root")
        scratch_root = ROOT / ".runtime" / "vast-supervisor"
        scratch_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{attempt_id}-", dir=scratch_root) as scratch:
            manifest_probe_dir = Path(scratch)
            while True:
                record = self._get_attempt(attempt_id)
                current_time = now()
                if current_time.tzinfo is None:
                    raise ValueError("supervisor clock must return a timezone-aware datetime")
                if record.get("status") in {"destroyed_verified", "settled"}:
                    return _status_view(record)
                instance_id = record.get("instance_id")
                if not isinstance(instance_id, str):
                    try:
                        record = self.reconcile(attempt_id)
                    except (CampaignError, subprocess.SubprocessError) as error:
                        self._update_attempt(
                            attempt_id,
                            supervisor_last_error_type=type(error).__name__,
                            supervisor_last_phase="reconcile",
                            supervisor_last_poll_at_utc=utc_now(),
                            updated_at_utc=utc_now(),
                        )
                    instance_id = record.get("instance_id")
                if isinstance(instance_id, str):
                    try:
                        record = self.attempt_status(attempt_id)
                    except (CampaignError, subprocess.SubprocessError) as error:
                        record = self._update_attempt(
                            attempt_id,
                            supervisor_last_error_type=type(error).__name__,
                            supervisor_last_phase="provider_status",
                            supervisor_last_poll_at_utc=utc_now(),
                            updated_at_utc=utc_now(),
                        )
                    collection_attempts = record.get("collection_attempts", 0)
                    collection_retry_allowed = (
                        isinstance(collection_attempts, int)
                        and not isinstance(collection_attempts, bool)
                        and collection_attempts < MAX_COLLECTION_ATTEMPTS
                    )
                    if (
                        current_time >= hard_stop
                        or record.get("status")
                        in {
                            "collected",
                            "destroy_pending",
                        }
                        or record.get("worker_terminal_failure") is True
                        or record.get("collection_terminal_failure") is True
                    ):
                        try:
                            record = self.destroy_owned(attempt_id, instance_id)
                        except (CampaignError, subprocess.SubprocessError) as error:
                            record = self._update_attempt(
                                attempt_id,
                                supervisor_last_error_type=type(error).__name__,
                                supervisor_last_phase="destroy",
                                supervisor_last_poll_at_utc=utc_now(),
                                updated_at_utc=utc_now(),
                            )
                    else:
                        if record.get("worker_launch_status") not in {
                            "launch_uncertain",
                            "worker_started",
                        }:
                            try:
                                record = self._stage_and_launch(
                                    attempt_id,
                                    collection_timeout_seconds=collection_timeout_seconds,
                                    prefetch_plan_path=prefetch_plan_path,
                                    prefetch_admission_path=prefetch_admission_path,
                                )
                            except (CampaignError, subprocess.SubprocessError) as error:
                                stage_record = self._get_attempt(attempt_id)
                                stage_attempts = stage_record.get("worker_stage_attempts", 0)
                                terminal_stage_failure = stage_record.get(
                                    "worker_launch_status"
                                ) not in {"launch_uncertain", "worker_started"} and (
                                    isinstance(
                                        error, (WorkerStagingIntegrityError, WorkerStartupFailure)
                                    )
                                    or "total upload deadline expired" in str(error)
                                    or "insufficient hard-stop runway" in str(error)
                                    or "staging retry limit reached" in str(error)
                                    or (
                                        isinstance(stage_attempts, int)
                                        and not isinstance(stage_attempts, bool)
                                        and stage_attempts >= 2
                                    )
                                )
                                record = self._update_attempt(
                                    attempt_id,
                                    status=(
                                        "failed_before_worker"
                                        if terminal_stage_failure
                                        else record.get("status")
                                    ),
                                    worker_terminal_failure=terminal_stage_failure,
                                    worker_failure_error_type=(
                                        type(error).__name__ if terminal_stage_failure else None
                                    ),
                                    supervisor_last_error_type=type(error).__name__,
                                    supervisor_last_phase=(
                                        "stage_failed_before_launch"
                                        if terminal_stage_failure
                                        else "stage_or_launch"
                                    ),
                                    supervisor_last_poll_at_utc=utc_now(),
                                    updated_at_utc=utc_now(),
                                )
                                if terminal_stage_failure:
                                    try:
                                        record = self.destroy_owned(attempt_id, instance_id)
                                    except (
                                        CampaignError,
                                        subprocess.SubprocessError,
                                    ) as destroy_error:
                                        record = self._update_attempt(
                                            attempt_id,
                                            supervisor_last_error_type=type(destroy_error).__name__,
                                            supervisor_last_phase="failed_worker_destroy",
                                            supervisor_last_poll_at_utc=utc_now(),
                                            updated_at_utc=utc_now(),
                                        )
                        if now() >= hard_stop:
                            try:
                                record = self.destroy_owned(attempt_id, instance_id)
                            except (CampaignError, subprocess.SubprocessError) as error:
                                record = self._update_attempt(
                                    attempt_id,
                                    supervisor_last_error_type=type(error).__name__,
                                    supervisor_last_phase="hard_stop_destroy",
                                    supervisor_last_poll_at_utc=utc_now(),
                                    updated_at_utc=utc_now(),
                                )
                        elif (
                            record.get("worker_terminal_failure") is not True
                            and record.get("collection_terminal_failure") is not True
                            and record.get("worker_launch_status")
                            in {"launch_uncertain", "worker_started"}
                            and collection_retry_allowed
                        ):
                            try:
                                terminal = self._worker_manifest_is_terminal(
                                    attempt_id, manifest_probe_dir, downloader
                                )
                            except (CampaignError, subprocess.SubprocessError) as error:
                                terminal = False
                                record = self._update_attempt(
                                    attempt_id,
                                    supervisor_last_error_type=type(error).__name__,
                                    supervisor_last_phase="manifest_probe",
                                    supervisor_last_poll_at_utc=utc_now(),
                                    updated_at_utc=utc_now(),
                                )
                            if terminal:
                                try:
                                    record = self.collect(
                                        attempt_id,
                                        output,
                                        downloader=downloader,
                                        timeout_seconds=collection_timeout_seconds,
                                    )
                                    record = self.destroy_owned(attempt_id, instance_id)
                                except (CampaignError, subprocess.SubprocessError) as error:
                                    failed_collection = self._get_attempt(attempt_id)
                                    collection_attempts = failed_collection.get(
                                        "collection_attempts", 0
                                    )
                                    terminal_collection_failure = (
                                        isinstance(collection_attempts, int)
                                        and not isinstance(collection_attempts, bool)
                                        and collection_attempts >= MAX_COLLECTION_ATTEMPTS
                                    )
                                    record = self._update_attempt(
                                        attempt_id,
                                        collection_terminal_failure=terminal_collection_failure,
                                        supervisor_last_error_type=type(error).__name__,
                                        supervisor_last_phase="collect_or_destroy",
                                        supervisor_last_poll_at_utc=utc_now(),
                                        updated_at_utc=utc_now(),
                                    )
                current_time = now()
                record = self._get_attempt(attempt_id)
                if record.get("status") == "destroyed_verified":
                    return _status_view(record)
                if current_time >= cleanup_deadline:
                    if isinstance(record.get("instance_id"), str):
                        try:
                            record = self.destroy_owned(
                                attempt_id, cast(str, record["instance_id"])
                            )
                        except (CampaignError, subprocess.SubprocessError) as error:
                            record = self._update_attempt(
                                attempt_id,
                                supervisor_last_error_type=type(error).__name__,
                                supervisor_last_phase="final_destroy",
                                supervisor_last_poll_at_utc=utc_now(),
                                updated_at_utc=utc_now(),
                            )
                    if record.get("status") == "destroyed_verified":
                        return _status_view(record)
                    self._update_attempt(
                        attempt_id,
                        supervisor_last_phase="cleanup_deadline_exhausted",
                        supervisor_last_poll_at_utc=utc_now(),
                        updated_at_utc=utc_now(),
                    )
                    raise CampaignError(
                        "Vast supervisor reached cleanup deadline without verified teardown; root intervention required"
                    )
                remaining = max(0.0, (cleanup_deadline - current_time).total_seconds())
                self._update_attempt(
                    attempt_id,
                    supervisor_last_phase=(
                        "destroy_pending"
                        if record.get("status") == "destroy_pending"
                        else "monitoring"
                    ),
                    supervisor_last_poll_at_utc=utc_now(),
                    updated_at_utc=utc_now(),
                )
                sleep_fn(min(float(poll_interval_seconds), remaining))

    def collect(
        self,
        attempt_id: str,
        destination: Path,
        downloader: Callable[[str, int, str, Path, int, int, Path], None] | None = None,
        timeout_seconds: int = 1800,
    ) -> dict[str, object]:
        """Download a worker manifest and exact expected artifacts, verifying size and SHA-256."""
        if timeout_seconds <= 0 or timeout_seconds > 3600:
            raise ValueError("collection timeout must be in (0, 3600] seconds")
        record = self._get_attempt(attempt_id)
        collection_attempts = record.get("collection_attempts", 0)
        if (
            isinstance(collection_attempts, bool)
            or not isinstance(collection_attempts, int)
            or collection_attempts < 0
        ):
            raise CampaignError("persisted collection attempt count is malformed")
        if collection_attempts >= MAX_COLLECTION_ATTEMPTS:
            raise CampaignError("bounded artifact collection retry limit reached")
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or INSTANCE_ID_RE.fullmatch(instance_id) is None:
            raise CampaignError("collection requires a persisted, owned instance ID")
        user = self.api.current_user()
        self._require_owned_scope(record, _account_scope(user))
        provider_record = _wrapped_instance(self.api.get_instance(instance_id))
        self._assert_provider_ownership(record, provider_record)
        host, port = _ssh_endpoint(provider_record)
        known_hosts_path = self._campaign_known_hosts(
            attempt_id, provider_record, allow_first_use=False
        )
        expected = record.get("expected_artifacts")
        if not isinstance(expected, list) or any(not isinstance(value, str) for value in expected):
            raise CampaignError("persisted expected_artifacts is malformed")
        output_root = _valid_remote_root(record.get("remote_output_dir"))
        limit = record.get("max_artifact_bytes")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise CampaignError("persisted artifact byte ceiling is malformed")
        output = destination.resolve()
        if not output.is_relative_to(ROOT):
            raise CampaignError("artifact destination must remain inside the project root")
        self._quarantine_failed_collection(attempt_id, output, collection_attempts)
        output.mkdir(parents=True, exist_ok=True)
        fetch = downloader or _rsync_fetch
        manifest_path = output / "artifact_manifest.json"
        manifest_remote = f"{output_root}/artifact_manifest.json"
        self._update_attempt(
            attempt_id,
            status="collecting",
            collection_attempts=collection_attempts + 1,
            collection_started_at_utc=utc_now(),
            collection_last_started_at_utc=utc_now(),
            collection_destination=str(output),
            updated_at_utc=utc_now(),
        )
        try:
            fetch(
                host,
                port,
                manifest_remote,
                manifest_path,
                timeout_seconds,
                5_000_000,
                known_hosts_path,
            )
            manifest_stat = manifest_path.lstat()
            if not stat.S_ISREG(manifest_stat.st_mode) or manifest_stat.st_size > 5_000_000:
                raise CampaignError("downloaded worker manifest is not a bounded regular file")
            manifest = _read_object(manifest_path)
            if manifest.get("attempt_id") != attempt_id:
                raise CampaignError("worker artifact manifest attempt_id mismatch")
            if manifest.get("packet_sha256") != record.get("packet_sha256"):
                raise CampaignError("worker artifact manifest packet SHA-256 mismatch")
            if manifest.get("worker_terminal") is not True:
                raise CampaignError("worker artifact manifest does not declare a terminal worker")
            files_value = manifest.get("files")
            if not isinstance(files_value, list) or any(
                not isinstance(row, dict) for row in files_value
            ):
                raise CampaignError("worker artifact manifest files list is malformed")
            file_map: dict[str, dict[str, object]] = {}
            for row in files_value:
                name = _valid_remote_relative(row.get("path"))
                if name in file_map:
                    raise CampaignError("worker artifact manifest contains duplicate paths")
                file_map[name] = row
            if set(file_map) != set(expected):
                raise CampaignError("worker outputs differ from the frozen expected artifact set")
            total_size = 0
            verified: list[dict[str, object]] = []
            for relative in expected:
                row = file_map[relative]
                expected_size = row.get("size_bytes")
                if (
                    isinstance(expected_size, bool)
                    or not isinstance(expected_size, int)
                    or expected_size < 0
                ):
                    raise CampaignError(f"artifact size is invalid for {relative}")
                expected_hash = _sha256(row.get("sha256"), f"artifact {relative} sha256")
                total_size += expected_size
                if total_size > limit:
                    raise CampaignError("artifact manifest exceeds the frozen total-byte ceiling")
                local_path = output.joinpath(*PurePosixPath(relative).parts)
                if not local_path.resolve().is_relative_to(output):
                    raise CampaignError("local artifact path escapes its attempt directory")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                fetch(
                    host,
                    port,
                    f"{output_root}/{relative}",
                    local_path,
                    timeout_seconds,
                    expected_size,
                    known_hosts_path,
                )
                file_stat = local_path.lstat()
                if not stat.S_ISREG(file_stat.st_mode):
                    raise CampaignError(f"downloaded artifact is not a regular file: {relative}")
                actual_size = file_stat.st_size
                actual_hash = file_sha256(local_path)
                if actual_size != expected_size:
                    raise CampaignError(f"artifact size mismatch for {relative}")
                if actual_hash != expected_hash:
                    raise CampaignError(f"artifact SHA-256 mismatch for {relative}")
                verified.append(
                    {"path": relative, "size_bytes": actual_size, "sha256": actual_hash}
                )
            manifest_hash = file_sha256(manifest_path)
        except (OSError, CampaignError, subprocess.SubprocessError) as error:
            self._update_attempt(
                attempt_id,
                status="needs_review",
                collection_error_type=type(error).__name__,
                updated_at_utc=utc_now(),
            )
            raise CampaignError(
                f"artifact collection failed for {attempt_id}: {type(error).__name__}; preserve partial files"
            ) from None
        result = self._update_attempt(
            attempt_id,
            status="collected",
            collection_manifest_sha256=manifest_hash,
            collected_files=verified,
            collected_bytes=total_size,
            collection_completed_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        return _status_view(result)

    def _quarantine_failed_collection(
        self, attempt_id: str, output: Path, previous_attempts: int
    ) -> None:
        """Preserve a failed partial download before retrying into a fresh destination."""
        if not output.exists():
            return
        if output.is_symlink() or not output.is_dir():
            raise CampaignError("collection destination must be a regular directory")
        if not any(output.iterdir()):
            return
        if previous_attempts == 0:
            raise CampaignError(
                "collection destination is not empty; never overwrite another attempt"
            )
        quarantine = output.with_name(f"{output.name}.collection-failed-{previous_attempts}")
        if quarantine.exists() or quarantine.is_symlink():
            raise CampaignError("failed collection quarantine already exists; preserve both paths")
        output.rename(quarantine)
        record = self._get_attempt(attempt_id)
        prior = record.get("collection_quarantine_paths", [])
        if not isinstance(prior, list) or any(not isinstance(value, str) for value in prior):
            raise CampaignError("persisted collection quarantine list is malformed")
        self._update_attempt(
            attempt_id,
            collection_quarantine_paths=[*prior, str(quarantine)],
            collection_last_quarantined_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )

    def destroy_owned(self, attempt_id: str, instance_id: str) -> dict[str, object]:
        """Delete and verify only the exact account-scoped instance in this creation receipt."""
        if INSTANCE_ID_RE.fullmatch(instance_id) is None:
            raise CampaignError("destroy requires a positive numeric instance ID")
        record = self._get_attempt(attempt_id)
        receipt_id = record.get("instance_id")
        if receipt_id != instance_id:
            raise CampaignError("instance ID is not in this attempt's persisted creation receipt")
        try:
            hard_stop = _parse_time(record.get("hard_stop_utc"), "hard_stop_utc")
            cleanup_deadline = _parse_time(
                record.get("cleanup_deadline_utc"), "cleanup_deadline_utc"
            )
        except CampaignError:
            raise CampaignError(
                "owned instance receipt has no valid hard teardown deadline"
            ) from None
        if cleanup_deadline < hard_stop:
            raise CampaignError("owned instance receipt cleanup deadline precedes its hard stop")
        failed_before_worker = record.get("worker_terminal_failure") is True and record.get(
            "worker_launch_status"
        ) not in {"launch_uncertain", "worker_started"}
        terminal_collection_failure = record.get("collection_terminal_failure") is True
        if (
            record.get("status") != "collected"
            and not failed_before_worker
            and not terminal_collection_failure
            and datetime.now(UTC) < hard_stop
        ):
            raise CampaignError(
                "collect and verify terminal artifacts before teardown, or wait until the hard stop"
            )
        user = self.api.current_user()
        self._require_owned_scope(record, _account_scope(user))
        instances = self.api.list_instances()
        matching = [row for row in instances if str(row.get("id")) == instance_id]
        if len(matching) > 1:
            raise CampaignError("provider returned duplicate rows for the owned instance ID")
        if not matching:
            updated = self._update_attempt(
                attempt_id,
                status="destroyed_verified",
                destroy_verified_at_utc=utc_now(),
                destroy_verification="instance absent from successful account instance listing",
                updated_at_utc=utc_now(),
            )
            return _status_view(updated)
        self._assert_provider_ownership(record, matching[0])
        self._update_attempt(
            attempt_id,
            status="destroy_pending",
            destroy_requested_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        try:
            self.api.destroy_instance(instance_id)
        except Exception as error:
            self._update_attempt(
                attempt_id,
                status="destroy_pending",
                destroy_error_type=type(error).__name__,
                updated_at_utc=utc_now(),
            )
        verify_user = self.api.current_user()
        self._require_owned_scope(record, _account_scope(verify_user))
        remaining = self.api.list_instances()
        still_present = [row for row in remaining if str(row.get("id")) == instance_id]
        if still_present:
            self._assert_provider_ownership(record, still_present[0])
            updated = self._update_attempt(
                attempt_id,
                status="destroy_pending",
                destroy_verified_at_utc=None,
                destroy_verification="instance remains in account listing after delete request",
                updated_at_utc=utc_now(),
            )
            return _status_view(updated)
        updated = self._update_attempt(
            attempt_id,
            status="destroyed_verified",
            destroy_verified_at_utc=utc_now(),
            destroy_verification="instance absent from successful account instance listing",
            updated_at_utc=utc_now(),
        )
        return _status_view(updated)

    def settle(self, attempt_id: str, receipt_path: Path) -> dict[str, object]:
        """Release unused reservation only from a root-authored final billing receipt."""
        receipt = _read_object(receipt_path)
        record = self._get_attempt(attempt_id)
        no_instance_settlement = (
            record.get("status") == "needs_review"
            and record.get("instance_id") is None
            and receipt.get("provider_absence_verified") is True
            and receipt.get("provider_terminal_confirmed") is True
        )
        if record.get("status") != "destroyed_verified" and not no_instance_settlement:
            raise CampaignError(
                "settlement requires verified teardown or root-confirmed no-instance reconciliation"
            )
        receipt_hash = file_sha256(receipt_path)
        required = {
            "schema_version": 1,
            "authorized_by": "root_controller",
            "decision": "settle_vast_attempt",
            "attempt_id": attempt_id,
            "instance_id": record.get("instance_id"),
            "account_scope_sha256": record.get("account_scope_sha256"),
            "job_sha256": record.get("job_sha256"),
        }
        if any(receipt.get(key) != value for key, value in required.items()):
            raise CampaignError("root settlement receipt does not bind this destroyed attempt")
        _sha256(receipt.get("billing_source_sha256"), "billing source SHA-256")
        actual = _money(receipt.get("actual_all_in_usd"), "actual_all_in_usd")
        reserved = _money(record.get("reserved_all_in_usd"), "reserved_all_in_usd")
        if actual > reserved:
            raise CampaignError("actual all-in spend exceeds the attempt's reserved ceiling")
        updated = self._update_attempt(
            attempt_id,
            status="settled",
            actual_all_in_usd=_money_text(actual),
            settlement_receipt_sha256=receipt_hash,
            settled_at_utc=utc_now(),
            updated_at_utc=utc_now(),
        )
        return _status_view(updated)

    def _get_attempt(self, attempt_id: str) -> dict[str, object]:
        if ATTEMPT_RE.fullmatch(attempt_id) is None:
            raise CampaignError("attempt ID is invalid")
        with ledger_transaction(self.ledger_path) as state:
            row = _attempts(state).get(attempt_id)
            if row is None:
                raise CampaignError("attempt has no local creation receipt")
            return dict(row)

    def _update_attempt(self, attempt_id: str, **updates: object) -> dict[str, object]:
        with ledger_transaction(self.ledger_path) as state:
            row = _attempts(state).get(attempt_id)
            if row is None:
                raise CampaignError("attempt disappeared from durable ledger")
            row.update(updates)
            return dict(row)

    @staticmethod
    def _require_owned_scope(record: Mapping[str, object], current_scope: str) -> None:
        if record.get("account_scope_sha256") != current_scope:
            raise CampaignError("current account scope differs from the persisted creation receipt")

    @staticmethod
    def _assert_provider_ownership(
        record: Mapping[str, object], provider_record: Mapping[str, object]
    ) -> None:
        expected_id = record.get("instance_id")
        if str(provider_record.get("id")) != expected_id:
            raise CampaignError("provider instance ID does not match persisted creation receipt")
        if provider_record.get("label") != record.get("campaign_label"):
            raise CampaignError("provider label does not match persisted campaign label")
        if provider_record.get("machine_id") != record.get("machine_id"):
            raise CampaignError("provider machine ID does not match persisted selected offer")


def _status_view(record: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "attempt_id",
        "campaign_label",
        "status",
        "instance_id",
        "offer_id",
        "job_sha256",
        "packet_sha256",
        "reserved_all_in_usd",
        "actual_all_in_usd",
        "last_provider_status",
        "last_provider_observed_at_utc",
        "upload_total_bytes",
        "upload_started_at_utc",
        "upload_verified_files",
        "upload_verified_bytes",
        "prefetch_status",
        "prefetch_admission_sha256",
        "prefetch_private_plan_sha256",
        "prefetch_remote_program_sha256",
        "prefetch_elapsed_ms",
        "prefetch_bytes",
        "prefetch_receipt",
        "prefetch_error_type",
        "worker_stage_attempts",
        "max_runtime_seconds",
        "worker_setup_timeout_seconds",
        "worker_runtime_seconds",
        "worker_startup_status",
        "worker_startup_deadline_utc",
        "worker_launch_status",
        "worker_terminal_failure",
        "ssh_key_attach_status",
        "known_hosts_fingerprints",
        "worker_failure_error_type",
        "worker_started_at_utc",
        "supervisor_last_phase",
        "supervisor_last_error_type",
        "supervisor_last_poll_at_utc",
        "collected_files",
        "collected_bytes",
        "collection_manifest_sha256",
        "collection_attempts",
        "collection_terminal_failure",
        "collection_quarantine_paths",
        "destroy_verification",
        "hard_stop_utc",
        "cleanup_deadline_utc",
        "settlement_receipt_sha256",
        "updated_at_utc",
    )
    return {field: record.get(field) for field in fields}


def _ssh_endpoint(instance: Mapping[str, object]) -> tuple[str, int]:
    proxy_host = instance.get("ssh_host")
    proxy_port = instance.get("ssh_port")
    if (proxy_host is None) != (proxy_port is None):
        raise CampaignError("provider instance has an incomplete paired SSH proxy endpoint")
    if proxy_host is not None and proxy_port is not None:
        host_value, port_value = proxy_host, proxy_port
    else:
        host_value = instance.get("public_ipaddr")
        mappings = instance.get("ports")
        mapped_ssh: object = None
        if isinstance(mappings, Mapping):
            candidates = mappings.get("22/tcp")
            if isinstance(candidates, list) and len(candidates) == 1:
                candidate = candidates[0]
                if isinstance(candidate, Mapping):
                    mapped_ssh = candidate.get("HostPort")
            elif isinstance(candidates, Mapping):
                mapped_ssh = candidates.get("HostPort")
        port_value = mapped_ssh
    if not isinstance(host_value, str) or re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host_value) is None:
        raise CampaignError("provider instance lacks a safe public SSH hostname")
    if isinstance(port_value, bool) or not isinstance(port_value, (str, int)):
        raise CampaignError("provider instance lacks a numeric SSH port")
    try:
        port = int(port_value)
    except ValueError:
        raise CampaignError("provider instance lacks a numeric SSH port") from None
    if not 1 <= port <= 65535:
        raise CampaignError("provider SSH port is outside TCP range")
    return host_value, port


def _ssh_command(
    host: str,
    port: int,
    remote_command: str,
    timeout: int,
    *,
    known_hosts_path: Path,
    allowed_exit_codes: tuple[int, ...] = (),
) -> int:
    """Run a bounded strict-host-key command and return its validated exit status."""
    command = _pinned_ssh_prefix(host, port, known_hosts_path) + [remote_command]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    allowed = {0, *allowed_exit_codes}
    if result.returncode not in allowed:
        raise CampaignError(f"SSH command failed with exit status {result.returncode}")
    return result.returncode


def _ssh_stdin(
    host: str,
    port: int,
    remote_command: str,
    timeout: int,
    *,
    known_hosts_path: Path,
    input_bytes: bytes,
    allowed_exit_codes: tuple[int, ...] = (),
) -> str:
    """Send bounded private input over pinned SSH stdin and return safe stdout only."""
    if re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host) is None or not 1 <= port <= 65535:
        raise CampaignError("unsafe SSH endpoint")
    if timeout <= 0 or timeout > kaggle_vast_prefetch.MAX_PREFETCH_SECONDS:
        raise CampaignError("signed prefetch timeout exceeds its hard limit")
    if len(input_bytes) > kaggle_vast_prefetch.MAX_REMOTE_PAYLOAD_BYTES:
        raise CampaignError("signed prefetch payload exceeds its hard limit")
    command = _pinned_ssh_prefix(host, port, known_hosts_path) + [remote_command]
    try:
        result = subprocess.run(
            command,
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise CampaignError("pinned SSH stdin transport failed") from None
    if result.returncode not in {0, *allowed_exit_codes}:
        raise CampaignError("pinned SSH stdin command failed")
    try:
        return result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise CampaignError("pinned SSH stdin command returned a malformed receipt") from None


def _pinned_ssh_prefix(host: str, port: int, known_hosts_path: Path) -> list[str]:
    """Construct the shared strict-host-key SSH argv prefix for this campaign."""
    if re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host) is None or not 1 <= port <= 65535:
        raise CampaignError("unsafe SSH endpoint")
    return [
        "ssh",
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts_path}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-i",
        str(DEFAULT_SSH_PUBLIC_KEY.with_suffix("")),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
        "root@" + host,
    ]


def _rsync_push(
    host: str,
    port: int,
    source: Path,
    remote_path: str,
    timeout: int,
    max_bytes: int,
    known_hosts_path: Path,
) -> None:
    """Upload one exact file over strict-host-key SSH without following links."""
    if re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host) is None or not 1 <= port <= 65535:
        raise CampaignError("unsafe SSH endpoint")
    source = source.resolve(strict=True)
    source_stat = source.lstat()
    if not stat.S_ISREG(source_stat.st_mode):
        raise CampaignError("upload source must be a regular file")
    if isinstance(max_bytes, bool) or max_bytes < source_stat.st_size:
        raise CampaignError("upload source exceeds its frozen byte ceiling")
    if (
        not remote_path.startswith("/")
        or SAFE_REMOTE_PART_RE.fullmatch(remote_path.lstrip("/")) is None
    ):
        raise CampaignError("unsafe remote upload path")
    ssh = " ".join(
        (
            "ssh",
            "-p",
            str(port),
            "-o BatchMode=yes",
            "-o StrictHostKeyChecking=yes",
            f"-o UserKnownHostsFile={shlex.quote(str(known_hosts_path))}",
            "-o GlobalKnownHostsFile=/dev/null",
            "-i",
            shlex.quote(str(DEFAULT_SSH_PUBLIC_KEY.with_suffix(""))),
            "-o IdentitiesOnly=yes",
            "-o ConnectTimeout=10",
        )
    )
    command = [
        "rsync",
        "--archive",
        "--no-owner",
        "--no-group",
        "--no-perms",
        "--protect-args",
        "--partial",
        "--append-verify",
        f"--max-size={max_bytes}",
        "--timeout=60",
        "-e",
        ssh,
        str(source),
        f"root@{host}:{remote_path}",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise CampaignError(f"rsync upload failed with exit status {result.returncode}")


def _validate_worker_bundle(path: Path, max_unpacked_bytes: int) -> int:
    """Require a plain tar bundle with one root run.sh and no path escapes or secrets."""
    secret_names = {".env", ".vast_api_key", "id_rsa", "id_ed25519", "credentials.json"}
    unpacked_bytes = 0
    run_script_count = 0
    try:
        with tarfile.open(path, mode="r:") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as error:
        raise CampaignError("worker packet must be an uncompressed tar archive") from error
    if not members:
        raise CampaignError("worker packet archive is empty")
    for member in members:
        relative = PurePosixPath(member.name)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise CampaignError("worker packet contains an unsafe archive path")
        if any(
            part.lower() in secret_names
            or part.lower().startswith(".env")
            or "api_key" in part.lower()
            or "credential" in part.lower()
            or part.lower() in {"secret", "secrets"}
            for part in relative.parts
        ):
            raise CampaignError("worker packet contains a credential-named file")
        if not (member.isfile() or member.isdir()) or member.issym() or member.islnk():
            raise CampaignError("worker packet may contain only regular files and directories")
        if member.isfile():
            unpacked_bytes += member.size
            if member.name == "run.sh":
                run_script_count += 1
    if unpacked_bytes > max_unpacked_bytes:
        raise CampaignError("worker packet unpacked bytes exceed the frozen upload ceiling")
    if run_script_count != 1:
        raise CampaignError("worker packet must contain exactly one root-level run.sh")
    return unpacked_bytes


def _rsync_fetch(
    host: str,
    port: int,
    remote_path: str,
    destination: Path,
    timeout: int,
    max_bytes: int,
    known_hosts_path: Path,
) -> None:
    """Fetch one exact file over strict-host-key SSH without shell interpolation."""
    if re.fullmatch(r"[A-Za-z0-9.-]{1,253}", host) is None or not 1 <= port <= 65535:
        raise CampaignError("unsafe SSH endpoint")
    if isinstance(max_bytes, bool) or max_bytes < 0:
        raise CampaignError("transfer byte ceiling must be a nonnegative integer")
    if (
        not remote_path.startswith("/")
        or SAFE_REMOTE_PART_RE.fullmatch(remote_path.lstrip("/")) is None
    ):
        raise CampaignError("unsafe remote artifact path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ssh = " ".join(
        (
            "ssh",
            "-p",
            str(port),
            "-o BatchMode=yes",
            "-o StrictHostKeyChecking=yes",
            f"-o UserKnownHostsFile={shlex.quote(str(known_hosts_path))}",
            "-o GlobalKnownHostsFile=/dev/null",
            "-i",
            shlex.quote(str(DEFAULT_SSH_PUBLIC_KEY.with_suffix(""))),
            "-o IdentitiesOnly=yes",
            "-o ConnectTimeout=10",
        )
    )
    command = [
        "rsync",
        "--archive",
        "--protect-args",
        "--partial",
        f"--max-size={max_bytes}",
        "--timeout=60",
        "-e",
        ssh,
        f"root@{host}:{remote_path}",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise CampaignError(f"rsync failed with exit status {result.returncode}")


def _json_file(path: Path) -> dict[str, object]:
    return _read_object(path)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--credentials", type=Path, default=ROOT / ".env")
    parser.add_argument("--timeout", type=float, default=15.0)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "inspect", help="read-only account balance and sanitized instance inventory"
    )
    search = commands.add_parser("search", help="read-only verified on-demand offer search")
    search.add_argument("--criteria", type=Path, required=True)
    create = commands.add_parser("create", help="disabled; use the supervised run command")
    create.add_argument("--intent", type=Path, required=True)
    create.add_argument("--admission", type=Path, required=True)
    create.add_argument("--dispatch-receipt", type=Path, required=True)
    create.add_argument("--confirm-create", required=True)
    run = commands.add_parser("run", help="create once, supervise, collect, and tear down")
    run.add_argument("--intent", type=Path, required=True)
    run.add_argument("--admission", type=Path, required=True)
    run.add_argument("--dispatch-receipt", type=Path, required=True)
    run.add_argument("--destination", type=Path, required=True)
    run.add_argument("--confirm-create", required=True)
    run.add_argument("--poll-interval", type=int, default=30)
    run.add_argument("--collection-timeout", type=int, default=300)
    run.add_argument("--prefetch-plan", type=Path)
    run.add_argument("--prefetch-admission", type=Path)
    supervise = commands.add_parser("supervise", help="resume one durable owned-instance monitor")
    supervise.add_argument("--attempt", required=True)
    supervise.add_argument("--destination", type=Path, required=True)
    supervise.add_argument("--poll-interval", type=int, default=30)
    supervise.add_argument("--collection-timeout", type=int, default=300)
    supervise.add_argument("--prefetch-plan", type=Path)
    supervise.add_argument("--prefetch-admission", type=Path)
    reconcile = commands.add_parser("reconcile", help="read-only reconcile an ambiguous create")
    reconcile.add_argument("--attempt", required=True)
    status = commands.add_parser("status", help="read-only status for one owned attempt")
    status.add_argument("--attempt", required=True)
    collect = commands.add_parser("collect", help="download and hash-check expected artifacts")
    collect.add_argument("--attempt", required=True)
    collect.add_argument("--destination", type=Path, required=True)
    collect.add_argument("--collection-timeout", type=int, default=1800)
    destroy = commands.add_parser("destroy", help="destroy and verify one receipt-owned instance")
    destroy.add_argument("--attempt", required=True)
    destroy.add_argument("--instance-id", required=True)
    destroy.add_argument("--confirm-destroy", required=True)
    settle = commands.add_parser("settle", help="settle actual cost from a root billing receipt")
    settle.add_argument("--attempt", required=True)
    settle.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one explicitly selected read or lifecycle operation."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "create":
            raise CampaignError(
                "direct Vast create is disabled; launch the combined run command under systemd"
            )
        if args.command == "settle":
            controller = VastCampaign(_NoApi(), args.ledger)
            _print_json(controller.settle(args.attempt, args.receipt))
            return 0
        if args.command in {"run", "supervise"} and (
            (args.prefetch_plan is None) != (args.prefetch_admission is None)
        ):
            raise CampaignError(
                "--prefetch-plan and --prefetch-admission must be supplied together"
            )
        if args.command in {"run", "supervise"} and not os.environ.get("INVOCATION_ID"):
            raise CampaignError(
                "paid lifecycle commands must run inside a systemd service with automatic restart"
            )
        if args.command == "run":
            intent = _read_object(args.intent)
            if args.confirm_create != intent.get("attempt_id"):
                raise CampaignError("--confirm-create must exactly match the frozen attempt_id")
            api = VastApi(load_api_key(args.credentials), args.timeout)
            _print_json(
                VastCampaign(api, args.ledger).run_attempt(
                    args.intent,
                    args.admission,
                    args.dispatch_receipt,
                    args.destination,
                    poll_interval_seconds=args.poll_interval,
                    collection_timeout_seconds=args.collection_timeout,
                    prefetch_plan_path=args.prefetch_plan,
                    prefetch_admission_path=args.prefetch_admission,
                )
            )
            return 0
        if args.command == "supervise":
            api = VastApi(load_api_key(args.credentials), args.timeout)
            _print_json(
                VastCampaign(api, args.ledger).supervise(
                    args.attempt,
                    args.destination,
                    poll_interval_seconds=args.poll_interval,
                    collection_timeout_seconds=args.collection_timeout,
                    prefetch_plan_path=args.prefetch_plan,
                    prefetch_admission_path=args.prefetch_admission,
                )
            )
            return 0
        api = VastApi(load_api_key(args.credentials), args.timeout)
        if args.command == "inspect":
            _print_json(inspect_account(api))
        elif args.command == "search":
            _print_json(search_verified_offers(api, _json_file(args.criteria)))
        elif args.command == "reconcile":
            _print_json(VastCampaign(api, args.ledger).reconcile(args.attempt))
        elif args.command == "status":
            _print_json(VastCampaign(api, args.ledger).attempt_status(args.attempt))
        elif args.command == "collect":
            _print_json(
                VastCampaign(api, args.ledger).collect(
                    args.attempt, args.destination, timeout_seconds=args.collection_timeout
                )
            )
        elif args.command == "destroy":
            if args.confirm_destroy != args.instance_id:
                raise CampaignError("--confirm-destroy must exactly match --instance-id")
            _print_json(
                VastCampaign(api, args.ledger).destroy_owned(args.attempt, args.instance_id)
            )
        return 0
    except (CampaignError, OSError, ValueError) as error:
        LOG.error("%s", error)
        return 2


class _NoApi:
    """API sentinel ensuring local settlement never performs a provider request."""

    @staticmethod
    def _fail(operation: str) -> Never:
        raise CampaignError(f"settlement unexpectedly attempted provider operation {operation}")

    def current_user(self) -> dict[str, object]:
        return self._fail("current_user")

    def list_instances(self) -> list[dict[str, object]]:
        return self._fail("list_instances")

    def search_offers(self, criteria: dict[str, object]) -> list[dict[str, object]]:
        del criteria
        return self._fail("search_offers")

    def search_offer_for_revalidation(
        self, offer: Mapping[str, object], allocated_storage: int
    ) -> list[dict[str, object]]:
        del offer, allocated_storage
        return self._fail("search_offer_for_revalidation")

    def create_instance(self, offer_id: int, payload: dict[str, object]) -> dict[str, object]:
        del offer_id, payload
        return self._fail("create_instance")

    def get_instance(self, instance_id: str) -> dict[str, object]:
        del instance_id
        return self._fail("get_instance")

    def destroy_instance(self, instance_id: str) -> dict[str, object]:
        del instance_id
        return self._fail("destroy_instance")

    def attach_ssh_key(self, instance_id: str, public_key: str) -> dict[str, object]:
        del instance_id, public_key
        return self._fail("attach_ssh_key")


if __name__ == "__main__":
    sys.exit(main())
