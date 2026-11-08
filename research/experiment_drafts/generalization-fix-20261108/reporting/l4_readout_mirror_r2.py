from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from collect_l4_readouts_r2 import CELLS, ENDPOINTS, MAX_FILE, MAX_RESPONSE

from scripts.sequence_monitor_mirror_io import atomic_bytes, atomic_json
from scripts.sequence_progress_report import _object

NAMES = {f"{cell}/{endpoint}/readout.json" for cell in CELLS for endpoint in ENDPOINTS}


def apply_readouts(snapshot: dict, root: Path) -> dict[str, str]:
    """Validate every source byte/hash/name before publishing immutable local receipts."""
    if snapshot.get("schema") != "wnn.l4-fixed-readout-bytes.v1" or root.is_symlink():
        raise ValueError("readout mirror schema/root differs")
    prepared = []
    seen = set()
    for item in snapshot["files"]:
        name = item["name"]
        if name not in NAMES or name in seen:
            raise ValueError("readout receipt name is not an exact fixed endpoint")
        seen.add(name)
        raw = base64.b64decode(item["data_base64"], validate=True)
        if (
            len(raw) > MAX_FILE
            or len(raw) != item["bytes"]
            or hashlib.sha256(raw).hexdigest() != item["sha256"]
        ):
            raise ValueError("readout source-byte digest/size differs")
        if not isinstance(json.loads(raw), dict):
            raise ValueError("readout receipt must be an object")
        target = root / name
        if any(path.is_symlink() for path in (target, target.parent, target.parent.parent)):
            raise ValueError("readout destination cannot be a symlink")
        if target.exists() and target.read_bytes() != raw:
            raise ValueError("immutable readout mirror already contains different source bytes")
        prepared.append((target, raw, item))
    result = {}
    for target, raw, item in prepared:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            atomic_bytes(target, raw)
        timestamp = item["source_mtime_ns"]
        os.utime(target, ns=(timestamp, timestamp))
        result[item["name"]] = item["sha256"]
    return result


class ReadoutMirror:
    """Queue at most one bounded read-only request on ROOT's existing local RPC stream."""

    def __init__(self, rpc: Path, native_root: str, local_root: Path, state_path: Path) -> None:
        self.rpc, self.native_root, self.local_root, self.path = (
            rpc,
            native_root,
            local_root,
            state_path,
        )
        self.state = _object(state_path) or {
            "schema": "wnn.l4-readout-mirror.v1",
            "known": {},
            "native_root": native_root,
            "local_root": str(local_root.absolute()),
        }
        if (
            self.state.get("schema") != "wnn.l4-readout-mirror.v1"
            or self.state.get("native_root") != native_root
            or self.state.get("local_root") != str(local_root.absolute())
        ):
            raise ValueError("readout mirror state belongs to another source")

    def tick(self, now: float) -> None:
        """Consume complete result JSON or atomically queue one exact collection request."""
        pending = self.state.get("pending")
        if pending:
            result_path = self.rpc / "results" / f"{pending}.json"
            if result_path.is_symlink():
                raise ValueError("RPC result cannot be a symlink")
            if not result_path.exists():
                request = self.rpc / "requests" / f"{pending}.json"
                if not request.exists():
                    atomic_json(request, self.state["pending_request"])
                return
            if result_path.stat().st_size > 2 * MAX_RESPONSE:
                raise ValueError("RPC readout result exceeds its bound")
            try:
                result = json.loads(result_path.read_bytes())
            except (ValueError, UnicodeDecodeError):
                # The ROOT client writes results non-atomically. Retry the same ID next poll.
                return
            output = result.get("output", "")
            begin, end = f"WNN_RPC_BEGIN {pending}\n", f"\nWNN_RPC_END {pending} ok"
            if (
                result.get("id") != pending
                or not output.startswith(begin)
                or not output.rstrip().endswith(end)
            ):
                raise ValueError("RPC readout result ID/framing/status differs")
            payload = output[len(begin) : output.rfind(end)]
            if len(payload.encode()) > MAX_RESPONSE:
                raise ValueError("RPC readout payload exceeds its bound")
            snapshot = json.loads(payload)
            # Root may itself be a declared symlink; collector returns its physical path.
            self.state["known"].update(apply_readouts(snapshot, self.local_root))
            self.state.update(
                last_collected_unix=snapshot["collected_at_unix"], errors=snapshot.get("errors", [])
            )
            self.state.pop("pending")
            self.state.pop("pending_request")
            atomic_json(self.path, self.state)
        if now - self.state.get("last_queued_unix", 0) < 30:
            return
        request_id = f"l4-readouts-{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        source = Path(__file__).with_name("collect_l4_readouts_r2.py").read_text(encoding="utf-8")
        arguments = json.dumps([self.native_root, self.state["known"]], allow_nan=False)
        code = f"import json\nnamespace={{'__name__':'readout_rpc'}}\nexec(compile({source!r},'collect_l4_readouts.py','exec'),namespace)\nargs=json.loads({arguments!r})\nprint(json.dumps(namespace['collect'](*args),allow_nan=False))\n"
        requests = self.rpc / "requests"
        if requests.is_symlink() or not requests.is_dir():
            raise ValueError("existing ROOT RPC request directory is required")
        self.state.update(pending=request_id, pending_request={"code": code}, last_queued_unix=now)
        atomic_json(self.path, self.state)
        atomic_json(requests / f"{request_id}.json", self.state["pending_request"])
