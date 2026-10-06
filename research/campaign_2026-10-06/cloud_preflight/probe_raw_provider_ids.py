"""Read raw Kaggle status/kernel JSON before generated SDK field filtering."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

OWNER = "vladkochetov07"
SLUG = "wnn-gru-device-preflight-20261006-v2"
VERSION = "1"


def fetch_raw_json() -> dict[str, Any]:
    """Read exact target status and metadata without printing auth headers."""
    api = KaggleApi()
    api.authenticate()
    captured: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    def capture_raw_response(response: Any) -> None:
        """Save response JSON before Kaggle's generated type drops extra fields."""
        payload = response.json()
        route = response.url.split("?", maxsplit=1)[0].split("/v1/", maxsplit=1)[-1]
        if route.endswith("/GetKernel") and isinstance(payload, dict):
            blob = payload.get("blob")
            if isinstance(blob, dict):
                payload = {
                    **payload,
                    "blob": {
                        key: blob[key]
                        for key in ("language", "kernelType", "kernel_type")
                        if key in blob
                    },
                }
        captured.append({"route": route, "http_status": response.status_code, "json": payload})

    api.get_response_processor = lambda: capture_raw_response
    sdk_status = None
    try:
        sdk_status = api.kernels_status(f"{OWNER}/{SLUG}")
    except Exception as error:
        errors.append(
            {"operation": "kernels_status", "type": type(error).__name__, "error": str(error)}
        )

    try:
        with api.build_kaggle_client() as client:
            request = ApiGetKernelRequest()
            request.user_name = OWNER
            request.kernel_slug = f"{SLUG}/{VERSION}"
            kernel = client.kernels.kernels_api_client.get_kernel(request)
    except Exception as error:
        errors.append(
            {"operation": "get_kernel", "type": type(error).__name__, "error": str(error)}
        )
    else:
        errors.append(
            {
                "operation": "get_kernel",
                "type": "completed",
                "error": f"sdk_kernel_id={kernel.metadata.id};current_version={kernel.metadata.current_version_number}",
            }
        )

    return {
        "observed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "kernel": f"{OWNER}/{SLUG}/{VERSION}",
        "source": "response_processor captured authenticated SDK JSON before generated model parsing",
        "raw_responses": captured,
        "operation_results": errors,
        "sdk_status_after_parse": None if sdk_status is None else str(sdk_status.status),
        "auth_headers_printed": False,
    }


if __name__ == "__main__":
    print(json.dumps(fetch_raw_json(), sort_keys=True))
