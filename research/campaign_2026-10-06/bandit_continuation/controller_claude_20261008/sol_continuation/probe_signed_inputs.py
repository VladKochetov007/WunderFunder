from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from kaggle.api.kaggle_api_extended import ApiDownloadDatasetRequest, KaggleApi

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OUT = Path(__file__).resolve().parent
PACKET = Path(
    "/home/vlad/development/wunder/worktrees/wnn-gru512-768-muon-fit-v5-20261008/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/gru512_768_muon_fit_v5r3_skip_accounting_20261008/packet"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only fresh signed URL retrieval; never downloads payload bodies"
    )
    parser.add_argument(
        "--runtime", type=Path, default=OUT.parents[4] / ".runtime/r5-signed-feasibility"
    )
    parser.add_argument("--attempt-id", default="e529-vast-capacity-r5-feasibility")
    parser.add_argument("--campaign-label", default="wnn-e529-capacity-r5-feasibility")
    parser.add_argument("--receipt", type=Path, default=OUT / "R5_SIGNED_INPUT_FEASIBILITY_R1.json")
    args = parser.parse_args()
    sys.path.insert(0, str(OUT.parents[4]))
    from scripts import kaggle_vast_prefetch as prefetch

    recipe = json.loads((PACKET / "GRU512_768_MUON_FIT_RECIPE_V5R3.json").read_text())
    assets = json.loads(
        (ROOT / "research/campaign_2026-10-04/kaggle/fold0_assets/assets-manifest.json").read_text()
    )
    by_path = {row["path"]: row for row in assets["files"]}
    api = KaggleApi()
    api.authenticate()
    inputs = []
    public = []
    with api.build_kaggle_client() as kaggle:
        for path in sorted(prefetch.ALLOWED_REMOTE_PATHS):
            identity = recipe["inputs"]["source_members"][path]
            if path.startswith("cache/"):
                asset = by_path[path]
                assert asset["sha256"] == identity["sha256"] and asset["size"] == identity["bytes"]
                dataset = "wnn-c20261004-fold0-assets"
                filename = asset["filename"]
            else:
                dataset = "wunder-funder"
                filename = Path(path).name
            request = ApiDownloadDatasetRequest()
            request.owner_slug = "vladkochetov07"
            request.dataset_slug = dataset
            request.file_name = filename
            response = kaggle.datasets.dataset_api_client.download_dataset(request)
            url = response.request.url
            parts = urlsplit(url)
            if parts.port == 443:
                url = parts._replace(netloc=parts.hostname).geturl()
            prefetch._validate_url(url)
            inputs.append(
                {
                    "remote_path": path,
                    "sha256": identity["sha256"],
                    "size_bytes": identity["bytes"],
                    "url": url,
                }
            )
            public.append(
                {
                    "remote_path": path,
                    "sha256": identity["sha256"],
                    "size_bytes": identity["bytes"],
                    "dataset": "vladkochetov07/" + dataset,
                    "filename": filename,
                    "signed_url_schema_pass": True,
                }
            )
    runtime = args.runtime
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(
        {"schema_version": 1, "inputs": inputs}, sort_keys=True, separators=(",", ":")
    ).encode()
    plan = runtime / "signed_plan.private.json"
    descriptor = os.open(plan, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
    admission = runtime / "prefetch_admission.json"
    admission.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt_id": args.attempt_id,
                "campaign_label": args.campaign_label,
                "private_plan_sha256": hashlib.sha256(raw).hexdigest(),
                "source_binding": [
                    {key: row[key] for key in ("remote_path", "sha256", "size_bytes")}
                    for row in inputs
                ],
            }
        )
    )
    frozen = [
        {
            "remote_path": row["remote_path"],
            "source_sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in inputs
    ]
    prepared = prefetch.prepare_prefetch(
        plan,
        admission,
        attempt_id=args.attempt_id,
        campaign_label=args.campaign_label,
        frozen_inputs=frozen,
        runtime_root=runtime,
    )
    receipt = {
        "status": "read_only_signed_plan_schema_pass",
        "inputs": public,
        "total_cloud_bytes": sum(row["size_bytes"] for row in inputs),
        "input_count": len(prepared.items),
        "private_plan_sha256": prepared.private_plan_sha256,
        "remote_program_sha256": prefetch.REMOTE_PROGRAM_SHA256,
        "private_plan_location": str(plan),
        "scope": "read-only URL retrieval and exact local schema; no byte downloads/provider creation. Refresh signed URLs after final intent before dispatch; hash verification only on remote actual bytes.",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "input_count": len(inputs),
                "total_cloud_bytes": receipt["total_cloud_bytes"],
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "exception_class": type(error).__name__}))
        raise SystemExit(1) from None
