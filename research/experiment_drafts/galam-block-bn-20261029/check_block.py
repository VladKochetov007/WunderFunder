from __future__ import annotations

import argparse
from pathlib import Path

from block_receipts import configure_cuda, identity, read_config, validate_output, write_receipt
from block_verification import run_checks


def main() -> None:
    """Write fail-closed GPU numerical acceptance for a frozen control config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--chunk-rows", type=int, default=64)
    args = parser.parse_args()
    validate_output(args.output)
    if not 2 <= args.batch_size <= 1024 or not 1 <= args.chunk_rows <= 256:
        raise ValueError("Batch within 2..1024, chunk within 1..256")
    device = configure_cuda(args.device)
    config = read_config(args.config)
    receipt: dict[str, object] = {
        "schema": "wnn.galam-block-acceptance.v1",
        "identity": identity(args.config, args.expected_commit, device),
        "representative_shape": [args.batch_size, args.chunk_rows, config.input_features],
        "suite": "state-gradient-bn-causal-update-v1",
        "status": "failed",
        "scope": "synthetic numerical checks; no scientific fit or validation quality claim",
    }
    try:
        receipt["results"] = run_checks(config, device, args.batch_size, args.chunk_rows)
        receipt["status"] = "passed"
    except (AssertionError, RuntimeError, ValueError, TypeError) as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        write_receipt(args.output, receipt)
        raise
    write_receipt(args.output, receipt)


if __name__ == "__main__":
    main()
