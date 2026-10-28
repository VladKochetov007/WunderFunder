from __future__ import annotations

import argparse
from pathlib import Path

from block_benchmark import benchmark
from block_receipts import (
    configure_cuda,
    digest_file,
    identity,
    read_config,
    validate_acceptance,
    validate_output,
    write_receipt,
)


def main() -> None:
    """Profile both backends only after source/config/device-matched GPU acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--updates", type=int, default=12)
    args = parser.parse_args()
    validate_output(args.output)
    if not 1 <= args.warmup <= 16 or not 4 <= args.updates <= 64:
        raise ValueError("Output must be fresh; warmup within 1..16, updates within 4..64")
    if not 2 <= args.batch_size <= 1024 or not 1 <= args.chunk_rows <= 256:
        raise ValueError("Bounded profile requires batch 2..1024 and chunk 1..256")
    device = configure_cuda(args.device)
    config = read_config(args.config)
    current = identity(args.config, args.expected_commit, device)
    validate_acceptance(
        args.acceptance, current, [args.batch_size, args.chunk_rows, config.input_features]
    )
    receipt: dict[str, object] = {
        "schema": "wnn.galam-block-profile.v1",
        "identity": current,
        "acceptance_sha256": digest_file(args.acceptance),
        "status": "failed",
        "scope": "synthetic forward/backward/AdamW update timing; excludes data loading and epoch evaluation",
        "precision": "fp32",
        "batch_size": args.batch_size,
        "chunk_rows": args.chunk_rows,
        "arms": [],
    }
    arms: list[dict[str, object]] = []
    try:
        for backend in ("step", "block"):
            arms.append(
                benchmark(
                    config,
                    device,
                    backend,
                    args.batch_size,
                    args.chunk_rows,
                    args.warmup,
                    args.updates,
                )
            )
        receipt["arms"] = arms
        receipt["wall_speedup"] = float(arms[0]["wall_median_ms"]) / float(
            arms[1]["wall_median_ms"]
        )
        receipt["status"] = "complete"
    except (AssertionError, RuntimeError, ValueError, TypeError) as error:
        receipt["arms"] = arms
        receipt["error"] = f"{type(error).__name__}: {error}"
        write_receipt(args.output, receipt)
        raise
    write_receipt(args.output, receipt)


if __name__ == "__main__":
    main()
