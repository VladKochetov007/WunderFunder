from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.galam import GALAM, galam_config_from_mapping


def main() -> None:
    """Inspect an explicit GALAM shape configuration without fitting or data access."""
    parser = argparse.ArgumentParser(description="Validate the source-traced GALAM graph; no fit.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("rb") as stream:
        raw = stream.read(100_001)
    if len(raw) > 100_000:
        parser.error("GALAM configuration exceeds the bounded metadata limit")
    mapping = json.loads(raw)
    if not isinstance(mapping, dict):
        parser.error("GALAM configuration must be an object")
    config = galam_config_from_mapping(mapping)
    model = GALAM(config)
    print(
        json.dumps(
            {
                "status": "schematic_reconstruction",
                "exact_replication": False,
                "shape": [config.input_features, config.output_features],
                "role": config.role,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "known_graph_details": sorted(config.faithful_details),
                "assumed_details": sorted(config.assumed_details),
                "scientific_fits": 0,
                "reference_parity": "not_available",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
