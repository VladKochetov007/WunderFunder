from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.galam import galam_config_from_mapping
from wnn.galam.execution import galam_execution_backend
from wnn.galam.variant_config import describe_variant, galam_variant_from_mapping
from wnn.galam.variants import build_galam_variant


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
    nested = mapping
    if "model" in mapping:
        model_settings = mapping["model"]
        if not isinstance(model_settings, dict) or model_settings.get("architecture") != "galam":
            parser.error("runtime config must select model.architecture=galam")
        nested = model_settings.get("galam")
        if not isinstance(nested, dict):
            parser.error("runtime config must contain model.galam")
    variant = galam_variant_from_mapping(nested)
    backend = galam_execution_backend(nested, variant)
    config = galam_config_from_mapping(
        {key: value for key, value in nested.items() if key not in ("variant", "execution_backend")}
    )
    model = build_galam_variant(config, variant, backend)
    print(
        json.dumps(
            {
                "status": "schematic_reconstruction",
                "exact_replication": False,
                "shape": [config.input_features, config.output_features],
                "role": config.role,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "effective_model": describe_variant(config, variant),
                "execution_backend": backend,
                "known_graph_details": sorted(config.faithful_details)
                if variant == "schematic_lstm_full"
                else [],
                "source_template_graph_details": sorted(config.faithful_details),
                "assumed_details": sorted(config.assumed_details),
                "scientific_fits": 0,
                "reference_parity": "not_available",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
