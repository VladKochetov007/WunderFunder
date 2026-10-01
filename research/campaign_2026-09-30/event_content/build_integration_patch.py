"""Generate a reviewable patch only; never apply to live controls."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path


def replace_once(source: str, needle: str, replacement: str) -> str:
    if source.count(needle) != 1:
        raise ValueError(f"ambiguous trainer context: {needle}")
    return source.replace(needle, replacement)


def main() -> None:
    own = Path(__file__).resolve().parent
    root = own.parents[2]
    original = (root / "scripts/train_rnn.py").read_text()
    proposed = replace_once(
        original,
        "from wnn.propensity import load_propensity, propensity\n",
        "from wnn.propensity import load_propensity, propensity\nfrom wnn.event_content import EventContentModel\n",
    )
    proposed = replace_once(
        proposed,
        "def make_model(args_dict, n_in, n_out):\n",
        "def make_model(args_dict, n_in, n_out):\n"
        '    if args_dict.get("arch") == "event-content":\n'
        '        if (args_dict["hidden"] != 96 or args_dict["layers"] != 1\n'
        '                or args_dict["dropout"] != 0 or args_dict.get("enc", "linear") != "linear"):\n'
        '            raise ValueError("D1 requires hidden96,layers1,dropout0,enc linear")\n'
        '        return EventContentModel(n_in, n_out, args_dict.get("memory_arm", "endpoint-content"))\n',
    )
    proposed = replace_once(
        proposed,
        '    ap.add_argument("--arch", choices=["gru", "dual", "gated"], default="gru")\n',
        '    ap.add_argument("--arch", choices=["gru", "dual", "gated", "event-content"], default="gru")\n'
        '    ap.add_argument("--memory-arm", choices=["endpoint-content", "endpoint-position", "raw-code-content"], default="endpoint-content")\n',
    )
    proposed = replace_once(
        proposed, '        "arch",\n', '        "arch",\n        "memory_arm",\n'
    )
    proposed = replace_once(
        proposed,
        "    if quote_cache_provenance is not None:\n",
        '    if args.arch == "event-content":\n'
        '        provenance["model_source_sha256"] = file_sha256(ROOT / "src/wnn/event_content.py")\n'
        "    if quote_cache_provenance is not None:\n",
    )
    ast.parse(proposed)
    model = (own / "model.py").read_text()
    patch = "".join(
        difflib.unified_diff(
            [],
            model.splitlines(keepends=True),
            fromfile="/dev/null",
            tofile="b/src/wnn/event_content.py",
        )
    )
    patch += "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile="a/scripts/train_rnn.py",
            tofile="b/scripts/train_rnn.py",
        )
    )
    (own / "integration.patch").write_text(patch)
    (own / "integration_manifest.json").write_text(
        json.dumps(
            {
                "applied": False,
                "base_trainer_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "proposed_trainer_sha256": hashlib.sha256(proposed.encode()).hexdigest(),
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "model_sha256": hashlib.sha256(model.encode()).hexdigest(),
                "state": "flat six tensors; existing forward_chunk/detach_state generic interface unchanged",
                "SSM": "no SSM changes; mutually separate unapplied patches require root merging",
                "prediction": "factory metadata required; native port and deployment path unimplemented",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
