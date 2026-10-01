from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path


def sha256(text: str) -> str:
    """Hash source text without changing the active control trainer."""
    return hashlib.sha256(text.encode()).hexdigest()


def replace_once(source: str, needle: str, replacement: str) -> str:
    """Stop if a trainer change invalidates the intended insertion context."""
    if source.count(needle) != 1:
        raise ValueError(f"integration context is ambiguous or absent: {needle}")
    return source.replace(needle, replacement)


def proposed_sources() -> tuple[str, str, str]:
    """Build source text for synthetic interface checks without touching live files."""
    own_dir = Path(__file__).resolve().parent
    root = own_dir.parents[2]
    model = (own_dir / "model.py").read_text()
    trainer = (root / "scripts/train_rnn.py").read_text()
    patched = replace_once(
        trainer,
        "from wnn.propensity import load_propensity, propensity\n",
        "from wnn.propensity import load_propensity, propensity\n"
        "from wnn.structured_state import StructuredStateModel\n",
    )
    patched = replace_once(
        patched,
        "def make_model(args_dict, n_in, n_out):\n",
        "def make_model(args_dict, n_in, n_out):\n"
        '    if args_dict.get("arch") == "structured":\n'
        '        if (args_dict["hidden"] != 96 or args_dict["layers"] != 2\n'
        '                or args_dict["dropout"] != 0\n'
        '                or args_dict.get("enc", "linear") != "linear"):\n'
        '            raise ValueError("structured card B requires hidden96, layers2, dropout0, enc linear")\n'
        '        model = StructuredStateModel(n_in, n_out, args_dict.get("decay_init", "multi"))\n'
        '        for prefix in filter(None, args_dict.get("freeze", "").split(",")):\n'
        "            if not any(name.startswith(prefix) for name, _ in model.named_parameters()):\n"
        '                raise ValueError(f"unknown structured freeze prefix {prefix!r}; backbone is inp,blocks")\n'
        "        return model\n",
    )
    patched = replace_once(
        patched,
        '    ap.add_argument("--arch", choices=["gru", "dual", "gated"], default="gru")\n',
        '    ap.add_argument("--arch", choices=["gru", "dual", "gated", "structured"], default="gru")\n'
        '    ap.add_argument("--decay-init", choices=["multi", "single"], default="multi",\n'
        '                    help="structured-state initial half-lives; identical learned bounds and phases")\n',
    )
    patched = replace_once(patched, '        "arch",\n', '        "arch",\n        "decay_init",\n')
    patched = replace_once(
        patched,
        "    if quote_cache_provenance is not None:\n",
        '    if args.arch == "structured":\n'
        '        provenance["model_source_sha256"] = file_sha256(ROOT / "src/wnn/structured_state.py")\n'
        "    if quote_cache_provenance is not None:\n",
    )
    ast.parse(patched)
    return model, trainer, patched


def main() -> None:
    """Produce a reviewable patch; applying it waits for completed controls."""
    own_dir = Path(__file__).resolve().parent
    model, trainer, patched = proposed_sources()
    patch = "".join(
        difflib.unified_diff(
            [],
            model.splitlines(keepends=True),
            fromfile="/dev/null",
            tofile="b/src/wnn/structured_state.py",
        )
    )
    patch += "".join(
        difflib.unified_diff(
            trainer.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="a/scripts/train_rnn.py",
            tofile="b/scripts/train_rnn.py",
        )
    )
    (own_dir / "integration.patch").write_text(patch)
    (own_dir / "integration_manifest.json").write_text(
        json.dumps(
            {
                "model_source_sha256": sha256(model),
                "trainer_base_sha256": sha256(trainer),
                "trainer_after_patch_sha256": sha256(patched),
                "integration_patch_sha256": sha256(patch),
                "applied": False,
                "new_module": "src/wnn/structured_state.py",
                "changed_module": "scripts/train_rnn.py",
                "native_port": "required after score promotion; not implemented",
                "launcher_changes": None,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
