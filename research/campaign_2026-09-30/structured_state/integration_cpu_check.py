from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import torch
from build_integration_patch import proposed_sources
from model import StructuredStateModel


def main() -> None:
    """Check proposed factory, strict loading, freeze rules, actual EMA helpers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="manual")
    options = parser.parse_args()
    if not options.label.replace("_", "").replace("-", "").isalnum():
        raise ValueError("label must contain letters, numbers, underscores or hyphens")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(3)
    own_dir = Path(__file__).resolve().parent
    root = own_dir.parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from train_rnn import FastEMA, detach_state, forward_chunk

    _, _, proposed = proposed_sources()
    tree = ast.parse(proposed)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"make_model", "inherit_init_arguments"}
    ]
    namespace: dict[str, object] = {
        "StructuredStateModel": StructuredStateModel,
        "argparse": argparse,
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "proposed_factory", "exec"), namespace
    )
    factory = namespace["make_model"]
    args = {
        "arch": "structured",
        "hidden": 96,
        "layers": 2,
        "dropout": 0.0,
        "enc": "linear",
        "decay_init": "single",
        "freeze": "",
    }
    model = factory(args, 211, 23)
    model.mean.copy_(torch.randn_like(model.mean))
    model.std.copy_(torch.rand_like(model.std) + 0.1)
    checkpoint_path = own_dir / f"integration_cpu_checkpoint_{options.label}.pt"
    checkpoint = {
        "model": model.state_dict(),
        "args": args,
        "n_in": 211,
        "epoch": 3,
        "synthetic_only": True,
    }
    if checkpoint_path.exists():
        raise FileExistsError("refusing to overwrite interface checkpoint")
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reconstructed = factory(
        loaded["args"], loaded["n_in"], loaded["model"]["head.2.weight"].shape[0]
    )
    reconstructed.load_state_dict(loaded["model"], strict=True)
    inputs = torch.randn(2, 34, 211)
    with torch.no_grad():
        original, original_state = model(inputs)
        restored, restored_state = reconstructed(inputs)
    assert torch.equal(original, restored)
    assert all(torch.equal(a, b) for a, b in zip(original_state, restored_state, strict=True))
    teacher = FastEMA(model, 0.999)
    student_state = teacher_state = None
    for start in [0, 17]:
        out, student_state = forward_chunk(model, inputs, slice(start, start + 17), student_state)
        student_state = detach_state(student_state)
        with torch.no_grad():
            expected, teacher_state = forward_chunk(
                teacher.module, inputs, slice(start, start + 17), teacher_state
            )
        teacher_state = detach_state(teacher_state)
        assert torch.equal(out, expected)
        assert all(not state.requires_grad for state in student_state)
        out.square().mean().backward()
        teacher.update_parameters(model)
        model.zero_grad(set_to_none=True)
    head_model = factory({**args, "freeze": "inp,blocks"}, 211, 23)
    for name, parameter in head_model.named_parameters():
        if name.startswith(("inp", "blocks")):
            parameter.requires_grad_(False)
    trainable = [name for name, p in head_model.named_parameters() if p.requires_grad]
    assert trainable == ["head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias"]
    try:
        factory({**args, "freeze": "inp,gru"}, 211, 23)
    except ValueError:
        old_freeze_rejected = True
    else:
        raise AssertionError("legacy GRU freeze was silently accepted")
    child = argparse.Namespace(**{**args, "decay_init": "multi"})
    namespace["inherit_init_arguments"](child, checkpoint, [])
    assert child.decay_init == "single"
    assert not torch.cuda.is_initialized()
    (own_dir / f"integration_cpu_check_{options.label}.json").write_text(
        json.dumps(
            {
                "passed": True,
                "synthetic_only": True,
                "cuda_initialized": False,
                "factory_and_strict_load_exact": True,
                "ema_and_detach_exact": True,
                "legacy_gru_freeze_rejected": old_freeze_rejected,
                "head_only_trainable": trainable,
                "inherited_decay_init": child.decay_init,
                "model_source_sha256": hashlib.sha256(
                    (own_dir / "model.py").read_bytes()
                ).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "proposed_trainer_sha256": hashlib.sha256(proposed.encode()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
