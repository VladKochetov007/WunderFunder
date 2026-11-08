from __future__ import annotations

from torch import Tensor


def _first_tensor(value: object) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, tuple):
        return next((tensor for item in value if (tensor := _first_tensor(item)) is not None), None)
    return None


def _model_path(name: str) -> str:
    if ".galam." in name:
        return name.split(".galam.", 1)[1]
    if name.startswith("galam."):
        return name.removeprefix("galam.")
    return name.removeprefix("module.")


def _attention_tag(path: str) -> str:
    return "local" if path.startswith("local_attention.") else "global"


def _activation_metric(path: str) -> str | None:
    if path in ("input_projection", "output_head"):
        return f"transformer/{path}_activation_rms"
    if path.startswith("blocks."):
        parts = path.split(".")
        if (
            len(parts) == 3
            and parts[1].isdigit()
            and parts[2]
            in (
                "attention_output",
                "feedforward_in",
                "feedforward_out",
            )
        ):
            return f"transformer/layer_{parts[1]}/{parts[2]}_activation_rms"
    if path.startswith("local_norms."):
        return f"local/activation_rms/{path.rsplit('.', 1)[-1]}"
    if path.startswith("global_attention_norms."):
        return f"global/attention_activation_rms/{path.rsplit('.', 1)[-1]}"
    if path.startswith("global_mixer_norms."):
        return f"global/mixer_activation_rms/{path.rsplit('.', 1)[-1]}"
    if path == "hidden_projection":
        return "local/hidden_projection_activation_rms"
    if path == "cell_projection":
        return "local/cell_projection_activation_rms"
    if path == "fusion":
        return "fusion/output_activation_rms"
    if path in ("fusion_norm", "feed_forward", "feed_forward_norm"):
        return f"fusion/{path}_activation_rms"
    if path.startswith(("local_attention.", "global_attention.")) and path.endswith(".output"):
        return f"attention/{_attention_tag(path)}/output_activation_rms"
    if ".relative_bias" in path and path.endswith("relative_bias"):
        return f"relative_bias/{_attention_tag(path)}/activation_rms"
    return None


def _gradient_key(key: str) -> str:
    if key.endswith("_activation_rms"):
        return f"{key.removesuffix('_activation_rms')}_gradient_rms"
    if key.endswith("_state_rms"):
        return f"{key.removesuffix('_state_rms')}_state_gradient_rms"
    if key.endswith("_output_rms"):
        return f"{key.removesuffix('_output_rms')}_output_gradient_rms"
    return key.replace("activation_rms", "gradient_rms")
