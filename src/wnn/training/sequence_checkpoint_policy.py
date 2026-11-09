from __future__ import annotations

CONFIGURED_FINAL_EPOCH = "configured_final_epoch"
RETAINED_EPOCH_PURPOSE = "retained_epoch_snapshot_no_automatic_selection"
FINAL_ENDPOINT_PURPOSE = "fixed_configured_final_endpoint"
RESUME_STATE_PURPOSE = "latest_full_safe_resume_state"


def validate_checkpoint_endpoint_policy(config: dict[str, object]) -> None:
    """Validate the optional future all-epochs fixed-endpoint overlay."""
    if "checkpoint_endpoint_policy" not in config:
        return
    if config["checkpoint_endpoint_policy"] != CONFIGURED_FINAL_EPOCH:
        raise ValueError("checkpoint_endpoint_policy must be configured_final_epoch")
    if config.get("checkpoint_retention") != "all_epochs":
        raise ValueError("checkpoint_endpoint_policy requires checkpoint_retention=all_epochs")
