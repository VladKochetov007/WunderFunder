from __future__ import annotations

from scripts.profile_activation_qat import _has_valid_latent_parameter_change


def _fingerprint(sha256: str) -> dict[str, object]:
    return {"sha256": sha256 * 64, "shape": [2, 3], "dtype": "torch.float32"}


def _update(before: str, after: str) -> dict[str, object]:
    before_fingerprints = {"gru.weight_ih_l0": _fingerprint(before)}
    after_fingerprints = {"gru.weight_ih_l0": _fingerprint(after)}
    changed = {
        name: before_fingerprints[name]["sha256"] != after_fingerprints[name]["sha256"]
        for name in before_fingerprints
    }
    return {
        "latent_parameter_fingerprints_before": before_fingerprints,
        "latent_parameter_fingerprints_after": after_fingerprints,
        "latent_parameters_changed_by_name": changed,
        "latent_parameters_changed": any(changed.values()),
        # Fused Adam may update values without advancing Tensor._version.
        "latent_versions_advanced": False,
    }


def test_actual_latent_change_validates_when_version_counter_is_unchanged() -> None:
    update = _update("a", "b")

    assert _has_valid_latent_parameter_change(update)


def test_unchanged_latent_tensor_is_rejected_even_with_claimed_change() -> None:
    update = _update("a", "a")
    update["latent_parameters_changed"] = True
    update["latent_parameters_changed_by_name"] = {"gru.weight_ih_l0": True}

    assert not _has_valid_latent_parameter_change(update)


def test_partial_or_malformed_fingerprint_sets_are_rejected() -> None:
    update = _update("a", "b")
    update["latent_parameter_fingerprints_after"] = {}

    assert not _has_valid_latent_parameter_change(update)
