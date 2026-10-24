from __future__ import annotations

import re
from typing import Any

from wnn.training.sequence_cache_ensemble_provenance import (
    canonical_sha256 as _canonical_sha256,
)
from wnn.training.sequence_cache_ensemble_provenance import (
    common_provenance,
)

MAX_ANCESTRY_DEPTH = 32
MAX_ANCESTRY_NODES = 256
ENSEMBLE_KIND = "mean_of_three_fp32"


def is_sha256(value: object) -> bool:
    """Return whether a value is a lowercase SHA-256 hex digest."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_teacher_reference(reference: object) -> dict[str, Any]:
    """Resolve a historical teacher identity while checking newer identity fields."""
    if not isinstance(reference, dict) or not isinstance(reference.get("producer"), dict):
        raise ValueError("teacher ancestry reference is malformed")
    producer = reference["producer"]
    if reference.get("manifest_sha256") is not None and not is_sha256(reference["manifest_sha256"]):
        raise ValueError("invalid teacher manifest hash")
    if producer.get("kind") == ENSEMBLE_KIND:
        if reference.get("checkpoint_sha256") is not None:
            raise ValueError("ensemble teacher identity cannot claim a checkpoint")
        if reference.get("composition_sha256") != producer.get("composition_sha256"):
            raise ValueError("ensemble teacher composition identity mismatch")
    elif reference.get("checkpoint_sha256") is not None and reference.get(
        "checkpoint_sha256"
    ) != producer.get("checkpoint_sha256"):
        raise ValueError("teacher checkpoint reference mismatch")
    return producer


def _validate_ensemble_node(current: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate parent receipts and return the three physical checkpoint producers."""
    if "checkpoint_sha256" in current:
        raise ValueError("ensemble producer cannot claim a physical checkpoint")
    if current.get("teacher") is not None:
        raise ValueError("ensemble producer must express ancestry through its parent branch")
    parents = current.get("parents")
    composition = current.get("composition")
    if not isinstance(parents, list) or len(parents) != 3 or not isinstance(composition, dict):
        raise ValueError("mean teacher requires three complete parent references")
    recipes: list[dict[str, Any]] = []
    seeds: list[int] = []
    checkpoint_hashes: list[str] = []
    manifest_hashes: list[str] = []
    references: list[dict[str, Any]] = []
    parent_producers: list[dict[str, Any]] = []
    common_provenances: list[dict[str, Any]] = []
    for parent in parents:
        if not isinstance(parent, dict) or not isinstance(parent.get("producer"), dict):
            raise ValueError("ensemble parent reference is malformed")
        producer = parent["producer"]
        config = producer.get("config")
        provenance = producer.get("provenance")
        seed = config.get("seed") if isinstance(config, dict) else None
        checkpoint_hash = parent.get("checkpoint_sha256")
        manifest_hash = parent.get("manifest_sha256")
        if producer.get("kind") == ENSEMBLE_KIND:
            raise ValueError("mean teacher inputs must be three single-checkpoint teachers")
        if not is_sha256(checkpoint_hash) or checkpoint_hash != producer.get("checkpoint_sha256"):
            raise ValueError("ensemble parent checkpoint reference mismatch")
        if not is_sha256(manifest_hash):
            raise ValueError("invalid ensemble parent manifest hash")
        if not isinstance(provenance, dict) or provenance != parent.get("provenance"):
            raise ValueError("ensemble parent provenance reference mismatch")
        common_provenances.append(common_provenance(producer, provenance))
        if not is_sha256(parent.get("producer_sha256")) or parent.get(
            "producer_sha256"
        ) != _canonical_sha256(producer):
            raise ValueError("ensemble parent producer reference mismatch")
        if not is_sha256(parent.get("provenance_sha256")) or parent.get(
            "provenance_sha256"
        ) != _canonical_sha256(provenance):
            raise ValueError("ensemble parent provenance hash mismatch")
        if type(seed) is not int or not 0 <= seed < 2**32 or parent.get("seed") != seed:
            raise ValueError("ensemble parent seed is invalid")
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("model"), dict)
            or type(config["model"].get("hint_dim")) is not int
            or config["model"].get("hint_dim") != 0
        ):
            raise ValueError("ensemble teacher recipe must include a zero-hint model")
        recipe = dict(config)
        recipe.pop("seed", None)
        recipe_hash = _canonical_sha256(recipe)
        if parent.get("recipe_sha256") != recipe_hash:
            raise ValueError("ensemble parent recipe reference mismatch")
        recipes.append(recipe)
        seeds.append(seed)
        checkpoint_hashes.append(checkpoint_hash)
        manifest_hashes.append(manifest_hash)
        references.append(
            {
                "manifest_sha256": manifest_hash,
                "checkpoint_sha256": checkpoint_hash,
                "seed": seed,
                "recipe_sha256": recipe_hash,
                "producer_sha256": parent["producer_sha256"],
                "provenance_sha256": parent["provenance_sha256"],
            }
        )
        parent_producers.append(producer)
    if len(set(seeds)) != 3:
        raise ValueError("mean teacher parents must use three distinct seeds")
    if len(set(checkpoint_hashes)) != 3:
        raise ValueError("mean teacher parents must use three distinct checkpoints")
    if len(set(manifest_hashes)) != 3:
        raise ValueError("mean teacher parents must have distinct cache manifests")
    if recipes[1:] != recipes[:1] * 2:
        raise ValueError("mean teacher parents use different recipes beyond seed")
    if common_provenances[1:] != common_provenances[:1] * 2:
        raise ValueError("mean teacher parents have different shared fit provenance")
    shared_provenance_hash = _canonical_sha256(common_provenances[0])
    if current.get("shared_provenance_sha256") != shared_provenance_hash:
        raise ValueError("ensemble shared provenance hash does not match its parents")
    expected_recipe_hash = _canonical_sha256(recipes[0])
    if current.get("recipe") != recipes[0] or current.get("recipe_sha256") != expected_recipe_hash:
        raise ValueError("ensemble recipe identity does not match its parents")
    expected_composition = {
        "schema": "sequence-teacher-composition-v1",
        "method": ENSEMBLE_KIND,
        "clip_before_mean": False,
        "shared_provenance_sha256": shared_provenance_hash,
        "parents": references,
    }
    if composition != expected_composition:
        raise ValueError("ensemble composition does not match its parent references")
    if current.get("composition_sha256") != _canonical_sha256(composition):
        raise ValueError("ensemble composition hash mismatch")
    return parent_producers


def validate_producer(
    producer: dict[str, Any],
    split_hash: str,
    members: list[tuple[str, int]],
) -> None:
    """Require same-fold fitting ancestry through linear and ensemble branches."""
    expected_members = [list(member) for member in members]
    active: set[int] = set()

    def visit(current: object, depth: int) -> int:
        """Validate a producer and its ancestors, returning its bounded node count."""
        if depth > MAX_ANCESTRY_DEPTH:
            raise ValueError("teacher ancestry exceeds supported depth")
        if not isinstance(current, dict):
            raise ValueError("teacher ancestry producer is malformed")
        identity = id(current)
        if identity in active:
            raise ValueError("teacher ancestry contains a cycle")
        active.add(identity)
        try:
            if current.get("split_sha256") != split_hash:
                raise ValueError("teacher ancestry crosses evaluation folds")
            if current.get("fit_members") != expected_members:
                raise ValueError("teacher fitted membership differs from the student's clean split")
            kind = current.get("kind")
            if kind == ENSEMBLE_KIND:
                children = _validate_ensemble_node(current)
            else:
                if kind not in (None, "single_checkpoint"):
                    raise ValueError("unsupported teacher producer kind")
                if not is_sha256(current.get("checkpoint_sha256")):
                    raise ValueError("invalid teacher checkpoint hash")
                teacher = current.get("teacher")
                children = [] if teacher is None else [_validate_teacher_reference(teacher)]
            count = 1
            for child in children:
                count += visit(child, depth + 1)
                if count > MAX_ANCESTRY_NODES:
                    raise ValueError("teacher ancestry exceeds supported node count")
            return count
        finally:
            active.remove(identity)

    visit(producer, 0)
