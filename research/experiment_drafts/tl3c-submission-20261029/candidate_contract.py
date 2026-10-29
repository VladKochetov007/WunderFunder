from __future__ import annotations

import hashlib
import json
from pathlib import Path

from candidate_io import JsonObject, object_value, pinned_path, read_json

SPLIT_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
TRAINING_SOURCE = "d174a35541ebfe6e0efc66fd124d491daf5c5643c338f5d0702c33f404418889"
ADOPTED_SOURCE = "cec4d6b3ad5fceefc5aa3b27b3badbaf4ff9c5555abc8358564cf0fb62f7bb1b"
PR49_COMMIT = "188c0788e571c6f981cdc6827fef0d50bd38cf14"
MEMBERS = ("tl3c_fbc1k_s5", "tl3c_fttall_tlead_s5", "tl3c_fttall_tlead_s6")
TEACHERS = {
    MEMBERS[0]: ("tl3c_fin256_s3", "tl3c_fin256_s4"),
    MEMBERS[1]: ("tl3c_fin256ft_s3", "tl3c_fin256ft_s4"),
    MEMBERS[2]: ("tl3c_fin256ft_s3", "tl3c_fin256ft_s4"),
}
INITIALIZERS = {
    "tl3c_fin256ft_s3": "tl3c_fin256_s3",
    "tl3c_fin256ft_s4": "tl3c_fin256_s4",
}
ALLOWED_RUNS = set(MEMBERS) | set(INITIALIZERS) | set(INITIALIZERS.values())


def endpoint(run: str) -> int:
    """Return the frozen zero-based endpoint, including two-epoch finetuned teachers."""
    if run not in ALLOWED_RUNS:
        raise ValueError(f"run is outside the frozen fresh TL3C lineage: {run}")
    return 1 if run in INITIALIZERS else 3


def expected_source(run: str) -> str:
    """Keep the R8 adopted first fit's source exception restricted to its one run."""
    endpoint(run)
    return ADOPTED_SOURCE if run == "tl3c_fin256_s3" else TRAINING_SOURCE


def ancestor_names(run: str) -> tuple[str, ...]:
    """List the complete declared teacher and initialization ancestry closure."""
    pending = list(TEACHERS[run])
    names: set[str] = set()
    while pending:
        name = pending.pop()
        names.add(name)
        if name in INITIALIZERS:
            pending.append(INITIALIZERS[name])
    return tuple(sorted(names))


def split_contract(identity: object) -> JsonObject:
    """Check the frozen train-only split without importing training or reading data payloads."""
    path = pinned_path(identity, "split_manifest")
    if object_value(identity, "split_manifest")["sha256"] != SPLIT_SHA256:
        raise ValueError("split differs from the admitted PR49 chronological-primary split")
    split = read_json(path)
    if (
        split.get("schema") != "wnn.chronological-primary.v1"
        or split.get("policy") != "official_train_fit_full_official_valid_evaluation_v1"
    ):
        raise ValueError("unexpected split policy")
    train = object_value(split.get("fit"), "split.fit").get("train")
    valid = object_value(split.get("evaluation"), "split.evaluation").get("valid")
    for label, groups, count in (("train", train, 10325), ("valid", valid, 1873)):
        if (
            not isinstance(groups, list)
            or any(type(group) is not int or group < 0 for group in groups)
            or len(groups) != count
            or len(set(groups)) != count
        ):
            raise ValueError(f"unexpected split {label} membership")
    if split["fit"].get("valid") != []:
        raise ValueError("official validation rows must not be fitted")
    return split


def verify_membership(provenance: JsonObject, split: JsonObject) -> None:
    """Require full, source-qualified train-only fitting and full-valid evaluation."""
    fit = object_value(provenance.get("fit_row_groups"), "provenance.fit_row_groups")
    for actual, expected in (
        (fit.get("train"), split["fit"]["train"]),
        (provenance.get("evaluation_row_groups"), split["evaluation"]["valid"]),
    ):
        if (
            not isinstance(actual, list)
            or any(type(group) is not int for group in actual)
            or len(actual) != len(expected)
            or set(actual) != set(expected)
        ):
            raise ValueError("run membership differs from the frozen split")
    if fit.get("valid") != [] or provenance.get("split_manifest_sha256") != SPLIT_SHA256:
        raise ValueError("run fitted validation rows or used another split")


def verify_source(provenance: JsonObject, run: str) -> None:
    """Check the declared trainer digest and recompute its source-file aggregate."""
    source = object_value(provenance.get("trainer_implementation"), "trainer_implementation")
    files = object_value(source.get("files"), "trainer source files")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    if source.get("sha256") != expected_source(run) or hashlib.sha256(
        encoded
    ).hexdigest() != expected_source(run):
        raise ValueError(f"{run}: trainer source differs from the R8 admitted source")


def verify_metadata(record: JsonObject, split: JsonObject) -> JsonObject:
    """Verify pinned checkpoint and sidecar identity, source, split and fixed endpoint."""
    run = str(record.get("run"))
    epoch = endpoint(run)
    if record.get("training_source_sha256") != expected_source(run):
        raise ValueError(f"{run}: manifest source is not the admitted source")
    checkpoint = pinned_path(record.get("checkpoint"), f"{run}.checkpoint")
    provenance_path = pinned_path(record.get("provenance"), f"{run}.provenance")
    if checkpoint.name != f"model_ep{epoch}.pt":
        raise ValueError(f"{run}: use the explicit fixed endpoint filename")
    provenance = read_json(provenance_path)
    hashes = object_value(provenance.get("checkpoint_sha256"), "checkpoint_sha256")
    if hashes.get(checkpoint.name) != record["checkpoint"]["sha256"]:
        raise ValueError(f"{run}: checkpoint bytes disagree with provenance")
    args = object_value(provenance.get("args"), "provenance.args")
    if args.get("name") != run or provenance.get("selected_epoch") != epoch:
        raise ValueError(f"{run}: sidecar run or selected endpoint differs")
    if args.get("checkpoint_policy") != "fixed" or args.get("checkpoint_epoch") != epoch:
        raise ValueError(f"{run}: checkpoint was not selected by the frozen policy")
    verify_source(provenance, run)
    verify_membership(provenance, split)
    return provenance


def verify_ancestry(member: JsonObject, split: JsonObject) -> list[JsonObject]:
    """Verify named teacher endpoints and initializer hashes; cache consumption stays pending."""
    run = str(member["run"])
    records = member.get("ancestors")
    if not isinstance(records, list):
        raise ValueError(f"{run}: ancestry records are required")
    by_name = {str(object_value(item, "ancestor").get("run")): item for item in records}
    if len(by_name) != len(records) or set(by_name) != set(ancestor_names(run)):
        raise ValueError(f"{run}: ancestry must exactly cover declared teacher and init closure")
    for name, record in by_name.items():
        provenance = verify_metadata(record, split)
        args = object_value(provenance["args"], "ancestor.args")
        expected_init = INITIALIZERS.get(name)
        if expected_init is None:
            if args.get("init"):
                raise ValueError(f"{name}: unexpected additional initializer")
        elif (
            args.get("init") != f"{expected_init}@ep3"
            or provenance.get("initial_checkpoint_sha256")
            != by_name[expected_init]["checkpoint"]["sha256"]
        ):
            raise ValueError(f"{name}: initialization ancestry hash differs")
    return records


def verify_candidate_args(args: JsonObject, run: str) -> None:
    """Reject unsupported preprocessing or architecture before writing weights."""
    expected = {
        "name": run,
        "hidden": 256,
        "layers": 2,
        "dropout": 0,
        "arch": "gru",
        "cell": "gru",
        "enc": "linear" if run == MEMBERS[0] else "tiedlead",
        "seed": 6 if run == MEMBERS[2] else 5,
        "precision": "fp32",
        "eval_ema": True,
        "ema_decay": 0.999,
        "ema_teacher": 1,
        "train_source": "train",
        "train_seqs": 10325,
        "epochs": 4,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 3,
        "full_valid": True,
        "batch": 32,
        "chunk": 1000,
        "loss": "wmse",
        "optim": "tl3-muon",
        "distill": ",".join(TEACHERS[run]),
        "distill_alpha": 0.5,
        "distill_boot_h": 8,
        "raw_only": False,
        "online_norm": False,
        "heuristics": False,
        "numeric_map": "none",
        "privileged_mode": "none",
        "research_features": "none",
        "readout_activation": "gelu",
        "head_layers": 1,
        "recurrent_rank": 0,
        "qat": "none",
        "aux": "",
        "aux_next_k": "",
        "aux_next": 0.3,
        "aux_struct": 0.3,
        "aux_tier": 0,
        "aux_mask": 0,
        "aux_sign": 0,
        "aux_quote": 0,
        "distill_family_weight": 0,
        "sentinel_values": "unchanged",
        "sentinel_indicators": "none",
        "swa_frac": 0,
    }
    for name, value in expected.items():
        if args.get(name) != value:
            raise ValueError(f"{run}: unsupported {name}={args.get(name)!r}; expected {value!r}")
    for name in ("encoder_width", "recurrent_width", "readout_width"):
        if args.get(name) not in (None, 256):
            raise ValueError(f"{run}: unsupported {name}")
    for name in (
        "input_lead",
        "snap_codebook",
        "init",
        "prop_input",
        "fit_groups_manifest",
        "fit_subset_manifest",
        "distill_cache_root",
        "student_frontier_contract",
    ):
        if args.get(name):
            raise ValueError(f"{run}: unsupported input/initialization override {name}")


def load_manifest(path: Path) -> tuple[JsonObject, JsonObject]:
    """Validate the frozen member set and split before receiver-side checkpoint export."""
    manifest = read_json(path)
    if manifest.get("schema") != "tl3c-submission-inputs-v1":
        raise ValueError("unsupported input manifest schema")
    members = manifest.get("members")
    if not isinstance(members, list) or [item.get("run") for item in members] != list(MEMBERS):
        raise ValueError("manifest must list the three frozen fresh TL3C runs in recipe order")
    split = split_contract(manifest.get("split_manifest"))
    return manifest, split
