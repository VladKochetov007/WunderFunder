"""One-shot admission for the fixed first-cell five-arm prediction export."""

PROJECT_BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
DATA_ROOT = PROJECT_BASE / "data"
SOURCE_ROOT = PROJECT_BASE / "source-fg19-cache-68ff6829"
SPLIT_PATH = PROJECT_BASE / "splits/fold0.json"
SCRATCH_ROOT = PROJECT_BASE / "research_tmp"
FIT_RECEIPTS_ROOT = SCRATCH_ROOT / "fg19_child_allthree_fit_v1"
CACHE_ROOT = SCRATCH_ROOT / "fg19_t3_fit_homogeneous_cache_v1"
CACHE_RECEIPTS_ROOT = PROJECT_BASE / "research/full_gru_tweaks_r1/cache_cpu_v2"
PROFILE_ROOT = SCRATCH_ROOT / "fg19_child_allthree_profile_v1"
SUPPORT_ROOT = SCRATCH_ROOT / "fg19_child_allthree_verify_v2"
EVIDENCE_ROOT = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_evaluation_v1"
OUTPUT_ROOT = EVIDENCE_ROOT / "exports"
CANONICAL_PYTHON = Path("/home/grinchenkoeg/WunderFunder/.venv/bin/python")
PORTABLE_REGISTRY = PROJECT_BASE / "support-source/terminal-v1/PORTABLE_SOURCES.json"
UNIT = "computer-vision-homework-fg19-first-cell-export-v1.service"
NATIVE_UNIT = "computer-vision-homework-fg19-child-allthree-fit-v1.service"
NATIVE_INVOCATION = "94d3a7132a434d448345b0206bd3daa3"
GPU_UUID = "GPU-ec74220d-78d7-cc47-69fc-0de1fb580944"
CANONICAL_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
CANONICAL_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
EXPORT_BINDING_SHA256 = "6cd8c46f31214623d97e54c32bac9395608bfc918b9b44e879e44ad2548f0eb9"
EXPORT_SOURCE_MANIFEST_SHA256 = "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
EXPORTER_SHA256 = "1587a29b2269f6592a6487fc48765a8e86d4cc23aab4951f00966d970ca3dfa1"
FIT_ADMISSION_BYTES = 12_912
FIT_ADMISSION_SHA256 = "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540"
FIT_LAUNCH_BYTES = 9_594
FIT_LAUNCH_SHA256 = "ec9d8dc819894e148753838ba33ba95080c192bacd5ac53413bd6847568fd538"
FIT_LAUNCH_INVOCATION = NATIVE_INVOCATION
PORTABLE_REGISTRY_BYTES = 2_157
PORTABLE_REGISTRY_SHA256 = "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378"
PORTABLE_SOURCE_NAMES = (
    "cache/chain_folds.npz",
    "cache/chains.npz",
    "cache/train_targets.npz",
    "cache/valid_targets.npz",
    "datasets/train.parquet",
    "datasets/valid.parquet",
)
READONLY_RECEIPTS = {
    "PARENT_TERMINAL_ACCEPTANCE.json": "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8",
    "PARENT_EXACT_STOP.json": "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8",
    "DICTIONARY_ACCEPTANCE.json": "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f",
    "DICTIONARY_EXACT_STOP.json": "2eb44a4381de06f28ed3d7438ace320ef3274eae37c89074fc481f58f5d88158",
    "PROFILE_TERMINAL_ACCEPTANCE.json": "9235b1670e80de8e18d322ab6e40dac102a7a8d8c028250f7b08262d8692c045",
    "PROFILE_EXACT_STOP.json": "a0437f4da0b2534721909087bab1ddf40e068c4ba6911870472b7374e4bb7dce",
    "CONTROL_TERMINAL_ACCEPTANCE.json": "92aaf5eeed156aaad25c4adbb6700d41b0527600b520dd051657a43a50bbb6d2",
    "CONTROL_EXACT_STOP.json": "600cb3c4a56c7d0731ed2bf6174d47f5b09623c7ca22a7e5fce77d4d358ae012",
    "EMA_TERMINAL_ACCEPTANCE.json": "0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8",
    "EMA_EXACT_STOP.json": "548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf",
    "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json": "283c447b86a9cc50c2ffab849a0d0839307accade1083ba53dcc7217a9da82b1",
    "DICTIONARY_CHILD_EXACT_STOP.json": "f1a73fc1bf13943e1dcc9cf018ebd4225986c4cfcdccea1b71c5ee961a54a5a3",
    "NLL_CHILD_TERMINAL_ACCEPTANCE.json": "d94457b11ce22a368cdc93df02e473c9a82cde6ddebd26cc6957bafc143750c5",
    "NLL_CHILD_EXACT_STOP.json": "c63be79b27d131acb0a38d2ad641fa7d72344cc344c26ab94ca77faf3b4a74f1",
}
PROFILE_PINS = {
    "ADMISSION.json": "908d066878b6a5b69549077a0002c6d6f1ed306bc10f5b059fa6ca423d2da2c5",
    "LAUNCH.json": "df171f46017dbb4b4b7bdfaaeb374db651d6d972eb11e8ffcebe917a4b464d44",
    "unit.log": "b4480cd8529ecbb0d5ec1cf840f0c9a312b541f9598cc4cd4cfa19084350c289",
}
CACHE_METADATA_SHA256 = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
CACHE_ACCEPTANCE_SHA256 = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"
CACHE_STOP_SHA256 = "286601bdf82dd19c94226ae06d5bef4841b59061c66e115c6fccb71370788ea4"
PARENT_CHECKPOINT_SHA256 = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
DICTIONARY_SHA256 = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
MAX_SOURCE_BYTES = 128_000_000
MAX_METADATA_BYTES = 16_000_000
PARENT_TARGET_BYTES = 17_000_000_000
PARENT_RESERVE_BYTES = 1_000_000_000
WORKER_MEMORY_HIGH_BYTES = 7_500_000_000
WORKER_MEMORY_MAX_BYTES = 8_000_000_000
WORKER_RUNTIME_SECONDS = 7_200
WORKER_GPU_ALLOCATOR_BYTES = 4_294_967_296
GPU_FREE_RESERVE_BYTES = 1_000_000_000
OUTPUT_BYTES = 374_400_000
OUTPUT_FREE_RESERVE_BYTES = 1_000_000_000
MINIMUM_OUTPUT_FREE_BYTES = 500_000_000
MINIMUM_FILESYSTEM_FREE_BYTES = 25 * 1024**3
EXPECTED_FS_DEVICE = 64_512
PAYLOAD_B64: dict[str, str] = {"__UNBOUND_PAYLOADS__": "__UNBOUND_PAYLOADS__"}
PAYLOAD_PINS: dict[str, tuple[int, str]] = {
    "export_first_cell_v1.py": (
        64_315,
        EXPORTER_SHA256,
    ),
    "EXPORT_BINDING_V1.json": (
        15_798,
        EXPORT_BINDING_SHA256,
    ),
    "EXPORT_SOURCE_MANIFEST.json": (
        27_569,
        EXPORT_SOURCE_MANIFEST_SHA256,
    ),
    "ALLTHREE_TERMINAL_ACCEPTANCE.json": (
        20_208,
        "fada500ede63a175d7d055d38fbef5d38110aef481a91771ae3c5717c1e7026f",
    ),
    "ALLTHREE_EXACT_STOP.json": (
        3_208,
        "0154b3f35c4fb849d0db7e7b60ce1f4ce2fa01ab2203c17db65877f89353f13d",
    ),
}


def payload_bytes() -> dict[str, bytes]:
    """Decode only five fixed small-code/receipt payloads before any mutation."""
    require(set(PAYLOAD_B64) == set(PAYLOAD_PINS), "rendered payload membership differs")
    result: dict[str, bytes] = {}
    for name, encoded in PAYLOAD_B64.items():
        expected_bytes, expected_sha = PAYLOAD_PINS[name]
        require(type(expected_bytes) is int and expected_bytes > 0, "bad fixed payload length")
        require(isinstance(encoded, str) and encoded, f"payload is unbound: {name}")
        raw = base64.b64decode(encoded, validate=True)
        require(len(raw) == expected_bytes, f"payload length differs: {name}")
        require(digest(raw) == expected_sha, f"payload SHA differs: {name}")
        result[name] = raw
    return result


def parse_json(raw: bytes, name: str) -> dict[str, object]:
    """Parse one already bounded JSON object without changing its original bytes."""
    return json_object(raw, name)


def stat_regular(path: Path, *, follow_leaf: bool = False) -> dict[str, object]:
    """Return one stat-only observation and reject unregistered leaf links."""
    before_link = path.lstat()
    islink = stat.S_ISLNK(before_link.st_mode)
    require(not islink or follow_leaf, f"unapproved symbolic-link leaf: {path}")
    before_target = path.stat() if islink else before_link
    resolved_before = path.resolve(strict=True)
    after_link = path.lstat()
    after_target = path.stat() if islink else after_link
    resolved_after = path.resolve(strict=True)
    require(
        (
            before_link.st_dev,
            before_link.st_ino,
            before_link.st_mode,
            before_link.st_size,
            before_link.st_mtime_ns,
            before_target.st_dev,
            before_target.st_ino,
            before_target.st_size,
            before_target.st_mtime_ns,
            resolved_before,
        )
        == (
            after_link.st_dev,
            after_link.st_ino,
            after_link.st_mode,
            after_link.st_size,
            after_link.st_mtime_ns,
            after_target.st_dev,
            after_target.st_ino,
            after_target.st_size,
            after_target.st_mtime_ns,
            resolved_after,
        ),
        f"file identity changed during stat-only observation: {path}",
    )
    observed = after_target
    require(stat.S_ISREG(observed.st_mode), f"expected regular file: {path}")
    return {
        "path": str(path),
        "resolved": str(resolved_after),
        "islink": islink,
        "bytes": observed.st_size,
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mtime_ns": observed.st_mtime_ns,
    }


def validate_registered_data(
    binding: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Join the pinned portable registry and six admission identities to fresh stat-only paths."""
    registered = binding.get("registered_data")
    require(
        isinstance(registered, dict)
        and set(registered)
        == {"admission", "portable_manifest", "source_identities", "valid_parquet"},
        "registered-data binding is malformed",
    )
    admission_pin = registered.get("admission")
    require(
        isinstance(admission_pin, dict)
        and set(admission_pin) == {"absolute_path", "bytes", "relative_path", "root_key", "sha256"}
        and admission_pin.get("root_key") == "fit_receipts_root"
        and admission_pin.get("relative_path") == "ADMISSION.json"
        and admission_pin.get("absolute_path") == str(FIT_RECEIPTS_ROOT / "ADMISSION.json"),
        "registered-data admission pin is missing or points elsewhere",
    )
    admission_path = Path(str(admission_pin.get("absolute_path", "")))
    raw, admission_stat = stable_file(admission_path, 20_000)
    require(
        (len(raw), digest(raw)) == (admission_pin.get("bytes"), admission_pin.get("sha256")),
        "original admission pin differs",
    )
    live_admission = parse_json(raw, "original AllThree admission")
    expected_sources = registered.get("source_identities")
    admitted_sources = live_admission.get("data_identity_stats_only")
    require(
        isinstance(expected_sources, dict)
        and len(expected_sources) == 6
        and admitted_sources == expected_sources
        and set(expected_sources) == set(PORTABLE_SOURCE_NAMES),
        "six registered source identities differ from the original admission",
    )
    allthree_arm = next(
        (
            arm
            for arm in binding.get("arms", [])
            if isinstance(arm, dict) and arm.get("arm") == "allthree"
        ),
        None,
    )
    require(
        isinstance(allthree_arm, dict)
        and live_admission.get("schema") == "fg19-child-allthree-fit-admission.v1"
        and live_admission.get("run_name") == allthree_arm.get("run_name")
        and live_admission.get("unit") == allthree_arm.get("unit")
        and live_admission.get("source_commit") == binding.get("source", {}).get("commit")
        and live_admission.get("split_sha256") == binding.get("split", {}).get("sha256"),
        "registered-data admission lineage differs",
    )
    registry_pin = registered.get("portable_manifest")
    require(
        isinstance(registry_pin, dict)
        and set(registry_pin) == {"absolute_path", "bytes", "sha256"}
        and registry_pin.get("absolute_path") == str(PORTABLE_REGISTRY)
        and registry_pin.get("bytes") == PORTABLE_REGISTRY_BYTES
        and registry_pin.get("sha256") == PORTABLE_REGISTRY_SHA256,
        "portable registry pin is missing or differs",
    )
    registry_path = Path(str(registry_pin.get("absolute_path", "")))
    require(registry_path == PORTABLE_REGISTRY, "portable registry path differs")
    require(
        registry_path.resolve(strict=True) == registry_path, "portable registry path is redirected"
    )
    registry_raw, registry_stat = stable_file(registry_path, 10_000)
    require(
        (len(registry_raw), digest(registry_raw))
        == (PORTABLE_REGISTRY_BYTES, PORTABLE_REGISTRY_SHA256)
        == (registry_pin.get("bytes"), registry_pin.get("sha256")),
        "portable registry bytes differ",
    )
    registry = parse_json(registry_raw, "portable source registry")
    sources = registry.get("sources")
    require(
        isinstance(registry, dict)
        and set(registry) == {"created_utc", "sources", "split_manifest_sha256", "version"}
        and type(registry.get("version")) is int
        and registry.get("version") == 1
        and registry.get("split_manifest_sha256") == binding.get("split", {}).get("sha256")
        and isinstance(sources, dict)
        and set(sources) == set(PORTABLE_SOURCE_NAMES),
        "portable source registry schema or membership differs",
    )
    observed: dict[str, object] = {}
    for relative in PORTABLE_SOURCE_NAMES:
        expected = expected_sources[relative]
        source_entry = sources[relative]
        identity_types = {
            "bytes": int,
            "device": int,
            "inode": int,
            "islink": bool,
            "mtime_ns": int,
            "path": str,
            "registered_sha256": str,
            "resolved": str,
        }
        require(
            isinstance(expected, dict)
            and set(expected) == set(identity_types)
            and all(type(expected.get(key)) is kind for key, kind in identity_types.items())
            and expected.get("path") == str(DATA_ROOT / relative)
            and Path(str(expected.get("resolved"))).is_absolute()
            and expected.get("bytes", 0) > 0
            and expected.get("device", -1) >= 0
            and expected.get("inode", 0) > 0
            and expected.get("mtime_ns", -1) >= 0
            and len(str(expected.get("registered_sha256", ""))) == 64
            and all(
                character in "0123456789abcdef"
                for character in str(expected.get("registered_sha256", ""))
            )
            and isinstance(source_entry, dict)
            and set(source_entry) == {"original_identity", "sha256"},
            f"source identity row is malformed: {relative}",
        )
        path = DATA_ROOT / relative
        actual = stat_regular(path, follow_leaf=expected.get("islink") is True)
        require(
            actual == {key: expected.get(key) for key in actual},
            f"registered source stat differs: {relative}",
        )
        original_identity = source_entry.get("original_identity")
        require(
            isinstance(original_identity, dict)
            and type(original_identity.get("bytes")) is int
            and type(original_identity.get("mtime_ns")) is int
            and original_identity.get("bytes") == expected.get("bytes")
            and original_identity.get("mtime_ns") == expected.get("mtime_ns")
            and source_entry.get("sha256") == expected.get("registered_sha256")
            and len(str(source_entry.get("sha256", ""))) == 64
            and all(
                character in "0123456789abcdef" for character in str(source_entry.get("sha256", ""))
            )
            and (
                "sha256" not in original_identity
                or original_identity.get("sha256") == source_entry.get("sha256")
            ),
            f"portable source membership/identity differs: {relative}",
        )
        observed[relative] = actual
    observations = {
        "admission": {**admission_stat, "sha256": digest(raw), "read_only": True},
        "portable_manifest": {**registry_stat, "sha256": digest(registry_raw), "read_only": True},
        "source_names": list(PORTABLE_SOURCE_NAMES),
        "source_identities_stat_only": observed,
        "payloads_read_or_hashed_by_this_gate": False,
    }
    return observations, live_admission


def check_source_tree(payload: dict[str, bytes], binding: dict[str, object]) -> dict[str, object]:
    """Hash the exact 167-file accepted source closure before creating the evidence directory."""
    manifest_raw = payload["EXPORT_SOURCE_MANIFEST.json"]
    source = binding.get("source")
    require(isinstance(source, dict), "export source binding is malformed")
    require(
        len(manifest_raw) == source.get("manifest_bytes")
        and digest(manifest_raw) == source.get("manifest_sha256") == EXPORT_SOURCE_MANIFEST_SHA256,
        "embedded export source manifest differs",
    )
    manifest = parse_json(manifest_raw, "export source manifest")
    entries = manifest.get("files")
    require(
        manifest.get("schema") == "fg19.teacher-cache-source-manifest.v1"
        and manifest.get("source_commit") == CANONICAL_COMMIT
        and manifest.get("source_root") == str(SOURCE_ROOT)
        and isinstance(entries, list)
        and len(entries) == 167,
        "source manifest identity differs",
    )
    require(
        source.get("source_set_sha256")
        == "9c461bcc5883572a071d41a86e18b9a91136099cf8fa87371ce7c3ee41445bd8",
        "source set pin differs",
    )
    require(
        SOURCE_ROOT.is_dir()
        and not SOURCE_ROOT.is_symlink()
        and SOURCE_ROOT.resolve(strict=True) == SOURCE_ROOT,
        "source root is redirected",
    )
    paths: set[str] = set()
    total = 0
    for row in entries:
        require(isinstance(row, dict), "source manifest row is malformed")
        relative = Path(str(row.get("path", "")))
        size, expected_sha = row.get("bytes"), row.get("sha256")
        require(
            not relative.is_absolute()
            and relative.parts
            and ".." not in relative.parts
            and type(size) is int
            and size >= 0
            and isinstance(expected_sha, str)
            and len(expected_sha) == 64
            and str(relative) not in paths,
            "source manifest entry is unsafe or duplicated",
        )
        path = SOURCE_ROOT / relative
        ancestor = path
        while ancestor != SOURCE_ROOT:
            require(not ancestor.is_symlink(), f"source path traverses a symbolic link: {ancestor}")
            ancestor = ancestor.parent
        actual_sha, info = file_hash(path, size)
        require(
            (actual_sha, info["bytes"]) == (expected_sha, size),
            f"source file pin differs: {relative}",
        )
        total += size
        require(total <= MAX_SOURCE_BYTES, "source closure exceeds the byte cap")
        paths.add(str(relative))
    actual_files: set[str] = set()
    for root, dirs, files in os.walk(SOURCE_ROOT, followlinks=False):
        for name in dirs + files:
            path = Path(root) / name
            require(not path.is_symlink(), "source inventory contains a symbolic link")
        actual_files.update(str((Path(root) / name).relative_to(SOURCE_ROOT)) for name in files)
    require(actual_files == paths, "source inventory differs from the 167-file manifest")
    return {
        "source_root": str(SOURCE_ROOT),
        "commit": CANONICAL_COMMIT,
        "files": len(paths),
        "bytes_hashed": total,
        "manifest_sha256": digest(manifest_raw),
        "source_set_sha256": source["source_set_sha256"],
    }


def check_support_helpers(binding: dict[str, object]) -> dict[str, object]:
    """Statically pin the two restricted-loader support modules before any private import."""
    source = binding.get("source")
    require(isinstance(source, dict), "source helper pins are missing")
    observed: dict[str, object] = {}
    for key in ("restricted_loader", "restricted_loader_support"):
        row = source.get(key)
        require(isinstance(row, dict), f"source helper pin is malformed: {key}")
        relative = Path(str(row.get("relative_path", "")))
        path = SUPPORT_ROOT / relative
        ancestor = path
        while ancestor != SUPPORT_ROOT:
            require(
                not ancestor.is_symlink(), f"support path traverses a symbolic link: {ancestor}"
            )
            ancestor = ancestor.parent
        sha, info = file_hash(path, int(row["bytes"]))
        require(
            (sha, info["bytes"]) == (row["sha256"], row["bytes"]), f"support helper changed: {key}"
        )
        observed[key] = {**info, "sha256": sha}
    for root in (SOURCE_ROOT, SUPPORT_ROOT):
        require(
            not any(path.name == "__pycache__" for path in root.rglob("__pycache__"))
            and not any(path.suffix == ".pyc" for path in root.rglob("*") if path.is_file()),
            f"bytecode exists under pinned code root: {root}",
        )
    return observed


def check_fit_receipts(payload: dict[str, bytes], binding: dict[str, object]) -> dict[str, object]:
    """Verify original AllThree receipt lineage and all five accepted/stopped models without loading weights."""
    original_admission, admission_stat = stable_file(FIT_RECEIPTS_ROOT / "ADMISSION.json", 20_000)
    original_launch, launch_stat = stable_file(FIT_RECEIPTS_ROOT / "LAUNCH.json", 20_000)
    require(
        (len(original_admission), digest(original_admission))
        == (FIT_ADMISSION_BYTES, FIT_ADMISSION_SHA256),
        "original AllThree admission changed",
    )
    require(
        (len(original_launch), digest(original_launch)) == (FIT_LAUNCH_BYTES, FIT_LAUNCH_SHA256),
        "original AllThree launch changed",
    )
    admission = parse_json(original_admission, "original AllThree admission")
    launch = parse_json(original_launch, "original AllThree launch")
    require(
        (admission.get("schema"), admission.get("run_name"), admission.get("unit"))
        == (
            "fg19-child-allthree-fit-admission.v1",
            "fg19_child_allthree_f0_s1001_fit_v1",
            NATIVE_UNIT,
        ),
        "original AllThree admission identity differs",
    )
    require(
        launch.get("schema") == "fg19-child-allthree-fit-launch.v1"
        and launch.get("unit") == NATIVE_UNIT
        and launch.get("invocation_id") == FIT_LAUNCH_INVOCATION
        and _is_invocation_id(launch.get("invocation_id"))
        and launch.get("submission", {}).get("outcome") == "submitted"
        and launch.get("submission", {}).get("returncode") == 0,
        "original AllThree launch receipt differs",
    )
    acceptance_row = payload["ALLTHREE_TERMINAL_ACCEPTANCE.json"]
    stop_raw = payload["ALLTHREE_EXACT_STOP.json"]
    receipt_pairs: dict[str, tuple[bytes, dict[str, object]]] = {}
    expected_acceptance_schemas = {
        "control": "fg19-child-control-fit-terminal-acceptance.v1",
        "ema": "fg19-child-ema-fit-terminal-acceptance.v1",
        "sparse": "fg19-child-dictionary-fit-terminal-acceptance.v1",
        "nll": "fg19-child-nll-fit-terminal-acceptance.v1",
        "allthree": "fg19-child-allthree-fit-terminal-acceptance.v1",
    }
    require(
        len(binding.get("arms", [])) == 5
        and [
            (row.get("arm"), row.get("representation_mode"))
            for row in binding["arms"]
            if isinstance(row, dict)
        ]
        == [
            ("control", "base"),
            ("ema", "ema"),
            ("sparse", "sparse"),
            ("nll", "base"),
            ("allthree", "ema_sparse"),
        ],
        "fixed five-arm order or representation modes differ",
    )
    for arm in binding["arms"]:
        require(isinstance(arm, dict), "arm binding is malformed")
        name = arm.get("arm")
        terminal = arm.get("terminal_acceptance")
        stop = arm.get("exact_stop")
        require(
            isinstance(name, str) and isinstance(terminal, dict) and isinstance(stop, dict),
            "arm receipt pins are malformed",
        )
        if name == "allthree":
            pair = (
                acceptance_row,
                {
                    "source": "fixed ROOT-authority payload; staged only after every pre-mutation gate",
                    "read_only_original": True,
                },
            )
            stop_pair = (
                stop_raw,
                {
                    "source": "fixed ROOT-authority payload; staged only after every pre-mutation gate",
                    "read_only_original": True,
                },
            )
        else:
            terminal_path = Path(str(terminal.get("absolute_path", "")))
            stop_path = Path(str(stop.get("absolute_path", "")))
            require(
                terminal_path.parent == FIT_RECEIPTS_ROOT and stop_path.parent == FIT_RECEIPTS_ROOT,
                f"non-AllThree receipt path is outside the read-only fit receipt root: {name}",
            )
            pair = stable_file(terminal_path, 32_000)
            stop_pair = stable_file(stop_path, 10_000)
        terminal_raw, terminal_stat = pair
        exact_stop_raw, exact_stop_stat = stop_pair
        require(
            (len(terminal_raw), digest(terminal_raw))
            == (terminal.get("bytes"), terminal.get("sha256")),
            f"terminal acceptance pin differs: {name}",
        )
        require(
            (len(exact_stop_raw), digest(exact_stop_raw))
            == (stop.get("bytes"), stop.get("sha256")),
            f"exact-stop receipt pin differs: {name}",
        )
        accepted = parse_json(terminal_raw, f"{name} terminal acceptance")
        stopped = parse_json(exact_stop_raw, f"{name} exact stop")
        require(
            accepted.get("schema") == expected_acceptance_schemas.get(name)
            and accepted.get("source_commit") == CANONICAL_COMMIT
            and accepted.get("source_file_count") == 167
            and accepted.get("source_manifest_sha256") == CANONICAL_MANIFEST_SHA256
            and (
                (name == "control" and accepted.get("split_sha256") is None)
                or accepted.get("split_sha256") == binding.get("split", {}).get("sha256")
            ),
            f"accepted source or split lineage differs: {name}",
        )
        require(
            accepted.get("status") == "ACCEPTED"
            and accepted.get("run_name") == arm.get("run_name")
            and accepted.get("unit") == arm.get("unit")
            and accepted.get("invocation_id") == arm.get("invocation_id")
            and type(accepted.get("selected_epoch")) is int
            and accepted.get("selected_epoch") == 5
            and type(accepted.get("successful_updates")) is int
            and accepted.get("successful_updates") == 44_760
            and accepted.get("quality_score") is None,
            f"accepted fit identity or fixed epoch differs: {name}",
        )
        representation = accepted.get("input_representation")
        expected_model = arm.get("model")
        accepted_model = accepted.get("files", {}).get("model_ep5.pt")
        require(
            isinstance(representation, dict)
            and representation.get("artifact_sha256")
            == binding.get("representation", {}).get("artifact_sha256")
            and representation.get("state_sha256")
            == binding.get("representation", {}).get("state_sha256")
            and (
                (name == "control" and "mode" not in representation)
                or representation.get("mode") == arm.get("representation_mode")
            )
            and isinstance(expected_model, dict)
            and expected_model.get("relative_path")
            == f"artifacts/runs/{arm.get('run_name')}/model_ep5.pt"
            and isinstance(accepted_model, dict)
            and accepted_model.get("bytes") == expected_model.get("bytes")
            and accepted_model.get("sha256") == expected_model.get("sha256"),
            f"accepted representation or epoch-5 model pin differs: {name}",
        )
        require(
            stopped.get("unit") == arm.get("unit")
            and stopped.get("invocation_id") == arm.get("invocation_id")
            and isinstance(stopped.get("stop"), dict)
            and stopped["stop"].get("returncode") == 0
            and stopped["stop"].get("stdout") == ""
            and stopped["stop"].get("stderr") == ""
            and stopped.get("after", {}).get("ActiveState") == "inactive"
            and stopped.get("after", {}).get("SubState") == "dead"
            and stopped.get("after", {}).get("MainPID") == "0",
            f"exact stop receipt differs: {name}",
        )
        model_relative = Path(str(arm.get("model", {}).get("relative_path", "")))
        require(
            not model_relative.is_absolute() and ".." not in model_relative.parts,
            "unsafe checkpoint path",
        )
        model_path = DATA_ROOT / model_relative
        ancestor = model_path
        while ancestor != DATA_ROOT:
            require(
                not ancestor.is_symlink(), f"checkpoint path traverses a symbolic link: {ancestor}"
            )
            ancestor = ancestor.parent
        model_stat = stat_regular(model_path)
        require(
            model_stat["bytes"] == arm.get("model", {}).get("bytes"),
            f"checkpoint stat differs: {name}",
        )
        receipt_pairs[name] = (
            terminal_raw,
            {
                "terminal_acceptance": {
                    **terminal_stat,
                    "sha256": digest(terminal_raw),
                    "read_only": name != "allthree",
                },
                "exact_stop": {
                    **exact_stop_stat,
                    "sha256": digest(exact_stop_raw),
                    "read_only": name != "allthree",
                },
                "checkpoint_stat_only": {
                    **model_stat,
                    "expected_sha256_from_binding_not_rehashed": arm["model"]["sha256"],
                },
                "accepted_epoch": 5,
                "successful_updates": 44_760,
                "quality_score": None,
            },
        )
    return {
        "original_admission": {
            **admission_stat,
            "sha256": digest(original_admission),
            "read_only": True,
        },
        "original_launch": {**launch_stat, "sha256": digest(original_launch), "read_only": True},
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": NATIVE_INVOCATION,
        "five_accepted_stop_receipt_pairs_and_model_stats_only": {
            name: row for name, (_, row) in receipt_pairs.items()
        },
        "current_fit_unit_states_queried": False,
        "fit_stop_authority": "exact pinned accepted stop receipts; no historical unit query",
        "checkpoint_payloads_read_or_hashed": False,
    }


def check_predecessors(
    admission: dict[str, object], binding: dict[str, object]
) -> dict[str, object]:
    """Recheck fourteen accepted predecessor receipts, profile records, cache, parent and dictionary by exact pins."""
    staged = admission.get("staged_receipt_sha256")
    require(
        isinstance(staged, dict)
        and set(staged) == set(READONLY_RECEIPTS)
        and {name: staged.get(name) for name in READONLY_RECEIPTS} == READONLY_RECEIPTS,
        "fourteen predecessor receipt map differs",
    )
    receipts: dict[str, object] = {}
    for name, expected_sha in READONLY_RECEIPTS.items():
        raw, info = stable_file(FIT_RECEIPTS_ROOT / name, 1_000_000)
        require(digest(raw) == expected_sha, f"accepted predecessor receipt changed: {name}")
        receipts[name] = {**info, "sha256": expected_sha, "read_only": True}
    profile: dict[str, object] = {}
    for name, expected_sha in PROFILE_PINS.items():
        raw, info = stable_file(PROFILE_ROOT / name, 1_000_000)
        require(digest(raw) == expected_sha, f"profile metadata changed: {name}")
        profile[name] = {**info, "sha256": expected_sha, "read_only": True}
    for key, path, expected_sha in (
        (
            "parent_checkpoint_identity",
            DATA_ROOT / "artifacts/runs/fg19_parent_h256x2_f0_s1001_fit_r2_v1/model_ep3.pt",
            PARENT_CHECKPOINT_SHA256,
        ),
        (
            "dictionary",
            DATA_ROOT / "artifacts/research/fg19_dictionary_f0_v1/input_representation.pt",
            DICTIONARY_SHA256,
        ),
    ):
        identity = admission.get(key)
        require(
            isinstance(identity, dict) and Path(str(identity.get("path", ""))) == path,
            f"{key} path pin differs",
        )
        observed = stat_regular(path)
        require(
            identity.get("sha256") == expected_sha
            and (identity.get("bytes"), identity.get("device"), identity.get("inode"))
            == (observed["bytes"], observed["device"], observed["inode"]),
            f"{key} stat-only identity differs",
        )
    split_raw, split_stat = stable_file(SPLIT_PATH, 2_000_000)
    split_pin = binding.get("split")
    require(isinstance(split_pin, dict), "fixed split binding is missing")
    require(
        digest(split_raw)
        == split_pin.get("sha256")
        == "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
        "F0 split bytes differ",
    )
    split_value = parse_json(split_raw, "canonical F0 split")
    require(
        split_value.get("fold") == 0 and split_value.get("n_folds") == 4,
        "canonical F0 split identity differs",
    )
    cache = check_cache(admission, binding)
    return {
        "fourteen_predecessor_receipts": receipts,
        "profile_files": profile,
        "parent_checkpoint_stat_only": True,
        "dictionary_artifact_stat_only": True,
        "canonical_split": {**split_stat, "sha256": digest(split_raw)},
        "cache_stat_only": cache,
        "all_read_only": True,
        "checkpoint_dictionary_cache_arrays_read_or_hashed": False,
    }


def check_cache(admission: dict[str, object], binding: dict[str, object]) -> dict[str, object]:
    """Check accepted cache metadata and the two large arrays by stat only."""
    cache_identity = admission.get("cache")
    require(isinstance(cache_identity, dict), "cache identity is missing")
    acceptance_raw, acceptance_stat = stable_file(
        CACHE_RECEIPTS_ROOT / "TERMINAL_ACCEPTANCE.json", 10_000
    )
    stop_raw, stop_stat = stable_file(CACHE_RECEIPTS_ROOT / "EXACT_STOP.json", 10_000)
    require(digest(acceptance_raw) == CACHE_ACCEPTANCE_SHA256, "cache acceptance differs")
    require(digest(stop_raw) == CACHE_STOP_SHA256, "cache exact stop differs")
    metadata_raw, metadata_stat = stable_file(CACHE_ROOT / "metadata.json", MAX_METADATA_BYTES)
    require(digest(metadata_raw) == CACHE_METADATA_SHA256, "cache metadata differs")
    metadata = parse_json(metadata_raw, "cache metadata")
    require(
        metadata.get("split_manifest_sha256") == binding["split"]["sha256"], "cache split differs"
    )
    acceptance = parse_json(acceptance_raw, "cache acceptance")
    arrays: dict[str, object] = {}
    for name, size, count, expected_sha in (
        (
            "train",
            1_685_600_128,
            10_535,
            "4abbb9be31ec1135a5e0ff5bb87c93a2db6e44943e95d23ff70835b8044c5194",
        ),
        (
            "valid",
            224_800_128,
            1_405,
            "6963d3c565caf469b023a61ba6d2dd376d929b2bcf724af88492dc8adb76f591",
        ),
    ):
        path = CACHE_ROOT / f"main_homogeneous_{name}.npy"
        observed = stat_regular(path)
        recorded = cache_identity.get("array_stats_only", {}).get(name)
        output = acceptance.get("output", {}).get(name)
        require(
            isinstance(recorded, dict) and isinstance(output, dict),
            f"cache {name} receipts are missing",
        )
        require(
            {key: observed[key] for key in ("bytes", "device", "inode", "mtime_ns")} == recorded
            and observed["bytes"] == size
            and output.get("bytes") == size
            and output.get("sha256") == expected_sha
            and output.get("shape") == [count, 20_000, 2]
            and output.get("dtype") == "float32",
            f"cache {name} stat/receipt differs",
        )
        arrays[name] = {
            "stat_only": {key: observed[key] for key in ("bytes", "device", "inode", "mtime_ns")},
            "accepted_sha256_not_rehashed": expected_sha,
        }
    return {
        "metadata": {**metadata_stat, "sha256": CACHE_METADATA_SHA256, "read_only": True},
        "acceptance": {**acceptance_stat, "sha256": CACHE_ACCEPTANCE_SHA256, "read_only": True},
        "exact_stop": {**stop_stat, "sha256": CACHE_STOP_SHA256, "read_only": True},
        "arrays": arrays,
        "payloads_read_or_hashed": False,
    }


def check_parent_resources() -> dict[str, object]:
    """Require an empty two-core campaign slice, 17 GB headroom, ext4 scratch and 25 GiB free."""
    fields = ("ControlGroup", "CPUQuotaPerSecUSec", "MemoryHigh", "MemoryMax", "MemorySwapMax")
    parent = show("wnn-campaign.slice", fields)
    require(
        (
            parent.get("CPUQuotaPerSecUSec"),
            parent.get("MemoryHigh"),
            parent.get("MemoryMax"),
            parent.get("MemorySwapMax"),
        )
        == ("2s", "18000000000", "20000000000", "0"),
        "aggregate parent resource envelope differs",
    )
    control_group = parent.get("ControlGroup", "")
    require(
        control_group not in ("", "/") and control_group.endswith("/wnn-campaign.slice"),
        "parent cgroup is not scoped",
    )
    cgroup = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    require(
        cgroup.is_dir() and cgroup.resolve(strict=True) == cgroup,
        "parent cgroup unavailable or redirected",
    )
    stack = [cgroup]
    directories: list[Path] = []
    while stack:
        current_dir = stack.pop()
        directories.append(current_dir)
        require(len(directories) <= 128, "parent cgroup directory inventory exceeds cap")
        with os.scandir(current_dir) as entries:
            for count, entry in enumerate(entries, start=1):
                require(count <= 256, "parent cgroup directory entry count exceeds cap")
                require(not entry.is_symlink(), "parent cgroup traversal encountered a symlink")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    require(
                        len(stack) + len(directories) <= 128, "parent cgroup traversal exceeds cap"
                    )
    require(directories, "parent cgroup directory inventory is empty")
    populated: dict[str, str] = {}
    aggregate_proc_bytes = 0
    for current_dir in directories:
        proc_path = current_dir / "cgroup.procs"
        proc_raw = _bounded_proc_read(proc_path, 32_768)
        aggregate_proc_bytes += len(proc_raw)
        require(aggregate_proc_bytes <= 262_144, "parent cgroup process inventory exceeds byte cap")
        if proc_raw.strip():
            populated[str(proc_path)] = proc_raw.decode("ascii", errors="strict")
    require(not populated, "campaign cgroup has populated descendants")
    current_raw = _bounded_proc_read(cgroup / "memory.current", 4096).decode("ascii").strip()
    require(current_raw.isdigit(), "campaign memory.current is unavailable")
    current = int(current_raw)
    require(
        current + WORKER_MEMORY_MAX_BYTES + PARENT_RESERVE_BYTES < PARENT_TARGET_BYTES,
        "parent memory headroom is insufficient for the export unit",
    )
    events = {
        name: _bounded_proc_read(cgroup / name, 4096).decode("ascii", errors="strict")
        for name in ("memory.events", "memory.events.local")
    }
    local_events = dict(
        line.split()
        for line in events["memory.events.local"].splitlines()
        if len(line.split()) == 2
    )
    require(
        all(local_events.get(name) == "0" for name in ("oom", "oom_kill", "oom_group_kill")),
        "campaign-local OOM event counters are nonzero",
    )
    require(
        SCRATCH_ROOT.is_dir()
        and not SCRATCH_ROOT.is_symlink()
        and SCRATCH_ROOT.resolve(strict=True) == SCRATCH_ROOT,
        "scratch root is redirected",
    )
    project_stat, scratch_stat = PROJECT_BASE.stat(), SCRATCH_ROOT.stat()
    require(
        project_stat.st_dev == scratch_stat.st_dev == EXPECTED_FS_DEVICE,
        "project scratch device differs",
    )
    filesystem_stats = os.statvfs(PROJECT_BASE)
    free = filesystem_stats.f_bavail * filesystem_stats.f_frsize
    required_output_free = max(MINIMUM_OUTPUT_FREE_BYTES, OUTPUT_BYTES + OUTPUT_FREE_RESERVE_BYTES)
    require(
        free >= MINIMUM_FILESYSTEM_FREE_BYTES and free >= required_output_free,
        "physical filesystem lacks required free space",
    )
    filesystem = subprocess.run(
        ["findmnt", "-T", str(SCRATCH_ROOT), "-n", "-o", "FSTYPE"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    require(
        filesystem.returncode == 0 and filesystem.stdout.strip() == "ext4",
        "scratch filesystem is not ext4",
    )
    return {
        "systemd_parent": parent,
        "cgroup": str(cgroup),
        "all_populated_descendants": populated,
        "memory_current_bytes": current,
        "headroom_formula": "parent current + 8000000000 worker max + 1000000000 reserve < 17000000000",
        "memory_events": events,
        "project_filesystem_free_bytes": free,
        "required_minimum_free_bytes": MINIMUM_FILESYSTEM_FREE_BYTES,
        "scratch_fstype": filesystem.stdout.strip(),
        "filesystem_device": project_stat.st_dev,
    }


def gpu_owners() -> dict[str, object]:
    """Observe every selected-GPU process without signaling or otherwise touching it."""
    prefix = ["nvidia-smi", "-i", GPU_UUID]
    identity_output, _ = _bounded_gpu_query(
        prefix + ["--query-gpu=uuid,name,memory.free,memory.total", "--format=csv,noheader"]
    )
    rows = [line.strip() for line in identity_output.splitlines() if line.strip()]
    require(len(rows) == 1, "selected GPU query did not return exactly one device")
    values = [item.strip() for item in rows[0].split(",")]
    require(
        len(values) == 4 and values[0] == GPU_UUID and values[1], "selected GPU identity differs"
    )
    free, total = _memory_bytes(values[2]), _memory_bytes(values[3])
    require(total > 0 and free <= total, "selected GPU memory sample is inconsistent")
    require(
        free >= WORKER_GPU_ALLOCATOR_BYTES + GPU_FREE_RESERVE_BYTES,
        "selected GPU lacks allocator plus free reserve",
    )
    process_output, _ = _bounded_gpu_query(
        prefix + ["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"]
    )
    process_rows = [line.strip() for line in process_output.splitlines() if line.strip()]
    if len(process_rows) == 1 and process_rows[0].startswith("No running processes found"):
        process_rows = []
    require(len(process_rows) <= 256, "selected GPU process inventory exceeds cap")
    owners: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in process_rows:
        fields = [value.strip() for value in row.split(",")]
        require(
            len(fields) == 3
            and fields[0].isascii()
            and fields[0].isdigit()
            and int(fields[0]) > 0
            and str(int(fields[0])) == fields[0]
            and fields[1] == GPU_UUID
            and fields[2]
            and fields[0] not in seen,
            "selected GPU process row is malformed or duplicated",
        )
        seen.add(fields[0])
        proc = Path("/proc") / fields[0]
        before = proc.stat(follow_symlinks=False)
        start_before = _proc_starttime(proc, fields[0])
        cgroup = (
            _bounded_proc_read(proc / "cgroup", 32_768).decode("ascii", errors="strict").strip()
        )
        after = proc.stat(follow_symlinks=False)
        start_after = _proc_starttime(proc, fields[0])
        require(
            (before.st_ino, before.st_uid, start_before)
            == (after.st_ino, after.st_uid, start_after),
            "selected GPU PID changed during observation",
        )
        campaign_owned = "wnn-campaign.slice" in cgroup
        if after.st_uid == os.getuid() and not campaign_owned:
            command = _bounded_proc_read(proc / "cmdline", 8_192).decode("utf-8", errors="replace")
            campaign_owned = str(PROJECT_BASE) in command
        require(not campaign_owned, "selected GPU is occupied by a campaign project process")
        owners.append(
            {
                "pid": int(fields[0]),
                "gpu_uuid": fields[1],
                "used_memory": fields[2],
                "uid": after.st_uid,
                "cgroup": cgroup,
                "proc_starttime": start_before,
                "preserved": True,
                "action": "none",
            }
        )
    return {
        "uuid": GPU_UUID,
        "name": values[1],
        "memory_free_bytes": free,
        "memory_total_bytes": total,
        "minimum_free_bytes": WORKER_GPU_ALLOCATOR_BYTES + GPU_FREE_RESERVE_BYTES,
        "process_count": len(process_rows),
        "owners": owners,
        "foreign_process_action": "none",
    }


def _bounded_submit(command: list[str]) -> tuple[int, bytes, bytes]:
    """Capture systemd-run output under strict time and per-stream byte bounds."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        start_new_session=True,
    )
    require(
        process.stdout is not None and process.stderr is not None,
        "systemd-run pipes unavailable",
    )
    output = {"stdout": bytearray(), "stderr": bytearray()}
    streams = {process.stdout: "stdout", process.stderr: "stderr"}
    deadline = time.monotonic() + 30
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                selector.register(stream, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                require(remaining > 0, "systemd-run submission timed out; outcome is unknown")
                events = selector.select(remaining)
                require(events, "systemd-run submission timed out; outcome is unknown")
                for key, _ in events:
                    label = streams[key.fileobj]
                    room = 65_537 - len(output[label])
                    chunk = os.read(key.fileobj.fileno(), min(4096, room))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output[label].extend(chunk)
                    require(
                        len(output[label]) <= 65_536,
                        f"systemd-run {label} exceeded 64 KiB; outcome is unknown",
                    )
        remaining = deadline - time.monotonic()
        require(remaining > 0, "systemd-run submission timed out; outcome is unknown")
        returncode = process.wait(timeout=remaining)
        return returncode, bytes(output["stdout"]), bytes(output["stderr"])
    except (OSError, RuntimeError, subprocess.SubprocessError):
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
        process.stderr.close()


def preflight(payload: dict[str, bytes]) -> dict[str, object]:
    """Run all read-only source, artifact, receipt and resource gates before creating D."""
    binding_raw = payload["EXPORT_BINDING_V1.json"]
    require(digest(binding_raw) == EXPORT_BINDING_SHA256, "export binding payload differs")
    binding = parse_json(binding_raw, "export binding")
    require(
        binding.get("schema") == "fg19.first-cell-evaluation-export-binding.v1",
        "export binding schema differs",
    )
    require(
        (
            binding.get("project_base"),
            binding.get("data_root"),
            binding.get("source_root"),
            binding.get("evidence_root"),
            binding.get("output_dir"),
        )
        == (
            str(PROJECT_BASE),
            str(DATA_ROOT),
            str(SOURCE_ROOT),
            str(EVIDENCE_ROOT),
            str(OUTPUT_ROOT),
        ),
        "export binding paths differ",
    )
    require(
        binding.get("resource", {}).get("selected_gpu_uuid") == GPU_UUID
        and binding.get("resource", {}).get("allocator_bytes") == WORKER_GPU_ALLOCATOR_BYTES
        and binding.get("resource", {}).get("outputs_bytes") == OUTPUT_BYTES,
        "fixed export resource contract differs",
    )
    require(
        binding.get("split", {}).get("valid_row_group_count") == 1_873
        and binding.get("split", {}).get("fit_valid_group_count") == 1_405
        and binding.get("split", {}).get("evaluation_group_count") == 468
        and binding.get("split", {}).get("sequence_rows") == 20_000,
        "split row-group and sequence-row counts differ",
    )
    require(
        PROJECT_BASE.is_dir()
        and not PROJECT_BASE.is_symlink()
        and PROJECT_BASE.resolve(strict=True) == PROJECT_BASE,
        "project base is redirected",
    )
    require(
        DATA_ROOT.is_dir()
        and not DATA_ROOT.is_symlink()
        and DATA_ROOT.resolve(strict=True) == DATA_ROOT,
        "data root is redirected",
    )
    require(
        CANONICAL_PYTHON.is_file() and os.access(CANONICAL_PYTHON, os.X_OK),
        "canonical project Python is unavailable",
    )
    require(
        EVIDENCE_ROOT.parent == SCRATCH_ROOT and not os.path.lexists(EVIDENCE_ROOT),
        "evidence root already exists",
    )
    require(not os.path.lexists(OUTPUT_ROOT), "export output directory already exists")
    require(
        show(UNIT, ("LoadState",)).get("LoadState") == "not-found", "export unit already exists"
    )
    sources, original_admission = validate_registered_data(binding)
    source_tree = check_source_tree(payload, binding)
    support_helpers = check_support_helpers(binding)
    predecessor_data = check_predecessors(original_admission, binding)
    fit_data = check_fit_receipts(payload, binding)
    parent = check_parent_resources()
    gpu = gpu_owners()
    return {
        "binding_sha256": EXPORT_BINDING_SHA256,
        "source_manifest_sha256": EXPORT_SOURCE_MANIFEST_SHA256,
        "source_tree": source_tree,
        "restricted_loader_support_stat_and_hash": support_helpers,
        "registered_six_source_paths_stat_only": sources,
        "read_only_predecessors_and_cache_split": predecessor_data,
        "all_five_fit_receipts_and_checkpoint_stats": fit_data,
        "parent_disk_cgroup": parent,
        "selected_gpu_ownership": gpu,
        "output_and_evidence_absent_before_mutation": True,
        "unit_absent_before_mutation": True,
    }


def write_payloads(payload: dict[str, bytes]) -> dict[str, object]:
    """Create the private evidence directory and stage only the fixed worker packet."""
    EVIDENCE_ROOT.mkdir(mode=0o700, exist_ok=False)
    accepted_root = EVIDENCE_ROOT / "accepted_receipts"
    accepted_root.mkdir(mode=0o700, exist_ok=False)
    stage_map = {
        "export_first_cell_v1.py": payload["export_first_cell_v1.py"],
        "EXPORT_BINDING_V1.json": payload["EXPORT_BINDING_V1.json"],
        "EXPORT_SOURCE_MANIFEST.json": payload["EXPORT_SOURCE_MANIFEST.json"],
        "accepted_receipts/ALLTHREE_TERMINAL_ACCEPTANCE.json": payload[
            "ALLTHREE_TERMINAL_ACCEPTANCE.json"
        ],
        "accepted_receipts/ALLTHREE_EXACT_STOP.json": payload["ALLTHREE_EXACT_STOP.json"],
    }
    for relative, raw in stage_map.items():
        path = EVIDENCE_ROOT / relative
        write_exclusive(path, raw)
    for directory in (accepted_root, EVIDENCE_ROOT):
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    staged: dict[str, object] = {}
    for relative, raw in stage_map.items():
        disk, info = stable_file(EVIDENCE_ROOT / relative, len(raw))
        require(disk == raw, f"staged payload differs: {relative}")
        staged[relative] = {**info, "sha256": digest(raw)}
    return staged


def submit_once(admission: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Submit exactly one bounded export unit and durably bind its actual invocation."""
    command = [
        "systemd-run",
        "--user",
        "--unit=" + UNIT,
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=Slice=wnn-campaign.slice",
        "--property=CPUQuota=100%",
        "--property=MemoryHigh=" + str(WORKER_MEMORY_HIGH_BYTES),
        "--property=MemoryMax=" + str(WORKER_MEMORY_MAX_BYTES),
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=" + str(WORKER_RUNTIME_SECONDS),
        "--property=TimeoutStopSec=45",
        "--property=StandardOutput=append:" + str(EVIDENCE_ROOT / "unit.log"),
        "--property=StandardError=append:" + str(EVIDENCE_ROOT / "unit.log"),
        "--working-directory=" + str(EVIDENCE_ROOT),
        "--setenv=CUDA_VISIBLE_DEVICES=" + GPU_UUID,
        "--setenv=WNN_PROJECT_ROOT=" + str(DATA_ROOT),
        "--setenv=TMPDIR=" + str(SCRATCH_ROOT),
        "--setenv=WNN_THREADS=1",
        "--setenv=WNN_GPU_MEMORY_BYTES=" + str(WORKER_GPU_ALLOCATOR_BYTES),
        "--setenv=WNN_PORTABLE_DATA_MANIFEST=" + str(PORTABLE_REGISTRY),
        "--setenv=WNN_LOADER_WORKERS=0",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=NUMEXPR_NUM_THREADS=1",
        "--setenv=NVIDIA_TF32_OVERRIDE=0",
        "--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8",
        str(CANONICAL_PYTHON),
        "-E",
        "-B",
        "-I",
        str(EVIDENCE_ROOT / "export_first_cell_v1.py"),
        "--project-base",
        str(PROJECT_BASE),
        "--data-root",
        str(DATA_ROOT),
        "--source-root",
        str(SOURCE_ROOT),
        "--support-root",
        str(SUPPORT_ROOT),
        "--evidence-root",
        str(EVIDENCE_ROOT),
        "--output-dir",
        str(OUTPUT_ROOT),
    ]
    try:
        returncode, stdout, stderr = _bounded_submit(command)
        submission = {
            "outcome": "submitted" if returncode == 0 else "submit_failed_or_unknown",
            "returncode": returncode,
            "stdout_b64": base64.b64encode(stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
            "stdout_truncated": False,
            "stderr_truncated": False,
        }
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        submission = {
            "outcome": "submit_error_or_timeout_unknown_never_retry",
            "returncode": None,
            "error": repr(error),
            "stdout_b64": "",
            "stderr_b64": "",
        }
    submission["argv"] = command
    fields = (
        "Id",
        "LoadState",
        "InvocationID",
        "ActiveState",
        "SubState",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "Result",
        "ControlGroup",
    )
    post: dict[str, object] = {"outcome": "captured", "properties": {}}
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            properties = show(UNIT, fields)
            post["properties"] = properties
            if properties.get("Id") == UNIT and _is_invocation_id(
                properties.get("InvocationID", "")
            ):
                break
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            post = {"outcome": "query_failed_unknown", "error": repr(error), "properties": {}}
        time.sleep(0.25)
    properties = post.get("properties", {})
    invocation_id = properties.get("InvocationID") if isinstance(properties, dict) else None
    if not _is_invocation_id(invocation_id):
        unresolved = {
            "schema": "fg19.first-cell-export-unresolved-launch.v1",
            "unit": UNIT,
            "invocation_id": None,
            "submission": submission,
            "systemctl_after_submission": post,
            "unknown_outcome_policy": "Inspect only this exact unit and evidence directory; never retry or recreate it.",
        }
        write_json_exclusive(EVIDENCE_ROOT / "LAUNCH_UNRESOLVED.json", unresolved)
        raise RuntimeError(
            "systemd invocation ID unresolved; durable unknown outcome retained, never retry"
        )
    launch = {
        "schema": "fg19.first-cell-export-launch.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "invocation_id": invocation_id,
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": NATIVE_INVOCATION,
        "admission_sha256": digest(
            (json.dumps(admission, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        ),
        "submission": submission,
        "systemctl_after_submission": post,
        "unknown_outcome_policy": "Inspect only this exact unit and evidence directory; never retry or recreate it.",
    }
    write_json_exclusive(EVIDENCE_ROOT / "LAUNCH.json", launch)
    if submission.get("outcome") != "submitted" or submission.get("returncode") != 0:
        raise RuntimeError(
            "submission outcome is not a confirmed single launch; inspect durable records and never retry"
        )
    return launch, submission


def _is_invocation_id(value: object) -> bool:
    """Accept only a lowercase 32-hex systemd invocation identifier."""
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def main() -> None:
    """Perform one read-only preflight, stage the exact packet, and submit at most once."""
    payload = payload_bytes()
    gates = preflight(payload)
    staged = write_payloads(payload)
    admission = {
        "schema": "fg19.first-cell-export-admission.v1",
        "unit": UNIT,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "FRESH_GATES_PASSED_SUBMISSION_PENDING",
        "native_unit": NATIVE_UNIT,
        "native_invocation_id": NATIVE_INVOCATION,
        "binding_sha256": EXPORT_BINDING_SHA256,
        "exporter_sha256": EXPORTER_SHA256,
        "source_manifest_sha256": EXPORT_SOURCE_MANIFEST_SHA256,
        "gates": gates,
        "staged_files": staged,
        "resource_limits": {
            "cpu_quota": "100%",
            "memory_high_bytes": WORKER_MEMORY_HIGH_BYTES,
            "memory_max_bytes": WORKER_MEMORY_MAX_BYTES,
            "memory_swap_max_bytes": 0,
            "runtime_seconds": WORKER_RUNTIME_SECONDS,
            "gpu_allocator_bytes": WORKER_GPU_ALLOCATOR_BYTES,
            "selected_gpu_uuid": GPU_UUID,
            "blas_threads": 1,
            "loader_workers": 0,
            "tf32_override": 0,
            "cublas_workspace_config": ":4096:8",
            "temporary_directory": str(SCRATCH_ROOT),
        },
        "scope": "One fixed five-arm FP32 feature-only F0/seed-1001 epoch-5 prediction export; no scoring, metric computation, checkpoint selection, training, inference outside the declared export, or quality claim.",
        "unknown_outcome_policy": "Never retry or recreate the evidence directory or unit. Inspect only the exact recorded unit and outputs.",
    }
    write_json_exclusive(EVIDENCE_ROOT / "ADMISSION.json", admission)
    write_exclusive(EVIDENCE_ROOT / "unit.log", b"")
    directory_fd = os.open(EVIDENCE_ROOT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    launch, _submission = submit_once(admission)
    admission_raw, _ = stable_file(EVIDENCE_ROOT / "ADMISSION.json", 1_000_000)
    launch_raw, _ = stable_file(EVIDENCE_ROOT / "LAUNCH.json", 1_000_000)
    require(
        len(admission_raw) + len(launch_raw) <= 4_000_000, "bounded admission response exceeds 4 MB"
    )
    durable_admission = parse_json(admission_raw, "durable export admission")
    durable_launch = parse_json(launch_raw, "durable export launch")
    require(
        durable_admission.get("unit") == UNIT
        and durable_launch.get("unit") == UNIT
        and durable_launch.get("invocation_id") == launch.get("invocation_id")
        and _is_invocation_id(durable_launch.get("invocation_id"))
        and durable_launch.get("submission", {}).get("outcome") == "submitted"
        and durable_launch.get("submission", {}).get("returncode") == 0,
        "durable export receipts do not bind one successful submission",
    )
    print(
        json.dumps(
            {
                "unit": UNIT,
                "outcome": "submitted",
                "admission_base64": base64.b64encode(admission_raw).decode("ascii"),
                "launch_base64": base64.b64encode(launch_raw).decode("ascii"),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
