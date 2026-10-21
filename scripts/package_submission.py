"""Package a fixed native ensemble from independently trained checkpoints.

The command accepts run references such as run-name@ep5:0.5. It writes a
root-level scorer ZIP and a build receipt that binds each exported member to
its source checkpoint and provenance sidecar. Outputs are create-only: an
existing ZIP or receipt is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from wnn.native_representation import native_input_metadata
from wnn.training.dynamics_auxiliary import strip_dynamics_auxiliary_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "artifacts" / "runs"
DEFAULT_OUT_ROOT = ROOT / "submissions"
TensorMap = dict[str, np.ndarray]


def sha256_file(path: Path) -> str:
    """Hash a file without reading the entire artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_manifest(state: TensorMap) -> dict[str, dict[str, object]]:
    """Describe exported float32 tensors with canonical little-endian hashes."""
    manifest: dict[str, dict[str, object]] = {}
    for name, value in sorted(state.items()):
        array = np.ascontiguousarray(value, dtype="<f4")
        manifest[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return manifest


def checkpoint_path(run: str, runs_dir: Path) -> Path:
    """Resolve a run reference to its selected checkpoint."""
    name, separator, epoch = run.partition("@ep")
    if not name:
        raise ValueError(f"empty run name in {run!r}")
    if separator and (not epoch.isdigit() or int(epoch) < 0):
        raise ValueError(f"invalid checkpoint epoch in {run!r}")
    filename = f"model_ep{int(epoch)}.pt" if separator else "model.pt"
    root = runs_dir.resolve(strict=True)
    path = (root / name / filename).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"checkpoint escapes the runs directory: {run!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_checkpoint_member(
    run: str, runs_dir: Path, require_provenance: bool
) -> tuple[TensorMap, dict[str, object]]:
    """Load one checkpoint and bind it to the run's provenance sidecar."""
    import torch

    path = checkpoint_path(run, runs_dir)
    provenance_path = path.parent / "provenance.json"
    if require_provenance and not provenance_path.is_file():
        raise FileNotFoundError(f"required provenance sidecar missing: {provenance_path}")
    provenance_record: dict[str, object] | None = None
    if provenance_path.is_file():
        json.loads(provenance_path.read_text())
        provenance_record = {
            "path": str(provenance_path.resolve()),
            "sha256": sha256_file(provenance_path),
        }

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"{run}: checkpoint arguments are malformed")
    if args.get("readout_activation", "gelu") != "gelu":
        raise ValueError(f"{run}: non-GELU readout is not supported by this native runtime")
    if checkpoint.get("sentinel_preprocessing"):
        raise ValueError(f"{run}: sentinel preprocessing is not supported by this native runtime")
    if (
        args.get("input_lead")
        or args.get("snap_codebook")
        or args.get("privileged_mode", "none") != "none"
    ):
        raise ValueError(f"{run}: teacher-only lookahead inputs are not deployable")

    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{run}: checkpoint has no model state dictionary")
    state = {
        name: np.ascontiguousarray(tensor.detach().cpu().float().numpy())
        for name, tensor in model.items()
    }
    state = strip_dynamics_auxiliary_state(state)
    input_layout = native_input_metadata(args, state, checkpoint.get("n_in"))
    if "head.2.weight" not in state or "head.2.bias" not in state:
        raise ValueError(f"{run}: checkpoint lacks the expected two-layer output head")
    if state["head.2.weight"].shape[0] < 2 or state["head.2.bias"].shape[0] < 2:
        raise ValueError(f"{run}: output head has fewer than two scored targets")
    state["head.2.weight"] = np.ascontiguousarray(state["head.2.weight"][:2])
    state["head.2.bias"] = np.ascontiguousarray(state["head.2.bias"][:2])
    if not all(np.isfinite(value).all() for value in state.values()):
        raise ValueError(f"{run}: checkpoint contains non-finite model tensors")

    record: dict[str, object] = {
        "run_reference": run,
        "checkpoint_path": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": sha256_file(path),
        "provenance": provenance_record,
        "input_layout": input_layout,
        "exported_tensors": tensor_manifest(state),
    }
    return state, record


def load_member(
    run: str, runs_dir: Path, require_provenance: bool
) -> tuple[TensorMap, list[dict[str, object]]]:
    """Load a run reference, including an optional shared-trunk head merge."""
    if "|" not in run:
        state, source = load_checkpoint_member(run, runs_dir, require_provenance)
        return state, [source]

    references = run.split("|")
    if len(references) < 2 or any(not reference for reference in references):
        raise ValueError(f"invalid merged-head run reference: {run!r}")
    loaded = [
        load_checkpoint_member(reference, runs_dir, require_provenance) for reference in references
    ]
    states = [item[0] for item in loaded]
    base = states[0]
    layouts = [item[1].get("input_layout") for item in loaded]
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError("merged-head members must use the same native input layout")
    for reference, state in zip(references[1:], states[1:], strict=True):
        for name, value in base.items():
            if not name.startswith("head.") and not np.array_equal(value, state[name]):
                raise ValueError(f"{reference}: trunk tensor {name} differs from {references[0]}")
    count = len(states)
    merged = {name: value for name, value in base.items() if not name.startswith("head.")}
    merged["head.0.weight"] = np.concatenate([state["head.0.weight"] for state in states], axis=0)
    merged["head.0.bias"] = np.concatenate([state["head.0.bias"] for state in states])
    merged["head.2.weight"] = (
        np.concatenate([state["head.2.weight"] for state in states], axis=1) / count
    )
    merged["head.2.bias"] = sum(state["head.2.bias"] for state in states) / count
    return (
        {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in merged.items()},
        [item[1] for item in loaded],
    )


def export_member(
    run: str, output: Path, runs_dir: Path, require_provenance: bool
) -> dict[str, object]:
    """Export one fixed member to NPZ and return its source and tensor hashes."""
    if run == "baseline":
        if require_provenance:
            raise ValueError("the provenance-required candidate path accepts trained runs only")
        source = ROOT / "submissions" / "_parts" / "baseline_rnnmodel.npz"
        if not source.is_file():
            raise FileNotFoundError(
                "baseline weights are not prepared; run scripts/convert_baseline.py first"
            )
        shutil.copyfile(source, output)
        with np.load(output) as archive:
            arrays = {
                name: np.ascontiguousarray(archive[name], dtype=np.float32)
                for name in archive.files
            }
        sources: list[dict[str, object]] = [
            {
                "source": "baseline",
                "path": str(source.resolve()),
                "sha256": sha256_file(source),
            }
        ]
    else:
        arrays, sources = load_member(run, runs_dir, require_provenance)
        np.savez(output, **arrays)
    input_layout = sources[0].get("input_layout")
    if not isinstance(input_layout, dict):
        input_layout = native_input_metadata({}, arrays, int(arrays["mean"].shape[0]))
    return {
        "run_reference": run,
        "sources": sources,
        "input_layout": input_layout,
        "weights_path_in_zip": output.name,
        "weights_bytes": output.stat().st_size,
        "weights_sha256": sha256_file(output),
        "exported_tensors": tensor_manifest(arrays),
    }


def parse_members(members: list[str], normalized: bool) -> list[tuple[str, float]]:
    """Parse fixed RUN:COEF inputs and validate blend coefficients."""
    result: list[tuple[str, float]] = []
    for member in members:
        run, separator, coefficient_text = member.rpartition(":")
        if not separator or not run:
            raise ValueError(f"member must be RUN:COEF, got {member!r}")
        coefficient = float(coefficient_text)
        if not math.isfinite(coefficient):
            raise ValueError(f"member coefficient must be finite: {member!r}")
        result.append((run, coefficient))
    if not result:
        raise ValueError("at least one member is required")
    if normalized and not math.isclose(
        sum(coefficient for _, coefficient in result), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("normalized fixed blend coefficients must sum to 1")
    return result


def add_package_files(
    package: Path,
    native_from: Path,
    solution: Path,
    members: list[tuple[str, float]],
    precision: str,
    runs_dir: Path,
    require_provenance: bool,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Create package files without modifying any source artifact."""
    shutil.copyfile(solution, package / "solution.py")
    (package / "precision.txt").write_text(precision + "\n")
    member_records: list[dict[str, object]] = []
    spec: list[dict[str, object]] = []
    for index, (run, coefficient) in enumerate(members):
        filename = f"weights_{index}.npz"
        record = export_member(run, package / filename, runs_dir, require_provenance)
        member_records.append(record)
        spec.append(
            {
                "weights": filename,
                "coef": coefficient,
                "source": run,
                "input_layout": record.get(
                    "input_layout",
                    {
                        "schema": "wnn.native_input_layout.v1",
                        "representation_mode": "none",
                        "input_width": None,
                    },
                ),
            }
        )
    (package / "members.json").write_text(json.dumps(spec, indent=1) + "\n")

    library_hashes: dict[str, str] = {}
    for level in ("v2", "avx2", "v3"):
        source = native_from / level
        library = source / "wnn_native.abi3.so"
        if not library.is_file():
            raise FileNotFoundError(f"native {level} library missing: {library}")
        shutil.copytree(source, package / level)
        library_hashes[f"{level}/wnn_native.abi3.so"] = sha256_file(library)
    return member_records, library_hashes


def package_files(package: Path) -> list[Path]:
    """Return regular package files in stable ZIP order."""
    files = sorted(path for path in package.rglob("*") if path.is_file())
    if not (package / "solution.py").is_file():
        raise ValueError("solution.py must be at the ZIP root")
    return files


def write_zip(package: Path, destination: Path, max_bytes: int) -> dict[str, object]:
    """Create a compressed scorer ZIP and validate its CRCs and root layout."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_files(package):
            archive.write(path, path.relative_to(package).as_posix())
    if destination.stat().st_size > max_bytes:
        raise ValueError(
            f"submission ZIP is {destination.stat().st_size} bytes; limit is {max_bytes}"
        )
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("submission ZIP failed its CRC check")
        names = set(archive.namelist())
        if "solution.py" not in names or "members.json" not in names:
            raise ValueError("submission ZIP is missing a required root file")
        required = {
            "v2/wnn_native.abi3.so",
            "avx2/wnn_native.abi3.so",
            "v3/wnn_native.abi3.so",
        }
        if not required <= names:
            raise ValueError("submission ZIP is missing one or more native libraries")
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"unsafe ZIP path: {name}")
        file_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)
        }
    return {
        "path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "members": file_hashes,
    }


def write_json_create_only(path: Path, value: dict[str, object]) -> None:
    """Write a JSON receipt while refusing to replace any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Build a fixed ensemble ZIP and its hash-bound source receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="safe package label without path components")
    parser.add_argument("members", nargs="+", help="fixed RUN:COEF entries")
    parser.add_argument(
        "--native-from",
        type=Path,
        default=ROOT / "submissions" / "ab18_wmse_struct" / "pkg",
    )
    parser.add_argument("--solution", type=Path, default=ROOT / "submission" / "solution_multi.py")
    parser.add_argument(
        "--precision",
        choices=("f32", "f16", "i16", "q16x", "q8vnni"),
        default="i16",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--output-zip", type=Path, help="write one flat ZIP at this exact path")
    parser.add_argument("--receipt", type=Path, help="write the build receipt at this exact path")
    parser.add_argument("--require-provenance", action="store_true")
    parser.add_argument("--normalized-coefficients", action="store_true")
    parser.add_argument("--evidence-note", default="")
    parser.add_argument("--evidence-file", type=Path, action="append", default=None)
    parser.add_argument("--max-zip-bytes", type=int, default=20_000_000)
    args = parser.parse_args()

    if Path(args.name).name != args.name or args.name in ("", ".", ".."):
        raise SystemExit("name must be a single safe path component")
    members = parse_members(args.members, args.normalized_coefficients)
    runs_dir = args.runs_dir.resolve(strict=True)
    native_from = args.native_from.resolve(strict=True)
    solution = args.solution.resolve(strict=True)
    if not solution.is_file():
        raise FileNotFoundError(solution)

    if args.output_zip:
        output_zip = args.output_zip.resolve()
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        if output_zip.exists():
            raise FileExistsError(f"refusing to replace existing ZIP: {output_zip}")
        default_receipt = output_zip.with_name(output_zip.stem + "_build_receipt.json")
        output_dir = None
        staging_parent = output_zip.parent
    else:
        output_dir = args.out_root.resolve() / args.name
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise FileExistsError(f"refusing to replace existing package directory: {output_dir}")
        default_receipt = output_dir / "BUILD_RECEIPT.json"
        staging_parent = output_dir.parent
        output_zip = output_dir / "submission.zip"
    receipt_path = args.receipt.resolve() if args.receipt else default_receipt
    if receipt_path.exists():
        raise FileExistsError(f"refusing to replace existing build receipt: {receipt_path}")

    evidence = []
    for path in args.evidence_file or []:
        resolved = path.resolve(strict=True)
        evidence.append({"path": str(resolved), "sha256": sha256_file(resolved)})

    with tempfile.TemporaryDirectory(
        prefix=f".{args.name}.package-", dir=staging_parent
    ) as temporary:
        stage = Path(temporary)
        package = stage / "pkg"
        package.mkdir()
        member_records, native_hashes = add_package_files(
            package,
            native_from,
            solution,
            members,
            args.precision,
            runs_dir,
            args.require_provenance,
        )
        temporary_zip = stage / "submission.zip"
        zip_record = write_zip(package, temporary_zip, args.max_zip_bytes)
        receipt: dict[str, object] = {
            "schema": "wnn.submission-build-receipt.v1",
            "candidate": args.name,
            "packager": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "fixed_members": [
                {"run_reference": run, "coefficient": coefficient} for run, coefficient in members
            ],
            "precision": args.precision,
            "training_and_selection_scope": args.evidence_note,
            "evidence_files": evidence,
            "public_score_forecast": None,
            "source_code": {
                "path": str(solution),
                "sha256": sha256_file(solution),
            },
            "native_runtime": {
                "source_directory": str(native_from),
                "libraries": native_hashes,
            },
            "members": member_records,
            "artifact": zip_record,
            "gate_status": "package_crc_passed; offline_import_and_streaming_gate_pending",
        }
        if args.output_zip:
            zip_record["path"] = str(output_zip)
            os.link(temporary_zip, output_zip)
            write_json_create_only(receipt_path, receipt)
        else:
            assert output_dir is not None
            output_dir.mkdir()
            os.rename(package, output_dir / "pkg")
            os.link(temporary_zip, output_zip)
            zip_record["path"] = str(output_zip)
            write_json_create_only(receipt_path, receipt)

    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
