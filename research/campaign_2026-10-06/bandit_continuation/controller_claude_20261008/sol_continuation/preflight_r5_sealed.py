from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main() -> None:
    archive_path = OUT / "r5_capacity_packet_v2.tar"
    with tempfile.TemporaryDirectory(prefix="r5-sealed-") as temporary:
        root = Path(temporary)
        packet = root / "packet"
        packet.mkdir()
        with tarfile.open(archive_path) as archive:
            archive.extractall(packet, filter="data")
        sys.path.insert(0, str(packet))
        import capacity_worker as runtime
        import capacity_fit_worker
        import worker

        worker.PACKET = packet
        spec = json.loads((packet / "worker_spec.json").read_text())
        for name, digest in spec["packet_files"].items():
            assert runtime.sha256_file(packet / name) == digest
        producer = capacity_fit_worker._verify_profile_evidence(packet)
        assert set(producer["profiles"]) == {"muon512", "muon768"}
        try:
            worker.verify_authority()
        except PermissionError as error:
            assert str(error) == "r5 root admission pending"
        else:
            raise AssertionError("pending authority accepted")
        project = root / "project"
        hashes = runtime._safe_extract_source(packet, project)
        sys.path.insert(0, str(project / "src"))
        sys.path.insert(0, str(project / "scripts"))
        os.environ["WNN_PROJECT_ROOT"] = str(project)
        trainer_source = project / "scripts/train_rnn.py"
        loader = importlib.util.spec_from_file_location("sealed_trainer", trainer_source)
        trainer = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(trainer)
        main_node = next(
            node
            for node in ast.parse(trainer_source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        stop = next(
            i
            for i, node in enumerate(main_node.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets)
        )
        main_node.body = main_node.body[: stop + 1] + [ast.Return(ast.Name("args", ast.Load()))]
        tree = ast.fix_missing_locations(ast.Module(body=[main_node], type_ignores=[]))
        exec(compile(tree, str(trainer_source), "exec"), trainer.__dict__)
        parsed = []
        for hidden in (512, 768):
            arm = {
                "id": f"muon{hidden}",
                "hidden": hidden,
                "optimizer": "muon",
                "run_name": f"r5_capacity_muon{hidden}_seed9_f0",
            }
            argv = runtime._fit_argv(arm, 8, Path("fold0.json"))
            sys.argv = [str(trainer_source)] + argv
            args = trainer.main()
            assert (
                args.hidden == hidden
                and args.epochs == 4
                and args.checkpoint_epoch == 3
                and args.seed == 9
                and args.optim == "muon"
                and args.profile_steps == 0
            )
            parsed.append({"hidden": hidden, "argv": argv, "passed": True})
        result = {
            "status": "actual_sealed_source_and_producer_preflight_pass",
            "archive_sha256": runtime.sha256_file(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "source_members_verified": len(hashes),
            "packet_members_verified": len(spec["packet_files"]),
            "actual_v4_profile_producer": "passed",
            "pending_authority": "rejected before data/GPU work",
            "fit_parsers": parsed,
            "gpu_used": False,
            "limitation": "root must embed final authorization, reseal, and run exact final admission validation before provider create; actual3090/4090 smoke gate runs on worker",
        }
        (OUT / "R5_SEALED_PREFLIGHT_R2.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {"status": result["status"], "source_members": len(hashes), "parsers": len(parsed)}
            )
        )


if __name__ == "__main__":
    main()
