import importlib, json, os, pathlib, sys, time, numpy as np, torch, pyarrow.parquet as pq

root = pathlib.Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/support-source/eda19-r9-v1"
)
manifest = json.loads((root / "idea_screens_r9/PREPARATION_MANIFEST.json").read_bytes())
old_load, old_reader, old_torch = np.load, pq.ParquetFile, torch.load
calls = []


def deny(*args, **kwargs):
    calls.append(True)
    raise AssertionError("import attempted array/parquet/checkpoint read")


np.load = pq.ParquetFile = torch.load = deny
started = time.monotonic()
cli = importlib.import_module("idea_screens_r9.run_probe")
inventory = cli.smoke_imports(
    root, root.parent.parent / "source-a266b036", manifest["files"]
)
assert not calls and not torch.cuda.is_initialized()
np.load, pq.ParquetFile, torch.load = old_load, old_reader, old_torch
with (root / "ROOT_IMPORT_SMOKE.json").open("x") as f:
    json.dump(
        dict(
            status="PASS",
            data_reads=0,
            runtime_import_inventory=inventory,
            numpy=np.__version__,
            torch=str(torch.__version__),
            cuda_initialized=False,
            elapsed_s=time.monotonic() - started,
        ),
        f,
        indent=2,
    )
args = [
    "--sample",
    "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-sample-r3-v2",
    "--checkpoint",
    "/home/grinchenkoeg/computer-vision-homework-20261018/checkpoint_snapshots/rr18_r01_wide_f0_s1001_e6/epoch5_terminal/model_ep5.pt",
    "--output",
    "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-r9-v1",
    "--expected-sample-sha256",
    "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64",
    "--expected-checkpoint-sha256",
    "78c1413de8eeaa51f803cf59d6b0d7d2eddaad74cc91f744b8b2f85466b61b7f",
    "--expected-preparation-manifest-sha256",
    "__R9_MANIFEST_SHA__",
]
os.execv(sys.executable, [sys.executable, "-B", "-m", "idea_screens_r9.run_probe", *args])
