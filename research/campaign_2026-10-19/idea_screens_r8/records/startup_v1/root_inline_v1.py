import importlib, json, os, pathlib, sys, time, numpy as np, torch, pyarrow.parquet as pq

root = pathlib.Path(
    "/home/grinchenkoeg/computer-vision-homework-20261018/support-source/eda19-r8-v1"
)
manifest = json.loads((root / "idea_screens_r8/PREPARATION_MANIFEST.json").read_bytes())
old_load, old_reader, old_torch = np.load, pq.ParquetFile, torch.load
calls = []


def deny(*args, **kwargs):
    calls.append(True)
    raise AssertionError("import attempted array/parquet/checkpoint read")


np.load = pq.ParquetFile = torch.load = deny
started = time.monotonic()
names = [
    n[:-3].replace("/", ".")
    for n in manifest["files"]
    if n.startswith("idea_screens_r8/")
    and n.endswith(".py")
    and not n.rsplit("/", 1)[1].startswith("test_")
    and not n.endswith("/__init__.py")
]
modules = [importlib.import_module(n) for n in names]
# Exercise the full pinned transitive inventory, including reused R7 helpers.
for m in list(modules):
    if hasattr(m, "import_helpers"):
        modules.extend(m.import_helpers())
cli = importlib.import_module("idea_screens_r8.run_probe")
inventory = cli._runtime_imports(
    root, root.parent.parent / "source-a266b036", manifest["files"]
)
assert not calls and not torch.cuda.is_initialized()
np.load, pq.ParquetFile, torch.load = old_load, old_reader, old_torch
with (root / "ROOT_IMPORT_SMOKE.json").open("x") as f:
    json.dump(
        dict(
            status="PASS",
            data_reads=0,
            modules={m.__name__: m.__file__ for m in modules},
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
    "--output",
    "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-r8-v1",
    "--expected-sample-sha256",
    "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64",
    "--expected-preparation-manifest-sha256",
    "__R8_MANIFEST_SHA__",
    "--expected-r7-result-sha256",
    "dd36e82a51672a9b9e2ab694dd92fcd827a2445d607927548200d333d1f4b9e0",
]
os.execv(sys.executable, [sys.executable, "-B", "-m", "idea_screens_r8.run_probe", *args])
