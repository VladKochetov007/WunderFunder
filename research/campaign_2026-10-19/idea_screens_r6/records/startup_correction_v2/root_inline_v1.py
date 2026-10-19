import importlib,json,pathlib,sys,time,numpy as np,torch,pyarrow.parquet as pq
root=pathlib.Path("/home/grinchenkoeg/computer-vision-homework-20261018/support-source/eda19-r6-v1")
manifest=json.loads((root/"idea_screens_r6/PREPARATION_MANIFEST.json").read_bytes())
sys.path.insert(0,str(root/"idea_screens_r6"))
old_load,old_reader,old_torch=np.load,pq.ParquetFile,torch.load
calls=[]
def deny(*args,**kwargs):
 calls.append(True)
 raise AssertionError("import attempted array/parquet/checkpoint access")
np.load=pq.ParquetFile=torch.load=deny
started=time.monotonic()
names=[name.rsplit("/",1)[1][:-3] for name in manifest["files"] if name.startswith("idea_screens_r6/") and name.endswith(".py") and not name.rsplit("/",1)[1].startswith("test_") and not name.endswith("/__init__.py")]
modules=[importlib.import_module(n) for n in names]
helper=importlib.import_module("sample_panel")._r1_helpers()
modules.append(helper)
assert not calls and not torch.cuda.is_initialized()
np.load,pq.ParquetFile,torch.load=old_load,old_reader,old_torch
with (root/"ROOT_IMPORT_SMOKE.json").open("x") as stream:
 json.dump(dict(status="PASS",data_reads=0,modules={m.__name__:m.__file__ for m in modules},numpy=np.__version__,torch=torch.__version__,cuda_initialized=False,elapsed_s=time.monotonic()-started),stream,indent=2)
sys.argv=["run_probe","--sample","/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-sample-r3-v2","--checkpoint","/home/grinchenkoeg/computer-vision-homework-20261018/checkpoint_snapshots/rr18_r01_wide_f0_s1001_e6/epoch5_terminal/model_ep5.pt","--output","/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/nf19-r6-v1","--expected-sample-sha256","b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64","--expected-checkpoint-sha256","78c1413de8eeaa51f803cf59d6b0d7d2eddaad74cc91f744b8b2f85466b61b7f","--expected-preparation-manifest-sha256","__R6_MANIFEST_SHA__"]
importlib.import_module("run_probe").main()
