#!/bin/bash
set -u
cd -- "$(dirname -- "$0")"
export WNN_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export WNN_LOADER_WORKERS=0 WNN_GPU_DUTY=1 WNN_GPU_MEMORY_BYTES=18000000000
export WNN_WORKER_RAM_BYTES=28000000000 WNN_WORKER_VRAM_BYTES=19000000000
mkdir -p "$WNN_OUTPUT_DIR"
timeout --signal=TERM --kill-after=10s 450s python -m pip install --disable-pip-version-check numpy==2.2.6 pyarrow==19.0.1 psutil==7.0.0 > "$WNN_OUTPUT_DIR/dependencies.log" 2>&1
setup_exit=$?
if [ "$setup_exit" -eq 0 ]; then
  python worker.py > "$WNN_OUTPUT_DIR/worker.log" 2>&1
  worker_exit=$?
else
  worker_exit=$setup_exit
  cp "$WNN_OUTPUT_DIR/dependencies.log" "$WNN_OUTPUT_DIR/worker.log"
fi
export WNN_WORKER_EXIT="$worker_exit"
python - <<'END'
import os,json,tarfile,hashlib
from pathlib import Path
out=Path(os.environ['WNN_OUTPUT_DIR'])
if not (out/'result.json').exists():
 (out/'result.json').write_text(json.dumps({'status':'failed','exit_code':int(os.environ['WNN_WORKER_EXIT']),'phase':'setup_or_worker_crash'})+'\n')
result=json.loads((out/'result.json').read_text())
result['exit_code']=int(os.environ['WNN_WORKER_EXIT'])
result['worker_success']=result['exit_code']==0 and result.get('status')=='delivered_ungraded'
(out/'result.tmp').write_text(json.dumps(result,indent=2)+'\n')
(out/'result.tmp').replace(out/'result.json')
if (out/'artifacts.tar').exists():
 (out/'artifacts.tar').rename(out/'artifact_bundle.tar')
else:
 with tarfile.open(out/'artifact_bundle.tar','w') as archive:
  for path in sorted(out.iterdir()):
   if path.is_file() and path.name not in ('artifact_bundle.tar','artifact_manifest.json'):
    archive.add(path,arcname=path.name,recursive=False)
records=[]
for name in json.loads(os.environ['WNN_EXPECTED_ARTIFACTS']):
 path=out/name;h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 records.append({'path':name,'size_bytes':path.stat().st_size,'sha256':h.hexdigest()})
manifest={'attempt_id':os.environ['WNN_ATTEMPT_ID'],'packet_sha256':os.environ['WNN_PACKET_SHA256'],'worker_terminal':True,'files':records}
(out/'artifact_manifest.tmp').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'artifact_manifest.tmp').replace(out/'artifact_manifest.json')
END
manifest_exit=$?
if [ "$manifest_exit" -ne 0 ]; then exit "$manifest_exit"; fi
exit "$worker_exit"
