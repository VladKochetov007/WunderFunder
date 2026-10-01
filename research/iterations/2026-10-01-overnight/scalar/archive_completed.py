"""Preserve all completed SCALAR run files; hash opaque bytes only, no scores."""
from pathlib import Path
import datetime,hashlib,json
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def main():
 plan=json.loads((P/'training-plan.seed3.v1.json').read_text());oldpath=P.parent/'completed-fit-archive-44.json';old=json.loads(oldpath.read_text());assert old['run_count']==44
 proof=json.loads((P.parent/'critic/scalar-training-seed3-actual-review.json').read_text());assert proof['status']=='PASS_INDEPENDENT_SCALAR8_FIT4_OOF_INTEGRITY' and proof['fit_count']==8
 runs=sorted(plan['runs']);assert len(runs)==8 and not set(runs)&set(old['runs']);files=[]
 for name in runs:
  folder=ROOT/'artifacts/runs'/name
  for path in sorted(folder.rglob('*')):
   if path.is_file():
    before=path.stat();digest=sha(path);after=path.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns);files.append({'path':str(path.relative_to(ROOT)),'bytes':before.st_size,'sha256':digest})
 for name in runs:
  job=plan['runs'][name];assert next(x['sha256'] for x in files if x['path']==job['endpoint'])==proof['checkpoints'][name]['endpoint_sha256']
 additions={'status':'PASS_OPAQUE_COMPLETED_RUN_ARCHIVE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':runs,'run_count':8,'files':files,'total_bytes':sum(x['bytes'] for x in files),'source_script_sha256':sha(Path(__file__)),'actual_review_sha256':sha(P.parent/'critic/scalar-training-seed3-actual-review.json'),'public_upload':False,'prediction_cache_copied':False,'WP_queries':0};(P/'archive-additions.json').write_text(json.dumps(additions,indent=2)+'\n')
 combined={**old,'created_utc':additions['created_utc'],'scope':'52 completed overnight forecasting fits; all run files. Large prediction caches remain on beleriand.','run_count':52,'runs':sorted(old['runs']+runs),'files':sorted(old['files']+files,key=lambda x:x['path']),'total_bytes':old['total_bytes']+additions['total_bytes'],'prior44_archive_sha256':sha(oldpath),'scalar8_additions_sha256':sha(P/'archive-additions.json'),'scalar_root_allocation_sha256':sha(P/'training-seed3-allocation.json'),'scalar_quality_decision_sha256':sha(P/'seed3-quality-decision.json')};out=P.parent/'completed-fit-archive-52.json';assert not out.exists();out.write_text(json.dumps(combined,indent=2)+'\n');print(json.dumps({'file':str(out.relative_to(ROOT)),'sha256':sha(out),'run_count':52,'total_bytes':combined['total_bytes'],'newfiles':len(files),'newbytes':additions['total_bytes']}))
if __name__=='__main__':main()
