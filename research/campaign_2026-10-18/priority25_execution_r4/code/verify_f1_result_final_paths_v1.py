import pathlib,json,hashlib,datetime,os
B=pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018')
receipt=B/'runtime/receipts/RANK01_F1_S1001_RESULT_COLLECTION.json'
output=B/'runtime/receipts/RANK01_F1_S1001_FINAL_PATH_VERIFICATION_V1.json'
assert not os.path.lexists(output)
raw=receipt.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest=='6905144e5534235ff1caa478e5f97d08b792f2621ec9c05bc5efe02e4bf30cd7'
d=json.loads(raw);assert d['status']=='collection_passed'
root=B/'result_snapshots/rr18_r01_f1_s1001_e6';assert d['snapshot_directory']==str(root) and not root.is_symlink()
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
assert actual==set(d['files']) and len(actual)==57
files={};total=0
for rel,item in d['files'].items():
 rp=pathlib.PurePosixPath(rel);assert not rp.is_absolute() and '..' not in rp.parts
 p=root/rel
 for parent in (p,*p.parents):
  assert not parent.is_symlink()
  if parent==root:break
 a=p.stat();assert a.st_size==item['bytes'] and a.st_size<=128*1024*1024
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 z=p.stat();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
 assert h.hexdigest()==item['sha256']
 files[rel]={'final_path':str(p),'bytes':a.st_size,'sha256':h.hexdigest(),'historical_copy_path':item['snapshot']};total+=a.st_size
assert total<256*1024*1024
result={'schema':'priority25.f1_final_path_verification.v1','status':'passed','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'helper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'collection_sha256':digest,'snapshot_directory':str(root),'files':files,'file_count':len(files),'total_bytes':total,'note':'Original collection per-file snapshot fields name pre-promotion copy locations. These create-only final-path bindings verify the actual promoted files. The original receipt remains unchanged.','scope':{'hash_only':True,'no_checkpoint_or_prediction_deserialization':True,'no_grade_or_export_rerun':True}}
with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','file_count':len(files),'total_bytes':total}))
