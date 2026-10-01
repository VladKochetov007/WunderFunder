"""Root-authorized existing prediction byte comparison; no inference or score."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1<<20):h.update(b)
 return h.hexdigest()
def main():
 out=P/'control-byte-identity.json';assert not out.exists();records=[]
 for fold in [0,1]:
  names=[f'pred_on_plectl_s3_f{fold}_valid.npy',f'pred_on_ctl128s3_f{fold}_valid.npy'];paths=[P.parent/'cache'/n for n in names];sizes=[p.stat().st_size for p in paths];assert sizes[0]==sizes[1]
  manifests=[json.loads(p.with_suffix('.inputs.json').read_text()) for p in paths];hashes=[sha(p) for p in paths];assert all(m['prediction_sha256']==h for m,h in zip(manifests,hashes))
  equal=True;differing_chunks=0;first_difference=None;offset=0
  with paths[0].open('rb') as left,paths[1].open('rb') as right:
   while a:=left.read(1<<20):
    b=right.read(len(a));assert len(a)==len(b)
    if a!=b:
     equal=False;differing_chunks+=1
     if first_difference is None:first_difference=offset+next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
    offset+=len(a)
   assert not right.read(1)
  records.append({'fold':fold,'new_path':str(paths[0].relative_to(ROOT)),'old_path':str(paths[1].relative_to(ROOT)),'bytes_each':sizes[0],'sha256':hashes,'complete_file_byte_equal_including_NPY_header_and_NaNs':equal,'differing_1MiB_chunks':differing_chunks,'first_different_file_byte':first_difference,'manifests_sha256':[sha(p.with_suffix('.inputs.json')) for p in paths]})
 result={'status':'COMPLETED_BYTE_ONLY_COMPARISON','folds':records,'both_full_files_byte_equal':all(r['complete_file_byte_equal_including_NPY_header_and_NaNs'] for r in records),'script_sha256':sha(Path(__file__)),'prediction_values_decoded':False,'targets_or_masks_read':False,'inference_executed':False,'WP_queries':0}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
