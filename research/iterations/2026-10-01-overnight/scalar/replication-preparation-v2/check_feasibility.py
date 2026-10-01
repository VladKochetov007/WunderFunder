"""Generated-only boundary checks; no actual job/data/quality receipts read."""
from pathlib import Path
import datetime,hashlib,json
from feasibility import calculation
D=Path(__file__).resolve().parent;UTC=datetime.timezone.utc
records=[]
def add(name,seconds):records.append({'label':name,'seconds':seconds,'exit_code':0,'failure':None})
add('inputs-before',20);add('inputs-after',30)
for f in [0,1]:
 for name,seconds in [(f'on_plectl_s3_f{f}_parent',100+f),(f'on_plectl_s3_f{f}',10+f),(f'on_ple_s3_f{f}_parent',200+f),(f'on_ple_s3_f{f}',12+f)]:add(name,seconds);add(name+'-check',2+f)
 add(f'on_plectl_s3_f{f}-OOF',15+f);add(f'on_ple_s3_f{f}-OOF',17+f)
c=.03292332775890827;p=.08629158139228821
r=calculation(records,datetime.datetime(2026,10,1,5,tzinfo=UTC),c,p)
assert r['PASS'] and r['actual_remaining_seconds']==6028.75
assert r['projected_PLE_parent_delay_seconds']==172.5
assert r['projected_screen_seconds']==527.5
late=calculation(records,datetime.datetime(2026,10,1,5,57,7,500000,tzinfo=UTC),0,0)
assert not late['gates']['all_parents_projected_start_before06']
previous=calculation(records,datetime.datetime(2026,10,1,5,57,7,499999,tzinfo=UTC),0,0)
assert previous['gates']['all_parents_projected_start_before06']
failed=[dict(x) for x in records];failed[2]['exit_code']=1
try:calculation(failed,datetime.datetime(2026,10,1,5,tzinfo=UTC),c,p)
except AssertionError:pass
else:raise AssertionError('failed prior job accepted')
result={'status':'PASS_GENERATED_ONLY','numeric_cases':4,'actual_seconds':r['actual_remaining_seconds'],'parent_delay':r['projected_PLE_parent_delay_seconds'],'screen_seconds':r['projected_screen_seconds'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256((D/'feasibility.py').read_bytes()).hexdigest(),'empirical_data_or_receipts_read':False,'WP_queries':0}
(D/'feasibility-generated-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
