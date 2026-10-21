from pathlib import Path
import ast,base64,copy,hashlib,json,os
from datetime import datetime,UTC
ROOT=Path('/Users/nb/Documents/wunder/WunderFunder-maintenance')
C=ROOT/'research/campaign_2026-10-19/full_gru_tweaks_r1'
R=C/'records/first_cell_evaluation_v1/score_v4'
D=R/'exact_stop'
P=C/'preparation/first_cell_evaluation_score_stop_v4'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def pin(path):
    raw=path.read_bytes();return {'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':sha(raw)}
def write(path,raw):
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def bind(template,values):
    original=ast.parse(template);lines=template.decode().splitlines(keepends=True);changes=[];old={}
    for node in original.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in values:
            key=node.targets[0].id;old[key]=node.value;changes.append((node.lineno-1,node.end_lineno,f'{key} = {values[key]!r}\n'))
    assert set(old)==set(values)
    for start,end,value in reversed(changes):lines[start:end]=[value]
    raw=''.join(lines).encode();normalized=ast.parse(raw)
    for node in normalized.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in old:node.value=copy.deepcopy(old[node.targets[0].id])
    assert ast.dump(normalized,include_attributes=False)==ast.dump(original,include_attributes=False)
    return raw
acceptance_raw=(R/'SCORE_ACCEPTANCE.json').read_bytes();a=json.loads(acceptance_raw)
assert a['schema']=='fg19.first-cell-score-root-acceptance.v1' and a['status']=='ACCEPTED_FIXED_FIRST_CELL_DEVELOPMENT_SCORE_REPORT'
assert a['unit']=='computer-vision-homework-fg19-first-cell-score-v3.service' and a['invocation_id']=='9bd64d6759e8462bb40985b06a9a0ad6'
native=a['native_terminal'];assert (native['active_state'],native['sub_state'])==('active','exited')
assert all(type(native[k]) is int and native[k]==v for k,v in [('main_pid',0),('exec_main_code',1),('exec_main_status',0)])
assert (a['result']['path'],a['result']['bytes'],a['result']['sha256'])==('/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v3/RESULT.json',33852,'1646d8656a042ce49b355f22521134206d601c1dbac3697991fcd308aa1a94d5')
root_binding=json.loads((R/'ROOT_ACCEPTANCE_BINDING.json').read_bytes());assert (root_binding['acceptance']['bytes'],root_binding['acceptance']['sha256'])==(len(acceptance_raw),sha(acceptance_raw))
template=(P/'STOP_SCORE_TEMPLATE_V3.py').read_bytes();assert (len(template),sha(template))==(11984,'34383a33d3d8c4efa1bdf84d2c22769ef55934b59b35b1d3f9a27c08effbb10c')
review=(P/'SCORE_STOP_SOURCE_REVIEW_V4.json').read_bytes();assert (len(review),sha(review))==(5134,'54c678bd248d039fdaa35212943b5a85ab6ed7b3f9f068604b743a76c2bca47f')
values={'SCORE_INVOCATION':a['invocation_id'],'ACCEPTANCE_BYTES':len(acceptance_raw),'ACCEPTANCE_SHA256':sha(acceptance_raw),'ACCEPTANCE_BASE64':base64.b64encode(acceptance_raw).decode()}
body=bind(template,values);D.mkdir(exist_ok=False)
write(D/'REMOTE.py',body)
write(D/'BIND_ACTUAL_STOP.py',Path(__file__).read_bytes())
proof={'schema':'fg19.first-cell-score-stop-actual-four-literal-binding.v1','status':'EXACT_FOUR_LITERAL_BINDING_AST_OTHERWISE_IDENTICAL_NO_REMOTE_ACTION','created_utc':datetime.now(UTC).isoformat(),'template':pin(P/'STOP_SCORE_TEMPLATE_V3.py'),'conditional_source_review':pin(P/'SCORE_STOP_SOURCE_REVIEW_V4.json'),'acceptance':pin(R/'SCORE_ACCEPTANCE.json'),'root_acceptance_binding':pin(R/'ROOT_ACCEPTANCE_BINDING.json'),'actual_body':pin(D/'REMOTE.py'),'renderer':pin(D/'BIND_ACTUAL_STOP.py'),'bound_literal_names':list(values),'actual_invocation':a['invocation_id'],'all_other_AST_identical':True,'result_path_explicitly_bound_to_fixed_export_evidence_root':True,'B_action':False,'failed_score_units_excluded':[1,2]}
write(D/'AST_BINDING.json',(json.dumps(proof,indent=2)+'\n').encode())
fd=os.open(D,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(proof))
