from pathlib import Path
import ast,copy,hashlib,json,os
ROOT=Path('/Users/nb/Documents/wunder/WunderFunder-maintenance')
C=ROOT/'research/campaign_2026-10-19/full_gru_tweaks_r1'
D=C/'records/first_cell_evaluation_v1/score_v4/exact_stop'
P=C/'preparation/first_cell_evaluation_score_stop_v4'
sha=lambda b:hashlib.sha256(b).hexdigest()
def pin(p):
    b=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':sha(b)}
def write(p,b):
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
source=(P/'ROOT_CAPTURE_SCORE_STOP_V4.py').read_bytes()
assert (len(source),sha(source))==(12159,'848e1573059dd2c3e829166fed018e750f6d8a312d0c6497262967ec0d96d160')
body=(D/'REMOTE.py').read_bytes();review=(D/'SOURCE_REVIEW.json').read_bytes();acceptance=(D.parent/'SCORE_ACCEPTANCE.json').read_bytes()
assert (len(body),sha(body))==(40291,'8304472c154f9a0f11066a61d8de97c8aa8d093b74ff1ba654dd5079661c553a')
assert (len(review),sha(review))==(6046,'27324ec08a9d170847b9d20015ed848ec91eb253e15430c63c838b00aaad9955')
r=json.loads(review);assert r['schema']=='fg19.first-cell-score-exact-stop-source-review.v1' and r['status']=='PASS_SOURCE_ONLY_ACTUAL_SCORE_STOP_V1'
assert (len(acceptance),sha(acceptance))==(21229,'6e9a8162321b2490d5131039e4fe9cf9aded13e58b789d5789491558861c0ed8')
values={'BODY_PIN':(len(body),sha(body)),'REVIEW_PIN':(len(review),sha(review)),'ACCEPTANCE_PIN':(len(acceptance),sha(acceptance)),'SCORE_INVOCATION':'9bd64d6759e8462bb40985b06a9a0ad6'}
tree=ast.parse(source);lines=source.decode().splitlines(keepends=True);old={};replacements=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in values:
        key=node.targets[0].id;old[key]=node.value;replacements.append((node.lineno-1,node.end_lineno,f'{key} = {values[key]!r}\n'))
assert set(old)==set(values)
for start,end,value in reversed(replacements):lines[start:end]=[value]
bound=''.join(lines).encode();check=ast.parse(bound)
for node in check.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in old:node.value=copy.deepcopy(old[node.targets[0].id])
assert ast.dump(check,include_attributes=False)==ast.dump(tree,include_attributes=False)
write(D/'ROOT_CAPTURE_STOP.py',bound);write(D/'BIND_ACTUAL_TRANSPORT.py',Path(__file__).read_bytes())
proof={'schema':'fg19.first-cell-score-stop-root-transport-actual-binding.v1','status':'EXACT_FOUR_LITERAL_BINDING_AST_OTHERWISE_IDENTICAL_NO_EXECUTION','template':pin(P/'ROOT_CAPTURE_SCORE_STOP_V4.py'),'conditional_source_review':pin(P/'SCORE_STOP_SOURCE_REVIEW_V4.json'),'actual_driver':pin(D/'ROOT_CAPTURE_STOP.py'),'actual_body':pin(D/'REMOTE.py'),'actual_body_source_review':pin(D/'SOURCE_REVIEW.json'),'acceptance':pin(D.parent/'SCORE_ACCEPTANCE.json'),'renderer':pin(D/'BIND_ACTUAL_TRANSPORT.py'),'literal_values':values,'all_other_AST_identical':True,'remote_action_performed':False}
write(D/'ROOT_TRANSPORT_AST_BINDING.json',(json.dumps(proof,indent=2)+'\n').encode())
fd=os.open(D,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(proof))
