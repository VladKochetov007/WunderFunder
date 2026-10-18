import pathlib,json,hashlib,datetime
p=pathlib.Path('/home/grinchenkoeg/computer-vision-homework-20261018/splits/fold2.json')
s=p.stat();assert p.is_file() and not p.is_symlink() and s.st_size<=2000000
raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h=='56b0e2ecb6bbda1f34569206cbf2c276544c275bac025793ae450488cf5d17a2'
j=json.loads(raw);assert j['fold']==2 and j['n_folds']==4
nodes=[(split,n['row_group'],n['component']) for split,items in j['sequences'].items() for n in items]
cm={(split,g):c for split,g,c in nodes};assert len(cm)==len(nodes)
m={};fits=set()
for section,split in [('fit','train'),('fit','valid'),('evaluation','valid'),('excluded','train'),('excluded','valid'),('excluded','corrupt_train')]:
 a=j[section][split];assert all(type(v)==int for v in a) and len(a)==len(set(a))
 m[section+'.'+split]={'count':len(a),'ordered_json_sha256':hashlib.sha256(json.dumps(a,separators=(',',':')).encode()).hexdigest()}
 if section=='fit':fits|={cm[split,g] for g in a}
evals={cm['valid',g] for g in j['evaluation']['valid']};assert not fits&evals
s2=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(s2.st_ino,s2.st_size,s2.st_mtime_ns)
out={'schema':'quality_gap.f2_split_metadata.v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_METADATA_ONLY','path':str(p),'resolved_path':str(p.resolve()),'bytes':len(raw),'mtime_ns':s.st_mtime_ns,'sha256':h,'fold':2,'n_folds':4,'counts':{'fit_train':len(j['fit']['train']),'fit_valid':len(j['fit']['valid']),'evaluation_valid':len(j['evaluation']['valid']),'evaluation_components':len(evals),'total_components':len(set(cm.values()))},'membership':m,'component_inventory_sha256':hashlib.sha256(json.dumps(nodes,separators=(',',':')).encode()).hexdigest(),'manifest_declared_fit_eval_components_disjoint':True,'source_generation_stable':True,'declared_sources':j['sources'],'limits':['No data, graph or target array payload read or rehash; declared source identities only.','No native loader replay, training, cache generation, allocation or claim check.']}
print(json.dumps(out,indent=2))
