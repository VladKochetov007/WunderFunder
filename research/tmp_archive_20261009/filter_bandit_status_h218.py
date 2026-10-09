import json,sys
s=json.load(sys.stdin)
print('admission_closed',s.get('admission_closed'))
print('parent_properties',json.dumps(s.get('parent_properties',{}),sort_keys=True))
active=[]
for a in s.get('attempts',[]):
    if a.get('attempt_active_state')=='active' and a.get('phase') not in ('terminal','reconciled'):
        active.append({k:a.get(k) for k in ('id','phase','kind','unit','reservation','allocator_admission_bytes','started_utc')})
print('active_nonterminal_attempts',json.dumps(active,sort_keys=True))
print('active_gpu_attempts',sum(x.get('kind')=='gpu' for x in active))
print('attempt_count',len(s.get('attempts',[])))
