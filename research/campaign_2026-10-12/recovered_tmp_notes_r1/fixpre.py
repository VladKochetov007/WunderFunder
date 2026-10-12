import json,sys
p=sys.argv[1];d=json.load(open(p))
for s in d['steps']:
    a=s['argv']
    if '--pretrain-next-in-budget' in a and '--checkpoint-epoch' in a:
        a[a.index('--checkpoint-epoch')+1]='1'
    if 'predict_run.py' in ' '.join(a) and '--epoch' in a and any(x in p for x in ('pre1',)):
        a[a.index('--epoch')+1]='1'
json.dump(d,open(p,'w'),indent=1);print('fixed',p)
