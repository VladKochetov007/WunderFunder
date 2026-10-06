"""Falsifiable prerequisites for decoded mirror and a four-event race model."""
import hashlib
import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import networkx as nx
from wnn.data import SequenceFile
from wnn.ticks import reconstruct
from online_c3 import rebuilt_labels,law
R=Path.home()/'WunderFunder-creative';D=R/'research/iterations/2026-10-05-wave9'
def main():
    moves,spreads=pickle.load(open(R/'cache/class3_moves_full.pkl','rb'))
    g=nx.DiGraph();gm=nx.DiGraph()
    for (a,b),(da,db) in moves.items():
        if a==b:continue
        g.add_edge(a,b,move=(da,db));gm.add_edge(a,b,move=(-db,-da))
    def signature(graph):
        return (graph.number_of_nodes(),graph.number_of_edges(),tuple(sorted(Counter(d['move'] for _,_,d in graph.edges(data=True)).items())))
    comps=[g.subgraph(c) for c in nx.weakly_connected_components(g)]
    mirrors=[gm.subgraph(c) for c in nx.weakly_connected_components(gm)]
    counts=Counter(signature(c) for c in mirrors)
    candidate_nodes=sum(c.number_of_nodes() for c in comps if counts[signature(c)]>0)
    # Exact quote-level reflection implies swapping the two targets, without a sign flip.
    b=np.repeat(np.arange(-10,11),12*81);sp=np.tile(np.repeat(np.arange(1,13),81),21)
    f=b+np.tile(np.arange(-40,41),21*12)
    a=b+sp
    y=law(f,b,a);mir=law(-f,-a,-b)
    assert np.array_equal(mir,y[:,::-1])
    assert np.array_equal(law(f,b,a),law(-(-f),-(-b),-(-a)))
    cf=np.load(R/'cache/chain_folds.npz')
    seen=set(c for s,rg,c,fold in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if (s=='valid' and fold!=0) or (s=='train' and int(hashlib.sha256(f'half:{rg}'.encode()).hexdigest(),16)%2==0))
    cm={rg:(c,fold) for s,rg,c,fold in zip(cf['split'],cf['row_group'],cf['chain'],cf['fold']) if s=='valid'}
    z=np.load(R/'cache/valid_targets.npz');sf=SequenceFile(R/'datasets/valid.parquet')
    ref=np.load(R/'cache/pred_w2_reference_valid.npy',mmap_mode='r')
    from wnn.gmetric import wp_sums,wp_from_sums
    moments=np.zeros((2,2,6));counts_law=Counter();mass=Counter()
    for i,rg in enumerate(z['row_group']):
        chain,fold=cm[rg]
        if fold!=0 or chain in seen or z['market_class'][i]!=3:continue
        seq=sf.read(int(rg));aa,bb,unknown=reconstruct(seq.features,moves,spreads)
        rb,known=rebuilt_labels(aa,bb)
        mask=seq.scored.copy();mask[:99]=False
        ok=known<len(rb);fail=np.any(abs(rb-seq.targets)>1e-3,axis=1)
        w=np.clip(abs(seq.targets),0,2).sum(-1)
        counts_law.update(sequences=1,scored=int(mask.sum()),known=int((mask&ok).sum()),failure=int((mask&ok&fail).sum()),unknown_transitions=int(unknown))
        mass['total']+=float(w[mask].sum());mass['known']+=float(w[mask&ok].sum());mass['failure']+=float(w[mask&ok&fail].sum())
        oracle=np.where(ok[:,None],rb,ref[i])
        for k in range(2):
            moments[0,k]+=wp_sums(seq.targets[:,k],ref[i,:,k],mask)
            moments[1,k]+=wp_sums(seq.targets[:,k],oracle[:,k],mask)
    report={'mirror':{'quote_law_swap_test_cases':len(y),'double_reflection':'PASS','graph_nodes':g.number_of_nodes(),'graph_edges':g.number_of_edges(),'components':len(comps),'nodes_in_signature_compatible_components':candidate_nodes,
        'gate':'NOT PASSED: no verified reversible mapping for all 112 encoded price, volume, trade and auxiliary columns. Quote-law symmetry alone does not authorize feature augmentation. Incomplete observed graph cannot prove impossibility.'},
        'race_prerequisite':{'counts':dict(counts_law),'metric_mass':dict(mass),'class3_reference_wp':float(wp_from_sums(moments[0]).mean()),'future_four_event_law_wp':float(wp_from_sums(moments[1]).mean()),'failure_weight_fraction_known':mass['failure']/mass['known'],
        'caveat':'Diagnostic future reconstruction only. Decoder table learned using labels including validation; results are mechanism audit, never a clean predictor evaluation. Race learner must use causal features and allow misspecification.'}}
    (D/'law-mirror-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
