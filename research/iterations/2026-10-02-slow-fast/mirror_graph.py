"""Diagnostic 10 gate: does the class-3 price-code transition graph admit a mirror automorphism
(move (da,db) -> (-db,-da))? Component-level check + VF2 on components."""
import pickle,numpy as np,collections,networkx as nx,time
from networkx.algorithms.isomorphism import DiGraphMatcher
moves,spreads=pickle.load(open("/home/grinchenkoeg/WunderFunder-creative/cache/class3_moves_full.pkl","rb"))
G=nx.DiGraph();Gm=nx.DiGraph()
for (a,b),(da,db) in moves.items():
    a,b=tuple(a),tuple(b)
    if a==b:continue
    G.add_edge(a,b,m=(da,db));Gm.add_edge(a,b,m=(-db,-da))
comps=[G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
compsm=[Gm.subgraph(c).copy() for c in nx.weakly_connected_components(Gm)]
sig=lambda g:(g.number_of_nodes(),g.number_of_edges(),tuple(sorted(collections.Counter(d["m"] for *_,d in g.edges(data=True)).items())))
print("nodes",G.number_of_nodes(),"edges",G.number_of_edges(),"components",len(comps),"sizes",sorted([c.number_of_nodes() for c in comps],reverse=True)[:10])
S=collections.Counter(sig(c) for c in comps);Sm=collections.Counter(sig(c) for c in compsm)
print("component signatures matched under mirror:",sum((S&Sm).values()),"of",len(comps))
big=max(comps,key=lambda c:c.number_of_nodes());bigm=Gm.subgraph(big.nodes()).copy()
print("largest comp",sig(big)[:2],"mirror label counts equal:",sig(big)[2]==sig(bigm)[2])
t=time.time();gm=DiGraphMatcher(big,bigm,edge_match=lambda x,y:x["m"]==y["m"])
found=None
for i,mp in enumerate(gm.isomorphisms_iter()):
    if any(k!=v for k,v in mp.items()):found=mp;break
    if time.time()-t>600:break
print("nontrivial mirror automorphism of largest component:",found is not None,"time %.0fs"%(time.time()-t))
if found:print("moved nodes",sum(k!=v for k,v in found.items()),"of",len(found))
else:
    t=time.time();ok=DiGraphMatcher(big,bigm,edge_match=lambda x,y:x["m"]==y["m"]).is_isomorphic();print("is_isomorphic",ok,"%.0fs"%(time.time()-t))
