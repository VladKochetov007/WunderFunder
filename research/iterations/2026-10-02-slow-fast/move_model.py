import pickle,numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
S="/home/grinchenkoeg/WunderFunder-creative/"
moves,spreads=pickle.load(open(S+"cache/class3_moves_full.pkl","rb"))
K=list(moves.keys());A=np.array([np.r_[np.array(a,float),np.array(b,float)] for a,b in K]);Ylab=[f"{moves[k][0]},{moves[k][1]}" for k in K]
Y=np.array(Ylab);print("pairs",len(K),"label dist",{u:int((Y==u).sum()) for u in np.unique(Y)})
X=np.c_[A,A[:,22:]-A[:,:22]]
for name,mk in (("ET",lambda:ExtraTreesClassifier(500,n_jobs=8,random_state=0)),("kNN1",lambda:KNeighborsClassifier(1))):
    acc=[]
    for tr,te in KFold(5,shuffle=True,random_state=0).split(X):
        m=mk().fit(X[tr],Y[tr]);acc.append((m.predict(X[te])==Y[te]).mean())
    print(name,"5-fold acc %.3f"%np.mean(acc))
# spreads per code predictability
Kc=list(spreads.keys());C=np.array(Kc,float);Sp=np.array([spreads[k] for k in Kc])
acc=[]
for tr,te in KFold(5,shuffle=True,random_state=0).split(C):
    m=ExtraTreesClassifier(500,n_jobs=8,random_state=0).fit(C[tr],Sp[tr]);acc.append((m.predict(C[te])==Sp[te]).mean())
print("spread-per-code 5-fold acc %.3f; spread dist"%np.mean(acc),np.unique(Sp,return_counts=True))
