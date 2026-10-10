"""Check the Takens maximum-likelihood estimator of D2 and its standard error
on (a) uniform points in the unit square (D2 = 2) and (b) Henon map (D2 ~ 1.22)."""
import numpy as np
from scipy.spatial import cKDTree
rng=np.random.default_rng(3)
def takens(X, eps0, w=0):
    tree=cKDTree(X); pairs=tree.query_pairs(eps0, output_type='ndarray')
    if w: pairs=pairs[np.abs(pairs[:,0]-pairs[:,1])>w]
    r=np.linalg.norm(X[pairs[:,0]]-X[pairs[:,1]],axis=1)
    r=r[r>0]; M=len(r)
    D=M/np.sum(np.log(eps0/r))
    return D, D/np.sqrt(M), M
# (a) uniform square
ests=[]
for t in range(200):
    X=rng.random((2000,2)); D,se,M=takens(X,0.05); ests.append(D)
print(f"uniform square: mean D={np.mean(ests):.4f}, empirical sd={np.std(ests):.4f}, formula se={se:.4f} (M={M})")
# (b) Henon
def henon(n,a=1.4,b=0.3):
    x,y=0.1,0.1; out=np.empty((n,2))
    for i in range(n+1000):
        x,y=a-x*x+b*y,x
        if i>=1000: out[i-1000]=(x,y)
    return out
X=henon(20000)
for eps0 in [0.02,0.05,0.1]:
    D,se,M=takens(X,eps0,w=10); print(f"Henon eps0={eps0}: D2={D:.3f} +- {se:.3f}, pairs M={M}")
# compare with local slope of correlation sum
tree=cKDTree(X); 
eps=np.logspace(-2.3,-0.7,12); C=np.array([tree.count_neighbors(tree,e)-len(X) for e in eps])/ (len(X)*(len(X)-1))
sl=np.diff(np.log(C))/np.diff(np.log(eps)); print("local slopes of log C:", np.round(sl,3))
