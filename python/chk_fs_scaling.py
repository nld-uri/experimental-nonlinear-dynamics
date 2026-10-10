"""Farmer-Sidorowich scaling check: one-step local-linear prediction error on the Henon map
versus record length N. Prediction E ~ N^{-(m+1)/D} with m=1 (linear), D~1.26 -> exponent ~ -1.6;
zeroth order (simple averaging, m=0) -> exponent ~ -0.8."""
import numpy as np
from scipy.spatial import cKDTree
def henon(n,a=1.4,b=0.3,x0=0.1,y0=0.1):
    x,y=x0,y0; out=np.empty(n)
    for i in range(n+1000):
        x,y=a-x*x+b*y,x
        if i>=1000: out[i-1000]=x
    return out
rng=np.random.default_rng(1)
def errs(N,r=8,ntest=2000):
    x=henon(N+2); X=np.c_[x[:-2],x[1:-1]]; Y=x[2:]      # forward delay vectors (x_n,x_{n+1}) -> x_{n+2}
    xt=henon(ntest+2,x0=0.3,y0=-0.1); Xt=np.c_[xt[:-2],xt[1:-1]]; Yt=xt[2:]
    tree=cKDTree(X); d,idx=tree.query(Xt,r)
    e0=[];e1=[]
    for q in range(ntest):
        nb=idx[q]; A=np.c_[X[nb],np.ones(r)]; coef=np.linalg.lstsq(A,Y[nb],rcond=None)[0]
        e1.append(np.r_[Xt[q],1]@coef-Yt[q]); e0.append(Y[nb].mean()-Yt[q])
    return np.sqrt(np.mean(np.square(e0))),np.sqrt(np.mean(np.square(e1)))
Ns=[1000,2000,4000,8000,16000,32000]; E=np.array([errs(N) for N in Ns])
for N,(a,b) in zip(Ns,E): print(N, f"avg {a:.2e}  loclin {b:.2e}")
s0=np.polyfit(np.log(Ns),np.log(E[:,0]),1)[0]; s1=np.polyfit(np.log(Ns),np.log(E[:,1]),1)[0]
print(f"fitted exponents: averaging {s0:.2f} (theory -1/D=-0.79), local linear {s1:.2f} (theory -2/D=-1.59)")
