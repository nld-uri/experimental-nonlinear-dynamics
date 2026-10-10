import os
import numpy as np, sys
from gsvd import gsvd, smooth_project
from nlutil import embed, pad, unshift, strand_weights
from scipy.special import gamma
rng=np.random.default_rng(1)
print("== 1. rank condition of centred strand for GSVD (d=6)")
d=6
for L in (5,6,7,8):
    S=rng.standard_normal((d,L)); S-=S.mean(1,keepdims=True)
    try:
        gsvd(S.T, np.diff(S,axis=1).T); print(f" L={L} (2l+1): ok, rank centred={np.linalg.matrix_rank(S)}")
    except Exception as e: print(f" L={L}: FAIL rank={np.linalg.matrix_rank(S)}")
print("== 2. curvature correction on unit circle, eps=0.3")
th=np.linspace(0,2*np.pi,20000,endpoint=False); P=np.c_[np.cos(th),np.sin(th)]
from scipy.spatial import cKDTree
T=cKDTree(P); eps=0.3
def com(i): return P[T.query_ball_point(P[i],eps)].mean(0)
i0=0; c0=com(i0); nb=T.query_ball_point(P[i0],eps)
cc=np.mean([com(j) for j in nb],axis=0)
print(" COM radius",np.linalg.norm(c0)," predicted 1-eps^2/6=",1-eps**2/6)
print(" corrected 2c-cc radius",np.linalg.norm(2*c0-cc))
ct=(com(i0-1)+com(i0+1))/2; print(" temporal version radius",np.linalg.norm(2*c0-ct))
print("== 3. SNR arithmetic")
for r in (20,10,5,2.5,1.25): print(f" 1/{r}: {20*np.log10(r):.2f} dB")
TL=(26.0206,33.8305,39.1658,13.9794,22.3897,27.4318); TD=(26.0206,30.7583,36.1687,13.9794,19.5963,24.5692)
for nm,t in (("Lorenz",TL),("Duffing",TD)):
    print(f" {nm}: POD gain {t[1]-t[0]:.2f},{t[4]-t[3]:.2f}; SOD gain {t[2]-t[0]:.2f},{t[5]-t[3]:.2f}")
TB={20:[19.83,26.56,27.88,27.55],40:[14.01,20.59,21.66,21.37]}
print(" bundle gain r1->r4:",TB[20][2]-TB[20][0],TB[40][2]-TB[40][0])
print(" D2 r=1 vs clean 1.939:",2.848/1.939-1, " r=8:",1.839/1.939-1, " 40% r=8:",1.914/1.939-1)
print(" lam r=8:",1-0.818/0.877,1-0.608/0.877)
print("== 4. averaging index formula / unshift on x_i=i with identity filter")
n=400; d=3; m=4; l=5; x=np.sin(2*np.pi*np.arange(1,n+1)/50.7)+0.3*np.sin(2*np.pi*np.arange(1,n+1)/13.1)
y,nrp=embed(x,d,m); assert nrp==n-(d-1)*m
npd=l+(d-1)*m
yp,_=pad(y,nrp,npd); assert yp.shape[1]==n+2*l+(d-1)*m
Yh=yp[:,l:yp.shape[1]-l]           # trimmed as in the codes
xf=unshift(Yh,d,m,nrp,(d-1)*m)
bad=np.nonzero(np.abs(xf-x)>1e-9)[0]+1
print(" n recovered exactly except samples:",bad, " (expected within first/last (d-1)tau =",(d-1)*m,")")
# formula check: padded Yhat(k,j) = x_{j-l-(d-1)tau+(k-1)tau} in interior
ok=True
for k in range(1,d+1):
    for i in range((d-1)*m+1, n-(d-1)*m+1):
        j=i+l+(d-k)*m
        ok &= yp[k-1,j-1]==x[i-1]
print(" Yhat(k, i+l+(d-k)tau)=x_i for interior i:",ok)
print("== 5. code bundle weight scale")
d=6;nn=20000-60
a=np.sqrt(2)*gamma((d+1)/2)*(np.sqrt(d)/nn)**(1/d)/gamma(d/2); print(" a=",a," a^(1/4)=",a**.25)
print("== 6. quantization rms", np.std(rng.uniform(-.5,.5,10**6)), 1/np.sqrt(12))
print("== 7. comb weights reduce to center replacement")
print(strand_weights(3,0), strand_weights(3,1), np.round(strand_weights(3,2),3))
print("== 8. SOD on a strand: smooth_project reproduces a smooth strand exactly when q>=rank")
t=np.linspace(0,1,11); S=np.c_[np.sin(t),np.cos(t),t**2].T  # 3 x 11 (d=3)
S=np.vstack([S,S[:1]*0.5]); S=S-S.mean(1,keepdims=True)     # d=4, rank 3
print(" err q=3:",np.abs(smooth_project(S,3)-S).max())
