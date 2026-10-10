import numpy as np
from scipy.integrate import solve_ivp
def ami(x, maxlag, nb=32):
    # plug-in AMI with equiprobable bins
    r=np.argsort(np.argsort(x)); q=(r*nb)//len(x)
    out=[]
    for L in range(1,maxlag+1):
        a=q[:-L]; b=q[L:]
        H=np.zeros((nb,nb)); np.add.at(H,(a,b),1); P=H/H.sum()
        px=P.sum(1); py=P.sum(0); nz=P>0
        out.append(np.sum(P[nz]*np.log(P[nz]/(px[:,None]*py[None,:])[nz])))
    return np.array(out)
def firstmin(v):
    for i in range(1,len(v)-1):
        if v[i]<v[i-1] and v[i]<=v[i+1]: return i+1
    return None
def zc(x,maxlag):
    x=x-x.mean(); c=np.array([np.dot(x[:-L],x[L:]) for L in range(1,maxlag+1)])/np.dot(x,x)
    idx=np.where(c<=0)[0]; return idx[0]+1 if len(idx) else None
def lorenz(t,u): return [10*(u[1]-u[0]), u[0]*(28-u[2])-u[1], u[0]*u[1]-8/3*u[2]]
sol=solve_ivp(lorenz,[0,2100],[5,-5,20],t_eval=np.arange(100,2100,0.02),rtol=1e-9,atol=1e-9)
x=sol.y[0]; print("Lorenz dt=0.02: ACF zero crossing lag", zc(x,200), " AMI first min lag", firstmin(ami(x,60)))
def duff(t,u): return [u[1], -0.25*u[1]+u[0]-u[0]**3+0.3*np.cos(t)]
sol=solve_ivp(duff,[0,4100],[-0.67,-0.37],t_eval=np.arange(100,4100,0.1),rtol=1e-9,atol=1e-9)
x=sol.y[0]; print("Duffing dt=0.1: ACF zero crossing lag", zc(x,400), " AMI first min lag", firstmin(ami(x,80)))
# Rossler
def ross(t,u): return [-u[1]-u[2], u[0]+0.2*u[1], 0.2+u[2]*(u[0]-5.7)]
sol=solve_ivp(ross,[0,5100],[1,1,1],t_eval=np.arange(100,5100,0.05),rtol=1e-9,atol=1e-9)
x=sol.y[0]; print("Rossler dt=0.05: ACF zero crossing lag", zc(x,400), " AMI first min lag", firstmin(ami(x,120)))
