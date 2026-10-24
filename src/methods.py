import os
"""The four recommendation methods + popularity baseline. All fit on training data only."""
import numpy as np, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

def matrix(H, items=None):
    """H -> (R dense float32 [n_users_in_H x n_items], row_index, item_list)."""
    if items is None:
        items = sorted({s for c in H.values() for s in c})
    iidx={s:i for i,s in enumerate(items)}
    rows=sorted(H)
    ridx={u:i for i,u in enumerate(rows)}
    R=np.zeros((len(rows),len(items)), dtype=np.float32)
    for u,c in H.items():
        r=ridx[u]
        for s,n in c.items():
            j=iidx.get(s)
            if j is not None: R[r,j]=n
    return R, ridx, items, iidx

# ---------------------------------------------------------------- popularity
def popularity_scores(H, items):
    p=lib.popularity(H)
    return np.array([p.get(s,0) for s in items], dtype=np.float32)

# ------------------------------------------------- 1. matrix factorization
def ials(R, factors=64, reg=0.05, alpha=20.0, iters=15, seed=lib.SEED):
    """Implicit ALS (Hu-Koren-Volinsky). R holds raw counts; confidence = 1 + alpha*log1p(count)."""
    rng=np.random.default_rng(seed)
    n,m=R.shape
    P=(R>0).astype(np.float32)
    C=1.0+alpha*np.log1p(R)
    X=rng.normal(0,0.01,(n,factors)).astype(np.float32)
    Y=rng.normal(0,0.01,(m,factors)).astype(np.float32)
    I=np.eye(factors,dtype=np.float32)*reg
    for it in range(iters):
        YtY=Y.T@Y
        for u in range(n):
            nz=np.flatnonzero(P[u])
            if nz.size==0: X[u]=0; continue
            Yn=Y[nz]; cu=C[u,nz]-1.0
            A=YtY+(Yn*cu[:,None]).T@Yn+I
            b=(Yn*C[u,nz][:,None]).sum(0)
            X[u]=np.linalg.solve(A,b)
        XtX=X.T@X
        for i in range(m):
            nz=np.flatnonzero(P[:,i])
            if nz.size==0: Y[i]=0; continue
            Xn=X[nz]; ci=C[nz,i]-1.0
            A=XtX+(Xn*ci[:,None]).T@Xn+I
            b=(Xn*C[nz,i][:,None]).sum(0)
            Y[i]=np.linalg.solve(A,b)
    return X,Y

# --------------------------------------------------- 2. content-based filter
def item_features(items):
    """TF-IDF over category, 3 genres, director, actors, decade."""
    docs=[]
    for s in items:
        m=lib.SHOW_META.get(s)
        if not m: docs.append([]); continue
        title,cat,g1,g2,g3,year,director,actors=m
        f=[]
        if cat: f.append("cat="+cat)
        for g in (g1,g2,g3):
            if g: f.append("gen="+g.strip())
        if year and year.isdigit(): f.append("dec=%s"%(int(year)//10*10))
        if director:
            for d in director.split(","):
                d=d.strip()
                if d: f.append("dir="+d)
        if actors:
            for a in actors.split(",")[:8]:
                a=a.strip()
                if a: f.append("act="+a)
        docs.append(f)
    vocab={}
    for d in docs:
        for t in set(d): vocab.setdefault(t,len(vocab))
    F=np.zeros((len(items),len(vocab)),dtype=np.float32)
    for i,d in enumerate(docs):
        for t in set(d): F[i,vocab[t]]=1.0
    df=F.sum(0); idf=np.log((len(items)+1)/(df+1))+1.0
    F*=idf
    nrm=np.linalg.norm(F,axis=1,keepdims=True); nrm[nrm==0]=1
    return F/nrm, len(vocab)

def cbf_scores(R, F):
    prof=R@F                                   # user profile = count-weighted sum of item vectors
    nrm=np.linalg.norm(prof,axis=1,keepdims=True); nrm[nrm==0]=1
    return (prof/nrm)@F.T

# ------------------------------------------------------ 3. association rules
def assoc_scores(R, min_support=5):
    """Confidence-weighted co-occurrence: score(u,i) = sum_j 1[u saw j] * conf(j->i)."""
    B=(R>0).astype(np.float32)
    co=B.T@B                                   # items x items co-occurrence over users
    supp=np.diag(co).copy()
    conf=co/np.maximum(supp[:,None],1.0)       # conf(j->i) = co(j,i)/supp(j)
    conf[supp<min_support,:]=0.0
    np.fill_diagonal(conf,0.0)
    return B@conf, conf, supp

# ------------------------------------------------- 4. personalized PageRank
def ppr_scores(R, alpha=0.85, iters=20):
    """PPR on the bipartite user-programme co-view graph, restart on the user's own node."""
    B=(R>0).astype(np.float32)
    du=B.sum(1,keepdims=True); du[du==0]=1
    di=B.sum(0,keepdims=True); di[di==0]=1
    Pui=B/du                                   # user -> item
    Piu=(B/di).T                               # item -> user
    v=np.eye(B.shape[0],dtype=np.float32)      # restart on self
    pu=v.copy(); pi=np.zeros((B.shape[0],B.shape[1]),dtype=np.float32)
    for _ in range(iters):
        pi=alpha*(pu@Pui)
        pu=alpha*(pi@Piu)+(1-alpha)*v
    return pi
