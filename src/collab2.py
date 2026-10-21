"""Extended collaborative signal block: several parameterisations per family."""
import sys, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, methods

KEYS=["assoc_conf","assoc_lift","cos","ppr85","ppr50","mf64","mf16","cov"]

def block(hist_mask, seed=lib.SEED):
    u=lib.UI[hist_mask].astype(np.int64); s=lib.SIDX[hist_mask].astype(np.int64)
    uu=np.unique(u); ii=np.unique(s)
    ru={int(a):k for k,a in enumerate(uu)}; ci={int(a):k for k,a in enumerate(ii)}
    R=np.zeros((len(uu),len(ii)),dtype=np.float32)
    uk,cn=np.unique(u*lib.N_ITEMS+s,return_counts=True)
    R[[ru[int(x)] for x in uk//lib.N_ITEMS],[ci[int(x)] for x in uk%lib.N_ITEMS]]=cn
    B=(R>0).astype(np.float32)
    co=B.T@B; supp=np.diag(co).copy()
    conf=co/np.maximum(supp[:,None],1.0); conf[supp<5,:]=0.0; np.fill_diagonal(conf,0.0)
    n_u=B.shape[0]
    lift=(co*n_u)/np.maximum(np.outer(supp,supp),1.0); lift[supp<5,:]=0.0; np.fill_diagonal(lift,0.0)
    nrm=np.sqrt(np.maximum(supp,1e-9))
    cos=co/np.outer(nrm,nrm); np.fill_diagonal(cos,0.0)
    out={}
    out["assoc_conf"]=B@conf
    out["assoc_lift"]=B@np.log1p(lift)
    out["cos"]=B@cos
    out["cov"]=B@co
    out["ppr85"]=methods.ppr_scores(R,alpha=0.85,iters=20)
    out["ppr50"]=methods.ppr_scores(R,alpha=0.50,iters=20)
    X1,Y1=methods.ials(R,factors=64,reg=100.0,alpha=5.0,iters=8,seed=seed); out["mf64"]=X1@Y1.T
    X2,Y2=methods.ials(R,factors=16,reg=100.0,alpha=5.0,iters=8,seed=seed); out["mf16"]=X2@Y2.T
    items=[int(lib.ITEMS_ALL[k]) for k in sorted(ci,key=ci.get)]
    F,_=methods.item_features(items)
    out["cbf"]=methods.cbf_scores(R,F); 
    return out, ru, ci, R

ALL=KEYS+["cbf"]

def attach(X, CU, CI, blk, start):
    out,ru,ci,_=blk
    r=np.array([ru.get(int(a),-1) for a in CU]); c=np.array([ci.get(int(b),-1) for b in CI])
    ok=(r>=0)&(c>=0)
    cols=[]
    for k in ALL:
        v=np.zeros(len(CU),np.float32); v[ok]=out[k][r[ok],c[ok]]
        cols.append(v)
    return np.hstack([X[:,:start], np.column_stack(cols).astype(np.float32)])
