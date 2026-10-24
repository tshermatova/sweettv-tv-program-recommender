"""Vectorised candidate generation and feature construction."""
import os
import sys, numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

_AR=np.load(lib.W+"airings_raw.npy")      # show, channel, start (sorted by start)
_AR_S=np.array([lib.IIDX_ALL.get(int(s),-1) for s in _AR[:,0]],dtype=np.int32)

def airing_stats(lo,hi):
    i=np.searchsorted(_AR[:,2],lo,"left"); j=np.searchsorted(_AR[:,2],hi,"right")
    s=_AR_S[i:j]; ch=_AR[i:j,1]; st=_AR[i:j,2]
    ok=s>=0; s,ch,st=s[ok],ch[ok],st[ok]
    cnt=np.bincount(s,minlength=lib.N_ITEMS).astype(np.float32)
    hour=np.zeros((lib.N_ITEMS,24),dtype=np.float32)
    np.add.at(hour,(s,((st//3600)%24).astype(np.int64)),1.0)
    main=np.full(lib.N_ITEMS,-1,dtype=np.int64)
    if len(s):
        key=s.astype(np.int64)*100000+ch
        uk,c=np.unique(key,return_counts=True)
        ks=uk//100000; kc=uk%100000
        order=np.lexsort((-c,ks))
        ks,kc=ks[order],kc[order]
        first=np.ones(len(ks),bool); first[1:]=ks[1:]!=ks[:-1]
        main[ks[first]]=kc[first]
    return cnt,hour,main

FEATS=["own_log","own_days","own_weeks","own_share","own_rank","recency",
       "catch_rate","pop_log","ev_log","air_log","air_tail","ch_aff","hour_aff",
       "assoc","ppr","cbf","mf"]

def user_hour_profile(mask):
    P=np.zeros((lib.N_USERS,24),dtype=np.float32)
    np.add.at(P,(lib.UI[mask].astype(np.int64),lib.HOUR[mask].astype(np.int64)),1.0)
    n=P.sum(1,keepdims=True); n[n==0]=1
    return P/n

def history_arrays(hi, thr=None, lo=None):
    thr = lib.WATCH_THR if thr is None else thr
    m=lib.WATCHED>=thr
    m &= lib.PST<hi
    if lo is not None: m &= lib.PST>=lo
    return m

def build(hist_mask, air_lo, air_hi, users, collab=None, cand_pool=None,
          n_pop=60, label_mask=None, min_n=1):
    u=lib.UI[hist_mask].astype(np.int64); s=lib.SIDX[hist_mask].astype(np.int64)
    t=lib.PST[hist_mask]; d=lib.DAY[hist_mask]
    key=u*lib.N_ITEMS+s
    uk,inv=np.unique(key,return_inverse=True)
    cnt=np.bincount(inv).astype(np.float32)
    last=np.zeros(len(uk),dtype=np.int64); np.maximum.at(last,inv,t)
    ndays=np.array([len(x) for x in np.split(np.sort(d[np.argsort(inv,kind="stable")]),
                     np.cumsum(np.bincount(inv))[:-1])],dtype=object) if False else None
    # distinct days / weeks per pair, vectorised
    dk=np.unique(inv.astype(np.int64)*10_000_000+d)
    ndays=np.bincount((dk//10_000_000).astype(np.int64),minlength=len(uk)).astype(np.float32)
    wk=np.unique(inv.astype(np.int64)*10_000_000+(d//7))
    nweeks=np.bincount((wk//10_000_000).astype(np.int64),minlength=len(uk)).astype(np.float32)

    pu=(uk//lib.N_ITEMS).astype(np.int64); pi=(uk%lib.N_ITEMS).astype(np.int64)
    utot=np.bincount(pu,weights=cnt,minlength=lib.N_USERS).astype(np.float32)
    pop_u=np.bincount(pi,minlength=lib.N_ITEMS).astype(np.float32)
    pop_e=np.bincount(pi,weights=cnt,minlength=lib.N_ITEMS).astype(np.float32)
    t_end=int(lib.PST[hist_mask].max())

    air_t,hour_t,main_ch = airing_stats(air_lo,air_hi)
    air_h,_,_            = airing_stats(0,air_lo)                 # airings during history
    air_tail,_,_         = airing_stats(air_lo-21*86400,air_lo)   # still on air at the end?
    hp=user_hour_profile(hist_mask)
    hn=np.linalg.norm(hour_t,axis=1,keepdims=True); hn[hn==0]=1
    hour_t_n=hour_t/hn

    # ---- candidates: own history + global top-N popular ----
    top_pop=np.argsort(-pop_u)[:n_pop]
    uu=np.array(sorted(users),dtype=np.int64)
    own={}
    order=np.argsort(pu,kind="stable")
    pu_s,pi_s=pu[order],pi[order]
    bounds=np.searchsorted(pu_s,uu); bounds_e=np.searchsorted(pu_s,uu,"right")
    pair_lookup={int(a)*lib.N_ITEMS+int(b):k for k,(a,b) in enumerate(zip(pu,pi))}

    rows_u=[]; rows_i=[]
    for q,user in enumerate(uu):
        o=pi_s[bounds[q]:bounds_e[q]]
        extra=np.setdiff1d(top_pop,o,assume_unique=False)
        c=np.concatenate([o,extra]) if len(o) else top_pop
        rows_u.append(np.full(len(c),user,dtype=np.int64)); rows_i.append(c)
    CU=np.concatenate(rows_u); CI=np.concatenate(rows_i)

    k=np.array([pair_lookup.get(int(a)*lib.N_ITEMS+int(b),-1) for a,b in zip(CU,CI)])
    has=k>=0
    c_cnt=np.where(has,cnt[np.clip(k,0,None)],0.0)
    c_days=np.where(has,ndays[np.clip(k,0,None)],0.0)
    c_weeks=np.where(has,nweeks[np.clip(k,0,None)],0.0)
    c_last=np.where(has,last[np.clip(k,0,None)],0)
    rec=np.where(has,(t_end-c_last)/86400.0,999.0)

    rk=np.zeros(len(CU),dtype=np.float32)
    ordr=np.lexsort((-c_cnt,CU))
    pos=np.empty(len(CU),dtype=np.int64); pos[ordr]=np.arange(len(CU))
    start=np.zeros(len(CU),dtype=np.int64)
    _,idx0=np.unique(CU[ordr],return_index=True)
    rank_within=np.arange(len(CU))-np.repeat(idx0,np.diff(np.append(idx0,len(CU))))
    rk[ordr]=rank_within

    X=np.column_stack([
        np.log1p(c_cnt), np.log1p(c_days), np.log1p(c_weeks),
        c_cnt/np.maximum(utot[CU],1), 1.0/(1.0+rk), np.log1p(rec),
        c_cnt/np.maximum(air_h[CI],1),
        np.log1p(pop_u[CI]), np.log1p(pop_e[CI]), np.log1p(air_t[CI]), np.log1p(air_tail[CI]),
        np.zeros(len(CU),np.float32), (hp[CU]*hour_t_n[CI]).sum(1),
        np.zeros(len(CU),np.float32), np.zeros(len(CU),np.float32),
        np.zeros(len(CU),np.float32), np.zeros(len(CU),np.float32),
    ]).astype(np.float32)

    # channel affinity
    ch_aff=np.zeros(len(CU),dtype=np.float32)
    for n,(a,b) in enumerate(zip(CU,CI)):
        ch=main_ch[b]
        if ch>=0:
            v=lib.CH_VIEWS.get(int(a))
            if v:
                tot=sum(v.values()) or 1
                ch_aff[n]=v.get(int(ch),0)/tot
    X[:,11]=ch_aff

    y=None
    if label_mask is not None:
        lu=lib.UI[label_mask].astype(np.int64); ls=lib.SIDX[label_mask].astype(np.int64)
        lk=lu*lib.N_ITEMS+ls
        luk,lc=np.unique(lk,return_counts=True)
        pos_set=set(luk[lc>=min_n].tolist())
        y=np.array([1 if (int(a)*lib.N_ITEMS+int(b)) in pos_set else 0
                    for a,b in zip(CU,CI)],dtype=np.int8)
    return X, CU, CI, y, dict(pop_u=pop_u, air_t=air_t, main_ch=main_ch)

def collab_block(hist_mask, mf_params=(64,100.0,5.0,8)):
    """Fit the four methods on the history window; return a lookup for (user, item)."""
    import methods
    u=lib.UI[hist_mask].astype(np.int64); s=lib.SIDX[hist_mask].astype(np.int64)
    uu=np.unique(u); ii=np.unique(s)
    ru={int(a):k for k,a in enumerate(uu)}; ci={int(a):k for k,a in enumerate(ii)}
    R=np.zeros((len(uu),len(ii)),dtype=np.float32)
    key=u*lib.N_ITEMS+s
    uk,cn=np.unique(key,return_counts=True)
    R[[ru[int(x)] for x in uk//lib.N_ITEMS],[ci[int(x)] for x in uk%lib.N_ITEMS]]=cn
    out={}
    out["assoc"],_,_=methods.assoc_scores(R,min_support=5)
    out["ppr"]=methods.ppr_scores(R,alpha=0.85,iters=20)
    F,_=methods.item_features([int(lib.ITEMS_ALL[k]) for k in ii])
    out["cbf"]=methods.cbf_scores(R,F)
    f,reg,al,it=mf_params
    X,Y=methods.ials(R,factors=f,reg=reg,alpha=al,iters=it)
    out["mf"]=X@Y.T
    return out, ru, ci, R

def attach_collab(X, CU, CI, block):
    out,ru,ci,_=block
    r=np.array([ru.get(int(a),-1) for a in CU]); c=np.array([ci.get(int(b),-1) for b in CI])
    ok=(r>=0)&(c>=0)
    for n,k in enumerate(["assoc","ppr","cbf","mf"]):
        v=np.zeros(len(CU),dtype=np.float32)
        v[ok]=out[k][r[ok],c[ok]]
        X[:,13+n]=v
    return X

def top5(score, CU, CI, fallback):
    order=np.lexsort((-score,CU))
    out={}; cur=None; buf=[]
    for k in order:
        a=int(CU[k])
        if a!=cur:
            if cur is not None: out[cur]=buf
            cur=a; buf=[]
        if len(buf)<5: buf.append(int(lib.ITEMS_ALL[CI[k]]))
    if cur is not None: out[cur]=buf
    for a in lib.TEST_UI:
        o=list(out.get(a,[]))
        for s in fallback:
            if len(o)>=5: break
            if s not in o: o.append(s)
        out[a]=o[:5]
    return out
