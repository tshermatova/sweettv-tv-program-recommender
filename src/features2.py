"""Richer feature set for learning-to-rank."""
import sys, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features

_AR=np.load(lib.W+"airings_raw.npy")
_ARS=np.array([lib.IIDX_ALL.get(int(s),-1) for s in _AR[:,0]],dtype=np.int32)

def air_block(lo,hi):
    i=np.searchsorted(_AR[:,2],lo,"left"); j=np.searchsorted(_AR[:,2],hi,"right")
    s=_ARS[i:j]; ch=_AR[i:j,1]; st=_AR[i:j,2]
    ok=s>=0; s,ch,st=s[ok],ch[ok],st[ok]
    N=lib.N_ITEMS
    cnt=np.bincount(s,minlength=N).astype(np.float32)
    days=np.bincount(np.unique(s.astype(np.int64)*10_000_000+st//86400)//10_000_000,minlength=N).astype(np.float32)
    nch =np.bincount(np.unique(s.astype(np.int64)*1_000_000+ch)//1_000_000,minlength=N).astype(np.float32)
    hour=np.zeros((N,24),dtype=np.float32); np.add.at(hour,(s,((st//3600)%24).astype(np.int64)),1.0)
    dow =np.zeros((N,7),dtype=np.float32);  np.add.at(dow,(s,((st//86400+4)%7).astype(np.int64)),1.0)
    main=np.full(N,-1,dtype=np.int64)
    if len(s):
        k=s.astype(np.int64)*100000+ch
        uk,c=np.unique(k,return_counts=True)
        ks,kc=uk//100000,uk%100000
        o=np.lexsort((-c,ks)); ks,kc=ks[o],kc[o]
        f=np.ones(len(ks),bool); f[1:]=ks[1:]!=ks[:-1]
        main[ks[f]]=kc[f]
    return dict(cnt=cnt,days=days,nch=nch,hour=hour,dow=dow,main=main)

def _norm(M):
    n=np.linalg.norm(M,axis=1,keepdims=True); n[n==0]=1
    return M/n

FEATS2=["own_log","own_days","own_weeks","own_share","own_rank","recency","tenure",
        "trend28","regularity","catch_rate","catch_recent",
        "pop_log","ev_log","pop_recent","pop_rank",
        "air_log","air_days","air_nch","air_first2w","air_last4w","air_tail",
        "ch_aff","ch_rank","hour_aff","dow_aff","hour_aff_ch",
        "assoc","ppr","cbf","mf"]

def build2(hist_mask, air_lo, air_hi, users, n_pop=80, label_mask=None):
    N=lib.N_ITEMS
    u=lib.UI[hist_mask].astype(np.int64); s=lib.SIDX[hist_mask].astype(np.int64)
    t=lib.PST[hist_mask]; d=lib.DAY[hist_mask]; h=lib.HOUR[hist_mask].astype(np.int64)
    t_end=int(t.max()); t_start=int(t.min())
    key=u*N+s
    uk,inv=np.unique(key,return_inverse=True)
    cnt=np.bincount(inv).astype(np.float32)
    last=np.zeros(len(uk),np.int64); np.maximum.at(last,inv,t)
    first=np.full(len(uk),2**62,np.int64); np.minimum.at(first,inv,t)
    ndays=np.bincount(np.unique(inv.astype(np.int64)*10_000_000+d)//10_000_000,minlength=len(uk)).astype(np.float32)
    nweeks=np.bincount(np.unique(inv.astype(np.int64)*10_000_000+d//7)//10_000_000,minlength=len(uk)).astype(np.float32)
    rec28=(t>=t_end-28*86400)
    c28=np.bincount(inv[rec28],minlength=len(uk)).astype(np.float32)
    pu=(uk//N).astype(np.int64); pi=(uk%N).astype(np.int64)
    utot=np.bincount(pu,weights=cnt,minlength=lib.N_USERS).astype(np.float32)
    pop_u=np.bincount(pi,minlength=N).astype(np.float32)
    pop_e=np.bincount(pi,weights=cnt,minlength=N).astype(np.float32)
    pop_r=np.bincount(pi[c28>0],minlength=N).astype(np.float32)
    pop_rank=np.empty(N); pop_rank[np.argsort(-pop_u)]=np.arange(N)

    AT=air_block(air_lo,air_hi)
    A1=air_block(air_lo,air_lo+14*86400)
    A4=air_block(air_hi-28*86400,air_hi)
    AH=air_block(0,air_lo)
    ATAIL=air_block(air_lo-21*86400,air_lo)
    AHR=air_block(t_end-28*86400,air_lo)

    hp=np.zeros((lib.N_USERS,24),np.float32); np.add.at(hp,(u,h),1.0); hp=_norm(hp)
    dp=np.zeros((lib.N_USERS,7),np.float32);  np.add.at(dp,(u,(d+4)%7),1.0); dp=_norm(dp)
    hour_n=_norm(AT["hour"]); dow_n=_norm(AT["dow"])

    # channel affinity, vectorised
    ch_share=np.zeros((lib.N_USERS,),dtype=object)
    top_pop=np.argsort(-pop_u)[:n_pop]
    uu=np.array(sorted(users),dtype=np.int64)
    o=np.argsort(pu,kind="stable"); pus,pis=pu[o],pi[o]
    b0=np.searchsorted(pus,uu); b1=np.searchsorted(pus,uu,"right")
    lut={int(a)*N+int(b):k for k,(a,b) in enumerate(zip(pu,pi))}
    # Optional third candidate source, switched on by CH_CAND=K: every programme airing
    # in the target window on the user's K most-watched channels. Measured on the true
    # target, this lifts the MAP@5 ceiling from 0.7974 to 0.8325 at K=3, against 0.8133
    # at K=1 — the first expansion that moves the ceiling materially, because 17.1% of
    # correct answers are not candidates at all and 175 of them are premieres the other
    # two sources cannot reach.
    import os as _os
    _K = int(_os.environ.get("CH_CAND", 0))
    _by_ch = {}
    if _K:
        import importlib as _il
        _share = _il.import_module("32_chviews").ch_share(air_lo)[0]
        _i = np.searchsorted(_AR[:, 2], air_lo, "left")
        _j = np.searchsorted(_AR[:, 2], air_hi, "right")
        for _s, _c in zip(_AR[_i:_j, 0], _AR[_i:_j, 1]):
            if int(_s) and int(_s) in lib.IIDX_ALL:
                _by_ch.setdefault(int(_c), set()).add(lib.IIDX_ALL[int(_s)])

    CU=[];CI=[]
    for q,user in enumerate(uu):
        own=pis[b0[q]:b1[q]]
        extra=np.setdiff1d(top_pop,own)
        c=np.concatenate([own,extra]) if len(own) else top_pop
        if _K:
            v = _share.get(int(user), {})
            add = set()
            for _ch, _ in sorted(v.items(), key=lambda kv: -kv[1])[:_K]:
                add |= _by_ch.get(int(_ch), set())
            add -= set(c.tolist())
            if add:
                c = np.concatenate([c, np.fromiter(add, np.int64)])
        CU.append(np.full(len(c),user,np.int64)); CI.append(c)
    CU=np.concatenate(CU); CI=np.concatenate(CI)
    k=np.array([lut.get(int(a)*N+int(b),-1) for a,b in zip(CU,CI)])
    has=k>=0; kk=np.clip(k,0,None)
    c_cnt=np.where(has,cnt[kk],0.0); c_days=np.where(has,ndays[kk],0.0)
    c_wk=np.where(has,nweeks[kk],0.0); c_28=np.where(has,c28[kk],0.0)
    c_last=np.where(has,last[kk],0); c_first=np.where(has,first[kk],t_end)
    rec=np.where(has,(t_end-c_last)/86400.0,999.0)
    ten=np.where(has,(t_end-c_first)/86400.0,0.0)
    rk=np.zeros(len(CU),np.float32)
    ordr=np.lexsort((-c_cnt,CU)); _,i0=np.unique(CU[ordr],return_index=True)
    rk[ordr]=np.arange(len(CU))-np.repeat(i0,np.diff(np.append(i0,len(CU))))

    chv_arr=np.zeros(len(CU),np.float32); chrk=np.zeros(len(CU),np.float32)
    haff_ch=np.zeros(len(CU),np.float32)
    for n,(a,b) in enumerate(zip(CU,CI)):
        ch=AT["main"][b]
        v=lib.CH_VIEWS.get(int(a))
        if v and ch>=0:
            tot=sum(v.values()) or 1
            chv_arr[n]=v.get(int(ch),0)/tot
            srt=sorted(v.values(),reverse=True)
            x=v.get(int(ch),0); chrk[n]=1.0/(1.0+(srt.index(x) if x in srt else 99))
    X=np.column_stack([
        np.log1p(c_cnt), np.log1p(c_days), np.log1p(c_wk), c_cnt/np.maximum(utot[CU],1),
        1.0/(1.0+rk), np.log1p(rec), np.log1p(ten),
        c_28/np.maximum(c_cnt,1), c_days/np.maximum(ten,1), 
        c_cnt/np.maximum(AH["cnt"][CI],1), c_28/np.maximum(AHR["cnt"][CI],1),
        np.log1p(pop_u[CI]), np.log1p(pop_e[CI]), np.log1p(pop_r[CI]), 1.0/(1.0+pop_rank[CI]),
        np.log1p(AT["cnt"][CI]), np.log1p(AT["days"][CI]), AT["nch"][CI],
        np.log1p(A1["cnt"][CI]), np.log1p(A4["cnt"][CI]), np.log1p(ATAIL["cnt"][CI]),
        chv_arr, chrk,
        (hp[CU]*hour_n[CI]).sum(1), (dp[CU]*dow_n[CI]).sum(1), haff_ch,
        np.zeros(len(CU),np.float32),np.zeros(len(CU),np.float32),
        np.zeros(len(CU),np.float32),np.zeros(len(CU),np.float32),
    ]).astype(np.float32)
    y=None
    if label_mask is not None:
        lu=lib.UI[label_mask].astype(np.int64); ls=lib.SIDX[label_mask].astype(np.int64)
        luk=np.unique(lu*N+ls); ps=set(luk.tolist())
        y=np.array([1 if (int(a)*N+int(b)) in ps else 0 for a,b in zip(CU,CI)],np.int8)
    return X,CU,CI,y,dict(pop_u=pop_u)

def attach(X,CU,CI,block):
    out,ru,ci,_=block
    r=np.array([ru.get(int(a),-1) for a in CU]); c=np.array([ci.get(int(b),-1) for b in CI])
    ok=(r>=0)&(c>=0)
    for n,k in enumerate(["assoc","ppr","cbf","mf"]):
        v=np.zeros(len(CU),np.float32); v[ok]=out[k][r[ok],c[ok]]
        X[:,26+n]=v
    return X

