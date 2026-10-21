import sys, time, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features
GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)
hm=features.history_arrays(lib.B_AB)
u=lib.UI[hm].astype(np.int64); s=lib.SIDX[hm].astype(np.int64); t=lib.PST[hm]
t_end=int(t.max())
pop=np.bincount(s,minlength=lib.N_ITEMS)
FB=[int(lib.ITEMS_ALL[k]) for k in np.argsort(-pop)[:5]]
air,_,_=features.airing_stats(lib.B_AB,lib.B_TEST)

def run(halflife_days, mult_air=True, thr=0.20):
    m=lib.FRAC>=thr; m&=lib.PST<lib.B_AB
    uu=lib.UI[m].astype(np.int64); ss=lib.SIDX[m].astype(np.int64); tt=lib.PST[m]
    w=np.ones(len(uu)) if halflife_days is None else 0.5**(((t_end-tt)/86400.0)/halflife_days)
    key=uu*lib.N_ITEMS+ss
    uk,inv=np.unique(key,return_inverse=True)
    sc=np.bincount(inv,weights=w)
    CU=(uk//lib.N_ITEMS); CI=(uk%lib.N_ITEMS)
    val=sc*(np.log1p(air[CI]) if mult_air else 1.0)
    keep=np.isin(CU,lib.TEST_UI)
    return lib.evaluate(features.top5(val[keep],CU[keep],CI[keep],FB),GT)

print(f"{'half-life':>10} {'x log(air)':>11} {'thr':>5} {'MAP@5':>8}")
for hl in [None,120,60,30,21,14,10,7,5,3]:
    print(f"{str(hl):>10} {'yes':>11} {0.20:>5} {run(hl):>8.5f}", flush=True)
print()
for hl in [None,21,14,7]:
    print(f"{str(hl):>10} {'no':>11} {0.20:>5} {run(hl,mult_air=False):>8.5f}", flush=True)
print()
for thr in [0.1,0.3,0.5,0.7]:
    print(f"{'14':>10} {'yes':>11} {thr:>5} {run(14,thr=thr):>8.5f}", flush=True)
