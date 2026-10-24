import numpy as np, pickle, time
W = ROOT + "work/"
M = np.load(W+"matches.npy")   # ui, show, overlap, prog_dur, prog_start
ui, sid, ov, pdur, pst = M[:,0], M[:,1], M[:,2], M[:,3], M[:,4]
print("events:", len(M))

print("\n--- programme duration of matched events (sec) ---")
q = [1,5,25,50,75,90,95,99,99.9,100]
print({f"p{p}": int(np.percentile(pdur,p)) for p in q})
for thr in [3600, 7200, 21600, 43200, 86400]:
    m = pdur >= thr
    print(f"  prog_dur >= {thr:>5}s : {m.sum():>9,} events ({100*m.mean():5.2f}%), {len(np.unique(sid[m])):>5} shows")

print("\n--- overlap seconds ---")
print({f"p{p}": int(np.percentile(ov,p)) for p in q})

frac = ov / np.maximum(pdur,1)
print("\n--- overlap / programme duration ---")
print({f"p{p}": round(float(np.percentile(frac,p)),4) for p in q})

print("\n--- candidate 'watched' thresholds (events kept / distinct user-show pairs) ---")
import itertools
pair = ui.astype(np.int64)*10_000_000 + sid
for name, mask in [
    ("no threshold",            np.ones(len(M), bool)),
    ("overlap >= 300s",         ov>=300),
    ("overlap >= 600s",         ov>=600),
    ("frac >= 0.10",            frac>=0.10),
    ("frac >= 0.20",            frac>=0.20),
    ("frac >= 0.30",            frac>=0.30),
    ("frac>=0.2 & dur<=14400",  (frac>=0.20)&(pdur<=14400)),
    ("frac>=0.2 & dur<=21600",  (frac>=0.20)&(pdur<=21600)),
]:
    print(f"  {name:<26} events={mask.sum():>9,}  pairs={len(np.unique(pair[mask])):>9,}  users={len(np.unique(ui[mask])):>5}  shows={len(np.unique(sid[mask])):>5}")
