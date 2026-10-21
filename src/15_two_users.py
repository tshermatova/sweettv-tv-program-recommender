"""Section 7: pick two users, show every method's Top-5 side by side with titles."""
import sys, numpy as np
sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, features, methods, rerank
from collections import Counter

def show_label(s):
    m=lib.SHOW_META.get(int(s))
    if not m: return f"{s} (no metadata)"
    return f"{m[0][:52]:<52} | {m[1]}"

def pick_users(hm, GT):
    """One heavy viewer and one with a thin usable history — both must be test users."""
    cnt=Counter()
    for u in lib.UI[hm]: cnt[int(u)]+=1
    cand=[(cnt.get(u,0),u) for u in lib.TEST_UI if GT.get(u)]
    cand.sort()
    light=[u for n,u in cand if n>0][:1]
    heavy=[u for n,u in cand][-1:]
    return heavy[0], light[0], cnt

if __name__=="__main__":
    hm=features.history_arrays(lib.B_AB)
    GT=lib.target(lo=lib.B_AB,hi=lib.B_TEST)
    heavy,light,cnt=pick_users(hm,GT)
    for tag,u in [("HEAVY",heavy),("LIGHT",light)]:
        print(f"\n{tag}  user_idx={u}  user_id={lib.USERS[u]}  events={cnt.get(u,0)}  |relevant in hold-out|={len(GT.get(u,()))}")
