"""Can tv_show_id be recovered for unlabelled airtime via tv_show_title?"""
import csv, collections
D="/Users/tshermatova/Downloads/week7-exam/data/"
title2ids=collections.defaultdict(collections.Counter)
n_lab=n_unlab=0; unlab_titles=collections.Counter(); blank=0
for fn in ["export_arh_11-20-final.csv","export_arh_21-30-final.csv","export_arh_31-42-final.csv"]:
    with open(D+fn, newline="") as f:
        r=csv.reader(f); next(r)
        for row in r:
            t=row[4].strip(); sid=int(row[5])
            if sid: n_lab+=1; title2ids[t][sid]+=1
            else:
                n_unlab+=1
                if not t: blank+=1
                else: unlab_titles[t]+=1
print(f"labelled rows {n_lab:,} | unlabelled {n_unlab:,} (of which blank title: {blank:,})")
print(f"distinct labelled titles: {len(title2ids):,}")
amb=sum(1 for t,c in title2ids.items() if len(c)>1)
print(f"titles mapping to >1 id (ambiguous): {amb:,} ({100*amb/len(title2ids):.1f}%)")
recov=sum(c for t,c in unlab_titles.items() if t in title2ids)
uniq=sum(c for t,c in unlab_titles.items() if t in title2ids and len(title2ids[t])==1)
print(f"unlabelled rows whose title IS a known labelled title: {recov:,} ({100*recov/n_unlab:.1f}% of unlabelled)")
print(f"   ... of those, title maps to exactly one id: {uniq:,} ({100*uniq/n_unlab:.1f}% of unlabelled)")
print(f"potential labelled coverage after recovery: {100*(n_lab+uniq)/(n_lab+n_unlab):.1f}% (was {100*n_lab/(n_lab+n_unlab):.1f}%)")
print("\ntop unlabelled titles that are NOT recoverable:")
for t,c in [(t,c) for t,c in unlab_titles.most_common(40) if t not in title2ids][:10]:
    print(f"   {c:>7,}  {t[:70]}")
