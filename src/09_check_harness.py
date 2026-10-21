import sys; sys.path.insert(0,"/Users/tshermatova/Downloads/week7-exam/src")
import lib, numpy as np
HA, GT = lib.val_split()
print("users with history in A:", len(HA))
print("test users with target in B:", sum(1 for u in lib.TEST_UI if GT.get(u)))
sizes=[len(GT[u]) for u in lib.TEST_UI if GT.get(u)]
print("median |R| = %d  mean %.1f" % (np.median(sizes), np.mean(sizes)))
pop=lib.popularity(HA); fb=[s for s,_ in pop.most_common(5)]
print("MAP@5 popularity baseline (all 1260 users, zeros included): %.5f" %
      lib.evaluate({u:fb for u in lib.TEST_UI}, GT))
