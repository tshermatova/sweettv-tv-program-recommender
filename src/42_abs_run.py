"""Score the shipped configuration under whatever 'watched' rule the environment sets."""
import sys, importlib
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib
lab = importlib.import_module("31_lab")
cache2 = importlib.import_module("30_cache2")
rule = f"OV>={int(lib.ABS_SEC)}s" if lib.ABS_SEC else f"FRAC>={lib.FEAT_THR}"
print(f"rule={rule}  cache={cache2.CACHE}", flush=True)
lab.run2(f"50 features, 1600 trees, {rule}",
         opt=dict(feats3=True, feats5=True), n_estimators=1600, learning_rate=0.015)
