"""Score the shipped configuration at whatever FEAT_THR the environment asks for.

FEAT_THR decides which viewing events exist at all, so it is fixed when `lib` is
imported and each threshold needs its own process and its own cache directory. This
script prints one line; run it once per threshold and compare the lines.

    FEAT_THR=0.10 python3 src/40_thr_run.py
"""
import sys, time, importlib, numpy as np
sys.path.insert(0, "/Users/tshermatova/Downloads/week7-exam/src")
import lib

lab = importlib.import_module("31_lab")
cache2 = importlib.import_module("30_cache2")

print(f"FEAT_THR={lib.FEAT_THR}  cache={cache2.CACHE}", flush=True)
lab.run2(f"50 features, 1600 trees, FEAT_THR={lib.FEAT_THR}",
         opt=dict(feats3=True, feats5=True), n_estimators=1600, learning_rate=0.015)
