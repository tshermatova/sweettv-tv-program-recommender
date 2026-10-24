"""Score the shipped configuration at whatever N_POP the environment asks for.

The candidate pool is own history + the N_POP most popular programmes. Oracle inside
the current pool is 0.85963 against a model at 0.564, so the pool is not the binding
constraint — but a wider pool was last tested with 30 features and 800 trees, when the
model lost 0.003 to the extra noise. A stronger model may now absorb it.
"""
import os
import sys, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
lab = importlib.import_module("31_lab")
cache2 = importlib.import_module("30_cache2")
print(f"N_POP={cache2.N_POP}  cache={cache2.CACHE}", flush=True)
lab.run2(f"50 features, 1600 trees, N_POP={cache2.N_POP}",
         opt=dict(feats3=True, feats5=True), n_estimators=1600, learning_rate=0.015)
