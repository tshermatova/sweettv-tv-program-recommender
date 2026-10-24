"""Build the second hold-out: evaluation set, its three training windows, and extras."""
import os
import sys, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
c2 = importlib.import_module("30_cache2")
ex = importlib.import_module("33_extra")

c2.outer2()
for t in c2.EVAL2_WIN:
    c2.get(t)
for t in ["outer2"] + c2.EVAL2_WIN:
    ex.extras(t)
    ex.extras4(t)
print("eval2 ready")
