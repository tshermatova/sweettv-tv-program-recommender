"""Block 2: the programme's broadcast history before the predicted window."""
import os
import sys, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("49_f10")   # reuses its run() and the true-target evaluation
