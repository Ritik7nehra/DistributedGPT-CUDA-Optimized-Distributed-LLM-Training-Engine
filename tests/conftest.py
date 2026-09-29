import os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("src","monitoring","cuda","."):
    p=os.path.join(ROOT,sub)
    if p not in sys.path: sys.path.insert(0,p)
