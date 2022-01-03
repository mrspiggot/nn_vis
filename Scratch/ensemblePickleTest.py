import pickle
import pandas as pd
import os

fname = os.path.join(os.pardir, "assets", "cifar10Output20220101_170642.pickle")
fname = os.path.join(os.pardir, "assets", "cifar10Output_ENS__4_20220101_170749_0.7382.pickle")
print(fname)
print(os.getcwd())
with open(fname, 'rb') as pkl:
    p_dict = pickle.load(pkl)

print(p_dict)