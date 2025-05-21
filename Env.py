import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from joblib import Parallel, delayed
import os
import tempfile
from RiskMeasureEst import *
import random


class Env:
    def __init__(self,p,c,gamma = 0.90,B = 5,A=2):
        self.gamma = gamma
        self.B = B
        self.A = A
        self.sidx = 0
        self.p = p
        self.c = c
        self.totalStep = 0
    
    def step(self,aidx):
        s0idx = self.sidx
        Pvector = self.p[self.sidx,aidx]
        s1idx = np.random.choice(list(range(0,self.B)),p=Pvector)
        cost = self.c[self.sidx,aidx,s1idx]
        self.sidx = s1idx
        self.totalStep += 1
        #print(s0idx,aidx,s1idx,cost)
        return s0idx,aidx,s1idx,cost
    
