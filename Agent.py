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
from Env import Env

def SampleP(alpha,N):
    B = alpha.shape[0]
    K = alpha.shape[1]
    samples = np.zeros((B, K, B, N))
    for nidx in range(N):
        for sidx in range(B):
            for aidx in range(K):
                samples[sidx,aidx,:,nidx] = np.random.dirichlet(alpha[sidx,aidx,:])
    return samples

class Agent:
    def __init__(self,env,\
                 alpha0,\
                 N=100,\
                  aph=None, \
                  kappa = None,\
                  risk_type='CVaR',\
                  aph_beta = 0.2,\
                  beta_type='Mean'):
        self.env = env
        self.gamma = env.gamma 
        self.B =env.B
        self.A = env.A
        self.c = env.c
        
        
        self.aph = aph
        self.kappa = kappa
        self.aph_beta = aph_beta
        self.risk_type = risk_type
        self.beta_type = beta_type
        self.alpha = alpha0
        self.N = N
        
        self.V = np.zeros(env.B)
        self.Q = np.zeros((env.B,env.A))
        self.p = np.zeros((env.B,env.A,env.B))
        
        self.totalStep = env.totalStep
        
        
    def DP(self):
        do = True
        V = self.V.copy()
        P = SampleP(self.alpha,self.N)
        for _ in range(1000):
            V0 = V
            Q = BellmanOperator(V0,self.c,self.alpha,P,\
                  aph=self.aph, \
                  kappa = self.kappa,\
                  risk_type=self.risk_type,\
                  aph_beta = self.aph_beta,\
                  beta_type=self.beta_type)
            V = Q.min(axis=1)
            if np.max(np.abs(V0-V))<=0.01:
                break
            
        self.V = V
        self.Q = Q
        return V,Q
        
    def selectAction(self,sidx):
        self.totalStep = self.env.totalStep
        u = random.random()
        epsilon = max(0.05,1.0-self.totalStep*0.05)
        if u<epsilon:
            aidx = random.sample(list(range(self.A)),1)[0]
        else:
            aidx = self.Q.argmin(1)[sidx]
        return aidx
        
        
    def updateAlpha(self,sidx,aidx,s1idx):
        self.alpha[sidx,aidx,s1idx] += 1
        return self.alpha
    

