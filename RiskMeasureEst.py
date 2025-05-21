import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from joblib import Parallel, delayed
import os
import tempfile

def sigmaEstimator(V,P,c,gamma=0.9,\
                  aph=None, \
                  kappa = None,\
                  risk_type='Mean'):
    B = P.shape[0]
    ActionL = P.shape[1]
    L = P.shape[-1]
    
    if risk_type == 'Mean':
        
        def compute_J(sidx, aidx, pidx):
            p = P[:,:,:,pidx]
            c_obj = - p[sidx, aidx] * (c[sidx, aidx] + gamma * V)

            return -c_obj.sum()
        
    elif risk_type == 'CVaR':
        def compute_J(sidx, aidx, pidx):
            p = P[:,:,:,pidx]
            c_obj = - p[sidx, aidx] * (c[sidx, aidx] + gamma * V)

            A_eq = np.array([
                p[sidx, aidx]   
            ])
            b_eq = np.array([1]) 

            bounds = [(0, 1/aph)] * B

            result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            
            return -result.fun
        
    elif risk_type == 'Mean_Semideviation':
        def compute_J(sidx, aidx, pidx):
            p = P[:,:,:,pidx]
            rewards = c[sidx, aidx] + gamma * V
            cost = rewards  
            mu = np.dot(p[sidx, aidx], cost)  

            semidev = np.maximum(0, cost - mu)

            risk = mu + kappa * np.sqrt(np.dot(p[sidx, aidx], semidev**2))

            return risk
                
    os.environ['JOBLIB_TEMP_FOLDER'] = r"C:\temp"

    results = Parallel(n_jobs=-1)(
        delayed(compute_J)(sidx, aidx,pidx)
        for sidx in range(B)
        for aidx in range(ActionL)
        for pidx in range(L)
    )
    J = np.array(results).reshape(B, ActionL, L)

    return J
def BellmanOperator(V,c,alpha,P,gamma=0.9,\
                  aph=None, \
                  kappa = None,\
                  risk_type='Mean',\
                  aph_beta = None,\
                  beta_type='Mean'):
    B = alpha.shape[0]
    ActionL = alpha.shape[1]
    L = P.shape[-1]
    
    Q = sigmaEstimator(V,P,c,\
                  gamma = gamma,\
                  aph=aph, \
                  kappa = kappa,\
                  risk_type=risk_type)
    
    if beta_type == 'Mean':
        
        def compute_J(sidx, aidx):
            return Q[sidx,aidx].mean()
        
    elif beta_type == 'CVaR':
        def compute_J(sidx, aidx):
            c_obj = - Q[sidx, aidx] /L
        
            A_eq = np.array([
                np.ones(L)/L
            ])
            b_eq = np.array([1]) 

            bounds = [(0, 1/aph_beta) for _ in range(L)] 

            result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            
            return -result.fun
        
    os.environ['JOBLIB_TEMP_FOLDER'] = r"C:\temp"
    
    results = Parallel(n_jobs=-1)(
        delayed(compute_J)(sidx, aidx)
        for sidx in range(B)
        for aidx in range(ActionL)
    )
    J = np.array(results).reshape(B, ActionL)
    
    return J
