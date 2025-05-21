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
from Agent import *
from tqdm import tqdm



import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar,brentq

def get(arr,alist = []):
    array_tuples = arr.reshape(arr.shape[0], -1).T 
    
    if len(alist)>0:
        list_array = np.array(alist)
        combined = np.vstack((array_tuples, list_array))
    else:
        combined = array_tuples
    unique_tuples = np.unique(combined, axis=0)
    return unique_tuples.tolist()

def ground_true(actionlist,p,c,N,risk_type='CVaR'):
    V = np.zeros(N)
    for _ in range(1000):
            V0 = V
            J = sigmaEstimator(V0,np.array([p.T]).T,c,\
                  aph=0.5, \
                  kappa = None,\
                  risk_type='CVaR')[:,:,0]
            V = np.array([J[c,actionlist[c]] for c in range(N)])
            if np.max(np.abs(V0-V))<=0.01:
                break
    #print(np.sum(np.abs(V0-V)))
    return V

def ground_true_all(allAction,p,c,B,risk_type):
    hhlist =[]
    print('Calculating Ground Truth')
    for actionlist in tqdm(allAction):
        #print(actionlist)
        hh = ground_true(actionlist,p,c,B,risk_type)
        #print(hh)
        hhlist.append(hh)
    return hhlist

def summarize(arr,hhlist,allAction):
    ddlist = arr.reshape(arr.shape[0],-1).T.tolist()

    for j in range(len(ddlist)):
        ddlist[j] = hhlist[allAction.index(ddlist[j])][0]
    ddlist = pd.DataFrame(np.array(ddlist).reshape((arr.shape[1],arr.shape[2])))
    
    ddlist = ddlist.iloc[[0,100,200,300,400,500,600,700,800,900]].reset_index(drop=True)
    return ddlist

def evaluate(p,r,B,ct='CVaR',nameslist = [['CVaR','Mean','1']],task='CP'):
    
    allAction = []
    for names in nameslist:
        risk_type,beta_type,Prior_name = names
        arr = np.load('res/rho_{}_beta_{}_Prior_{}_task_{}.npy'.format(risk_type,beta_type,Prior_name,task)).astype('int')
        allAction=get(arr,alist = allAction)
    hhlist = ground_true_all(allAction,p,-r,B,risk_type=ct)    
    
    adict = {}
    for names in nameslist:
        risk_type,beta_type,Prior_name = names
        arr = np.load('res/rho_{}_beta_{}_Prior_{}_task_{}.npy'.format(risk_type,beta_type,Prior_name,task)).astype('int')
        
        adict['rho_{}_beta_{}_Prior_{}'.format(risk_type,beta_type,Prior_name)]=summarize(arr,hhlist,allAction)
        
    return adict


def evaluate_robust_expectation(p,r,kllist,gamma,B,\
                                nameslist = [['Mean','Mean','1']],read_file='False',task='CP'):

    


    def summarize(arr,resdf,allAction,kllist):
        resultlist = []
        for kl in kllist:
            hhlist = resdf[kl].tolist()
            ddlist = arr.reshape(arr.shape[0],-1).T.tolist()
            
            for j in range(len(ddlist)):
                ddlist[j] = hhlist[allAction.index(ddlist[j])]
            ddlist = pd.DataFrame(np.array(ddlist).reshape((arr.shape[1],arr.shape[2])))
            resultlist.append(ddlist.mean(axis=1).iloc[-1])
        return resultlist
        
        
            
    def worst_case_expectation(values, p, epsilon):
        values = np.array(values)
        p = np.array(p)
        support = p > 0
        values = values[support]
        p = p[support]
        p /= np.sum(p)

        def dual(eta):
            if eta <= 0:
                return np.inf
            log_exp = logsumexp(values / eta, b=p)
            return eta * (log_exp) + eta * epsilon

        res = minimize_scalar(dual, bounds=(1e-3, 1e3), method='bounded')

        eta_opt = res.x
        log_weighted = np.log(p) + values / eta_opt
        q = np.exp(log_weighted - logsumexp(log_weighted))
        return np.dot(q, values)


    def cvar_kl_projection_strict(values, p, epsilon, alpha=0.5):
        values = np.array(values)
        p = np.array(p)

        support = p > 0
        values = values[support]
        p = p[support]
        p /= np.sum(p)
        
        h = values
        worst_eq = worst_case_expectation(h, p, epsilon)
        return worst_eq

    def robust_e_value_iteration(P, R, pi, gamma=0.95, alpha=0.5, epsilon=0.5, tol=1e-6, max_iter=1000):
        S, A, _ = P.shape
        V = np.zeros(S)

        for it in range(max_iter):
            V_old = V.copy()
            for s in range(S):
                a = pi[s]
                p_sa = P[s, a]
                r_sa = R[s, a]
                values = r_sa + gamma * V
                V[s] = cvar_kl_projection_strict(values, p_sa, epsilon, alpha)
            if np.max(np.abs(V - V_old)) < tol:
                break

        return V

    def acList2pi(actionlist):
        return np.array([[1.0 if c ==actionlist[s]\
                          else 0.0 for c in range(2)] for s in range(B)])

    allAction = []
    for names in nameslist:
        risk_type,beta_type,Prior_name = names
        arr = np.load('res/rho_{}_beta_{}_Prior_{}_task_{}.npy'.format(risk_type,beta_type,Prior_name,task)).astype('int')[:,-2:,:]
        allAction=get(arr,alist = allAction)
    #hhlist = ground_true_all(allAction,p,-r,B,risk_type='Mean')
    if read_file==True:
        resdf = pd.read_pickle('res/KLmean{}.pickle'.format(task))
    else:
        res = []
        for each in tqdm(allAction):
            reslist = {}
            for kl in kllist:
                a = robust_e_value_iteration(p, -r, each\
                         , gamma=gamma, epsilon=kl, max_iter=1000, tol=1e-1)
                reslist[kl] = a[0]
            res.append(reslist)
        resdf = pd.DataFrame(res)
        resdf.to_pickle('res/KLmean{}.pickle'.format(task))

    adict = {}
    for names in nameslist:
        risk_type,beta_type,Prior_name = names
        arr = np.load('res/rho_{}_beta_{}_Prior_{}_task_{}.npy'.format(risk_type,beta_type,Prior_name,task)).astype('int')[:,-2:,:]
        adict['rho_{}_beta_{}_Prior_{}'.format(risk_type,beta_type,Prior_name)]=summarize(arr,resdf,allAction,kllist)
        
    return adict


def evaluate_robust_CVaR(p,r,kllist,gamma,B,\
                                nameslist = [['CVaR','Mean','1']],read_file=False,task='CP'):

    
    def summarize(arr,resdf,allAction,kllist):
        resultlist = []
        for kl in kllist:
            hhlist = resdf[kl].tolist()
            ddlist = arr.reshape(arr.shape[0],-1).T.tolist()
            
            for j in range(len(ddlist)):
                ddlist[j] = hhlist[allAction.index(ddlist[j])]
            ddlist = pd.DataFrame(np.array(ddlist).reshape((arr.shape[1],arr.shape[2])))
            resultlist.append(ddlist.mean(axis=1).iloc[-1])
        return resultlist
        
        
            
    def worst_case_expectation(values, p, epsilon):

        values = np.array(values)
        p = np.array(p)

        # Restrict to support
        support = p > 0
        values = values[support]
        p = p[support]
        p /= np.sum(p)

        def dual(eta):
            if eta <= 0:
                return np.inf
            log_exp = logsumexp(values / eta, b=p)
            return eta * (log_exp) + eta * epsilon

        res = minimize_scalar(dual, bounds=(1e-3, 1e3), method='bounded')

        eta_opt = res.x
        log_weighted = np.log(p) + values / eta_opt
        q = np.exp(log_weighted - logsumexp(log_weighted))
        return np.dot(q, values)


    def cvar_kl_projection_strict(values, p, epsilon, alpha=0.5):
        values = np.array(values)
        p = np.array(p)

        support = p > 0
        values = values[support]
        p = p[support]
        p /= np.sum(p)

        def cvar_objective(v):
            h = np.maximum(values - v, 0)
            worst_eq = worst_case_expectation(h, p, epsilon)
            return (v + (1 / alpha) * worst_eq)

        res = minimize_scalar(cvar_objective, bounds=(min(values)-10, max(values)+10), method='bounded')
        if not res.success:
            raise ValueError("CVaR v-optimization failed.")
        return res.fun

    def robust_cvar_value_iteration(P, R, pi, gamma=0.95, alpha=0.5, epsilon=0.5, tol=1e-6, max_iter=1000):
        S, A, _ = P.shape
        V = np.zeros(S)

        for it in range(max_iter):
            V_old = V.copy()
            for s in range(S):
                a = pi[s]
                p_sa = P[s, a]
                r_sa = R[s, a]
                values = r_sa + gamma * V
                V[s] = cvar_kl_projection_strict(values, p_sa, epsilon, alpha)
            #print(V)
            if np.max(np.abs(V - V_old)) < tol:
                break

        return V


    def acList2pi(actionlist):
        return np.array([[1.0 if c ==actionlist[s]\
                          else 0.0 for c in range(2)] for s in range(B)])

    allAction = []
    for names in nameslist:
        risk_type,beta_type,Prior_name = names
        arr = np.load('res/rho_{}_beta_{}_Prior_{}_task_{}.npy'.format(risk_type,beta_type,Prior_name,task)).astype('int')[:,-2:,:]
        allAction=get(arr,alist = allAction)
    #hhlist = ground_true_all(allAction,p,-r,B,risk_type='Mean')
    if read_file==True:
        resdf = pd.read_pickle('res/KLcvar{}.pickle'.format(task))
    else:
        res = []
        for each in tqdm(allAction):
            reslist = {}
            for kl in kllist:
                a = robust_cvar_value_iteration(p, -r, each\
                         , gamma=gamma, alpha=0.5, epsilon=kl, max_iter=1000, tol=1e-1)
                reslist[kl] = a[0]
            res.append(reslist)
        resdf = pd.DataFrame(res)
        resdf.to_pickle('res/KLcvar{}.pickle'.format(task))

    adict = {}
    for names in nameslist:
        risk_type,beta_type,Prior_name = names
        arr = np.load('res/rho_{}_beta_{}_Prior_{}_task_{}.npy'.format(risk_type,beta_type,Prior_name,task)).astype('int')[:,-2:,:]
        adict['rho_{}_beta_{}_Prior_{}'.format(risk_type,beta_type,Prior_name)]=summarize(arr,resdf,allAction,kllist)
        
    return adict
