import math
import scipy.stats
import numpy as np
from Env import conf
import copy
from scipy.optimize import linear_sum_assignment

from numba import jit

# @jit(nopython=True)
def is_power2(n):
    return n > 0 and ((n & (n - 1)) == 0)

# @jit(nopython=True)
def is_power2(n):
    return n > 0 and ((n & (n - 1)) == 0)

# @jit(nopython=True)
def KL(p,q):
    '''
    KL divergence
    '''
    res=scipy.stats.entropy([p,1-p],[q,1-q]) 
    return res

# @jit(nopython=True)
def dKL_q(p,q):
    '''
    return the first derivate of KL w.r.t. q when p is a fixed constant
    '''
    return -p/q+(1-p)/(1-q)

def binary_search(l,r,iteNum,fun,val,k):
    '''
        fun is monotonically increasing
        
        for KL
    '''
    m=(l+r)/2
    for i in range(iteNum):
        mV=fun(k,m)
        if mV>val:
            r=m
        else:
            l=m
        m=(l+r)/2
    return m

def binary_search_increasing(l,r,iteNum,fun,val,p):
    '''
        fun is monotonically increasing
        not for KL
    '''
    m=(l+r)/2
    for i in range(iteNum):
        mV=fun(p,m)
        if mV>val:
            r=m
        else:
            l=m
        m=(l+r)/2
    return m

def binary_search_decreasing(l,r,iteNum,fun,val,p):
    '''
        fun is monotonically decreasing
        not for KL
    '''
    m=(l+r)/2
    for i in range(iteNum):
        mV=fun(p,m)
        if mV>val:
            l=m
        else:
            r=m
        m=(l+r)/2
    return m

def test(x):
    print(x+10)

def pseudo_regret_ub():
    d=conf.L*conf.K
    return conf.L*math.sqrt(2*(1+math.e**2)*conf.T*d*(1+math.log(d)))

def matrix_decompose(N_tilde):
    '''
    decompose a doubly stochastic matrix into the convex combination of several purmutation matrices

    N_tilde: (K,K) ndarray
    '''
    tem2=N_tilde
    c_req,E=[],[]
    K=N_tilde.shape[0]
    while np.sum(N_tilde)>0:
        if  len(c_req)>K*K:
            break
        row_idx,col_idx=linear_sum_assignment(N_tilde,maximize=True)
        c_req.append(np.min(N_tilde[row_idx,col_idx]))
        tem=np.zeros((K,K))
        tem[row_idx,col_idx]=1
        E.append(copy.deepcopy(tem))
        tem*=c_req[-1]
        N_tilde-=tem

    c_req=np.array(c_req)
    E=np.array(E)
    filtIdx=c_req>1e-8

    c_req=c_req[filtIdx]
    E=E[filtIdx]

    if c_req.shape[0]==0:
        c_req=np.array([1])
        E=np.array([np.identity(K)])
    else:
        c_req+=(1-min(1,sum(c_req)))/c_req.shape[0]
    if sum(c_req<0)>0:
        print('in utils mat decom negative c_req idx={} val={}'.format(c_req<0,c_req[c_req<0]))
    return c_req,E