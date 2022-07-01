import numpy as np
import sys

sys.path.append("..")

from sklearn.preprocessing import normalize
import math
from itertools import permutations  
import pandas as pd
import gzip
from icecream import ic
from tqdm import tqdm

class PbmEnv():
    '''
    stochastic setting
    '''
    def __init__(self, args, conf):
        self.conf = conf
        theta, kappa = self.conf.get_the_kap(args.type)
        self.kap=np.array(kappa) 
        self.theta=np.array(theta) 
        self.T=self.conf.T
        if args.data.startswith('KDD'):
            idx=int(args.data[-1])
            self.theta,self.kap=np.array(self.conf.KDD[idx][0]),np.array(self.conf.KDD[idx][1])
        self.K,self.L=len(self.theta),len(self.kap)
        
    def _getTK(self,t):
        return self.theta,self.kap

    def feedback(self, At, t):
        theta,kap=self._getTK(t)
        means = theta[At] * kap
        if self.conf.VERBOSE:
            print('in env At={} theta={} kap={} means={}'.format(At,theta,kap,means))
        Zt=np.random.binomial(1, means)
        curReg=self.get_best_reward(theta,kap)-sum(means)
        return Zt, sum(means), curReg

    def get_best_reward(self,theta,kap):
        '''
        get current round's best reward, for computing the real regret
        '''
        return sum((np.sort(theta)[self.K-self.L:])*np.sort(kap))

class AdvPbmEnv():
    '''
    the adversarial position-based model without known kappa.
    '''
    def __init__(self, args, conf):
        self.conf = conf
        theta, kappa = self.conf.get_the_kap(args.type)
        self.ADVERSARIAL_SETTING=self.conf.ADVERSARIAL_SETTING
        self.Theta,self.Kappa=np.array(theta),np.array(kappa)
        self.PHASE_BASE=self.conf.PHASE_BASE
        self.T=self.conf.T
        self.K,self.L=self.conf.K,self.conf.L

    def _getCurrentPhase(self,t):
        '''
        phase is exponentially reversed if phase_base < 10 or uniformly reversed otherwise
        '''
        res = -1
        if self.PHASE_BASE < 10:
            res = int(math.log(t+1,self.PHASE_BASE))&1
        else:
            res = (t//self.PHASE_BASE)&1
        return res

    def _getTK(self,t):
        '''
        get current time's theta & kappa
        odd phase or even phase
        '''
        lastBit = self._getCurrentPhase(t)
        ic(t, lastBit)
        if self.ADVERSARIAL_SETTING==1:
            return self.Theta,self.Kappa
        elif self.ADVERSARIAL_SETTING==2:
            if lastBit:
                return self.Theta,self.Kappa
            else:
                return self.Theta[::-1],self.Kappa
        elif self.ADVERSARIAL_SETTING==3:
            if lastBit:
                return self.Theta,self.Kappa
            else:
                return self.Theta[::-1],self.Kappa[::-1]
        elif self.ADVERSARIAL_SETTING == 4:
            if lastBit:
                return self.Theta, self.Kappa
            else:
                return np.hstack((self.Theta[5:], self.Theta[:5])) , self.Kappa

    def feedback(self, At, t):
        theta,kap=self._getTK(t)
        means = theta[At] * kap
        if self.conf.VERBOSE:
            print('in env At={} theta={} kap={} means={}'.format(At,theta,kap,means))
            print('in env real loss={}'.format(1-means))
        Zt=np.random.binomial(1, means)
        curReg=self.get_best_reward(theta,kap)-sum(means)
        return Zt, sum(means), curReg

    def get_best_reward(self,theta,kap):
        '''
        get current round's best reward, for computing the real regret
        '''
        return sum((np.sort(theta)[self.K-self.L:])*np.sort(kap))