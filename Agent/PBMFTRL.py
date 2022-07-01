import sys,copy

sys.path.append("..")

import numpy as np
import math
from utils import utils
# from Env import conf
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from Agent.Base import Base
import wandb

class PBMFTRL(Base):
    def __init__(self, env, args):
        self.conf = env.conf
        self.args = args
        self.K = env.K # num of items
        self.L = env.L # num of positions
        self.env = env
        self.T = env.T
        self.rewards = np.zeros(self.T)
        self.cum_regrets = np.zeros(self.T)
        self.KNOWN_KAPPA=self.conf.KNOWN_KAPPA
        if self.KNOWN_KAPPA:
            self.kap=env.kap
            self.kap_prime=np.array([i for i in self.kap]+[0]*(self.K-self.L))
            if self.conf.VERBOSE:
                print('kappa={}\n kappa_prime={}'.format(self.kap,self.kap_prime))
        self.iteNum=self.conf.iteNum
        # self.eps=self.conf.eps
        self.d=self.K*self.L
        self.FWEps=self.conf.FTRL['FWEps']
        
        self.eta=-1
        self.n_FWIte=self.conf.FTRL['n_FWIte']
        self.Beta=np.zeros((self.K,self.K,self.K)) # for mix decomposition 

        self.scaleEta=self.conf.FTRL['scaleEta']
        self.type = self.conf.FTRL['type']
        self.gradient = self.conf.FTRL['gradient'][0] if self.type == 'ours' else self.conf.FTRL['gradient'][1]
        

    def getEta(self,t):
        return 1/math.sqrt(t) * self.scaleEta

    def dPsi(self,A):
        '''
        compute the gradient of Psi w.r.t. A

        A: (L,K) ndarray

        gradient: -1-log(1-x)-x^(1/2)
        gradient: -1*x**(-1/2)
        '''
        res=eval(self.gradient)
        return res

    def getAlgoName(self):
        return super().getAlgoName()+'_{}_{}'.format(self.type, self.n_FWIte)

    def dFun(self,A,Loss_hat):
        '''
        compute the gradient of line3
        '''
        return Loss_hat+(1/self.eta)*self.dPsi(A)

    def __linear_optimization(self,gradient):
        '''
        compute the linear optimization in FW algo.
        '''
        res=''
        if not self.KNOWN_KAPPA:
            s=linear_sum_assignment(gradient)
            tem=np.zeros_like(gradient)
            tem[s]=1
            res=tem
        return res
        
    def FW(self,a0,Loss_hat):
        '''
        a0 is a (L,K) matrix and is the result of the last time step 

        return: a (L,K) convex combination of several sub-permutation matrices 
        '''
        a=a0
        acc=0
        for ite in range(1,self.n_FWIte+1):
            gradient=self.dFun(a,Loss_hat)
            s=self.__linear_optimization(gradient)
            acc=np.sum((a-s)*gradient)
            if acc<self.FWEps:
                break
            gamma=2/(ite+2)
            a=(1-gamma)*a+gamma*s
        if self.conf.VERBOSE:
            print('\n********************** in FW acc={}\n'.format(acc))
        return a

    def _decompose(self,sP):
        if not self.KNOWN_KAPPA:
            M_tilde=np.zeros((self.K,self.K))
            M_tilde[:self.L]=np.copy(sP)
            tem1=np.sum(sP,axis=0)
            for i in range(self.L,self.K):
                for j in range(self.K):
                    M_tilde[i,j]=1/(self.K-self.L)*(1-tem1[j])
            q,M=utils.matrix_decompose(M_tilde)
            return q,M           

    def run_adv(self):
        if not self.KNOWN_KAPPA:
            reg=0
            rews=0
            sus=0
            Loss_hat=np.zeros((self.L,self.K))
            a_t=np.ones((self.L,self.K))/self.K
            for t in tqdm(range(self.T)):
                self.eta=self.getEta(t+1)
                a_t=self.FW(a_t,Loss_hat)
                q,M=self._decompose(a_t)
                pi=np.random.choice(a=len(q),size=1,p=q)[0]
                slate=M[pi][:self.L]
                row_idx,col_idx=np.nonzero(slate)
                At=col_idx
                # interact with env.
                Zt, r, pseudo_regret= self.env.feedback(At,t)
                loss=np.zeros((self.L,self.K))
                loss[row_idx,col_idx]=1-Zt
                if self.conf.VERBOSE:
                    print('in ftrl a_t={}'.format(a_t[row_idx,col_idx]))
                if np.sum(a_t[row_idx,col_idx]<=0):
                    if self.conf.VERBOSE:
                        print('pi={} q={} q[pi]={} slate=\n{} a_t=\n{}'.format(pi,q,q[pi],slate,a_t))
                else:
                    loss[row_idx,col_idx]/=np.maximum(a_t[row_idx,col_idx], 1e-5)
                    Loss_hat+=loss
                rews+=r
                self.rewards[t] = rews
                rD=pseudo_regret
                reg+=rD
                self.cum_regrets[t]=reg
                if rD<=1e-5:
                    sus+=1
                if self.args.useWandb:
                    wandb.log({"avg_reward": rews/(t+1), "cum_reward": rews})
                if self.conf.VERBOSE:
                    print('Round:{} reward:{} cumRewards:{} cur_reg:{} cumReg:{} susRate={}'.format(t,r,self.rewards[t],rD,reg,sus/(t+1)))
            return self.rewards,self.cum_regrets