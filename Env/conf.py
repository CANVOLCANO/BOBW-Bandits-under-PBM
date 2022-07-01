import numpy as np

K=10
L=5

dt=0.01
t1=0.1
t2=0.7
t3=0.5

dk=0.05
k1=0.15
k2=0.45

SHRINK_BASE = int(1)
# PHASE_BASE=1.6
PHASE_BASE = int(1e5) // SHRINK_BASE
VERBOSE = 0

KNOWN_KAPPA=0

iteNum=5 # for binary search
eps=0.1 # for algo
epsilon=np.finfo(np.float32).eps # for compute  

T=int(5e5) // SHRINK_BASE
# T = int(1e2)
PMED={'mle_ite_num':5,'cutting_plane_iteNum':5,'LR':0.0001}
FTPL={'scaleZ':1}
FTRL = {
        'n_FWIte':int(1e3),
        'FWEps':0.01,
        'scaleEta':1,
        'type':'ours',  
        'gradient': ['-1*(A**(-1/2))', '-0.5*(A**(-1/2))-np.log(1-A)-1'] # '-1*(A**(-1/2))' is standard
    } 

'''
for stochastic setting
'''
# type = None
def get_the_kap(type):
    theta, kappa = None, None
    if type == 'synthetic1':
        theta=[0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86] # 0.01
        kappa=[1/i for i in range(1,1+L)]
    elif type == 'synthetic2':
        theta=[0.95, 0.92, 0.89, 0.86, 0.83, 0.8, 0.77, 0.74, 0.71, 0.68] # 0.03
        # theta=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5] # 0.05
        kappa=[1/i for i in range(1,1+L)]
    elif type == 'yandex':
        theta=[0.8941321809618571, 0.23103019795887, 0.13932148712964998, 0.07445376605518324, 0.05846070308961704, 0.04239223303728405, 0.02371431519709203, 0.023434456759181728, 0.023075837829763263, 0.017830056679033292] # yandex
        kappa=[0.8911321809618571, 0.2274335318498291, 0.07782579219572691, 0.041191773955868205, 0.0378496545411373] # yandex
    return theta, kappa


'''
for adversarial setting.
# 1: Fix \kappa, order of \theta is fixed.
2: \theta will be inversed in each phase, but \kappa is fixed.
3: \theta and \kappa will be inversed in each phase.  (periodic setting 2 in paper)
4: (periodic setting 1 in paper)
'''
ADVERSARIAL_SETTING = 4


def list2str(l):
    return '[{}]'.format(','.join([str(round(i,2)) for i in l]))


def genParamTuple(data, type):
    theta, kappa = get_the_kap(type)
    if data=='sto': 
        name=[]
        name.append('[type={}]'.format( type))
        name.append('[K={}]'.format(len(theta)))
        name.append('[L={}]'.format(len(kappa)))
        name.append('[T={}]'.format(T))
        name.append('[KK={}]'.format(1 if KNOWN_KAPPA else 0))
    elif data.startswith('adv'):
        name=[]
        name.append('[AdvSetting{}]'.format(ADVERSARIAL_SETTING))
        name.append('[type={}]'.format( type))
        name.append('[Phase_base={}]'.format(PHASE_BASE))
        name.append('[K={}]'.format(K))
        # the='[{}]'.format(list2str(theta))
        # kap='[{}]'.format(list2str(kappa))
        # name.append('[the={}]'.format(the))
        name.append('[L={}]'.format(L))
        # name.append('[kap={}]'.format( kap))
        name.append('[T={}]'.format(T))
        name.append('[KK={}]'.format(1 if KNOWN_KAPPA else 0))
    return '+'.join(name)