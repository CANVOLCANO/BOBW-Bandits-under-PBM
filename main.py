import numpy as np
import random,os
from Env.Env import PbmEnv,AdvPbmEnv
from Agent.PBMFTRL import PBMFTRL
from Env import conf
import argparse, wandb
import datetime
from time import strftime
import shutil
import time

def main(envname,args,filename=''):
    seed=args.seed
    print("Seed = %d" % seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.data.startswith('adv'):
        conf.ADVERSARIAL_SETTING = int(args.data[-1])
        env = AdvPbmEnv(args=args, conf=conf) 
    elif args.data=='sto':
        env = PbmEnv(args=args, conf=conf)

    agent=''
    if args.algo=='PBMFTRL':
        agent=PBMFTRL(env,args)

    resFile='./Res/{}/{}/{}'.format(args.data,conf.genParamTuple(args.data, args.type),agent.getAlgoName())
    if not os.path.exists(resFile):
        try:
            os.makedirs(resFile)
        except:
            print('file exists')
    print('in main refFile={}'.format(resFile))
    filename=resFile
    starttime = time.time()
    cum_rewards,cum_regs = agent.run_adv()
    rruntime = time.time() - starttime
    if envname in ['cas', 'pbm']:
        np.savez(filename+'/'+envname+'_{}'.format(args.algo)+'_'+str(seed), seed, cum_rewards, cum_regs , rruntime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--output', dest='output', default=1, type=int)
    parser.add_argument('--algo', dest='algo', default='PBMFTRL', type=str)
    parser.add_argument('--data', dest='data', default='sto', type=str, choices=['sto', 'adv3', 'adv4'])
    parser.add_argument('--type', dest='type', default='synthetic1', type=str, choices=['synthetic1', 'synthetic2', 'yandex'])
    parser.add_argument('--scaleZ', dest='scaleZ', default='1', type=float)
    parser.add_argument('--scaleEta', dest='scaleEta', default='1', type=float)
    parser.add_argument('--useWandb', dest='useWandb', default=1, type=int)
    args = parser.parse_args()

    resFile=''
    now = datetime.datetime.now()
    ttime = str(now.strftime("%H:%M"))
    if args.useWandb:
        wandb.init(project='PBM-Hidden',
                    name=str(args.seed),
                    group=args.data,
                    job_type=args.type+'-'+args.algo+'-'+ttime,
                    reinit=True)
        wandb.config.update(args)
    main(args=args,envname='pbm',filename=resFile)


