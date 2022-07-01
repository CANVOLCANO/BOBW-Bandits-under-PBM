with open('./run_ftrl.sh','w') as f:
    algo=['PBMFTRL']
    data=['sto', 'adv3', 'adv4']
    type=['synthetic1', 'synthetic2', 'yandex']
    for ag in algo:
        for d in data:
            for t in type:
                for seed in range(10):
                    print('nohup python ./main.py  --algo={} --seed={} --data={} --type={}& \n'.format(ag, seed, d, t),file=f)
