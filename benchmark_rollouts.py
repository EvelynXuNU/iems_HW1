"""Validate complete batched episodes; benchmark actual rollout throughput.

Imports of torch stay inside main so spawned environment workers stay lightweight.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path


def main():
    import numpy as np
    import torch
    import q2_policy_gradient as p1
    import q3_policy_gradient as p2
    from batched_rollout import EpisodeBatch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game',choices=['cartpole','pong'],default='pong')
    parser.add_argument('--checkpoint')
    parser.add_argument('--episodes',type=int,default=32)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',default='results/q3/collector_validation.json')
    args=parser.parse_args()
    p1.seed_everything(3817)
    if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('Exactly one visible GPU required')
    config=p1.Config(game=args.game,reward_scale=1 if args.game=='pong' else 20)
    policy=p1.Policy(config)
    if args.checkpoint:
        c=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        config=p1.Config(**c['config'])
        policy=p1.Policy(config)
        policy.load_state_dict(c['policy'])
    policy.eval()
    count=16
    env_seeds=list(range(1_300_000_000,1_300_000_000+count))
    action_seeds=list(range(78000,78000+count))
    with EpisodeBatch(config.game,count,1) as pool:
        reference=pool.collect(policy,config,env_seeds,action_seeds,trace=True)
    with EpisodeBatch(config.game,count,4) as pool:
        parallel=pool.collect(policy,config,env_seeds,action_seeds,trace=True)
        repeated=pool.collect(policy,config,env_seeds[:3],action_seeds[:3],trace=True)
    for a,b in zip(reference,parallel):
        for key in ('states','actions','rewards','weights'):
            np.testing.assert_array_equal(a[key],b[key])
        assert (a['score'],a['length'],a['terminated'],a['truncated'])==(
            b['score'],b['length'],b['terminated'],b['truncated'])
    for a,b in zip(reference[:3],repeated):
        np.testing.assert_array_equal(a['rewards'],b['rewards'])
        np.testing.assert_array_equal(a['states'],b['states'])
    # Confirm actor update and baseline subtraction remain identical across workers.
    one,four=copy.deepcopy(policy),copy.deepcopy(policy)
    for batch in (reference,parallel):
        for e in batch:e['weights']=p2.baseline_weights(e['rewards'],config.gamma,2,config.reward_scale)
    o1=torch.optim.Adam(one.parameters(),lr=.01)
    o4=torch.optim.Adam(four.parameters(),lr=.01)
    p1.update(one,o1,reference,config,'cpu')
    p1.update(four,o4,parallel,config,'cpu')
    error=max(float((one.state_dict()[k]-four.state_dict()[k]).abs().max()) for k in one.state_dict())
    assert error==0
    result={'game':config.game,'complete_episodes_compared':count,
            'one_worker_four_worker_trajectories_identical':True,
            'reset_and_partial_batch_episodes_compared':3,'actor_update_max_difference':error,
            'versions':p1.versions(),'benchmarks':[]}
    # Benchmark the same frozen policy, episode seeds, and per-episode RNG streams.
    target=args.episodes
    for mode in ('sequential_cpu','sequential_device','batched_4','batched_8'):
        device='cpu' if mode=='sequential_cpu' else args.device
        model=copy.deepcopy(policy).to(device)
        startup=time.monotonic()
        pool=None
        env=None
        if mode.startswith('batched'):
            pool=EpisodeBatch(config.game,count,int(mode.split('_')[1]))
            pool.collect(model,config,list(range(6000,6016)),list(range(7000,7016)),device)
        else:
            env=p1.make_env(config)
            p1.rollout(env,model,config,6000,np.random.default_rng(7000),device)
        startup=time.monotonic()-startup
        start=time.monotonic()
        episodes=[]
        try:
            for first in range(0,target,count if pool else 1):
                size=min(count,target-first) if pool else 1
                seeds=list(range(1_350_000_000+first,1_350_000_000+first+size))
                actions=list(range(90000+first,90000+first+size))
                if pool:episodes.extend(pool.collect(model,config,seeds,actions,device))
                else:episodes.append(p1.rollout(env,model,config,seeds[0],np.random.default_rng(actions[0]),device))
            elapsed=time.monotonic()-start
        finally:
            if pool:pool.close()
            if env:env.close()
        row={'mode':mode,'device':device,'episodes':len(episodes),
             'startup_and_warmup_seconds':startup,'measured_seconds':elapsed,
             'decisions':sum(e['length'] for e in episodes),
             'decisions_per_second':sum(e['length'] for e in episodes)/elapsed,
             'episode_scores':[e['score'] for e in episodes],
             'episode_lengths':[e['length'] for e in episodes]}
        result['benchmarks'].append(row)
        print(json.dumps({k:v for k,v in row.items() if not k.startswith('episode_')},indent=2),flush=True)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',output,flush=True)


if __name__=='__main__':main()
