"""Same REINFORCE updates with batched CPU environments and one policy GPU.

Training always finishes exactly the requested episode batch before updating.
Supports explicit continuation of legacy checkpoints, recording the collector
and RNG protocol change. Evaluations collect scores without training-state arrays.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path


def train(config,part,episodes,output,device='cuda',workers=4,resume=None,save_every=4,
          feature_mode='diff',network='mlp',baseline_kind='ema',cuda_graph=False):
    import numpy as np
    import torch
    import q2_policy_gradient as p1
    import q3_policy_gradient as p2
    from batched_rollout import EpisodeBatch
    from vision_policy import make_policy
    if episodes<1 or save_every<1:raise ValueError('Positive episode count and save interval required')
    p1.seed_everything(config.seed)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    policy=make_policy(config,feature_mode,network).to(device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=config.learning_rate)
    baseline=p2.Baseline()
    if part==1 and baseline_kind=='state':raise ValueError('Part 1 cannot use a baseline')
    critic=None
    if part==2 and baseline_kind=='state':
        from learned_baseline import LearnedBaseline
        critic=LearnedBaseline(config,feature_mode,device)
    rng=np.random.default_rng(config.seed)
    history,updates,elapsed=[],[],0.
    method=('REINFORCE with lagged learned state baseline' if critic else p2.METHOD) if part==2 else 'REINFORCE without baseline'
    resume_digest=None
    if resume:
        resume_digest=hashlib.sha256(Path(resume).read_bytes()).hexdigest()
        c=torch.load(resume,map_location='cpu',weights_only=False)
        if c['config']!=asdict(config):raise ValueError('Resume actor configuration mismatch')
        checkpoint_part=c.get('part',2 if ('baseline_state' in c or 'critic_state' in c) else 1)
        if checkpoint_part!=part:raise ValueError('Resume Part 1/Part 2 mismatch')
        if c.get('baseline_kind','ema')!=baseline_kind:raise ValueError('Resume baseline kind mismatch')
        previous=c.get('fast_collector',{})
        if previous.get('feature_mode','diff')!=feature_mode or previous.get('network','mlp')!=network:
            raise ValueError('Resume vision variant mismatch')
        policy.load_state_dict(c['policy'])
        optimizer.load_state_dict(c['optimizer'])
        rng.bit_generator.state=c['action_rng']
        history,updates,elapsed=c['history'],c['updates'],c['elapsed_seconds']
        if critic:critic.load_state_dict(c['critic_state'])
        elif part==2:baseline=p2.Baseline(**c['baseline_state'])
    elif (output/'checkpoint.pt').exists():raise FileExistsError('Use --resume or a fresh run directory')
    if len(history)>=episodes:raise ValueError('Target total must exceed completed episodes')
    start_episode=len(history)
    metadata={'method':method,'config':asdict(config),'env_id':config.env_id,'gamma':config.gamma,
      'versions':p1.versions(),'device':device,'gpu_name':torch.cuda.get_device_name(0) if device=='cuda' else None,
      'cuda_runtime':torch.version.cuda if device=='cuda' else None,'target_episodes':episodes,
      'collector':'complete batched episodes, no autoreset, no replacement', 'workers':workers,
      'feature_mode':feature_mode,'network':network,
      'cuda_graph_inference':cuda_graph,
      'action_rng_protocol':'one uint63 seed drawn per episode from saved batch generator',
      'checkpoint_interval_batches':save_every,'resume_episode':start_episode,
      'resume_elapsed_seconds':elapsed,'resume_sha256':resume_digest,
      'baseline_kind':baseline_kind if part==2 else 'none',
      'baseline_decay':baseline.decay if part==2 and not critic else None,
      'critic_target':'Monte Carlo G_t, discounted occupancy weighting; fit only after actor' if critic else None,
      'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       map(Path,[__file__,p1.__file__,p2.__file__,'batched_rollout.py','vision_policy.py']+
                           (['learned_baseline.py'] if critic else [])+
                           (['cuda_inference.py'] if cuda_graph else []))}}
    if (output/'run.json').exists() and not (output/'run_before_fast_collector.json').exists():
        (output/'run_before_fast_collector.json').write_bytes((output/'run.json').read_bytes())
    (output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    start=time.monotonic()
    with EpisodeBatch(config.game,config.batch_episodes,min(workers,config.batch_episodes),feature_mode,cuda_graph) as pool:
        while len(history)<episodes:
            first=len(history)
            size=min(config.batch_episodes,episodes-first)
            action_seeds=rng.integers(0,2**63,size=size,dtype=np.uint64)
            batch=pool.collect(policy,config,list(range(config.seed+first,config.seed+first+size)),
                               action_seeds,device,trace=first in (start_episode,start_episode+config.batch_episodes))
            b=baseline.value if part==2 and not critic else 0.
            for i,e in enumerate(batch):
                if critic:
                    e['baseline_predictions']=critic.predict(e['states'])
                    e['weights']=config.gamma**np.arange(e['length'])*(
                        p1.discounted_returns(e['rewards'],config.gamma)-e['baseline_predictions'])/config.reward_scale
                elif part==2:e['weights']=p2.baseline_weights(e['rewards'],config.gamma,b,config.reward_scale)
                for row in e['trace']:
                    value=float(e['baseline_predictions'][row['step']]) if critic else b
                    row['baseline_used']=value
                    row['advantage']=row['reward_to_go']-value
                    row['gradient_weight']=float(e['weights'][row['step']])
                if e['trace']:
                    (output/f'first_10_fast_steps_batch_{first}.json').write_text(json.dumps(e['trace'],indent=2)+'\n')
                record={'episode':first+i+1,'seed':config.seed+first+i,
                  **{k:e[k] for k in ('score','discounted_return','length','terminated','truncated')}}
                if critic:record['baseline_prediction_mean']=float(e['baseline_predictions'].mean())
                elif part==2:record['baseline_used']=b
                history.append(record)
            diagnostic=p1.update(policy,optimizer,batch,config,device)
            critic_diagnostic=critic.fit_after_actor(batch,config.gamma) if critic else {}
            if part==2 and not critic:baseline.observe([e['discounted_return'] for e in batch])
            record={'update':len(updates)+1,'through_episode':len(history),'batch_episodes':size,
                    **diagnostic}
            if critic:record.update(critic_diagnostic)
            elif part==2:record.update({'baseline_used':b,'baseline_for_next_batch':baseline.value,
                'advantage_mean_diagnostic_only':float(np.concatenate([
                    p1.discounted_returns(e['rewards'],config.gamma)-b for e in batch]).mean())})
            # Match existing diagnostic column order when continuing legacy runs.
            if updates:record={k:record[k] for k in updates[0]}
            updates.append(record)
            if len(updates)%save_every==0 or len(history)==episodes:
                payload={'method':method,'part':part,'baseline_kind':baseline_kind,
                  'config':asdict(config),'policy':policy.state_dict(),
                  'optimizer':optimizer.state_dict(),'history':history,'updates':updates,
                  'action_rng':rng.bit_generator.state,'elapsed_seconds':elapsed+time.monotonic()-start,
                  'fast_collector':metadata}
                if critic:payload['critic_state']=critic.state_dict()
                elif part==2:payload['baseline_state']=asdict(baseline)
                p1.save_checkpoint(output/'checkpoint.pt',payload)
                p1.write_csv(output/'training.csv',history)
                p1.write_csv(output/'updates.csv',updates)
            if len(updates)%10==0 or len(history)==episodes:
                print(f'Part {part} {config.game} episode={len(history)} mean_last_100='
                      f'{np.mean([r["score"] for r in history[-100:]]):.2f} '
                      f'baseline={b:.4f} entropy={diagnostic["entropy"]:.3f}',flush=True)
    if critic:
        plt=p1.plotting()
        scores=np.array([r['score'] for r in history])
        fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
        ax.plot(np.arange(1,len(scores)+1),scores,alpha=.3,color='#6eae9b',label='Episode score')
        if len(scores)>=100:ax.plot(np.arange(100,len(scores)+1),np.convolve(scores,np.ones(100)/100,'valid'),color='#803b7b',label='Trailing mean, 100 episodes')
        ax.set(xlabel='Training episode',ylabel='Undiscounted episode reward (score)',title=f'{config.env_id}: learned state baseline')
        ax.legend();fig.savefig(output/'training_rewards.png');plt.close(fig)
    else:(p2.plot_training if part==2 else p1.plot_training)(history,output,config)


def evaluate(checkpoint,output,episodes=500,seed=1_500_000_000,device='cuda',workers=4):
    import numpy as np
    import torch
    import q2_policy_gradient as p1
    import q3_policy_gradient as p2
    from batched_rollout import EpisodeBatch
    from vision_policy import make_policy
    if episodes<2:raise ValueError('At least two episodes needed for sample SD')
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=p1.Config(**c['config'])
    part=c.get('part',2 if ('baseline_state' in c or 'critic_state' in c) else 1)
    if max(seed,config.seed)<min(seed+episodes,config.seed+len(c['history'])):
        raise ValueError('Evaluation and training reset seeds overlap')
    p1.seed_everything(seed)
    variant=c.get('fast_collector',{})
    feature_mode,network=variant.get('feature_mode','diff'),variant.get('network','mlp')
    policy=make_policy(config,feature_mode,network).to(device)
    policy.load_state_dict(c['policy'])
    policy.eval()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'evaluation.json').exists():raise FileExistsError('Preserve existing evaluation; use a fresh output directory')
    rows=[]
    with EpisodeBatch(config.game,16,min(workers,16),feature_mode) as pool:
        for first in range(0,episodes,16):
            size=min(16,episodes-first)
            action_seeds=[int(np.random.SeedSequence([seed,i,42]).generate_state(1,dtype=np.uint64)[0])
                          for i in range(first,first+size)]
            batch=pool.collect(policy,config,list(range(seed+first,seed+first+size)),action_seeds,
                               device,store_states=False,trace=first==0)
            for i,e in enumerate(batch):
                rows.append({'episode':first+i+1,'seed':seed+first+i,
                             **{k:e[k] for k in ('score','discounted_return','length','terminated','truncated')}})
                if e['trace']:(output/'first_10_steps.json').write_text(json.dumps(e['trace'],indent=2)+'\n')
            p1.write_csv(output/'episodes.csv',rows)
            print(f'Part {part} evaluation {len(rows)}/{episodes}',flush=True)
    scores=np.array([r['score'] for r in rows])
    mean,sd=float(scores.mean()),float(scores.std(ddof=1))
    result={'env_id':config.env_id,'episodes':episodes,'mean':mean,'sample_standard_deviation':sd,
      'population_standard_deviation':float(scores.std()),
      'mean_approx_95_ci':[mean-1.96*sd/np.sqrt(episodes),mean+1.96*sd/np.sqrt(episodes)],
      'min':float(scores.min()),'max':float(scores.max()),
      'discounted_return_mean':float(np.mean([r['discounted_return'] for r in rows])),
      'terminated_episodes':sum(r['terminated'] for r in rows),'truncated_episodes':sum(r['truncated'] for r in rows),
      'win_fraction':float(np.mean(scores>0)) if config.game=='pong' else None,
      'cartpole_500_fraction':float(np.mean(scores==500)) if config.game=='cartpole' else None,
      'policy':'trained stochastic policy','training_episodes':len(c['history']),
      'training_seed':config.seed,'evaluation_seed':seed,'method':c.get('method',p2.METHOD if part==2 else 'REINFORCE without baseline'),
      'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
      'versions':p1.versions(),'collector':'batched score-only, no autoreset','workers':workers,
      'feature_mode':feature_mode,'network':network,
      'action_rng_protocol':'independent SeedSequence([evaluation_seed, episode_index, 42])'}
    plt=p1.plotting()
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    bins=np.arange(-21.5,22.5,1) if config.game=='pong' else np.linspace(.5,500.5,26)
    ax.hist(scores,bins=bins,color='#488c78' if part==2 else '#377c9b',edgecolor='white')
    ax.axvline(mean,color='#803b7b',linestyle='--',label=f'Mean {mean:.2f}; sample SD {sd:.2f}')
    ax.set(xlabel='Undiscounted episode reward (score)',ylabel='Number of episodes',
           title=f'{config.env_id}: Part {part}, {episodes} held-out episodes\n{result["method"]}')
    ax.legend()
    fig.savefig(output/'reward_histogram.png')
    plt.close(fig)
    # Completion marker comes after the required histogram has been saved.
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    import torch
    from q2_policy_gradient import Config
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    t.add_argument('--part',type=int,choices=[1,2],required=True)
    t.add_argument('--game',choices=['cartpole','pong'],default='pong')
    t.add_argument('--episodes',type=int,required=True)
    t.add_argument('--seed',type=int,default=3817)
    t.add_argument('--lr',type=float,default=.01)
    t.add_argument('--reward-scale',type=float,default=1)
    t.add_argument('--batch-episodes',type=int,default=16)
    t.add_argument('--hidden',type=int,default=64)
    t.add_argument('--feature-mode',choices=['diff','current_diff'],default='diff')
    t.add_argument('--network',choices=['mlp','cnn'],default='mlp')
    t.add_argument('--baseline',choices=['ema','state'],default='ema')
    t.add_argument('--cuda-graph',action='store_true')
    t.add_argument('--resume')
    t.add_argument('--save-every',type=int,default=4,help='checkpoint every N actor batches, plus final')
    e=sub.add_parser('evaluate')
    e.add_argument('--checkpoint',required=True)
    e.add_argument('--episodes',type=int,default=500)
    e.add_argument('--seed',type=int,default=1_500_000_000)
    for p in (t,e):
        p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        p.add_argument('--workers',type=int,default=4)
        p.add_argument('--output-dir',required=True)
    a=parser.parse_args()
    if a.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('Exactly one visible GPU required')
    if a.command=='train':
        config=Config(a.game,a.seed,a.lr,a.hidden,a.batch_episodes,a.reward_scale)
        train(config,a.part,a.episodes,a.output_dir,a.device,a.workers,a.resume,a.save_every,a.feature_mode,a.network,a.baseline,a.cuda_graph)
    else:evaluate(a.checkpoint,a.output_dir,a.episodes,a.seed,a.device,a.workers)


if __name__=='__main__':main()
