"""Time-bounded PPO follow-up; distinct from required REINFORCE experiments.

Uses Stable Baselines3 PPO. Pong inputs are object centroids and motion extracted
from assignment-preprocessed frames, never emulator RAM or a scripted action.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


class PongPixels(gym.Wrapper):
    def __init__(self):
        import ale_py
        gym.register_envs(ale_py)
        super().__init__(gym.make('ALE/Pong-v5',frameskip=4,
            repeat_action_probability=.25,full_action_space=False,obs_type='rgb'))
        assert self.env.unwrapped.get_action_meanings()[2:4]==['RIGHT','LEFT']
        self.action_space=gym.spaces.Discrete(2)
        self.observation_space=gym.spaces.Box(-2.,2.,shape=(12,),dtype=np.float32)
        self.previous=None

    def features(self,frame):
        image=frame[35:195:2,::2,0]
        binary=(image!=0)&(image!=144)&(image!=109)
        right=np.nonzero(binary[:,70:73])[0]
        left=np.nonzero(binary[:,8:10])[0]
        ball_y,ball_x=np.nonzero(binary[:,10:70])
        visible=bool(len(ball_y))
        py=float(right.mean()) if len(right) else 40.
        oy=float(left.mean()) if len(left) else 40.
        bx=float(ball_x.mean()+10) if visible else (self.previous[0] if self.previous else 40.)
        by=float(ball_y.mean()) if visible else (self.previous[1] if self.previous else 40.)
        dx=dy=dp=0.
        if self.previous:
            dp=py-self.previous[2]
            if visible and self.previous[3] and abs(bx-self.previous[0])<12:
                dx=bx-self.previous[0];dy=by-self.previous[1]
        previous_y=self.previous[1] if self.previous else by
        self.previous=(bx,by,py,visible)
        return np.array([(bx-40)/40,(by-40)/40,(py-40)/40,(oy-40)/40,
            np.clip(dx/5,-2,2),np.clip(dy/5,-2,2),(by-py)/80,
            np.clip(dp/5,-2,2),(previous_y-40)/40,float(visible),
            (70-bx)/80,(py-oy)/80],dtype=np.float32)

    def reset(self,**kwargs):
        self.previous=None
        frame,info=self.env.reset(**kwargs)
        return self.features(frame),info

    def step(self,action):
        frame,reward,terminated,truncated,info=self.env.step(int(action)+2)
        return self.features(frame),reward,terminated,truncated,info


class SeedBank(gym.Wrapper):
    def __init__(self,env,base,slot,stride):
        super().__init__(env)
        self.next_seed=base+slot
        self.stride=stride

    def reset(self,**kwargs):
        self.current_seed=self.next_seed
        self.next_seed+=self.stride
        kwargs['seed']=self.current_seed
        return self.env.reset(**kwargs)

    def step(self,action):
        observation,reward,terminated,truncated,info=self.env.step(action)
        info.update(reset_seed=self.current_seed,terminated=terminated,truncated=truncated)
        return observation,reward,terminated,truncated,info


class PongPixelsLead(PongPixels):
    """Add an image-velocity-based wall-reflection feature, not an action rule."""
    def __init__(self):
        super().__init__()
        self.observation_space=gym.spaces.Box(-2.,2.,shape=(14,),dtype=np.float32)

    def features(self,frame):
        features=super().features(frame)
        bx,by,py,visible=self.previous
        dx,dy=features[4]*5,features[5]*5
        target=by
        if visible and dx>0.01:
            unfolded=by+dy*(70-bx)/dx
            phase=(unfolded-1)%156
            target=1+(phase if phase<=78 else 156-phase)
        return np.concatenate((features,np.array([(target-40)/40,(target-py)/80],dtype=np.float32)))


def make_environment(game,base,slot,stride):
    from stable_baselines3.common.monitor import Monitor
    env=(PongPixelsLead() if game=='pong_lead' else PongPixels()) if game.startswith('pong') else gym.make('CartPole-v1')
    return Monitor(SeedBank(env,base,slot,stride))


def vector(game,base,count):
    from functools import partial
    from stable_baselines3.common.vec_env import SubprocVecEnv
    return SubprocVecEnv([partial(make_environment,game,base,i,count) for i in range(count)],start_method='spawn')


def evaluate(model,game,root,deadline,count=500,base=1960000000,workers=16):
    assert model.device.type=='cuda' and torch.cuda.device_count()==1
    env=vector(game,base,workers)
    torch.manual_seed(base)
    rows={}
    last_printed=-1
    obs=env.reset()
    try:
        while len(rows)<count and time.time()<deadline:
            action,_=model.predict(obs,deterministic=False)
            obs,reward,done,infos=env.step(action)
            for ended,info in zip(done,infos):
                if ended and base<=info['reset_seed']<base+count:
                    seed=info['reset_seed']
                    assert seed not in rows
                    rows[seed]={'episode':seed-base+1,'seed':seed,'score':info['episode']['r'],
                        'length':info['episode']['l'],'terminated':info['terminated'],'truncated':info['truncated']}
            milestone=len(rows)//50
            if milestone>last_printed:
                last_printed=milestone
                print(f'evaluation {game} {len(rows)}/{count}',flush=True)
    finally:env.close()
    ordered=[rows[k] for k in sorted(rows)]
    write_csv(root/'evaluation/episodes.csv',ordered)
    result={'complete':len(rows)==count,'episodes':len(rows),'requested_episodes':count,
        'evaluation_seed_base':base,'policy':'frozen stochastic PPO policy',
        'seed_protocol':'fixed round-robin bank; wait for every requested seed; ignore extra games',
        'model_sha256':hashlib.sha256((root/'selected_model.zip').read_bytes()).hexdigest(),
        'device':str(model.device),'visible_gpu_count':torch.cuda.device_count(),
        'gpu_name':torch.cuda.get_device_name(0),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'all_policy_parameters_on_cuda':all(p.device.type=='cuda' for p in model.policy.parameters())}
    if len(rows)==count:
        scores=np.array([r['score'] for r in ordered])
        mean,sd=float(scores.mean()),float(scores.std(ddof=1))
        result.update(mean=mean,sample_standard_deviation=sd,min=float(scores.min()),max=float(scores.max()),
            mean_approx_95_ci=[mean-1.96*sd/np.sqrt(count),mean+1.96*sd/np.sqrt(count)],
            wins=int((scores>0).sum()) if game.startswith('pong') else None,
            reached_500=int((scores==500).sum()) if game=='cartpole' else None,
            terminated_episodes=sum(r['terminated'] for r in ordered),truncated_episodes=sum(r['truncated'] for r in ordered))
        if sd==0:
            result['mean_approx_95_ci']=None
            result['interval_caveat']='Zero empirical variance does not imply zero population uncertainty.'
            if game=='cartpole' and bool((scores==500).all()):
                result['cap_success_probability_one_sided_95_lower_bound']=float(.05**(1/count))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
        ax.hist(scores,bins=np.arange(-21.5,22.5) if game.startswith('pong') else np.linspace(.5,500.5,26),color='#488c78',edgecolor='white')
        ax.axvline(mean,color='#803b7b',linestyle='--',label=f'Mean {mean:.3f}; sample SD {sd:.3f}')
        ax.set(xlabel='Undiscounted episode score',ylabel='Episodes',title=f'{game}: PPO follow-up, 500 frozen stochastic games')
        ax.legend();fig.savefig(root/'evaluation/reward_histogram.png');plt.close(fig)
    (root/'evaluation/evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def train(args):
    from stable_baselines3 import PPO,__version__
    from stable_baselines3.common.callbacks import BaseCallback
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():raise FileExistsError('Use a fresh experiment directory')
    train_end=datetime.fromisoformat(args.train_until).timestamp()
    final_end=datetime.fromisoformat(args.finish_until).timestamp()
    gamma=.95 if args.game=='cartpole' else .99
    plan=vars(args)|{'algorithm':'Stable Baselines3 PPO-Clip; GAE and normalized advantages',
        'assignment_status':'supplemental algorithm; not basic REINFORCE',
        'gamma':gamma,'sb3_version':__version__,'torch_version':torch.__version__,
        'gymnasium_version':gym.__version__,'numpy_version':np.__version__,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'device':args.device,'visible_gpu_count':torch.cuda.device_count(),
        'gpu_name':torch.cuda.get_device_name(0),'cuda_runtime':torch.version.cuda,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'selection_rule':'best completed trailing 100 training episodes, no test use',
        'evaluation_games':500,'evaluation_seed_base':1960000000,
        'observation':('14 image-derived motion and kinematic features' if args.game=='pong_lead' else '12 image-derived centroid/motion features') if args.game.startswith('pong') else 'official four-state observation'}
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[];window=deque(maxlen=100)
    class BudgetCallback(BaseCallback):
        def __init__(self):super().__init__();self.best=-float('inf')
        def _on_training_start(self):
            assert self.model.device.type=='cuda'
            assert all(p.device.type=='cuda' for p in self.model.policy.parameters())
            print(f'Training on {self.model.device}; visible GPUs={torch.cuda.device_count()}; GPU={torch.cuda.get_device_name(0)}',flush=True)
        def _on_step(self):
            for info in self.locals['infos']:
                if 'episode' in info:
                    row={'episode':len(rows)+1,'seed':info['reset_seed'],'score':info['episode']['r'],
                        'length':info['episode']['l'],'through_timesteps':self.num_timesteps}
                    rows.append(row);window.append(row['score'])
            if len(rows)>=100 and self.num_timesteps%args.workers==0:
                mean=float(np.mean(window))
                if mean>self.best:
                    self.best=mean;self.model.save(root/'selected_model.zip')
                    write_csv(root/'training.csv',rows)
                    (root/'selection.json').write_text(json.dumps({'training_episodes_observed':len(rows),
                        'training_timesteps':self.num_timesteps,'last100_training_score_mean':mean,
                        'selected_using':'training episodes only'},indent=2)+'\n')
            if self.n_calls%256==0:
                write_csv(root/'training.csv',rows)
                print(f'{args.game} steps={self.num_timesteps} episodes={len(rows)} mean100={np.mean(window) if window else 0:.2f} best={self.best:.2f}',flush=True)
            return time.time()<train_end
    env=vector(args.game,args.seed,args.workers)
    try:
        model=PPO('MlpPolicy',env,learning_rate=args.lr,n_steps=128,
            batch_size=256,n_epochs=4 if args.game.startswith('pong') else 10,
            gamma=gamma,gae_lambda=.95,clip_range=.2,ent_coef=.01 if args.game.startswith('pong') else 0.,
            vf_coef=.5,max_grad_norm=.5,policy_kwargs={'net_arch':dict(pi=[64,64],vf=[64,64])},
            seed=args.seed,device=args.device,verbose=0)
        model.learn(total_timesteps=args.steps,callback=BudgetCallback())
        model.save(root/'final_model.zip')
        if not (root/'selected_model.zip').exists():model.save(root/'selected_model.zip')
    finally:env.close()
    write_csv(root/'training.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores=np.array([r['score'] for r in rows])
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    ax.plot(np.arange(1,len(scores)+1),scores,alpha=.3,color='#6eae9b',label='Episode score')
    if len(scores)>=100:ax.plot(np.arange(100,len(scores)+1),np.convolve(scores,np.ones(100)/100,'valid'),color='#803b7b',label='Trailing mean, 100 episodes')
    ax.set(xlabel='Completed training episode',ylabel='Undiscounted episode score',title=f'{args.game}: supplemental PPO training')
    ax.legend();fig.savefig(root/'training_rewards.png');plt.close(fig)
    if not args.training_only:
        selected=PPO.load(root/'selected_model.zip',device=args.device)
        evaluate(selected,args.game,root,final_end,workers=args.workers)
    (root/'complete.json').write_text(json.dumps({'finished_at':datetime.now(timezone.utc).isoformat(),
        'training_timesteps':model.num_timesteps,'training_episodes':len(rows),'new_runs_queued':False,
        'training_device':str(model.device),'evaluation_device':args.device,
        'all_policy_parameters_on_cuda':all(p.device.type=='cuda' for p in model.policy.parameters()),
        'optimizer_state_tensor_devices':sorted({str(v.device) for state in model.policy.optimizer.state.values() for k,v in state.items() if torch.is_tensor(v) and k!='step'}),
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game',choices=['cartpole','pong','pong_lead'],required=True)
    parser.add_argument('--training-only',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--train-until',required=True)
    parser.add_argument('--finish-until',required=True)
    parser.add_argument('--steps',type=int,default=10000000)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--seed',type=int,default=3817)
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--device',choices=['cuda'],default='cuda')
    torch.set_num_threads(1)
    args=parser.parse_args()
    if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('Expose exactly one assigned GPU with CUDA_VISIBLE_DEVICES')
    train(args)
