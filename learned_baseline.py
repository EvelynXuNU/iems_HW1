"""Past-data state baseline for Monte Carlo REINFORCE (no TD bootstrap)."""
import numpy as np
import torch
from torch import nn


class LearnedBaseline:
    def __init__(self,config,feature_mode,device,learning_rate=.001,epochs=2):
        dimension=4 if config.game=='cartpole' else 6400*(2 if feature_mode=='current_diff' else 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed+100000)
            self.model=nn.Sequential(nn.Linear(dimension,64),
                nn.Tanh() if config.game=='cartpole' else nn.ReLU(),nn.Linear(64,1))
            nn.init.orthogonal_(self.model[0].weight)
            nn.init.zeros_(self.model[0].bias)
            nn.init.zeros_(self.model[-1].weight)
            nn.init.zeros_(self.model[-1].bias)
        self.model.to(device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        self.device,self.epochs,self.learning_rate=device,epochs,learning_rate
        self.warmstarted=False

    @torch.no_grad()
    def predict(self,states):
        return np.concatenate([self.model(torch.as_tensor(states[first:first+1024],
                    dtype=torch.float32,device=self.device)).squeeze(-1).cpu().numpy()
                    for first in range(0,len(states),1024)]).astype(np.float64)

    def fit_after_actor(self,episodes,gamma):
        from q2_policy_gradient import discounted_returns
        targets=[discounted_returns(e['rewards'],gamma) for e in episodes]
        weights=[gamma**np.arange(e['length']) for e in episodes]
        denominator=sum(w.sum() for w in weights)
        # This data-dependent initialization happens only AFTER the first actor step.
        if not self.warmstarted:
            mean=sum(float(w@g) for w,g in zip(weights,targets))/denominator
            with torch.no_grad():self.model[-1].bias.fill_(mean)
            self.warmstarted=True
        losses=[]
        norm=0.
        for _ in range(self.epochs):
            self.optimizer.zero_grad(set_to_none=True)
            total=torch.zeros((),dtype=torch.float64,device=self.device)
            for e,g,w in zip(episodes,targets,weights):
                for first in range(0,e['length'],1024):
                    prediction=self.model(torch.as_tensor(e['states'][first:first+1024],
                        dtype=torch.float32,device=self.device)).squeeze(-1)
                    target=torch.as_tensor(g[first:first+1024],dtype=torch.float32,device=self.device)
                    weight=torch.as_tensor(w[first:first+1024],dtype=torch.float64,device=self.device)
                    loss=(weight*(prediction-target).square()).sum()/denominator
                    loss.backward()
                    total+=loss.detach()
            norm=float(nn.utils.clip_grad_norm_(self.model.parameters(),5,error_if_nonfinite=True))
            self.optimizer.step()
            losses.append(float(total))
        return {'critic_loss':losses[-1],'critic_gradient_norm':norm}

    def state_dict(self):
        return {'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                'warmstarted':self.warmstarted,'epochs':self.epochs,'learning_rate':self.learning_rate}

    def load_state_dict(self,state):
        if state['epochs']!=self.epochs or state['learning_rate']!=self.learning_rate:
            raise ValueError('Critic settings changed across resume')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.warmstarted=state['warmstarted']
