"""Pong observation ablations; the legacy diff/MLP policy remains unchanged."""
import math
import torch
from torch import nn


class VisionPolicy(nn.Module):
    def __init__(self,config,feature_mode='current_diff',network='mlp'):
        super().__init__()
        channels=1 if feature_mode=='diff' else 2
        self.network=network
        self.channels=channels
        if network=='mlp':
            self.net=nn.Sequential(nn.Linear(channels*6400,config.hidden),nn.ReLU(),
                                   nn.Linear(config.hidden,2))
            nn.init.orthogonal_(self.net[0].weight)
            nn.init.zeros_(self.net[0].bias)
            nn.init.orthogonal_(self.net[2].weight,gain=.01)
            nn.init.zeros_(self.net[2].bias)
        else:
            self.encoder=nn.Sequential(nn.Conv2d(channels,16,5,stride=2),nn.ReLU(),
                                       nn.Conv2d(16,32,5,stride=2),nn.ReLU(),nn.Flatten())
            self.head=nn.Sequential(nn.Linear(32*17*17,config.hidden),nn.ReLU(),nn.Linear(config.hidden,2))
            for layer in list(self.encoder)+list(self.head):
                if isinstance(layer,(nn.Conv2d,nn.Linear)):
                    nn.init.orthogonal_(layer.weight,gain=math.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.head[-1].weight,gain=.01)

    def forward(self,x):
        single=x.ndim==1
        if single:x=x.unsqueeze(0)
        result=self.net(x) if self.network=='mlp' else self.head(self.encoder(x.reshape(-1,self.channels,80,80)))
        return result.squeeze(0) if single else result


def make_policy(config,feature_mode='diff',network='mlp'):
    if feature_mode=='diff' and network=='mlp':
        from q2_policy_gradient import Policy
        return Policy(config)
    if config.game!='pong':raise ValueError('Vision variants apply only to Pong')
    if feature_mode not in ('diff','current_diff') or network not in ('mlp','cnn'):
        raise ValueError('Invalid observation/network variant')
    return VisionPolicy(config,feature_mode,network)
