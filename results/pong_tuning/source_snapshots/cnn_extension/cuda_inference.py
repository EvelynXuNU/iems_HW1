"""Fixed-batch CUDA graph for feedforward policy inference only.

Parameters remain the live tensors updated by Adam; inactive rows are ignored.
No actor update or random-number sampling is captured.
"""
import torch
from torch import nn


class CudaPolicyInference:
    def __init__(self,policy,slots,dimension):
        if any(isinstance(m,(nn.modules.batchnorm._BatchNorm,nn.Dropout)) for m in policy.modules()):
            raise ValueError('Padded inference requires independent deterministic rows')
        self.input=torch.zeros((slots,dimension),dtype=torch.float32,device='cuda')
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):self.output=policy(self.input).softmax(-1)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(self.graph):
            self.output=policy(self.input).softmax(-1)

    @torch.no_grad()
    def __call__(self,states):
        count=len(states)
        self.input[:count].copy_(torch.as_tensor(states,dtype=torch.float32))
        self.graph.replay()
        return self.output[:count].cpu().numpy()
