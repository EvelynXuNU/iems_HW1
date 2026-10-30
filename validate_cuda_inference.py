"""Compare graphed probabilities before/after parameter updates and benchmark."""
import argparse
import json
import time
from pathlib import Path


def main():
    import numpy as np
    import torch
    from q2_policy_gradient import Config,seed_everything
    from vision_policy import make_policy
    from cuda_inference import CudaPolicyInference
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    seed_everything(3817)
    c=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    variant=c['fast_collector']
    model=make_policy(Config(**c['config']),variant['feature_mode'],variant['network']).cuda()
    model.load_state_dict(c['policy'])
    # Keep the diagnostic away from saturated logits, so stale parameter reads
    # cannot pass merely because both outputs round to probabilities zero/one.
    with torch.no_grad():
        model.head[-1].weight.mul_(.01)
        model.head[-1].bias.zero_()
    states=np.random.default_rng(7).integers(-1,2,size=(16,12800),dtype=np.int8)
    graph=CudaPolicyInference(model,16,12800)
    errors=[]
    before=graph(states).copy()
    with torch.no_grad():
        for changed in [False,True]:
            if changed:
                model.head[-1].bias[0].add_(.25)
            for n in [16,7,1]:
                reference=model(torch.tensor(states[:n],dtype=torch.float32,device='cuda')).softmax(-1).cpu().numpy()
                observed=graph(states[:n])
                error=float(np.max(np.abs(reference-observed)));errors.append(error)
                assert error<2e-5,error
        change=float(np.max(np.abs(graph(states)-before)))
        assert change>.001,'Parameter-update test did not change probabilities meaningfully'
        timings={}
        for method in ['eager','graph']:
            start=time.perf_counter()
            for _ in range(200):
                if method=='graph':graph(states)
                else:model(torch.tensor(states,dtype=torch.float32,device='cuda')).softmax(-1).cpu().numpy()
            timings[method]=time.perf_counter()-start
    result={'max_probability_error':max(errors),'partial_batches':[16,7,1],
      'parameter_updates_read_correctly':True,'iterations':200,'seconds':timings,
      'probability_change_after_parameter_update':change,
      'inference_speedup':timings['eager']/timings['graph'],
      'scope':'inference-only benchmark; other same-GPU pilots may be active'}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
