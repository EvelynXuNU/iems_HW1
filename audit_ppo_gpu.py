"""Audit single-GPU PPO artifacts and draw training curves; no new rollouts."""
import csv
import hashlib
import json
import statistics
from pathlib import Path


def read_rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('results/ppo_gpu')
    records={}
    for game in ('cartpole','pong'):
        folder=root/game
        plan=json.loads((folder/'plan.json').read_text())
        result=json.loads((folder/'evaluation/evaluation.json').read_text())
        complete=json.loads((folder/'complete.json').read_text())
        assert plan['device']=='cuda' and plan['visible_gpu_count']==1
        assert complete['training_device']=='cuda' or complete['training_device']=='cuda:0'
        assert complete['all_policy_parameters_on_cuda']
        assert complete['optimizer_state_tensor_devices']==['cuda:0']
        assert complete['cuda_peak_allocated_bytes']>0
        assert result['device'] in ('cuda','cuda:0') and result['visible_gpu_count']==1
        assert result['all_policy_parameters_on_cuda']
        assert plan['cuda_visible_devices']==result['cuda_visible_devices']
        assert result['complete'] and result['episodes']==500
        data=read_rows(folder/'evaluation/episodes.csv')
        assert len(data)==500
        assert {int(r['seed']) for r in data}==set(range(1960000000,1960000500))
        scores=[float(r['score']) for r in data]
        mean,sd=statistics.mean(scores),statistics.stdev(scores)
        assert abs(mean-result['mean'])<1e-10
        assert abs(sd-result['sample_standard_deviation'])<1e-10
        digest=hashlib.sha256((folder/'selected_model.zip').read_bytes()).hexdigest()
        assert digest==result['model_sha256']
        assert hashlib.sha256((root/'source/ppo_gpu.py').read_bytes()).hexdigest()==plan['source_sha256']
        selection=json.loads((folder/'selection.json').read_text())
        history=read_rows(folder/'training.csv')
        chosen=selection['training_episodes_observed']
        assert len(history)>=chosen>=100
        assert abs(statistics.mean(float(r['score']) for r in history[chosen-100:chosen])-
                   selection['last100_training_score_mean'])<1e-10
        assert not ({int(r['seed']) for r in data}&{int(r['seed']) for r in history})
        assert all(r['terminated']=='True' or r['truncated']=='True' for r in data)
        for key in ('terminated','truncated'):
            assert sum(r[key]=='True' for r in data)==result[key+'_episodes']
        record={'games':500,'mean':mean,'sample_sd':sd,'model_sha256':digest,
            'training_device':complete['training_device'],'evaluation_device':result['device'],
            'optimizer_state_devices':complete['optimizer_state_tensor_devices'],
            'gpu_uuid':plan['cuda_visible_devices'],'gpu_name':plan['gpu_name'],
            'selected_at_completed_training_episode':chosen,
            'selected_at_observed_environment_steps':selection['training_timesteps'],
            'training_only_selection_checked':True,'exact_reserved_seed_bank_checked':True}
        if game=='cartpole':
            assert result['reached_500']==sum(s==500 for s in scores)
            record['reached_500']=result['reached_500']
            if result['reached_500']==500:
                record['cap_probability_exact_one_sided_95_lower_bound']=.05**(1/500)
        else:
            assert result['wins']==sum(s>0 for s in scores)
            record['wins']=result['wins']
            record['naturally_terminated_positive_scores']=sum(float(r['score'])>0 and r['terminated']=='True' for r in data)
        records[game]=record
        training=np.array([float(r['score']) for r in history])
        fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
        ax.plot(np.arange(1,len(training)+1),training,color='#6eae9b',alpha=.35,label='Episode score')
        ax.plot(np.arange(100,len(training)+1),np.convolve(training,np.ones(100)/100,'valid'),color='#803b7b',label='Trailing mean, 100 episodes')
        ax.axvline(chosen,color='#333333',linestyle='--',label=f'Selected checkpoint: episode {chosen}')
        ax.set(xlabel='Completed training episode',ylabel='Undiscounted episode score',title=f'{game}: single-GPU PPO screening')
        ax.legend();fig.savefig(folder/'training_rewards.png');plt.close(fig)
    assert records['cartpole']['gpu_uuid']==records['pong']['gpu_uuid']
    evidence={'status':'passed','scope':'single-GPU provenance and saved-artifact audit',
        'same_physical_gpu_for_both_trials':True,'runs':records}
    (root/'completed_audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
