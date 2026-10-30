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
    cp,pong=records['cartpole'],records['pong']
    supplement=f'''# Supplement: Single-GPU PPO Performance Study

The required experiments above use the requested basic policy-gradient
algorithm. This additional study tests PPO-Clip with a learned state-value
baseline, based on the [original PPO paper](https://arxiv.org/abs/1707.06347)
and [OpenAI's explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
It is a different algorithm and is not substituted for the required comparison.
Pong uses engineered image features rather than directly feeding binary frames
to the neural network; [object-centric Atari research](https://arxiv.org/abs/2306.08649)
motivates this representation. No emulator RAM, downloaded trained model,
scripted action, shaped reward, extra frame skip, or Pong point reset is used.

## Single-GPU execution and method

Fresh training and frozen evaluation both ran on one NVIDIA RTX PRO 5000
Blackwell GPU on Deepdish. Both trials selected the same physical GPU using
CUDA_VISIBLE_DEVICES. The runner rejects anything other than exactly one visible
GPU. Saved metadata verifies that all actor/value parameters and the optimizer's
moment tensors are on CUDA; evaluation also records CUDA parameters. CPU workers
simulate the official environments and prepare observations for the GPU.

Stable Baselines3 2.9.2 PPO uses separate actor and value networks, each with two
64-unit tanh layers, Adam 0.0003, 128-step rollouts per environment, minibatches
of 256, GAE lambda 0.95, clip range 0.2, value coefficient 0.5, and gradient-norm
limit 0.5. CartPole uses eight workers, ten optimization epochs and no entropy
bonus. Pong uses sixteen workers, four epochs and entropy coefficient 0.01.
The assigned discounts remain 0.95 and 0.99. Official environment versions,
CartPole's 500-step limit, and Pong actions 2/3, frame skip 4 and sticky probability
0.25 are preserved. The
[SB3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
describes the maintained PPO implementation and its normalized advantages.

Pong's 12 inputs are image-derived ball and paddle positions, motion, relative
positions, previous ball height and a visibility flag. Missing balls retain the
last position with a missing flag; reset clears motion. These features do not
prescribe the action. The feature validator checks synthetic centroids and
motion, missing objects, unchanged input frames, and resets. Four actual games
(4,135 transitions) exactly match official rewards and endings under matched
actions. The kinematic-input option was validated but not used in these trials.

## Frozen-policy results

Checkpoint selection uses only the best trailing mean over 100 completed
training episodes. The selected CartPole checkpoint was saved after observing
{cp['selected_at_observed_environment_steps']:,} steps and {cp['selected_at_completed_training_episode']:,} completed episodes;
Pong after {pong['selected_at_observed_environment_steps']:,} steps and {pong['selected_at_completed_training_episode']:,} episodes. Counts can include
an in-progress rollout; optimizer updates use completed rollout batches.
Screening continues after selection, as shown in the plots. No final test games
are used to choose checkpoints.

| GPU-trained stochastic policy | Games | Mean score | Sample SD | Outcome |
|---|---:|---:|---:|---|
| CartPole PPO | 500 | {cp['mean']:.3f} | {cp['sample_sd']:.4f} | {cp['reached_500']}/500 at the cap |
| Pong PPO | 500 | {pong['mean']:.3f} | {pong['sample_sd']:.4f} | {pong['wins']}/500 positive-score games |

Each test requests exactly 500 distinct seeds beginning at 1,960,000,000,
disjoint from training. All requested seeds must finish; fast extra autoreset
games are ignored rather than substituted for slow games. Parameters stay frozen
and actions are sampled stochastically. The source snapshot, evaluated model
SHA-256, CUDA provenance, seed bank, training-only selection and raw score
statistics pass the independent saved-artifact audit.

![Single-GPU PPO CartPole training](results/ppo_gpu/cartpole/training_rewards.png)

**Figure S1.** Raw training scores and exact trailing 100-episode mean. The
vertical line marks the selected checkpoint; later screening scores are retained.

![Single-GPU PPO CartPole evaluation](results/ppo_gpu/cartpole/evaluation/reward_histogram.png)

**Figure S2.** Exactly 500 frozen stochastic games. Reaching 500 is the official
time-limit cap, not an observed physical failure. Zero empirical SD, if observed,
does not imply zero population uncertainty.

![Single-GPU PPO Pong training](results/ppo_gpu/pong/training_rewards.png)

**Figure S3.** Raw full-game scores and trailing mean. A training window mixes
changing policies and is not an estimate of one frozen checkpoint's performance.

![Single-GPU PPO Pong evaluation](results/ppo_gpu/pong/evaluation/reward_histogram.png)

**Figure S4.** Exactly 500 reserved full games. Natural positive-score endings,
truncations, and raw scores are retained separately in the audit and episode CSV.

## Interpretation and reproduction

This study changes the baseline, estimator, optimization, network and Pong
representation. Score differences cannot be attributed to the baseline alone.
The results are conditional on one training seed, with different budgets and
evaluation banks from the original experiments. Pong should not be described as
solved based on a small number of wins. For CartPole, if all 500 games reach the
cap, the exact one-sided 95% lower bound on its cap probability is 0.9940 under
independent trials; the observed zero SD does not prove universal reliability.

Install requirements-ppo.txt in a separate environment with CUDA-enabled PyTorch.
Expose your one assigned GPU and set CUBLAS_WORKSPACE_CONFIG=:4096:8. For a fresh
run, replace both ISO timestamps below with future UTC deadlines, reserving time
for all 500 final games:

```bash
python ppo_gpu.py --game cartpole --device cuda --workers 8 --steps 600000 \\
  --train-until "YYYY-MM-DDTHH:MM:SS+00:00" \\
  --finish-until "YYYY-MM-DDTHH:MM:SS+00:00" \\
  --output results/ppo_gpu/reproduction_cartpole
```

Use --game pong --workers 16 and a fresh output directory for Pong. Run
`python validate_ppo_gpu.py` for feature/environment validation and
`python audit_ppo_gpu.py` to recompute the saved statistics, CUDA/source/model
checks, training plots and this supplement. The full environment lock, plans,
source snapshots, selected models and raw data are under results/ppo_gpu.
'''
    (root/'report_section.md').write_text(supplement)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
