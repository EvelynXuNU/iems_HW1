"""Recompute PPO performance evidence, plots and report text; no game rollouts."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('results/research20')
    records={}
    for name in ('cartpole','pong_selected'):
        folder=root/name
        result=json.loads((folder/'evaluation/evaluation.json').read_text())
        assert result['complete'] and result['episodes']==500
        data=rows(folder/'evaluation/episodes.csv')
        assert len(data)==500
        assert {int(r['seed']) for r in data}==set(range(1950000000,1950000500))
        scores=[float(r['score']) for r in data]
        mean,sd=statistics.mean(scores),statistics.stdev(scores)
        assert abs(mean-result['mean'])<1e-10
        assert abs(sd-result['sample_standard_deviation'])<1e-10
        digest=hashlib.sha256((folder/'selected_model.zip').read_bytes()).hexdigest()
        assert digest==result['model_sha256']
        selection=json.loads((folder/'selection.json').read_text())
        history=rows(folder/'training.csv')
        chosen=selection['training_episodes_observed']
        prefix=history[:chosen]
        assert len(prefix)==chosen
        close_mean=statistics.mean(float(r['score']) for r in prefix[-100:])
        assert abs(close_mean-selection['last100_training_score_mean'])<1e-10
        assert not ({int(r['seed']) for r in data}&{int(r['seed']) for r in history})
        assert all(r['terminated']=='True' or r['truncated']=='True' for r in data)
        for key in ('terminated','truncated'):
            assert sum(r[key]=='True' for r in data)==result[key+'_episodes']
        record={'evaluation_games':500,'mean':mean,'sample_sd':sd,
            'model_sha256':digest,'checkpoint_choice_recomputed_from_training_only':True,
            'chosen_at_completed_training_episode':chosen,
            'chosen_at_observed_environment_steps':selection['training_timesteps'],
            'training_episodes_in_screening_csv':len(history),
            'evaluation_seed_bank_complete_and_disjoint':True}
        if name=='cartpole':
            assert result['reached_500']==sum(s==500 for s in scores)
            record['reached_500']=result['reached_500']
            if result['reached_500']==500:
                record['cap_probability_exact_one_sided_95_lower_bound']=.05**(1/500)
                record['normal_interval_warning']='Raw [500,500] plug-in interval is degenerate; no population perfection claim.'
        else:
            assert result['wins']==sum(s>0 for s in scores)
            record['wins']=result['wins']
        records[name]=record
        training=np.array([float(r['score']) for r in history])
        fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
        ax.plot(np.arange(1,len(training)+1),training,color='#6eae9b',alpha=.35,label='Episode score')
        if len(training)>=100:ax.plot(np.arange(100,len(training)+1),np.convolve(training,np.ones(100)/100,'valid'),color='#803b7b',label='Trailing mean, 100 episodes')
        ax.axvline(chosen,color='#333333',linestyle='--',label=f'Selected checkpoint: episode {chosen}')
        ax.set(xlabel='Completed training episode',ylabel='Undiscounted episode score',title=f'{name}: supplemental PPO screening')
        ax.legend();fig.savefig(folder/'training_rewards.png');plt.close(fig)
    evidence={'status':'passed','scope':'raw evidence audit; no new training or evaluation',
        'runs':records,'all_metrics_conditional_on_one_training_seed':True}
    (root/'completed_audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    cp,pong=records['cartpole'],records['pong_selected']
    text=f'''
## Internet-informed PPO performance follow-up

Within a separate 20-minute budget, I tested a different policy-gradient approach
on Deepdish: Stable Baselines3 PPO-Clip with a learned state-value baseline,
GAE, normalized advantages, multiple minibatch updates, and clipping. The
[original PPO paper](https://arxiv.org/abs/1707.06347) and
[OpenAI explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
motivate limiting policy changes. The
[SB3 documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
recommends CPU execution for small MLPs; both trials therefore used Deepdish CPUs.
Pong adds image-derived object/motion features, motivated by
[object-centric Atari research](https://arxiv.org/abs/2306.08649).

This is a supplemental algorithm comparison, not the required basic REINFORCE
experiment. It retains the assigned discounts, official environments, and Pong
actions 2/3. No emulator RAM, downloaded pretrained agent, scripted action rule,
reward shaping or Pong point reset was used. RESEARCH_20MIN.md explains the
changes, hyperparameters, source citations, reproduction, and limitations.

| Frozen stochastic policy | Games | Mean score | Sample SD | At cap / wins |
|---|---:|---:|---:|---:|
| CartPole PPO | 500 | {cp['mean']:.3f} | {cp['sample_sd']:.4f} | {cp['reached_500']}/500 at 500 |
| Pong PPO, image-derived features | 500 | {pong['mean']:.3f} | {pong['sample_sd']:.4f} | {pong['wins']}/500 wins |

Both improve over their prior Part 2 means (438.342 for CartPole; -18.492 for
the earlier Pong CNN follow-up). The baseline, estimator, architecture, input
representation and budgets differ, so this does not isolate a baseline effect.
Each result is conditional on one training seed. CartPole's observed zero SD
does not imply population certainty: its exact one-sided 95% lower bound on the
probability of reaching the cap is {cp.get('cap_probability_exact_one_sided_95_lower_bound',0):.4f}.

![PPO CartPole training](results/research20/cartpole/training_rewards.png)

**Figure 10.** Raw PPO CartPole screening scores and exact trailing 100-episode
mean. The vertical line marks training-only checkpoint selection; later training
was retained in the plot but did not determine the frozen test result.

![PPO CartPole frozen evaluation](results/research20/cartpole/evaluation/reward_histogram.png)

**Figure 11.** All 500 requested seeds reached the official time limit. These
are time-limit truncations, rather than physical failure terminations.

![PPO Pong training](results/research20/pong_selected/training_rewards.png)

**Figure 12.** PPO Pong screening with 12 pixel-derived features. Its best
trailing training mean was -7.10, versus -8.12 for a shorter second pilot adding
approximate intercept features. The shorter pilot is inconclusive, not evidence
that intercept features cannot help. Selection preceded all final Pong games.

![PPO Pong frozen evaluation](results/research20/pong_selected/evaluation/reward_histogram.png)

**Figure 13.** Exactly 500 fixed reserved seeds, stochastic learned actions, and
full official games. All requested seeds must finish; faster extra autoreset
games are ignored. The model SHA-256 and training-only selection were audited.
The feature extractor passed synthetic centroid/motion/reset checks and matched
4,135 official Pong transitions under the same actions. Execution used an
isolated dependency directory, preserving the original assignment environment.

The raw-data audit and selection evidence are under results/research20. Run
`python audit_research_results.py` to recompute the statistics and training plots;
`python validate_research_ppo.py` validates feature and environment behavior.
'''
    report=Path('q3_report.md');existing=report.read_text()
    marker='## Internet-informed PPO performance follow-up'
    if marker in existing:
        start=existing.index(marker);end=existing.index('## With three additional months',start)
        existing=existing[:start]+existing[end:]
    existing=existing.replace('## With three additional months',text+'\n## With three additional months')
    report.write_text(existing)
    document=Path('RESEARCH_20MIN.md');original=document.read_text()
    marker='\n## Completed performance metrics\n'
    if marker in original:original=original.split(marker)[0]
    original+=marker+f'\nCartPole: **{cp["mean"]:.3f} ± {cp["sample_sd"]:.4f}**, {cp["reached_500"]}/500 at the cap.\n\nPong: **{pong["mean"]:.3f} ± {pong["sample_sd"]:.4f}**, **{pong["wins"]}/500 wins**.\n\nEach ± quantity is a sample SD, not a confidence interval. The complete raw-data\naudit is results/research20/completed_audit.json.\n'
    document.write_text(original)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
