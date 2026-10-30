"""Rebuild paired-seed comparison tables and plots from saved episode evidence."""
import csv
import json
from pathlib import Path

import numpy as np

from q2_policy_gradient import plotting,write_csv


def run():
    output=Path('results/q3/comparison')
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    histories={1:[],2:[]}
    for seed in (3817,3818,3819):
        name='final_seed3817' if seed==3817 else f'seed{seed}'
        for part in (1,2):
            root=Path(f'results/q{part+1}/cartpole/{name}')
            r=json.loads((root/'evaluation_trained/evaluation.json').read_text())
            with (root/'training.csv').open() as f:
                scores=np.array([float(x['score']) for x in csv.DictReader(f)])
            assert len(scores)==r['training_episodes']==6000
            assert r['episodes']==500
            histories[part].append(scores)
            rows.append({'training_seed':seed,'part':part,'training_episodes':6000,
              'evaluation_episodes':500,'mean_score':r['mean'],
              'sample_sd':r['sample_standard_deviation'],'reached_500_fraction':r['cartpole_500_fraction']})
    write_csv(output/'cartpole_three_seed_comparison.csv',rows)
    summary={'cartpole':{},'pong':{}}
    for part in (1,2):
        means=[r['mean_score'] for r in rows if r['part']==part]
        summary['cartpole'][f'part{part}']={'mean_of_three_seed_means':float(np.mean(means)),
            'sample_sd_of_three_seed_means':float(np.std(means,ddof=1))}
    for part in (1,2):
        p=Path(f'results/q{part+1}/pong/seed3817_lr001/evaluation_trained/evaluation.json')
        summary['pong'][f'part{part}']=json.loads(p.read_text()) if p.exists() else {'status':'pending'}
    summary['limitation']='Three distinct initialization/action seeds; reset seed banks overlap. Descriptive comparison, no independent-seed confidence claim.'
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt=plotting()
    colors={1:'#377c9b',2:'#488c78'}
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for part in (1,2):
        curves=np.stack([np.convolve(s,np.ones(100)/100,'valid') for s in histories[part]])
        for curve in curves:ax.plot(np.arange(100,6001),curve,color=colors[part],alpha=.2,linewidth=.7)
        ax.plot(np.arange(100,6001),curves.mean(0),color=colors[part],linewidth=2,
                label=f'Part {part}: '+('no baseline' if part==1 else 'lagged EMA baseline'))
    ax.set(xlabel='Training episode',ylabel='Trailing 100-episode mean score',
           title='CartPole-v1: matched 6,000-episode budgets\nBold: mean of 3 training-seed curves; faint: individual seeds')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output/'cartpole_training_comparison.png')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    for j,seed in enumerate((3817,3818,3819)):
        values=[next(r['mean_score'] for r in rows if r['part']==part and r['training_seed']==seed) for part in (1,2)]
        ax.plot([j-.12,j+.12],values,color='#777777',linewidth=1)
        for part,value in enumerate(values,1):
            ax.scatter(j+(-.12 if part==1 else .12),value,color=colors[part],s=55,
                       label=f'Part {part}' if j==0 else None)
            ax.annotate(f'{value:.1f}',(j+(-.12 if part==1 else .12),value),
                        xytext=(0,8),textcoords='offset points',ha='center',fontsize=10)
    ax.set(xticks=range(3),xticklabels=['3817','3818','3819'],xlabel='Training initialization/action seed',
           ylabel='Mean score over 500 held-out episodes',ylim=(0,500),
           title='CartPole-v1: final frozen-policy evaluation')
    ax.legend()
    ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'cartpole_seed_comparison.png')
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
