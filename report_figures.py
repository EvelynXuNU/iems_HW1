"""Compact publication figures, regenerated from the existing raw evidence."""
import csv
import json
import os
from pathlib import Path

Path('tmp/report/matplotlib').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(Path('tmp/report/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def main():
    output=Path('report/figures');output.mkdir(parents=True,exist_ok=True)
    audit={'required_plot_count':11,'figures':{}}
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,'savefig.dpi':220})
    data=rows('results/q1/state_results.csv')
    states=np.array([int(r['state']) for r in data])
    fig=plt.figure(figsize=(7.6,5.1),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[1.15,1])
    for column,key,title in [(0,'finite_cost_t0','(a) Enumeration / backward induction, T=500'),
                              (1,'vi_cost','(b) Infinite-horizon value iteration')]:
        ax=fig.add_subplot(grid[0,column])
        ax.plot(states,[float(r[key]) for r in data],color='#377c9b')
        ax.set(title=title,xlabel='Initial passengers',ylabel='Expected discounted cost')
        ax.grid(alpha=.2)
    ax=fig.add_subplot(grid[1,:])
    actions=[int(r['pi_action']) for r in data]
    ax.step(states,actions,where='post',color='#488c78')
    ax.axvline(13,color='#803b7b',linestyle='--',linewidth=1,label='Dispatch from 13')
    ax.set(title='(c) Policy iteration: optimal policy over all 201 states',
           xlabel='Passengers waiting',ylabel='Action',yticks=[0,1],yticklabels=['Wait','Dispatch'],ylim=(-.1,1.2))
    ax.legend(loc='center right');ax.grid(alpha=.2)
    audit['figures']['shuttle.png']={'panels':len(fig.axes),'state_count':len(states),'full_state_range':[int(states.min()),int(states.max())]}
    fig.savefig(output/'shuttle.png');plt.close(fig)
    for part in (1,2):
        fig,axes=plt.subplots(2,2,figsize=(7.6,5.3),layout='constrained')
        cases=[]
        for column,game in enumerate(('cartpole','pong')):
            name='final_seed3817' if game=='cartpole' else 'seed3817_lr001'
            root=Path(f'results/q{part+1}/{game}/{name}')
            history=rows(root/'training.csv')
            scores=np.array([float(r['score']) for r in history])
            ax=axes[0,column]
            ax.plot(np.arange(1,len(scores)+1),scores,color='#6eae9b',alpha=.45,linewidth=.45,label='Raw episode score')
            ax.plot(np.arange(100,len(scores)+1),np.convolve(scores,np.ones(100)/100,'valid'),color='#803b7b',linewidth=1.2,label='Trailing mean, exactly 100')
            label='CartPole-v1' if game=='cartpole' else 'ALE/Pong-v5'
            ax.set(title=f'({chr(97+column)}) {label}: training',xlabel='Completed training episode',ylabel='Episode reward (score)')
            ax.legend(loc='best');ax.grid(alpha=.15)
            result=json.loads((root/'evaluation_trained/evaluation.json').read_text())
            tested=np.array([float(r['score']) for r in rows(root/'evaluation_trained/episodes.csv')])
            assert len(tested)==500
            assert abs(tested.mean()-result['mean'])<1e-10
            assert abs(tested.std(ddof=1)-result['sample_standard_deviation'])<1e-10
            cases.append({'game':game,'training_episodes':len(history),'moving_average_window':100,
                'first_moving_average_episode':100,'evaluation_episodes':len(tested),
                'mean':float(tested.mean()),'sample_sd':float(tested.std(ddof=1)),
                'training_csv':str(root/'training.csv'),'evaluation_csv':str(root/'evaluation_trained/episodes.csv')})
            ax=axes[1,column]
            bins=np.linspace(.5,500.5,26) if game=='cartpole' else np.arange(-21.5,22.5,1)
            ax.hist(tested,bins=bins,color='#488c78' if part==2 else '#377c9b',edgecolor='white',linewidth=.45)
            ax.axvline(tested.mean(),color='#803b7b',linestyle='--',linewidth=1)
            ax.set(title=f'({chr(99+column)}) {label}: 500 frozen games',xlabel='Episode reward (score)',ylabel='Episodes')
            ax.text(.98,.97,f'Mean {tested.mean():.3f}\nSample SD {tested.std(ddof=1):.4f}',
                    transform=ax.transAxes,ha='right',va='top',fontsize=8,
                    bbox={'facecolor':'white','alpha':.85,'edgecolor':'#ddd','pad':3})
        audit['figures'][f'part{part}.png']={'panels':len(fig.axes),'cases':cases}
        fig.savefig(output/f'part{part}.png');plt.close(fig)
    assert sum(v['panels'] for v in audit['figures'].values())==11
    (output/'figure_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Regenerated the 11 required plots in three readable composite figures.')


if __name__=='__main__':main()
