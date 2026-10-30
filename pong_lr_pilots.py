"""Short training-only Pong LR screen; all processes share ONE visible GPU.

No final evaluation is run here. The existing Part 2 0.01 history is the matched
reference at the same episode budget. Per-pilot CPU workers are bounded.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=3072)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output-dir',default='results/pong_tuning/lr_screen')
    args=parser.parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible.startswith('GPU-') or ',' in visible:
        raise RuntimeError('Select exactly one GPU UUID for all pilots')
    root=Path(args.output_dir)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise FileExistsError('Preserve existing pilots; use a fresh directory')
    manifest={'started_at':datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(),
      'hypothesis':'Adam 0.01 produces premature policy concentration; smaller actor steps may improve learning',
      'part':2,'learning_rates':[.003,.001,.0003], 'episodes_per_rate':args.episodes,
      'workers_per_pilot':args.workers,'gpu_uuid':visible,'physical_gpu_count':1,
      'seed':3817,'hidden_units':64,'observation':'80x80 frame difference',
      'batch_episodes':16,'baseline':'lagged EMA, decay 0.9','gamma':.99,
      'control':'existing Part 2 lr=0.01 history, first N episodes only',
      'selection':'last 512 training score mean; inspect discounted return and entropy for stability; no heldout-game selection',
      'reserved_new_final_evaluation_seed':1900000000,'reserved_final_episodes':500,
      'reference_source_commit':'5b1d407',
      'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ['fast_pg.py','batched_rollout.py','q2_policy_gradient.py','q3_policy_gradient.py']},
      'jobs':[],'status':'running'}
    jobs=[]
    for rate in manifest['learning_rates']:
        name=f'part2_lr{rate:g}'
        folder=root/name
        folder.mkdir(exist_ok=True)
        log=(folder/'console.log').open('w')
        command=[sys.executable,'-u','fast_pg.py','train','--part','2','--game','pong',
          '--episodes',str(args.episodes),'--lr',str(rate),'--workers',str(args.workers),
          '--reward-scale','1','--save-every','8','--device','cuda','--output-dir',str(folder)]
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        jobs.append((rate,folder,process,log))
        manifest['jobs'].append({'learning_rate':rate,'pid':process.pid,'directory':str(folder)})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
    summary=[]
    for rate,folder,process,log in jobs:
        code=process.wait()
        log.close()
        manifest['jobs'][manifest['learning_rates'].index(rate)]['exit_code']=code
        if code:continue
        rows=list(csv.DictReader((folder/'training.csv').open()))
        updates=list(csv.DictReader((folder/'updates.csv').open()))
        last=rows[-512:]
        summary.append({'learning_rate':rate,'episodes':len(rows),
          'last512_score_mean':sum(float(r['score']) for r in last)/len(last),
          'last512_discounted_return_mean':sum(float(r['discounted_return']) for r in last)/len(last),
          'last_batch_entropy':float(updates[-1]['entropy']),
          'last_batch_gradient_norm':float(updates[-1]['gradient_norm']),
          'directory':str(folder)})
    manifest['status']='complete' if len(summary)==len(jobs) else 'some_pilots_failed'
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
