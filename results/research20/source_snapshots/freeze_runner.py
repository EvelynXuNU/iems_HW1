"""Freeze the training-only Pong choice, stop trials, and evaluate exactly once."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import torch


def main():
    from stable_baselines3 import PPO
    from research_ppo import evaluate
    torch.set_num_threads(1)
    # Fixed before either final Pong test: leave over six minutes for evaluation.
    freeze=1793335105.0  # 2026-10-30 04:38:25 UTC
    # Convert explicitly to avoid any dependence on a local timezone.
    from datetime import datetime
    freeze=datetime.fromisoformat('2026-10-30T04:38:25+00:00').timestamp()
    while time.time()<freeze:time.sleep(min(1,freeze-time.time()))
    runtime=Path(__file__).resolve().parent
    root=runtime.parent/'results/research20'
    # These are exclusively the trial processes launched by this campaign.
    for name in ('pong','pong_lead'):
        pid=int((runtime/f'{name}.pid').read_text())
        process=Path(f'/proc/{pid}')
        if process.exists() and b'research_ppo.py' in (process/'cmdline').read_bytes():
            if name=='pong_lead' and (root/name/'complete.json').exists():continue
            os.kill(pid,signal.SIGINT)
    until=time.time()+15
    while time.time()<until:
        live=False
        for name in ('pong','pong_lead'):
            pid=int((runtime/f'{name}.pid').read_text())
            status=Path(f'/proc/{pid}/status')
            if status.exists() and '\nState:\tZ' not in status.read_text():live=True
        if not live:break
        time.sleep(.25)
    candidates=[]
    for name in ('pong','pong_lead'):
        folder=root/name
        if (folder/'selection.json').exists():
            selection=json.loads((folder/'selection.json').read_text())
            candidates.append((selection['last100_training_score_mean'],name,selection))
    assert candidates,'No completed 100-game training window'
    score,name,selection=max(candidates,key=lambda x:x[0])
    target=root/'pong_selected';target.mkdir(exist_ok=False)
    for filename in ('selected_model.zip','training.csv','plan.json'):
        shutil.copyfile(root/name/filename,target/filename)
    selection.update(chosen_variant=name,chosen_training_score_mean=score,
        compared_training_windows=[{'variant':n,'last100_training_mean':v} for v,n,_ in candidates],
        final_evaluation_used_for_selection=False,
        model_sha256=hashlib.sha256((target/'selected_model.zip').read_bytes()).hexdigest())
    (target/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(json.dumps(selection),flush=True)
    model=PPO.load(target/'selected_model.zip',device='cpu')
    result=evaluate(model,name,target,
        datetime.fromisoformat('2026-10-30T04:45:20+00:00').timestamp(),workers=16)
    (target/'complete.json').write_text(json.dumps({'evaluation_complete':result['complete'],
        'finished_at':datetime.now().astimezone().isoformat(),'new_runs_queued':False},indent=2)+'\n')


if __name__=='__main__':main()
