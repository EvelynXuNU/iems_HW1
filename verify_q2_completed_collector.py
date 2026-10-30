"""Audit final Pong checkpoint and collector without training any policy."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    import csv, json
    import numpy as np
    import torch
    import q2_policy_gradient as p1
    from batched_rollout import EpisodeBatch
    root = ROOT/'results/q2/pong/seed3817_lr001'
    checkpoint = torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=False)
    config = p1.Config(**checkpoint['config'])
    p1.seed_everything(1500000000)
    policy = p1.Policy(config)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()
    seeds = [1500000000+i for i in range(2)]
    action_seeds = [int(np.random.SeedSequence([1500000000,i,42]).generate_state(1,dtype=np.uint64)[0]) for i in range(2)]
    serial = []
    env = p1.make_env(config)
    try:
        for seed, action_seed in zip(seeds, action_seeds):
            serial.append(p1.rollout(env,policy,config,seed,np.random.default_rng(action_seed)))
    finally:
        env.close()
    with EpisodeBatch('pong',slots=4,workers=2) as pool:
        batch = pool.collect(policy,config,seeds,action_seeds,store_states=True)
        score_only = pool.collect(policy,config,seeds,action_seeds,store_states=False)
    for reference, full, scores in zip(serial,batch,score_only):
        for name in ('states','actions','rewards','weights'):
            np.testing.assert_array_equal(reference[name], full[name])
        for name in ('score','length','terminated','truncated','discounted_return'):
            assert reference[name] == full[name] == scores[name], name
    rows=list(csv.DictReader((root/'evaluation_trained/episodes.csv').open()))
    for result,row in zip(batch,rows[:2]):
        assert result['score'] == float(row['score'])
        assert result['length'] == int(row['length'])
        assert result['terminated'] == (row['terminated']=='True')
        assert result['truncated'] == (row['truncated']=='True')
    evidence={'games':2,'partial_batch_slots':4,'workers':2,
       'serial_batched_states_actions_rewards_weights_identical':True,
       'score_only_after_reset_identical':True,'saved_gpu_evaluation_scores_lengths_flags_reproduced_on_cpu':True,
       'scores':[x['score'] for x in batch],'lengths':[x['length'] for x in batch]}
    path=ROOT/'results/q2/verifier/completed_collector_check.json'
    path.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
