"""Verify fast-runner replay, baseline isolation, and exact evaluation counts."""
from pathlib import Path
import argparse
import json
import tempfile


def main():
    import torch
    import q2_policy_gradient as p1
    import fast_pg
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/q3/fast_runner_validation.json')
    a=parser.parse_args()
    Path('tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='fast_replay_',dir='tmp') as directory:
        root=Path(directory)
        config=p1.Config(seed=17,hidden=64,batch_episodes=4)
        fast_pg.train(config,2,16,root/'full','cpu',2,save_every=1)
        fast_pg.train(config,2,8,root/'split','cpu',1,save_every=1)
        fast_pg.train(config,2,16,root/'split','cpu',2,resume=root/'split/checkpoint.pt',save_every=1)
        full=torch.load(root/'full/checkpoint.pt',weights_only=False,map_location='cpu')
        split=torch.load(root/'split/checkpoint.pt',weights_only=False,map_location='cpu')
        error=max(float((full['policy'][k]-split['policy'][k]).abs().max()) for k in full['policy'])
        assert error==0 and full['history']==split['history']
        assert full['baseline_state']==split['baseline_state'] and full['updates']==split['updates']
        fast_pg.train(config,1,4,root/'first1','cpu',2,save_every=1)
        fast_pg.train(config,2,4,root/'first2','cpu',2,save_every=1)
        one=torch.load(root/'first1/checkpoint.pt',weights_only=False,map_location='cpu')
        two=torch.load(root/'first2/checkpoint.pt',weights_only=False,map_location='cpu')
        first_error=max(float((one['policy'][k]-two['policy'][k]).abs().max()) for k in one['policy'])
        assert first_error==0
        evaluation=fast_pg.evaluate(root/'full/checkpoint.pt',root/'eval',17,device='cpu',workers=2)
        assert evaluation['episodes']==17
        assert len((root/'eval/episodes.csv').read_text().splitlines())==18
        # Continue a legacy checkpoint without discarding its completed episodes.
        p1.train(config,8,root/'legacy','cpu')
        legacy=torch.load(root/'legacy/checkpoint.pt',weights_only=False,map_location='cpu')
        fast_pg.train(config,1,16,root/'legacy','cpu',2,resume=root/'legacy/checkpoint.pt',save_every=1)
        continued=torch.load(root/'legacy/checkpoint.pt',weights_only=False,map_location='cpu')
        assert continued['history'][:8]==legacy['history']
        assert len(continued['history'])==16
        result={'resumed_parameter_error':error,'resumed_baseline_state_identical':True,
                'resumed_trajectory_history_identical':True,'worker_count_changed_from':1,'worker_count_changed_to':2,
                'part1_part2_first_batch_parameter_error':first_error,
                'partial_evaluation_exact_episode_count':17,'legacy_history_preserved_episodes':8,
                'legacy_continued_total_episodes':16}
    output=Path(a.output)
    output.parent.mkdir(exist_ok=True,parents=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
