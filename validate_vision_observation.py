"""Check that adding current pixels preserves motion and all game outcomes."""
from pathlib import Path
import json


def main():
    import numpy as np
    import torch
    from batched_rollout import EpisodeBatch
    from q2_policy_gradient import Config,Policy,seed_everything
    from vision_policy import make_policy
    seed_everything(3817)
    config=Config(game='pong',reward_scale=1)
    old=Policy(config)
    new=make_policy(config,'current_diff','mlp')
    # Equal action probabilities isolate representation from the action sequence.
    for model in (old,new):
        torch.nn.init.zeros_(model.net[-1].weight)
        torch.nn.init.zeros_(model.net[-1].bias)
    seeds=[1101,1102,1103,1104]
    action_seeds=[2101,2102,2103,2104]
    with EpisodeBatch('pong',4,2,'diff') as pool:
        difference=pool.collect(old,config,seeds,action_seeds)
    with EpisodeBatch('pong',4,2,'current_diff') as pool:
        full=pool.collect(new,config,seeds,action_seeds)
        repeat=pool.collect(new,config,seeds[:1],action_seeds[:1])
    for a,b in zip(difference,full):
        np.testing.assert_array_equal(a['actions'],b['actions'])
        np.testing.assert_array_equal(a['rewards'],b['rewards'])
        np.testing.assert_array_equal(a['states'],b['states'][:,6400:])
        assert b['states'].shape[1]==12800
        assert set(np.unique(b['states'][:,:6400]))<={0,1}
        assert np.count_nonzero(b['states'][0,:6400])>0
        assert np.count_nonzero(b['states'][0,6400:])==0
    np.testing.assert_array_equal(full[0]['states'],repeat[0]['states'])
    result={'episodes':4,'actions_and_rewards_identical_with_matched_uniform_policy':True,
      'motion_channel_identical_to_legacy_features':True,'current_channel_retains_stationary_foreground':True,
      'episode_reset_has_zero_motion_and_preserves_current_frame':True,'reset_replay_identical':True,
      'input_shape':[2,80,80]}
    path=Path('results/pong_tuning/vision_observation_validation.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
