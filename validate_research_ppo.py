"""Validate image-derived features and unchanged official Pong transitions."""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from research_ppo import PongPixels, PongPixelsLead


def frame(ball_x=78,ball_y=100):
    image=np.full((210,160,3),144,dtype=np.uint8)
    image[80:96,140:144]=92
    image[90:106,16:20]=92
    image[ball_y:ball_y+2,ball_x:ball_x+2]=236
    return image


def main():
    wrapped=PongPixels()
    wrapped.previous=None
    original=frame()
    before=original.copy()
    first=wrapped.features(original)
    np.testing.assert_array_equal(original,before)
    np.testing.assert_allclose(first[:4],[-.025,-.175,-.3375,-.2125],atol=1e-7)
    np.testing.assert_array_equal(first[4:6],[0,0])
    second=wrapped.features(frame(82,104))
    np.testing.assert_allclose(second[4:6],[.4,.4],atol=1e-7)
    lead=PongPixelsLead()
    try:
        lead.features(original)
        prediction=lead.features(frame(82,104))
        np.testing.assert_allclose(prediction[-2:],[.6,.46875],atol=1e-7)
        assert lead.observation_space.contains(prediction)
    finally:lead.close()
    missing=frame();missing[100:102,78:80]=144
    absent=wrapped.features(missing)
    assert absent[9]==0 and np.isfinite(absent).all()
    wrapped.previous=None
    np.testing.assert_array_equal(wrapped.features(original),first)
    raw=gym.make('ALE/Pong-v5',frameskip=4,repeat_action_probability=.25,
                 full_action_space=False,obs_type='rgb')
    actions=np.random.default_rng(517)
    transitions=0
    try:
        for seed in (120001,120002,120003,120004):
            state,_=wrapped.reset(seed=seed)
            raw.reset(seed=seed)
            assert state[4]==state[5]==0
            while True:
                action=int(actions.integers(2))
                state,reward,terminated,truncated,_=wrapped.step(action)
                _,reference_reward,reference_terminated,reference_truncated,_=raw.step(action+2)
                assert (reward,terminated,truncated)==(reference_reward,reference_terminated,reference_truncated)
                assert wrapped.observation_space.contains(state)
                transitions+=1
                if terminated or truncated:break
    finally:
        wrapped.close();raw.close()
    output={'synthetic_centroids_and_motion_correct':True,'observation_reset_clears_motion':True,
        'missing_ball_finite_and_masked':True,'input_frames_not_mutated':True,
        'official_v5_rewards_and_endings_identical':True,'actual_episodes':4,
        'transitions_checked':transitions,'action_mapping':'policy 0/1 maps to official RIGHT/LEFT 2/3',
        'privileged_RAM_or_scripted_actions':False,'synthetic_kinematic_feature_correct':True}
    path=Path('results/research20/feature_validation.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
