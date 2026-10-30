"""Part 2 scientific validation: unbiasedness, variance, learning, and replay.

--variance measures paired gradient variance at frozen real-game policies.
Baseline calibration trajectories are independent of the measurement sample.
"""
from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import q2_policy_gradient as p1
import q3_policy_gradient as p2


def exact_tree(output, variable_length=False):
    theta = torch.tensor([-0.4, 0.6, -0.2, 0.3, -0.7, 0.2, 0.8],
                         dtype=torch.float64, requires_grad=True)
    gamma = 0.7
    direct = theta.sum() * 0
    surrogates = {b: theta.sum() * 0 for b in (-10.0, 0.0, 5.0)}
    bad = theta.sum() * 0
    paths = ([(0,)] + [(1,a,b) for a,b in itertools.product([0,1],repeat=2)]
             if variable_length else list(itertools.product([0,1],repeat=3)))
    for path in paths:
        nodes = [0] if len(path)==1 else [0, 1 + path[0], 3 + 2 * path[0] + path[1]]
        logits = torch.stack([torch.zeros(len(path), dtype=torch.float64), theta[nodes]], dim=1)
        actions = torch.tensor(path)
        logp = Categorical(logits=logits).log_prob(actions)
        probability = logp.sum().exp()
        rewards = ([-1] if len(path)==1 else
                   [2 * path[0] - 1, 3 * (path[1] == path[0]) - 2,
                    5 * (path[2] != path[0]) - path[1] - 2])
        direct = direct + probability * sum(gamma ** t * r for t, r in enumerate(rewards))
        for b in surrogates:
            w = torch.tensor(p2.baseline_weights(rewards, gamma, b))
            surrogates[b] = surrogates[b] + probability.detach() * (logp * w).sum()
        invalid_weights = torch.tensor(gamma ** np.arange(len(path)) * (
            p1.discounted_returns(rewards, gamma) - np.asarray(path)))
        bad = bad + probability.detach() * (logp * invalid_weights).sum()
    target = torch.autograd.grad(direct, theta, retain_graph=True)[0]
    rows = []
    for b, surrogate in surrogates.items():
        gradient = torch.autograd.grad(surrogate, theta, retain_graph=True)[0]
        error = float((target - gradient).abs().max())
        assert error < 1e-12
        rows.append({"baseline": b, "max_gradient_error": error})
    invalid_error = float((target - torch.autograd.grad(bad, theta)[0]).abs().max())
    assert invalid_error > 0.01
    name="exact_variable_length_gradient.csv" if variable_length else "exact_baseline_gradient.csv"
    p1.write_csv(output / name, rows)
    return {"trajectories": len(paths), "variable_length":variable_length,"constant_baselines": rows,
            "action_dependent_baseline_error": invalid_error}


def exact_bandit_variance(output):
    rows = []
    # A deterministic-reward task has a constant baseline eliminating all variance.
    # A noisy task demonstrates that a return-mean baseline need not be optimal.
    for p, r0, r1 in ((0.3, 0.0, 1.0), (0.3, 0.2, 0.8), (0.98, 0.2, 0.8)):
        mean_reward = (1-p)*r0 + p*r1
        optimal = p*r0 + (1-p)*r1
        for label, b in (("none", 0.0), ("return_mean", mean_reward), ("variance_optimal", optimal)):
            expected, second = 0.0, 0.0
            for a in (0, 1):
                pa = p if a else 1-p
                pr = r1 if a else r0
                for reward, probability in ((1, pr), (0, 1-pr)):
                    sample = (reward-b) * (a-p)
                    expected += pa * probability * sample
                    second += pa * probability * sample**2
            variance = max(0.0, second-expected**2)
            assert abs(expected - p*(1-p)*(r1-r0)) < 1e-12
            rows.append({"p_action1": p, "reward_p_action0": r0, "reward_p_action1": r1,
                         "baseline_kind": label, "baseline": b,
                         "expected_gradient": expected, "gradient_variance": variance})
    assert rows[2]["gradient_variance"] < 1e-15
    assert rows[-2]["gradient_variance"] > rows[-3]["gradient_variance"]
    p1.write_csv(output / "exact_bandit_variance.csv", rows)
    return rows


def current_batch_bias():
    p = 0.3
    rows = []
    for batch in (1, 4):
        expectation = 0.0
        for path in itertools.product([0, 1], repeat=batch):
            a = np.array(path, dtype=float)
            probability = p**a.sum() * (1-p)**(batch-a.sum())
            expectation += probability * float(np.mean((a-a.mean()) * (a-p)))
        target = p*(1-p)
        assert abs(expectation-(1-1/batch)*target) < 1e-12
        rows.append({"batch_episodes": batch, "true_gradient": target,
                     "same_batch_mean_baseline_expected_gradient": expectation,
                     "multiplicative_bias_factor": 1-1/batch})
    return rows


def baseline_limits():
    baseline = p2.Baseline()
    assert baseline.value == 0
    baseline.observe([20, 20])
    assert abs(baseline.value-20) < 1e-12  # EMA startup correction
    before = baseline.value
    frozen = p2.baseline_weights([1, 1, 1], 0.95, before)
    baseline.observe([10, 10])
    np.testing.assert_array_equal(frozen, p2.baseline_weights([1, 1, 1], 0.95, before))
    np.testing.assert_array_equal(p2.baseline_weights([1,-2,3], 0.5, 0),
                                  p1.objective_weights([1,-2,3], 0.5))
    np.testing.assert_allclose(p2.baseline_weights([1,2,3], 0, 2), [-1,0,0])
    zeros = p2.Baseline()
    for _ in range(20): zeros.observe([0]*16)
    assert zeros.value == 0 and np.isfinite(p2.baseline_weights([0]*1000, .99, zeros.value)).all()
    negative = p2.Baseline()
    for _ in range(20): negative.observe([-2]*16)
    assert abs(negative.value+2) < 1e-12
    # A reward below expectation receives negative credit even if reward > 0.
    theta = nn.Parameter(torch.zeros(1,2))
    optimizer = torch.optim.Adam([theta], lr=.01)
    loss = p1.reinforce_loss(theta, torch.tensor([1]), torch.tensor(p2.baseline_weights([1], .95, 2)))
    loss.backward()
    optimizer.step()
    probability = float(theta.detach().softmax(-1)[0,1])
    assert probability < .5
    # A stale nonzero baseline can add zero-mean noise to zero-reward episodes.
    p,b=.3,2.0
    samples=np.array([-b*(0-p),-b*(1-p)])
    expected=float(np.array([1-p,p])@samples)
    assert abs(expected)<1e-12
    zero_reward_variance=float(np.array([1-p,p])@(samples**2)-expected**2)
    return {"ema_after_first_batch_20": before, "ema_after_second_batch_10": baseline.value,
            "all_zero_baseline": zeros.value, "negative_return_baseline": negative.value,
            "below_baseline_action_probability_after": probability,
            "zero_reward_stale_baseline_expected_gradient":expected,
            "zero_reward_stale_baseline_gradient_variance":zero_reward_variance,
            "gamma0_weights": p2.baseline_weights([1,2,3],0,2).tolist()}


def bandit_learning(output):
    rows = []
    for seed in (7, 19, 41):
        p1.seed_everything(seed)
        theta = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        optimizer = torch.optim.Adam([theta], lr=.01)
        baseline = p2.Baseline()
        for _ in range(400):
            b = baseline.value
            actions = Categorical(logits=theta).sample((64,))
            reward_probability = torch.where(actions == 1, .8, .2)
            rewards = torch.bernoulli(reward_probability).double()
            loss = p1.reinforce_loss(theta.expand(64,2), actions, rewards-b) / 64
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            baseline.observe(rewards.numpy())
        probability = float(theta.detach().softmax(-1)[1])
        assert probability > .95
        rows.append({"seed":seed, "p_best_action":probability,
                     "expected_reward":.2+.6*probability, "final_baseline":baseline.value})
    p1.write_csv(output / "bandit_learning.csv", rows)
    return rows


def replay_and_first_batch(device):
    Path("tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="q3_replay_",dir="tmp") as directory:
        root = Path(directory)
        config = p1.Config(seed=17, hidden=8, batch_episodes=4)
        full, history = p2.train(config,16,root/"full",device=device)
        p2.train(config,8,root/"split",device=device)
        resumed, split_history = p2.train(config,16,root/"split",device=device,
                                          resume=root/"split/checkpoint.pt")
        error=max(float((full.state_dict()[k]-resumed.state_dict()[k]).abs().max()) for k in full.state_dict())
        assert error == 0 and history == split_history
        c1=torch.load(root/"full/checkpoint.pt",weights_only=False,map_location="cpu")
        c2=torch.load(root/"split/checkpoint.pt",weights_only=False,map_location="cpu")
        assert c1["baseline_state"] == c2["baseline_state"]
        assert c1["updates"] == c2["updates"]
        # First batch has b=0 and MUST reproduce Part 1, including the Adam step.
        first1, h1=p1.train(config,4,root/"part1",device=device)
        first2, h2=p2.train(config,4,root/"part2",device=device)
        initial_error=max(float((first1.state_dict()[k]-first2.state_dict()[k]).abs().max()) for k in first1.state_dict())
        assert initial_error == 0
        assert [{k:v for k,v in r.items() if k!='baseline_used'} for r in h2] == h1
        return {"device":device,"replay_episodes":16,"interruption_after":8,
                "max_parameter_difference":error,"baseline_and_optimizer_diagnostics_identical":True,
                "part1_part2_first_batch_parameter_difference":initial_error}


def trace_variance(gram):
    n=len(gram)
    return max(0.0,float((np.trace(gram)-gram.sum()/n)/(n-1)))


def frozen_variance(config, output, label, checkpoint=None, calibrate=64, probes=128):
    """Compare raw and centered gradients on IDENTICAL frozen-policy trajectories.

    Exact Gram matrices avoid storing bootstrap copies of a large Pong network.
    The baseline is fitted on independent trajectories and frozen before probing.
    """
    p1.seed_everything(config.seed)
    policy=p1.Policy(config)
    if checkpoint:
        c=torch.load(checkpoint,map_location="cpu",weights_only=False)
        policy.load_state_dict(c["policy"])
    policy.eval()
    env=p1.make_env(config)
    baseline=p2.Baseline()
    rng=np.random.default_rng(8761)
    try:
        targets=[]
        for i in range(calibrate):
            e=p1.rollout(env,policy,config,1_100_000_000+i,rng)
            targets.append(e["discounted_return"])
            if len(targets)==16 or i==calibrate-1:
                baseline.observe(targets)
                targets=[]
        b=baseline.value
        raw, scores, episode_rows=[],[],[]
        params=tuple(policy.parameters())
        rng=np.random.default_rng(7654)
        for i in range(probes):
            e=p1.rollout(env,policy,config,1_200_000_000+i,rng)
            logits=policy(torch.tensor(e["states"],dtype=torch.float32))
            logp=Categorical(logits=logits).log_prob(torch.tensor(e["actions"]))
            weights=torch.tensor(p1.objective_weights(e["rewards"],config.gamma,config.reward_scale))
            discount=torch.tensor(config.gamma**np.arange(e["length"])/config.reward_scale)
            gy=torch.autograd.grad((logp*weights).sum(),params,retain_graph=True)
            gz=torch.autograd.grad((logp*discount).sum(),params)
            raw.append(np.concatenate([g.detach().numpy().ravel() for g in gy]).astype(np.float64))
            scores.append(np.concatenate([g.detach().numpy().ravel() for g in gz]).astype(np.float64))
            episode_rows.append({"probe":i+1,"score":e["score"],"discounted_return":e["discounted_return"],
                                 "baseline":b,"raw_gradient_norm":float(np.linalg.norm(raw[-1])),
                                 "centered_gradient_norm":float(np.linalg.norm(raw[-1]-b*scores[-1]))})
        y,z=np.stack(raw),np.stack(scores)
        yy,zz,yz=y@y.T,z@z.T,y@z.T
        centered=yy-b*(yz+yz.T)+b*b*zz
        raw_var=trace_variance(yy)
        centered_var=trace_variance(centered)
        # Paired bootstrap: re-use the same trajectory indices for both estimates.
        counts=np.random.default_rng(987).multinomial(probes,np.ones(probes)/probes,size=2000)
        def bootvar(k):
            return (counts@np.diag(k)-np.einsum('bi,ij,bj->b',counts,k,counts)/probes)/(probes-1)
        ratios=bootvar(centered)/np.maximum(bootvar(yy),np.finfo(float).tiny)
        ratio=centered_var/raw_var if raw_var else None
        result={"policy":label,"env_id":config.env_id,"calibration_episodes":calibrate,
                "probe_episodes":probes,"baseline":b,"baseline_decay":baseline.decay,
                "gradient_dimension":y.shape[1],"variance_measure":"trace of sample gradient covariance (ddof=1)",
                "raw_gradient_variance":raw_var,"baseline_gradient_variance":centered_var,
                "variance_ratio":ratio,"paired_bootstrap_ratio_95_interval":np.quantile(ratios,[.025,.975]).tolist(),
                "gradient_mean_difference_norm":float(np.linalg.norm(y.mean(0)-(y-b*z).mean(0))),
                "checkpoint":str(checkpoint) if checkpoint else None,
                "note":"conditional on a fixed policy and independently fitted baseline; not training-seed uncertainty"}
        p1.write_csv(output/f"{label}_gradient_samples.csv",episode_rows)
        (output/f"{label}_gradient_variance.json").write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        return result
    finally:
        env.close()


def plot_variances(results,output):
    plt=p1.plotting()
    fig,ax=plt.subplots(figsize=(8.5,4.8),layout="constrained")
    x=np.arange(len(results))
    means=np.array([r['variance_ratio'] for r in results])
    limits=np.array([r['paired_bootstrap_ratio_95_interval'] for r in results]).T
    ax.bar(x,means,color="#488c78",width=.5)
    ax.vlines(x,limits[0],limits[1],color="#333333")
    ax.scatter(np.repeat(x,2),limits.T.ravel(),color="#333333",s=14)
    ax.axhline(1,color="#a63328",linestyle="--",label="No variance change")
    ax.set(xticks=x,xticklabels=[r['policy'].replace('_','\n') for r in results],
           ylabel="Gradient variance with baseline / without baseline",
           title="Frozen-policy gradient variance: paired trajectories")
    ax.legend()
    fig.savefig(output/"gradient_variance_ratios.png")
    plt.close(fig)


def run(output,device="cpu",variance=False,pong=False):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    evidence={"exact_gradient":exact_tree(output),
              "exact_variable_length_gradient":exact_tree(output,variable_length=True),
              "exact_variance":exact_bandit_variance(output),
              "current_batch_mean_bias":current_batch_bias(),"limits":baseline_limits(),
              "bandit_learning":bandit_learning(output),"checkpoint_and_first_batch":replay_and_first_batch(device)}
    if variance:
        config=p1.Config(learning_rate=.003)
        measurements=[frozen_variance(config,output,"cartpole_initial")]
        checkpoint=Path("results/q2/cartpole/final_seed3817/checkpoint.pt")
        if checkpoint.exists():
            measurements.append(frozen_variance(config,output,"cartpole_part1_final",checkpoint))
        if pong:
            measurements.append(frozen_variance(p1.Config(game="pong",reward_scale=1),output,
                                                 "pong_initial",calibrate=32,probes=32))
        evidence["frozen_policy_variance"]=measurements
        plot_variances(measurements,output)
    (output/"validation.json").write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",default="results/q3/validation")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--variance",action="store_true")
    parser.add_argument("--pong",action="store_true")
    args=parser.parse_args()
    if args.device=="cuda" and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError("CUDA validation requires exactly one visible GPU")
    run(args.output_dir,args.device,args.variance,args.pong)
