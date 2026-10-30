"""Part 2: REINFORCE with a lagged, bias-corrected EMA scalar baseline.

Part 1's environment, policy, collector, actor update, and evaluator are reused
without modifying that source. Only the actor weights change to gamma^t(G_t-b)/c.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

import q2_policy_gradient as part1
from q2_policy_gradient import Config, Policy, discounted_returns

METHOD = "REINFORCE with lagged EMA baseline"


@dataclass
class Baseline:
    """A scalar, fixed during each batch, learned only for subsequent batches."""
    decay: float = 0.9
    numerator: float = 0.0
    mass: float = 0.0
    batches_seen: int = 0
    episodes_seen: int = 0

    def __post_init__(self):
        if not np.isfinite(self.decay) or not 0 <= self.decay < 1:
            raise ValueError("Baseline decay must be finite in [0,1)")

    @property
    def value(self):
        return self.numerator / self.mass if self.mass else 0.0

    def observe(self, episode_start_returns):
        """Call AFTER the actor step. No current trajectory uses its own target."""
        targets = np.asarray(episode_start_returns, dtype=np.float64)
        if targets.size == 0 or not np.isfinite(targets).all():
            raise ValueError("Baseline targets must be nonempty and finite")
        self.numerator = self.decay * self.numerator + (1 - self.decay) * float(targets.mean())
        self.mass = self.decay * self.mass + (1 - self.decay)
        self.batches_seen += 1
        self.episodes_seen += len(targets)


def baseline_weights(rewards, gamma, baseline, scale=1.0):
    if not np.isfinite(baseline):
        raise ValueError("Baseline must be finite")
    return gamma ** np.arange(len(rewards), dtype=np.float64) * (
        discounted_returns(rewards, gamma) - baseline) / scale


def plot_training(history, output, config):
    plt = part1.plotting()
    scores = np.array([r["score"] for r in history])
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.plot(np.arange(1, len(scores) + 1), scores, color="#6eae9b", alpha=0.5,
            linewidth=0.7, label="Episode score")
    if len(scores) >= 100:
        ax.plot(np.arange(100, len(scores) + 1), np.convolve(scores, np.ones(100)/100, "valid"),
                color="#803b7b", linewidth=2, label="Trailing mean, 100 episodes")
    ax.set(xlabel="Training episode", ylabel="Undiscounted episode reward (score)",
           title=f"{config.env_id}: Part 2, lagged EMA baseline (seed {config.seed})")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(Path(output) / "training_rewards.png")
    plt.close(fig)


def train(config, episodes, output, device="cpu", resume=None, baseline_decay=0.9):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    part1.seed_everything(config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = Policy(config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    baseline = Baseline(decay=baseline_decay)
    history, updates, elapsed = [], [], 0.0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("method") != METHOD or checkpoint["config"] != asdict(config):
            raise ValueError("Resume requires a matching Part 2 experiment")
        if checkpoint["baseline_state"]["decay"] != baseline_decay:
            raise ValueError("Resume baseline decay must match the checkpoint")
        policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["action_rng"]
        baseline = Baseline(**checkpoint["baseline_state"])
        history, updates = checkpoint["history"], checkpoint["updates"]
        elapsed = checkpoint["elapsed_seconds"]
    elif (output / "checkpoint.pt").exists() or (output / "training.csv").exists():
        raise FileExistsError("Output contains a run; choose another directory or --resume")
    if len(history) >= episodes:
        raise ValueError("Target episode count must exceed completed episodes")
    env = part1.make_env(config)
    start = time.monotonic()
    metadata = {"method": METHOD, "config": asdict(config), "gamma": config.gamma,
                "env_id": config.env_id, "device": device, "versions": part1.versions(),
                "target_episodes": episodes, "baseline_decay": baseline_decay,
                "baseline_target": "previous batches' mean episode-start discounted return G_0",
                "baseline_update_timing": "after actor update; frozen within next batch",
                "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "cuda_runtime": torch.version.cuda if device == "cuda" else None,
                "max_episode_steps": env.spec.max_episode_steps,
                "pong_max_emulator_frames": 108000 if config.game == "pong" else None,
                "source_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in (__file__, part1.__file__)},
                "objective": "E[sum_t gamma^t r_t]",
                "evaluation_policy": "stochastic, sampled from trained policy"}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        while len(history) < episodes:
            # A snapshot from past batches; this value never sees current actions.
            b = baseline.value
            batch = []
            advantage_sum, steps = 0.0, 0
            for _ in range(min(config.batch_episodes, episodes - len(history))):
                index = len(history)
                do_trace = index in (0, config.batch_episodes)
                episode = part1.rollout(env, policy, config, config.seed + index, rng,
                                        device=device, trace=do_trace)
                episode["weights"] = baseline_weights(episode["rewards"], config.gamma, b,
                                                        config.reward_scale)
                returns = discounted_returns(episode["rewards"], config.gamma)
                advantage_sum += float((returns - b).sum())
                steps += episode["length"]
                for row in episode["trace"]:
                    row["baseline_used"] = b
                    row["advantage"] = row["reward_to_go"] - b
                    row["gradient_weight"] = float(episode["weights"][row["step"]])
                if do_trace:
                    name = "first_10_training_steps.json" if index == 0 else "first_10_steps_after_baseline.json"
                    (output / name).write_text(json.dumps(episode["trace"], indent=2) + "\n")
                batch.append(episode)
                history.append({"episode": index + 1, "seed": config.seed + index,
                                **{k: episode[k] for k in ("score", "discounted_return", "length", "terminated", "truncated")},
                                "baseline_used": b})
            diagnostic = part1.update(policy, optimizer, batch, config, device)
            # Observe current returns only AFTER the actor has used the old b.
            baseline.observe([e["discounted_return"] for e in batch])
            updates.append({"update": len(updates) + 1, "through_episode": len(history),
                            "batch_episodes": len(batch), "baseline_used": b,
                            "baseline_for_next_batch": baseline.value,
                            "advantage_mean_diagnostic_only": advantage_sum / steps, **diagnostic})
            part1.save_checkpoint(output / "checkpoint.pt", {
                "method": METHOD, "config": asdict(config), "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history, "updates": updates,
                "baseline_state": asdict(baseline), "action_rng": rng.bit_generator.state,
                "elapsed_seconds": elapsed + time.monotonic() - start,
            })
            part1.write_csv(output / "training.csv", history)
            part1.write_csv(output / "updates.csv", updates)
            if len(updates) % 10 == 0 or len(history) == episodes:
                recent = np.mean([r["score"] for r in history[-100:]])
                print(f"part2 {config.game} episode={len(history)} mean_last_100={recent:.2f} "
                      f"baseline={b:.4f} grad={diagnostic['gradient_norm']:.4g} "
                      f"entropy={diagnostic['entropy']:.3f}", flush=True)
        plot_training(history, output, config)
    finally:
        env.close()
    return policy, history


def evaluate(checkpoint_path, output, episodes=500, seed=1_500_000_000, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != METHOD:
        raise ValueError("A Part 2 checkpoint is required")
    # Identical frozen stochastic evaluation; a baseline is used only in training.
    result = part1.evaluate(checkpoint_path, output, episodes, seed, device)
    result.update({"method": METHOD, "baseline_decay": checkpoint["baseline_state"]["decay"],
                   "final_baseline": Baseline(**checkpoint["baseline_state"]).value})
    output = Path(output)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "episodes.csv").open() as stream:
        scores = np.array([float(r["score"]) for r in csv.DictReader(stream)])
    config = Config(**checkpoint["config"])
    plt = part1.plotting()
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    bins = np.arange(-21.5, 22.5, 1) if config.game == "pong" else np.linspace(0.5, 500.5, 26)
    ax.hist(scores, bins=bins, color="#488c78", edgecolor="white")
    ax.axvline(result["mean"], color="#803b7b", linestyle="--",
               label=f"Mean {result['mean']:.2f}; sample SD {result['sample_standard_deviation']:.2f}")
    ax.set(xlabel="Undiscounted episode reward (score)", ylabel="Number of episodes",
           title=f"{config.env_id}: Part 2, lagged EMA baseline\n{episodes} held-out stochastic-policy episodes")
    ax.legend()
    fig.savefig(output / "reward_histogram.png")
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--game", choices=["cartpole", "pong"], required=True)
    t.add_argument("--episodes", type=int, required=True)
    t.add_argument("--seed", type=int, default=3817)
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--batch-episodes", type=int, default=16)
    t.add_argument("--reward-scale", type=float, default=20)
    t.add_argument("--baseline-decay", type=float, default=0.9)
    t.add_argument("--resume")
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--episodes", type=int, default=500)
    e.add_argument("--seed", type=int, default=1_500_000_000)
    for p in (t, e):
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        p.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("CUDA runs require exactly one visible GPU")
    if args.command == "train":
        config = Config(args.game, args.seed, args.lr, args.hidden, args.batch_episodes, args.reward_scale)
        train(config, args.episodes, args.output_dir, args.device, args.resume, args.baseline_decay)
    else:
        evaluate(args.checkpoint, args.output_dir, args.episodes, args.seed, args.device)


if __name__ == "__main__":
    main()
