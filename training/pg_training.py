from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.distributions import Categorical

import environment

_REGISTERED_ENV = environment.DisasterResponseEnv


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ReinforceSpec:
    learning_rate: float
    gamma: float
    entropy_beta: float


@dataclass
class SB3PGSpec:
    learning_rate: float
    gamma: float
    ent_coef: float
    n_steps: int


def make_env(seed: int, render_mode: str | None = None) -> gym.Env:
    env = gym.make("DisasterResponseMission-v0", render_mode=render_mode)
    env.reset(seed=seed)
    return Monitor(env)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    return returns_t


def evaluate_reinforce(policy: PolicyNetwork, n_episodes: int = 5, seed: int = 101) -> tuple[float, float]:
    policy.eval()
    scores: list[float] = []
    for epi in range(n_episodes):
        env = gym.make("DisasterResponseMission-v0")
        obs, _ = env.reset(seed=seed + epi)
        done = False
        truncated = False
        total_reward = 0.0
        while not (done or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)
        env.close()
        scores.append(total_reward)

    arr = np.array(scores, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def render_reinforce_episode(
    policy: PolicyNetwork,
    output_path: Path,
    seed: int = 9,
    max_frames: int = 700,
) -> None:
    env = gym.make("DisasterResponseMission-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []

    done = False
    truncated = False
    steps = 0
    policy.eval()
    while not (done or truncated) and steps < max_frames:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = int(torch.argmax(policy(obs_t), dim=-1).item())
        obs, _, done, truncated, _ = env.step(action)
        steps += 1

    env.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(output_path, frames, fps=20)


def train_reinforce(
    total_timesteps: int = 80_000,
    seed: int = 42,
    render_best: bool = False,
    run_name: str = "default",
    config: ReinforceSpec | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = config or ReinforceSpec(learning_rate=8e-4, gamma=0.99, entropy_beta=0.01)

    log_dir = Path("logs") / "reinforce" / run_name
    model_dir = Path("models") / "pg" / "reinforce" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("DisasterResponseMission-v0")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=spec.learning_rate)

    episode_rewards: list[float] = []
    timesteps = 0
    episode = 0

    while timesteps < total_timesteps:
        obs, _ = env.reset(seed=seed + episode)
        done = False
        truncated = False
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []

        while not (done or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            obs, reward, done, truncated, _ = env.step(int(action.item()))

            log_probs.append(log_prob)
            entropies.append(dist.entropy())
            rewards.append(float(reward))
            timesteps += 1

            if timesteps >= total_timesteps:
                break

        returns = discounted_returns(rewards, spec.gamma)
        losses = []
        for lp, ret in zip(log_probs, returns):
            losses.append(-lp * ret)

        policy_loss = torch.stack(losses).sum()
        entropy_bonus = torch.stack(entropies).mean()
        loss = policy_loss - spec.entropy_beta * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(float(np.sum(rewards)))
        episode += 1

    env.close()

    torch.save(policy.state_dict(), model_dir / "policy.pt")

    mean_reward, std_reward = evaluate_reinforce(policy, n_episodes=6)

    metrics = {
        "algo": "reinforce",
        "run_name": run_name,
        "timesteps": total_timesteps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "learning_rate": spec.learning_rate,
        "gamma": spec.gamma,
        "entropy_beta": spec.entropy_beta,
        "episodes": episode,
        "last_10_ep_mean": float(np.mean(episode_rewards[-10:])),
    }

    metrics_file = Path("logs") / "reinforce_results.csv"
    write_header = not metrics_file.exists()
    with metrics_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    if render_best:
        render_reinforce_episode(policy, Path("assets") / "videos" / f"reinforce_{run_name}.mp4")

    return metrics


def evaluate_sb3(model: PPO | A2C, n_episodes: int = 5, seed: int = 202) -> tuple[float, float]:
    scores: list[float] = []
    for epi in range(n_episodes):
        env = gym.make("DisasterResponseMission-v0")
        obs, _ = env.reset(seed=seed + epi)
        done = False
        truncated = False
        total_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(int(action))
            total_reward += float(reward)
        env.close()
        scores.append(total_reward)

    arr = np.array(scores, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def render_sb3_episode(model: PPO | A2C, output_path: Path, seed: int = 11, max_frames: int = 700) -> None:
    env = gym.make("DisasterResponseMission-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []

    done = False
    truncated = False
    steps = 0
    while not (done or truncated) and steps < max_frames:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(int(action))
        steps += 1

    env.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(output_path, frames, fps=20)


def _train_sb3_pg(
    *,
    algo: str,
    total_timesteps: int,
    seed: int,
    render_best: bool,
    run_name: str,
    config: SB3PGSpec,
) -> dict[str, Any]:
    log_dir = Path("logs") / algo / run_name
    model_dir = Path("models") / "pg" / algo / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([lambda: make_env(seed=seed)])

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            ent_coef=config.ent_coef,
            n_steps=config.n_steps,
            batch_size=128,
            tensorboard_log=str(log_dir),
            verbose=1,
            seed=seed,
        )
    else:
        model = A2C(
            "MlpPolicy",
            vec_env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            ent_coef=config.ent_coef,
            n_steps=config.n_steps,
            tensorboard_log=str(log_dir),
            verbose=1,
            seed=seed,
        )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(model_dir / "model.zip")

    mean_reward, std_reward = evaluate_sb3(model, n_episodes=6)

    metrics = {
        "algo": algo,
        "run_name": run_name,
        "timesteps": total_timesteps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "ent_coef": config.ent_coef,
        "n_steps": config.n_steps,
    }

    metrics_file = Path("logs") / f"{algo}_results.csv"
    write_header = not metrics_file.exists()
    with metrics_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    if render_best:
        render_sb3_episode(model, Path("assets") / "videos" / f"{algo}_{run_name}.mp4")

    vec_env.close()
    return metrics


def train_ppo(
    total_timesteps: int = 80_000,
    seed: int = 42,
    render_best: bool = False,
    run_name: str = "default",
    config: SB3PGSpec | None = None,
) -> dict[str, Any]:
    spec = config or SB3PGSpec(learning_rate=3e-4, gamma=0.99, ent_coef=0.01, n_steps=1024)
    return _train_sb3_pg(
        algo="ppo",
        total_timesteps=total_timesteps,
        seed=seed,
        render_best=render_best,
        run_name=run_name,
        config=spec,
    )


def train_a2c(
    total_timesteps: int = 80_000,
    seed: int = 42,
    render_best: bool = False,
    run_name: str = "default",
    config: SB3PGSpec | None = None,
) -> dict[str, Any]:
    spec = config or SB3PGSpec(learning_rate=7e-4, gamma=0.99, ent_coef=0.01, n_steps=20)
    return _train_sb3_pg(
        algo="a2c",
        total_timesteps=total_timesteps,
        seed=seed,
        render_best=render_best,
        run_name=run_name,
        config=spec,
    )


def run_reinforce_sweep(total_timesteps: int = 70_000, seed: int = 42) -> list[dict[str, Any]]:
    # Improved hyperparameter sweep with higher entropy for better exploration
    configs = [
        ReinforceSpec(8e-4, 0.99, 0.05),
        ReinforceSpec(8e-4, 0.95, 0.05),
        ReinforceSpec(5e-4, 0.99, 0.08),
        ReinforceSpec(5e-4, 0.995, 0.03),
        ReinforceSpec(3e-4, 0.99, 0.10),
        ReinforceSpec(3e-4, 0.97, 0.05),
        ReinforceSpec(1e-3, 0.99, 0.02),
        ReinforceSpec(1e-3, 0.95, 0.08),
        ReinforceSpec(6e-4, 0.98, 0.06),
        ReinforceSpec(4e-4, 0.99, 0.04),
    ]

    runs: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        run_name = f"sweep_{idx:02d}"
        runs.append(
            train_reinforce(
                total_timesteps=total_timesteps,
                seed=seed + idx,
                render_best=False,
                run_name=run_name,
                config=cfg,
            )
        )
    return runs


def run_ppo_sweep(total_timesteps: int = 70_000, seed: int = 42) -> list[dict[str, Any]]:
    configs = [
        SB3PGSpec(3e-4, 0.95, 0.00, 512),
        SB3PGSpec(3e-4, 0.99, 0.00, 1024),
        SB3PGSpec(2e-4, 0.99, 0.01, 1024),
        SB3PGSpec(2e-4, 0.995, 0.02, 2048),
        SB3PGSpec(1e-4, 0.99, 0.02, 1024),
        SB3PGSpec(1e-4, 0.98, 0.03, 512),
        SB3PGSpec(5e-4, 0.97, 0.00, 512),
        SB3PGSpec(5e-4, 0.99, 0.01, 2048),
        SB3PGSpec(7e-4, 0.96, 0.00, 256),
        SB3PGSpec(1.5e-4, 0.995, 0.015, 1024),
    ]

    runs: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        run_name = f"sweep_{idx:02d}"
        runs.append(
            train_ppo(
                total_timesteps=total_timesteps,
                seed=seed + idx,
                render_best=False,
                run_name=run_name,
                config=cfg,
            )
        )
    return runs


def run_a2c_sweep(total_timesteps: int = 70_000, seed: int = 42) -> list[dict[str, Any]]:
    configs = [
        SB3PGSpec(7e-4, 0.95, 0.00, 5),
        SB3PGSpec(7e-4, 0.99, 0.00, 20),
        SB3PGSpec(5e-4, 0.99, 0.01, 20),
        SB3PGSpec(5e-4, 0.995, 0.02, 40),
        SB3PGSpec(3e-4, 0.99, 0.02, 20),
        SB3PGSpec(3e-4, 0.98, 0.03, 10),
        SB3PGSpec(1e-3, 0.97, 0.00, 5),
        SB3PGSpec(1e-3, 0.99, 0.01, 40),
        SB3PGSpec(2e-4, 0.96, 0.00, 10),
        SB3PGSpec(4e-4, 0.995, 0.015, 20),
    ]

    runs: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        run_name = f"sweep_{idx:02d}"
        runs.append(
            train_a2c(
                total_timesteps=total_timesteps,
                seed=seed + idx,
                render_best=False,
                run_name=run_name,
                config=cfg,
            )
        )
    return runs
