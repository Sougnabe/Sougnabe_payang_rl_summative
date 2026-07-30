from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import environment

_REGISTERED_ENV = environment.DisasterResponseEnv


@dataclass
class DQNSpec:
    learning_rate: float
    gamma: float
    buffer_size: int
    batch_size: int
    train_freq: int
    target_update_interval: int


def make_env(seed: int, render_mode: str | None = None) -> gym.Env:
    env = gym.make("DisasterResponseMission-v0", render_mode=render_mode)
    env.reset(seed=seed)
    return Monitor(env)


def evaluate(model: DQN, n_episodes: int = 5, seed: int = 123) -> tuple[float, float]:
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


def render_episode(model: DQN, output_path: Path, seed: int = 7, max_frames: int = 700) -> None:
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


def train_dqn(
    total_timesteps: int = 80_000,
    seed: int = 42,
    render_best: bool = False,
    run_name: str = "default",
    config: DQNSpec | None = None,
) -> dict[str, Any]:
    spec = config or DQNSpec(
        learning_rate=5e-4,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=128,
        train_freq=4,
        target_update_interval=600,
    )

    log_dir = Path("logs") / "dqn" / run_name
    model_dir = Path("models") / "dqn" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([lambda: make_env(seed=seed)])

    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=spec.learning_rate,
        gamma=spec.gamma,
        buffer_size=spec.buffer_size,
        batch_size=spec.batch_size,
        train_freq=spec.train_freq,
        target_update_interval=spec.target_update_interval,
        learning_starts=4_000,
        exploration_fraction=0.85,
        exploration_final_eps=0.25,
        tensorboard_log=str(log_dir),
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model_path = model_dir / "model.zip"
    model.save(model_path)

    mean_reward, std_reward = evaluate(model, n_episodes=6)

    metrics = {
        "algo": "dqn",
        "run_name": run_name,
        "timesteps": total_timesteps,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "learning_rate": spec.learning_rate,
        "gamma": spec.gamma,
        "buffer_size": spec.buffer_size,
        "batch_size": spec.batch_size,
        "train_freq": spec.train_freq,
        "target_update_interval": spec.target_update_interval,
    }

    metrics_file = Path("logs") / "dqn_results.csv"
    write_header = not metrics_file.exists()
    with metrics_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    if render_best:
        render_episode(model, Path("assets") / "videos" / f"dqn_{run_name}.mp4")

    vec_env.close()
    return metrics


def run_dqn_sweep(total_timesteps: int = 70_000, seed: int = 42) -> list[dict[str, Any]]:
    configs = [
        DQNSpec(1e-3, 0.95, 50_000, 64, 4, 400),
        DQNSpec(1e-3, 0.99, 100_000, 64, 4, 800),
        DQNSpec(5e-4, 0.97, 100_000, 128, 8, 600),
        DQNSpec(5e-4, 0.995, 120_000, 128, 4, 1000),
        DQNSpec(3e-4, 0.99, 150_000, 256, 4, 1200),
        DQNSpec(3e-4, 0.98, 80_000, 128, 8, 900),
        DQNSpec(2e-4, 0.99, 200_000, 256, 4, 1500),
        DQNSpec(2e-4, 0.995, 200_000, 128, 2, 1600),
        DQNSpec(1e-4, 0.99, 120_000, 64, 4, 700),
        DQNSpec(7e-4, 0.96, 60_000, 64, 8, 500),
    ]

    all_metrics: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        run_name = f"sweep_{idx:02d}"
        metrics = train_dqn(
            total_timesteps=total_timesteps,
            seed=seed + idx,
            render_best=False,
            run_name=run_name,
            config=cfg,
        )
        all_metrics.append(metrics)
    return all_metrics
