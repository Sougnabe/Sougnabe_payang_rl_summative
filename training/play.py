from __future__ import annotations

import argparse
from pathlib import Path
import time

import imageio
import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import DisasterResponseEnv
from training.pg_training import PolicyNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained model and save video")
    parser.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="assets/videos/demo_play.mp4")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--live", action="store_true", help="Show the rollout in a live window")
    parser.add_argument("--live-fps", type=float, default=12.0, help="Frame rate for the live window")
    parser.add_argument("--stay-open", action="store_true", help="Keep the live window open until you close it manually")
    parser.add_argument("--assist-energy", action="store_true", help="Override actions to return/recharge when battery is low")
    parser.add_argument("--assist-threshold", type=float, default=0.30, help="Battery fraction threshold for return-to-base assist")
    return parser.parse_args()


def load_agent(algo: str, model_path: str):
    if algo == "dqn":
        return DQN.load(model_path)
    if algo == "ppo":
        return PPO.load(model_path)
    if algo == "a2c":
        return A2C.load(model_path)

    env = DisasterResponseEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    env.close()

    policy = PolicyNetwork(obs_dim, act_dim)
    state = torch.load(model_path, map_location="cpu")
    policy.load_state_dict(state)
    policy.eval()
    return policy


def predict_action(agent, algo: str, obs: np.ndarray) -> int:
    if algo in {"dqn", "ppo", "a2c"}:
        action, _ = agent.predict(obs, deterministic=True)
        return int(action)

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits = agent(obs_t)
    return int(torch.argmax(logits, dim=-1).item())


def maybe_override_for_energy(env: DisasterResponseEnv, action: int, threshold: float) -> int:
    battery_frac = float(np.clip(env.battery / env.battery_capacity, 0.0, 1.0))
    dist_to_base = float(np.linalg.norm(env.base_pos - env.agent_pos))

    # Emergency return if battery is low and we are far from base.
    if battery_frac <= threshold and dist_to_base > env.recharge_radius:
        return 11

    # Once at base, hold position until almost full so the drone does not orbit the charge zone.
    if dist_to_base <= env.recharge_radius and battery_frac < 0.98:
        return 0

    return action


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agent = load_agent(args.algo, args.model_path)
    render_mode = "human" if args.live else "rgb_array"
    env = DisasterResponseEnv(render_mode=render_mode)

    if args.live and args.stay_open:
        env.max_steps = 10_000_000

    obs, _ = env.reset(seed=args.seed)
    done = False
    truncated = False
    total_reward = 0.0
    frames = []
    steps = 0

    while True:
        if done:
            break
        if truncated and not (args.live and args.stay_open):
            break
        if args.live and args.stay_open:
            keep_open = env.render()
            if keep_open is False:
                break
        else:
            if steps >= args.max_steps:
                break
            frame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)

        if args.live and args.stay_open:
            action = predict_action(agent, args.algo, obs)
            if args.assist_energy:
                action = maybe_override_for_energy(env, action, threshold=args.assist_threshold)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            if steps % 40 == 0:
                print(
                    f"step={steps} reward={total_reward:.2f} battery={info['battery']:.3f} "
                    f"delivered={info['delivered']} kits_left={info['kits_left']}"
                )

            if args.live and args.live_fps > 0:
                time.sleep(1.0 / args.live_fps)
            continue

        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame)

        action = predict_action(agent, args.algo, obs)
        if args.assist_energy:
            action = maybe_override_for_energy(env, action, threshold=args.assist_threshold)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

        if steps % 40 == 0:
            print(
                f"step={steps} reward={total_reward:.2f} battery={info['battery']:.3f} "
                f" delivered={info['delivered']} kits_left={info['kits_left']}"
            )

        if args.live and args.live_fps > 0:
            time.sleep(1.0 / args.live_fps)

    env.close()

    if frames:
        imageio.mimsave(out_path, frames, fps=20)

    print(f"Saved video to {out_path}")
    print(f"Episode reward: {total_reward:.2f} | steps: {steps}")


if __name__ == "__main__":
    main()
