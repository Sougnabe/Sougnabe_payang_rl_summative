from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import environment

_REGISTERED_ENV = environment.DisasterResponseEnv


def rollout(policy_fn, seed: int, max_steps: int = 420) -> tuple[float, dict]:
    env = gym.make("DisasterResponseMission-v0")
    obs, info = env.reset(seed=seed)
    total = 0.0
    done = truncated = False
    steps = 0
    while not (done or truncated) and steps < max_steps:
        action = policy_fn(obs)
        obs, reward, done, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
    env.close()
    return total, info


def random_policy(env_action_space):
    def _fn(obs: np.ndarray) -> int:
        return env_action_space.sample()

    return _fn


def parked_policy(obs: np.ndarray) -> int:
    # Degenerate strategy: always issue the emergency-return / hover-at-base command.
    return 11


def scan_spam_policy(obs: np.ndarray) -> int:
    # Degenerate strategy: always scan, regardless of distance to any victim. Guards
    # against a near-miss "close enough to scan" bonus being farmable from a fixed spot.
    return 9


def heuristic_policy(obs: np.ndarray) -> int:
    battery_frac = obs[4]
    kits_left = obs[5] * 3
    discovered = [obs[8 + i * 4 + 3] > 0 for i in range(3)]
    delivered = [obs[8 + i * 4 + 2] < 0 for i in range(3)]

    def move_toward(dx: float, dy: float) -> int:
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 4
        return 1 if dy > 0 else 2

    if all(delivered):
        base_dx, base_dy = obs[2], obs[3]
        dist_to_base = np.sqrt(base_dx**2 + base_dy**2) * 100
        return 0 if dist_to_base < 5 else move_toward(base_dx, base_dy)

    if battery_frac < 0.25:
        base_dx, base_dy = obs[2], obs[3]
        dist_to_base = np.sqrt(base_dx**2 + base_dy**2) * 100
        return 0 if dist_to_base < 7 else move_toward(base_dx, base_dy)

    target_idx, target_dist = -1, float("inf")
    for i in range(3):
        if discovered[i] and not delivered[i] and kits_left > 0:
            dx, dy = obs[8 + i * 4], obs[8 + i * 4 + 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < target_dist:
                target_dist, target_idx = dist, i

    if target_idx >= 0:
        dx, dy = obs[8 + target_idx * 4], obs[8 + target_idx * 4 + 1]
        dist = np.sqrt(dx**2 + dy**2) * 100
        return 10 if dist < 5.5 else move_toward(dx, dy)

    target_idx, target_dist = -1, float("inf")
    for i in range(3):
        if not delivered[i]:
            dx, dy = obs[8 + i * 4], obs[8 + i * 4 + 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < target_dist:
                target_dist, target_idx = dist, i

    if target_idx >= 0:
        dx, dy = obs[8 + target_idx * 4], obs[8 + target_idx * 4 + 1]
        dist = np.sqrt(dx**2 + dy**2) * 100
        return 9 if dist < 9.5 else move_toward(dx, dy)

    return 0


def evaluate_policy(name: str, policy_fn, seeds: range) -> dict:
    rewards, delivereds = [], []
    for seed in seeds:
        total, info = rollout(policy_fn, seed=seed)
        rewards.append(total)
        delivereds.append(info["delivered"])
    rewards_arr = np.array(rewards, dtype=np.float64)
    return {
        "name": name,
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "mean_delivered": float(np.mean(delivereds)),
        "n_seeds": len(rewards),
    }


def main() -> None:
    seeds = range(20)
    probe_env = gym.make("DisasterResponseMission-v0")
    results = [
        evaluate_policy("random", random_policy(probe_env.action_space), seeds),
        evaluate_policy("parked_at_base (exploit)", parked_policy, seeds),
        evaluate_policy("scan_spam (exploit)", scan_spam_policy, seeds),
        evaluate_policy("hand_written_heuristic", heuristic_policy, seeds),
    ]
    probe_env.close()

    print(f"{'Policy':<28} {'MeanReward':>12} {'StdReward':>12} {'MeanDelivered':>14}")
    for r in results:
        print(f"{r['name']:<28} {r['mean_reward']:>12.2f} {r['std_reward']:>12.2f} {r['mean_delivered']:>14.2f}")

    parked = next(r for r in results if r["name"].startswith("parked"))
    scan_spam = next(r for r in results if r["name"].startswith("scan_spam"))
    heuristic = next(r for r in results if r["name"] == "hand_written_heuristic")
    random_r = next(r for r in results if r["name"] == "random")

    for exploit in (parked, scan_spam, random_r):
        assert heuristic["mean_reward"] > exploit["mean_reward"], (
            f"Heuristic must beat '{exploit['name']}' for the task to be worth training on."
        )
    assert heuristic["mean_delivered"] > parked["mean_delivered"], (
        "Heuristic must actually deliver kits, unlike the parked exploit."
    )
    assert heuristic["mean_delivered"] > scan_spam["mean_delivered"], (
        "Heuristic must actually deliver kits, unlike the scan-spam exploit."
    )

    print(
        "\nSanity check PASSED: hand-written heuristic "
        f"({heuristic['mean_reward']:.2f}) clears the parked-at-base exploit "
        f"({parked['mean_reward']:.2f}), the scan-spam exploit ({scan_spam['mean_reward']:.2f}), "
        f"and random policy ({random_r['mean_reward']:.2f})."
    )

    out_path = Path("logs") / "sanity_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
