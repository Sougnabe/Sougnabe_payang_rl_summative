from __future__ import annotations

import argparse
from pathlib import Path

from training.dqn_training import train_dqn
from training.pg_training import train_a2c, train_ppo, train_reinforce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mission RL runner")
    parser.add_argument(
        "--algo",
        choices=["dqn", "reinforce", "ppo", "a2c", "all"],
        default="all",
        help="Algorithm to run",
    )
    parser.add_argument("--timesteps", type=int, default=80_000, help="Training horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--render-best",
        action="store_true",
        help="Render a short rollout with the best checkpoint after training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    if args.algo in {"dqn", "all"}:
        train_dqn(total_timesteps=args.timesteps, seed=args.seed, render_best=args.render_best)

    if args.algo in {"reinforce", "all"}:
        train_reinforce(total_timesteps=args.timesteps, seed=args.seed, render_best=args.render_best)

    if args.algo in {"ppo", "all"}:
        train_ppo(total_timesteps=args.timesteps, seed=args.seed, render_best=args.render_best)

    if args.algo in {"a2c", "all"}:
        train_a2c(total_timesteps=args.timesteps, seed=args.seed, render_best=args.render_best)


if __name__ == "__main__":
    main()
