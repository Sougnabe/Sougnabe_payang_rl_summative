from __future__ import annotations

import argparse

from training.dqn_training import run_dqn_sweep
from training.pg_training import run_a2c_sweep, run_ppo_sweep, run_reinforce_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps")
    parser.add_argument(
        "--algo",
        choices=["dqn", "reinforce", "ppo", "a2c", "all"],
        default="all",
        help="Which sweep to run",
    )
    parser.add_argument("--timesteps", type=int, default=70_000)
    parser.add_argument("--seed", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.algo in {"dqn", "all"}:
        run_dqn_sweep(total_timesteps=args.timesteps, seed=args.seed)

    if args.algo in {"reinforce", "all"}:
        run_reinforce_sweep(total_timesteps=args.timesteps, seed=args.seed)

    if args.algo in {"ppo", "all"}:
        run_ppo_sweep(total_timesteps=args.timesteps, seed=args.seed)

    if args.algo in {"a2c", "all"}:
        run_a2c_sweep(total_timesteps=args.timesteps, seed=args.seed)


if __name__ == "__main__":
    main()
