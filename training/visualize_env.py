from __future__ import annotations

from pathlib import Path

import imageio

from environment.custom_env import DisasterResponseEnv


def main() -> None:
    out = Path("assets") / "environment_snapshot.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    env = DisasterResponseEnv(render_mode="rgb_array")
    env.reset(seed=11)
    frame = env.render()
    env.close()

    if frame is None:
        raise RuntimeError("Environment did not return an RGB frame.")

    imageio.imwrite(out, frame)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
