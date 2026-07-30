import numpy as np

from environment.custom_env import DisasterResponseEnv
from environment.rendering import MissionRenderer


def test_env_reset_and_step_shapes() -> None:
    env = DisasterResponseEnv()
    obs, info = env.reset(seed=123)

    assert obs.shape == (20,)
    assert "battery" in info
    assert "battery_frac" in info
    assert "recharged" in info

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (20,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "delivered" in info

    env.close()


def test_new_cell_visits_are_rewarded_more_strongly() -> None:
    env = DisasterResponseEnv()
    env.reset(seed=11)

    env.agent_pos = np.array([10.0, 10.0], dtype=np.float32)
    env._visited_cells = set()
    env._recent_cells = []
    env._recent_positions = [env.agent_pos.copy()]

    _, reward, _, _, _ = env.step(0)
    env.close()

    assert reward > 0.05


def test_env_rgb_render() -> None:
    env = DisasterResponseEnv(render_mode="rgb_array")
    env.reset(seed=1)
    frame = env.render()
    env.close()

    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3


def test_env_rgb_render_contains_scene_features() -> None:
    env = DisasterResponseEnv(render_mode="rgb_array")
    env.reset(seed=1)
    frame = env.render()
    env.close()

    # 3D scene should have non-zero content (not all black or uniform)
    assert frame.shape == (700, 900, 3)
    assert frame.dtype == np.uint8
    # Verify there's color variation (scene has ground, hazards, victims, base, drone)
    unique_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
    assert unique_colors > 10, f"Expected >10 unique colors in 3D scene, got {unique_colors}"
    # Verify the frame is not completely dark
    mean_brightness = float(np.mean(frame))
    assert mean_brightness > 10.0, f"Frame too dark: mean brightness={mean_brightness}"


def test_looping_motion_is_penalized() -> None:
    env = DisasterResponseEnv()
    env.reset(seed=5)
    env.wind_scale = 0.0

    rewards = []
    for action in [3, 4, 3, 4, 3, 4, 3, 4]:
        _, reward, _, _, _ = env.step(action)
        rewards.append(float(reward))

    env.close()

    assert sum(rewards) < 0.0


def test_passive_discovery_when_close() -> None:
    env = DisasterResponseEnv()
    env.reset(seed=19)

    site = env.victim_sites[0]
    env.agent_pos = site.position + np.array([1.0, 0.0], dtype=np.float32)
    site.discovered = False

    _, _, _, _, _ = env.step(0)

    env.close()

    assert site.discovered is True


def test_renderer_requests_close_on_escape_key() -> None:
    renderer = MissionRenderer(world_size=100.0)
    renderer.on_key_press(27, 0)

    assert renderer._close_requested is True


def test_renderer_toggles_pause_on_spacebar() -> None:
    renderer = MissionRenderer(world_size=100.0)
    renderer.on_key_press(32, 0)
    assert renderer._paused is True
    renderer.on_key_press(32, 0)
    assert renderer._paused is False


def test_charging_docks_at_base_no_orbit() -> None:
    env = DisasterResponseEnv()
    env.reset(seed=21)

    env.battery = 0.5 * env.battery_capacity
    env.agent_pos = env.base_pos + np.array([env.recharge_radius * 0.7, 0.0], dtype=np.float32)

    for _ in range(30):
        _, _, done, truncated, _ = env.step(0)
        if done or truncated:
            break

    dist_to_base = float(np.linalg.norm(env.agent_pos - env.base_pos))
    env.close()

    assert dist_to_base < 1e-3


def test_stale_explored_zone_moves_faster() -> None:
    env = DisasterResponseEnv()
    env.reset(seed=33)

    env.wind_scale = 0.0
    env.agent_pos = np.array([50.0, 50.0], dtype=np.float32)
    env._visited_cells.add(env._cell_index(env.agent_pos))
    env._steps_since_new_cell = env.stale_steps_for_boost + 2

    start = env.agent_pos.copy()
    env.step(3)
    displacement = float(np.linalg.norm(env.agent_pos - start))
    env.close()

    assert displacement > env.move_speed


def test_revisit_penalty_stronger_when_far_from_victim_signal() -> None:
    def step_reward_at(pos: np.ndarray) -> float:
        env = DisasterResponseEnv()
        env.reset(seed=34)
        env.wind_scale = 0.0
        env.agent_pos = pos.copy()
        env._visited_cells.add(env._cell_index(env.agent_pos))
        env._steps_since_new_cell = env.stale_steps_for_boost + 2
        _, reward, _, _, _ = env.step(0)
        env.close()
        return float(reward)

    far_reward = step_reward_at(np.array([50.0, 50.0], dtype=np.float32))
    near_site = np.array([80.0, 20.0], dtype=np.float32) + np.array([8.5, 0.0], dtype=np.float32)
    near_reward = step_reward_at(near_site)

    assert far_reward < near_reward


def test_battery_recharges_near_base() -> None:
    env = DisasterResponseEnv()
    env.reset(seed=7)

    # Spend some battery first.
    for _ in range(40):
        env.step(4)

    battery_before = env.battery

    # Use emergency return and hover to trigger recharge at base.
    recharged_values = []
    for _ in range(30):
        _, _, done, truncated, info = env.step(11)
        recharged_values.append(float(info["recharged"]))
        if done or truncated:
            break
    for _ in range(10):
        _, _, done, truncated, info = env.step(0)
        recharged_values.append(float(info["recharged"]))
        if done or truncated:
            break

    env.close()

    assert any(v > 0.0 for v in recharged_values)
    assert env.battery > battery_before
