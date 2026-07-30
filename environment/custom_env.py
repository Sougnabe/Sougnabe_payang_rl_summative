from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .rendering import MissionRenderer, RenderState


@dataclass
class VictimSite:
    position: np.ndarray
    discovered: bool = False
    pending: bool = True


class DisasterResponseEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode

        self.world_size = 100.0
        self.max_steps = 420
        self.move_speed = 1.8
        self.wind_scale = 0.55
        self.scan_radius = 9.5
        self.delivery_radius = 5.5
        self.no_victim_relief_radius = 1.4 * self.scan_radius
        self.battery_capacity = 100.0
        self.recharge_radius = 7.0
        self.charge_time_seconds = 7.5
        self.drain_time_seconds = 15.0
        self.stale_steps_for_boost = 5
        self.stale_speed_boost = 1.75
        self.no_victim_revisit_penalty = 0.35
        self.step_cost = 0.05
        self.battery_safety_margin = 1.15

        fps = float(self.metadata.get("render_fps", 30))
        # Make the battery drain over about 15 seconds and recharge over about 7.5 seconds.
        self.base_drain_rate = self.battery_capacity / (self.drain_time_seconds * fps)
        self.movement_drain_rate = 0.12 * self.base_drain_rate
        self.scan_drain_extra = 0.06 * self.base_drain_rate
        self.return_drain_extra = 0.08 * self.base_drain_rate
        self.recharge_rate = self.battery_capacity / (self.charge_time_seconds * fps)

        self.base_pos = np.array([10.0, 10.0], dtype=np.float32)
        self.hazard_centers = np.array([[30.0, 70.0], [65.0, 35.0], [75.0, 80.0]], dtype=np.float32)
        self.hazard_radii = np.array([8.0, 10.0, 7.0], dtype=np.float32)

        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(20,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.renderer = MissionRenderer(self.world_size)

        self.agent_pos = self.base_pos.copy()
        self.wind = np.array([0.0, 0.0], dtype=np.float32)
        self.battery = self.battery_capacity
        self.kits_left = 3
        self.step_count = 0
        self._visited_cells: set[tuple[int, int]] = set()
        self._last_action: int | None = None
        self._repeat_count = 0
        self._recent_positions: list[np.ndarray] = []
        self._recent_cells: list[tuple[int, int]] = []
        self._steps_since_new_cell = 0
        self.last_recharged = 0.0
        self._auto_returning = False
        self.victim_sites = self._build_victim_sites()

        self.reset(seed=seed)

    def _build_victim_sites(self) -> list[VictimSite]:
        return [
            VictimSite(position=np.array([80.0, 20.0], dtype=np.float32)),
            VictimSite(position=np.array([25.0, 85.0], dtype=np.float32)),
            VictimSite(position=np.array([90.0, 75.0], dtype=np.float32)),
        ]

    def _movement_vector(self, action: int) -> np.ndarray:
        vectors = {
            0: np.array([0.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 1.0], dtype=np.float32),
            2: np.array([0.0, -1.0], dtype=np.float32),
            3: np.array([1.0, 0.0], dtype=np.float32),
            4: np.array([-1.0, 0.0], dtype=np.float32),
            5: np.array([1.0, 1.0], dtype=np.float32),
            6: np.array([-1.0, 1.0], dtype=np.float32),
            7: np.array([1.0, -1.0], dtype=np.float32),
            8: np.array([-1.0, -1.0], dtype=np.float32),
        }
        if action == 11:
            to_base = self.base_pos - self.agent_pos
            norm = np.linalg.norm(to_base) + 1e-8
            return (to_base / norm).astype(np.float32)

        move = vectors.get(action, np.array([0.0, 0.0], dtype=np.float32))
        norm = np.linalg.norm(move)
        if norm > 0:
            move = move / norm
        return move

    def _get_obs(self) -> np.ndarray:
        to_base = (self.base_pos - self.agent_pos) / self.world_size
        battery_frac = np.clip(self.battery / self.battery_capacity, 0.0, 1.0)

        obs: list[float] = [
            (self.agent_pos[0] / self.world_size) * 2 - 1,
            (self.agent_pos[1] / self.world_size) * 2 - 1,
            np.clip(to_base[0], -1.0, 1.0),
            np.clip(to_base[1], -1.0, 1.0),
            battery_frac,
            self.kits_left / 3.0,
            np.clip(self.wind[0], -1.0, 1.0),
            np.clip(self.wind[1], -1.0, 1.0),
        ]

        for site in self.victim_sites:
            rel = (site.position - self.agent_pos) / self.world_size
            obs.extend([
                float(np.clip(rel[0], -1.0, 1.0)),
                float(np.clip(rel[1], -1.0, 1.0)),
                1.0 if site.pending else -1.0,
                1.0 if site.discovered else -1.0,
            ])

        return np.array(obs, dtype=np.float32)

    def _closest_hazard_penalty(self) -> float:
        dists = np.linalg.norm(self.hazard_centers - self.agent_pos, axis=1)
        overlaps = self.hazard_radii - dists
        danger = float(np.max(overlaps))
        return -1.0 if danger > 0 else -0.01 * max(0.0, danger)

    def _all_delivered(self) -> bool:
        return all(not site.pending for site in self.victim_sites)

    def _cell_index(self, pos: np.ndarray) -> tuple[int, int]:
        # A coarse occupancy grid rewards genuine map coverage and discourages loops.
        grid = 10
        ix = int(np.clip(pos[0] / self.world_size * grid, 0, grid - 1))
        iy = int(np.clip(pos[1] / self.world_size * grid, 0, grid - 1))
        return ix, iy

    def _target_distance(self) -> float:
        discovered_pending = [site for site in self.victim_sites if site.pending and site.discovered]
        if discovered_pending:
            return float(min(np.linalg.norm(site.position - self.agent_pos) for site in discovered_pending))

        undiscovered_pending = [site for site in self.victim_sites if site.pending and not site.discovered]
        if undiscovered_pending:
            return float(min(np.linalg.norm(site.position - self.agent_pos) for site in undiscovered_pending))

        return float(np.linalg.norm(self.base_pos - self.agent_pos))

    def _distance_to_undiscovered_pending(self) -> float | None:
        undiscovered_pending = [site for site in self.victim_sites if site.pending and not site.discovered]
        if not undiscovered_pending:
            return None
        return float(min(np.linalg.norm(site.position - self.agent_pos) for site in undiscovered_pending))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agent_pos = self.base_pos.copy()
        self.wind = np.array([0.0, 0.0], dtype=np.float32)
        self.battery = self.battery_capacity
        self.kits_left = 3
        self.step_count = 0
        self._visited_cells = {self._cell_index(self.agent_pos)}
        self._last_action = None
        self._repeat_count = 0
        self._recent_positions = [self.agent_pos.copy()]
        self._recent_cells = [self._cell_index(self.agent_pos)]
        self._steps_since_new_cell = 0
        self.last_recharged = 0.0
        self._auto_returning = False
        self.victim_sites = self._build_victim_sites()

        return self._get_obs(), self._info()

    def _info(self) -> dict[str, Any]:
        delivered = sum(0 if site.pending else 1 for site in self.victim_sites)
        return {
            "battery": self.battery,
            "battery_frac": np.clip(self.battery / self.battery_capacity, 0.0, 1.0),
            "recharged": self.last_recharged,
            "kits_left": self.kits_left,
            "delivered": delivered,
            "step": self.step_count,
            "auto_return": self._auto_returning,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        reward = -self.step_cost
        self.last_recharged = 0.0

        action = int(action)

        # Auto return-to-base safety override: once the remaining battery is just
        # enough (with a small margin) to reach base at the current distance, force
        # a return regardless of what the agent chose. Mirrors a real drone's
        # auto-RTL failsafe and removes "died from battery neglect" as an outcome.
        dist_to_base_now = float(np.linalg.norm(self.base_pos - self.agent_pos))
        self._auto_returning = False
        if dist_to_base_now > self.recharge_radius:
            steps_to_base = dist_to_base_now / self.move_speed
            per_step_return_cost = self.base_drain_rate + self.movement_drain_rate + self.return_drain_extra
            battery_needed = steps_to_base * per_step_return_cost * self.battery_safety_margin
            if self.battery <= battery_needed:
                action = 11
                self._auto_returning = True

        prev_target_dist = self._target_distance()
        prev_undiscovered_dist = self._distance_to_undiscovered_pending()
        prev_base_dist = float(np.linalg.norm(self.base_pos - self.agent_pos))

        if self._last_action == action:
            self._repeat_count += 1
        else:
            self._repeat_count = 1
        self._last_action = action

        stale_no_target = (
            self._steps_since_new_cell >= self.stale_steps_for_boost
            and prev_undiscovered_dist is not None
            and prev_undiscovered_dist > self.no_victim_relief_radius
        )

        if stale_no_target:
            reward -= self.no_victim_revisit_penalty

        speed_multiplier = 1.0
        if stale_no_target and action in {1, 2, 3, 4, 5, 6, 7, 8}:
            speed_multiplier = self.stale_speed_boost

        move = self._movement_vector(action)
        self.wind = self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32) * self.wind_scale
        delta = (self.move_speed * speed_multiplier) * move + self.wind
        self.agent_pos = self.agent_pos + delta

        # While charging near base, lock position to prevent wind-driven circling.
        battery_frac_before_recharge = float(np.clip(self.battery / self.battery_capacity, 0.0, 1.0))
        dist_to_base_after_move = float(np.linalg.norm(self.base_pos - self.agent_pos))
        if dist_to_base_after_move <= self.recharge_radius and battery_frac_before_recharge < 0.98 and action in (0, 11):
            self.agent_pos = self.base_pos.copy()
            self.wind = np.array([0.0, 0.0], dtype=np.float32)

        out_of_bounds = np.any(self.agent_pos < 0.0) or np.any(self.agent_pos > self.world_size)
        if out_of_bounds:
            self.agent_pos = np.clip(self.agent_pos, 0.0, self.world_size)
            reward -= 0.5

        # Exploration reward for visiting new cells
        cell = self._cell_index(self.agent_pos)
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            reward += 1.0
            self._steps_since_new_cell = 0
        else:
            self._steps_since_new_cell += 1

        # Battery drain
        max_delta = self.move_speed * speed_multiplier + self.wind_scale * np.sqrt(2.0)
        movement_intensity = float(np.clip(np.linalg.norm(delta) / max_delta, 0.0, 1.5))
        travel_cost = self.base_drain_rate + self.movement_drain_rate * movement_intensity
        if action == 9:
            travel_cost += self.scan_drain_extra
        if action == 11:
            travel_cost += self.return_drain_extra
        self.battery -= travel_cost

        hazard = self._closest_hazard_penalty()
        reward += hazard

        # Encourage going back to base when battery is low.
        new_base_dist = float(np.linalg.norm(self.base_pos - self.agent_pos))
        battery_frac = float(np.clip(self.battery / self.battery_capacity, 0.0, 1.0))
        if battery_frac < 0.35:
            toward_base = np.clip((prev_base_dist - new_base_dist) / self.world_size, -0.25, 0.25)
            reward += 2.0 * toward_base
        # Mild penalty for low battery when far from base
        if battery_frac < 0.15 and new_base_dist > self.recharge_radius:
            reward -= 1.0

        # Recharge at base area (mechanical only: no reward, so idling at base is not
        # a free-money strategy relative to the per-step cost above).
        if new_base_dist <= self.recharge_radius and self.battery > 0.0 and self.battery < self.battery_capacity:
            recharge_amount = min(self.battery_capacity - self.battery, self.recharge_rate)
            if recharge_amount > 0.0:
                self.battery += recharge_amount
                self.last_recharged = float(recharge_amount)

        # Passive detection
        passive_reveals = 0
        for site in self.victim_sites:
            if not site.discovered and site.pending:
                distance = float(np.linalg.norm(site.position - self.agent_pos))
                if distance <= self.scan_radius * 0.8:
                    site.discovered = True
                    passive_reveals += 1
        if passive_reveals > 0:
            reward += 30.0 * passive_reveals

        # Scan action (reward only for an actual discovery; a "near-miss" bonus here would
        # let the agent farm reward by scanning just outside scan_radius forever without
        # ever closing the distance).
        if action == 9:
            for site in self.victim_sites:
                if not site.discovered and site.pending:
                    distance = float(np.linalg.norm(site.position - self.agent_pos))
                    if distance <= self.scan_radius:
                        site.discovered = True
                        reward += 40.0

        # Deliver action
        if action == 10:
            delivered_here = False
            for site in self.victim_sites:
                if site.pending and site.discovered and self.kits_left > 0:
                    distance = float(np.linalg.norm(site.position - self.agent_pos))
                    if distance <= self.delivery_radius:
                        site.pending = False
                        self.kits_left -= 1
                        delivered_here = True
                        reward += 50.0
            if not delivered_here:
                reward -= 0.5

        # Progress toward undiscovered victims (strong shaping)
        if prev_undiscovered_dist is not None:
            new_undiscovered_dist = self._distance_to_undiscovered_pending()
            if new_undiscovered_dist is not None:
                discover_progress = np.clip(
                    (prev_undiscovered_dist - new_undiscovered_dist) / self.world_size,
                    -0.5,
                    0.5,
                )
                reward += 25.0 * discover_progress

        # Progress toward nearest target (strong shaping)
        new_target_dist = self._target_distance()
        progress = np.clip((prev_target_dist - new_target_dist) / self.world_size, -0.5, 0.5)
        reward += 12.0 * progress

        # Mission completion
        if self._all_delivered():
            dist_to_base = float(np.linalg.norm(self.base_pos - self.agent_pos))
            if dist_to_base <= self.delivery_radius:
                reward += 150.0
                obs = self._get_obs()
                return obs, reward, True, False, self._info()
            reward += 2.0

        # Battery death
        if self.battery <= 0.0:
            reward -= 10.0
            obs = self._get_obs()
            return obs, reward, True, False, self._info()

        truncated = self.step_count >= self.max_steps
        if truncated and self._all_delivered():
            reward += 30.0

        obs = self._get_obs()
        return obs, reward, False, truncated, self._info()

    def _render_state(self) -> RenderState:
        victim_positions = np.stack([site.position for site in self.victim_sites], axis=0)
        victim_pending = np.array([site.pending for site in self.victim_sites], dtype=bool)
        victim_discovered = np.array([site.discovered for site in self.victim_sites], dtype=bool)

        return RenderState(
            agent_pos=self.agent_pos,
            base_pos=self.base_pos,
            victim_positions=victim_positions,
            victim_pending=victim_pending,
            victim_discovered=victim_discovered,
            hazard_centers=self.hazard_centers,
            hazard_radii=self.hazard_radii,
            battery=float(np.clip(self.battery / self.battery_capacity, 0.0, 1.0)),
            kits_left=int(self.kits_left),
            step_count=int(self.step_count),
            max_steps=int(self.max_steps),
        )

    def render(self) -> np.ndarray | bool | None:
        state = self._render_state()
        if self.render_mode == "rgb_array":
            return self.renderer.render_rgb(state)
        if self.render_mode == "human":
            return self.renderer.render_human(state)
        return None

    def close(self) -> None:
        self.renderer.close()