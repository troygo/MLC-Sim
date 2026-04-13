"""
simulation.py
-------------
ArmSimulation: wraps the MuJoCo model, steps physics, and delegates
control to an external ML controller each step.
"""

import time
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from .controller import BaseMLController
from .types import ArmAction, ArmObservation

MODEL_PATH = Path(__file__).parent.parent / "models" / "arm.xml"


class ArmSimulation:
    """
    Runs the 2-DOF arm simulation and queries the ML controller each step.

    Physics:  500 Hz
    Control:   50 Hz  (every CONTROL_DECIMATION=10 physics sub-steps)
    """

    PHYSICS_HZ = 500
    CONTROL_HZ = 50
    CONTROL_DECIMATION = PHYSICS_HZ // CONTROL_HZ  # 10

    def __init__(self, controller: BaseMLController, render: bool = True):
        self.controller = controller
        self.render = render

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self._shoulder_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor"
        )
        self._elbow_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_motor"
        )
        self._ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site"
        )

        self.target_pos: Optional[np.ndarray] = np.array([0.4, 0.2, 0.1])
        self._update_target_marker()

        self.step_count = 0
        self.episode_reward = 0.0

    # ── public API ────────────────────────────────────────────────────

    def get_observation(self) -> ArmObservation:
        """Read current physics state into an ArmObservation."""
        return ArmObservation(
            joint_positions=self.data.sensordata[0:2].copy(),
            joint_velocities=self.data.sensordata[2:4].copy(),
            end_effector_pos=self.data.site_xpos[self._ee_site].copy(),
            target_pos=self.target_pos.copy() if self.target_pos is not None else None,
            time=self.data.time,
        )

    def apply_action(self, action: ArmAction) -> None:
        """Write ML model torques into the MuJoCo actuator array."""
        self.data.ctrl[self._shoulder_act] = action.torques[0]
        self.data.ctrl[self._elbow_act] = action.torques[1]

    def compute_reward(self, obs: ArmObservation) -> float:
        """
        Dense reward: negative Euclidean distance to target.
        Override or replace with your own reward shaping.
        """
        if obs.target_pos is None:
            return 0.0
        return -float(np.linalg.norm(obs.end_effector_pos - obs.target_pos))

    def set_target(self, pos: np.ndarray) -> None:
        """Move the target marker to a new world position."""
        self.target_pos = pos.copy()
        self._update_target_marker()

    def reset(self, randomise: bool = False) -> ArmObservation:
        """Reset physics; optionally randomise starting joint angles."""
        mujoco.mj_resetData(self.model, self.data)
        if randomise:
            self.data.qpos[0] = np.random.uniform(-1.5, 1.5)
            self.data.qpos[1] = np.random.uniform(-1.2, 1.2)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.episode_reward = 0.0
        self.controller.reset()
        return self.get_observation()

    def step(self) -> tuple[ArmObservation, float, bool]:
        """
        One control step = CONTROL_DECIMATION physics sub-steps.

        Returns:
            obs    — observation after the step
            reward — scalar reward
            done   — episode termination flag
        """
        obs = self.get_observation()
        action = self.controller.predict(obs)
        self.apply_action(action)

        for _ in range(self.CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self.get_observation()
        reward = self.compute_reward(obs)
        self.episode_reward += reward
        done = self._check_termination(obs)
        return obs, reward, done

    def run(
        self,
        max_steps: int = 2000,
        episode_steps: int = 300,
        randomise_reset: bool = True,
        print_freq: int = 50,
    ) -> None:
        """Main loop: runs episodes, calling the ML controller each step."""
        if self.render:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self._run_loop(viewer, max_steps, episode_steps, randomise_reset, print_freq)
        else:
            self._run_loop(None, max_steps, episode_steps, randomise_reset, print_freq)

    # ── internal helpers ──────────────────────────────────────────────

    def _run_loop(self, viewer, max_steps, episode_steps, randomise_reset, print_freq):
        obs = self.reset(randomise=randomise_reset)
        global_step = 0

        while global_step < max_steps:
            if viewer is not None and not viewer.is_running():
                break

            obs, reward, done = self.step()
            global_step += 1

            if global_step % print_freq == 0:
                ee = obs.end_effector_pos
                dist = (
                    np.linalg.norm(ee - self.target_pos)
                    if self.target_pos is not None
                    else float("nan")
                )
                print(
                    f"[step {global_step:>5}] "
                    f"t={obs.time:.2f}s | "
                    f"shoulder={np.degrees(obs.joint_positions[0]):+7.2f}°  "
                    f"elbow={np.degrees(obs.joint_positions[1]):+7.2f}°  | "
                    f"EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})  "
                    f"dist={dist:.4f}m  |  "
                    f"ep_reward={self.episode_reward:.2f}"
                )

            if done or self.step_count >= episode_steps:
                print(
                    f"\n── Episode ended at global step {global_step} "
                    f"| total reward = {self.episode_reward:.2f} ──\n"
                )
                self.set_target(self._random_reachable_target())
                obs = self.reset(randomise=randomise_reset)

            if viewer is not None:
                viewer.sync()
                time.sleep(self.model.opt.timestep * self.CONTROL_DECIMATION)

    def _check_termination(self, obs: ArmObservation) -> bool:
        if obs.target_pos is not None:
            dist = np.linalg.norm(obs.end_effector_pos - obs.target_pos)
            if dist < 0.03:
                print(f"  ✓ Target reached! dist={dist:.4f} m")
                return True
        return False

    def _update_target_marker(self) -> None:
        if self.target_pos is not None:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
            mocap_id = self.model.body_mocapid[body_id]
            if mocap_id >= 0:
                self.data.mocap_pos[mocap_id] = self.target_pos

    @staticmethod
    def _random_reachable_target() -> np.ndarray:
        r = np.random.uniform(0.10, 0.50)
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(theta), r * np.sin(theta), 0.1])
