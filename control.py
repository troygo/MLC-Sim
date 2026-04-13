"""
claude chat: https://claude.ai/chat/4ba27c92-e0a0-4bf5-a9ba-c7488c5b4a94

2 DOF Robotic Arm Simulation using MuJoCo
==========================================
This module simulates a 2-DOF planar robotic arm in MuJoCo.
Control inputs are provided by an external ML model via a clean interface.

Dependencies:
    pip install mujoco numpy

Usage:
    python arm_2dof.py


"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class ArmObservation:
    """Full state observation passed to the ML model each step."""
    joint_positions: np.ndarray      # shape (2,)  — shoulder & elbow angles [rad]
    joint_velocities: np.ndarray     # shape (2,)  — angular velocities [rad/s]
    end_effector_pos: np.ndarray     # shape (3,)  — (x, y, z) in world frame [m]
    target_pos: Optional[np.ndarray] # shape (3,)  — goal position, if set
    time: float                      # simulation time [s]


@dataclass
class ArmAction:
    """Control output produced by the ML model."""
    torques: np.ndarray              # shape (2,)  — shoulder & elbow torques [N·m]


# ─────────────────────────────────────────────
#  ML model interface (plug your model here)
# ─────────────────────────────────────────────

class BaseMLController(ABC):
    """
    Abstract base class for an external ML controller.

    Implement this class with your trained model (e.g. PPO, SAC, BC …).
    The simulation will call `predict()` once per control step.
    """

    @abstractmethod
    def predict(self, observation: ArmObservation) -> ArmAction:
        """
        Given the current arm state, return joint torques.

        Args:
            observation: Current arm state.

        Returns:
            ArmAction with torques for shoulder (index 0) and elbow (index 1).
        """
        ...

    def reset(self) -> None:
        """Called at the start of each episode. Override if your model is stateful."""
        pass


class RandomMLController(BaseMLController):
    """
    Placeholder random controller — replace with your trained model.

    Generates smooth random torques via a low-pass filtered random walk so the
    arm moves visibly without flying apart.
    """

    def __init__(self, torque_scale: float = 1.0, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self._torque_scale = torque_scale
        self._current = np.zeros(2)

    def predict(self, observation: ArmObservation) -> ArmAction:
        # Smooth random walk (low-pass filter)
        noise = self._rng.uniform(-1.0, 1.0, size=2)
        self._current = 0.85 * self._current + 0.15 * noise
        torques = np.clip(self._current * self._torque_scale, -5.0, 5.0)
        return ArmAction(torques=torques)

    def reset(self) -> None:
        self._current = np.zeros(2)


# ─────────────────────────────────────────────
#  MuJoCo XML model
# ─────────────────────────────────────────────

ARM_XML = """
<mujoco model="2dof_arm">

  <compiler angle="radian"/>

  <option gravity="0 0 -9.81" timestep="0.002"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="link"        rgba="0.3 0.6 0.9 1"/>
    <material name="joint_cap"   rgba="0.9 0.5 0.1 1"/>
    <material name="ee"          rgba="0.2 0.9 0.2 1"/>
    <material name="target"      rgba="1.0 0.2 0.2 0.7"/>
  </asset>

  <worldbody>

    <!-- Ground -->
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>

    <!-- Fixed base -->
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.1" material="joint_cap"/>

      <!-- === Link 1 (shoulder) === -->
      <body name="link1" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 0 1"
               range="-3.14159 3.14159" damping="0.5" armature="0.1"/>
        <!-- upper arm geometry -->
        <geom type="capsule" fromto="0 0 0  0.3 0 0" size="0.025" material="link"/>
        <!-- shoulder cap -->
        <geom type="sphere" pos="0 0 0" size="0.035" material="joint_cap"/>

        <!-- === Link 2 (elbow) === -->
        <body name="link2" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 0 1"
                 range="-2.617 2.617" damping="0.3" armature="0.05"/>
          <!-- forearm geometry -->
          <geom type="capsule" fromto="0 0 0  0.25 0 0" size="0.02" material="link"/>
          <!-- elbow cap -->
          <geom type="sphere" pos="0 0 0" size="0.030" material="joint_cap"/>
          <!-- end-effector marker -->
          <geom name="end_effector" type="sphere" pos="0.25 0 0" size="0.025" material="ee"/>
          <!-- end-effector site (used to read position) -->
          <site name="end_effector_site" pos="0.25 0 0" size="0.01"/>
        </body>
      </body>
    </body>

    <!-- Target marker (visual only – repositioned at runtime) -->
    <body name="target" pos="0.4 0.2 0.1" mocap="true">
      <geom type="sphere" size="0.03" material="target" contype="0" conaffinity="0"/>
    </body>

  </worldbody>

  <!-- Actuators — one per joint -->
  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
    <motor name="elbow_motor"    joint="elbow"    gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>

  <!-- Sensors -->
  <sensor>
    <jointpos  name="shoulder_pos" joint="shoulder"/>
    <jointpos  name="elbow_pos"    joint="elbow"/>
    <jointvel  name="shoulder_vel" joint="shoulder"/>
    <jointvel  name="elbow_vel"    joint="elbow"/>
    <framepos  name="ee_pos"       objtype="site" objname="end_effector_site"/>
  </sensor>

</mujoco>
"""


# ─────────────────────────────────────────────
#  Simulation
# ─────────────────────────────────────────────

class ArmSimulation:
    """
    Wraps the MuJoCo model and steps the physics at a fixed rate,
    delegating control to an external ML controller each step.
    """

    # Physics runs at 500 Hz; ML controller queried every CONTROL_DECIMATION steps → 50 Hz
    PHYSICS_HZ       = 500
    CONTROL_HZ       = 50
    CONTROL_DECIMATION = PHYSICS_HZ // CONTROL_HZ  # 10

    def __init__(self, controller: BaseMLController, render: bool = True):
        self.controller = controller
        self.render = render

        # Load model
        self.model = mujoco.MjModel.from_xml_string(ARM_XML)
        self.data  = mujoco.MjData(self.model)

        # Cache actuator / sensor indices
        self._shoulder_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")
        self._elbow_act    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_motor")
        self._ee_site      = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,     "end_effector_site")

        # Target position (updated externally / by curriculum)
        self.target_pos: Optional[np.ndarray] = np.array([0.4, 0.2, 0.1])
        self._update_target_marker()

        # Metrics
        self.step_count = 0
        self.episode_reward = 0.0

    # ── public API ────────────────────────────────────────────────────

    def get_observation(self) -> ArmObservation:
        """Read current physics state and return a clean observation object."""
        return ArmObservation(
            joint_positions  = self.data.sensordata[0:2].copy(),
            joint_velocities = self.data.sensordata[2:4].copy(),
            end_effector_pos = self.data.site_xpos[self._ee_site].copy(),
            target_pos       = self.target_pos.copy() if self.target_pos is not None else None,
            time             = self.data.time,
        )

    def apply_action(self, action: ArmAction) -> None:
        """Write ML model torques into the MuJoCo actuator array."""
        self.data.ctrl[self._shoulder_act] = action.torques[0]
        self.data.ctrl[self._elbow_act]    = action.torques[1]

    def compute_reward(self, obs: ArmObservation) -> float:
        """
        Simple dense reward: negative distance to target.
        Override or replace with your own reward shaping.
        """
        if obs.target_pos is None:
            return 0.0
        dist = np.linalg.norm(obs.end_effector_pos - obs.target_pos)
        return -dist

    def set_target(self, pos: np.ndarray) -> None:
        """Move the target marker to a new position."""
        self.target_pos = pos.copy()
        self._update_target_marker()

    def reset(self, randomise: bool = False) -> ArmObservation:
        """Reset physics and optionally randomise starting joint angles."""
        mujoco.mj_resetData(self.model, self.data)
        if randomise:
            self.data.qpos[0] = np.random.uniform(-1.5, 1.5)  # shoulder
            self.data.qpos[1] = np.random.uniform(-1.2, 1.2)  # elbow
        mujoco.mj_forward(self.model, self.data)
        self.step_count    = 0
        self.episode_reward = 0.0
        self.controller.reset()
        return self.get_observation()

    def step(self) -> tuple[ArmObservation, float, bool]:
        """
        Run one *control* step (CONTROL_DECIMATION physics sub-steps).

        Returns:
            obs     — observation after the step
            reward  — scalar reward
            done    — episode termination flag
        """
        obs    = self.get_observation()
        action = self.controller.predict(obs)
        self.apply_action(action)

        for _ in range(self.CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs    = self.get_observation()
        reward = self.compute_reward(obs)
        self.episode_reward += reward

        done = self._check_termination(obs)
        return obs, reward, done

    def run(
        self,
        max_steps: int        = 2000,
        episode_steps: int    = 500,
        randomise_reset: bool = True,
        print_freq: int       = 100,
    ) -> None:
        """
        Main loop: runs episodes, calling the ML controller each step.
        If render=True a MuJoCo viewer is opened.
        """
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
                dist = np.linalg.norm(ee - self.target_pos) if self.target_pos is not None else float("nan")
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
                print(f"\n── Episode ended at global step {global_step} "
                      f"| total reward = {self.episode_reward:.2f} ──\n")
                # Cycle target to a new random reachable position
                self.set_target(self._random_reachable_target())
                obs = self.reset(randomise=randomise_reset)

            if viewer is not None:
                viewer.sync()
                # Throttle rendering to real time
                time.sleep(self.model.opt.timestep * self.CONTROL_DECIMATION)

    def _check_termination(self, obs: ArmObservation) -> bool:
        """Return True when the episode should end (goal reached, fallen, etc.)."""
        if obs.target_pos is not None:
            dist = np.linalg.norm(obs.end_effector_pos - obs.target_pos)
            if dist < 0.03:          # within 3 cm → success
                print(f"  ✓ Target reached! dist={dist:.4f} m")
                return True
        return False

    def _update_target_marker(self) -> None:
        if self.target_pos is not None:
            # mocap body index for 'target'
            body_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
            mocap_id = self.model.body_mocapid[body_id]
            if mocap_id >= 0:
                self.data.mocap_pos[mocap_id] = self.target_pos

    @staticmethod
    def _random_reachable_target() -> np.ndarray:
        """Sample a target inside the arm's reachable workspace (0.05 m – 0.55 m radius)."""
        r     = np.random.uniform(0.10, 0.50)
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(theta), r * np.sin(theta), 0.1])


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" 2-DOF Arm Simulation with ML Controller Interface")
    print("=" * 60)
    print()
    print("Controller : RandomMLController  (placeholder)")
    print("Replace    : Subclass BaseMLController and pass it to ArmSimulation")
    print()

    # ── Swap this line to plug in your real model ──────────────────────
    controller = RandomMLController(torque_scale=3.0, seed=42)
    # controller = YourTrainedController(checkpoint="model.pth")
    # ──────────────────────────────────────────────────────────────────

    sim = ArmSimulation(controller=controller, render=True)

    sim.run(
        max_steps       = 5000,
        episode_steps   = 300,
        randomise_reset = True,
        print_freq      = 50,
    )


if __name__ == "__main__":
    main()