"""
tests/test_simulation.py
------------------------
Basic sanity checks — no MuJoCo viewer required (render=False).
Run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ArmObservation, ArmAction, ArmSimulation, RandomMLController


@pytest.fixture
def sim():
    controller = RandomMLController(torque_scale=1.0, seed=0)
    return ArmSimulation(controller=controller, render=False)


def test_observation_shapes(sim):
    obs = sim.get_observation()
    assert obs.joint_positions.shape == (2,)
    assert obs.joint_velocities.shape == (2,)
    assert obs.end_effector_pos.shape == (3,)


def test_reset_zeroes_state(sim):
    # Run a few steps to dirty the state
    for _ in range(10):
        sim.step()
    sim.reset(randomise=False)
    obs = sim.get_observation()
    np.testing.assert_allclose(obs.joint_positions, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(obs.joint_velocities, [0.0, 0.0], atol=1e-6)


def test_step_returns_correct_types(sim):
    sim.reset()
    obs, reward, done = sim.step()
    assert isinstance(obs, ArmObservation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_reward_is_negative_distance(sim):
    sim.reset()
    obs, reward, _ = sim.step()
    dist = np.linalg.norm(obs.end_effector_pos - sim.target_pos)
    assert abs(reward - (-dist)) < 1e-6


def test_set_target_updates_position(sim):
    new_target = np.array([0.3, 0.1, 0.1])
    sim.set_target(new_target)
    np.testing.assert_array_equal(sim.target_pos, new_target)


def test_action_torques_applied(sim):
    sim.reset()
    action = ArmAction(torques=np.array([5.0, -5.0]))
    sim.apply_action(action)
    assert sim.data.ctrl[sim._shoulder_act] == pytest.approx(5.0)
    assert sim.data.ctrl[sim._elbow_act] == pytest.approx(-5.0)


def test_multiple_episodes(sim):
    for _ in range(3):
        sim.reset(randomise=True)
        for _ in range(20):
            obs, reward, done = sim.step()
            if done:
                break
