"""
types.py
--------
Shared data structures passed between the simulation and the ML controller.
No MuJoCo or heavy dependencies — safe to import anywhere.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ArmObservation:
    """Full state observation passed to the ML model each step."""
    joint_positions: np.ndarray       # shape (2,) — shoulder & elbow angles [rad]
    joint_velocities: np.ndarray      # shape (2,) — angular velocities [rad/s]
    end_effector_pos: np.ndarray      # shape (3,) — (x, y, z) in world frame [m]
    target_pos: Optional[np.ndarray]  # shape (3,) — goal position, if set
    time: float                       # simulation time [s]


@dataclass
class ArmAction:
    """Control output produced by the ML model."""
    torques: np.ndarray               # shape (2,) — shoulder & elbow torques [N·m]
