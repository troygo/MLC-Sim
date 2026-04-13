"""
controller.py
-------------
Defines the ML controller interface (BaseMLController) and a placeholder
random controller (RandomMLController).

To plug in a real model, subclass BaseMLController and implement predict().
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .types import ArmObservation, ArmAction


class BaseMLController(ABC):
    """
    Abstract base class for an external ML controller.

    Implement this with your trained model (e.g. PPO, SAC, BC …).
    The simulation calls predict() once per control step.
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
