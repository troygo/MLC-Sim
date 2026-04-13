"""
scripts/run.py
--------------
Entry point. Swap out RandomMLController for your trained model here.
"""

import sys
from pathlib import Path

# Allow `python scripts/run.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ArmSimulation, RandomMLController


def main() -> None:
    print("=" * 60)
    print(" 2-DOF Arm Simulation with ML Controller Interface")
    print("=" * 60)
    print()
    print("Controller : RandomMLController  (placeholder)")
    print("Replace    : Subclass BaseMLController and pass it below")
    print()

    # ── Swap this line to plug in your real model ──────────────────
    controller = RandomMLController(torque_scale=3.0, seed=42)
    # from src.ppo_controller import PPOController
    # controller = PPOController(checkpoint="checkpoints/best.pth")
    # ──────────────────────────────────────────────────────────────

    sim = ArmSimulation(controller=controller, render=True)
    sim.run(
        max_steps=5000,
        episode_steps=300,
        randomise_reset=True,
        print_freq=50,
    )


if __name__ == "__main__":
    main()
