# 2-DOF Arm Simulation

A MuJoCo-based 2-DOF planar robotic arm with a clean interface for plugging in an external ML controller.

## Structure

```
2dof_arm/
├── src/
│   ├── __init__.py          # Public re-exports
│   ├── types.py             # ArmObservation, ArmAction dataclasses
│   ├── controller.py        # BaseMLController + RandomMLController
│   └── simulation.py        # ArmSimulation (physics loop)
├── models/
│   └── arm.xml              # MuJoCo XML model
├── scripts/
│   └── run.py               # Entry point
├── tests/
│   └── test_simulation.py   # Pytest unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python scripts/run.py
```

## Plug in your ML model

Subclass `BaseMLController` and implement `predict()`:

```python
# src/ppo_controller.py
from src import BaseMLController, ArmObservation, ArmAction

class PPOController(BaseMLController):
    def __init__(self, checkpoint: str):
        self.model = load_your_model(checkpoint)

    def predict(self, observation: ArmObservation) -> ArmAction:
        obs_tensor = observation_to_tensor(observation)
        torques = self.model(obs_tensor)
        return ArmAction(torques=torques.numpy())
```

Then swap it in `scripts/run.py`:

```python
from src.ppo_controller import PPOController
controller = PPOController(checkpoint="checkpoints/best.pth")
```

## Tests

```bash
pytest tests/
```
