"""
scripts/view.py
---------------
Launch the MuJoCo passive viewer for the arm model.
"""

from pathlib import Path

import mujoco


MODEL_PATH = Path(__file__).parent.parent / "models" / "2dof-robot" / "robot.mjcf.xml"


def main() -> None:
    # model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mjModel* m = mj_loadXML(MODEL_PATH,)


if __name__ == "__main__":
    main()
