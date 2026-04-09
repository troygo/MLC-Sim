import mujoco
import mujoco.viewer
import time

TEST_MODEL_PATH = "TestProject.xml"

#initialize model
model = mujoco.MjModel.from_xml_path(TEST_MODEL_PATH)
data = mujoco.MjData(model)

#model parameters
model.opt.timestep = 0.002

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)