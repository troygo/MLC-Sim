import mujoco

TEST_MODEL_PATH = "TestProject.xml"

test_model = mujoco.MjModel.from_xml_path(TEST_MODEL_PATH)
test_data = mujoco.MjData(test_model)

viewer = mujoco.viewer.launch_passive(test_model, test_data)

