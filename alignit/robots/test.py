import mujoco
from mujoco import viewer
import threading

def safe_viewer():
    model = mujoco.MjModel.from_xml_path("lite6mjcf.xml")
    data = mujoco.MjData(model)
    with viewer.launch_passive(model, data) as v:
        while v.is_running():  # Safer than while True
            mujoco.mj_step(model, data)
            v.sync()

# Run in main thread (don't spawn new threads)
safe_viewer()