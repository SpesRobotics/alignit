# test_mujoco.py
import mujoco as mj
import mujoco.viewer
import time
import os

# Optional: Test with OSMesa if you're trying to pinpoint graphics issues
# os.environ['MUJOCO_GL'] = 'osmesa' 

# A very simple XML model
xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body name="box" pos="0 0 0.2">
      <joint name="box_joint" type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

try:
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)

    # Only try to launch viewer if not in headless mode
    if 'MUJOCO_GL' not in os.environ or os.environ['MUJOCO_GL'] != 'osmesa':
        viewer = mujoco.viewer.launch_passive(model, data)
        print("MuJoCo viewer launched successfully. Keeping open for 5 seconds.")
        viewer.sync()
        time.sleep(5)
        viewer.close()
        print("Viewer closed.")
    else:
        print("Running in OSMesa (headless) mode. No viewer will appear.")
        # Perform a few steps to ensure the model can be simulated
        for _ in range(100):
            mj.mj_step(model, data)
        print("Model stepped successfully in headless mode.")

    print("Minimal MuJoCo test successful.")
except Exception as e:
    print(f"Minimal MuJoCo test failed: {e}")
except KeyboardInterrupt:
    print("Test interrupted.")