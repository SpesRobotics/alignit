# test_mujoco_import.py
import mujoco as mj
import sys

print(f"MuJoCo version: {mj.__version__}")
print(f"Path to mujoco package: {mj.__path__}")
print(f"Does mj have 'xml' attribute? {hasattr(mj, 'xml')}")
if hasattr(mj, 'xml'):
    print(f"Does mj.xml have 'load_model_from_path' attribute? {hasattr(mj.xml, 'load_model_from_path')}")
else:
    print("mj.xml submodule is missing.")

try:
    # Try to load a simple dummy model to ensure functionality
    model = mj.MjModel.from_xml_string('<mujoco><worldbody><geom name="sphere" type="sphere" size="0.1"/></worldbody></mujoco>')
    print("Successfully loaded a simple dummy model.")
except Exception as e:
    print(f"Error loading dummy model: {e}")