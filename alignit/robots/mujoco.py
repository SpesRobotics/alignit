from teleop.utils.jacobi_robot import JacobiRobot
import mujoco.viewer
import numpy as np
import time
import os

# Ensure you have 'pinocchio' installed for JacobiRobot to work
# from teleop.utils.jacobi_robot import JacobiRobot # Uncomment if you've fixed teleop import

# Path to your downloaded lite6.urdf file
LITE6_URDF_PATH = "/home/nikola/code/alignit/alignit/lite6.urdf"
# Path where the converted MJCF will be saved
LITE6_MJCF_PATH = "/home/nikola/code/alignit/alignit/lite6_converted.xml" # A new path for the converted file

# The end-effector link name for the xArm Lite6.
LITE6_EE_LINK = "link6"


class MuJoCo_xArmLite6:
    def __init__(self, model_path=None, is_urdf=True): # Added is_urdf flag
        if model_path is None:
            raise ValueError("Robot model path must be provided")

        print(f"\n--- Attempting to load model from: {model_path} ---")
        try:
            with open(model_path, 'r') as f:
                file_content = f.read()
                print("Content of file being loaded (last 500 characters):\n", file_content[-500:])
        except FileNotFoundError:
            print(f"Error: File not found at {model_path}")
            exit()
        print("-----------------------------------------------------\n")

        # Load the model. If it's a URDF, MuJoCo will convert it internally.
        # If it's the converted MJCF, it loads directly.
        self.model = mujoco.MjModel.from_xml_path(model_path)

        # --- IMPORTANT CONVERSION STEP HERE ---
        # If we loaded a URDF, save it as an MJCF for future use and robustness
        if is_urdf:
            print(f"Saving converted MJCF model to: {LITE6_MJCF_PATH}")
            mujoco.mj_saveLastXML(LITE6_MJCF_PATH, self.model)
            print("Conversion complete. You can now use the MJCF file directly.")
            # Optionally, you could exit here and tell the user to run again with MJCF_PATH
            # For now, we'll continue with the model we just loaded.
        # --- END CONVERSION STEP ---

        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                          for i in range(self.model.njnt)]
        self.joint_names = [name for name in self.joint_names if name is not None]
        print(f"Found joints: {self.joint_names}")

        self.actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                             for i in range(self.model.nu)]
        self.actuator_joint_ids = [self.model.actuator_trnid[i][0] for i in range(self.model.nu)]

        print(f"Found actuators: {self.actuator_names}")
        print(f"Actuator-joint mapping: {self.actuator_joint_ids}")

        self.num_arm_joints = 6
        self.num_joints = len(self.joint_names)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

    def set_joint_positions(self, positions):
        if len(positions) != self.num_arm_joints:
            raise ValueError(f"Expected {self.num_arm_joints} arm joint positions from IK, got {len(positions)}")
        
        # Check if actuators were found before attempting to control
        if self.model.nu == 0:
            print("WARNING: No actuators found in MuJoCo model. Cannot set joint positions.")
            return # Exit the function if no actuators

        for i in range(self.num_arm_joints):
            self.data.ctrl[i] = positions[i]

        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.close()

if __name__ == "__main__":
    # FIRST RUN: Load the URDF to convert it to MJCF
    print("\n--- FIRST RUN: Loading URDF to generate MJCF ---")
    temp_sim = MuJoCo_xArmLite6(model_path=LITE6_URDF_PATH, is_urdf=True)
    temp_sim.close() # Close the viewer after conversion

    print("\n--- SECOND RUN: Loading the newly generated MJCF ---")
    # NOW, load the generated MJCF file which should contain proper actuator definitions
    sim = MuJoCo_xArmLite6(model_path=LITE6_MJCF_PATH, is_urdf=False)

    # JacobiRobot (Pinocchio) still uses the URDF
    robot = JacobiRobot(LITE6_URDF_PATH, ee_link=LITE6_EE_LINK)

    # ... (rest of your debug print statements for Pinocchio model) ...
    print("\n--- Pinocchio Model Debug Info ---")
    print(f"Number of frames in Pinocchio model: {robot.model.nframes}")
    print("Available frame names and their IDs:")
    frame_names = [f.name for f in robot.model.frames]
    for i, frame in enumerate(robot.model.frames):
        print(f"  ID: {i}, Name: {frame.name}")

    if LITE6_EE_LINK in frame_names:
        ee_frame_id_check = robot.model.getFrameId(LITE6_EE_LINK)
        print(f"'{LITE6_EE_LINK}' found in Pinocchio model with ID: {ee_frame_id_check}")
    else:
        print(f"ERROR: '{LITE6_EE_LINK}' NOT found in Pinocchio model frames.")
        print("Please check the correct end-effector link name in your URDF.")
        exit()
    print("----------------------------------\n")


    robot.start_visualization()

    target_poses = [
        np.array([
            [1, 0, 0, 0.35],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.20],
            [0, 0, 0, 1]
        ]),
        np.array([
            [0, -1, 0, 0.25],
            [1, 0, 0, 0.0],
            [0, 0, 1, 0.30],
            [0, 0, 0, 1]
        ]),
    ]

    current_target_idx = 0
    target_reached = False

    try:
        while True:
            if target_reached:
                current_target_idx = (current_target_idx + 1) % len(target_poses)
                target_reached = False
                print(f"Moving to target pose {current_target_idx}")

            target_pose = target_poses[current_target_idx]
            target_reached = robot.servo_to_pose(target_pose, dt=sim.model.opt.timestep)
            ik_joint_positions = robot.q
            sim.set_joint_positions(ik_joint_positions)
            robot.update_visualization()
            time.sleep(sim.model.opt.timestep)

    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        sim.close()
        robot.stop_visualization()