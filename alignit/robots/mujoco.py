import mujoco
import mujoco.viewer
import numpy as np
import time

import os
from teleop.utils.jacobi_robot import JacobiRobot

URDF_PATH = os.path.join(os.path.dirname(__file__), "robot.urdf")
XARM_7_MJCF_PATH = "mujoco_menagerie/xarm7/xarm7.xml"

class MuJoCo_xArmLite6:
    def __init__(self, mjcf_model_path=None):
        if mjcf_model_path is None:
            raise ValueError("MJCF model path must be provided")
        
        self.model = mujoco.MjModel.from_xml_path(mjcf_model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Get all joint names
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                          for i in range(self.model.njnt)]
        self.joint_names = [name for name in self.joint_names if name is not None]
        self.num_joints = len(self.joint_names)
        print(f"Found joints: {self.joint_names}")

        # Get actuator names and their corresponding joint indices
        self.actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) 
                             for i in range(self.model.nu)]
        self.actuator_joint_ids = [self.model.actuator_trnid[i][0] for i in range(self.model.nu)]
        
        print(f"Found actuators: {self.actuator_names}")
        print(f"Actuator-joint mapping: {self.actuator_joint_ids}")

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

    def set_joint_positions(self, positions):
        if len(positions) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint positions, got {len(positions)}")
        
        # Set position for each actuator based on its corresponding joint
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i][0]
            self.data.ctrl[i] = positions[joint_id]

        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.close()
if __name__ == "__main__":
    # Initialize simulation and IK solver
    sim = MuJoCo_xArmLite6(mjcf_model_path=XARM_7_MJCF_PATH)
    robot = JacobiRobot(URDF_PATH, ee_link="end_effector")
    
    # Start visualization (optional)
    robot.start_visualization()
    
    # Define target poses for the end-effector
    target_poses = [
        # Example poses (position x,y,z in meters, orientation as rotation matrix)
        np.array([
            [1, 0, 0, 0.3],  # 0.3m in x direction
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.2],
            [0, 0, 0, 1]
        ]),
        np.array([
            [0, -1, 0, 0.2],  # Different orientation and position
            [1, 0, 0, 0.0],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ]),
        # Add more target poses as needed
    ]
    
    current_target_idx = 0
    target_reached = False
    
    try:
        while True:
            if target_reached:
                # Move to next target pose
                current_target_idx = (current_target_idx + 1) % len(target_poses)
                target_reached = False
                print(f"Moving to target pose {current_target_idx}")
            
            # Get current target pose
            target_pose = target_poses[current_target_idx]
            
            # Run IK solver to get joint positions
            target_reached = robot.servo_to_pose(target_pose, dt=sim.model.opt.timestep)
            
            # Get the resulting joint positions from IK
            ik_joint_positions = robot.q
            
            # Apply to simulation
            sim.set_joint_positions(ik_joint_positions)
            
            # Update visualization
            robot.update_visualization()
            
            # Sleep for simulation timestep
            time.sleep(sim.model.opt.timestep)
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        sim.close()
        robot.stop_visualization()