import mujoco as mj
import mujoco.viewer
import numpy as np
import transforms3d as t3d
import time
import pinocchio as pin
from pathlib import Path

# Placeholder for your abstract Robot class if it exists
# from alignit.robots.robot import Robot

class MuJoCoRobot: # Inherit from Robot if you have that base class
    def __init__(self, mjcf_path, urdf_path_pinocchio, end_effector_frame_name_pinocchio):
        # MuJoCo setup
        self.model = mj.xml.load_model_from_path(str(mjcf_path))
        self.data = mj.mjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Get joint IDs for control (assuming 6 joints for xArm Lite 6)
        self.joint_names = [f"joint{i}" for i in range(1, 7)] # Adjust if names are different
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        self.num_joints = len(self.joint_ids)

        # Get end-effector site ID for tracking its pose
        self.eef_site_id = self.model.site("gripper_site").id # Ensure this matches your MJCF

        # Pinocchio setup
        self.robot_pin = pin.RobotWrapper.BuildFromURDF(urdf_path_pinocchio)
        try:
            self.end_effector_frame_id_pin = self.robot_pin.model.getFrameId(end_effector_frame_name_pinocchio)
        except Exception as e:
            raise ValueError(f"End-effector frame '{end_effector_frame_name_pinocchio}' not found in Pinocchio model. "
                             "Please check your URDF and Pinocchio end-effector name. {e}")

        # Initial Pinocchio configuration (important for IK iterations)
        self.current_q_pin = pin.neutral(self.robot_pin.model)

        print("MuJoCo and Pinocchio robot initialized.")


    def _calculate_ik_pinocchio(self, target_pose_matrix):
        """
        Calculates inverse kinematics using Pinocchio.
        Returns the target joint angles.
        """
        target_position = target_pose_matrix[:3, 3]
        target_rotation = target_pose_matrix[:3, :3]
        target_se3 = pin.SE3(target_rotation, target_position)

        eps = 1e-6
        IT_MAX = 1000
        DT = 1e-2
        damp = 1e-6

        q = self.current_q_pin.copy() # Start IK from the current robot configuration

        for i in range(IT_MAX):
            pin.forwardKinematics(self.robot_pin.model, self.robot_pin.data, q)
            pin.updateFramePlacements(self.robot_pin.model, self.robot_pin.data)

            current_se3 = self.robot_pin.data.oMf[self.end_effector_frame_id_pin]
            err = pin.log6(current_se3.inverse() * target_se3).vector

            if np.linalg.norm(err) < eps:
                break

            J = pin.computeFrameJacobian(self.robot_pin.model, self.robot_pin.data, q,
                                         self.end_effector_frame_id_pin, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

            # Use least squares for numerical stability
            dq, residuals, rank, s = np.linalg.lstsq(J, err, rcond=damp)
            q = pin.integrate(self.robot_pin.model, q, dq * DT)

        # Update current Pinocchio state for the next IK call
        self.current_q_pin = q
        return q


    def send_action(self, action_pose_matrix):
        """
        Calculates IK for the target pose and applies joint positions to MuJoCo.
        """
        # Calculate target joint positions using Pinocchio IK
        # Note: Pinocchio's joint order and indexing might differ from MuJoCo's.
        # You might need to reorder or map the results if your URDF/MJCF has
        # different joint definitions/ordering.
        target_joint_poses = self._calculate_ik_pinocchio(action_pose_matrix)

        # Apply the target joint positions to MuJoCo motors
        # Ensure that target_joint_poses has the same number of elements
        # and corresponds to the `self.joint_ids` order.
        if len(target_joint_poses) != self.num_joints:
            print(f"Warning: Pinocchio IK returned {len(target_joint_poses)} joints, expected {self.num_joints}.")
            # You might need a mapping here if the joint count/order differs.
            # For simplicity, assuming they match for xArm Lite 6 (6 DoF robot).
            target_joint_poses = target_joint_poses[:self.num_joints]

        self.data.ctrl[self.joint_ids] = target_joint_poses # Set motor targets

        # Step the MuJoCo simulation
        mj.mj_step(self.model, self.data)
        self.viewer.sync() # Update the viewer
        # time.sleep(self.model.opt.timestep) # Optional: sleep to match real-time or slow down

    def pose(self):
        """
        Returns the current pose of the end-effector (gripper_site) in MuJoCo.
        """
        # MuJoCo stores site positions in data.site_xpos
        pos = self.data.site_xpos[self.eef_site_id]
        # MuJoCo stores site orientations as 3x3 rotation matrices in data.site_xmat
        rot_mat = self.data.site_xmat[self.eef_site_id].reshape(3, 3)

        return t3d.affines.compose(pos, rot_mat, [1, 1, 1])

    def get_observation(self):
        """
        Captures camera images from MuJoCo.
        This assumes a camera is defined in your MJCF.
        If not, you'll need to add a <camera> tag to your MJCF.
        """
        # For simplicity, assuming a camera named "robot_camera" is defined
        # in your MJCF, perhaps attached to your end-effector or a fixed position.
        # If not, you'll need to define one in your MJCF.
        # Example MJCF camera:
        # <body name="link6">
        #     <camera name="robot_camera" pos="0.1 0 0" xyaxes="0 1 0 0 0 1" fovy="60"/>
        # </body>
        camera_name = "robot_camera" # Name of the camera in your MJCF
        if camera_name not in self.model.camera_names:
            print(f"Warning: Camera '{camera_name}' not found in MJCF. No image observation available.")
            return {}

        cam_id = self.model.camera(camera_name).id

        width, height = 320, 240 # Match your PyBullet resolution
        rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
        depth_data = np.zeros((height, width), dtype=np.float32) # Depth data is often useful

        # Render the camera image
        mj.mjv_updateScene(self.model, self.data, mj.mjvOption(), mj.mjrContext(),
                           self.viewer.scn)
        mj.mjr_render(mj.MjrRect(0, 0, width, height), self.viewer.scn, mj.mjrContext(),
                      rgb_data, depth_data, None)

        # MuJoCo images are typically flipped vertically and have colors in a different order (BGRA or RGBA)
        # You might need to adjust the channel order if it's not RGB (e.g., OpenCV expects BGR).
        # This example assumes RGB for consistency with PyBullet's output (after processing).
        # And usually images are flipped, so flip them back.
        rgb_array = np.flipud(rgb_data) # Flip vertically

        return {
            "camera.rgb": rgb_array
        }

    def close(self):
        """
        Cleans up MuJoCo viewer and resources.
        """
        if self.viewer is not None:
            self.viewer.close()
        print("MuJoCo simulation closed.")


if __name__ == "__main__":
    # Define paths to your MuJoCo XML and Pinocchio URDF files
    mjcf_file = "xarm_lite_6.xml" # This is the MJCF you created/converted
    urdf_file_pinocchio = "path/to/your/xarm_lite_6.urdf" # Your xArm Lite 6 URDF

    # Verify your end-effector frame name in your xArm Lite 6 URDF
    end_effector_link_name = "link6" # Common for 6-axis robots, check your URDF!

    # Create the simulation instance
    try:
        sim = MuJoCoRobot(mjcf_file, urdf_file_pinocchio, end_effector_link_name)
    except Exception as e:
        print(f"Failed to initialize MuJoCoRobot: {e}")
        exit()

    # Example simulation loop
    try:
        for t in range(1000): # Run for a limited number of steps for testing
            # Define a target pose for the end-effector
            # Example: Move in a circle or to a fixed point
            # current_time = sim.data.time # Get current simulation time
            # x = 0.5 + 0.1 * np.sin(current_time * 2)
            # y = 0.1 * np.cos(current_time * 2)
            # z = 0.4
            # target_pos = np.array([x, y, z])

            target_pos = np.array([0.5, 0.0, 0.3]) # Fixed target position
            target_rot = t3d.quaternions.quat2mat([0, 0, 1, 0]) # Example: rotation about Z by 180 degrees (upwards)
            target_rot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) # Example from your PyBullet code

            pose = t3d.affines.compose(target_pos, target_rot, [1, 1, 1])

            # Send the action (target pose) to the robot
            sim.send_action(pose)

            # Get observations (e.g., camera image)
            observation = sim.get_observation()
            if "camera.rgb" in observation:
                # You can now process or display observation["camera.rgb"]
                # For example, using matplotlib:
                # import matplotlib.pyplot as plt
                # plt.imshow(observation["camera.rgb"])
                # plt.pause(0.01) # Small pause to update plot
                pass

            # Optional: Add a small delay for visualization if not running real-time
            # time.sleep(0.01) # Adjust as needed

            if sim.viewer.is_closed(): # Check if viewer is closed by user
                break

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        sim.close() # Ensure resources are cleaned up