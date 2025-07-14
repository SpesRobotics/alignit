#import mujoco as mj
import mujoco.viewer
import numpy as np
import transforms3d as t3d
import time
import pinocchio as pin
from pathlib import Path
import os
os.environ['MUJOCO_GL'] = 'osmesa'  # Add this BEFORE importing mujoco
import mujoco as mj

# Placeholder for your abstract Robot class if it exists
# from alignit.robots.robot import Robot

class MuJoCoRobot: # Inherit from Robot if you have that base class
    def __init__(self, mjcf_path, urdf_path_pinocchio, end_effector_frame_name_pinocchio):
        # MuJoCo setup
        try:
            self.model = mj.MjModel.from_xml_path(str(mjcf_path))
            self.data = mj.MjData(self.model)
            print(f"DEBUG: Successfully loaded MuJoCo model from: {mjcf_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {mjcf_path}: {e}")

        # Initialize passive viewer - this automatically handles its own scene/context
        # The viewer object itself has .scn and .con attributes internally
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        print("DEBUG: MuJoCo passive viewer launched.")

        # Initialize offscreen rendering objects only when needed (lazy initialization)
        # This prevents OpenGL context conflicts with the passive viewer
        self.scn = None
        self.cam = None
        self.vopt = None
        self.pert = None
        self.mjr_context = None
        self._offscreen_initialized = False
        print("DEBUG: Offscreen rendering objects will be initialized on first use.")

        # Get joint IDs for control (assuming 6 joints for xArm Lite 6)
        self.joint_names = [f"joint{i}" for i in range(1, 7)] # Adjust if names are different
        # Note: self.joint_ids here seem to be used by old code, new code uses mujoco_actuator_ids
        # self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        # self.num_joints = len(self.joint_ids)

        # Get end-effector site ID for tracking its pose
        self.eef_site_id = self.model.site("gripper_site").id # Ensure this matches your MJCF
        
        # Pinocchio setup
        try:
            self.robot_pin = pin.RobotWrapper.BuildFromURDF(urdf_path_pinocchio)
            self.end_effector_frame_id_pin = self.robot_pin.model.getFrameId(end_effector_frame_name_pinocchio)
            print(f"DEBUG: Pinocchio model loaded from: {urdf_path_pinocchio}")
            print(f"DEBUG: Pinocchio end-effector frame ID: {self.end_effector_frame_id_pin}")
            if self.end_effector_frame_id_pin == self.robot_pin.model.nframes: # pin.NO_JOINT
                raise ValueError(f"End-effector frame '{end_effector_frame_name_pinocchio}' not found in Pinocchio URDF.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinocchio: {e}")

        # Initial Pinocchio configuration (important for IK iterations)
        self.current_q_pin = pin.neutral(self.robot_pin.model)

        # Map MuJoCo actuator IDs for control (0-indexed) - MOVED HERE FROM _initialize_offscreen_rendering
        self.mujoco_actuator_ids = []
        for i in range(1, self.robot_pin.model.nq + 1): # Iterate from 1 to nq (for joint1 to jointN from Pinocchio)
            actuator_name = f"joint{i}_ctrl" # Construct the actuator name expected in MJCF
            try:
                actuator_id = self.model.actuator(actuator_name).id
                self.mujoco_actuator_ids.append(actuator_id)
            except KeyError:
                raise ValueError(f"Actuator '{actuator_name}' not found in MuJoCo model. Check MJCF.")

        print(f"DEBUG: MuJoCo actuator IDs: {self.mujoco_actuator_ids}")

        # For observation return consistency
        self.nq = self.robot_pin.model.nq
        self.nv = self.robot_pin.model.nv

    def _initialize_offscreen_rendering(self):
        """
        Initialize offscreen rendering objects when first needed.
        This prevents OpenGL context conflicts with the passive viewer.
        """
        if self._offscreen_initialized:
            return
            
        try:
            # Initialize MuJoCo visualization objects for OFFSCREEN rendering
            self.scn = mj.MjvScene(self.model, maxgeom=1000)
            self.cam = mj.MjvCamera()
            mj.mjv_defaultCamera(self.cam)
            self.vopt = mj.MjvOption()
            mj.mjv_defaultOption(self.vopt)
            self.pert = mj.MjvPerturb()
            mj.mjv_defaultPerturb(self.pert)
            
            # Initialize the rendering context
            self.mjr_context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
            
            self._offscreen_initialized = True
            print("DEBUG: Offscreen rendering objects initialized successfully.")
        except Exception as e:
            print(f"WARNING: Failed to initialize offscreen rendering: {e}")
            self._offscreen_initialized = False

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
        damp = 1e-3

        q = self.current_q_pin.copy() # Start IK from the current robot configuration

        for i in range(IT_MAX):
            pin.forwardKinematics(self.robot_pin.model, self.robot_pin.data, q)
            pin.updateFramePlacements(self.robot_pin.model, self.robot_pin.data)

            current_se3 = self.robot_pin.data.oMf[self.end_effector_frame_id_pin]
            err = pin.log6(current_se3.inverse() * target_se3).vector

            if np.linalg.norm(err) < eps:
                break

            # computeFrameJacobian for LOCAL_WORLD_ALIGNED frame
            J = pin.computeFrameJacobian(self.robot_pin.model, self.robot_pin.data, q,
                                         self.end_effector_frame_id_pin, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

            # Use least squares for numerical stability
            # dq is delta_q, which is delta_velocity in tangent space
            dq, residuals, rank, s = np.linalg.lstsq(J, err, rcond=damp)
            q = pin.integrate(self.robot_pin.model, q, dq * DT)
    
        for i in range(self.robot_pin.model.nq):
            if self.model.jnt_limited[i]:
                q[i] = np.clip(
                    q[i],
                    self.model.jnt_range[i, 0],
                    self.model.jnt_range[i, 1]
                )

        # Update current Pinocchio state for the next IK call
        self.current_q_pin = q
        return q[self.robot_pin.nq-self.model.nu:] # Return only the actuated joint positions

    def send_action(self, action_pose_matrix):
        """
        Calculates IK for the target pose and applies joint positions to MuJoCo.
        """
        # Calculate target joint positions using Pinocchio IK
        # Note: Pinocchio's joint order and indexing might differ from MuJoCo's.
        # Ensure that the returned target_joint_poses matches the order of self.mujoco_actuator_ids.
        target_joint_poses = self._calculate_ik_pinocchio(action_pose_matrix)

          # Check joint limits
        for i, q in enumerate(target_joint_poses):
            joint_name = self.model.joint(i).name
            limit_low = self.model.jnt_range[i,0] if self.model.jnt_limited[i] else -np.pi
            limit_high = self.model.jnt_range[i,1] if self.model.jnt_limited[i] else np.pi
            
            if not (limit_low <= q <= limit_high):
                print(f"WARNING: Joint {joint_name} target {q} exceeds limits [{limit_low}, {limit_high}]")

        if len(target_joint_poses) != self.model.nu: # Check against number of actuators
            # This check is crucial if IK returns more/less than expected
            # For a 6-DOF fixed-base robot, Pinocchio.nq == MuJoCo.nu == 6
            raise ValueError(f"Expected {self.model.nu} joint positions from IK, got {len(target_joint_poses)}")

        # Use the correctly mapped actuator IDs
        self.data.ctrl[self.mujoco_actuator_ids] = target_joint_poses
        
        # Step the MuJoCo simulation
        mj.mj_step(self.model, self.data)


        # Update the viewer (if it's open)
        if self.viewer.is_running(): # Check if the passive viewer is still active
            self.viewer.sync()
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

    def get_observation(self, camera_name: str = "front_camera"):
        """
        Captures camera images from MuJoCo and returns robot state.
        """
        # Initialize offscreen rendering if not already done
        if self.viewer.is_running():
            return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "eef_pose": self.pose(),
        }
    

        self._initialize_offscreen_rendering()
        
        # If offscreen rendering failed to initialize, return observation without camera
        if not self._offscreen_initialized:
            print("WARNING: Offscreen rendering not available, returning observation without camera image.")
            observation = {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose(), # Current end-effector pose
            }
            return observation
        
        # Get all camera names from the model
        all_camera_names = [self.model.camera(i).name for i in range(self.model.ncam)]

        if camera_name in all_camera_names:
            camera_id = self.model.camera(camera_name).id
            # Correct way to set the MjvCamera object (self.cam) to use a fixed camera from the model
            self.cam.type = mj.mjtCamera.mjCAMERA_FIXED # Set camera type to fixed
            self.cam.fixedcamid = camera_id             # Set the ID of the fixed camera
        else:
            print(f"Warning: Camera '{camera_name}' not found in MJCF. Using default scene camera setup for observation.")
            # If the camera is not found, reset the camera object to default free view
            self.cam.type = mj.mjtCamera.mjCAMERA_FREE # Set camera type to free
            mj.mjv_defaultCamera(self.cam)     

        width, height = 320, 240 # Define your desired image resolution for offscreen rendering
        
        try:
            # Update the scene for rendering (using the objects initialized in __init__)
            # Arguments: model, data, option, perturb, camera, category_mask (int), scene
            mj.mjv_updateScene(self.model, self.data, self.vopt, self.pert, self.cam, 0, self.scn)

            # Prepare buffers for RGB and depth data
            rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
            depth_data = np.zeros((height, width), dtype=np.float32)

            # Render the scene to the buffers (using the context initialized in __init__)
            viewport = mj.MjrRect(0, 0, width, height)
            # Arguments: viewport, scene, context, RGB output, Depth output, segmentation output
            mj.mjr_render(viewport, self.scn, self.mjr_context) # Render to internal buffer

            # Read the pixels from MuJoCo's internal buffer into your numpy arrays
            # Arguments: rgb_output_array, depth_output_array, viewport, context
            # Note: rgb_data and depth_data are populated directly by this function
            mj.mjr_readPixels(rgb_data, depth_data, viewport, self.mjr_context)    
            # MuJoCo images are typically flipped vertically; flip them back for standard image processing
            rgb_array = np.flipud(rgb_data)

            # Combine numerical state and image data for observation
            observation = {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose(), # Current end-effector pose
                "camera.rgb": rgb_array
            }
            return observation
        except Exception as e:
            print(f"WARNING: Failed to render camera image: {e}")
            # Return observation without camera image if rendering fails
            observation = {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose(), # Current end-effector pose
            }
            return observation

    def close(self):
        """
        Cleans up MuJoCo viewer and resources.
        """
        if self.viewer is not None and self.viewer.is_running(): # Check if viewer is still running before closing
            self.viewer.close()
        # Free the rendering context if it was initialized
        if hasattr(self, 'mjr_context') and self.mjr_context is not None and self._offscreen_initialized:
            try:
                mj.mjr_freeContext(self.mjr_context)
            except Exception as e:
                print(f"WARNING: Failed to free rendering context: {e}")
        print("MuJoCo simulation closed.")


if __name__ == "__main__":
    # Define paths to your MuJoCo XML and Pinocchio URDF files
    mjcf_file = Path("/home/nikola/code/alignit/alignit/lite6mjcf.xml") # Use Path objects directly
    urdf_file_pinocchio = Path("/home/nikola/code/alignit/alignit/lite6.urdf") # Your xArm Lite 6 URDF

    # Verify your end-effector frame name in your xArm Lite 6 URDF
    end_effector_link_name = "link6" # Common for 6-axis robots, check your URDF!

    sim = None
    # Create the simulation instance
    try:
        sim = MuJoCoRobot(mjcf_file, urdf_file_pinocchio, end_effector_link_name)
        print("Initial qpos:", sim.data.qpos)
        print("Initial qvel:", sim.data.qvel)
        print("DEBUG: Robot initialization successful!")
    except Exception as e:
        print(f"Failed to initialize MuJoCoRobot: {e}")
        if sim is not None:
            sim.close()
        exit(1)

    # Example simulation loop
    try:
        initial_time = time.time()
        for t_step in range(1000): # Run for a limited number of steps for testing
            # Define a target pose for the end-effector
            current_sim_time = sim.data.time # Get current simulation time

            # Example: Move in a small circle in XY plane, fixed Z
            x = 0.5 + 0.05 * np.sin(current_sim_time * 1.0)
            y = 0.05 * np.cos(current_sim_time * 1.0)
            z = 0.4
            target_pos = np.array([x, y, z])



            # Fixed orientation (e.g., end-effector pointing downwards)
            # This rotation matrix aligns the end-effector's Z-axis (pointing outwards) with world -Z
            # And its X-axis with world +Y.
            target_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # Example: -90 deg around X, then 90 deg around Z

            pose = t3d.affines.compose(target_pos, target_rot, [1, 1, 1])

            # Send the action (target pose) to the robot
            sim.send_action(pose)

            # Get observations (e.g., camera image and robot state)
            observation = sim.get_observation(camera_name="front_camera") # Use front_camera as you have it

            # Optional: Process observation or save image
            # if "camera.rgb" in observation:
            #     # You can now process or display observation["camera.rgb"]
            #     # import matplotlib.pyplot as plt
            #     # plt.imshow(observation["camera.rgb"])
            #     # plt.pause(0.01) # Small pause to update plot
            #     pass

            # Check if viewer is closed by user in passive mode
            if not sim.viewer.is_running(): # is_running() is the correct method for passive viewer
                print("Viewer closed by user.")
                break

            # Optional: Regulate loop speed to visualize at a consistent pace
            # Desired real-time factor, e.g., 1.0 for real-time
            # while sim.data.time / (time.time() - initial_time) < 1.0:
            #     time.sleep(0.001)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sim is not None:
            sim.close() # Ensure resources are cleaned up