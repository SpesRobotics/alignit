import mujoco as mj
import mujoco.viewer
import numpy as np
import transforms3d as t3d
import time
import pinocchio as pin
from pathlib import Path
import os
import logging

# --- Configure the root logger once at the top ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    filename='robot_simulation.log', # Log to this file
                    filemode='w') # 'w' for overwrite, 'a' for append

# Get a logger for the current module for top-level messages
main_logger = logging.getLogger(__name__)

# --- Pinocchio specific import ---
from pinocchio import RobotWrapper 

class MuJoCoRobot:
    def __init__(self, mjcf_path, urdf_path_pinocchio, end_effector_frame_name_pinocchio):
        """
        Initializes the MuJoCo simulation environment and the Pinocchio robot model.

        Args:
            mjcf_path (Path): Path to the MuJoCo MJCF model file (e.g., scene.xml).
            urdf_path_pinocchio (Path): Path to the URDF model file for Pinocchio (e.g., lite6.urdf).
            end_effector_frame_name_pinocchio (str): Name of the end-effector frame in the Pinocchio model.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # --- MuJoCo setup ---
        try:
            self.model = mj.MjModel.from_xml_path(str(mjcf_path))
            self.data = mj.MjData(self.model)
            
            # Configure simulation for stability and accuracy
            self.model.opt.timestep = 0.1
            self.model.opt.iterations = 50
            self.model.opt.tolerance = 1e-8
            self.model.opt.solver = mj.mjtSolver.mjSOL_NEWTON
            
            # Add damping to all joints to improve stability and prevent oscillations
            for i in range(self.model.nv):
                self.model.dof_damping[i] = 0.5 # Increased damping for better stability
                
            self.logger.info(f"Successfully loaded MuJoJo model from: {mjcf_path}")
            self.logger.info(f"MuJoCo model has {self.model.nq} generalized coordinates (qpos size).")
            self.logger.info(f"MuJoCo model has {self.model.nv} degrees of freedom (qvel size).")
            self.logger.info(f"MuJoCo model has {self.model.nu} actuators (ctrl size).")

            # --- Log all actuator names to debug potential empty names ---
            self.logger.info("MuJoCo Actuator Names (from model.actuator(i).name):")
            for j in range(self.model.nu):
                act_name = self.model.actuator(j).name
                self.logger.info(f"  Actuator {j}: '{act_name}'")
                if not act_name:
                    self.logger.warning(f"  WARNING: Actuator at index {j} has an empty name in the MJCF model!")


            # --- Identify End-Effector in MuJoCo ---
            # For scene.xml, 'gripper_site' might not exist. 'link6' is a common body name.
            try:
                # Try to use a site first if it exists and is defined in the model
                # The Lite6 menagerie scene.xml doesn't define a 'gripper_site'
                self.eef_id_type = 'site'
                self.eef_id = self.model.site("gripper_site").id
                self.logger.info("Using 'gripper_site' as MuJoCo end-effector.")
            except KeyError:
                try:
                    # Fallback to using the body ID if 'gripper_site' is not found
                    self.eef_id_type = 'body'
                    self.eef_id = self.model.body("link6").id # Common for Lite6
                    self.logger.info("Using 'link6' body as MuJoCo end-effector.")
                except KeyError:
                    self.logger.critical("Neither 'gripper_site' nor 'link6' body found in MuJoCo model for end-effector. Please check your MJCF.")
                    raise RuntimeError("MuJoCo end-effector not found.")

        except Exception as e:
            self.logger.critical(f"Failed to load MuJoCo model from {mjcf_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load MuJoCo model from {mjcf_path}: {e}")

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.logger.debug("MuJoCo passive viewer launched.")

        # --- Offscreen rendering setup (lazy initialization) ---
        self.scn = None
        self.cam = None
        self.vopt = None
        self.pert = None
        self.mjr_context = None
        self._offscreen_initialized = False

        # --- Pinocchio setup ---
        try:
            self.robot_pin = RobotWrapper.BuildFromURDF(str(urdf_path_pinocchio))
            self.end_effector_frame_id_pin = self.robot_pin.model.getFrameId(end_effector_frame_name_pinocchio)
            if self.end_effector_frame_id_pin == self.robot_pin.model.nframes:
                raise ValueError(f"End-effector frame '{end_effector_frame_name_pinocchio}' not found in Pinocchio model.")
            self.logger.info(f"Pinocchio model loaded from: {urdf_path_pinocchio}")
            self.logger.info(f"Pinocchio model has {self.robot_pin.model.nq} generalized coordinates (q size).")
            self.logger.info(f"Pinocchio model has {self.robot_pin.model.nv} velocity dimensions (v size).")
        except Exception as e:
            self.logger.critical(f"Failed to initialize Pinocchio: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Pinocchio: {e}")

        # Initialize Pinocchio's current configuration to its neutral pose
        self.current_q_pin = pin.neutral(self.robot_pin.model)
        
        # --- Map MuJoCo actuators to Pinocchio joint indices ---
        self.mujoco_actuator_ids = []
        self.mujoco_actuator_to_pinocchio_q_idx = [] 
        self.mujoco_qpos_indices_for_actuators = [] # Store MuJoCo qpos index for each actuated joint

        pinocchio_joint_name_to_q_idx_map = {}
        self.logger.info("\nPinocchio Joint Names and their q_indices:")
        for joint_id in range(1, self.robot_pin.model.njoints): # Skip universe joint (id 0)
            joint_model = self.robot_pin.model.joints[joint_id]
            joint_name = self.robot_pin.model.names[joint_id] 
            pinocchio_joint_name_to_q_idx_map[joint_name] = joint_model.idx_q
            self.logger.info(f"  Pinocchio Joint: '{joint_name}' (ID: {joint_id}), Q Index: {joint_model.idx_q}")

        self.logger.info("\nAttempting to map MuJoCo actuators to Pinocchio joints:")

        for i in range(self.model.nu): 
            actuator_name = self.model.actuator(i).name # This might be empty, but we'll use the joint name for mapping
            
            # Removed: if not actuator_name: continue (as the joint name is what matters for mapping)
            
            mujoco_joint_id = self.model.actuator_trnid[i, 0] 
            
            if mujoco_joint_id == -1: 
                self.logger.warning(f"MuJoCo actuator '{actuator_name}' (index {i}) does not control a standard joint (mujoco_joint_id is -1). Skipping.")
                continue

            mujoco_joint_name = self.model.joint(mujoco_joint_id).name
            self.logger.info(f"  Processing MuJoCo Actuator '{actuator_name}' (index {i}), linked to MuJoCo Joint '{mujoco_joint_name}' (ID: {mujoco_joint_id}).")


            if mujoco_joint_name in pinocchio_joint_name_to_q_idx_map:
                self.mujoco_actuator_ids.append(self.model.actuator(i).id) 
                self.mujoco_actuator_to_pinocchio_q_idx.append(pinocchio_joint_name_to_q_idx_map[mujoco_joint_name])
                self.mujoco_qpos_indices_for_actuators.append(self.model.joint(mujoco_joint_id).qposadr[0])
                self.logger.info(f"    SUCCESS: Mapped MuJoCo joint '{mujoco_joint_name}' to Pinocchio joint.")
            else:
                self.logger.warning(f"    WARNING: MuJoCo joint '{mujoco_joint_name}' (controlled by actuator '{actuator_name}' at index {i}) DOES NOT have a corresponding Pinocchio joint with the same name. Skipping this actuator.")
        
        if not self.mujoco_actuator_ids:
            self.logger.critical("No MuJoCo actuators were successfully mapped to Pinocchio joints. IK will not function. Please check your URDF/MJCF joint and actuator naming conventions.")
        else:
            self.num_mapped_joints = len(self.mujoco_actuator_ids)
            self.logger.info(f"Successfully mapped {self.num_mapped_joints} MuJoCo actuators to Pinocchio joints.")
            self.logger.debug(f"Mapped MuJoCo Actuator IDs: {self.mujoco_actuator_ids}")
            self.logger.debug(f"Corresponding Pinocchio q indices for mapped actuators: {self.mujoco_actuator_to_pinocchio_q_idx}")
            self.logger.debug(f"Corresponding MuJoCo qpos indices for mapped actuators: {self.mujoco_qpos_indices_for_actuators}")


        # --- Initializing MuJoCo qpos to a valid starting pose ---
        # Reset data to model's default initial state first.
        mj.mj_resetData(self.model, self.data)
        
        # If the model has at least 6 DOFs (for the Lite6), set its initial pose.
        # This assumes the Lite6 joints are the first 6 DOFs in the scene.xml's qpos.
        if self.model.nq >= 6:
            initial_robot_qpos_rad = np.array([0.0, 0.0, np.deg2rad(5.0), 0.0, 0.0, 0.0])
            # Ensure we don't try to set more qpos than available
            num_to_set = min(len(initial_robot_qpos_rad), self.model.nq)
            self.data.qpos[0:num_to_set] = initial_robot_qpos_rad[0:num_to_set]
            self.logger.info(f"Initial MuJoCo qpos for first {num_to_set} joints set to: {np.degrees(self.data.qpos[0:num_to_set])} degrees")
        else:
            self.logger.warning(f"MuJoCo model has fewer than 6 generalized coordinates ({self.model.nq}). Robot-specific initial pose not set.")
        
        mj.mj_forward(self.model, self.data) # Propagate this initial state through the model
        
        self.viewer.sync() # Update viewer to show initial pose
        time.sleep(1.0) # Pause to see initial pose (can be removed later)

        # --- Warmup simulation ---
        for _ in range(10):
            mj.mj_step(self.model, self.data)
            self.data.ctrl[:] = 0 # Ensure controls are zeroed during warmup

        self.logger.debug("MuJoCo model loaded. Checking actuator properties:")
        for i in range(self.model.nu):
            actuator_name = self.model.actuator(i).name
            joint_id = self.model.actuator_trnid[i, 0] 
            if joint_id != -1: 
                joint_name = self.model.joint(joint_id).name
                forcerange = self.model.actuator(i).forcerange 
                stiffness = self.model.joint(joint_id).stiffness 
                self.logger.debug(f"  Actuator '{actuator_name}' (controls joint '{joint_name}'): Forcerange={forcerange}, Stiffness={stiffness}")
            else:
                self.logger.debug(f"  Actuator '{actuator_name}' (not controlling a joint or unknown type)")


    def _initialize_offscreen_rendering(self):
        """
        Lazy initialization of offscreen rendering components for camera observations.
        This is called only if get_observation is requested and the viewer is not running.
        """
        if self._offscreen_initialized:
            return
            
        try:
            self.scn = mj.MjvScene(self.model, maxgeom=1000) 
            self.cam = mj.MjvCamera() 
            mj.mjv_defaultCamera(self.cam) 
            self.vopt = mj.MjvOption() 
            self.pert = mj.MjvPerturb() 
            self.mjr_context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
            self._offscreen_initialized = True
            self.logger.debug("Offscreen rendering initialized.")
        except Exception as e:
            self.logger.warning(f"Failed offscreen rendering initialization: {e}", exc_info=True)

    def _calculate_ik_pinocchio(self, target_pose_matrix):
        """
        Calculates the Inverse Kinematics (IK) solution using Pinocchio.
        Finds the joint configuration (q) that places the end-effector at the target_pose_matrix.

        Args:
            target_pose_matrix (np.ndarray): A 4x4 homogeneous transformation matrix
                                             representing the target pose of the end-effector
                                             relative to the robot's base frame.

        Returns:
            np.ndarray: The full Pinocchio configuration vector (q) for the robot.
        """
        if not self.mujoco_actuator_ids: # If no actuators mapped, IK cannot work
            self.logger.error("IK cannot be calculated: No MuJoCo actuators mapped to Pinocchio joints.")
            return self.current_q_pin # Return current Pinocchio pose

        target_se3 = pin.SE3(target_pose_matrix[:3, :3], target_pose_matrix[:3, 3])
        q = self.current_q_pin.copy()  
        
        eps = 1e-4   
        IT_MAX = 10000 # Increased iterations for better convergence
        DT = 0.01     # Reduced IK step size for smoother trajectory
        damp = 1e-2  # Increased damping for better stability
        
        self.logger.debug("\nStarting IK calculation with Pinocchio...")
        for i in range(IT_MAX):
            pin.forwardKinematics(self.robot_pin.model, self.robot_pin.data, q)
            pin.updateFramePlacements(self.robot_pin.model, self.robot_pin.data)
            
            current_se3 = self.robot_pin.data.oMf[self.end_effector_frame_id_pin]
            err = pin.log6(current_se3.inverse() * target_se3).vector 
            
            error_norm = np.linalg.norm(err)
            self.logger.debug(f"  IK Iteration {i}: Error norm = {error_norm:.6f}") # Uncommented for detailed IK per-iteration debug

            if error_norm < eps:
                self.logger.debug(f"IK converged in {i} iterations. Final error norm: {error_norm:.6f}")
                break

            J = pin.computeFrameJacobian(
                self.robot_pin.model, self.robot_pin.data, q,
                self.end_effector_frame_id_pin, # Ensure this uses the correct frame ID
                pin.ReferenceFrame.LOCAL
            )
            
            U, S, Vh = np.linalg.svd(J)
            S_inv = np.zeros(J.shape[1])
            S_inv[:len(S)] = S / (S**2 + damp) 
            dq = Vh.T @ np.diag(S_inv) @ U.T @ err

            q_neutral = pin.neutral(self.robot_pin.model)
            nullspace_projection = (np.eye(J.shape[1]) - np.linalg.pinv(J) @ J) @ (q - q_neutral)
            dq += 0.01 * nullspace_projection 
            
            q = pin.integrate(self.robot_pin.model, q, dq * DT) 
            
            q = np.clip(q, self.robot_pin.model.lowerPositionLimit, self.robot_pin.model.upperPositionLimit)

        if np.linalg.norm(err) >= eps:
            self.logger.warning(f"IK did not fully converge after {IT_MAX} iterations. Final error norm: {np.linalg.norm(err):.6f}")
        
        self.current_q_pin = q.copy() 
        return q 

    def send_action(self, action_pose_matrix):
        """
        Sends a target end-effector pose to the robot.
        This function calculates the required joint angles using IK and applies them to MuJoCo.

        Args:
            action_pose_matrix (np.ndarray): A 4x4 homogeneous transformation matrix
                                             representing the desired world-frame pose of the end-effector.

        Returns:
            bool: True if the action was successfully sent and within joint limits, False otherwise.
        """
        # If no actuators are mapped, we cannot send actions to the robot.
        if not self.mujoco_actuator_ids:
            self.logger.error("Cannot send action: No MuJoCo actuators mapped to Pinocchio joints.")
            return False

        # Optional: Visualize the target in MuJoCo if a 'gripper_site' exists
        # Note: The Lite6 menagerie scene.xml typically does NOT have a 'gripper_site'
        # If you want to visualize the target, you'd need to add a site to your scene.xml
        # or dynamically create a geom in the viewer.
        # if self.eef_id_type == 'site':
        #     self.model.site_pos[self.eef_id] = action_pose_matrix[:3, 3]
        #     self.model.site_quat[self.eef_id] = t3d.quaternions.mat2quat(action_pose_matrix[:3, :3])
        
        # Ensure 'link_base' exists in the scene.xml
        try:
            base_pos = self.data.xpos[self.model.body("link_base").id]
            base_rot = self.data.xmat[self.model.body("link_base").id].reshape(3,3)
        except KeyError:
            self.logger.error("Body 'link_base' not found in MuJoCo model. Cannot calculate IK relative to base.")
            return False

        world_to_base = t3d.affines.compose(base_pos, base_rot, [1,1,1])
        base_target = np.linalg.inv(world_to_base) @ action_pose_matrix

        full_pinocchio_q = self._calculate_ik_pinocchio(base_target)
        
        # Ensure target_joint_poses has the correct size based on mapped actuators
        # This check should be against the number of *mapped* Pinocchio joints, not necessarily full_pinocchio_q length
        if len(self.mujoco_actuator_to_pinocchio_q_idx) == 0: # If no joints mapped, IK result is irrelevant
             self.logger.error("No Pinocchio joints mapped to MuJoCo actuators. Cannot process IK result.")
             return False
        
        # This check is more accurate: ensure the indices are valid for full_pinocchio_q
        # Find the maximum index requested from full_pinocchio_q
        max_pin_idx_requested = -1
        if self.mujoco_actuator_to_pinocchio_q_idx: # Check if list is not empty
            max_pin_idx_requested = max(self.mujoco_actuator_to_pinocchio_q_idx)

        if max_pin_idx_requested >= len(full_pinocchio_q):
            self.logger.error(f"Pinocchio IK result (full_pinocchio_q) has size {len(full_pinocchio_q)}, but indices up to {max_pin_idx_requested} are requested by mapped actuators. Data access error.")
            return False

        target_joint_poses = np.array([full_pinocchio_q[idx] for idx in self.mujoco_actuator_to_pinocchio_q_idx])
        
        self.logger.debug(f"IK calculated target_joint_poses (for MuJoCo actuators, shape: {target_joint_poses.shape}): {np.degrees(target_joint_poses)}")
        
        # Get current joint positions from MuJoCo using the *correct* qpos indices for the mapped actuators
        # Ensure current_joint_qpos has the correct size
        max_mujoco_qpos_idx_requested = -1
        if self.mujoco_qpos_indices_for_actuators: # Check if list is not empty
            max_mujoco_qpos_idx_requested = max(self.mujoco_qpos_indices_for_actuators)

        if max_mujoco_qpos_idx_requested >= len(self.data.qpos):
            self.logger.error(f"MuJoCo qpos array has size {len(self.data.qpos)}, but indices up to {max_mujoco_qpos_idx_requested} are requested by mapped actuators. Data access error.")
            return False
        current_joint_qpos = self.data.qpos[self.mujoco_qpos_indices_for_actuators]

        # Define a maximum joint velocity (radians per second) or a maximum position change per step.
        max_joint_vel = np.deg2rad(2000) # Increased max joint velocity for faster movement
        max_angle_change_per_step = max_joint_vel * self.model.opt.timestep
        
        # Calculate the desired change for this step
        angle_diff = target_joint_poses - current_joint_qpos
        
        # Limit the angle change per step to ensure smooth motion and prevent instability
        limited_angle_diff = np.clip(angle_diff, -max_angle_change_per_step, max_angle_change_per_step)
        
        # The new control target is the current position plus the limited change
        new_control_target_q = current_joint_qpos + limited_angle_diff

        # --- Joint Limit Check (Apply to the *new_control_target_q* before applying) ---
        for i, actuator_id in enumerate(self.mujoco_actuator_ids):
            q_target_for_this_actuator = new_control_target_q[i]
            
            mujoco_joint_id = self.model.actuator_trnid[actuator_id, 0] # Get joint ID from actuator_trnid
            if mujoco_joint_id == -1: continue # Should have been caught during mapping

            if self.model.jnt_limited[mujoco_joint_id]: 
                lower_limit = self.model.jnt_range[mujoco_joint_id, 0]
                upper_limit = self.model.jnt_range[mujoco_joint_id, 1]
                buffer = 0.005 # radians
                
                # Check bounds before accessing qposadr
                if self.model.joint(mujoco_joint_id).qposadr[0] >= len(self.data.qpos):
                    self.logger.error(f"MuJoCo qpos index out of bounds for joint '{self.model.joint(mujoco_joint_id).name}'.")
                    return False
                current_q_val = self.data.qpos[self.model.joint(mujoco_joint_id).qposadr[0]] # Actual current qpos

                if not (lower_limit + buffer <= q_target_for_this_actuator <= upper_limit - buffer):
                    self.logger.warning(f"Joint limit violation detected for MuJoCo joint '{self.model.joint(mujoco_joint_id).name}' (controlled by actuator '{self.model.actuator(actuator_id).name}'):")
                    self.logger.warning(f"  Target angle (limited and buffered): {np.degrees(q_target_for_this_actuator):.2f}°")
                    self.logger.warning(f"  Current angle: {np.degrees(current_q_val):.2f}°")
                    self.logger.warning(f"  Allowed limits (with buffer): [{np.degrees(lower_limit + buffer):.2f}°, {np.degrees(upper_limit - buffer):.2f}°]")
                    self.logger.warning(f"  Original limits: [{np.degrees(lower_limit):.2f}°, {np.degrees(upper_limit):.2f}°]")
                    return False  

        # Apply the smoothed target to the MuJoCo actuators
        # Ensure new_control_target_q has the same size as mujoco_actuator_ids
        if len(new_control_target_q) != len(self.mujoco_actuator_ids):
            self.logger.error("Mismatch in size between new_control_target_q and mujoco_actuator_ids. Control cannot be applied.")
            return False
            
        self.data.ctrl[self.mujoco_actuator_ids] = new_control_target_q
        self.logger.debug(f"MuJoCo data.ctrl after assignment: {np.degrees(self.data.ctrl[self.mujoco_actuator_ids])}")
        
        mj.mj_step(self.model, self.data)
        
        if self.viewer.is_running():
            self.viewer.sync()
        return True 

    def pose(self):
        """
        Gets the current end-effector pose from the MuJoCo simulation.
        Uses the identified end-effector body or site.
        """
        if self.eef_id_type == 'site':
            pos = self.data.site_xpos[self.eef_id]
            rot_mat = self.data.site_xmat[self.eef_id].reshape(3, 3)
        else: # Must be 'body'
            pos = self.data.xpos[self.eef_id]
            rot_mat = self.data.xmat[self.eef_id].reshape(3, 3)
        return t3d.affines.compose(pos, rot_mat, [1, 1, 1])

    def get_observation(self, camera_name: str = "front_camera"):
        """
        Gets the current observation from the simulation, including joint states
        and optionally a camera image if offscreen rendering is enabled.
        """
        if self.viewer.is_running():
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose()
            }
            
        self._offscreen_initialized = False 
        self._initialize_offscreen_rendering()
        if not self._offscreen_initialized:
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose()
            }
            
        try:
            if camera_name in [self.model.camera(i).name for i in range(self.model.ncam)]:
                self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
                self.cam.fixedcamid = self.model.camera(camera_name).id
            else:
                self.cam.type = mj.mjtCamera.mjCAMERA_FREE
                mj.mjv_defaultCamera(self.cam)
                
            rgb_data = np.zeros((240, 320, 3), dtype=np.uint8) 
            depth_data = np.zeros((240, 320), dtype=np.float32) 
            viewport = mj.MjrRect(0, 0, 320, 240) 
            mj.mjr_render(viewport, self.scn, self.mjr_context)
            mj.mjr_readPixels(rgb_data, depth_data, viewport, self.mjr_context)
            
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose(),
                "camera.rgb": np.flipud(rgb_data) 
            }
        except Exception as e:
            self.logger.warning(f"Camera rendering failed: {e}", exc_info=True)
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose()
            }

    def close(self):
        """
        Cleans up MuJoCo and Pinocchio resources.
        """
        if self.viewer and self.viewer.is_running():
            self.logger.debug("Closing MuJoCo viewer.")
            self.viewer.close()
            self.viewer = None # Mark as closed
        if hasattr(self, 'mjr_context') and self.mjr_context and self._offscreen_initialized:
            try:
                self.logger.debug("Freeing MuJoCo rendering context.")
                mj.mjr_freeContext(self.mjr_context)
                self.mjr_context = None # Mark as freed
                self._offscreen_initialized = False
            except Exception as e:
                self.logger.warning(f"Failed to free MjrContext: {e}", exc_info=True)

if __name__ == "__main__":
    #os.environ['MUJOCO_GL'] = 'osmesa' # Uncomment this if running headless (without a display server)
    
    # --- Configuration ---
    mjcf_file = Path("/home/nikola/code/alignit/alignit/mujoco_menagerie/ufactory_lite6/scene.xml")
    urdf_file_pinocchio = Path("/home/nikola/code/alignit/alignit/lite6.urdf")
    end_effector_link_name = "link6" # This should match the end-effector body/frame name in your URDF/MJCF

    main_logger.info("=== Initializing Robot Simulation for Scene Visualization ===")
    sim = None
    try:
        main_logger.info(f"Loading MJCF from: {mjcf_file} and URDF from: {urdf_file_pinocchio}...")
        sim = MuJoCoRobot(mjcf_file, urdf_file_pinocchio, end_effector_link_name) 
        
        main_logger.info(f"MuJoCo model nq (generalized coordinates): {sim.model.nq}")
        main_logger.info(f"MuJoCo model nv (degrees of freedom): {sim.model.nv}")
        main_logger.info(f"MuJoCo model nu (actuators): {sim.model.nu}")
        
        # Log all MuJoCo joint names and their qpos addresses to help diagnose
        main_logger.info("\n--- MuJoCo Joint Details ---")
        for i in range(sim.model.njnt):
            joint_name = sim.model.joint(i).name
            # FIX: Access scalar type explicitly to avoid DeprecationWarning
            joint_type = mj.mjtJoint(sim.model.joint(i).type.item()).name 
            qpos_adr = sim.model.joint(i).qposadr[0] if sim.model.joint(i).type != mj.mjtJoint.mjJNT_FREE else -1
            main_logger.info(f"- Joint: '{joint_name}' (ID: {i}), Type: {joint_type}, Qpos Address: {qpos_adr}")

        # Log all MuJoCo actuator names and the joint they control
        main_logger.info("\n--- MuJoCo Actuator Details ---")
        for i in range(sim.model.nu):
            actuator_name = sim.model.actuator(i).name
            joint_id = sim.model.actuator_trnid[i, 0] # Use actuator_trnid here too for consistency in logging
            joint_name = sim.model.joint(joint_id).name if joint_id != -1 else "N/A (not controlling a joint)"
            main_logger.info(f"- Actuator: '{actuator_name}' (ID: {i}), Controls Joint: '{joint_name}'")

        # Pinocchio joint details are now logged within the MuJoCoRobot class init for better context
        # main_logger.info("\n--- Pinocchio Joint Details ---")
        # for joint_id in range(1, sim.robot_pin.model.njoints): # Skip universe joint (id 0)
        #     joint_name = sim.robot_pin.model.names[joint_id]
        #     main_logger.info(f"- Joint: '{joint_name}' (ID: {joint_id}), Q Index: {sim.robot_pin.model.joints[joint_id].idx_q}")

        main_logger.info(f"\nInitial End-Effector Position (World Frame): {sim.pose()[:3, 3]}")
        main_logger.info(f"Initial Joint Angles (MuJoCo qpos, degrees): {np.degrees(sim.data.qpos)}")
        
        main_logger.info("\nViewer opened. It will remain open until you close the window manually.")
        main_logger.info("Check the log file for detailed model information if the viewer closes immediately.")

        # --- Motion Parameters ---
        distance = 0.1  # 10 cm movement in the X direction
        duration = 5.0  # seconds for one direction (e.g., 5s to move forward)
        # Calculate steps based on desired duration and MuJoCo's timestep
        steps = int(duration / sim.model.opt.timestep) 
        
        main_logger.info(f"\nMotion Parameters:")
        main_logger.info(f"- Movement Distance: {distance}m (along X-axis)")
        main_logger.info(f"- Duration per direction: {duration}s")
        main_logger.info(f"- Total simulation steps per direction: {steps}")
        main_logger.info(f"- Simulation Timestep: {sim.model.opt.timestep}s")

        # Capture the initial state of the robot at the start of the entire simulation
        initial_simulation_pose = sim.pose()
        initial_simulation_pos = initial_simulation_pose[:3, 3].copy()
        initial_simulation_rot = initial_simulation_pose[:3, :3]
        
        # --- Main Simulation Loop: Move Forward and Backward ---
        # Outer loop for cycles, inner loop for forward/backward movement
        for cycle in range(2): 
            for direction_sign in [1, -1]: # 1 for forward, -1 for backward
                dir_name = "forward" if direction_sign > 0 else "backward"
                main_logger.info(f"\n=== Cycle {cycle+1} - Moving {dir_name} ===")
                
                # Determine the start and end X positions for this segment of movement
                if direction_sign > 0: # Moving forward
                    start_x_value = initial_simulation_pos[0]
                    end_x_value = initial_simulation_pos[0] + distance
                else: # Moving backward (from the advanced position back to initial)
                    start_x_value = initial_simulation_pos[0] + distance
                    end_x_value = initial_simulation_pos[0]

                # Create an array of target X positions for this phase
                # We use steps + 1 to include both the start and end points in the trajectory
                target_x_values_for_phase = np.linspace(start_x_value, end_x_value, steps + 1)
                
                for t_step in range(steps):
                    if not sim.viewer.is_running(): # Check if viewer is still open
                        main_logger.info("Viewer window closed by user. Stopping simulation.")
                        break 

                    # The target position for this step is taken from the pre-calculated array.
                    # t_step + 1 because t_step is 0-indexed, and we want to move towards the next point in trajectory.
                    target_pos = initial_simulation_pos.copy() # Keep Y and Z from the original start
                    target_pos[0] = target_x_values_for_phase[t_step + 1]
                    
                    # Progress towards the *end of this specific phase/segment*
                    progress = (t_step + 1) / steps 
                    
                    main_logger.info(f"\n--- Step {t_step+1}/{steps} ({dir_name} movement) ---")
                    main_logger.info(f"  Target End-Effector X position (World Frame): {target_pos[0]:.4f}m")
                    main_logger.info(f"  Progress towards target (of this segment): {progress*100:.1f}%")
                    
                    pose = t3d.affines.compose(target_pos, initial_simulation_rot, [1, 1, 1])
                    
                    main_logger.info("  Calling send_action() to move robot...")
                    success = sim.send_action(pose)
                    
                    current_pose = sim.pose()
                    current_pos = current_pose[:3, 3]
                    current_joints = np.degrees(sim.data.qpos[sim.mujoco_qpos_indices_for_actuators]) # Only show actuated joints
                    
                    main_logger.info(f"  Current Robot State (after step):")
                    main_logger.info(f"  - Actual End-Effector X position (World Frame): {current_pos[0]:.4f}m")
                    main_logger.info(f"  - Joint angles (MuJoCo qpos, degrees, actuated only): {current_joints}")
                    
                    if not success:
                        main_logger.error("\n!!! Movement failed due to joint limits or other issues. Stopping simulation. !!!")
                        main_logger.info("\n  Detailed Joint Limit Status (MuJoCo Joints):")
                        for j_idx, mujoco_qpos_idx in enumerate(sim.mujoco_qpos_indices_for_actuators): 
                            mujoco_joint_id = sim.model.actuator_trnid[sim.mujoco_actuator_ids[j_idx], 0]
                            if mujoco_joint_id != -1 and sim.model.jnt_limited[mujoco_joint_id]: 
                                limit_min = np.degrees(sim.model.jnt_range[mujoco_joint_id,0])
                                limit_max = np.degrees(sim.model.jnt_range[mujoco_joint_id,1])
                                current = np.degrees(sim.data.qpos[mujoco_qpos_idx])
                                status = "OK" if limit_min <= current <= limit_max else "VIOLATION!"
                                main_logger.info(f"    Joint '{sim.model.joint(mujoco_joint_id).name}' (ID: {mujoco_joint_id}): {current:.2f}° (Limits: {limit_min:.2f} to {limit_max:.2f}°) - {status}")
                        break 
                    
                    # Removed time.sleep to allow simulation to run at maximum speed
                    # time.sleep(sim.model.opt.timestep) 
                
                if not success or not sim.viewer.is_running():
                    break 
            
            if not success or not sim.viewer.is_running():
                break

    except Exception as e:
        main_logger.critical(f"\n!!! An unexpected exception occurred during simulation: {str(e)}", exc_info=True)
    finally:
        if sim:
            main_logger.info("\nClosing simulation resources...")
            sim.close()
    main_logger.info("Simulation process finished.")
