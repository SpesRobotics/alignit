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

class MuJoCoRobot:
    # Removed debug_mode parameter, as logging handles this now
    def __init__(self, mjcf_path, urdf_path_pinocchio, end_effector_frame_name_pinocchio):
        """
        Initializes the MuJoCo simulation environment and the Pinocchio robot model.

        Args:
            mjcf_path (Path): Path to the MuJoCo MJCF model file.
            urdf_path_pinocchio (Path): Path to the URDF model file for Pinocchio.
            end_effector_frame_name_pinocchio (str): Name of the end-effector frame in the Pinocchio model.
        """
        # Get a specific logger for this class instance
        self.logger = logging.getLogger(self.__class__.__name__)
        # Optionally, you can set a specific level for this logger if you want it
        # to be more or less verbose than the root logger.
        # Example: self.logger.setLevel(logging.DEBUG) # Uncomment to see debug from ONLY this class

        # --- MuJoCo setup ---
        try:
            self.model = mj.MjModel.from_xml_path(str(mjcf_path))
            self.data = mj.MjData(self.model)
            
            # Configure simulation for stability and accuracy
            self.model.opt.timestep = 0.0005# Smaller timestep for more precise integration
            self.model.opt.iterations = 500   # More solver iterations for better constraint satisfaction
            self.model.opt.tolerance = 1e-12  # Solver tolerance
            self.model.opt.solver = mj.mjtSolver.mjSOL_NEWTON  # Use Newton solver for better convergence
            
            # Add damping to all joints to improve stability and prevent oscillations
            for i in range(self.model.nv): # nv is the number of degrees of freedom (qpos size)
                self.model.dof_damping[i] = 0.1
                
            self.logger.debug(f"Successfully loaded MuJoCo model from: {mjcf_path}")
            self.logger.debug(f"MuJoCo model has {self.model.nv} degrees of freedom (qpos size).")
            self.logger.debug(f"MuJoCo model has {self.model.nu} actuators (ctrl size).")
        except Exception as e:
            # Use logger.exception() to automatically include traceback info
            self.logger.critical(f"Failed to load MuJoCo model from {mjcf_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load MuJoCo model from {mjcf_path}: {e}")

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.logger.debug("MuJoCo passive viewer launched.")

        # --- Offscreen rendering setup (lazy initialization) ---
        # These variables are initialized only when get_observation is called without a viewer
        self.scn = None
        self.cam = None
        self.vopt = None
        self.pert = None
        self.mjr_context = None
        self._offscreen_initialized = False

        # Get the ID of the end-effector site in the MuJoCo model
        self.eef_site_id = self.model.site("gripper_site").id
        
        # --- Pinocchio setup ---
        try:
            self.robot_pin = pin.RobotWrapper.BuildFromURDF(str(urdf_path_pinocchio))
            # Get the ID of the end-effector frame in the Pinocchio model
            self.end_effector_frame_id_pin = self.robot_pin.model.getFrameId(end_effector_frame_name_pinocchio)
            if self.end_effector_frame_id_pin == self.robot_pin.model.nframes:
                raise ValueError(f"End-effector frame '{end_effector_frame_name_pinocchio}' not found in Pinocchio model.")
            self.logger.debug(f"Pinocchio model loaded from: {urdf_path_pinocchio}")
            self.logger.debug(f"Pinocchio model has {self.robot_pin.model.nq} generalized coordinates (q size).")
            self.logger.debug(f"Pinocchio model has {self.robot_pin.model.nv} velocity dimensions (v size).")
        except Exception as e:
            self.logger.critical(f"Failed to initialize Pinocchio: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Pinocchio: {e}")

        # Initialize Pinocchio's current configuration to its neutral pose
        self.current_q_pin = pin.neutral(self.robot_pin.model)
        
        # --- Map MuJoCo actuators to Pinocchio joint indices ---
        self.mujoco_actuator_ids = []
        self.mujoco_actuator_to_pinocchio_q_idx = [] 
        
        pinocchio_joint_name_to_q_idx_map = {}
        for joint_id in range(1, self.robot_pin.model.njoints): 
            joint_model = self.robot_pin.model.joints[joint_id]
            joint_name = self.robot_pin.model.names[joint_id] 
            pinocchio_joint_name_to_q_idx_map[joint_name] = joint_model.idx_q

        self.logger.debug(f"Pinocchio Joint Name to q_idx Map: {pinocchio_joint_name_to_q_idx_map}")

        for i in range(self.model.nu): 
            actuator_name = self.model.actuator(i).name
            
            if actuator_name.endswith("_ctrl"):
                joint_name_from_actuator = actuator_name.replace("_ctrl", "")
                
                if joint_name_from_actuator in pinocchio_joint_name_to_q_idx_map:
                    self.mujoco_actuator_ids.append(self.model.actuator(actuator_name).id)
                    self.mujoco_actuator_to_pinocchio_q_idx.append(pinocchio_joint_name_to_q_idx_map[joint_name_from_actuator])
                else:
                    self.logger.warning(f"MuJoCo actuator '{actuator_name}' does not have a corresponding Pinocchio joint '{joint_name_from_actuator}'. Skipping this actuator.")
            else:
                self.logger.warning(f"MuJoCo actuator '{actuator_name}' does not follow '_ctrl' naming convention. Skipping this actuator.")
        
        if len(self.mujoco_actuator_ids) != self.model.nu:
            self.logger.warning(f"Mismatch in number of mapped actuators. MuJoCo has {self.model.nu} actuators, but only {len(self.mujoco_actuator_ids)} were successfully mapped to Pinocchio joints.")
        
        if not self.mujoco_actuator_ids:
            self.logger.critical("No MuJoCo actuators were successfully mapped. Please check your URDF/MJCF joint and actuator naming conventions.")
            raise RuntimeError("No MuJoCo actuators were successfully mapped. Please check your URDF/MJCF joint and actuator naming conventions.")
        
        self.logger.debug(f"Mapped MuJoCo Actuator IDs: {self.mujoco_actuator_ids}")
        self.logger.debug(f"Corresponding Pinocchio q indices for mapped actuators: {self.mujoco_actuator_to_pinocchio_q_idx}")

        # --- Initializing MuJoCo qpos to a valid starting pose ---
        initial_qpos_rad = np.array([0.0, 0.0, np.deg2rad(5.0), 0.0, 0.0, 0.0])
        self.data.qpos[:] = initial_qpos_rad
        mj.mj_forward(self.model, self.data) # Propagate this initial state through the model
        
        self.logger.debug(f"Initial MuJoCo qpos set to: {np.degrees(self.data.qpos)} degrees")
        self.viewer.sync() # Update viewer to show initial pose
        time.sleep(1.0) # Pause to see initial pose

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
        target_se3 = pin.SE3(target_pose_matrix[:3, :3], target_pose_matrix[:3, 3])
        q = self.current_q_pin.copy()  
        
        eps = 1e-4   
        IT_MAX = 100 
        DT = 0.5     
        damp = 1e-3  
        
        self.logger.debug("\nStarting IK calculation with Pinocchio...")
        for i in range(IT_MAX):
            pin.forwardKinematics(self.robot_pin.model, self.robot_pin.data, q)
            pin.updateFramePlacements(self.robot_pin.model, self.robot_pin.data)
            
            current_se3 = self.robot_pin.data.oMf[self.end_effector_frame_id_pin]
            err = pin.log6(current_se3.inverse() * target_se3).vector 
            
            error_norm = np.linalg.norm(err)
            # self.logger.debug(f"  IK Iteration {i}: Error norm = {error_norm:.6f}") # Uncomment for detailed IK per-iteration debug

            if error_norm < eps:
                self.logger.debug(f"IK converged in {i} iterations. Final error norm: {error_norm:.6f}")
                break

            J = pin.computeFrameJacobian(
                self.robot_pin.model, self.robot_pin.data, q,
                self.end_effector_frame_id_pin,
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
        self.model.site_pos[self.model.site("gripper_site").id] = action_pose_matrix[:3, 3]
        self.model.site_quat[self.model.site("gripper_site").id] = t3d.quaternions.mat2quat(action_pose_matrix[:3, :3])
        
        base_pos = self.data.body("link_base").xpos
        base_rot = self.data.body("link_base").xmat.reshape(3,3)
        world_to_base = t3d.affines.compose(base_pos, base_rot, [1,1,1])
        base_target = np.linalg.inv(world_to_base) @ action_pose_matrix

        full_pinocchio_q = self._calculate_ik_pinocchio(base_target)
        
        target_joint_poses = np.array([full_pinocchio_q[idx] for idx in self.mujoco_actuator_to_pinocchio_q_idx])
        
        self.logger.debug(f"IK calculated target_joint_poses (for MuJoCo actuators, shape: {target_joint_poses.shape}): {np.degrees(target_joint_poses)}")
        
        for i, actuator_id in enumerate(self.mujoco_actuator_ids):
            q_target_for_this_actuator = target_joint_poses[i]
            self.logger.debug(f"q target for actuator {actuator_id} : {np.degrees(q_target_for_this_actuator):.4f} degrees")
            
            actuator_name = self.model.actuator(actuator_id).name
            joint_name_from_actuator = actuator_name.replace("_ctrl", "")
            mujoco_joint_id = self.model.joint(joint_name_from_actuator).id
            
            if self.model.jnt_limited[mujoco_joint_id]: 
                lower_limit = self.model.jnt_range[mujoco_joint_id, 0]
                upper_limit = self.model.jnt_range[mujoco_joint_id, 1]
                self.logger.debug(f"lower limit: {np.degrees(lower_limit):.4f} degrees")
                self.logger.debug(f"upper limit: {np.degrees(upper_limit):.4f} degrees")
                buffer = 0.005 
                
                if not (lower_limit + buffer <= q_target_for_this_actuator <= upper_limit - buffer):
                    self.logger.warning(f"Joint limit violation detected for MuJoCo joint '{self.model.joint(mujoco_joint_id).name}' (controlled by actuator '{self.model.actuator(actuator_id).name}'):")
                    self.logger.warning(f"  Target angle: {np.degrees(q_target_for_this_actuator):.2f}°")
                    self.logger.warning(f"  Allowed limits (with buffer): [{np.degrees(lower_limit + buffer):.2f}°, {np.degrees(upper_limit - buffer):.2f}°]")
                    self.logger.warning(f"  Original limits: [{np.degrees(lower_limit):.2f}°, {np.degrees(upper_limit):.2f}°]")
                    return False  

        self.data.ctrl[self.mujoco_actuator_ids] = target_joint_poses
        self.logger.debug(f"MuJoCo data.ctrl after assignment: {self.data.ctrl}")
        mj.mj_step(self.model, self.data)
        
        if self.viewer.is_running():
            self.viewer.sync()
        return True 

    def pose(self):
        """
        Gets the current end-effector pose from the MuJoCo simulation.
        """
        pos = self.data.site_xpos[self.eef_site_id]
        rot_mat = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
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
        if hasattr(self, 'mjr_context') and self.mjr_context:
            try:
                self.logger.debug("Freeing MuJoCo rendering context.")
                mj.mjr_freeContext(self.mjr_context)
            except Exception as e:
                self.logger.warning(f"Failed to free MjrContext: {e}", exc_info=True)

if __name__ == "__main__":
    # os.environ['MUJOCO_GL'] = 'osmesa' # Keep commented out if you want to see the viewer window
    
    # --- Configuration ---
    mjcf_file = Path("/home/nikola/code/alignit/alignit/lite6mjcf.xml")
    urdf_file_pinocchio = Path("/home/nikola/code/alignit/alignit/lite6.urdf")
    end_effector_link_name = "link6" 

    main_logger.info("=== Initializing Robot Simulation ===")
    sim = None
    try:
        main_logger.info("Loading MJCF and URDF models...")
        sim = MuJoCoRobot(mjcf_file, urdf_file_pinocchio, end_effector_link_name) 
        
        # Capture the initial state of the robot at the start of the entire simulation
        initial_simulation_pose = sim.pose()
        initial_simulation_pos = initial_simulation_pose[:3, 3].copy()
        initial_simulation_rot = initial_simulation_pose[:3, :3]
        main_logger.info(f"\nInitial End-Effector Position (World Frame): {initial_simulation_pos}")
        main_logger.info(f"Initial Joint Angles (MuJoCo qpos, degrees): {np.degrees(sim.data.qpos)}")
        
        # --- Motion Parameters ---
        distance = 0.1  # 10 cm movement in the X direction
        duration = 5.0  # seconds for one direction (e.g., 5s to move forward)
        steps = int(duration / sim.model.opt.timestep) 
        
        main_logger.info(f"\nMotion Parameters:")
        main_logger.info(f"- Movement Distance: {distance}m (along X-axis)")
        main_logger.info(f"- Duration per direction: {duration}s")
        main_logger.info(f"- Total simulation steps per direction: {steps}")
        main_logger.info(f"- Simulation Timestep: {sim.model.opt.timestep}s")
        
        # --- Main Simulation Loop: Move Forward and Backward ---
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
                    current_joints = np.degrees(sim.data.qpos)
                    
                    main_logger.info(f"  Current Robot State (after step):")
                    main_logger.info(f"  - Actual End-Effector X position (World Frame): {current_pos[0]:.4f}m")
                    main_logger.info(f"  - Joint angles (MuJoCo qpos, degrees): {current_joints}")
                    
                    if not success:
                        main_logger.error("\n!!! Movement failed due to joint limits or other issues. Stopping simulation. !!!")
                        main_logger.info("\n  Detailed Joint Limit Status (MuJoCo Joints):")
                        for j in range(sim.model.nq): 
                            if sim.model.jnt_limited[j]: 
                                limit_min = np.degrees(sim.model.jnt_range[j,0])
                                limit_max = np.degrees(sim.model.jnt_range[j,1])
                                current = current_joints[j]
                                status = "OK" if limit_min <= current <= limit_max else "VIOLATION!"
                                main_logger.info(f"    Joint '{sim.model.joint(j).name}' (ID: {j}): {current:.2f}° (Limits: {limit_min:.2f} to {limit_max:.2f}°) - {status}")
                        break 
                    
                    if not sim.viewer.is_running():
                        main_logger.info("Viewer window closed by user. Stopping simulation.")
                        break 
                    
                    time.sleep(sim.model.opt.timestep)
                
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
