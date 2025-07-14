import mujoco as mj
import mujoco.viewer
import numpy as np
import transforms3d as t3d
import time
import pinocchio as pin
from pathlib import Path
import os
# os.environ['MUJOCO_GL'] = 'osmesa'  # Keep commented out if you want to see the viewer window

class MuJoCoRobot:
    def __init__(self, mjcf_path, urdf_path_pinocchio, end_effector_frame_name_pinocchio):
        """
        Initializes the MuJoCo simulation environment and the Pinocchio robot model.

        Args:
            mjcf_path (Path): Path to the MuJoCo MJCF model file.
            urdf_path_pinocchio (Path): Path to the URDF model file for Pinocchio.
            end_effector_frame_name_pinocchio (str): Name of the end-effector frame in the Pinocchio model.
        """
        # --- MuJoCo setup ---
        try:
            self.model = mj.MjModel.from_xml_path(str(mjcf_path))
            self.data = mj.MjData(self.model)
            
            # Configure simulation for stability and accuracy
            self.model.opt.timestep = 0.002  # Smaller timestep for more precise integration
            self.model.opt.iterations = 100    # More solver iterations for better constraint satisfaction
            self.model.opt.tolerance = 1e-10   # Solver tolerance
            self.model.opt.solver = mj.mjtSolver.mjSOL_NEWTON  # Use Newton solver for better convergence
            
            # Add damping to all joints to improve stability and prevent oscillations
            for i in range(self.model.nv): # nv is the number of degrees of freedom (qpos size)
                self.model.dof_damping[i] = 2.0
                
            print(f"DEBUG: Successfully loaded MuJoCo model from: {mjcf_path}")
            print(f"DEBUG: MuJoCo model has {self.model.nv} degrees of freedom (qpos size).")
            print(f"DEBUG: MuJoCo model has {self.model.nu} actuators (ctrl size).")
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {mjcf_path}: {e}")

        # Initialize passive viewer for real-time visualization
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        print("DEBUG: MuJoCo passive viewer launched.")

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
            print(f"DEBUG: Pinocchio model loaded from: {urdf_path_pinocchio}")
            print(f"DEBUG: Pinocchio model has {self.robot_pin.model.nq} generalized coordinates (q size).")
            print(f"DEBUG: Pinocchio model has {self.robot_pin.model.nv} velocity dimensions (v size).")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinocchio: {e}")

        # Initialize Pinocchio's current configuration to its neutral pose
        self.current_q_pin = pin.neutral(self.robot_pin.model)
        
        # --- Map MuJoCo actuators to Pinocchio joint indices ---
        # This is crucial for correctly transferring IK solutions (from Pinocchio's q)
        # to MuJoCo's control inputs (self.data.ctrl).
        self.mujoco_actuator_ids = []
        # This list will store the corresponding index in Pinocchio's 'q' vector
        # for each MuJoCo actuator.
        self.mujoco_actuator_to_pinocchio_q_idx = [] 
        
        # Create a map from Pinocchio joint name to its starting index in Pinocchio's 'q' vector.
        # Pinocchio's model.njoints is the number of actual joints (excluding the universe joint at index 0).
        pinocchio_joint_name_to_q_idx_map = {}
        # Iterate through all joints in the Pinocchio model, skipping the 'universe' joint (index 0).
        for joint_id in range(1, self.robot_pin.model.njoints): 
            # Get the JointModel object for the current joint_id
            joint_model = self.robot_pin.model.joints[joint_id]
            # Get the name of the joint using its ID (which is the same as its index in model.names for joints)
            joint_name = self.robot_pin.model.names[joint_id] 
            
            # idx_q is the starting index in the configuration vector 'q' for this joint.
            # For revolute joints, it's typically just the index of that joint's angle in 'q'.
            pinocchio_joint_name_to_q_idx_map[joint_name] = joint_model.idx_q

        print("DEBUG: Pinocchio Joint Name to q_idx Map:", pinocchio_joint_name_to_q_idx_map)

        # Iterate through all MuJoCo actuators to find their corresponding Pinocchio joints.
        for i in range(self.model.nu): # self.model.nu is the total number of actuators in MuJoCo.
            actuator_name = self.model.actuator(i).name
            
            # Assuming MuJoCo actuator names follow a convention like "jointX_ctrl"
            # and the corresponding Pinocchio joint name is "jointX".
            if actuator_name.endswith("_ctrl"):
                joint_name_from_actuator = actuator_name.replace("_ctrl", "")
                
                if joint_name_from_actuator in pinocchio_joint_name_to_q_idx_map:
                    self.mujoco_actuator_ids.append(self.model.actuator(actuator_name).id)
                    # Store the Pinocchio 'q' index for this MuJoCo actuator.
                    self.mujoco_actuator_to_pinocchio_q_idx.append(pinocchio_joint_name_to_q_idx_map[joint_name_from_actuator])
                else:
                    print(f"WARNING: MuJoCo actuator '{actuator_name}' does not have a corresponding Pinocchio joint '{joint_name_from_actuator}'. Skipping this actuator.")
            else:
                print(f"WARNING: MuJoCo actuator '{actuator_name}' does not follow '_ctrl' naming convention. Skipping this actuator.")
        
        # Check if all MuJoCo actuators were successfully mapped.
        if len(self.mujoco_actuator_ids) != self.model.nu:
            print(f"WARNING: Mismatch in number of mapped actuators. MuJoCo has {self.model.nu} actuators, but only {len(self.mujoco_actuator_ids)} were successfully mapped to Pinocchio joints.")
        
        if not self.mujoco_actuator_ids:
            raise RuntimeError("No MuJoCo actuators were successfully mapped. Please check your URDF/MJCF joint and actuator naming conventions.")
        
        print(f"DEBUG: Mapped MuJoCo Actuator IDs: {self.mujoco_actuator_ids}")
        print(f"DEBUG: Corresponding Pinocchio q indices for mapped actuators: {self.mujoco_actuator_to_pinocchio_q_idx}")

        # --- Warmup simulation ---
        # Perform a few steps to stabilize the simulation and initialize physics states.
        for _ in range(10):
            mj.mj_step(self.model, self.data)
            self.data.ctrl[:] = 0 # Ensure controls are zeroed during warmup

    def _initialize_offscreen_rendering(self):
        """
        Lazy initialization of offscreen rendering components for camera observations.
        This is called only if get_observation is requested and the viewer is not running.
        """
        if self._offscreen_initialized:
            return
            
        try:
            self.scn = mj.MjvScene(self.model, maxgeom=1000) # Scene object for rendering
            self.cam = mj.MjvCamera() # Camera object
            mj.mjv_defaultCamera(self.cam) # Set default camera parameters
            self.vopt = mj.MjvOption() # Visualization options
            self.pert = mj.MjvPerturb() # Perturbation object (for camera control)
            # Graphics rendering context
            self.mjr_context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
            self._offscreen_initialized = True
            print("DEBUG: Offscreen rendering initialized.")
        except Exception as e:
            print(f"WARNING: Failed offscreen rendering initialization: {e}")

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
        # Convert target pose matrix to Pinocchio's SE3 object
        target_se3 = pin.SE3(target_pose_matrix[:3, :3], target_pose_matrix[:3, 3])
        # Start IK from the current Pinocchio configuration
        q = self.current_q_pin.copy()  
        
        # Parameters for the IK solver (Damped Least Squares with Nullspace Control)
        eps = 1e-4   # Convergence tolerance: IK stops when error norm is below this
        IT_MAX = 100 # Maximum number of iterations for the IK solver
        DT = 0.5     # Integration step for updating 'q' (larger steps can converge faster but risk instability)
        damp = 1e-6  # Damping factor for DLS, helps with singularities
        
        print("\nDEBUG: Starting IK calculation with Pinocchio...")
        for i in range(IT_MAX):
            # Update forward kinematics and frame placements for the current 'q'
            pin.forwardKinematics(self.robot_pin.model, self.robot_pin.data, q)
            pin.updateFramePlacements(self.robot_pin.model, self.robot_pin.data)
            
            # Get the current end-effector pose
            current_se3 = self.robot_pin.data.oMf[self.end_effector_frame_id_pin]
            # Calculate the error (difference) between current and target pose in SE(3) space
            err = pin.log6(current_se3.inverse() * target_se3).vector 
            
            error_norm = np.linalg.norm(err)
            # print(f"  IK Iteration {i}: Error norm = {error_norm:.6f}") # Uncomment for detailed IK per-iteration debug

            if error_norm < eps:
                print(f"DEBUG: IK converged in {i} iterations. Final error norm: {error_norm:.6f}")
                break

            # Compute the Jacobian for the end-effector frame in its LOCAL reference frame.
            # LOCAL frame is generally good for IK as it describes velocities relative to the end-effector.
            J = pin.computeFrameJacobian(
                self.robot_pin.model, self.robot_pin.data, q,
                self.end_effector_frame_id_pin,
                pin.ReferenceFrame.LOCAL
            )
            
            # Damped Least-Squares (DLS) solution for the change in joint angles (dq)
            # This is a more numerically stable way to compute the pseudo-inverse with damping.
            U, S, Vh = np.linalg.svd(J)
            S_inv = np.zeros(J.shape[1])
            # Apply damping to singular values to avoid large joint velocity commands near singularities
            S_inv[:len(S)] = S / (S**2 + damp) 
            dq = Vh.T @ np.diag(S_inv) @ U.T @ err

            # Nullspace control: Add a component to dq that doesn't affect the end-effector pose,
            # but moves the robot towards a "neutral" configuration. This helps avoid awkward poses.
            q_neutral = pin.neutral(self.robot_pin.model)
            # Project (q - q_neutral) into the nullspace of J (i.e., motions that don't move the end-effector)
            nullspace_projection = (np.eye(J.shape[1]) - np.linalg.pinv(J) @ J) @ (q - q_neutral)
            dq += 0.01 * nullspace_projection # Small scaling factor to gently nudge towards neutral
            
            # Update the configuration 'q' using Pinocchio's integration function.
            # This is important for Lie group manifolds (like rotations) to ensure proper updates.
            q = pin.integrate(self.robot_pin.model, q, dq * DT) 
            
            # Clip 'q' to Pinocchio's model-defined joint limits.
            # This is a general check within the IK solver to keep 'q' within valid ranges.
            # More precise MuJoCo-specific limit checks are done in send_action.
            q = np.clip(q, self.robot_pin.model.lowerPositionLimit, self.robot_pin.model.upperPositionLimit)

        # Warn if IK did not fully converge within max iterations
        if np.linalg.norm(err) >= eps:
            print(f"WARNING: IK did not fully converge after {IT_MAX} iterations. Final error norm: {np.linalg.norm(err):.6f}")
        
        self.current_q_pin = q.copy() # Store the last successful IK configuration
        return q # Return the full Pinocchio 'q' vector

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
        # Update the visual target site (red dot) in the MuJoCo viewer.
        # This is purely for visualization and does not control the robot's actual joints.
        self.model.site_pos[self.model.site("gripper_site").id] = action_pose_matrix[:3, 3]
        self.model.site_quat[self.model.site("gripper_site").id] = t3d.quaternions.mat2quat(action_pose_matrix[:3, :3])
        
        # Transform the world-frame target pose into the robot's base frame.
        # Pinocchio's IK typically expects targets relative to the robot's base.
        base_pos = self.data.body("link_base").xpos
        base_rot = self.data.body("link_base").xmat.reshape(3,3)
        world_to_base = t3d.affines.compose(base_pos, base_rot, [1,1,1])
        base_target = np.linalg.inv(world_to_base) @ action_pose_matrix

        # Calculate the full joint configuration (q) using Pinocchio's IK.
        full_pinocchio_q = self._calculate_ik_pinocchio(base_target)
        
        # Extract the relevant joint angles from Pinocchio's 'q' for MuJoCo's actuators.
        # This uses the mapping created in __init__ to ensure correct indexing.
        target_joint_poses = np.array([full_pinocchio_q[idx] for idx in self.mujoco_actuator_to_pinocchio_q_idx])
        
        print(f"DEBUG: IK calculated target_joint_poses (for MuJoCo actuators, shape: {target_joint_poses.shape}): {np.degrees(target_joint_poses)}")
        
        # --- Validate joint limits for MuJoCo's actuated joints ---
        # This is a critical step to prevent the robot from trying to move into invalid poses.
        for i, actuator_id in enumerate(self.mujoco_actuator_ids):
            q_target_for_this_actuator = target_joint_poses[i]
            
            # Get the MuJoCo joint ID that this specific actuator controls.
            # Corrected: Get joint name from actuator name, then get joint ID from model.
            actuator_name = self.model.actuator(actuator_id).name
            # Assuming actuator name is like "jointX_ctrl" and joint name is "jointX"
            joint_name_from_actuator = actuator_name.replace("_ctrl", "")
            mujoco_joint_id = self.model.joint(joint_name_from_actuator).id
            
            # Check if this MuJoCo joint has defined limits.
            if self.model.jnt_limited[mujoco_joint_id]: 
                lower_limit = self.model.jnt_range[mujoco_joint_id, 0]
                upper_limit = self.model.jnt_range[mujoco_joint_id, 1]
                
                # Add a small buffer to the limits. This helps to:
                # 1. Account for numerical precision issues in IK.
                # 2. Prevent the robot from getting stuck exactly at the limit, which can cause instability.
                buffer = 0.005 # radians (approximately 0.28 degrees)
                
                # Check if the target joint angle is within the allowed range (with buffer).
                if not (lower_limit + buffer <= q_target_for_this_actuator <= upper_limit - buffer):
                    print(f"WARNING: Joint limit violation detected for MuJoCo joint '{self.model.joint(mujoco_joint_id).name}' (controlled by actuator '{self.model.actuator(actuator_id).name}'):")
                    print(f"  Target angle: {np.degrees(q_target_for_this_actuator):.2f}°")
                    print(f"  Allowed limits (with buffer): [{np.degrees(lower_limit + buffer):.2f}°, {np.degrees(upper_limit - buffer):.2f}°]")
                    print(f"  Original limits: [{np.degrees(lower_limit):.2f}°, {np.degrees(upper_limit):.2f}°]")
                    return False  # Return False to indicate that the action cannot be performed safely.

        # Assign the calculated joint positions to MuJoCo's control inputs.
        # This is where the robot's actual movement is commanded.
        self.data.ctrl[self.mujoco_actuator_ids] = target_joint_poses
        
        # Step the MuJoCo simulation forward by one timestep.
        mj.mj_step(self.model, self.data)
        
        # Sync the viewer to update the display with the new robot state.
        if self.viewer.is_running():
            self.viewer.sync()
        return True # Action successfully processed

    def pose(self):
        """
        Gets the current end-effector pose from the MuJoCo simulation.

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix representing
                        the current world-frame pose of the end-effector.
        """
        pos = self.data.site_xpos[self.eef_site_id]
        rot_mat = self.data.site_xmat[self.eef_site_id].reshape(3, 3)
        return t3d.affines.compose(pos, rot_mat, [1, 1, 1])

    def get_observation(self, camera_name: str = "front_camera"):
        """
        Gets the current observation from the simulation, including joint states
        and optionally a camera image if offscreen rendering is enabled.

        Args:
            camera_name (str): The name of the camera to render from (e.g., "front_camera").

        Returns:
            dict: A dictionary containing 'qpos', 'qvel', 'eef_pose', and optionally 'camera.rgb'.
        """
        # If the viewer is running, we don't attempt offscreen rendering to avoid conflicts.
        if self.viewer.is_running():
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose()
            }
            
        # Initialize offscreen rendering if it hasn't been already.
        self._offscreen_initialized = False # Reset for retry if it failed previously
        self._initialize_offscreen_rendering()
        if not self._offscreen_initialized:
            # If offscreen rendering failed to initialize, return basic state.
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose()
            }
            
        try:
            # Configure the camera for rendering.
            if camera_name in [self.model.camera(i).name for i in range(self.model.ncam)]:
                self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
                self.cam.fixedcamid = self.model.camera(camera_name).id
            else:
                self.cam.type = mj.mjtCamera.mjCAMERA_FREE
                mj.mjv_defaultCamera(self.cam)
                
            # Render the scene to an RGB image and depth map.
            mj.mjv_updateScene(self.model, self.data, self.vopt, self.pert, self.cam, 0, self.scn)
            rgb_data = np.zeros((240, 320, 3), dtype=np.uint8) # Image buffer
            depth_data = np.zeros((240, 320), dtype=np.float32) # Depth buffer
            viewport = mj.MjrRect(0, 0, 320, 240) # Define the viewport size
            mj.mjr_render(viewport, self.scn, self.mjr_context)
            mj.mjr_readPixels(rgb_data, depth_data, viewport, self.mjr_context)
            
            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose(),
                "camera.rgb": np.flipud(rgb_data) # Flip vertically as MuJoCo renders upside down
            }
        except Exception as e:
            print(f"WARNING: Camera rendering failed: {e}")
            # If rendering fails, return basic state without camera image.
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
            print("DEBUG: Closing MuJoCo viewer.")
            self.viewer.close()
        if hasattr(self, 'mjr_context') and self.mjr_context:
            try:
                print("DEBUG: Freeing MuJoCo rendering context.")
                mj.mjr_freeContext(self.mjr_context)
            except Exception as e:
                print(f"WARNING: Failed to free MjrContext: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace these paths with the actual paths to your MJCF and URDF files.
    # Example for Windows: mjcf_file = Path("C:/Users/YourUser/Documents/alignit/lite6mjcf.xml")
    # Example for Linux/macOS: urdf_file_pinocchio = Path("/home/youruser/alignit/lite6.urdf")
    mjcf_file = Path("/home/nikola/code/alignit/alignit/lite6mjcf.xml")
    urdf_file_pinocchio = Path("/home/nikola/code/alignit/alignit/lite6.urdf")
    end_effector_link_name = "link6" # This should be the name of the end-effector link in your URDF/MJCF

    print("=== Initializing Robot Simulation ===")
    sim = None
    try:
        print("Loading MJCF and URDF models...")
        sim = MuJoCoRobot(mjcf_file, urdf_file_pinocchio, end_effector_link_name)
        
        # Get the initial end-effector pose and joint angles from the simulation.
        initial_pose = sim.pose()
        initial_pos = initial_pose[:3, 3].copy()
        initial_rot = initial_pose[:3, :3]
        print(f"\nInitial End-Effector Position (World Frame): {initial_pos}")
        print(f"Initial Joint Angles (MuJoCo qpos, degrees): {np.degrees(sim.data.qpos)}")
        
        # --- Motion Parameters ---
        distance = 0.1  # 10 cm movement in the X direction
        duration = 5.0  # seconds for one direction (e.g., 5s to move forward)
        # Calculate the number of simulation steps required for one direction of movement.
        steps = int(duration / sim.model.opt.timestep) 
        
        print(f"\nMotion Parameters:")
        print(f"- Movement Distance: {distance}m (along X-axis)")
        print(f"- Duration per direction: {duration}s")
        print(f"- Total simulation steps per direction: {steps}")
        print(f"- Simulation Timestep: {sim.model.opt.timestep}s")
        
        # --- Main Simulation Loop: Move Forward and Backward ---
        for cycle in range(2):  # Perform 2 complete cycles (forward then backward, then forward then backward)
            for direction in [1, -1]: # 1 for moving forward (positive X), -1 for moving backward (negative X)
                dir_name = "forward" if direction > 0 else "backward"
                print(f"\n=== Cycle {cycle+1} - Moving {dir_name} ===")
                
                for t_step in range(steps):
                    # Calculate the target end-effector position for the current step.
                    # 'progress' goes from 0 to 1 over the 'steps'.
                    progress = t_step / steps 
                    target_pos = initial_pos.copy()
                    # Increment/decrement the X-coordinate based on direction and progress.
                    target_pos[0] += direction * distance * progress 
                    
                    # Print current step details for monitoring.
                    print(f"\n--- Step {t_step+1}/{steps} ({dir_name} movement) ---")
                    print(f"  Target End-Effector X position (World Frame): {target_pos[0]:.4f}m")
                    print(f"  Progress towards target: {progress*100:.1f}%")
                    
                    # Create the 4x4 homogeneous transformation matrix for the target pose.
                    # Orientation remains constant (initial_rot).
                    pose = t3d.affines.compose(target_pos, initial_rot, [1, 1, 1])
                    
                    # Send the action to the robot. This triggers IK calculation and MuJoCo simulation step.
                    print("  Calling send_action() to move robot...")
                    success = sim.send_action(pose)
                    
                    # Get the actual current state of the robot from MuJoCo after the simulation step.
                    current_pose = sim.pose()
                    current_pos = current_pose[:3, 3]
                    current_joints = np.degrees(sim.data.qpos)
                    
                    print(f"  Current Robot State (after step):")
                    print(f"  - Actual End-Effector X position (World Frame): {current_pos[0]:.4f}m")
                    print(f"  - Joint angles (MuJoCo qpos, degrees): {current_joints}")
                    
                    # Check if the action was successful (e.g., no joint limit violations).
                    if not success:
                        print("\n!!! Movement failed due to joint limits or other issues. Stopping simulation. !!!")
                        # Provide a detailed report of joint limit status for debugging.
                        print("\n  Detailed Joint Limit Status (MuJoCo Joints):")
                        for j in range(sim.model.nq): # Iterate through all MuJoCo joints
                            if sim.model.jnt_limited[j]: # Check if the joint has limits defined
                                limit_min = np.degrees(sim.model.jnt_range[j,0])
                                limit_max = np.degrees(sim.model.jnt_range[j,1])
                                current = current_joints[j]
                                status = "OK" if limit_min <= current <= limit_max else "VIOLATION!"
                                print(f"    Joint '{sim.model.joint(j).name}' (ID: {j}): {current:.2f}° (Limits: {limit_min:.2f}° to {limit_max:.2f}°) - {status}")
                        break # Exit the inner 't_step' loop
                    
                    # Check if the viewer window has been closed by the user.
                    if not sim.viewer.is_running():
                        print("Viewer window closed by user. Stopping simulation.")
                        break # Exit the inner 't_step' loop
                    
                    # Introduce a small delay to make the simulation visually observable.
                    time.sleep(sim.model.opt.timestep)
                
                # If any issue occurred in the inner loop, break from the direction loop as well.
                if not success or not sim.viewer.is_running():
                    break 
            
            # If any issue occurred in the direction loop, break from the cycle loop as well.
            if not success or not sim.viewer.is_running():
                break

    except Exception as e:
        # Catch any unexpected exceptions and print a detailed traceback for debugging.
        print(f"\n!!! An unexpected exception occurred during simulation: {str(e)}")
        import traceback
        traceback.print_exc() 
    finally:
        # Ensure simulation resources are properly closed even if an error occurs.
        if sim:
            print("\nClosing simulation resources...")
            sim.close()
    print("Simulation process finished.")
