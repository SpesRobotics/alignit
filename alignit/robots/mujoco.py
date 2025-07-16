import mujoco as mj
import mujoco.viewer
import numpy as np
import transforms3d as t3d
import time
from pathlib import Path
import os
import logging
import pinocchio as pin


# Import JacobiRobot
from teleop.utils.jacobi_robot import JacobiRobot


class MuJoCoRobot:
    def __init__(self):
        """
        Initializes the MuJoCo simulation environment and the JacobiRobot model.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        mjcf_path = Path("/home/nikola/code/alignit/alignit/mujoco_menagerie/ufactory_lite6/scene.xml")
        urdf_path_jacobi = Path("/home/nikola/code/alignit/alignit/lite6.urdf")
        end_effector_frame_name_jacobi = "link6"

        try:
            self.model = mj.MjModel.from_xml_path(str(mjcf_path))
            self.data = mj.MjData(self.model)

            self.model.opt.timestep = 0.005 
            self.model.opt.iterations = 50
            self.model.opt.tolerance = 1e-8
            self.model.opt.solver = mj.mjtSolver.mjSOL_NEWTON

            for i in range(self.model.nv):
                self.model.dof_damping[i] = 0.5 


            # --- Identify End-Effector in MuJoCo ---
            try:
                self.eef_id_type = 'site'
                self.eef_id = self.model.site("gripper_site").id
                self.logger.info("Using 'gripper_site' as MuJoCo end-effector.")
            except KeyError:
                try:
                    self.eef_id_type = 'body'
                    self.eef_id = self.model.body("link6").id
                    self.logger.info("Using 'link6' body as MuJoCo end-effector.")
                except KeyError:
                    self.logger.critical("Neither 'gripper_site' nor 'link6' body found in MuJoCo model for end-effector.")
                    raise RuntimeError("MuJoCo end-effector not found.")

        except Exception as e:
            self.logger.critical(f"Failed to load MuJoCo model from {mjcf_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load MuJoCo model from {mjcf_path}: {e}")

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.scn = None
        self.cam = None
        self.vopt = None
        self.pert = None
        self.mjr_context = None
        self._offscreen_initialized = False

        try:
            self.robot_jacobi = JacobiRobot(str(urdf_path_jacobi), ee_link=end_effector_frame_name_jacobi)
            self.logger.info(f"JacobiRobot model loaded from: {urdf_path_jacobi}")
        except Exception as e:
            self.logger.critical(f"Failed to initialize JacobiRobot: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize JacobiRobot: {e}")

        jacobi_neutral_q = self.robot_jacobi.q.copy()

        # --- Map MuJoCo actuators to JacobiRobot joint indices ---
        self.mujoco_actuator_ids = []
        self.mujoco_actuator_to_jacobi_joint_idx = []
        self.mujoco_qpos_indices_for_actuators = []

        jacobi_joint_name_to_idx_map = {}
        jacobi_joint_names = self.robot_jacobi.get_joint_names() # Call the method!
        self.logger.info("\nJacobiRobot Joint Names and their indices:")
        for joint_idx, joint_name in enumerate(jacobi_joint_names):
            jacobi_joint_name_to_idx_map[joint_name] = joint_idx


        for i in range(self.model.nu):
            actuator_name = self.model.actuator(i).name
            mujoco_joint_id = self.model.actuator_trnid[i, 0]

            mujoco_joint_name = self.model.joint(mujoco_joint_id).name

            if mujoco_joint_name in jacobi_joint_name_to_idx_map:
                self.mujoco_actuator_ids.append(self.model.actuator(i).id)
                self.mujoco_actuator_to_jacobi_joint_idx.append(jacobi_joint_name_to_idx_map[mujoco_joint_name])
                self.mujoco_qpos_indices_for_actuators.append(self.model.joint(mujoco_joint_id).qposadr[0])
            else:
                self.logger.warning(f"    WARNING: MuJoCo joint '{mujoco_joint_name}' (controlled by actuator '{actuator_name}' at index {i}) DOES NOT have a corresponding JacobiRobot joint with the same name. Skipping this actuator.")
       
        mj.mj_resetData(self.model, self.data)

        if len(self.mujoco_qpos_indices_for_actuators) > 0:
            for i, mj_qpos_idx in enumerate(self.mujoco_qpos_indices_for_actuators):
                jacobi_joint_idx = self.mujoco_actuator_to_jacobi_joint_idx[i]
                if jacobi_joint_idx < len(jacobi_neutral_q):
                    self.data.qpos[mj_qpos_idx] = jacobi_neutral_q[jacobi_joint_idx]
                else:
                    self.logger.warning(f"JacobiRobot neutral configuration index {jacobi_joint_idx} out of bounds for MuJoCo qpos index {mj_qpos_idx}.")
            self.logger.info(f"Initial MuJoCo qpos for mapped joints set from JacobiRobot's neutral pose.")
        else:
            self.logger.warning("No mapped joints. MuJoCo's initial pose might not match JacobiRobot's neutral pose.")

        mj.mj_forward(self.model, self.data)
        self.viewer.sync()
        time.sleep(1.0)

        # --- Warmup simulation ---
        for _ in range(10):
            mj.mj_step(self.model, self.data)
            self.data.ctrl[:] = 0



    def send_action(self, target_pose_matrix):
        """
        Sends a target end-effector pose to the robot.
        This function now drives JacobiRobot's internal state using its servo_to_pose,
        and then applies JacobiRobot's internal joint configuration to MuJoCo.

        Args:
            target_pose_matrix (np.ndarray): A 4x4 homogeneous transformation matrix
                                             representing the target pose of the end-effector
                                             relative to the world frame.

        Returns:
            bool: True if the action was successfully processed and simulation step taken, False otherwise.
        """
        if not self.mujoco_actuator_ids:
            self.logger.error("Cannot send action: No MuJoCo actuators mapped to JacobiRobot joints.")
            return False

        try:
            base_pos = self.data.xpos[self.model.body("link_base").id]
            base_rot = self.data.xmat[self.model.body("link_base").id].reshape(3,3)
            world_to_base = t3d.affines.compose(base_pos, base_rot, [1,1,1])
        except KeyError:
            self.logger.error("Body 'link_base' not found in MuJoCo model. Assuming base at world origin (0,0,0, identity).")
            world_to_base = np.eye(4)

        base_target_pose = np.linalg.inv(world_to_base) @ target_pose_matrix
        servo_dt = self.model.opt.timestep   
        self.robot_jacobi.servo_to_pose(base_target_pose, dt=servo_dt)

        full_jacobi_q = self.robot_jacobi.q # JacobiRobot's internal q state
        target_joint_qpos_for_mujoco = np.array([full_jacobi_q[idx] for idx in self.mujoco_actuator_to_jacobi_joint_idx])    
        self.data.ctrl[self.mujoco_actuator_ids] = target_joint_qpos_for_mujoco

        # Step the MuJoCo simulation
        mj.mj_step(self.model, self.data)

        self.viewer.sync()
        return True 
    
    def pose(self):
        """Gets the current end-effector pose from the MuJoCo simulation."""
        if self.eef_id_type == 'site':
            pos = self.data.site_xpos[self.eef_id]
            rot_mat = self.data.site_xmat[self.eef_id].reshape(3, 3)
        else:
            pos = self.data.xpos[self.eef_id]
            rot_mat = self.data.xmat[self.eef_id].reshape(3, 3)
        return t3d.affines.compose(pos, rot_mat, [1, 1, 1])

    def get_observation(self, camera_name: str = "front_camera"):
        """Gets the current observation from the simulation."""
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
        self.logger.debug("Closing MuJoCo viewer.")
        self.viewer.close()
        self.viewer = None
        if hasattr(self, 'mjr_context') and self.mjr_context and self._offscreen_initialized:
            try:
                self.logger.debug("Freeing MuJoCo rendering context.")
                mj.mjr_freeContext(self.mjr_context)
                self.mjr_context = None
                self._offscreen_initialized = False
            except Exception as e:
                self.logger.warning(f"Failed to free MjrContext: {e}", exc_info=True)

if __name__ == "__main__":
   
    sim = None
    try:
        sim = MuJoCoRobot()

        initial_pose = sim.pose()
        initial_pos = initial_pose[:3, 3].copy()
        initial_rot = initial_pose[:3, :3].copy()

        target_pos = initial_pos + np.array([0.1, 0.0, 0.0]) # Move 10cm in positive X
        target_rot = initial_rot # Keep initial orientation
        target_pose_matrix = t3d.affines.compose(target_pos, target_rot, [1, 1, 1])


        duration_s = 3.0 # seconds
        steps_to_simulate = int(duration_s / sim.model.opt.timestep)
        
        for i in range(steps_to_simulate):
            success = sim.send_action(target_pose_matrix) # Continuously send the same target

            if (i % 100 == 0) or (i == steps_to_simulate - 1):
                current_pose = sim.pose()
                current_pos = current_pose[:3, 3]
                pos_error = np.linalg.norm(current_pos - target_pos)


        final_pose = sim.pose()
        final_pos = final_pose[:3, 3]
        final_rot = final_pose[:3, :3]

        position_error = np.linalg.norm(final_pos - target_pos)
       
        tolerance = 0.002 # meters (0.5 cm)

        if position_error <= tolerance:
            print(f"Robot has arrived to a desired pose with a tolerance of {position_error*1000} mm")
        else:
            print(f"Robot did not arrive at a desired position,position error ({position_error} > tolerance error ({tolerance}))")
    except Exception as e:
        print(f"\n!!! An unexpected exception occurred: {str(e)}", exc_info=True)
    finally:
        if sim:
            print("\nClosing simulation resources...")
            time.sleep(5)
            sim.close()
    print("Simulation finished.")