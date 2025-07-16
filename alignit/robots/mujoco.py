import mujoco as mj
import mujoco.viewer
import numpy as np
import transforms3d as t3d
import time
from pathlib import Path
import os
import logging
import pinocchio as pin
from PIL import Image  
from teleop.utils.jacobi_robot import JacobiRobot
from alignit.robots.robot import Robot

GLOBAL_GL_CONTEXT_WIDTH = 320
GLOBAL_GL_CONTEXT_HEIGHT = 240
_mujoco_gl_context_initialized = True
mj.GL_RENDER = mj.GLContext(max_width=GLOBAL_GL_CONTEXT_WIDTH, max_height=GLOBAL_GL_CONTEXT_HEIGHT)
mj.GL_RENDER.make_current()

class MuJoCoRobot(Robot ):
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

    def _initialize_offscreen_rendering(self):
        """
        Initializes the offscreen rendering context for capturing images.
        This method sets up the necessary MuJoCo rendering components.
        It should ideally be called once successfully.
        """
        global _mujoco_gl_context_initialized

        try:
            self.scn = mj.MjvScene(self.model, maxgeom=1000)
            self.cam = mj.MjvCamera()
            self.vopt = mj.MjvOption()
            self.pert = mj.MjvPerturb()

            mj.mjv_defaultCamera(self.cam)
            mj.mjv_defaultOption(self.vopt)

            self.mjr_context = mj.MjrContext(self.model, 100)  # 100 for font scale

            self._offscreen_initialized = True
        except Exception as e:
            self._offscreen_initialized = False
            self.mjr_context = None # Ensure context is reset on failure

    def get_observation(self, camera_name: str = "front_camera"):
        """
        Gets the current observation from the simulation, including camera RGB data if offscreen rendering is active.

        Args:
            camera_name (str): The name of the camera to use for observation. Defaults to "front_camera".

        Returns:
            dict: A dictionary containing 'qpos', 'qvel', 'eef_pose', and optionally 'camera.rgb'.
        """
        global _mujoco_gl_context_initialized

        if not self._offscreen_initialized:
            self._initialize_offscreen_rendering()
          
        try:
            mj.mjv_updateScene(self.model, self.data, self.vopt, self.pert, self.cam, mj.mjtCatBit.mjCAT_ALL, self.scn)

            if camera_name in [self.model.camera(i).name for i in range(self.model.ncam)]:
                self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
                self.cam.fixedcamid = self.model.camera(camera_name).id
                self.logger.debug(f"Using fixed camera: {camera_name}")
            else:
                self.cam.type = mj.mjtCamera.mjCAMERA_FREE
                mj.mjv_defaultCamera(self.cam)  # Reset to default if the named camera is not found
                self.logger.debug("Using free camera (default).")

            width, height = 320, 240
            viewport = mj.MjrRect(0, 0, width, height)

            rgb_data = np.zeros((height, width, 3), dtype=np.uint8)
            depth_data = np.zeros((height, width), dtype=np.float32)

            mj.mjr_render(viewport, self.scn, self.mjr_context)

            mj.mjr_readPixels(rgb_data, depth_data, viewport, self.mjr_context)

            return {
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "eef_pose": self.pose(),
                "camera.rgb": np.flipud(rgb_data)
            }
        except Exception:
            return False

    def close(self):
        """
        Closes the MuJoCo viewer (if active) and frees the offscreen rendering context.
        """
        self.logger.debug("Closing MuJoCo resources.")
        if self.viewer:
            self.logger.debug("Closing MuJoCo viewer.")
            self.viewer.close()
            self.viewer = None
        if self.mjr_context and self._offscreen_initialized:
            try:
                self.mjr_context = None
                self._offscreen_initialized = False
            except Exception as e:
                quit
        global _mujoco_gl_context_initialized
        if _mujoco_gl_context_initialized:
            try:
                _mujoco_gl_context_initialized = False 
            except Exception as e:
                quit
if __name__ == "__main__":

    sim = None
    output_dir = "captured_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {os.path.abspath(output_dir)}")

    try:
        sim = MuJoCoRobot()
        initial_pose = sim.pose()
        initial_pos = initial_pose[:3, 3].copy()
        initial_rot = initial_pose[:3, :3].copy()

        target_pos = initial_pos + np.array([0.1, 0.0, 0.0])
        target_rot = initial_rot
        target_pose_matrix = t3d.affines.compose(target_pos, target_rot, [1, 1, 1])

        duration_s = 3.0  
        steps_to_simulate = int(duration_s / sim.model.opt.timestep)

        for i in range(steps_to_simulate):
            success = sim.send_action(target_pose_matrix)
            print(f"Current robot pose: {sim.pose()}")
            if success:
                # Capture and save an image periodically (e.g., every 50 steps) or at the end
                if (i % 50 == 0) or (i == steps_to_simulate - 1):
                    observation = sim.get_observation(camera_name="gripper_camera")
                    if "camera.rgb" in observation and observation["camera.rgb"] is not None:
                        rgb_image = Image.fromarray(observation["camera.rgb"])
                        image_filename = os.path.join(output_dir, f"frame_{i:05d}.png")
                        rgb_image.save(image_filename)
                    else:
                        quit

        final_pose = sim.pose()
        final_pos = final_pose[:3, 3]
        final_rot = final_pose[:3, :3]

        position_error = np.linalg.norm(final_pos - target_pos)
        tolerance = 0.002 

        if position_error <= tolerance:
            print(f"Robot has arrived to a desired pose with a tolerance of {position_error*1000:.2f} mm")
        else:
            print(f"Robot did not arrive at a desired position, position error ({position_error:.4f} m) > tolerance error ({tolerance} m)")

    except Exception as e:
        quit
    finally:
        if sim:
            print("\nClosing simulation resources...")
            sim.close()
    print("Simulation finished.")
