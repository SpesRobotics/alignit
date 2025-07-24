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
import threading

GLOBAL_GL_CONTEXT_WIDTH = 320
GLOBAL_GL_CONTEXT_HEIGHT = 240
_mujoco_gl_context_initialized = True
mj.GL_RENDER = mj.GLContext(max_width=GLOBAL_GL_CONTEXT_WIDTH, max_height=GLOBAL_GL_CONTEXT_HEIGHT)
mj.GL_RENDER.make_current()

class MuJoCoRobot(Robot):
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

            self.model.opt.timestep = 0.008
            self.model.opt.iterations = 50
            self.model.opt.tolerance = 1e-8
            self.model.opt.solver = mj.mjtSolver.mjSOL_NEWTON

            for i in range(self.model.nv):
                self.model.dof_damping[i] = 0.5

            # --- Identify End-Effector in MuJoCo ---
            try:
                self.eef_id_type = 'site'
                self.eef_id = self.model.site("gripper_site").id
            except KeyError:
                try:
                    self.eef_id_type = 'body'
                    self.eef_id = self.model.body("link6").id
                except KeyError:
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
        except Exception as e:
            raise RuntimeError(f"Failed to initialize JacobiRobot: {e}")

        jacobi_neutral_q = self.robot_jacobi.q.copy()

        # --- Map MuJoCo actuators to JacobiRobot joint indices ---
        self.mujoco_actuator_ids = []
        self.mujoco_actuator_to_jacobi_joint_idx = []
        self.mujoco_qpos_indices_for_actuators = []
        self.gripper_ctrl_id = None

        jacobi_joint_name_to_idx_map = {}
        jacobi_joint_names = self.robot_jacobi.get_joint_names()
        for joint_idx, joint_name in enumerate(jacobi_joint_names):
            jacobi_joint_name_to_idx_map[joint_name] = joint_idx

        for i in range(self.model.nu):
            actuator_name = self.model.actuator(i).name
            mujoco_joint_id = self.model.actuator_trnid[i, 0]
            mujoco_joint_name = self.model.joint(mujoco_joint_id).name

            # Special handling for gripper actuator
            if actuator_name == "gripper":
                self.gripper_ctrl_id = i
                self.logger.info(f"Found gripper actuator at index {i}")
                continue  # Skip adding to Jacobi mapping

            if mujoco_joint_name in jacobi_joint_name_to_idx_map:
                self.mujoco_actuator_ids.append(self.model.actuator(i).id)
                self.mujoco_actuator_to_jacobi_joint_idx.append(jacobi_joint_name_to_idx_map[mujoco_joint_name])
                self.mujoco_qpos_indices_for_actuators.append(self.model.joint(mujoco_joint_id).qposadr[0])
            else:
                self.logger.warning(f"MuJoCo joint '{mujoco_joint_name}' (actuator '{actuator_name}' at index {i}) has no JacobiRobot counterpart")

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

        self.gripper_open_pos = 0.008
        self.gripper_close_pos = -0.008
        self.current_gripper_pos = 0.0

        mj.mj_forward(self.model, self.data)
        self.viewer.sync()
        time.sleep(1.0)
    def stop_object_movement(self, object_name="pickup_object"):
        start_time = self.data.time
        while self.data.body("pickup_object").xpos[2] > 0.08:  
            mj.mj_step(self.model, self.data)
            self.viewer.sync()
        self.model.opt.gravity[:]=[0,0,0]
        obj_id = self.model.body(object_name).id
        joint_id = self.model.body_jntadr[obj_id]
        if joint_id != -1:
            qvel_adr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[qvel_adr:qvel_adr+6] = 0 
            mj.mj_forward(self.model, self.data)
    def groff(self):
        self.model.opt.gravity[:]=[0,0,0]
        mj.mj_forward(self.model, self.data)

    def set_object_pose(self, object_name, pose_matrix):
        """
        Set the object's pose using a 4x4 transformation matrix.
        Compatible with modern MuJoCo Python bindings.
        """
        try:
            # Get body ID
            body_id = self.model.body(object_name).id
            
            # Set position directly in xpos
            self.data.xpos[body_id] = pose_matrix[:3, 3]
            
            # Convert rotation matrix to quaternion (w, x, y, z)
            quat = t3d.quaternions.mat2quat(pose_matrix[:3, :3])
            self.data.xquat[body_id] = quat
            
            # Alternative method for free joints (if direct xpos/xquat doesn't work)
            # Find associated joint (if any)
            joint_id = self.model.body_jntadr[body_id]
            if joint_id >= 0:  # If body has a joint
                jnt_type = self.model.jnt_type[joint_id]
                qpos_adr = self.model.jnt_qposadr[joint_id]
                
                if jnt_type == 0:  # mjJNT_FREE
                    # Set position (first 3) and orientation (next 4)
                    self.data.qpos[qpos_adr:qpos_adr+3] = pose_matrix[:3, 3]
                    self.data.qpos[qpos_adr+3:qpos_adr+7] = quat
                    # Zero velocity
                    qvel_adr = self.model.jnt_dofadr[joint_id]
                    self.data.qvel[qvel_adr:qvel_adr+6] = 0
            
            # Update simulation
            mj.mj_forward(self.model, self.data)
            
        except Exception as e:
            print(f"Error setting pose for {object_name}: {str(e)}")
            print("Available bodies:", [self.model.body(i).name for i in range(self.model.nbody)])
            raise
    def reset_auto(self):
        self.gripper_close()
        self.model.opt.gravity[:]=[0,0,0]
        # Get initial poses
        obj_pose = self.get_object_pose("pickup_object")
        initial_pose = self.pose()
        initial_rot = initial_pose[:3,:3]
        obj_pos = obj_pose[:3, 3]
        obj_rot = obj_pose[:3,:3]
        print(initial_rot)
        # Approach in world Z-axis (unchanged)
        local_offset1 = np.array([0, 0, -0.15])
        world_offset1 = obj_rot @ local_offset1 
        approach_pos = obj_pos + world_offset1
        approach_pose = t3d.affines.compose(approach_pos, obj_rot, [1, 1, 1])
        
        self.servo_to_pose(approach_pose, lin_tol=0.003, ang_tol=0.1)
        print("Approaching")
        # First offset - transformed to object frame
        local_offset1 = np.array([-0.030, 0, 0.015])  # In object frame
        world_offset1 = obj_rot @ local_offset1  # Transform to world frame
        current_pos = approach_pose[:3,3] + world_offset1
        off_rot = t3d.euler.euler2mat(0, 0, np.pi/2)
        new_rot = obj_rot @ off_rot
        rotated_pose = t3d.affines.compose(current_pos, new_rot, [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.003, ang_tol=0.1)

        # Second offset - in current gripper frame
        curr_pose = self.pose()
        local_offset2 = np.array([0.013, 0, 0])  # In gripper frame
        world_offset2 = curr_pose[:3,:3] @ local_offset2
        current_pos = curr_pose[:3,3] + world_offset2
        rotated_pose = t3d.affines.compose(current_pos, curr_pose[:3,:3], [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.0015, ang_tol=0.1)

        # Third offset - in current gripper frame (Z-axis)
        curr_pose = self.pose()
        local_offset3 = np.array([0, 0, 0.07])  # In gripper frame
        world_offset3 = curr_pose[:3,:3] @ local_offset3
        current_pos = curr_pose[:3,3] + world_offset3
        rotated_pose = t3d.affines.compose(current_pos, curr_pose[:3,:3], [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.0015, ang_tol=0.1)
        
        # Open gripper and lift - world Z-axis
        self.gripper_open()
    
        current_pos = curr_pose[:3,3] + np.array([0, 0, 0.1])  # World frame
        rotated_pose = t3d.affines.compose(current_pos, curr_pose[:3,:3], [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.003, ang_tol=0.1)
        
        # Apply random rotation in object frame
        angle_x = np.random.uniform(-0.3, 0.3) 
        angle_y = np.random.uniform(-0.3, 0.3)   
        angle_z = np.random.uniform(-np.pi, np.pi)
        random_rot = t3d.euler.euler2mat(np.pi/8 + angle_x,
                                        np.pi/8 + angle_y,
                                        np.pi/8 + angle_z)
        const_rot = np.array([[9.99996083e-01, -5.31966273e-07, -2.79879023e-03], [-5.40655149e-07, -1.00000000e+00, -3.10375475e-06], [-2.79879023e-03, 3.10525578e-06, -9.99996083e-01]])
        curr_pose = self.pose()
        new_rot = const_rot @ random_rot
        trans = np.array([0.25,np.random.uniform(-0.01,0.01),0.2])
        print(f"Moved to {trans}")
        rotated_pose = t3d.affines.compose(trans, new_rot, [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.003, ang_tol=0.1)
        
        # Final grasp and lift
        self.gripper_close()
        self.model.opt.gravity[:]=[0,0,-9.81]
        mj.mj_forward(self.model, self.data)
        self.stop_object_movement()
        
        curr_obj_pose = self.get_object_pose()
        current_pos = curr_obj_pose[:3,3] + np.array([0, 0, 0.2])  # World Z
        rotated_pose = t3d.affines.compose(current_pos, initial_rot, [1, 1, 1])
        self.servo_to_pose(rotated_pose, lin_tol=0.003, ang_tol=0.1)



    def _sync_viewer_loop(self):
        while self._viewer_active:
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()
            time.sleep(0.005)  # ~200Hz refresh
    def update_sim(self):
        mj.mj_forward(self.model, self.data)
        self.viewer.sync()

    def gripper_close(self):
        self._set_gripper_position(self.gripper_close_pos)
        
    def gripper_open(self):
        self._set_gripper_position(self.gripper_open_pos)

    def gripper_off(self):
            self._set_gripper_position(0.0)


    def _set_gripper_position(self, pos, tolerance=1e-3, max_sim_steps=2000): # Increased max_sim_steps for safety
        target_pos = np.clip(pos, self.gripper_close_pos, self.gripper_open_pos)
        self.data.ctrl[self.gripper_ctrl_id] = target_pos
        steps_taken = 0
        while abs(self.data.qpos[self.gripper_ctrl_id] - target_pos) > tolerance and steps_taken < max_sim_steps:
            mj.mj_step(self.model, self.data)
            self.viewer.sync() 
            steps_taken += 1
            self.current_gripper_pos = target_pos 


    def send_action(self, target_pose_matrix):
        """
        Sends a target end-effector pose to the robot.
        This function now drives JacobiRobot's internal state using its servo_to_pose,
        and then applies JacobiRobot's internal joint configuration to MuJoCo.
        """

        try:
            base_pos = self.data.xpos[self.model.body("link_base").id]
            base_rot = self.data.xmat[self.model.body("link_base").id].reshape(3,3)
            world_to_base = t3d.affines.compose(base_pos, base_rot, [1,1,1])
        except KeyError:
            world_to_base = np.eye(4)
        base_target_pose = np.linalg.inv(world_to_base) @ target_pose_matrix
        servo_dt = self.model.opt.timestep   
        self.robot_jacobi.servo_to_pose(base_target_pose, dt=servo_dt)

        full_jacobi_q = self.robot_jacobi.q
        target_joint_qpos_for_mujoco = np.array([full_jacobi_q[idx] for idx in self.mujoco_actuator_to_jacobi_joint_idx])    
        self.data.ctrl[self.mujoco_actuator_ids] = target_joint_qpos_for_mujoco

        mj.mj_step(self.model, self.data)

        self.viewer.sync()
        return True
    
    def get_object_pose(self, object_name="pickup_object"):
        """Returns the 4x4 homogeneous transformation matrix of an object"""
        try:
            obj_id = self.model.body(object_name).id
            
            pos = self.data.body(obj_id).xpos
            rot = self.data.body(obj_id).xmat.reshape(3, 3)
            
            return t3d.affines.compose(pos, rot, [1, 1, 1])
        except Exception as e:
            return None

    def pose(self):
        if self.eef_id_type == 'site':
            pos = self.data.site_xpos[self.eef_id]
            rot_mat = self.data.site_xmat[self.eef_id].reshape(3, 3)
        else:
            pos = self.data.xpos[self.eef_id]
            rot_mat = self.data.xmat[self.eef_id].reshape(3, 3)
        return t3d.affines.compose(pos, rot_mat, [1, 1, 1])

    def _initialize_offscreen_rendering(self):
        """Initializes the offscreen rendering context for capturing images."""
        global _mujoco_gl_context_initialized

        try:
            self.scn = mj.MjvScene(self.model, maxgeom=1000)
            self.cam = mj.MjvCamera()
            self.vopt = mj.MjvOption()
            self.pert = mj.MjvPerturb()

            mj.mjv_defaultCamera(self.cam)
            mj.mjv_defaultOption(self.vopt)

            self.mjr_context = mj.MjrContext(self.model, 100)
            self._offscreen_initialized = True
        except Exception as e:
            self._offscreen_initialized = False
            self.mjr_context = None

    def get_observation(self, camera_name: str = "gripper_camera"):
        """Gets the current observation from the simulation."""
        global _mujoco_gl_context_initialized

        if not self._offscreen_initialized:
            self._initialize_offscreen_rendering()
          
        try:
            mj.mjv_updateScene(self.model, self.data, self.vopt, self.pert, self.cam, mj.mjtCatBit.mjCAT_ALL, self.scn)

            if camera_name in [self.model.camera(i).name for i in range(self.model.ncam)]:
                self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
                self.cam.fixedcamid = self.model.camera(camera_name).id
            else:
                self.cam.type = mj.mjtCamera.mjCAMERA_FREE
                mj.mjv_defaultCamera(self.cam)

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
        """Closes the MuJoCo viewer and resources."""
        self.logger.debug("Closing MuJoCo resources.")
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.mjr_context and self._offscreen_initialized:
            self.mjr_context = None
            self._offscreen_initialized = False

if __name__ == "__main__":
    sim = None
    output_dir = "captured_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {os.path.abspath(output_dir)}")

    try:
        sim = MuJoCoRobot()
        #sim.viewer.cam.fixedcamid = sim.model.camera('gripper_camera').id
        #sim.viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
        initial_pose = sim.pose()
        initial_pos = initial_pose[:3, 3].copy()
        initial_rot = initial_pose[:3, :3].copy()

        target_pos = initial_pos + np.array([0.1, 0.0, 0.06])
        target_rot = initial_rot


        target_pose_matrix = t3d.affines.compose(target_pos, target_rot, [1, 1, 1])

        duration_s = 3.0  
        steps_to_simulate = int(duration_s / sim.model.opt.timestep)
        
        for i in range(steps_to_simulate):
            success = sim.send_action(target_pose_matrix)
            if success and (i % 50 == 0 or i == steps_to_simulate - 1):
                observation = sim.get_observation(camera_name="gripper_camera")
                if observation and "camera.rgb" in observation:
                    Image.fromarray(observation["camera.rgb"]).save(
                        os.path.join(output_dir, f"frame_{i:05d}.png"))

        final_pose = sim.pose()
        position_error = np.linalg.norm(final_pose[:3, 3] - target_pos)
        tolerance = 0.002 

        if position_error <= tolerance:
            print(f"Robot arrived with tolerance of {position_error*1000:.2f} mm")
        else:
            print(f"Position error ({position_error:.4f}m) > tolerance ({tolerance}m)")


        obj_pose = sim.get_object_pose("pickup_object")
        print(obj_pose)
        obj_pos = obj_pose[:3, 3]
        obj_rot = obj_pose[:3, :3]
        angle = np.deg2rad(180)
        rot_off = t3d.axangles.axangle2mat([0, 1, 0], angle)

        approach_pos = obj_pos + np.array([0, 0, 0.2])
        approach_rot = obj_rot @ rot_off # Match object orientation
        
        # Create pose matrix
        approach_pose = t3d.affines.compose(approach_pos, approach_rot, [1, 1, 1])
        
        for i in range(steps_to_simulate):
            success = sim.send_action(approach_pose)
        time.sleep(1)
        sim.gripper_close()
        time.sleep(1)
        approach_pos = obj_pos + np.array([0, 0, 0.12])
        grip_pose = t3d.affines.compose(approach_pos, approach_rot, [1, 1, 1])
        for i in range(steps_to_simulate):
            sim.send_action(grip_pose)
        sim.gripper_open()
        time.sleep(1)
        approach_pos = obj_pos + np.array([0, 0, 0.12])
        grip_pose = t3d.affines.compose(approach_pos, approach_rot, [1, 1, 1])
        for i in range(steps_to_simulate):
            sim.send_action(grip_pose)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if sim:
            time.sleep(30)
            sim.close()
    print("Simulation finished.")